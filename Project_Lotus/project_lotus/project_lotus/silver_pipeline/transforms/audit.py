"""Audit normalization and left enrichment; no fabricated reference mappings."""
from pyspark.sql import functions as F
from ..common import prepare, string_inputs, cast_long, timestamp, date_column, pad_identifier, finish


def transform(bronze, cfg, partners, providers, api_types):
    """References must be normalized/validated by references.prepare_references.

    Returns (included, excluded). Excluded test rows are counted and recorded,
    so reconciliation includes legitimate business exclusions separately.
    """
    df = string_inputs(prepare(bronze, cfg), cfg)
    test = (F.coalesce(F.lower("_s_partner_id").isin([x.lower() for x in cfg["test_partner_ids"]]), F.lit(False)) |
            F.coalesce(F.lower("_s_service_provider_id").isin([x.lower() for x in cfg["test_provider_ids"]]), F.lit(False)))
    excluded = df.filter(test).select(F.col("_s_id").alias("record_id"), "_raw_payload", "_source_file")
    df = df.filter(~test)
    df = timestamp(df, "_in_timestamp", "api_timestamp", cfg["null_tokens"])
    df = date_column(df, "_s_date", "date")
    df = cast_long(df, "_s_processing_time", "processing_time_ms")
    df = cast_long(df, "_s_timestamp_ms", "_epoch_ms")
    # Bound epoch milliseconds before converting to avoid timestamp overflow.
    valid_epoch = F.col("_epoch_ms").between(-62135596800000, 253402300799999)
    df = df.withColumn("timestamp_ms", F.timestamp_millis(F.when(valid_epoch, F.col("_epoch_ms"))))
    df = df.withColumn("_bad_timestamp_ms", F.col("_s_timestamp_ms").isNotNull() & F.col("timestamp_ms").isNull())
    for target, source in {"record_id": "id", "msisdn": "msisdn", "phone_number_AT_hash": "hashed_msisdn",
                           "encrypted_msisdn": "encrypted_msisdn", "brand": "brand", "source_ip": "source_ip",
                           "source_event_id": "request_id"}.items():
        df = df.withColumn(target, F.col("_s_" + source))
    for name in ("operation", "mno", "response_code"):
        df = df.withColumn(name, F.upper("_s_" + name))
    for name in ("partner_id", "service_provider_id"):
        df = df.withColumn(name, pad_identifier("_s_" + name))
    # Provider name and industry share one validated provider key, avoiding an
    # accidental many-to-many expansion from separate non-unique joins.
    df = df.join(partners, "partner_id", "left").join(providers, "service_provider_id", "left")
    mapping = F.create_map(*[F.lit(x) for pair in api_types.items() for x in pair])
    df = (df.withColumn("api_type", F.coalesce(mapping[F.col("operation")], F.col("operation")))
          .withColumn("_warn_unknown_api", F.col("operation").isNotNull() & ~F.col("operation").isin(list(api_types)))
          .withColumn("_warn_partner_reference", F.col("partner_id").isNotNull() & F.col("_partner_match").isNull())
          .withColumn("_warn_provider_reference", F.col("service_provider_id").isNotNull() & F.col("_provider_match").isNull())
          .withColumn("correlation_id", F.lit(None).cast("string"))
          .withColumn("source_name", F.lit(cfg["source_name"]))
          .withColumn("schema_version", F.lit(cfg["schema_version"]))
          .withColumn("event_date", F.to_date("api_timestamp")))
    # request_id is lineage, not evidence of a correlation-id mapping.
    return finish(df.drop("_bad__epoch_ms"), cfg), excluded
