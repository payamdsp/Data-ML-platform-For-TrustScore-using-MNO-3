"""Device events: carrier-specific statuses, nullable notes and source date."""
from pyspark.sql import functions as F
from ..common import prepare, string_inputs, notes, cast_long, timestamp, date_column, finish


def transform(bronze, cfg, ingestion_ts):
    """Normalize one Bronze device_lookup_batch batch into the Silver schema.

    Unlike account_changes_batch, MNO arrives directly on the source row here,
    so it only needs uppercasing rather than notes-based inference.
    """
    df = notes(string_inputs(prepare(bronze, cfg), cfg), cfg)
    df = cast_long(df, "_s_id", "record_id", unsigned=True)
    df = timestamp(df, "_in_timestamp", "event_timestamp", cfg["null_tokens"])
    df = date_column(df, "_s_date", "date")
    # MNO is a top-level source column in the actual client device notebook.
    for target, source in {"phone_number_AC_hash": "change_key", "phone_number_AT_hash": "change_key_ats",
                           "imei": "imei", "imsi": "imsi", "related_id": "related_id"}.items():
        df = df.withColumn(target, F.col("_s_" + source))
    for name in ("tac", "oldIMSI", "oldIMEI", "fromADCSnapshot"):
        df = df.withColumn(name, F.col("_n_" + name))
    df = (df.withColumn("mno", F.upper("_s_mno"))
          .withColumn("event_type", F.upper("_s_status"))
          .withColumn("correlation_id", F.col("_n_correlationId"))
          .withColumn("event_date", F.to_date("event_timestamp"))
          .withColumn("ingestion_ts", F.lit(ingestion_ts).cast("timestamp"))
          .withColumn("source_name", F.lit(cfg["source_name"]))
          .withColumn("source_event_id", F.col("record_id"))
          .withColumn("schema_version", F.lit(cfg["schema_version"])))
    return finish(df, cfg)
