"""Shared Spark expressions. No Python UDFs and no implicit destructive casts."""
from functools import reduce
from pyspark.sql import functions as F, types as T


def quoted(name):
    """Backtick-quote one identifier so Spark treats it literally, not as an expression."""
    return "`" + name.replace("`", "``") + "`"


def clean(column, tokens=("", "\\N")):
    """Preserve identifier/hash case; only enumerations are uppercased separately."""
    c = F.col(column) if isinstance(column, str) else column
    value = F.trim(c.cast("string"))
    return F.when(value.isin(list(tokens)), F.lit(None)).otherwise(value)


def prepare(df, cfg):
    """Resolve source names once, without repeating the Lambda schema registry.

    Required *columns* must exist for the transform to be meaningful. Optional
    fields are typed nulls. Row values are evaluated later by YAML rules.
    Unexpected top-level columns are allowed and retained only in issue payloads.
    """
    names = {}
    for name in df.columns:
        if name.lower() in names:
            raise ValueError("Ambiguous case-insensitive source column: " + name)
        names[name.lower()] = name
    if any(name.startswith("_") for name in df.columns):
        raise ValueError("Source columns beginning '_' are reserved by this pipeline")
    # This JSON is only written for problematic rows, never to the Silver table.
    # Retaining explicit nulls is important for remediation: a null source value
    # must not look identical to a column that was absent from captured evidence.
    df = df.withColumn(
        "_raw_payload",
        F.to_json(
            F.struct(*[F.col(quoted(c)) for c in df.columns]),
            {"ignoreNullFields": "false"},
        ),
    )
    df = df.withColumn("_source_file", F.input_file_name())
    # Two source rows can have identical values.  This distributed token keeps
    # them distinct in record-level quarantine without treating it as a business
    # identifier or collecting rows on the driver.
    df = df.withColumn(
        "_source_row_id", F.monotonically_increasing_id().cast("string")
    )
    for canonical, spec in cfg["inputs"].items():
        found = [names[n.lower()] for n in spec["aliases"] if n.lower() in names]
        if len(found) > 1:
            raise ValueError(f"Multiple source aliases for {canonical}: {found}; select one upstream")
        if not found and spec["required"]:
            raise ValueError(f"Missing transform input {canonical}; accepted names: {spec['aliases']}")
        df = df.withColumn("_in_" + canonical, F.col(quoted(found[0])) if found else F.lit(None).cast("string"))
    return df


def string_inputs(df, cfg):
    """Trim and null-normalize every declared input into a parallel `_s_*` string column.

    Keep source types until timestamp parsing has distinguished instants from strings.
    """
    for name in cfg["inputs"]:
        df = df.withColumn("_s_" + name, clean("_in_" + name, cfg["null_tokens"]))
    return df


def cast_long(df, source, target, unsigned=False):
    """'4321' -> 4321; decimals, exponents and overflow -> null + an issue flag.

    SQL try_cast is supported on the Spark 3.5 API baseline. Column.try_cast
    would require Spark 4. A digit guard prevents rounding/truncating 4321.7.
    """
    regex = "^[0-9]+$" if unsigned else "^[+-]?[0-9]+$"
    df = df.withColumn(target, F.when(F.col(source).rlike(regex),
                                      F.expr(f"try_cast({quoted(source)} AS BIGINT)")))
    return df.withColumn("_bad_" + target, F.col(source).isNotNull() & F.col(target).isNull())


def timestamp(df, source, target, tokens):
    """Strings are Toronto wall time; TimestampType values already are instants.

    Do not call from_utc_timestamp on an already local source string. Explicit
    offset strings are accepted and represent their own instant. Fractional
    seconds extend the POC formats without changing whole-second/date parsing.
    """
    dtype = df.schema[source].dataType
    if isinstance(dtype, (T.TimestampType, T.TimestampNTZType)):
        result = F.col(source).cast("timestamp")
    else:
        s = clean(source, tokens)
        result = F.coalesce(*[F.try_to_timestamp(s, F.lit(p)) for p in (
            "yyyy-MM-dd HH:mm:ss[.SSSSSSSSS]", "yyyy-MM-dd",
            "yyyy-MM-dd'T'HH:mm:ss[.SSSSSSSSS]XXX",
            "yyyy-MM-dd'T'HH:mm:ss[.SSSSSSSSS]",
        )])
    df = df.withColumn(target, result)
    return df.withColumn("_bad_" + target, clean(source, tokens).isNotNull() & F.col(target).isNull())


def date_column(df, source, target):
    """Parse a strict yyyy-MM-dd string into a DATE; anything else becomes null + a bad flag.

    try_to_timestamp also rejects impossible calendar dates under ANSI mode.
    """
    s = F.when(F.col(source).rlike("^[0-9]{4}-[0-9]{2}-[0-9]{2}$"), F.col(source))
    df = df.withColumn(target, F.to_date(F.try_to_timestamp(s, F.lit("yyyy-MM-dd"))))
    return df.withColumn("_bad_" + target, F.col(source).isNotNull() & F.col(target).isNull())


def notes(df, cfg):
    """Fixed typed projection, as in the POC; unknown JSON keys do not evolve DDL.

    Absent notes/fields are null. Non-object/malformed JSON is distinguished
    from absent notes so account MNO inference cannot hide corrupt input.
    """
    fields = cfg["notes_fields"]
    schema = T.StructType([T.StructField(n, T.StringType()) for n in fields])
    raw = F.col("_s_notes")
    df = df.withColumn("_notes", F.from_json(raw, schema, {"mode": "PERMISSIVE"}))
    # json_object_keys validates the outer object without reserving a JSON key
    # such as _corrupt_record that a legitimate additional field could contain.
    df = df.withColumn("_bad_notes", raw.isNotNull() & F.json_object_keys(raw).isNull())
    for name in fields:
        df = df.withColumn("_n_" + name, clean(F.col("_notes").getField(name), cfg["null_tokens"]))
    return df


def pad_identifier(c):
    """Match main and reference keys; never truncate identifiers longer than ten."""
    value = clean(c)
    return F.when(value.rlike("^[0-9]{1,10}$"), F.lpad(value, 10, "0")).otherwise(value)


def finish(df, cfg):
    """Publishable schema is explicit; raw data/flags accompany only DQ evaluation."""
    output = list(cfg["output"])
    internal = [c for c in df.columns if c.startswith("_bad_") or c.startswith("_warn_")]
    return df.select(
        *[F.col(c).cast(dtype).alias(c) for c, dtype in cfg["output"].items()],
        *internal,
        "_raw_payload",
        "_source_file",
        "_source_row_id",
    )


def business_columns(cfg):
    """Output columns that define whether two rows are the same event.

    Run time does not turn an otherwise identical event into a new event, so
    ingestion_ts is excluded from dedup/conflict comparisons.
    """
    return [c for c in cfg["output"] if c != "ingestion_ts"]


def exact_deduplicate(df, cfg):
    """Remove identical business rows, never select a winner just by record_id.

    Raw-row DQ must run first: two bad source values can both normalize to null.
    Group by all semantic columns. Ingestion time is constant per job and its
    minimum makes the function stable even when callers combine validated runs.
    Spark hashes all group keys, spreading large audit operations across tasks.
    """
    keys = business_columns(cfg)
    # Candidate DQ and existing-target conflicts occur after exact deduplication.
    # Retain a deterministic representative of the source evidence so any later
    # quarantine entry remains recoverable instead of containing only normalized
    # Silver columns.
    lineage_columns = [
        column
        for column in ("_raw_payload", "_source_file", "_source_row_id")
        if column in df.columns
    ]
    aggregations = []
    if "ingestion_ts" in cfg["output"]:
        aggregations.append(F.min("ingestion_ts").alias("ingestion_ts"))
    aggregations.extend(
        F.min(F.col(quoted(column))).alias(column) for column in lineage_columns
    )
    if not aggregations:
        return df.select(*keys).dropDuplicates()
    return df.groupBy(*keys).agg(*aggregations).select(
        *cfg["output"], *lineage_columns
    )
