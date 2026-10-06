"""Account changes: notes-driven carrier inference and event normalization."""
from functools import reduce
from pyspark.sql import functions as F
from ..common import prepare, string_inputs, notes, cast_long, timestamp, finish


def transform(bronze, cfg, ingestion_ts):
    """Normalize one Bronze account_changes_batch batch into the Silver schema.

    MNO is not a reliable source field for this dataset, so it is inferred from
    the notes JSON before any DQ rule runs: an explicit notes.mno wins, certain
    notes fields imply BELL, and everything else defaults to ROGERS.
    """
    df = notes(string_inputs(prepare(bronze, cfg), cfg), cfg)
    df = cast_long(df, "_s_id", "record_id", unsigned=True)
    df = timestamp(df, "_in_timestamp", "event_timestamp", cfg["null_tokens"])
    mno = F.upper(F.col("_n_mno"))
    indicators = reduce(lambda a, b: a | b, [F.col("_n_" + n).isNotNull() for n in
        ("subId", "msisdnChangeSide", "correlationId", "otherMSISDN")])
    df = df.withColumn("mno", F.when(F.col("_bad_notes"), F.lit(None).cast("string"))
        .when(mno == "BELL_LANDLINE", "BELL")
        .when(mno.isNotNull(), mno)
        .when(indicators, "BELL").otherwise("ROGERS"))
    for name in ("subId", "msisdnChangeSide", "correlationId", "otherMSISDN"):
        df = df.withColumn(name, F.col("_n_" + name))
    df = (df.withColumn("phone_number_AC_hash", F.col("_s_change_key"))
          .withColumn("event_type", F.upper("_s_change_type"))
          .withColumn("event_date", F.to_date("event_timestamp"))
          .withColumn("ingestion_ts", F.lit(ingestion_ts).cast("timestamp"))
          .withColumn("source_name", F.lit(cfg["source_name"]))
          .withColumn("source_event_id", F.col("record_id"))
          .withColumn("schema_version", F.lit(cfg["schema_version"])))
    # The POC diagnoses, rather than removes, unexpected Rogers notes extras.
    df = df.withColumn("_warn_rogers_notes", (F.col("mno") == "ROGERS") & indicators)
    return finish(df, cfg)
