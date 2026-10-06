"""Unit tests: one per §6.1.2 metric family.

Each test builds a tiny in-memory frame, runs that family's aggregation
expressions through a single ``.agg(...)``, and checks the parser output.
"""

from datetime import datetime

from pyspark.sql.types import (
    LongType, StringType, StructField, StructType, TimestampType,
)

from silver_dq import metrics as M
from silver_dq import parse_checks

BELL_ROGERS = ["IMEI_START", "IMEI_END", "IMSI_START", "IMSI_END"]
TELUS_SET = BELL_ROGERS + [
    "IMEI_START_A", "IMEI_START_NC", "IMSI_START_A", "IMSI_START_NC", "IMSI_END_C",
]


# #
# completeness
# #
def test_completeness_null_rate(spark):
    schema = StructType([
        StructField("record_id", LongType()),
        StructField("event_type", StringType()),
    ])
    df = spark.createDataFrame([(1, "A"), (2, None), (3, "C")], schema)
    cols = df.columns

    row = df.agg(*M.completeness_exprs(cols)).collect()[0]
    out = M.parse_completeness(row, cols, row_count=3, required={"event_type"})

    assert out["per_column"]["record_id"]["null_count"] == 0
    assert out["per_column"]["event_type"]["null_count"] == 1
    assert out["per_column"]["event_type"]["null_rate"] == round(1 / 3, 6)

    rule = next(r for r in out["rules"] if r["column"] == "event_type")
    assert rule["violations"] == 1 and rule["passed"] is False


# #
def test_completeness_matches_required_case_insensitively(spark):
    """Live find 2026-07-17: account_changes_batch has phone_number_AC_hash (Iceberg
    preserves column case) while the yaml says phone_number_ac_hash; 7 required
    columns produced only 6 rules. The match must work the way Spark resolves
    columns: case-insensitively, with the rule labeled by the table's casing."""
    schema = StructType([
        StructField("record_id", StringType()),
        StructField("phone_number_AC_hash", StringType()),
    ])
    df = spark.createDataFrame([("1", "h1"), ("2", None)], schema)
    row = df.agg(*M.completeness_exprs(df.columns)).collect()[0]
    out = M.parse_completeness(row, df.columns, 2, {"phone_number_ac_hash"})

    assert len(out["rules"]) == 1, f"required column produced no rule: {out['rules']}"
    rule = out["rules"][0]
    assert rule["column"] == "phone_number_AC_hash"
    assert rule["violations"] == 1 and rule["passed"] is False


def test_completeness_missing_required_column_goes_red(spark):
    """A required column absent from the table (any casing) must fail loudly as a
    rule, not vanish; a typo'd or renamed column would otherwise disable its gate."""
    schema = StructType([StructField("record_id", StringType())])
    df = spark.createDataFrame([("1",)], schema)
    row = df.agg(*M.completeness_exprs(df.columns)).collect()[0]
    out = M.parse_completeness(row, df.columns, 1, {"record_id", "no_such_column"})

    missing = [r for r in out["rules"] if r["rule"] == "no_such_column exists in the table"]
    assert len(missing) == 1 and missing[0]["passed"] is False
    assert len(out["rules"]) == 2


# consistency: flat mno domain AND per-MNO event_type domain (Telus extended)
# #
def test_consistency_flat_and_per_mno(spark):
    schema = StructType([
        StructField("mno", StringType()),
        StructField("event_type", StringType()),
    ])
    rows = [
        ("BELL", "IMEI_START"),    # both valid
        ("FREEDOM", "IMEI_START"),  # mno out of domain; not in per-MNO map -> no et viol
        ("BELL", "IMSI_END_C"),    # IMSI_END_C is Telus-only -> et violation for BELL
        ("TELUS", "IMSI_END_C"),   # valid: Telus extended set allows it
    ]
    df = spark.createDataFrame(rows, schema)

    allowed = {"mno": ["BELL", "ROGERS", "TELUS"]}
    by_mno = {"event_type": {"BELL": BELL_ROGERS, "ROGERS": BELL_ROGERS, "TELUS": TELUS_SET}}

    row = df.agg(*M.consistency_exprs(allowed, by_mno, "mno")).collect()[0]
    out = M.parse_consistency(row, allowed, by_mno, row_count=4)

    assert out["per_column"]["mno"]["violation_count"] == 1          # FREEDOM
    assert out["per_column"]["event_type"]["violation_count"] == 1   # BELL + IMSI_END_C


# #
# uniqueness: primary key, and null-safe full-row distinctness
# #
def test_uniqueness_primary_key(spark):
    schema = StructType([
        StructField("record_id", LongType()),
        StructField("payload", StringType()),
    ])
    df = spark.createDataFrame([(10, "a"), (20, "b"), (10, "c")], schema)

    row = df.agg(*M.uniqueness_exprs(["record_id"], df.columns)).collect()[0]
    out = M.parse_uniqueness(row, ["record_id"], row_count=3)

    assert out["metrics"]["distinct_keys"] == 2
    assert out["metrics"]["duplicate_count"] == 1
    # AC asks metrics.json for "distinct %" alongside null %
    assert out["metrics"]["distinct_rate"] == round(2 / 3, 6)
    assert out["metrics"]["distinct_rate"] + out["metrics"]["duplicate_rate"] == 1.0


def test_uniqueness_full_row_with_null_column(spark):
    # msisdn is 100% null (audit_trail case). Full-row dedup must still count these
    # rows: a naive countDistinct over the columns would drop every null-key row.
    schema = StructType([
        StructField("msisdn", StringType()),
        StructField("record_id", LongType()),
        StructField("val", StringType()),
    ])
    rows = [(None, 1, "a"), (None, 2, "b"), (None, 2, "b")]  # rows 2 & 3 identical
    df = spark.createDataFrame(rows, schema)

    row = df.agg(*M.uniqueness_exprs([], df.columns)).collect()[0]  # [] -> full row
    out = M.parse_uniqueness(row, [], row_count=3)

    assert out["metrics"]["primary_keys"] == ["<full row>"]
    assert out["metrics"]["distinct_keys"] == 2
    assert out["metrics"]["duplicate_count"] == 1


# #
# timeliness: event-time (no ingestion_ts): coverage window, future, recency
# #
def test_timeliness_future_and_window(spark):
    schema = StructType([StructField("event_timestamp", TimestampType())])
    rows = [
        (datetime(2026, 6, 20, 12, 0, 0),),  # within run_date -> not future
        (datetime(2026, 6, 10, 12, 0, 0),),  # oldest -> min
        (datetime(2026, 6, 25, 12, 0, 0),),  # after run_date 2026-06-22 -> FUTURE
    ]
    df = spark.createDataFrame(rows, schema)
    cfg = parse_checks({"dataset": "t", "timeliness": {"mode": "batch", "future_grace_seconds": 0}})
    row = df.agg(*M.timeliness_exprs(cfg, "2026-06-22", df.columns)).collect()[0]
    res = M.parse_timeliness(row, cfg, df.columns, row_count=3)
    m = res["metrics"]

    assert m["future_count"] == 1
    assert m["null_event_time_count"] == 0
    assert m["min_event_time"].startswith("2026-06-10")
    assert m["max_event_time"].startswith("2026-06-25")
    # only the future-event rule (no recency threshold set); it fails on the 2026-06-25 row
    assert len(res["rules"]) == 1
    frule = res["rules"][0]
    assert frule["rule"] == "no future event_time"
    assert frule["violations"] == 1 and frule["passed"] is False


def test_timeliness_recency_gate(spark):
    # newest event 2026-06-01, run_date 2026-06-22 -> recency ~22 days; gate at 7 days -> fail
    schema = StructType([StructField("event_timestamp", TimestampType())])
    rows = [(datetime(2026, 6, 1, 12, 0, 0),), (datetime(2026, 5, 1, 12, 0, 0),)]
    df = spark.createDataFrame(rows, schema)
    cfg = parse_checks({"dataset": "t", "timeliness": {
        "mode": "batch", "future_grace_seconds": 0, "max_recency_seconds": 604800}})
    row = df.agg(*M.timeliness_exprs(cfg, "2026-06-22", df.columns)).collect()[0]
    res = M.parse_timeliness(row, cfg, df.columns, row_count=2)
    m = res["metrics"]

    assert m["future_count"] == 0
    assert m["recency_seconds"] > 604800                    # ~22 days old
    rec = next(r for r in res["rules"] if r["rule"].startswith("recency"))
    assert rec["violations"] == 1 and rec["passed"] is False


# S12 additions: validity / victim exclusion / currency #
def _gold_frame(spark, rows):
    """Tiny bad_actor_gold-shaped frame: attestation + phone + ingestion time."""
    from pyspark.sql.types import StringType, StructField, StructType, TimestampType
    schema = StructType([StructField("record_id", StringType()),
                         StructField("victim_attestation", StringType()),
                         StructField("phone_number", StringType()),
                         StructField("ingestion_ts", TimestampType())])
    return spark.createDataFrame(rows, schema)


def test_validity_counts_non_null_format_mismatches(spark):
    from datetime import datetime

    from silver_dq import compute_dq, parse_checks
    df = _gold_frame(spark, [
        ("1", "CONFIRMED_BAD_ACTOR", "+14165551234", datetime(2026, 7, 10)),
        ("2", "CONFIRMED_BAD_ACTOR", "abc",          datetime(2026, 7, 10)),
        ("3", "CONFIRMED_BAD_ACTOR", None,           datetime(2026, 7, 10)),
    ])
    # Yaml names the column in UPPER case: resolution must be case-blind and the
    # output must carry the TABLE's casing (same contract as completeness).
    cfg = parse_checks({"dataset": "g", "primary_keys": ["record_id"],
                        "validity": {"formats": {"PHONE_NUMBER": r"^\+1[0-9]{10}$"}}})
    summary, metrics = compute_dq(df, cfg, "2026-07-16")
    col = metrics["overall"]["validity"]["per_column"]["phone_number"]
    assert col["violation_count"] == 1          # "abc" only; the NULL is completeness's job
    rule = next(r for r in summary["rules"] if r["family"] == "validity")
    assert rule["rule"] == "phone_number matches format" and rule["passed"] is False


def test_validity_missing_column_goes_red(spark):
    from datetime import datetime

    from silver_dq import compute_dq, parse_checks
    df = _gold_frame(spark, [("1", "CONFIRMED_BAD_ACTOR", "+14165551234", datetime(2026, 7, 10))])
    cfg = parse_checks({"dataset": "g", "primary_keys": ["record_id"],
                        "validity": {"formats": {"no_such_col": "^x$"}}})
    summary, _ = compute_dq(df, cfg, "2026-07-16")
    rule = next(r for r in summary["rules"] if r["family"] == "validity")
    assert rule["rule"] == "no_such_col exists in the table"
    assert rule["passed"] is False and summary["passed"] is False


def test_victim_exclusion_counts_wrong_and_null(spark):
    from datetime import datetime

    from silver_dq import compute_dq, parse_checks
    df = _gold_frame(spark, [
        ("1", "CONFIRMED_BAD_ACTOR", "+14165551234", datetime(2026, 7, 10)),
        ("2", "VICTIM",              "+14165551235", datetime(2026, 7, 10)),
        ("3", None,                  "+14165551236", datetime(2026, 7, 10)),
    ])
    cfg = parse_checks({"dataset": "g", "primary_keys": ["record_id"],
                        "victim_exclusion": {"column": "VICTIM_ATTESTATION",
                                             "allowed_values": ["CONFIRMED_BAD_ACTOR"]}})
    summary, metrics = compute_dq(df, cfg, "2026-07-16")
    vx = metrics["overall"]["victim_exclusion"]["metrics"]
    # VICTIM row + NULL row both fail: an unattested record must not be matchable.
    assert vx["column"] == "victim_attestation" and vx["violation_count"] == 2
    rule = next(r for r in summary["rules"] if r["family"] == "victim_exclusion")
    assert rule["passed"] is False and rule["violations"] == 2


def test_victim_exclusion_clean_table_passes_and_missing_column_goes_red(spark):
    from datetime import datetime

    from silver_dq import compute_dq, parse_checks
    clean = _gold_frame(spark, [("1", "CONFIRMED_BAD_ACTOR", "+14165551234", datetime(2026, 7, 10))])
    good = parse_checks({"dataset": "g", "primary_keys": ["record_id"],
                         "victim_exclusion": {"column": "victim_attestation",
                                              "allowed_values": ["CONFIRMED_BAD_ACTOR"]}})
    summary, _ = compute_dq(clean, good, "2026-07-16")
    assert summary["passed"] is True

    bad = parse_checks({"dataset": "g", "primary_keys": ["record_id"],
                        "victim_exclusion": {"column": "nope",
                                             "allowed_values": ["CONFIRMED_BAD_ACTOR"]}})
    summary, metrics = compute_dq(clean, bad, "2026-07-16")
    rule = next(r for r in summary["rules"] if r["family"] == "victim_exclusion")
    assert rule["rule"] == "nope exists in the table" and summary["passed"] is False
    assert metrics["overall"]["victim_exclusion"]["metrics"]["violation_count"] is None


def test_currency_counts_rows_past_review_cycle(spark):
    from datetime import datetime

    from silver_dq import compute_dq, parse_checks
    df = _gold_frame(spark, [
        ("1", "CONFIRMED_BAD_ACTOR", "+14165551234", datetime(2026, 7, 10)),   # 7 days old
        ("2", "CONFIRMED_BAD_ACTOR", "+14165551235", datetime(2026, 7, 12)),   # 5 days old
        ("3", "CONFIRMED_BAD_ACTOR", "+14165551236", datetime(2026, 5, 1)),    # 77 days old
    ])
    cfg = parse_checks({"dataset": "g", "primary_keys": ["record_id"],
                        "event_timestamp_col": "ingestion_ts",
                        "timeliness": {"mode": "batch",
                                       "currency": {"timestamp_col": "ingestion_ts",
                                                    "max_age_seconds": 30 * 86400}}})
    summary, metrics = compute_dq(df, cfg, "2026-07-16")
    cur = metrics["overall"]["timeliness"]["metrics"]["currency"]
    assert cur == {"timestamp_col": "ingestion_ts", "max_age_seconds": 30 * 86400,
                   "stale_count": 1, "stale_rate": 0.333333}
    rule = next(r for r in summary["rules"] if "review cycle" in r["rule"])
    assert rule["violations"] == 1 and rule["passed"] is False


def test_new_blocks_absent_leave_outputs_unchanged(spark, account_changes_df):
    """A config with no validity/victim/currency block emits none of the new keys;
    the S12 blocks are pure add-ons, off by default. (Built inline rather than from a
    committed yaml so it keeps holding regardless of what the shipped yamls configure.)"""
    from silver_dq import compute_dq, parse_checks
    cfg = parse_checks({
        "dataset": "account_changes_batch",
        "primary_keys": ["record_id"],
        "partition_by": "mno",
        "event_timestamp_col": "event_timestamp",
        "completeness": {"required_columns": ["record_id", "mno", "event_timestamp"]},
        "consistency": {"allowed_values": {"mno": ["BELL", "ROGERS", "TELUS"]}},
        "timeliness": {"mode": "batch", "future_grace_seconds": 0},
    })
    summary, metrics = compute_dq(account_changes_df, cfg, "2026-07-16")
    assert "validity" not in metrics["overall"]
    assert "victim_exclusion" not in metrics["overall"]
    assert "currency" not in metrics["overall"]["timeliness"]["metrics"]
    assert {r["family"] for r in summary["rules"]} <= {"completeness", "consistency",
                                                       "uniqueness", "timeliness"}


def test_config_rejects_bad_new_blocks():
    import pytest as _pytest

    from silver_dq import parse_checks
    with _pytest.raises(ValueError):
        parse_checks({"dataset": "g", "victim_exclusion": {"column": "c"}})            # no allowed_values
    with _pytest.raises(ValueError):
        parse_checks({"dataset": "g", "validity": {"formats": {"c": ""}}})             # empty pattern
    with _pytest.raises(ValueError):
        parse_checks({"dataset": "g",
                      "timeliness": {"mode": "batch",
                                     "currency": {"max_age_seconds": -1}}})            # non-positive cycle
