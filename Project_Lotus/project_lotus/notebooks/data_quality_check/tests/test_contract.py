"""Contract test: compute_dq outputs must validate against the committed schemas.

Loads the real account_changes_batch checks.yaml and fixture, runs the
full pipeline, validates both documents, and confirms a malformed document is
rejected (so the schema is actually constraining, not vacuous).
"""

import jsonschema
import pytest

from silver_dq import compute_dq, load_checks, validate

RUN_DATE = "2026-06-22"


def test_outputs_validate_against_schema(account_changes_df, acct_yaml, schema_dir):
    cfg = load_checks(acct_yaml)
    summary, metrics = compute_dq(account_changes_df, cfg, RUN_DATE)

    # must not raise
    validate(summary, metrics, schema_dir)

    assert summary["dataset"] == "account_changes_batch"
    assert summary["run_date"] == RUN_DATE
    assert summary["row_count"] == 4

    # duplicate record_id (rows 1 & 4) -> uniqueness rule fails -> overall fails
    uniq = next(r for r in summary["rules"] if r["family"] == "uniqueness")
    assert uniq["violations"] == 1
    assert summary["passed"] is False

    assert metrics["partition_by"] == "mno"
    assert set(metrics["per_mno"]) == {"BELL", "ROGERS", "TELUS"}


def test_invalid_summary_is_rejected(account_changes_df, acct_yaml, schema_dir):
    cfg = load_checks(acct_yaml)
    summary, metrics = compute_dq(account_changes_df, cfg, RUN_DATE)

    broken = dict(summary)
    broken["row_count"] = "not-an-integer"  # violates type: integer

    with pytest.raises(jsonschema.ValidationError):
        validate(broken, metrics, schema_dir)


def test_outputs_with_s12_blocks_validate_against_committed_schema(spark, schema_dir):
    """validity + victim_exclusion + currency configured together must still fit the
    committed contract; guards the schema extension itself."""
    from datetime import datetime

    from pyspark.sql.types import StringType, StructField, StructType, TimestampType

    from silver_dq import compute_dq, parse_checks, validate
    schema = StructType([StructField("record_id", StringType()),
                         StructField("victim_attestation", StringType()),
                         StructField("phone_number", StringType()),
                         StructField("ingestion_ts", TimestampType())])
    df = spark.createDataFrame([
        ("1", "CONFIRMED_BAD_ACTOR", "+14165551234", datetime(2026, 7, 10)),
        ("2", "VICTIM",              "abc",          datetime(2026, 5, 1)),
    ], schema)
    cfg = parse_checks({"dataset": "g", "primary_keys": ["record_id"],
                        "event_timestamp_col": "ingestion_ts",
                        "validity": {"formats": {"phone_number": r"^\+1[0-9]{10}$"}},
                        "victim_exclusion": {"column": "victim_attestation",
                                             "allowed_values": ["CONFIRMED_BAD_ACTOR"]},
                        "timeliness": {"mode": "batch",
                                       "currency": {"timestamp_col": "ingestion_ts",
                                                    "max_age_seconds": 30 * 86400}}})
    summary, metrics = compute_dq(df, cfg, "2026-07-16")
    assert summary["passed"] is False                       # planted violations fire
    validate(summary, metrics, schema_dir)                  # and the red run still fits the contract
