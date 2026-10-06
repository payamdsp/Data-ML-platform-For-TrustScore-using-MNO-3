"""An empty silver table must NOT report passed=true.

Regression test for a real 2026-07-16 run: device_lookup_batch at row_count=0 produced
16/16 rules passed and validated cleanly against the committed schema, which would have
told Sprint 2 lineage that an unpublished table was trustworthy.
"""
from silver_dq import build_summary, compute_dq, load_checks, validate


def test_empty_table_fails_summary(spark, account_changes_df, acct_yaml):
    empty = account_changes_df.limit(0)
    cfg = load_checks(acct_yaml)
    summary, metrics = compute_dq(empty, cfg, "2026-07-16")

    assert summary["row_count"] == 0
    assert summary["passed"] is False, "empty table must not report passed=true"

    empty_rules = [r for r in summary["rules"] if r["rule"] == "table is not empty"]
    assert len(empty_rules) == 1
    assert empty_rules[0]["passed"] is False
    assert empty_rules[0]["violations"] == 1
    assert summary["failed_count"] >= 1


def test_empty_table_still_matches_schema(spark, account_changes_df, acct_yaml, schema_dir):
    """The red verdict must still be a contract-valid artifact, not a crash."""
    cfg = load_checks(acct_yaml)
    summary, metrics = compute_dq(account_changes_df.limit(0), cfg, "2026-07-16")
    validate(summary, metrics, schema_dir)


def test_non_empty_table_has_no_empty_rule(spark, account_changes_df, acct_yaml):
    """The guard must not fire on a table that has rows."""
    cfg = load_checks(acct_yaml)
    summary, _ = compute_dq(account_changes_df, cfg, "2026-07-16")
    assert summary["row_count"] == 4
    assert not [r for r in summary["rules"] if r["rule"] == "table is not empty"]


def test_build_summary_empty_without_spark():
    """build_summary is pure Python: the guard is testable with no SparkSession."""
    cfg = load_checks  # not used; keep import surface honest
    from silver_dq.config import parse_checks

    families = {"row_count": 0, "uniqueness": {"metrics": {}, "rules": []}}
    summary = build_summary(families, parse_checks({"dataset": "demo"}), "2026-07-16")
    assert summary["passed"] is False
    assert summary["rule_count"] == 1
