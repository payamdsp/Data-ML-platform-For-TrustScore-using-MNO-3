"""Every committed checks.yaml must load and run.

The AC requires a check definition per silver table in the same PR; these tests
assert each one parses into a usable CheckConfig and drives a real DQ run, so a broken
yaml fails here rather than at 19:17 on a Glue session.
"""
import glob
import os

import pytest

from silver_dq import compute_dq, load_checks, validate

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAMLS = sorted(glob.glob(os.path.join(PKG_ROOT, "checks", "dataset=*", "checks.yaml")))
DATASETS = [os.path.basename(os.path.dirname(p)).split("=", 1)[1] for p in YAMLS]


def test_all_datasets_committed():
    assert set(DATASETS) == {
        "account_changes_batch", "audit_trail_services_3",
        "device_lookup_batch", "tu_portps", "tu_portps_tulip",
    }, f"found {DATASETS}"


@pytest.mark.parametrize("path", YAMLS, ids=DATASETS)
def test_yaml_parses(path):
    cfg = load_checks(path)
    assert cfg.dataset == os.path.basename(os.path.dirname(path)).split("=", 1)[1]
    assert cfg.required_columns, "no required_columns -> completeness produces no rules"
    assert cfg.timeliness_mode in ("batch", "streaming")
    if cfg.allowed_values_by_mno:
        assert cfg.partition_by, "per-MNO allowed values need a partition_by column"


@pytest.mark.parametrize("path", YAMLS, ids=DATASETS)
def test_yaml_columns_are_lowercase(path):
    """Convention only: keep yaml column names lowercase for consistency. Iceberg
    PRESERVES column case (the live table has phone_number_AC_hash), the engine
    matches case-insensitively, and a required column that matches nothing now
    produces a failed `exists in the table` rule instead of silence."""
    cfg = load_checks(path)
    named = (list(cfg.required_columns) + list(cfg.primary_keys)
             + list(cfg.allowed_values) + list(cfg.allowed_values_by_mno)
             + [cfg.event_timestamp_col] + ([cfg.partition_by] if cfg.partition_by else [])
             + list(cfg.validity_formats) + [cfg.currency_timestamp_col]
             + ([cfg.victim_exclusion_col] if cfg.victim_exclusion_col else []))
    bad = [c for c in named if c != c.lower()]
    assert not bad, f"{cfg.dataset}: not lowercase -> will never match Glue: {bad}"


def test_account_changes_yaml_runs_end_to_end(spark, account_changes_df, acct_yaml, schema_dir):
    """The one dataset we have a real fixture for: parse -> run -> validate."""
    cfg = load_checks(acct_yaml)
    summary, metrics = compute_dq(account_changes_df, cfg, "2026-07-16")
    validate(summary, metrics, schema_dir)
    assert summary["row_count"] == 4
    assert summary["passed"] is False        # fixture has a planted duplicate
