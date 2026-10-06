"""The CLI is the module's scheduled entry point, so its exit code is a contract:
0 = trustworthy, 1 = a rule failed (results still written), 2 = could not run.

Everything goes through main(argv), not a subprocess: main() calls
SparkSession.builder.getOrCreate(), which hands back the session conftest already built.
"""

import json
import os

import pytest

from silver_dq.__main__ import EXIT_ERROR, EXIT_FAILED_RULES, EXIT_PASS, main

RUN_DATE = "2026-07-16"


def result_files(dq_root):
    base = os.path.join(dq_root, "results", "dataset=account_changes_batch", f"run_date={RUN_DATE}")
    return os.path.join(base, "summary.json"), os.path.join(base, "metrics.json")


@pytest.fixture
def clean_table(account_changes_df):
    """--table needs a name whose last segment equals the yaml's dataset.

    The shared fixture plants a duplicate record_id on purpose; drop it so this frame
    is genuinely clean and a red verdict here would mean a real regression.
    """
    account_changes_df.dropDuplicates(["record_id"]).createOrReplaceTempView("account_changes_batch")
    return "account_changes_batch"


def test_cli_writes_both_files_and_exits_pass(clean_table, acct_yaml, tmp_path):
    code = main(["--checks", acct_yaml, "--table", clean_table,
                 "--dq-root", str(tmp_path), "--run-date", RUN_DATE])
    summary_path, metrics_path = result_files(str(tmp_path))
    assert code == EXIT_PASS
    assert json.load(open(summary_path))["passed"] is True
    assert json.load(open(metrics_path))["dataset"] == "account_changes_batch"


def test_cli_failed_rule_exits_one_and_still_writes(account_changes_df, acct_yaml, tmp_path):
    """A red run must leave the artifact behind; the red summary IS the signal Sprint 2 reads."""
    account_changes_df.createOrReplaceTempView("account_changes_batch")  # the planted duplicate
    code = main(["--checks", acct_yaml, "--table", "account_changes_batch",
                 "--dq-root", str(tmp_path), "--run-date", RUN_DATE])
    summary_path, _ = result_files(str(tmp_path))
    summary = json.load(open(summary_path))
    assert code == EXIT_FAILED_RULES
    assert summary["passed"] is False
    assert any(r["family"] == "uniqueness" and not r["passed"] for r in summary["rules"])


def test_cli_rejects_table_and_checks_mismatch(clean_table, tmp_path):
    """Wrong pair must not write: output is filed under the YAML's dataset, so this would
    put account_changes_batch's numbers under tu_portps's name."""
    code = main(["--checks", "checks/dataset=tu_portps/checks.yaml", "--table", clean_table,
                 "--dq-root", str(tmp_path), "--run-date", RUN_DATE])
    assert code == EXIT_ERROR
    assert not os.path.exists(os.path.join(str(tmp_path), "results"))


def test_cli_dry_run_reports_without_writing(clean_table, acct_yaml, tmp_path):
    code = main(["--checks", acct_yaml, "--table", clean_table, "--dq-root", str(tmp_path),
                 "--run-date", RUN_DATE, "--dry-run"])
    assert code == EXIT_PASS
    assert not os.path.exists(os.path.join(str(tmp_path), "results"))


def test_cli_empty_table_writes_a_red_verdict(account_changes_df, acct_yaml, tmp_path):
    """Unlike the notebook (which stops early for a human at the keyboard), the CLI runs an
    empty table through: a scheduler needs the red artifact, not a missing one."""
    account_changes_df.limit(0).createOrReplaceTempView("account_changes_batch")
    code = main(["--checks", acct_yaml, "--table", "account_changes_batch",
                 "--dq-root", str(tmp_path), "--run-date", RUN_DATE])
    summary_path, _ = result_files(str(tmp_path))
    summary = json.load(open(summary_path))
    assert code == EXIT_FAILED_RULES
    assert summary["row_count"] == 0
    assert [r["rule"] for r in summary["rules"] if not r["passed"]] == ["table is not empty"]


def test_cli_validates_against_the_repo_schema(clean_table, acct_yaml, schema_dir, tmp_path):
    """--schema-dir takes the repo copy here; on a cluster it takes the s3:// published copy."""
    code = main(["--checks", acct_yaml, "--table", clean_table, "--dq-root", str(tmp_path),
                 "--run-date", RUN_DATE, "--schema-dir", schema_dir])
    assert code == EXIT_PASS
