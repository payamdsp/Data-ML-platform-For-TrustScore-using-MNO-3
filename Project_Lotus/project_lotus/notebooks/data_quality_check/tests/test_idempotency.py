"""Idempotency test.

Rerunning DQ against the same silver snapshot must yield identical dicts and,
once serialized, byte-identical files. compute_dq takes run_date as an argument
and never reads wall-clock time, and write_results serializes with sort_keys.
"""

from silver_dq import compute_dq, load_checks, write_results

RUN_DATE = "2026-06-22"


def test_compute_is_deterministic(account_changes_df, acct_yaml):
    cfg = load_checks(acct_yaml)
    s1, m1 = compute_dq(account_changes_df, cfg, RUN_DATE)
    s2, m2 = compute_dq(account_changes_df, cfg, RUN_DATE)

    assert s1 == s2
    assert m1 == m2


def test_written_bytes_identical(account_changes_df, acct_yaml, tmp_path):
    cfg = load_checks(acct_yaml)
    summary, metrics = compute_dq(account_changes_df, cfg, RUN_DATE)

    root_a = str(tmp_path / "run_a")
    root_b = str(tmp_path / "run_b")
    sa, ma = write_results(summary, metrics, root_a, cfg, RUN_DATE)
    sb, mb = write_results(summary, metrics, root_b, cfg, RUN_DATE)

    assert open(sa, "rb").read() == open(sb, "rb").read()
    assert open(ma, "rb").read() == open(mb, "rb").read()
