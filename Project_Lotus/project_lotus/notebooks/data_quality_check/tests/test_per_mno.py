"""Per-MNO breakdown test.

A mixed-MNO fixture (BELL x2, ROGERS, TELUS) must produce a per-MNO block for
each carrier in addition to the overall block, and the per-MNO numbers must
reflect each carrier's own rows (BELL owns the duplicate; the others are clean).
"""

from silver_dq import compute_dq, load_checks

RUN_DATE = "2026-06-22"


def test_per_mno_breakdown(account_changes_df, acct_yaml):
    cfg = load_checks(acct_yaml)
    _summary, metrics = compute_dq(account_changes_df, cfg, RUN_DATE)

    per_mno = metrics["per_mno"]
    assert set(per_mno) == {"BELL", "ROGERS", "TELUS"}

    # overall still computed
    assert metrics["overall"]["row_count"] == 4

    # row counts per carrier
    assert per_mno["BELL"]["row_count"] == 2
    assert per_mno["ROGERS"]["row_count"] == 1
    assert per_mno["TELUS"]["row_count"] == 1

    # the duplicate record_id lives entirely within BELL
    assert per_mno["BELL"]["uniqueness"]["metrics"]["duplicate_count"] == 1
    assert per_mno["ROGERS"]["uniqueness"]["metrics"]["duplicate_count"] == 0
    assert per_mno["TELUS"]["uniqueness"]["metrics"]["duplicate_count"] == 0
