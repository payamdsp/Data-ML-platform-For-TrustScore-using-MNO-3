"""Reusable silver DQ metrics module (§6.1.2).

Computes completeness / consistency / uniqueness / timeliness against any silver
Iceberg table from a per-table checks.yaml, with optional per-MNO breakdown, and
writes summary.json + metrics.json to the canonical poc/dq/results layout.
Phase 1 = summary-level only; row-level failed_rows.parquet is Phase 2.
"""

from .config import CheckConfig, load_checks, parse_checks
from .runner import (build_metrics, build_summary, compute_dq, read_schema,
                     result_paths, validate, write_results)

__all__ = [
    "CheckConfig",
    "load_checks",
    "parse_checks",
    "compute_dq",
    "build_summary",
    "build_metrics",
    "read_schema",
    "validate",
    "write_results",
    "result_paths",
]
