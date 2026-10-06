"""ML stage 1: inventory the feature tables and decide whether to proceed.

    python -m trust_score_05.ml.jobs.discovery_job \
      --config conf/ml/base.yaml --config conf/ml/local.yaml

Reads no data — only each prefix's schema — so it is the cheapest possible way
to find out that a configured month is absent, that a rename upstream has taken
the join key away, or that the fraud and non-fraud extracts no longer share a
numeric column. All three are conditions under which every later stage produces
numbers rather than errors, which is the expensive way to find out.

The job's whole contribution over :func:`trust_score_05.ml.discovery.run_discovery`
is the marker. ``_READY.json`` is written when, and only when, no check of
severity ``error`` failed; otherwise ``_FAILED.json`` goes down with the failing
checks inside it, and the exit status is non-zero so the orchestrator does not
launch the sweep behind it. The notebook wrote ``_READY.json`` unconditionally at
the end of its discovery cell — a run in which every prefix failed to read
produced a green marker and eleven empty reports, and the stages downstream
believed it.

Reports are written either way, before the marker is chosen. A discovery that
refuses to pass is the run most in need of its diagnostics.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from trust_score_05.common.io.s3 import write_marker
from trust_score_05.lineage.logging_utils import get_logger, log_banner
from trust_score_05.ml.discovery import run_discovery, write_discovery_reports
from trust_score_05.ml.jobs.base import (
    JobResult,
    MLRunContext,
    build_arg_parser,
    main,
    run_job,
)
from trust_score_05.ml.paths import schema_prefix

LOGGER = get_logger(__name__)

JOB_NAME = "ml_schema_discovery"

__all__ = ["JOB_NAME", "build_parser", "main_cli", "run", "run_stage"]


class DiscoveryGateFailed(RuntimeError):
    """Discovery ran and its error-severity checks did not pass.

    A distinct type so that the non-zero exit is attributable. The alternative —
    returning a summary with ``status="FAILED"`` and exiting 0 — is what makes a
    scheduler run the next step anyway.
    """


def build_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser(
        JOB_NAME,
        "ML stage 1: schema inventory and data-quality gates over the feature tables.",
    )
    parser.add_argument(
        "--allow-failed-checks",
        action="store_true",
        help=(
            "Write the reports and exit 0 even when an error-severity check "
            "failed. The marker is still _FAILED, so the sweep behind this still "
            "refuses to start; this only stops the step itself from failing. For "
            "inspecting a broken extract, not for scheduled runs."
        ),
    )
    return parser


def run_stage(ctx: MLRunContext) -> JobResult:
    """Inventory, check, write the reports, then choose the marker."""
    cfg = ctx.cfg
    prefix = schema_prefix(cfg)
    write_marker(prefix, "STARTED", {"run_id": cfg.run_id, "mode": cfg.mode})

    report, failures = run_discovery(ctx.spark, cfg)
    written = write_discovery_reports(report, prefix, failures)

    manifest = report.manifest()
    marker = "READY" if report.ok else "FAILED"
    write_marker(prefix, marker, {**manifest, "reports": written, "run_id": cfg.run_id})

    log_banner(LOGGER, f"discovery {marker} | {len(report.checks)} checks | {prefix}")
    for check in report.failures:
        LOGGER.error("FAILED %s: %s", check.name, check.message)
    for check in report.warnings:
        LOGGER.warning("warning %s: %s", check.name, check.message)

    summary = JobResult(
        status=marker,
        prefix=prefix,
        checks_total=len(report.checks),
        checks_failed=manifest["checks_failed"],
        errors=len(report.failures),
        warnings=len(report.warnings),
        unreadable_prefixes=len(failures),
        columns_seen=int(report.classification["column"].nunique())
        if not report.classification.empty
        else 0,
    )
    if not report.ok and not ctx.args.allow_failed_checks:
        # Raised after the summary has been assembled but not returned, so the
        # driver's `finally` still logs the counts. What an operator needs first
        # is how many checks failed, not the traceback.
        raise DiscoveryGateFailed(
            f"{len(report.failures)} error-severity checks failed; see {prefix}. "
            f"First: {report.failures[0].message}"
        )
    return summary


def run(argv: Optional[Sequence[str]] = None) -> JobResult:
    return run_job(JOB_NAME, build_parser(), run_stage, argv)


def main_cli(argv: Optional[Sequence[str]] = None) -> int:
    return main(JOB_NAME, build_parser(), run_stage, argv)


if __name__ == "__main__":  # pragma: no cover - entrypoint
    sys.exit(main_cli())
