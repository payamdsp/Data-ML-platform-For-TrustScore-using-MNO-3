"""Shared job driver: CLI, run order, and the run summary.

Both Gold jobs are the same steps in the same order, so the order lives here
once and each job supplies only what is genuinely its own: its name, and the
function that turns a slice into a set of Gold frames. If a step is added,
neither job can quietly skip it.

Order of operations, and why
----------------------------
1.  **Config**, so a bad config fails before a cluster-sized session is built.
2.  **Spark session**, with the session timezone pinned.
3.  **Run control.** Refuse to start when a previous run of this job never
    finished, resolve the watermark window ``(from, to]`` and record the run as
    ``running`` before anything is read.
4.  **Batch read**, watermarked on ``ingestion_ts``.
5.  **Slice.** The batch names phone numbers; the closure adds the ones linked to
    them. This is the step that makes everything after it bounded.
6.  **Manifest**, written before any output is touched, so that a run which dies
    mid-merge leaves behind the list of phone numbers it was in the middle of.
7.  **Transform.** Full history for the slice, recomputed from scratch. Nothing
    downstream reads the previous Gold rows for those phone numbers - they are
    about to be replaced.
8.  **DQ** on the recomputed frames, then **write the metrics**, then
    **abort or merge**. Metrics go out before the abort so a blocked run still
    leaves evidence of why it was blocked.
9.  **Merge**, in lineage order, each one scoped to the slice.
10. **Merge metrics**, appended as a second batch of ``info`` rows, and the run
    closed out as succeeded or failed.

Two notes on that order.

The DQ step runs against the frames the run built, not against the tables after
the merge. The question DQ answers here is "did this pipeline build something
coherent", and reading the merged table back would fold the merge's own
behaviour into the answer and blur which of the two went wrong.

The merge statistics are therefore a second metrics write rather than part of the
first. They do not exist until the merges have run, and the merges must not run
until the first metrics write has happened. Splitting the write is the only order
that keeps both properties.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from pyspark.sql import DataFrame, SparkSession

from ..config import Config, ConfigError, load_config
from ..dq.framework import counter_results
from ..timezone import DEFAULT_SESSION_TIMEZONE, pin_process_timezone
from ..dq import (
    REFERENCE_TABLE,
    Check,
    CheckResult,
    apply_severity_overrides,
    checks_for,
    log_results,
    merge_stat_checks,
    raise_on_abort,
    results_to_dataframe,
    run_checks,
    summarize,
    write_metrics,
)
from ..io import (
    MergeStats,
    RunControl,
    SliceScope,
    STATUS_FAILED,
    STATUS_SUCCEEDED,
    build_sink,
    new_run_id,
)
from ..logging_utils import configure_logging, get_logger, log_banner, log_mapping
from ..schemas import SLICE_KEY
from ..slice import SliceResult
from ..spark import build_spark_session

LOGGER = get_logger(__name__)

__all__ = [
    "TS_FORMAT",
    "WATERMARK_FLOOR",
    "JobResult",
    "RunContext",
    "JobPlan",
    "build_arg_parser",
    "parse_args",
    "read_phone_number_scope",
    "run_gold_job",
    "main",
]

#: Watermark arguments and the DQ ``check_ts`` all use this format.
TS_FORMAT = "%Y-%m-%d %H:%M:%S"

#: The window a repair run records when no window has ever been consumed.
#: ``watermark_to`` is not nullable and a repair run must not advance the
#: watermark, so it records a floor low enough that it cannot raise the ``max()``
#: :meth:`RunControl.last_watermark` takes.
#:
#: The epoch and not ``datetime.min``: Spark refuses to write a timestamp from
#: before the Julian/Gregorian cutover to Parquet without a rebase mode, and no
#: event in this lake predates 1970 anyway, so as a lower bound the two are the
#: same value with one of them writable.
WATERMARK_FLOOR = _dt.datetime(1970, 1, 1)


class JobResult(dict):
    """The run summary. A plain dict, so it logs and serializes cleanly."""


# --------------------------------------------------------------------------
# what a job hands back
# --------------------------------------------------------------------------


@dataclass
class RunContext:
    """Everything a job's plan function is given.

    It is a value object on purpose. A plan function that needed to reach back
    into the driver for something would be a sign that the step order belongs in
    the job rather than here, and the whole point of this module is that it does
    not.
    """

    spark: SparkSession
    cfg: Config
    args: argparse.Namespace
    job_name: str
    run_id: str
    run_ts: _dt.datetime
    watermark_from: _dt.datetime | None
    watermark_to: _dt.datetime


@dataclass
class JobPlan:
    """What one run computed, before anything was written.

    ``outputs`` maps Gold table short name to the frame that should replace that
    table's rows *for this slice*. Order matters - the driver merges in the order
    given - so a job lists its tables in lineage order and a reader of the job
    file sees the dependency order without having to know it already.
    """

    scope: SliceScope
    outputs: dict[str, DataFrame] = field(default_factory=dict)
    slice_result: SliceResult | None = None
    extra_checks: dict[str, list[Check]] = field(default_factory=dict)
    #: Checks on what the run was *given*, keyed by a label for the feed. Each
    #: entry is the frame and the rules to evaluate against it. They are kept
    #: apart from ``extra_checks`` because they are about a different thing: an
    #: entry here says the input was malformed, and an entry there says the
    #: output is. Conflating the two would make the metrics table unable to
    #: answer which of the two a failed run was.
    input_checks: dict[str, tuple[DataFrame, list[Check]]] = field(
        default_factory=dict
    )
    notes: dict = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return self.scope is None or self.scope.is_empty()


PlanFn = Callable[[RunContext], JobPlan]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_arg_parser(job_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=job_name,
        description=f"Silver -> Gold job: {job_name}.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "Config file (YAML or JSON). Repeatable; later files overlay earlier "
            "ones. Defaults to conf/lineage/base.yaml plus conf/lineage/local.yaml."
        ),
    )
    parser.add_argument(
        "--set",
        action="append",
        default=None,
        dest="overrides",
        metavar="KEY=VALUE",
        help="Dotted-key override applied last, e.g. --set catalog.profile=glue",
    )
    parser.add_argument(
        "--watermark-from",
        default=None,
        metavar="'YYYY-MM-DD HH:MM:SS'",
        help=(
            "Lower bound of the batch window, exclusive. Defaults to the "
            "watermark_to of the last succeeded run of this job. Pass this to "
            "re-consume a window deliberately; re-consuming is always safe "
            "because every write is a merge."
        ),
    )
    parser.add_argument(
        "--watermark-to",
        default=None,
        metavar="'YYYY-MM-DD HH:MM:SS'",
        help="Upper bound of the batch window, inclusive. Defaults to now.",
    )
    parser.add_argument(
        "--phone-numbers",
        default=None,
        metavar="PATH_OR_LIST",
        help=(
            "Repair mode: take the slice from this comma-separated list of "
            "phone_number_AC_hash values, or from a file with one per line, "
            "instead of from the batch. The closure still runs on top of it."
        ),
    )
    parser.add_argument("--run-id", default=None, help="Override the generated run id.")
    parser.add_argument(
        "--first-run",
        action="store_true",
        help=(
            "Assert that this is the first incremental run after the base "
            "ownership build. Allows the initial all-history slice past the "
            "daily size ceiling; delete protection and every DQ check remain active."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build everything and run DQ, but merge nothing.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Python logging level for the driver.",
    )
    return parser


def parse_args(job_name: str, argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_arg_parser(job_name).parse_args(
        list(argv) if argv is not None else None
    )


def parse_ts(raw: str | None) -> _dt.datetime | None:
    if raw is None:
        return None
    try:
        return _dt.datetime.strptime(raw, TS_FORMAT)
    except ValueError:
        try:
            return _dt.datetime.fromisoformat(raw)
        except ValueError:
            raise ConfigError(
                f"could not read {raw!r} as a timestamp; expected "
                f"'{TS_FORMAT}' or an ISO-8601 value"
            ) from None


def _parse_phone_number_text(text: str) -> list[str]:
    """One hash per line, or the JSON payload a run manifest holds.

    The manifest is the artefact an operator reaches for first - it is the only
    record of what a failed run was in the middle of - so it has to be accepted
    here directly. Requiring somebody to extract the array from it by hand,
    during the exact incident the manifest exists for, is how a retry ends up
    scoped to the wrong set of phone numbers.
    """
    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return [str(h).strip() for h in payload.get(SLICE_KEY, []) if str(h).strip()]
        if isinstance(payload, list):
            return [str(h).strip() for h in payload if str(h).strip()]
    return [line.strip() for line in text.splitlines() if line.strip()]


def read_phone_numbers(raw: str | None) -> list[str]:
    """Read ``--phone-numbers`` as a path or an inline comma-separated list.

    A path may be a file - plain text or a JSON manifest - or the *directory*
    Spark wrote a manifest into. The manifest is written through Spark so that
    the same code path serves a local directory and ``s3://``, and the price of
    that is a directory of part files rather than the single ``slice.json`` the
    logged path implies. Accepting the directory is what makes the path in the
    run log usable as-is.
    """
    if not raw:
        return []
    path = Path(raw)

    if path.is_dir():
        # Spark's convention, and the one it reads back with: a leading ``_`` or
        # ``.`` marks metadata (``_SUCCESS``, the binary ``.crc`` sidecars).
        parts = sorted(
            p for p in path.rglob("*")
            if p.is_file() and not p.name.startswith(("_", "."))
        )
        if not parts:
            raise ConfigError(
                f"--phone-numbers {raw!r} is a directory with no manifest files in it"
            )
        text = "\n".join(p.read_text(encoding="utf-8") for p in parts)
    elif path.is_file():
        text = path.read_text(encoding="utf-8")
    else:
        return [part.strip() for part in raw.split(",") if part.strip()]

    hashes = _parse_phone_number_text(text)
    if not hashes:
        raise ConfigError(
            f"--phone-numbers {raw!r} contained no phone numbers. Expected one "
            f"{SLICE_KEY} per line, or a run manifest carrying a "
            f"{SLICE_KEY!r} array."
        )
    return hashes


def read_phone_number_scope(
    spark: SparkSession,
    raw: str | None,
    *,
    inline_limit: int,
) -> SliceScope | None:
    """Read either a small text list or a distributed resolver manifest.

    A path ending in ``.parquet`` is the scalable manifest written by
    :meth:`RunControl.write_manifest_frame`. It must be read through Spark: that
    is what makes an S3 handoff work and what keeps a first-run backlog out of
    driver memory. All existing text and JSON forms retain their small-list
    behaviour.
    """
    if not raw:
        return None
    if str(raw).rstrip("/").endswith(".parquet"):
        try:
            frame = spark.read.parquet(str(raw))
        except Exception as exc:  # noqa: BLE001 - turn a reader error into CLI context
            raise ConfigError(
                f"--phone-numbers could not read distributed manifest {raw!r}: {exc}"
            ) from exc
        if SLICE_KEY not in frame.columns:
            raise ConfigError(
                f"--phone-numbers manifest {raw!r} has no {SLICE_KEY!r} column; "
                f"found {frame.columns}"
            )
        scope = SliceScope.from_frame(frame, inline_limit=inline_limit)
        if scope.is_empty():
            raise ConfigError(
                f"--phone-numbers manifest {raw!r} contained no phone numbers"
            )
        return scope

    hashes = read_phone_numbers(raw)
    if not hashes:
        return None
    return SliceScope.from_hashes(spark, hashes, inline_limit=inline_limit)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def _severity_overrides(cfg: Config, table: str) -> dict:
    """Global overrides, then per-table ones on top.

    A global entry re-tiers a check wherever it appears - the slice-key checks
    are the same five rules on all six tables - and a per-table entry re-tiers it
    on one table only. The per-table entry wins, which is the order an operator
    expects from anything else that layers.
    """
    merged = dict(cfg.get("dq.severity_overrides", {}) or {})
    merged.update(cfg.get(f"dq.severity_overrides_by_table.{table}", {}) or {})
    return merged


def _run_table_checks(
    frame: DataFrame,
    table: str,
    cfg: Config,
    reference: DataFrame | None,
    extra: Sequence[Check] = (),
) -> list[CheckResult]:
    checks = list(checks_for(table, reference)) + list(extra)
    checks = apply_severity_overrides(checks, _severity_overrides(cfg, table))
    results = list(run_checks(frame, checks))
    log_results(results, title=f"{table} data quality")
    return results


def _write_results(
    spark: SparkSession,
    cfg: Config,
    results: Sequence[CheckResult],
    run_id: str,
    dataset: str,
    check_ts: _dt.datetime,
    batch_id: str | None,
    layer: str = "gold",
) -> str | None:
    if not results:
        return None
    metrics = results_to_dataframe(
        spark,
        results,
        run_id=run_id,
        dataset=dataset,
        check_ts=check_ts,
        layer=layer,
        batch_id=batch_id,
    )
    return write_metrics(spark, cfg, metrics)


def run_gold_job(
    job_name: str,
    plan_fn: PlanFn,
    argv: Sequence[str] | None = None,
) -> JobResult:
    """Run one Silver -> Gold batch end to end. Returns the run summary."""
    args = parse_args(job_name, argv)
    configure_logging(args.log_level)

    started = time.time()

    # Config first, then the clock. `run_ts` becomes this run's `watermark_to`,
    # and it reaches Spark as a Python `datetime` through `F.lit`, which converts
    # it using the *driver's* zone - while the Silver `ingestion_ts` it is
    # compared against was written as a string parsed in the session zone. Two
    # zones there means the window `(from, to]` selects nothing and the job still
    # reports SUCCESS. See timezone.py, which is the whole argument.
    cfg = load_config(config_paths=args.config, overrides=args.overrides)
    pin_process_timezone(cfg.get("spark.session_timezone", DEFAULT_SESSION_TIMEZONE))

    run_ts = _dt.datetime.now()
    run_id = args.run_id or f"{job_name}-{new_run_id()}"

    log_banner(LOGGER, f"{job_name} | silver -> gold | run_id={run_id}")
    if args.first_run:
        log_banner(
            LOGGER,
            "FIRST-RUN MODE ASSERTED | all-history slice allowed | "
            "delete-ratio and DQ guards remain active",
        )

    spark = build_spark_session(cfg, app_name=job_name)
    summary = JobResult(
        run_id=run_id,
        job=job_name,
        dry_run=bool(args.dry_run),
        first_run=bool(args.first_run),
    )
    control = RunControl(spark, cfg)
    record = None

    try:
        # -- 3. run control and the watermark window -----------------------
        control.ensure_exists()
        control.assert_no_running(job_name)
        watermark_to = parse_ts(args.watermark_to) or run_ts
        watermark_from = parse_ts(args.watermark_from)
        if watermark_from is None and not args.phone_numbers:
            watermark_from = control.last_watermark(job_name)
        summary.update(watermark_from=watermark_from, watermark_to=watermark_to)
        if watermark_from is not None and watermark_to <= watermark_from:
            raise ConfigError("--watermark-to must be later than --watermark-from")

        log_mapping(
            LOGGER,
            "run parameters",
            {
                "run_id": run_id,
                "window": f"({watermark_from}, {watermark_to}]",
                "config sources": ", ".join(cfg.sources) or "(defaults)",
                "catalog.profile": cfg.get("catalog.profile", "local"),
                "repair slice": bool(args.phone_numbers),
                "first_run": args.first_run,
                "watermark": (
                    "held, this is a validation or repair run"
                    if args.phone_numbers or args.dry_run
                    else f"advances to {watermark_to}"
                ),
                "dry_run": args.dry_run,
            },
        )
        # A repair run answers an explicit list of phone numbers, not a window,
        # so the window it *records* has to be zero-width: the watermark must
        # come out of it exactly where it went in. Advancing it to now would
        # permanently skip every phone number that changed inside the current
        # window and was not on the list - the failure would be silent, and the
        # run that caused it would read as a success.
        #
        # The run is still recorded. It can fail, and ``assert_no_running`` has
        # to be able to see it. ``watermark_to`` is not nullable, so a repair run
        # made before any window has ever been consumed records the floor rather
        # than a null: it is the smallest value ``max()`` can return, so it
        # cannot move a watermark either.
        #
        # Only the recorded window is pinned. The run still carries the real
        # window in its ``RunContext``, because a repair run reads the complete
        # history of its phone numbers and ignores the watermark anyway - but
        # the batch selection and the manifest's ``batch_id`` both read it, and
        # a window forced to zero width there would describe the run wrongly.
        if args.phone_numbers:
            held = control.last_watermark(job_name) or WATERMARK_FLOOR
            control_from, control_to = held, held
        else:
            control_from, control_to = watermark_from, watermark_to
        # Validation reads the last publish watermark but must never write a
        # successful control row: that would consume a window without publishing.
        # DQ metrics are still written, so failed validation is diagnosable.
        if not args.dry_run:
            record = control.start(job_name, control_from, control_to, run_id=run_id)

        ctx = RunContext(
            spark=spark,
            cfg=cfg,
            args=args,
            job_name=job_name,
            run_id=run_id,
            run_ts=run_ts,
            watermark_from=watermark_from,
            watermark_to=watermark_to,
        )

        # -- 4-7. batch, slice, transform ----------------------------------
        plan = plan_fn(ctx)
        summary.update(plan.notes)
        if plan.slice_result is not None:
            summary.update(
                slice_l0=plan.slice_result.l0_size,
                slice_size=plan.slice_result.size,
                slice_iterations=plan.slice_result.iterations,
                slice_converged=plan.slice_result.converged,
            )

        if plan.empty:
            # Not a failure. A quiet window is the normal state of this pipeline
            # outside business hours, and the watermark still advances - there is
            # nothing in the window to come back for.
            LOGGER.info("no phone numbers in the window; nothing to rebuild")
            summary.update(
                status="VALIDATED_NO_DATA" if args.dry_run else "SUCCESS_NO_DATA",
                slice_size=0, tables_merged=0,
                watermark_advanced=not args.dry_run and not args.phone_numbers,
            )
            if record is not None:
                control.finish(record, STATUS_SUCCEEDED, slice_size=0)
            record = None
            return summary

        scope = plan.scope
        summary["slice_size"] = scope.size

        # -- 6. manifest, before anything is written -----------------------
        if record is not None and scope.inlineable:
            summary["manifest"] = control.write_manifest(
                record, scope.hashes or (), extra={"job": job_name}
            )
        elif record is not None:
            summary["manifest"] = control.write_manifest_frame(
                record, scope.frame, row_count=scope.size, extra={"job": job_name}
            )

        # -- 8. data quality ------------------------------------------------
        # Inputs first. They are written under their own dataset name so that a
        # failed run is attributable: "the feed we were sent was malformed" and
        # "the rows we built are malformed" are different incidents with
        # different owners, and one query has to be able to tell them apart.
        #
        # The rebuild has already run by the time these are evaluated, which
        # costs the run its compute but nothing else - the abort still lands
        # before the first merge, so Gold is exactly as the previous run left
        # it. Paying that is preferable to teaching each job to run its own DQ
        # step, which is the thing this module exists to prevent.
        input_results: list[CheckResult] = []
        for label, (frame, checks) in plan.input_checks.items():
            checks = apply_severity_overrides(checks, _severity_overrides(cfg, label))
            this_feed = list(run_checks(frame, checks))
            log_results(this_feed, title=f"{label} data quality")
            _write_results(
                spark,
                cfg,
                this_feed,
                run_id=run_id,
                dataset=label,
                check_ts=_dt.datetime.now(),
                batch_id=f"({watermark_from}, {watermark_to}]",
                layer="silver",
            )
            input_results += this_feed

        cached: dict[str, DataFrame] = {}
        for table, frame in plan.outputs.items():
            cached[table] = frame.cache()

        output_results: list[CheckResult] = []
        rows_by_table: dict[str, int] = {}
        for table, frame in cached.items():
            reference = cached.get(REFERENCE_TABLE.get(table, ""), None)
            table_results = _run_table_checks(
                frame, table, cfg, reference, plan.extra_checks.get(table, ())
            )
            rows_by_table[table] = table_results[0].total_rows if table_results else frame.count()
            output_results += table_results
        results = input_results + output_results
        summary["rows_by_table"] = rows_by_table
        summary.update(summarize(results))

        check_ts = _dt.datetime.now()
        summary["dq_metrics_target"] = _write_results(
            spark,
            cfg,
            output_results,
            run_id=run_id,
            dataset=job_name,
            check_ts=check_ts,
            batch_id=f"({watermark_from}, {watermark_to}]",
        )

        # -- 8b. abort or merge ---------------------------------------------
        raise_on_abort(results, job_name)

        # Check every existing output contract before writing the first table.
        # A previously provisioned target with incompatible types should fail
        # here, not leave a half-published set of Gold tables.
        sinks = {table: build_sink(table, cfg, spark) for table in cached}
        for sink in sinks.values():
            validate = getattr(sink, "validate_existing_schema", None)
            if validate is not None:
                validate()

        if args.dry_run:
            LOGGER.warning("--dry-run: merging nothing")
            summary.update(status="VALIDATED", tables_merged=0, watermark_advanced=False)
            return summary

        # -- 9. merge, in lineage order --------------------------------------
        merge_stats: list[MergeStats] = []
        for table, frame in cached.items():
            sink = sinks[table]
            merge_stats.append(sink.merge(frame, scope, run_ts, source_rows=rows_by_table[table]))
        summary["tables_merged"] = len(merge_stats)
        summary["merges"] = [s.format_line() for s in merge_stats]
        summary["merge_counts"] = {
            s.table: {
                "source_rows": s.source_rows,
                "target_rows_before": s.target_rows_before,
                "inserted": s.inserted,
                "updated": s.updated,
                "deleted": s.deleted,
            }
            for s in merge_stats
        }

        # -- 10. merge metrics, then close out --------------------------------
        # Both the effects and the source row count were already measured.
        # Reuse them instead of scanning each output again for counter metrics.
        # These are all ``info``, so none of them can fail a run.
        merge_results: list[CheckResult] = []
        for stats in merge_stats:
            merge_results += counter_results(merge_stat_checks(stats, stats.table), stats.source_rows)
        _write_results(
            spark,
            cfg,
            merge_results,
            run_id=run_id,
            dataset=f"{job_name}_merge",
            check_ts=_dt.datetime.now(),
            batch_id=f"({watermark_from}, {watermark_to}]",
        )

        control.finish(record, STATUS_SUCCEEDED, slice_size=scope.size)
        record = None
        summary["status"] = "SUCCESS"
        summary["watermark_advanced"] = not bool(args.phone_numbers)
        return summary

    except Exception as exc:
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if record is not None:
            # Reached only when something raised between ``start`` and one of the
            # ``finish`` calls. Marking the run failed is what keeps the next
            # run's watermark where it is, so this window is re-consumed rather
            # than skipped.
            try:
                control.finish(record, STATUS_FAILED, slice_size=summary.get("slice_size", 0))
            except Exception:  # pragma: no cover - never mask the original error
                LOGGER.exception("could not record the run as failed")
        summary["duration_seconds"] = round(time.time() - started, 2)
        summary.setdefault("status", "FAILED")
        log_mapping(LOGGER, f"{job_name} run summary", summary)
        try:
            summary["summary_uri"] = control.write_summary(summary)
        except Exception:
            # Diagnostic-output failure must not disguise the original error
            # or turn an already committed publish into a misleading retry.
            LOGGER.exception("could not write summary.json; use the EMR driver log")
        if bool(cfg.get("runtime.stop_spark", True)):
            spark.stop()


def main(
    job_name: str,
    plan_fn: PlanFn,
    argv: Sequence[str] | None = None,
) -> int:
    """Entrypoint wrapper: turn exceptions into a non-zero exit code.

    ``spark-submit`` reports failure from the driver's exit status, so a DQ abort
    and a delete-ratio veto reach the orchestrator the same way any other failure
    does. Exit 2 is reserved for configuration errors, which are worth telling
    apart because they are fixed by editing a file rather than by looking at data.
    """
    try:
        run_gold_job(job_name, plan_fn, argv)
    except ConfigError as exc:
        LOGGER.error("configuration error: %s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001 - top-level driver boundary
        LOGGER.error("%s failed: %s", job_name, exc, exc_info=True)
        return 1
    return 0
