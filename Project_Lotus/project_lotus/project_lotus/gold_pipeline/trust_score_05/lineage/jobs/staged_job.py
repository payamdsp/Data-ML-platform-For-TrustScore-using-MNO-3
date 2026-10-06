"""Five sequential EMR applications sharing one logical lineage run.

prepare -> accounts -> customers -> validate -> publish

The transformations and DQ rules are the same functions as lineage_job.plan.
Private work columns survive each handoff; published Gold projections do not
contain enough evidence to substitute for those work frames. Only publish may
change the five Gold tables, and only after ALL output checks have completed.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import asdict
from types import SimpleNamespace

from pyspark.sql import functions as F

from ..config import ConfigError, load_config
from ..dq import (
    REFERENCE_TABLE, apply_severity_overrides, log_results, merge_stat_checks,
    raise_on_abort, results_to_dataframe, run_checks, summarize, write_metrics,
)
from ..dq.framework import counter_results, pack_results, unpack_results
from ..execution import (
    STAGES, TABLES, RunStore, current_snapshot, digest, pin_inputs, pinned_config, timed,
)
from ..io import RunControl, SliceScope, STATUS_SUCCEEDED, build_sink
from ..io.control import RunRecord
from ..io.merge_sink import MergeStats
from ..logging_utils import configure_logging, get_logger, log_mapping
from ..release import CODE_SHA256
from ..spark import build_spark_session
from ..timezone import DEFAULT_SESSION_TIMEZONE, pin_process_timezone
from ..transforms import accounts, customers, lifecycle
from ..transforms.common import windows_from_config
from ..transforms.normalized import build_normalized_events
from .base import (
    RunContext, WATERMARK_FLOOR, _run_table_checks, _severity_overrides,
    build_arg_parser, parse_ts, read_phone_number_scope,
)
from .lineage_job import JOB_NAME, _enrichment_input_checks, _report_checks, prepare_lineage

LOGGER = get_logger(__name__)


def _read_scope(store, prepared, cfg):
    info = prepared["slice"]
    return SliceScope(
        store.read_frame("prepare", "scope"), info["size"], info["hashes"],
        int(cfg.get("slice.inline_limit", 20_000)),
    )


def _prepare(ctx, store):
    """Materialize histories and work frames once, at the original rule order."""
    with timed(ctx.spark, "prepare.business_transformations", store.timings):
        p = prepare_lineage(ctx, materialize=store.write_frame)
    sliced = p.sliced
    info = {
        "size": sliced.size, "l0_size": sliced.l0_size,
        "iterations": sliced.iterations, "converged": sliced.converged,
        "tu_l0_size": sliced.tu_l0_size, "hashes": sliced.scope.hashes,
        "sizes_by_iteration": sliced.sizes_by_iteration,
    }
    if sliced.scope.is_empty():
        return {"slice": info, "no_data": True, "notes": {}}

    # Evaluate against exactly the same full enrichment histories as the old
    # driver. Save the small results; validate writes and gates them later.
    checks = _enrichment_input_checks(
        ctx, p.tu_batch, p.tu_history, p.activations, p.account_changes, p.device_history,
    )
    measured = {}
    for label, (frame, rules) in checks.items():
        rules = apply_severity_overrides(rules, _severity_overrides(ctx.cfg, label))
        with timed(ctx.spark, f"prepare.dq.{label}", store.timings):
            measured[label] = pack_results(run_checks(frame, rules))
    return {
        "slice": info, "no_data": False, "input_results": measured,
        "boundary_report": asdict(p.boundary_report), "tu_report": asdict(p.tu_report),
        "notes": {
            "tu_l0_size": sliced.tu_l0_size,
            "tu_lifecycles_built": p.tu_report.lifecycles_built,
            "tu_lifecycle_carriers": p.tu_report.unsupported_carriers,
            "starts_firmed": p.boundary_report.starts_firmed,
            "ends_firmed": p.boundary_report.ends_firmed,
        },
    }


def _accounts(ctx, store):
    """Same account graph/walk, starting from short Parquet scan plans."""
    with timed(ctx.spark, "accounts.graph_walk_and_assembly", store.timings):
        frame, report = accounts.build_accounts(
            ctx.spark, store.read_frame("prepare", "events"),
            store.read_frame("prepare", "lifecycles"),
            store.read_frame("prepare", "account_changes"),
            store.read_frame("prepare", "device_history"),
            windows=windows_from_config(ctx.cfg), cfg=ctx.cfg,
        )
    store.write_frame("accounts", frame)
    LOGGER.info(report.format_line())
    return {"account_report": asdict(report), "notes": {
        "account_edges": report.edge_count, "account_edges_by_route": report.edges_by_route,
        "accounts": report.account_count,
    }}


def _customers(ctx, store):
    """Preserve account work columns, event evidence and all porting rules."""
    with timed(ctx.spark, "customers.graph_walk_and_assembly", store.timings):
        frame, report = customers.build_customers(
            ctx.spark, store.read_frame("accounts", "accounts"),
            store.read_frame("prepare", "lifecycles"),
            store.read_frame("prepare", "events"), windows=windows_from_config(ctx.cfg),
        )
    store.write_frame("customers", frame)
    LOGGER.info(report.format_line())
    return {"customer_report": asdict(report), "notes": {
        "customers": report.customer_count, "port_edges": report.port_edge_count,
        "quiet_port_candidates": report.quiet_port_candidates,
    }}


def _write_dq(ctx, results, dataset, *, layer="gold"):
    if not results:
        return None
    metrics = results_to_dataframe(
        ctx.spark, results, run_id=ctx.run_id, dataset=dataset,
        # One stable audit clock across attempts. These columns describe the
        # logical run, not whichever EMR application happened to finish last.
        check_ts=ctx.run_ts, layer=layer,
        batch_id=f"({ctx.watermark_from}, {ctx.watermark_to}]",
    )
    return write_metrics(ctx.spark, ctx.cfg, metrics, idempotent=True)


def _validate(ctx, store):
    """Stage all five published shapes and run the unchanged complete DQ gate.

    Only the current output and its DQ reference are cached. Releasing them
    after each table bounds memory; subsequent reads are from durable staging.
    No source history or Python state machine is recomputed during this step.
    """
    prepared = store.require("prepare")
    events = store.read_frame("prepare", "events")
    # Canonical events already have their final published shape; reuse the
    # immutable asset instead of writing another copy of the same data.
    store.assets[TABLES[0]] = prepared["assets"]["events"]
    store.write_frame(TABLES[1], lifecycle.to_gold(
        store.read_frame("prepare", "lifecycles"), ctx.run_ts))
    store.write_frame(TABLES[2], accounts.to_gold(
        store.read_frame("accounts", "accounts"), ctx.run_ts))
    store.write_frame(TABLES[3], customers.to_gold(
        store.read_frame("customers", "customers"), ctx.run_ts))
    with timed(ctx.spark, "validate.normalize_and_materialize", store.timings):
        store.write_frame(TABLES[4], build_normalized_events(
            events, store.read_frame("customers", "customers"),
            store.read_frame("prepare", "device_history"), ctx.run_ts, cfg=ctx.cfg,
        ))

    # Reuse the original report-to-rule definitions rather than copying the
    # rule lists into another driver. JSON lists suffice for the report counts.
    extras = _report_checks(
        SimpleNamespace(**prepared["slice"]),
        SimpleNamespace(**store.require("accounts")["account_report"]),
        SimpleNamespace(**store.require("customers")["customer_report"]),
        SimpleNamespace(**prepared["boundary_report"]),
        SimpleNamespace(**prepared["tu_report"]),
    )
    input_results = []
    for label, packed in prepared["input_results"].items():
        results = unpack_results(packed)
        log_results(results, title=f"{label} data quality")
        _write_dq(ctx, results, label, layer="silver")
        input_results.extend(results)

    output_results, rows = [], {}
    for table in TABLES:
        frame = store.read_asset(store.assets[table]).persist()
        ref_name = REFERENCE_TABLE.get(table)
        reference = store.read_asset(store.assets[ref_name]).persist() if ref_name else None
        try:
            with timed(ctx.spark, f"validate.dq.{table}", store.timings):
                results = _run_table_checks(frame, table, ctx.cfg, reference, extras.get(table, ()))
            # run_checks already counts rows in its aggregate: no second count.
            rows[table] = results[0].total_rows
            output_results.extend(results)
        finally:
            frame.unpersist()
            if reference is not None:
                reference.unpersist()
    all_results = input_results + output_results
    target = _write_dq(ctx, output_results, JOB_NAME)
    return {
        "results": pack_results(all_results), "rows_by_table": rows,
        "passed": not any(r.blocking for r in all_results),
        "notes": {"rows_by_table": rows, **summarize(all_results), "dq_metrics_target": target},
    }


def _publish(ctx, store, state):
    """Publish validated immutable assets; record each committed table head.

    An interrupted MERGE may have committed without returning to Python. The
    journal distinguishes that uncertainty from a known completed table. Exact
    replay uses the same run timestamp, outputs and scoped MERGE/DELETE rules.
    """
    validated = store.require("validate")
    raise_on_abort(unpack_results(validated["results"]), JOB_NAME)
    if ctx.args.dry_run:
        return {"notes": {"status": "VALIDATED", "tables_merged": 0, "watermark_advanced": False}}
    scope = _read_scope(store, store.require("prepare"), ctx.cfg)
    sinks = {name: build_sink(name, ctx.cfg, ctx.spark) for name in TABLES}

    # Validate EVERY target before the first write, as the original driver did.
    # Check known completed heads too: an external writer must not slip into
    # the middle of a resumed publication unnoticed.
    for name, sink in sinks.items():
        sink.validate_existing_schema()
        journal = store.get(f"publish/{name}")
        expected = journal["head_after"] if journal and journal["status"] == "complete" else state["pins"]["sinks"][name]
        actual = current_snapshot(ctx.spark, ctx.cfg, state["pins"]["sinks"][name]["table"])
        if actual != expected:
            uncertain_own_write = journal and journal["status"] == "writing"
            if not (uncertain_own_write and ctx.args.resume_publish):
                extra = (
                    " The previous write may have committed. Confirm the old EMR application "
                    "has stopped and no other writer changed Gold, then retry publish with "
                    "--resume-publish to reapply this run's exact staged output."
                    if uncertain_own_write else " Another writer changed Gold; investigate before starting a new run."
                )
                raise ConfigError(f"Target snapshot changed for {name}.{extra}")

    stats_list = []
    for name, sink in sinks.items():
        prior = store.get(f"publish/{name}")
        if prior and prior["status"] == "complete":
            LOGGER.info("publish: already committed %s; skipping", name)
            stats_list.append(MergeStats(**prior["stats"]))
            continue

        def before_write(stats, table=name, existing=prior):
            # Keep the original planned effects if a crash required a replay.
            # A replay itself may report zero inserts because they already landed.
            store.put(f"publish/{table}", {
                "status": "writing", "stats": existing["stats"] if existing else asdict(stats),
            })

        frame = store.read_asset(validated["assets"][name])
        result = sink.merge(
            frame, scope, ctx.run_ts, source_rows=validated["rows_by_table"][name],
            before_write=before_write,
            retry_stats=MergeStats(**prior["stats"]) if prior else None,
        )
        journal = store.get(f"publish/{name}")
        original_stats = MergeStats(**journal["stats"])
        store.put(f"publish/{name}", {
            "status": "complete", "stats": asdict(original_stats),
            "last_attempt_stats": asdict(result),
            "head_after": current_snapshot(ctx.spark, ctx.cfg, state["pins"]["sinks"][name]["table"]),
            "timings_seconds": sink.timings,
        })
        store.timings.update(sink.timings)
        stats_list.append(original_stats)

    metrics = []
    for stats in stats_list:
        metrics.extend(counter_results(merge_stat_checks(stats, stats.table), stats.source_rows))
    with timed(ctx.spark, "publish.merge_metrics", store.timings):
        _write_dq(ctx, metrics, f"{JOB_NAME}_merge")
    return {"notes": {
        "status": "SUCCESS", "watermark_advanced": not bool(ctx.args.phone_numbers),
        "tables_merged": len(stats_list), "merges": [s.format_line() for s in stats_list],
        "merge_counts": {s.table: {key: getattr(s, key) for key in (
            "source_rows", "target_rows_before", "inserted", "updated", "deleted")}
            for s in stats_list},
    }}


def _initialize(ctx_args, cfg, spark, store, control):
    """Freeze one identity before reservation; recover even if start() lost its reply."""
    state = store.get("run")
    if state is not None:
        if state["config_sha256"] != digest(cfg.as_dict()) or state["code_sha256"] != CODE_SHA256:
            raise ConfigError("This run was staged with different code/config. Resume with its original files or use a new run ID.")
        for flag in ("first_run", "dry_run", "phone_numbers", "watermark_from", "watermark_to", "run_timestamp"):
            supplied = getattr(ctx_args, flag, None)
            if supplied and supplied != state["arguments"].get(flag):
                raise ConfigError(f"--{flag.replace('_', '-')} differs from this run's frozen arguments")
        return state
    if ctx_args.stage != "prepare":
        raise ConfigError("This run has no preparation manifest. Start with --stage prepare.")
    control.assert_no_running(JOB_NAME)
    # Reject reusing an ID from the old monolithic driver or a recovered run.
    if control.read().where((F.col("job_name") == JOB_NAME) & (F.col("run_id") == ctx_args.run_id)).take(1):
        raise ConfigError("Run ID already exists in gold_run_control; use a new ID")
    run_ts = parse_ts(ctx_args.run_timestamp) or dt.datetime.now()
    watermark_to = parse_ts(ctx_args.watermark_to) or run_ts
    watermark_from = parse_ts(ctx_args.watermark_from)
    if watermark_from is None and not ctx_args.phone_numbers:
        watermark_from = control.last_watermark(JOB_NAME)
    if any(v is not None and v.tzinfo is not None for v in (run_ts, watermark_from, watermark_to)):
        raise ConfigError("Use naive business-time timestamps; spark.session_timezone supplies their zone")
    if watermark_from is not None and watermark_to <= watermark_from:
        raise ConfigError("--watermark-to must be later than --watermark-from")
    held = control.last_watermark(JOB_NAME) or WATERMARK_FLOOR
    for name in TABLES:
        build_sink(name, cfg, spark).validate_existing_schema()
    repair_path = None
    if ctx_args.phone_numbers:
        # A repair file can itself be replaced between attempts. Freeze its
        # actual keys, not just the caller's mutable path, before starting.
        repair = read_phone_number_scope(
            spark, ctx_args.phone_numbers,
            inline_limit=int(cfg.get("slice.inline_limit", 20_000)),
        )
        store.begin("inputs")
        store.write_frame("repair_scope.parquet", repair.frame)
        repair_path = store.assets["repair_scope.parquet"]["path"]
    state = {
        "run_id": ctx_args.run_id, "run_ts": run_ts.isoformat(),
        "watermark_from": watermark_from.isoformat() if watermark_from else None,
        "watermark_to": watermark_to.isoformat(),
        "control_from": (held if ctx_args.phone_numbers else watermark_from),
        "control_to": held if ctx_args.phone_numbers else watermark_to,
        "arguments": {key: getattr(ctx_args, key) for key in (
            "first_run", "dry_run", "phone_numbers", "watermark_from", "watermark_to", "run_timestamp")},
        "pins": pin_inputs(spark, cfg), "config_sha256": digest(cfg.as_dict()),
        "code_sha256": CODE_SHA256,
        "repair_scope_path": repair_path,
    }
    store.put("run", state, create=True)
    # Use one representation whether this is the original attempt or a retry.
    return store.get("run")


def _reservation(control, state, dry_run):
    """Keep the logical job reserved across steps; failures remain resumable."""
    run_id = state["run_id"]
    control.assert_no_running(JOB_NAME, except_run_id=run_id)
    rows = control.read().where((F.col("job_name") == JOB_NAME) & (F.col("run_id") == run_id)).collect()
    statuses = {row["status"] for row in rows}
    if "failed" in statuses:
        raise ConfigError("This run was explicitly abandoned/recovered as failed. Use a new run ID; do not resume its staging.")
    if "succeeded" in statuses:
        return None, True
    if dry_run:
        return None, False
    running = [row for row in rows if row["status"] == "running"]
    if len(running) > 1:
        raise ConfigError("Duplicate running records; verify that EMR step concurrency is 1")
    if running:
        return RunRecord(**running[0].asDict()), False
    return control.start(
        JOB_NAME, parse_ts(state["control_from"]), parse_ts(state["control_to"]), run_id=run_id,
    ), False


def _summary(store, state):
    summary = {"job": JOB_NAME, "run_id": state["run_id"],
               "dry_run": state["arguments"]["dry_run"], "first_run": state["arguments"]["first_run"],
               "watermark_from": state["watermark_from"], "watermark_to": state["watermark_to"],
               "staging_uri": store.uri, "input_snapshots": state["pins"], "stage_timings_seconds": {}}
    for stage in STAGES:
        done = store.get(f"completed/{stage}")
        if done is None:
            continue
        summary.update(done.get("notes", {}))
        summary["stage_timings_seconds"][stage] = done.get("duration_seconds", 0)
        if stage == "prepare":
            info = done["slice"]
            summary.update(slice_l0=info["l0_size"], slice_size=info["size"],
                           slice_iterations=info["iterations"], slice_converged=info["converged"])
            if "scope" in done.get("assets", {}):
                summary["manifest"] = done["assets"]["scope"]["path"]
    # Sum successful step compute time; exclude operator waiting between steps.
    summary["duration_seconds"] = round(sum(summary["stage_timings_seconds"].values()), 2)
    return summary


def run(argv=None):
    parser = build_arg_parser(JOB_NAME)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--staging-root", help="Optional override of staging.root (an S3 prefix)")
    parser.add_argument("--run-timestamp", help="Optional fixed business clock for output comparisons; prepare only")
    parser.add_argument("--resume-publish", action="store_true", help="Replay an uncertain interrupted write after operator verification")
    args = parser.parse_args(argv)
    if not args.run_id:
        raise ConfigError("Every staged step requires the SAME explicit --run-id")
    if args.resume_publish and args.stage != "publish":
        raise ConfigError("--resume-publish applies only to publish")
    configure_logging(args.log_level)
    cfg = load_config(args.config, args.overrides)
    if cfg.get("run_control.mode", "iceberg") != "iceberg" or cfg.get("dq.output.mode", "iceberg") != "iceberg":
        raise ConfigError("Staged EMR execution requires Iceberg run-control and DQ tables")
    pin_process_timezone(str(cfg.get("spark.session_timezone", DEFAULT_SESSION_TIMEZONE)))
    root = args.staging_root or cfg.get("staging.root", None)
    if not root:
        root = str(cfg.require("run_control.manifest_root")).rstrip("/") + "/staged"
    spark = build_spark_session(cfg, app_name=f"{JOB_NAME}.{args.stage}")
    store = RunStore(spark, root, args.run_id)
    control = RunControl(spark, cfg)
    summary = {"job": f"{JOB_NAME}.{args.stage}", "run_id": args.run_id, "stage": args.stage}
    started = time.monotonic()
    try:
        control.ensure_exists()
        state = _initialize(args, cfg, spark, store, control)
        for name, value in state["arguments"].items():
            setattr(args, name, value)
        if state.get("repair_scope_path"):
            args.phone_numbers = state["repair_scope_path"]
        record, already_succeeded = _reservation(control, state, args.dry_run)
        done = store.get(f"completed/{args.stage}")
        if already_succeeded and done is None:
            raise ConfigError("Control says succeeded but this stage manifest is missing; investigate staging integrity")
        if done is None:
            index = STAGES.index(args.stage)
            if index:
                store.require(STAGES[index - 1])
            store.begin(args.stage)
            ctx = RunContext(
                spark, pinned_config(cfg, state["pins"]), args, JOB_NAME, args.run_id,
                parse_ts(state["run_ts"]), parse_ts(state["watermark_from"]), parse_ts(state["watermark_to"]),
            )
            prepared = store.get("completed/prepare")
            if prepared and prepared["no_data"]:
                notes = {}
                if args.stage == "publish":
                    notes = {"status": "VALIDATED_NO_DATA" if args.dry_run else "SUCCESS_NO_DATA",
                             "tables_merged": 0, "watermark_advanced": not args.dry_run and not args.phone_numbers}
                detail = {"no_data": True, "notes": notes}
            elif args.stage == "publish":
                detail = _publish(ctx, store, state)
            else:
                detail = {"prepare": _prepare, "accounts": _accounts,
                          "customers": _customers, "validate": _validate}[args.stage](ctx, store)
            detail["duration_seconds"] = round(time.monotonic() - started, 2)
            done = store.complete(detail)
        else:
            LOGGER.info("Stage %s already completed; reusing its immutable handoff", args.stage)

        # Failed DQ is itself a complete, reviewable result. Never overwrite it
        # or publish it. A config/business-rule change needs a fresh logical run.
        if args.stage == "validate" and not done.get("no_data"):
            raise_on_abort(unpack_results(done["results"]), JOB_NAME)
        if args.stage == "publish":
            if record is not None:
                size = store.require("prepare")["slice"]["size"]
                control.finish(record, STATUS_SUCCEEDED, slice_size=size)
            final = _summary(store, state)
            log_mapping(LOGGER, "logical run summary", final)
            control.write_summary(final)
        summary.update(status="SUCCESS", staging_uri=store.uri,
                       timings_seconds=done.get("timings_seconds", {}),
                       next_stage=STAGES[STAGES.index(args.stage) + 1] if args.stage != "publish" else None)
        return summary
    except Exception as exc:
        # Do NOT close the logical reservation: a failed stage is retryable with
        # the same ID. The existing recovery utility explicitly abandons a run.
        summary.update(status="FAILED", error={"type": type(exc).__name__, "message": str(exc)},
                       staging_uri=store.uri, retry_stage=args.stage,
                       timings_seconds=store.timings)
        raise
    finally:
        summary["duration_seconds"] = round(time.monotonic() - started, 2)
        log_mapping(LOGGER, "EMR stage summary", summary)
        try:
            control.write_summary(summary)
        except Exception:
            LOGGER.exception("Could not write stage summary; use the EMR driver log")
        spark.stop()


def main_cli(argv=None):
    try:
        run(argv)
    except ConfigError as exc:
        LOGGER.error("configuration error: %s", exc)
        return 2
    except Exception:
        LOGGER.exception("Gold stage failed")
        return 1
    return 0
