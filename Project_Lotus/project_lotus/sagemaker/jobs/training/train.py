"""SageMaker Training entry point: data checks, feature selection, the model sweep,
and choosing the champion.

Four modes, selected with ``--mode`` (or the ``mode`` hyperparameter):

``prepare``
    Runs the package's ``discovery_job`` (schema inventory and data-quality
    gates; a failed gate stops here with a non-zero exit) and then its
    ``selection_job`` (ranks features and writes the per-model feature lists
    the sweep reads), both unchanged, in a local Spark session. The selection
    is written under this run's id, so the sweep can name it exactly.

``sweep``
    Runs one slice of the sweep - ``--arm-start``/``--arm-count`` - by calling
    ``trust_score_05.ml.jobs.sweep_job.main_cli`` unchanged. Every job of one
    fanned-out sweep must carry the same run id (``--run-id`` or the
    ``TS05_RUN_ID`` environment variable); the launcher sets it.

``finalize``
    Runs once, after every slice has finished. Needed because each slice writes
    the cross-arm leaderboard from *its own* arms only, all to the same
    ``sweep-summary/run_id=`` prefix, so after a fan-out that file holds
    whichever slice finished last. This mode:

    1. checks every arm of the run has a ``_READY.json`` (refuses otherwise,
       unless ``--allow-partial``);
    2. merges every arm's ``all_hyperparams_all_scenario_k_metrics.csv`` and ranks
       *configurations* across all arms with the package's own
       ``evaluation.leaderboard`` and the ``best_config`` criteria from config;
    3. loads the winning configuration's ``model.joblib`` bundle;
    4. draws the model's training sample again (same seed, same code) and
       writes a drift reference - the champion's training score distribution
       and a sample of its input features - that ``inference.py`` compares every
       nightly batch against;
    5. publishes ``champion/current.json`` (unless ``--no-promote``) and keeps a
       copy under ``champion/history/run_id=``.

``all``
    ``prepare``, then ``sweep`` over the whole grid, then ``finalize``, in this
    one job and pinned to this run's selection. For a small grid or a first
    rehearsal; a full sweep should be fanned out (see the Step Functions
    definition in ``terraform/``, which runs prepare -> N sweep slices ->
    finalize as separate jobs).

Nothing here re-implements a modelling step. Selection, preprocessing, fitting,
evaluation and the ranking all run inside the package exactly as they do on EMR.

Exit codes: 0 success, 2 configuration problem, 1 anything else. A sweep slice
exits with the package job's own code, which is 0 even when some arms FAILED
(its summary says PARTIAL) - that is deliberate, so one bad arm does not make
SageMaker retry a slice whose other arms are fine. ``finalize`` is the gate: it
refuses to choose a champion from a sweep with arms that are not READY.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from typing import Any, Dict, List, Optional, Sequence

import sm_common  # noqa: F401 - puts PROGRAM_DIR on sys.path before the package import
from sm_common import (
    CHAMPION_FILE,
    EXIT_CONFIG,
    EXIT_FAILED,
    EXIT_OK,
    REFERENCE_FEATURES_FILE,
    REFERENCE_SCORES_FILE,
    SM_MODEL_DIR,
    build_local_spark,
    champion_history_prefix,
    configure,
    container_argv,
    copy_object,
    current_champion_uri,
    read_csv_any,
    record_failure,
    resolve_configs,
    utc_run_id,
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trust_score_05.common.io.s3 import (  # noqa: E402
    join_uri,
    read_json,
    ready_exists,
    s3_exists,
    write_json,
    write_marker,
    write_pandas,
)
from trust_score_05.common.io.serialization import BUNDLE_FILENAME, load_bundle  # noqa: E402
from trust_score_05.lineage.config import ConfigError  # noqa: E402
from trust_score_05.lineage.logging_utils import get_logger, log_banner, log_mapping  # noqa: E402
from trust_score_05.ml.config import MLConfig, load_ml_config  # noqa: E402
from trust_score_05.ml.evaluation import SCORE_COLUMN, leaderboard  # noqa: E402
from trust_score_05.ml.jobs.base import (  # noqa: E402
    read_feature_selection,
    resolve_selection_run_id,
)
from trust_score_05.ml.jobs.sweep_job import ARM_GROUP_COLUMNS  # noqa: E402
from trust_score_05.ml.models import MODEL_NAMES  # noqa: E402
from trust_score_05.ml.paths import config_prefix, sweep_summary_prefix  # noqa: E402
from trust_score_05.ml.selection import METHOD_NAME  # noqa: E402
from trust_score_05.ml.sweep import SweepArm, sweep_arms  # noqa: E402

LOGGER = get_logger("ts05.sagemaker.train")

#: The run-level metric table every READY arm writes (``sweep.run_arm``).
ARM_K_METRICS = ("metrics", "all_hyperparams_all_scenario_k_metrics.csv")

#: What a configuration is identified by across the whole sweep: its arm, plus
#: its hyperparameter id within the arm.
CONFIG_GROUP_COLUMNS = tuple(ARM_GROUP_COLUMNS) + ("hyperparam_id",)


class FinalizeError(RuntimeError):
    """The sweep cannot be finalized: arms missing or failed, or nothing ranked."""


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Trust Score 0.5 SageMaker training: sweep slices and champion selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("prepare", "sweep", "finalize", "all"), default="all")
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="ML config overlay(s). Default: conf/ml/base.yaml + conf/ml/sandbox.yaml.",
    )
    parser.add_argument("--set", action="append", default=None, dest="overrides",
                        metavar="KEY=VALUE", help="Dotted-key override, as in the package jobs.")
    parser.add_argument(
        "--run-id",
        default=None,
        help="The sweep's run id. Default: TS05_RUN_ID, else a new UTC timestamp (mode=all only).",
    )
    parser.add_argument("--models", default=",".join(MODEL_NAMES))
    parser.add_argument("--method", default=METHOD_NAME)
    parser.add_argument("--selection-run-id", default=None)
    # prepare-only
    parser.add_argument("--allow-failed-checks", action="store_true",
                        help="Let selection run even when a discovery data-quality gate failed.")
    # sweep-only
    parser.add_argument("--arm-start", type=int, default=0)
    parser.add_argument("--arm-count", type=int, default=0)
    parser.add_argument("--max-minutes-per-arm", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    # finalize-only
    parser.add_argument("--allow-partial", action="store_true",
                        help="Finalize even if some arms are not READY. The gaps are recorded.")
    parser.add_argument("--no-promote", action="store_true",
                        help="Write the champion under history/ but do not move current.json.")
    parser.add_argument("--reference-rows", type=int, default=50_000,
                        help="Rows of the champion's training sample kept as the feature-drift reference.")
    # runtime
    parser.add_argument("--driver-memory", default=None,
                        help="spark.driver.memory for the in-container session. Default: 70%% of RAM.")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def _models(args: argparse.Namespace) -> List[str]:
    models = [name.strip() for name in str(args.models).split(",") if name.strip()]
    unknown = [name for name in models if name not in MODEL_NAMES]
    if unknown:
        raise ConfigError(f"unknown model names {unknown}; expected a subset of {MODEL_NAMES}")
    if not models:
        raise ConfigError("--models selected nothing")
    return models


def _pin_run_id(args: argparse.Namespace) -> str:
    """Decide the run id once and publish it the way the package reads it.

    Through ``TS05_RUN_ID`` rather than ``--run-id``, because that is the
    mechanism ``conf/ml/sandbox.yaml`` documents for a fanned-out sweep: the
    config leaves ``run.run_id`` null and every submission agrees via the
    environment.
    """
    run_id = args.run_id or os.environ.get("TS05_RUN_ID")
    if not run_id:
        if args.mode != "all":
            raise ConfigError(
                f"--mode {args.mode} needs the sweep's run id: pass --run-id or set "
                "TS05_RUN_ID, the same value on every slice and on the finalize job"
            )
        run_id = utc_run_id()
    os.environ["TS05_RUN_ID"] = run_id
    return run_id


def _load_cfg(args: argparse.Namespace) -> MLConfig:
    return load_ml_config(config_paths=resolve_configs(args.config),
                          overrides=list(args.overrides or []))


def _config_argv(args: argparse.Namespace) -> List[str]:
    """The ``--config``/``--set`` flags every package job takes, in order."""
    argv: List[str] = []
    for path in resolve_configs(args.config):
        argv += ["--config", path]
    for override in args.overrides or []:
        argv += ["--set", override]
    return argv


# --------------------------------------------------------------------------
# mode: prepare
# --------------------------------------------------------------------------

def run_prepare(args: argparse.Namespace) -> int:
    """Discovery (data-quality gates), then feature selection - the package's jobs.

    Each job stops its session on the way out, so a fresh local session is
    built before each. A discovery gate that fails exits non-zero and selection
    does not run, unless ``--allow-failed-checks``: training on a schema nobody
    validated is exactly what the gate exists to prevent.
    """
    from trust_score_05.ml.jobs import discovery_job, selection_job

    common = _config_argv(args) + ["--log-level", str(args.log_level)]

    build_local_spark("ts05_ml.ml_discovery", driver_memory=args.driver_memory)
    discovery_argv = list(common)
    if args.allow_failed_checks:
        discovery_argv.append("--allow-failed-checks")
    code = discovery_job.main_cli(discovery_argv)
    if code != EXIT_OK:
        return code

    build_local_spark("ts05_ml.ml_selection", driver_memory=args.driver_memory)
    selection_argv = common + ["--models", ",".join(_models(args)), "--method", str(args.method)]
    if args.force:
        selection_argv.append("--force")
    return selection_job.main_cli(selection_argv)


# --------------------------------------------------------------------------
# mode: sweep
# --------------------------------------------------------------------------

def run_sweep_slice(args: argparse.Namespace) -> int:
    """Delegate one slice to the package's sweep job, in a local Spark session."""
    from trust_score_05.ml.jobs import sweep_job

    argv = _config_argv(args)
    argv += [
        "--models", ",".join(_models(args)),
        "--method", str(args.method),
        "--arm-start", str(int(args.arm_start)),
        "--arm-count", str(int(args.arm_count)),
        "--max-minutes-per-arm", str(float(args.max_minutes_per_arm)),
        "--log-level", str(args.log_level),
    ]
    if args.selection_run_id:
        argv += ["--selection-run-id", str(args.selection_run_id)]
    if args.force:
        argv.append("--force")

    # Built here, before the job asks for one, so the job's getOrCreate attaches
    # to a session that is sized to the instance and can read s3://.
    build_local_spark("ts05_ml.ml_sweep", driver_memory=args.driver_memory)
    code = sweep_job.main_cli(argv)
    # The job returns 0 even when some arms FAILED (its summary says PARTIAL);
    # finalize is what refuses a partial sweep, so the slice itself is not
    # second-guessed here beyond the exit code.
    return code


# --------------------------------------------------------------------------
# mode: finalize
# --------------------------------------------------------------------------

def _arm_status(prefix: str) -> str:
    if ready_exists(prefix):
        return "READY"
    for marker in ("FAILED", "INCOMPLETE", "STARTED"):
        if s3_exists(join_uri(prefix, f"_{marker}.json")):
            return marker
    return "MISSING"


def audit_arms(cfg: MLConfig, models: Sequence[str], method: str,
               selection_run_id: Optional[str]) -> pd.DataFrame:
    """Every arm the sweep *should* have run, and the marker each one carries.

    The expected set is rebuilt the way ``sweep_job`` builds it - the full
    ``sweep_arms`` order, minus the arms whose feature count the model's
    selection does not hold - so an arm that no slice ever ran shows up as
    ``MISSING`` rather than simply not being looked for.
    """
    counts: Dict[str, List[int]] = {}
    selections: Dict[str, str] = {}
    for model in models:
        run_id = selection_run_id or resolve_selection_run_id(cfg, model, method)
        counts[model] = sorted(read_feature_selection(cfg, model, method, run_id=run_id))
        selections[model] = run_id

    rows = []
    for arm in sweep_arms(cfg, list(models), method):
        if arm.top_n_features not in counts[arm.model]:
            continue
        prefix = arm.prefix(cfg)
        rows.append({**arm.as_dict(), "prefix": prefix, "status": _arm_status(prefix),
                     "selection_run_id": selections[arm.model]})
    return pd.DataFrame(rows)


def merge_k_metrics(arms: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for prefix in arms.loc[arms["status"] == "READY", "prefix"]:
        uri = join_uri(prefix, *ARM_K_METRICS)
        if s3_exists(uri):
            frames.append(read_csv_any(uri))
        else:
            LOGGER.warning("READY arm has no run-level k-metrics table: %s", uri)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def rank_configurations(cfg: MLConfig, k_metrics: pd.DataFrame):
    table, summary = leaderboard(
        k_metrics,
        metric=cfg.best_config_metric,
        level=cfg.best_config_level,
        aggregate=cfg.best_config_aggregate,
        k_percent=cfg.best_config_k_percent,
        group_columns=CONFIG_GROUP_COLUMNS,
    )
    if table.empty:
        raise FinalizeError(
            f"no configuration has a usable {cfg.best_config_metric!r} at "
            f"level={cfg.best_config_level!r} (k_percent={cfg.best_config_k_percent}); "
            f"leaderboard summary: {summary}"
        )
    return table, summary


def build_reference(cfg: MLConfig, bundle: Any, prefix: str, rows: int,
                    driver_memory: Optional[str]) -> Dict[str, Any]:
    """The champion's drift reference: training scores and a feature sample.

    The training sample is drawn with the package's own loader and the run's
    seed, so it is the sample the champion was fitted on. Scores come from
    ``training_scores`` - the call the sweep's own drift report used - so the
    reference distribution is the one the sweep would have compared against.
    """
    from trust_score_05.ml.datasets import load_training_matrix

    spark = build_local_spark("ts05_ml.finalize_reference", driver_memory=driver_memory)
    try:
        frame, present, manifest = load_training_matrix(
            spark, cfg, bundle.features, join_uri(prefix, "reference_sample")
        )
    finally:
        spark.stop()

    matrix = bundle.preprocessor.transform(frame)
    scores = np.asarray(bundle.model.training_scores(matrix), dtype=float)
    if scores.shape != (len(frame),):
        raise FinalizeError(f"training_scores returned {scores.shape} for {len(frame)} rows")

    scores_uri = join_uri(prefix, REFERENCE_SCORES_FILE)
    write_pandas(pd.DataFrame({SCORE_COLUMN: scores}), scores_uri)

    sample = frame[list(bundle.features)].copy()
    sample[SCORE_COLUMN] = scores
    if rows and len(sample) > rows:
        sample = sample.sample(n=int(rows), random_state=int(cfg.seed))
    features_uri = join_uri(prefix, REFERENCE_FEATURES_FILE)
    write_pandas(sample.reset_index(drop=True), features_uri)

    return {
        "reference_scores_uri": scores_uri,
        "reference_features_uri": features_uri,
        "n_reference_scores": int(scores.size),
        "n_reference_feature_rows": int(len(sample)),
        "reference_features_present": len(present),
        "reference_months": int(len(manifest)),
    }


def _export_to_model_dir(champion: Dict[str, Any]) -> None:
    """Put the champion into ``/opt/ml/model`` so SageMaker packages it.

    SageMaker uploads that directory as ``model.tar.gz`` at the end of the job,
    which is what the Model Registry and any later SageMaker model object point
    at. Skipped outside a SageMaker container.
    """
    if not SM_MODEL_DIR.is_dir():
        return
    write_json(str(SM_MODEL_DIR / CHAMPION_FILE), champion)
    copy_object(champion["bundle_uri"], str(SM_MODEL_DIR / BUNDLE_FILENAME))
    LOGGER.info("champion exported to %s", SM_MODEL_DIR)


def run_finalize(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _load_cfg(args)
    models = _models(args)
    method = str(args.method)
    log_banner(LOGGER, f"finalize | {cfg.models_version} | run_id={cfg.run_id}")

    summary_prefix = join_uri(sweep_summary_prefix(cfg), "merged")
    arms = audit_arms(cfg, models, method, args.selection_run_id)
    if arms.empty:
        raise FinalizeError("the selection supplies no arm for this sweep's feature counts")
    write_pandas(arms, join_uri(summary_prefix, "arms.csv"))

    by_status = arms["status"].value_counts().to_dict()
    log_mapping(LOGGER, "arm status", by_status)
    not_ready = arms.loc[arms["status"] != "READY"]
    if not not_ready.empty and not args.allow_partial:
        raise FinalizeError(
            f"{len(not_ready)} of {len(arms)} arms are not READY ({by_status}); "
            "re-run their slices (a re-run resumes, it does not refit READY work) "
            "or pass --allow-partial to choose among the finished ones. First few: "
            + "; ".join(f"{r.model}/{r.top_n_features}/{r.preprocessing}={r.status}"
                        for r in not_ready.head(5).itertuples())
        )

    k_metrics = merge_k_metrics(arms)
    if k_metrics.empty:
        raise FinalizeError("no READY arm produced top-k metrics")
    table, board = rank_configurations(cfg, k_metrics)
    write_pandas(table, join_uri(summary_prefix, "config_leaderboard.csv"))

    best = table.iloc[0]
    arm = SweepArm(
        model=str(best["model"]),
        method=str(best["feature_selection_method"]),
        top_n_features=int(best["top_n_features"]),
        preprocessing_id=str(best["preprocessing"]),
    )
    cfg_prefix = config_prefix(arm.prefix(cfg), str(best["hyperparam_id"]))
    if not ready_exists(cfg_prefix):
        raise FinalizeError(f"the winning configuration is not READY: {cfg_prefix}")
    bundle_uri = join_uri(cfg_prefix, "model_artifacts", BUNDLE_FILENAME)
    bundle = load_bundle(bundle_uri)
    LOGGER.info("champion: %s | %s=%.4f", cfg_prefix, cfg.best_config_metric, float(best["value"]))

    history = champion_history_prefix(cfg, cfg.run_id)
    reference = build_reference(cfg, bundle, history, int(args.reference_rows), args.driver_memory)

    champion: Dict[str, Any] = {
        "run_id": cfg.run_id,
        "models_version": cfg.models_version,
        **arm.as_dict(),
        "hyperparam_id": str(best["hyperparam_id"]),
        "params": bundle.params,
        "features": list(bundle.features),
        "bundle_uri": bundle_uri,
        "config_prefix": cfg_prefix,
        "criteria": {
            "metric": cfg.best_config_metric,
            "level": cfg.best_config_level,
            "k_percent": cfg.best_config_k_percent,
            "aggregate": cfg.best_config_aggregate,
        },
        "value": float(best["value"]),
        "value_min": float(best.get("value_min", np.nan)),
        "value_spread": float(best.get("value_spread", np.nan)),
        "n_scenarios": int(best.get("n_scenarios", 0)),
        "n_configs_ranked": int(len(table)),
        "leaderboard": board,
        "arms_by_status": by_status,
        "partial": bool(not not_ready.empty),
        "drift_thresholds": {"score_psi": cfg.score_drift_psi_threshold},
        **reference,
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    current_uri = current_champion_uri(cfg)
    if s3_exists(current_uri):
        previous = read_json(current_uri)
        champion["previous_run_id"] = previous.get("run_id")
        champion["previous_value"] = previous.get("value")
    write_json(join_uri(history, CHAMPION_FILE), champion)
    write_json(join_uri(summary_prefix, CHAMPION_FILE), champion)

    promoted = not args.no_promote
    if promoted:
        write_json(current_uri, champion)
        LOGGER.info("promoted: %s", current_uri)
    write_marker(history, "READY", {"promoted": promoted, "bundle_uri": bundle_uri})
    _export_to_model_dir(champion)
    return {"status": "SUCCESS", "promoted": promoted, "champion": current_uri,
            "model": arm.model, "value": champion["value"]}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(container_argv(argv))
    configure(args.log_level)
    try:
        run_id = _pin_run_id(args)
        LOGGER.info("train.py | mode=%s | run_id=%s", args.mode, run_id)
        if args.mode in ("prepare", "all"):
            code = run_prepare(args)
            if code != EXIT_OK:
                record_failure(f"prepare (discovery/selection) exited {code}; see the job log")
                return code
            if args.mode == "all" and not args.selection_run_id:
                # Train on the selection this job just wrote, not "the latest".
                args.selection_run_id = run_id
        if args.mode in ("sweep", "all"):
            code = run_sweep_slice(args)
            if code != EXIT_OK:
                record_failure(f"sweep slice exited {code}; see the job log")
                return code
        if args.mode in ("finalize", "all"):
            result = run_finalize(args)
            log_mapping(LOGGER, "finalize result", result)
        return EXIT_OK
    except ConfigError as exc:
        LOGGER.error("configuration error: %s", exc)
        record_failure(f"configuration error: {exc}")
        return EXIT_CONFIG
    except Exception as exc:  # noqa: BLE001 - container boundary
        LOGGER.error("train.py failed: %s", exc, exc_info=True)
        record_failure(f"{type(exc).__name__}: {exc}")
        return EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
