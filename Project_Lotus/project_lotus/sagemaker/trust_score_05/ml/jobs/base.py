"""What the three ML jobs share: the CLI, the session, and the run summary.

Three things live here rather than in each job.

The **argument parser**, because the four flags every job takes have to mean the
same thing in all three. ``--config`` is repeatable and later files overlay
earlier ones; ``--set`` applies after the files; ``--run-id`` overrides both.
That is the same contract as
:func:`trust_score_05.lineage.jobs.base.build_arg_parser`, deliberately — an
operator moving between the Gold jobs and these should not have to relearn it.

The **SparkSession**, which is much smaller than the lineage one. These jobs
read parquet by path and write parquet, CSV and JSON by path; they resolve no
table names, so they need no Iceberg catalog and no catalog profile. Everything
else an operator wants to tune is a ``--conf`` on the ``spark-submit`` and needs
no code here to pass it through. The one thing the session must get right is its
timezone — see :attr:`~trust_score_05.ml.config.MLConfig.spark_session_timezone`.

The **feature-selection artifact contract**, because it is the one thing two of
the jobs have to agree on: ``selection_job`` writes it and ``sweep_job`` reads
it. Keeping :func:`write_feature_selection` and :func:`read_feature_selection`
next to each other is what makes it hard for one side to change shape without
the other. It is not in :mod:`trust_score_05.ml.selection`, which stays free of
object-store imports so that its ranking can be tested on a bare frame.

The **exit codes** follow the lineage jobs: 0 success, 2 a configuration error,
1 anything else. ``spark-submit`` reports the driver's status to the
orchestrator, so a discovery gate that refuses to pass reaches the scheduler the
same way a crash does — which is the point, since a run that proceeded on a
schema nobody validated is the failure being prevented.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from trust_score_05.common.io.s3 import (
    join_uri,
    list_common_prefixes,
    read_json,
    s3_exists,
    write_json,
    write_pandas,
)
from trust_score_05.lineage.config import ConfigError
from trust_score_05.lineage.logging_utils import (
    configure_logging,
    get_logger,
    log_banner,
    log_mapping,
)
from trust_score_05.ml.config import MLConfig, load_ml_config
from trust_score_05.ml.paths import feature_selection_prefix
from trust_score_05.ml.selection import selection_manifest

LOGGER = get_logger(__name__)

__all__ = [
    "APP_NAME_PREFIX",
    "RANKED_FEATURES_FILE",
    "SELECTED_FEATURES_FILE",
    "SELECTION_MANIFEST_FILE",
    "JobResult",
    "MLRunContext",
    "build_arg_parser",
    "build_spark_session",
    "load",
    "main",
    "parse_args",
    "read_feature_selection",
    "resolve_selection_run_id",
    "run_job",
    "write_feature_selection",
]

#: Prefixed onto every job's Spark application name, so a cluster's application
#: list groups this pipeline's steps together.
APP_NAME_PREFIX = "ts05_ml"

#: The three objects one model's feature selection consists of. Named as
#: constants because :mod:`trust_score_05.ml.jobs.sweep_job` reads two of them
#: back and a mismatched filename would present as "no selection was ever run".
RANKED_FEATURES_FILE = "ranked_features.csv"
SELECTED_FEATURES_FILE = "selected_features.json"
SELECTION_MANIFEST_FILE = "selection_manifest.json"


class JobResult(dict):
    """The run summary. A plain dict, so it logs and serializes cleanly."""


@dataclass
class MLRunContext:
    """Everything a job's ``run`` function is given.

    ``spark`` is typed loosely on purpose: this module must be importable — and
    its parser exercisable — on a machine with no pyspark, which is how
    ``--help`` and the config validation are tested without a cluster.
    """

    spark: Any
    cfg: MLConfig
    args: argparse.Namespace
    job_name: str


RunFn = Callable[[MLRunContext], JobResult]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_arg_parser(job_name: str, description: str) -> argparse.ArgumentParser:
    """The flags common to every ML job. Jobs add their own on top."""
    parser = argparse.ArgumentParser(
        prog=job_name,
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "Config file (YAML or JSON). Repeatable; later files overlay earlier "
            "ones. conf/ml/base.yaml is always applied underneath and does not "
            "need to be passed."
        ),
    )
    parser.add_argument(
        "--set",
        action="append",
        default=None,
        dest="overrides",
        metavar="KEY=VALUE",
        help="Dotted-key override applied after the files, e.g. --set run.mode=pilot",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Override run.run_id. A fanned-out sweep must instead leave the key "
            "unset and give every submission the same TS05_RUN_ID, so that one "
            "sweep's arms land under one prefix; see conf/ml/sandbox.yaml."
        ),
    )
    parser.add_argument(
        "--keep-spark",
        action="store_true",
        help=(
            "Do not stop the SparkSession on the way out. For a notebook or a "
            "shell that calls two jobs in sequence and owns the session."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Python logging level for the driver.",
    )
    return parser


def parse_args(
    parser: argparse.ArgumentParser,
    argv: Optional[Sequence[str]],
) -> argparse.Namespace:
    return parser.parse_args(list(argv) if argv is not None else None)


def load(args: argparse.Namespace) -> MLConfig:
    """Build the run's :class:`MLConfig` from the parsed arguments.

    ``--run-id`` is applied as an override on the config *tree* rather than onto
    the resulting object, so that it takes precedence over ``TS05_RUN_ID`` the
    same way a config file's ``run.run_id`` does. Setting it afterwards would
    make the flag win over the environment on some paths and lose on others,
    depending on whether the config happened to name a run id.
    """
    overrides = list(args.overrides or [])
    if args.run_id:
        overrides.append(f"run.run_id={args.run_id}")
    return load_ml_config(config_paths=args.config, overrides=overrides)


# --------------------------------------------------------------------------
# Spark
# --------------------------------------------------------------------------

def build_spark_session(cfg: MLConfig, job_name: str) -> Any:
    """A session for reading and writing parquet by path. No catalog.

    ``spark.sql.session.timeZone`` is pinned from config. Nothing else is set:
    an ML job resolves no table names, so the catalog configuration the lineage
    jobs need has no counterpart here, and inventing one would only create a
    setting that could be wrong without being used.

    ``getOrCreate`` rather than a fresh session, so that a notebook that already
    has one — the exploration track under ``notebooks/`` — can call a job's
    ``run`` directly and have it attach.
    """
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName(f"{APP_NAME_PREFIX}.{job_name}").getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", str(cfg.spark_session_timezone))
    LOGGER.info(
        "SparkSession ready | app=%s | version=%s | session tz=%s",
        spark.sparkContext.appName,
        spark.version,
        cfg.spark_session_timezone,
    )
    return spark


# --------------------------------------------------------------------------
# the feature-selection artifact
# --------------------------------------------------------------------------

def write_feature_selection(
    cfg: MLConfig,
    model: str,
    method: str,
    report: pd.DataFrame,
    ranked: pd.DataFrame,
    selections: Mapping[int, Sequence[str]],
    profile: pd.DataFrame,
    profiled_rows: Optional[int] = None,
) -> Dict[str, str]:
    """Write one model's selection under its prefix. Returns the URIs written.

    Four objects: the ranked table with its keep flag and drop reason, the
    ``{count: [names]}`` mapping the sweep reads, the profile the ranking was
    computed from, and the manifest of counts. The profile is written as well as
    the ranking because they answer different questions — the ranking says which
    features were chosen, the profile says what they looked like — and a later
    argument about a choice cannot be settled from the ranking alone.

    The mapping is keyed by count and written as one object rather than as one
    object per count. A sweep arm reads it once per ``(model, method)`` group and
    then serves every feature count and every preprocessing variant in that group
    from memory, which is nine to sixty-three arms per read.
    """
    prefix = feature_selection_prefix(cfg, model, method)
    written: Dict[str, str] = {}

    written["ranked_features"] = join_uri(prefix, RANKED_FEATURES_FILE)
    write_pandas(ranked, written["ranked_features"])

    written["feature_profile"] = join_uri(prefix, "feature_profile.csv")
    write_pandas(profile, written["feature_profile"])

    written["column_classification"] = join_uri(prefix, "column_classification.csv")
    write_pandas(report, written["column_classification"])

    written["selected_features"] = join_uri(prefix, SELECTED_FEATURES_FILE)
    write_json(
        written["selected_features"],
        {
            "model": str(model),
            "method": str(method),
            "run_id": cfg.run_id,
            "models_version": cfg.models_version,
            # Keys are strings because they go through JSON, which has no integer
            # keys. `read_feature_selection` casts them back rather than leaving
            # a caller to discover that `selections[25]` misses and
            # `selections["25"]` hits.
            "selections": {str(count): list(names) for count, names in selections.items()},
        },
    )

    written["selection_manifest"] = join_uri(prefix, SELECTION_MANIFEST_FILE)
    write_json(
        written["selection_manifest"],
        selection_manifest(
            model=model,
            method=method,
            report=report,
            ranked=ranked,
            selections=selections,
            profiled_rows=profiled_rows,
        ),
    )
    return written


def read_feature_selection(
    cfg: MLConfig,
    model: str,
    method: str,
    run_id: Optional[str] = None,
) -> Dict[int, List[str]]:
    """The ``{count: [feature names]}`` mapping one selection run produced.

    Raises when the object is absent. That is not a case to work around by
    falling back to "all numeric columns", which is what the notebook's training
    stage did when it could not find a selection: every arm then trained on the
    same unranked column list, ``top_n_features`` became decoration, and the
    sweep's most expensive dimension measured nothing.
    """
    prefix = feature_selection_prefix(cfg, model, method, run_id=run_id)
    uri = join_uri(prefix, SELECTED_FEATURES_FILE)
    if not s3_exists(uri):
        raise ConfigError(
            f"no feature selection for model {model!r} at {uri}. Run "
            "trust_score_05.ml.jobs.selection_job first, or point "
            "--selection-run-id at a selection that exists."
        )
    payload = read_json(uri)
    if not isinstance(payload, dict) or not isinstance(payload.get("selections"), dict):
        raise ConfigError(f"{uri} is not a selection document")
    out: Dict[int, List[str]] = {}
    for count, names in payload["selections"].items():
        try:
            key = int(count)
        except (TypeError, ValueError):
            LOGGER.warning("ignoring non-integer feature count %r in %s", count, uri)
            continue
        out[key] = [str(name) for name in names]
    if not out:
        raise ConfigError(f"{uri} holds no feature lists")
    return out


def resolve_selection_run_id(cfg: MLConfig, model: str, method: str) -> str:
    """The most recent selection run id for one ``(model, method)``.

    Run ids default to a UTC timestamp (see
    :class:`~trust_score_05.ml.config.MLConfig`), so the greatest one
    lexicographically is the latest one chronologically. That holds for the
    generated ids and not for an operator-supplied name, which is exactly why
    the resolved value is logged by the caller and why ``--selection-run-id``
    exists: a sweep whose selection matters should name it rather than inherit
    whatever sorted last.
    """
    if not cfg.feature_selection_root:
        raise ConfigError("feature_selection_root is not configured")
    parent = join_uri(cfg.feature_selection_root, f"model={model}", f"method={method}")
    candidates = [
        prefix.rstrip("/").rsplit("run_id=", 1)[-1]
        for prefix in list_common_prefixes(parent + "/")
        if "run_id=" in prefix
    ]
    if not candidates:
        raise ConfigError(
            f"no feature selection has ever been written under {parent}/ for model "
            f"{model!r}; run trust_score_05.ml.jobs.selection_job first"
        )
    return sorted(candidates)[-1]


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run_job(
    job_name: str,
    parser: argparse.ArgumentParser,
    run_fn: RunFn,
    argv: Optional[Sequence[str]] = None,
) -> JobResult:
    """Parse, configure, build a session, run, and log the summary.

    Config before Spark, so a bad config fails in seconds rather than after a
    cluster-sized session has been built. The summary is logged from a ``finally``
    so that a run which raises still leaves its elapsed time and whatever the job
    had recorded before the failure — which for a sweep is the count of arms that
    did finish, and is the difference between resuming and restarting.
    """
    args = parse_args(parser, argv)
    configure_logging(args.log_level)
    started = time.time()

    cfg = load(args)
    log_banner(LOGGER, f"{job_name} | {cfg.models_version} | run_id={cfg.run_id}")
    log_mapping(
        LOGGER,
        "run parameters",
        {
            "run_id": cfg.run_id,
            "models_version": cfg.models_version,
            "mode": cfg.mode,
            "config sources": ", ".join(cfg.sources) or "(defaults)",
            "training months": len(cfg.training_months),
            "scenarios": ", ".join(cfg.scenario_names()) or "(none)",
            "models_root": cfg.models_root or "(unset)",
        },
    )

    summary = JobResult(job=job_name, run_id=cfg.run_id, models_version=cfg.models_version)
    spark = build_spark_session(cfg, job_name)
    try:
        summary.update(run_fn(MLRunContext(spark=spark, cfg=cfg, args=args, job_name=job_name)))
        summary.setdefault("status", "SUCCESS")
        return summary
    finally:
        summary["duration_seconds"] = round(time.time() - started, 2)
        summary.setdefault("status", "FAILED")
        log_mapping(LOGGER, f"{job_name} run summary", summary)
        # Left running deliberately when a notebook owns the session. A job that
        # stopped a session it merely attached to would break the cell after it.
        if not bool(getattr(args, "keep_spark", False)):
            spark.stop()


def main(
    job_name: str,
    parser: argparse.ArgumentParser,
    run_fn: RunFn,
    argv: Optional[Sequence[str]] = None,
) -> int:
    """Entrypoint wrapper: turn exceptions into an exit code."""
    try:
        run_job(job_name, parser, run_fn, argv)
    except ConfigError as exc:
        LOGGER.error("configuration error: %s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001 - top-level driver boundary
        LOGGER.error("%s failed: %s", job_name, exc, exc_info=True)
        return 1
    return 0
