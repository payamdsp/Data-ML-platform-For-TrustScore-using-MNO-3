"""ML stage 2: profile the candidate features and rank them per model.

    python -m trust_score_05.ml.jobs.selection_job \
      --config conf/ml/base.yaml --config conf/ml/local.yaml

One profiling pass, then one ranking per model. The pass is shared because
profiling is the expensive half and it does not depend on the model: null rate,
variance and quantile spread are properties of the training population. Only
:func:`~trust_score_05.ml.selection.family_boost` differs by model, and it is a
multiplier applied to an already-computed score. The notebook re-profiled per
model, which is ten passes over the training union to compute the same three
numbers ten times.

Three things this job does that the notebook's selection cell did not.

It **profiles the union of the training months**, not the most recent one. A
feature that exists only in the latest month is null across most of what the
model will be fitted on, and that null rate is the honest description of it.

It **restricts candidates to columns the fraud population also has**. A feature
present in training and absent from the fraud extract is all-null on the
positive side of every scored population, so a model can only ever learn that
"this column exists" separates the classes — and since the union's projection
fills it rather than failing, nothing downstream says so. Discovery reports the
condition; this job acts on it, and records the count it dropped.

It **writes what it decided**, in four objects per model, so that a later
argument about why a feature was or was not used is settled by reading rather
than by re-running. See
:func:`trust_score_05.ml.jobs.base.write_feature_selection`.

The output prefix carries the *selection's* run id, not the sweep's — a selection
is reused across many training runs (see
:func:`trust_score_05.ml.paths.feature_selection_prefix`), which is why
``sweep_job`` takes ``--selection-run-id`` rather than assuming its own.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional, Sequence

from trust_score_05.common.io.s3 import write_marker
from trust_score_05.lineage.config import ConfigError
from trust_score_05.lineage.logging_utils import get_logger, log_banner
from trust_score_05.ml.datasets import (
    CUSTOMER_KEY_COLUMN,
    SOURCE_PATH_COLUMN,
    discover_fraud_paths,
    discover_training_nonfraud_paths,
    read_parquet_path,
    read_parquet_paths,
)
from trust_score_05.ml.jobs.base import (
    JobResult,
    MLRunContext,
    build_arg_parser,
    main,
    run_job,
    write_feature_selection,
)
from trust_score_05.ml.models import MODEL_NAMES
from trust_score_05.ml.paths import feature_selection_prefix
from trust_score_05.ml.selection import (
    METHOD_NAME,
    SelectionError,
    candidate_features,
    is_numeric_dtype,
    profile_features,
    rank_features,
    select_top_n,
)

LOGGER = get_logger(__name__)

JOB_NAME = "ml_feature_selection"

__all__ = ["JOB_NAME", "build_parser", "main_cli", "run", "run_stage"]

#: Columns added by the readers rather than by the feature extract. Passed to
#: the classifier as provenance so they are excluded with a reason rather than
#: silently surviving as high-cardinality strings.
PROVENANCE_COLUMNS = (SOURCE_PATH_COLUMN, "time_window", "scenario", "label")


def build_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser(
        JOB_NAME,
        "ML stage 2: profile the training features and write each model's ranked list.",
    )
    parser.add_argument(
        "--models",
        default=",".join(MODEL_NAMES),
        metavar="NAME[,NAME...]",
        help="Which models to write selections for. Default: all of them.",
    )
    parser.add_argument(
        "--method",
        default=METHOD_NAME,
        help=(
            "The method label written into the output path. Change it only to "
            "keep two selections apart; the ranking itself is this module's."
        ),
    )
    parser.add_argument(
        "--skip-fraud-intersection",
        action="store_true",
        help=(
            "Do not restrict candidates to columns the fraud population also "
            "carries. For selecting against a training extract before the "
            "matching fraud extract exists; the resulting lists must not be used "
            "for a sweep whose metrics anybody will read."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-select even when this run id's prefix is already _READY.",
    )
    return parser


def _training_union(ctx: MLRunContext):
    """Every configured training month as one frame, for profiling."""
    discovered = discover_training_nonfraud_paths(ctx.cfg).require("training non-fraud")
    if discovered.missing:
        LOGGER.warning(
            "%d configured training month(s) have no data prefix and are not "
            "profiled: %s",
            len(discovered.missing),
            list(discovered.missing),
        )
    return read_parquet_paths(ctx.spark, discovered.paths())


def _fraud_numeric_columns(ctx: MLRunContext) -> Optional[List[str]]:
    """Numeric column names the fraud population carries, or None if unreadable.

    Schema only — one ``dtypes`` per partition, no rows read. ``None`` rather
    than an empty list when nothing could be read, because the two mean opposite
    things: an empty list would intersect every candidate away, and returning it
    would turn "the fraud extract is unreachable" into "no feature is usable".
    """
    paths = [p for prefixes in discover_fraud_paths(ctx.cfg).values() for p in prefixes]
    if not paths:
        LOGGER.warning("no fraud prefixes were discovered; skipping the intersection")
        return None
    names: set = set()
    read = 0
    for path in paths:
        try:
            dtypes = read_parquet_path(ctx.spark, path).dtypes
        except Exception as exc:  # noqa: BLE001 - one partition, not the job
            LOGGER.warning("cannot read the schema of fraud prefix %s: %s", path, exc)
            continue
        read += 1
        names |= {
            name for name, dtype in dtypes if is_numeric_dtype(dtype)
        }
    if not read:
        LOGGER.warning(
            "none of the %d fraud prefixes could be read; skipping the intersection",
            len(paths),
        )
        return None
    return sorted(names)


def run_stage(ctx: MLRunContext) -> JobResult:
    """Profile once, then rank and write once per model."""
    cfg = ctx.cfg
    models = [name.strip() for name in str(ctx.args.models).split(",") if name.strip()]
    unknown = [name for name in models if name not in MODEL_NAMES]
    if unknown:
        raise ConfigError(f"unknown model names {unknown}; expected a subset of {MODEL_NAMES}")
    if not models:
        raise ConfigError("--models selected nothing")
    method = str(ctx.args.method)

    frame = _training_union(ctx)
    candidates, report = candidate_features(
        frame.dtypes,
        join_key=CUSTOMER_KEY_COLUMN,
        provenance_columns=PROVENANCE_COLUMNS,
    )
    log_banner(
        LOGGER,
        f"{len(candidates)} eligible of {len(report)} columns | method={method}",
    )

    dropped_not_in_fraud: List[str] = []
    if not ctx.args.skip_fraud_intersection:
        fraud_numeric = _fraud_numeric_columns(ctx)
        if fraud_numeric is not None:
            shared = set(fraud_numeric)
            dropped_not_in_fraud = [name for name in candidates if name not in shared]
            candidates = [name for name in candidates if name in shared]
            if dropped_not_in_fraud:
                LOGGER.warning(
                    "%d candidate features are absent from the fraud population and "
                    "were dropped; they would be all-null on the positive side of "
                    "every scored population. First 20: %s",
                    len(dropped_not_in_fraud),
                    dropped_not_in_fraud[:20],
                )
            if not candidates:
                raise SelectionError(
                    "no candidate feature is present in both the training and the "
                    "fraud populations; every model would be fitted on one set of "
                    "columns and scored on a disjoint one"
                )
        # `report` keeps its own verdict and gains this one as a separate column,
        # rather than having `eligible` overwritten. Two different reasons a
        # column is unusable — its name, and its absence from the other
        # population — are worth telling apart in the report an operator reads.
        report["in_fraud_population"] = ~report["column"].isin(dropped_not_in_fraud)

    profile = profile_features(
        frame,
        candidates,
        sample_rows=cfg.profile_rows,
        seed=cfg.seed,
    )
    profiled_rows = int(profile["n_rows"].max()) if not profile.empty else 0
    LOGGER.info("profiled %d features over %s rows", len(profile), f"{profiled_rows:,}")

    per_model: Dict[str, Dict[str, object]] = {}
    for model in models:
        prefix = feature_selection_prefix(cfg, model, method)
        write_marker(prefix, "STARTED", {"model": model, "method": method, "run_id": cfg.run_id})
        try:
            ranked = rank_features(
                profile,
                model,
                max_null_rate=cfg.max_null_rate,
                min_variance=cfg.min_variance,
            )
            selections = select_top_n(ranked, cfg.top_n_feature_counts)
        except Exception as exc:  # noqa: BLE001 - one model, not the job
            LOGGER.exception("selection failed for model %s", model)
            write_marker(prefix, "FAILED", {"model": model, "error": repr(exc)})
            per_model[model] = {"status": "FAILED", "error": repr(exc)}
            continue

        written = write_feature_selection(
            cfg=cfg,
            model=model,
            method=method,
            report=report,
            ranked=ranked,
            selections=selections,
            profile=profile,
            profiled_rows=profiled_rows,
        )
        kept = int(ranked["keep"].sum())
        write_marker(
            prefix,
            "READY",
            {
                "model": model,
                "method": method,
                "run_id": cfg.run_id,
                "features_profiled": int(len(ranked)),
                "features_kept": kept,
                # The counts the sweep will actually see. Requested counts above
                # the number of kept features are dropped rather than clamped —
                # see `select_top_n` — so this list can be shorter than
                # `sweep.top_n_feature_counts`, and an arm for a missing count
                # would otherwise fail with no explanation on this side.
                "selected_counts": sorted(selections),
                "reports": written,
            },
        )
        LOGGER.info(
            "model %-18s | %3d kept of %3d profiled | counts %s",
            model,
            kept,
            len(ranked),
            sorted(selections),
        )
        per_model[model] = {
            "status": "READY",
            "features_kept": kept,
            "selected_counts": sorted(selections),
        }

    failed = sorted(name for name, info in per_model.items() if info["status"] == "FAILED")
    return JobResult(
        status="SUCCESS" if not failed else "PARTIAL",
        method=method,
        columns_seen=int(len(report)),
        candidates=len(candidates),
        dropped_not_in_fraud=len(dropped_not_in_fraud),
        features_profiled=int(len(profile)),
        profiled_rows=profiled_rows,
        models_ready=sorted(set(per_model) - set(failed)),
        models_failed=failed,
        prefix=feature_selection_prefix(cfg, models[0], method).rsplit("/model=", 1)[0]
        if cfg.feature_selection_root
        else "",
    )


def run(argv: Optional[Sequence[str]] = None) -> JobResult:
    return run_job(JOB_NAME, build_parser(), run_stage, argv)


def main_cli(argv: Optional[Sequence[str]] = None) -> int:
    return main(JOB_NAME, build_parser(), run_stage, argv)


if __name__ == "__main__":  # pragma: no cover - entrypoint
    sys.exit(main_cli())
