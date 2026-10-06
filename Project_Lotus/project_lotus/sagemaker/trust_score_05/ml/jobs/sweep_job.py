"""ML stage 3: walk the arms, fit every configuration, write the leaderboards.

    python -m trust_score_05.ml.jobs.sweep_job \
      --config conf/ml/base.yaml --config conf/ml/local.yaml \
      --models ecod --method variance

The expensive stage. :func:`trust_score_05.ml.sweep.run_sweep` holds the loop and
the resumption logic and takes no Spark import; this module is the half that
does: it resolves the feature selection, turns the three data dependencies into
callables over :mod:`trust_score_05.ml.datasets`, and writes the one artifact
that belongs to no single arm.

Four things happen here that did not happen in the notebook's training cells.

**The selection is named, not inferred.** A selection is versioned separately
from the models (see
:func:`trust_score_05.ml.paths.feature_selection_prefix`), so this job cannot
assume its own run id points at one. ``--selection-run-id`` names it; without the
flag :func:`~trust_score_05.ml.jobs.base.resolve_selection_run_id` takes the
latest and *logs which*, because a sweep whose feature lists came from a
selection nobody recorded is a sweep whose results cannot be reproduced.

**Arms are sliced before they are filtered.** ``--arm-start``/``--arm-count``
slice :func:`~trust_score_05.ml.sweep.sweep_arms`' unfiltered order, and only
then are arms dropped whose feature count the selection does not contain. Doing
it the other way — filtering first — makes the slice boundaries depend on which
counts each model's selection happens to hold, so two submissions given
``0,100`` and ``100,100`` would overlap on some models and skip arms on others.
The notebook sliced the *hyperparameter* grid, after an auto-selection step that
could reorder it, and had neither property.

**The training sample and the fraud population are loaded once per feature
list.** ``sweep_arms`` orders preprocessing innermost precisely so that
consecutive arms share a feature list; the loaders here memoise on it, so nine
preprocessing variants share one Spark read instead of nine. The fraud frame is
collected to pandas at the same time and reused across every scenario of the
group, which is what :func:`~trust_score_05.ml.datasets.load_scenario_frame`'s
pandas branch is for.

**The cross-arm ranking is written where it belongs.** Under
:func:`trust_score_05.ml.paths.sweep_summary_prefix`, a sibling of the ``model=``
trees, grouped by the run-prefix segments the metric rows carry. The notebook
wrote its cross-model comparison under whichever model ran last.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from trust_score_05.common.io.s3 import join_uri, write_json, write_marker, write_pandas
from trust_score_05.lineage.config import ConfigError
from trust_score_05.lineage.logging_utils import get_logger, log_banner
from trust_score_05.ml.datasets import (
    bounded_to_pandas,
    discover_testing_nonfraud_paths,
    load_fraud_frame,
    load_scenario_frame,
    load_training_matrix,
)
from trust_score_05.ml.evaluation import leaderboard
from trust_score_05.ml.jobs.base import (
    JobResult,
    MLRunContext,
    build_arg_parser,
    main,
    read_feature_selection,
    resolve_selection_run_id,
    run_job,
)
from trust_score_05.ml.models import MODEL_NAMES
from trust_score_05.ml.paths import RUN_PREFIX_SEGMENTS, sweep_summary_prefix
from trust_score_05.ml.selection import METHOD_NAME
from trust_score_05.ml.sweep import RunOutcome, SweepArm, run_sweep, sweep_arms

LOGGER = get_logger(__name__)

JOB_NAME = "ml_sweep"

__all__ = ["JOB_NAME", "build_parser", "main_cli", "run", "run_stage"]

#: The metric-row columns that identify an arm. They are the run-prefix segments
#: minus the two that are constant across a sweep — ``version_id`` and ``run_id``
#: — and they are the columns
#: :func:`trust_score_05.ml.sweep._evaluate_scenarios` stamps onto every row from
#: :meth:`~trust_score_05.ml.sweep.SweepArm.as_dict`. Grouping the cross-arm
#: leaderboard by these needs no path parsing, because the identity is already in
#: the frame.
ARM_GROUP_COLUMNS = tuple(
    name for name in RUN_PREFIX_SEGMENTS if name not in ("version_id", "run_id")
)


def build_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser(
        JOB_NAME,
        "ML stage 3: fit and evaluate every arm of the sweep, and rank them.",
    )
    parser.add_argument(
        "--models",
        default=",".join(MODEL_NAMES),
        metavar="NAME[,NAME...]",
        help="Which models to sweep. Default: all of them.",
    )
    parser.add_argument(
        "--method",
        default=METHOD_NAME,
        help="The feature-selection method whose lists to read.",
    )
    parser.add_argument(
        "--selection-run-id",
        default=None,
        metavar="RUN_ID",
        help=(
            "Which selection run to take feature lists from. Defaults to the "
            "latest one written for each model, which is logged. Name it for any "
            "run whose results will be compared with another's."
        ),
    )
    parser.add_argument(
        "--arm-start",
        type=int,
        default=0,
        metavar="N",
        help="Index of the first arm to run, in sweep_arms order. For fanning out.",
    )
    parser.add_argument(
        "--arm-count",
        type=int,
        default=0,
        metavar="N",
        help="How many arms to run from --arm-start. 0 means all of them.",
    )
    parser.add_argument(
        "--max-minutes-per-arm",
        type=float,
        default=0.0,
        metavar="MINUTES",
        help=(
            "Per-arm wall-clock budget; an arm stopped by it is marked INCOMPLETE "
            "with its unattempted count and can be resumed. 0, the default, "
            "disables it. A flag rather than a config key because it bounds the "
            "*submission* — the session it runs in, not the experiment — and two "
            "submissions of one sweep with different budgets still sweep the same "
            "grid, which is not true of `sweep.max_hyperparam_configs`."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refit configurations and arms that are already _READY.",
    )
    parser.add_argument(
        "--no-skip-ready-arms",
        action="store_true",
        help=(
            "Enter every arm even when its run prefix is already _READY, so that "
            "its per-configuration markers are re-checked. Slower by one training "
            "read per finished arm; use it when an arm's marker is suspect."
        ),
    )
    return parser


class _SparkLoaders:
    """The three data dependencies of a sweep, memoised per feature list.

    Stateful on purpose. :func:`~trust_score_05.ml.sweep.run_sweep` hands the
    arm to ``feature_loader`` and then calls ``training_loader`` with only the
    feature list, but both Spark readers need a prefix to write their manifests
    to — ``load_training_matrix`` writes the per-month sample manifest and
    ``load_fraud_frame`` writes the quarantine report — and that prefix is the
    arm's. So the arm is recorded when the features are handed out and read back
    on the next call, which is safe because ``run_sweep`` is a single-threaded
    loop over arms and asks for features immediately before loading.

    The memoisation is keyed on the feature tuple rather than on the arm, since
    that is the thing the load actually depends on: the nine preprocessing
    variants of one ``(model, top_n)`` group differ in how the matrix is built,
    not in which rows and columns are read. Only the most recent group is kept —
    ``sweep_arms`` walks the groups contiguously, so a cache of one has the same
    hit rate as a cache of all of them and does not hold every month's sample
    for the length of the sweep.
    """

    def __init__(self, ctx: MLRunContext, selections: Dict[str, Dict[int, List[str]]]) -> None:
        self.ctx = ctx
        self.selections = selections
        self.scenario_paths = self._scenario_paths()
        self._arm: Optional[SweepArm] = None
        self._key: Optional[Tuple[str, ...]] = None
        self._training: Optional[pd.DataFrame] = None
        self._fraud: Optional[pd.DataFrame] = None

    # -- discovery --------------------------------------------------------

    def _scenario_paths(self) -> Dict[str, str]:
        """``{scenario: non-fraud prefix}`` for every window that resolved.

        Keyed by scenario name rather than by time window because that is what
        ``run_sweep`` passes back, but built from the window-keyed discovery so
        the mapping is explicit. A window that resolved to no prefix is dropped
        here and its scenario is absent from ``scenarios`` below, rather than
        failing inside the first arm that reaches it.
        """
        discovered = discover_testing_nonfraud_paths(self.ctx.cfg).require("testing non-fraud")
        if discovered.missing:
            LOGGER.warning(
                "%d configured evaluation window(s) have no data prefix and are not "
                "scored: %s",
                len(discovered.missing),
                list(discovered.missing),
            )
        return {window[:7]: path for window, path in discovered.found.items()}

    @property
    def scenarios(self) -> List[str]:
        return list(self.scenario_paths)

    # -- the three callables ----------------------------------------------

    def features_for(self, arm: SweepArm) -> Sequence[str]:
        """The arm's selected feature list, and the note of which arm asked."""
        self._arm = arm
        by_count = self.selections[arm.model]
        return by_count[arm.top_n_features]

    def training_for(self, features: Sequence[str]) -> pd.DataFrame:
        key = tuple(features)
        if self._key == key and self._training is not None:
            return self._training
        self._reset(key)
        frame, present, manifest = load_training_matrix(
            self.ctx.spark, self.ctx.cfg, features, self._prefix()
        )
        LOGGER.info(
            "training sample: %s rows x %d of %d features over %d month(s)",
            f"{len(frame):,}",
            len(present),
            len(features),
            int(len(manifest)),
        )
        self._training = frame
        return frame

    def scenario_for(self, arm: SweepArm, scenario: str) -> pd.DataFrame:
        """One evaluation population, with the fraud side collected once.

        The fraud frame is collected to pandas the first time a feature list
        needs it and handed to every scenario of that list afterwards. Reading it
        per scenario instead is a full pass over every fraud partition per
        window, for a frame that does not vary by window: the evaluation set is
        the same labelled population scored against each month's negatives.
        """
        path = self.scenario_paths[scenario]
        features = list(self.features_for(arm))
        if self._fraud is None:
            spark_frame, manifest = load_fraud_frame(
                self.ctx.spark, self.ctx.cfg, features, self._prefix()
            )
            self._fraud = bounded_to_pandas(
                spark_frame, max_rows=0, seed=self.ctx.cfg.seed, context="fraud population"
            )
            LOGGER.info(
                "fraud population: %s rows kept of %s (%s null, %s unparseable, %s "
                "before %s)",
                f"{manifest['rows_kept']:,}",
                f"{manifest['rows_in']:,}",
                f"{manifest['rows_null_timestamp']:,}",
                f"{manifest['rows_unparseable_timestamp']:,}",
                f"{manifest['rows_before_min_timestamp']:,}",
                manifest["fraud_min_timestamp"],
            )
        return load_scenario_frame(
            self.ctx.spark,
            self.ctx.cfg,
            scenario=scenario,
            nonfraud_path=path,
            fraud_frame=self._fraud,
            features=features,
        )

    # -- internals --------------------------------------------------------

    def _prefix(self) -> str:
        if self._arm is None:  # pragma: no cover - run_sweep always asks first
            raise ConfigError("a loader was called before any arm requested its features")
        return self._arm.prefix(self.ctx.cfg)

    def _reset(self, key: Tuple[str, ...]) -> None:
        """Drop the previous group's frames before loading the next one's.

        Explicitly, and before the new load rather than after, so that the two
        groups' training samples are never both resident. Each is
        ``rows_per_month`` times the number of months by
        ``top_n_features`` floats, and the largest arms of a full sweep are close
        enough to the driver's memory that holding two costs the job.
        """
        self._key = key
        self._training = None
        self._fraud = None


def _resolve_selections(ctx: MLRunContext, models: Sequence[str], method: str):
    """Read each model's feature lists, and record which selection they came from.

    Returns ``({model: {count: [names]}}, {model: run_id})``. A model whose
    selection is missing fails the job rather than being dropped: the arms of a
    sweep are the cross-product of what was asked for, and quietly sweeping nine
    models when ten were requested produces a leaderboard that looks complete.
    """
    requested = ctx.args.selection_run_id
    selections: Dict[str, Dict[int, List[str]]] = {}
    resolved: Dict[str, str] = {}
    for model in models:
        run_id = requested or resolve_selection_run_id(ctx.cfg, model, method)
        selections[model] = read_feature_selection(ctx.cfg, model, method, run_id=run_id)
        resolved[model] = run_id
        LOGGER.info(
            "model %-18s | selection run_id=%s | counts %s",
            model,
            run_id,
            sorted(selections[model]),
        )
    if not requested and len(set(resolved.values())) > 1:
        # Not an error — the models can legitimately have been selected in
        # separate runs — but it means the arms of this sweep were not built from
        # one selection, so a cross-model comparison of them compares two
        # different feature-ranking populations. Worth saying out loud, since
        # nothing in the leaderboard's own columns would reveal it.
        LOGGER.warning(
            "the models' feature lists come from %d different selection runs: %s. "
            "Pass --selection-run-id to compare models on one selection.",
            len(set(resolved.values())),
            resolved,
        )
    return selections, resolved


def _usable_arms(
    arms: Sequence[SweepArm],
    selections: Dict[str, Dict[int, List[str]]],
) -> Tuple[List[SweepArm], List[SweepArm]]:
    """Split the sliced arms into the ones the selection can supply and the rest.

    An arm asks for ``top_n_features`` features and the selection holds only the
    counts that did not exceed the number of features kept —
    :func:`trust_score_05.ml.selection.select_top_n` drops an oversized request
    rather than clamping it, so that two counts above the ceiling do not become
    the same list under two names. Those arms are skipped with a count rather
    than run: entering one would write a ``_FAILED`` marker whose reason is a
    ``KeyError`` on an integer, and a sweep littered with those is
    indistinguishable from one that is actually broken.
    """
    usable: List[SweepArm] = []
    unsupplied: List[SweepArm] = []
    for arm in arms:
        if arm.top_n_features in selections.get(arm.model, {}):
            usable.append(arm)
        else:
            unsupplied.append(arm)
    return usable, unsupplied


def _write_sweep_summary(
    ctx: MLRunContext,
    outcomes: Sequence[RunOutcome],
    payload: Dict[str, Any],
) -> Dict[str, str]:
    """The cross-arm tables: what every arm did, and how the arms rank.

    Three objects. ``arms.csv`` is one row per arm with its status and counts, so
    "which arms of this sweep finished" is answered by reading one file instead
    of listing a tree of markers. ``cross_arm_leaderboard.csv`` ranks the arms on
    the configured best-config metric, grouped by
    :data:`ARM_GROUP_COLUMNS` — the comparison across models, feature counts and
    preprocessing variants that no arm's own reports can make.
    ``sweep_manifest.json`` records the slice, the selection run ids and the
    tallies.

    The leaderboard is best-effort and its absence is recorded in the manifest
    rather than raised, because the arms' own outputs are already written by the
    time this runs: failing here would leave a sweep whose results exist and
    whose marker says it failed.
    """
    prefix = sweep_summary_prefix(ctx.cfg)
    written: Dict[str, str] = {}

    rows = []
    for outcome in outcomes:
        counts = outcome.counts()
        rows.append({
            **outcome.arm.as_dict(),
            "status": outcome.status,
            "prefix": outcome.prefix,
            "matrix_columns": outcome.n_features,
            "best_hyperparam_id": (outcome.best or {}).get("hyperparam_id", ""),
            **{f"configs_{name}": value for name, value in counts.items()},
            "error": outcome.error,
        })
    if rows:
        uri = join_uri(prefix, "arms.csv")
        write_pandas(pd.DataFrame(rows), uri)
        written["arms"] = uri

    # Only the k-metric frames, because the configured criterion is a top-k one
    # (`best_config_k_percent`) and the plain metric table has no `k_percent`
    # column to filter on. An arm that produced no top-k rows contributes
    # nothing, which `_concat`-style dropping of empties handles.
    frames = [o.k_metrics for o in outcomes if o.k_metrics is not None and not o.k_metrics.empty]
    if frames:
        k_metrics = pd.concat(frames, ignore_index=True)
        uri = join_uri(prefix, "all_arms_k_metrics.csv")
        write_pandas(k_metrics, uri)
        written["k_metrics"] = uri
        try:
            table, summary = leaderboard(
                k_metrics,
                metric=ctx.cfg.best_config_metric,
                level=ctx.cfg.best_config_level,
                aggregate=ctx.cfg.best_config_aggregate,
                k_percent=ctx.cfg.best_config_k_percent,
                group_columns=ARM_GROUP_COLUMNS,
            )
        except Exception as exc:  # noqa: BLE001 - the arms' own outputs are written
            LOGGER.warning("the cross-arm leaderboard could not be built: %s", exc)
            payload["leaderboard_error"] = repr(exc)
        else:
            uri = join_uri(prefix, "cross_arm_leaderboard.csv")
            write_pandas(table, uri)
            written["leaderboard"] = uri
            payload["leaderboard"] = summary
            if not table.empty:
                payload["best_arm"] = {
                    key: value
                    for key, value in table.iloc[0].to_dict().items()
                    if key in set(ARM_GROUP_COLUMNS) | {"value", "n_scenarios", "value_spread"}
                }
    else:
        payload["leaderboard_error"] = "no arm produced top-k metrics"

    uri = join_uri(prefix, "sweep_manifest.json")
    write_json(uri, payload)
    written["manifest"] = uri
    return written


def run_stage(ctx: MLRunContext) -> JobResult:
    """Resolve the selection, build the arms, run them, then rank them."""
    cfg = ctx.cfg
    models = [name.strip() for name in str(ctx.args.models).split(",") if name.strip()]
    unknown = [name for name in models if name not in MODEL_NAMES]
    if unknown:
        raise ConfigError(f"unknown model names {unknown}; expected a subset of {MODEL_NAMES}")
    if not models:
        raise ConfigError("--models selected nothing")
    method = str(ctx.args.method)
    max_minutes = float(ctx.args.max_minutes_per_arm or 0.0)

    selections, selection_run_ids = _resolve_selections(ctx, models, method)

    # Sliced on the unfiltered order, then filtered. See the module docstring:
    # filtering first would make the slice boundaries depend on each model's
    # selection, and two fanned-out submissions would no longer partition the
    # arms.
    sliced = sweep_arms(cfg, models, method, start=ctx.args.arm_start, count=ctx.args.arm_count)
    arms, unsupplied = _usable_arms(sliced, selections)
    if unsupplied:
        LOGGER.warning(
            "%d of %d sliced arms ask for a feature count their model's selection "
            "does not hold and are skipped: %s",
            len(unsupplied),
            len(sliced),
            sorted({(arm.model, arm.top_n_features) for arm in unsupplied}),
        )
    if not arms:
        raise ConfigError(
            f"none of the {len(sliced)} arms in this slice has a feature list; the "
            f"selection for {method!r} holds "
            f"{ {model: sorted(counts) for model, counts in selections.items()} } "
            f"but the sweep asks for {list(cfg.top_n_feature_counts)}"
        )

    loaders = _SparkLoaders(ctx, selections)
    scenarios = loaders.scenarios
    if not scenarios:
        raise ConfigError("no evaluation window resolved to a prefix; there is nothing to score")

    log_banner(
        LOGGER,
        f"{len(arms)} arms | {len(models)} models | method={method} | "
        f"{len(scenarios)} scenarios | run_id={cfg.run_id}",
    )

    outcomes = run_sweep(
        cfg=cfg,
        arms=arms,
        feature_loader=loaders.features_for,
        training_loader=loaders.training_for,
        scenarios=scenarios,
        scenario_loader=loaders.scenario_for,
        force=bool(ctx.args.force),
        max_minutes_per_arm=max_minutes,
        skip_ready_arms=not bool(ctx.args.no_skip_ready_arms),
    )

    by_status: Dict[str, int] = {}
    for outcome in outcomes:
        by_status[outcome.status] = by_status.get(outcome.status, 0) + 1
        LOGGER.info(
            "arm %-52s | %-10s | %s",
            outcome.arm,
            outcome.status,
            outcome.counts(),
        )

    payload: Dict[str, Any] = {
        "run_id": cfg.run_id,
        "models_version": cfg.models_version,
        "mode": cfg.mode,
        "models": models,
        "method": method,
        "selection_run_ids": selection_run_ids,
        "scenarios": scenarios,
        "arm_start": int(ctx.args.arm_start),
        "arm_count": int(ctx.args.arm_count),
        "arms_in_slice": len(sliced),
        "arms_run": len(arms),
        "arms_unsupplied": len(unsupplied),
        "arms_by_status": by_status,
        "max_minutes_per_arm": max_minutes,
        "config": cfg.as_dict(),
    }
    written = _write_sweep_summary(ctx, outcomes, payload)

    # The marker goes down last, and it is READY only when no arm failed and none
    # was truncated. A sweep of which two arms are INCOMPLETE is not a sweep whose
    # leaderboard should be read as final, and the notebook's unconditional
    # `_READY.json` at the end of its training loop is what made that
    # indistinguishable.
    failed = int(by_status.get("FAILED", 0))
    incomplete = int(by_status.get("INCOMPLETE", 0))
    marker = "READY" if not failed and not incomplete and not unsupplied else "FAILED"
    write_marker(sweep_summary_prefix(cfg), marker, {**payload, "reports": written})

    return JobResult(
        status="SUCCESS" if marker == "READY" else "PARTIAL",
        method=method,
        arms_in_slice=len(sliced),
        arms_run=len(arms),
        arms_unsupplied=len(unsupplied),
        arms_ready=int(by_status.get("READY", 0)),
        arms_failed=failed,
        arms_incomplete=incomplete,
        scenarios=len(scenarios),
        best_arm=payload.get("best_arm"),
        prefix=sweep_summary_prefix(cfg),
    )


def run(argv: Optional[Sequence[str]] = None) -> JobResult:
    return run_job(JOB_NAME, build_parser(), run_stage, argv)


def main_cli(argv: Optional[Sequence[str]] = None) -> int:
    return main(JOB_NAME, build_parser(), run_stage, argv)


if __name__ == "__main__":  # pragma: no cover - entrypoint
    sys.exit(main_cli())
