"""The driver: walk the cross-product, and be resumable while doing it.

A sweep is a nested loop over five dimensions — model, feature-selection method,
feature count, preprocessing variant, hyperparameter configuration — and the
first four of those name a *run prefix* while the fifth names a *config prefix*
inside it (see :mod:`trust_score_05.ml.paths`). The unit of work is one config
prefix: it holds one fitted bundle, one metrics table per scenario, and one
marker saying whether it finished. The unit of resumption is the same thing.

Resumability is the whole reason this module is separate from the job
entrypoints. A full sweep is thousands of fits over six-figure matrices, it runs
on a session that will be killed before it finishes, and it therefore has to be
restartable without redoing what it already did. That works only if a
configuration's identity is a function of the configuration. It was not:

* The notebook's ``safe_hyperparam_id(params, config_index)`` began the
  identifier with ``v{config_index:03d}``, so the same parameters got a
  different identifier at a different position in the grid. Restarting a job
  with a different ``CONFIG_SLICE_START``, or after any change that reordered
  the grid, meant every configuration looked new, found no ``_READY.json``, and
  was refitted and rewritten beside its own earlier output under a fresh key.
  :func:`trust_score_05.ml.paths.hyperparam_id` takes no index.
* There were two live definitions of that function with different key lists —
  one skipped ``variant``, the other included it — so identifiers written by one
  cell were not found by the other.

The second reason is that the notebook's run-level outputs were assembled from
whatever the *current process* happened to execute. ``all_metric_frames``
collected a frame per configuration it ran, and configurations skipped because
they were already ready contributed nothing. So the leaderboard and the "best
configuration" of a resumed run were computed over a subset of the grid
determined by how the job had been restarted, and two resumes of the same sweep
could name different winners from identical data. :func:`collect_metrics` reads
the per-configuration tables back for the skipped ones, so the run-level answer
depends on the grid rather than on the restart history.

The third is that its markers did not mean what they said. The per-configuration
loop caught every exception, wrote ``_FAILED.json``, and continued — which is
right — but the run then wrote ``_READY.json`` unconditionally, so a run in
which every configuration failed was indistinguishable from one in which all
succeeded. The runtime guard was worse: it ``break``\\ ed out of the loop and
fell through to the same unconditional ``_READY.json``, so a run truncated after
one of five hundred configurations was marked complete. Here a run is ``READY``
only if at least one configuration succeeded and none were left unattempted;
otherwise it is ``FAILED`` or, when the guard stopped it, ``INCOMPLETE``, and the
marker carries the counts either way.

This module holds no Spark import. The three things that need one — the training
sample, the fraud population and each scenario's non-fraud window — arrive as
callables, so the sweep is exercisable on in-memory frames and the Spark reads
stay in :mod:`trust_score_05.ml.datasets` where the partition-inference and
schema-alignment problems live.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
import traceback
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from trust_score_05.common.io.s3 import (
    join_uri,
    read_json,
    ready_exists,
    s3_exists,
    write_json,
    write_marker,
    write_pandas,
)
from trust_score_05.common.io.serialization import bundle_from_parts, save_bundle
from trust_score_05.ml.config import MLConfig
from trust_score_05.ml.drift import DriftThresholds, drift_report
from trust_score_05.ml.evaluation import (
    ScenarioEvaluation,
    evaluate_population,
    leaderboard,
    score_population,
    select_best_config,
)
from trust_score_05.ml.grids import config_grid_for_model
from trust_score_05.ml.models import MODEL_NAMES, ModelUnavailableError, build_model
from trust_score_05.ml.paths import (
    config_prefix,
    hyperparam_id,
    run_prefix,
    scenario_scores_prefix,
)
from trust_score_05.ml.preprocessing import fit_preprocessor

__all__ = [
    "CONFIG_STATUSES",
    "ConfigOutcome",
    "RunOutcome",
    "SweepArm",
    "SweepError",
    "collect_metrics",
    "run_arm",
    "run_config",
    "run_sweep",
    "sweep_arms",
]

LOGGER = logging.getLogger(__name__)

#: What one configuration can end as. ``skipped`` is a configuration that was
#: already ``_READY`` and was not refitted; ``unavailable`` is one whose model
#: needs an optional package that is not installed. The two are separated from
#: ``failed`` because neither is a defect in the run, and a leaderboard that
#: counts them as failures reads as though the sweep is broken when it is
#: merely resuming or under-provisioned.
CONFIG_STATUSES = ("ready", "skipped", "failed", "unavailable")

#: Written to the run prefix instead of ``READY`` when the wall-clock guard
#: stopped the loop early. A distinct marker name rather than a field inside
#: ``_READY.json`` so that a consumer looking for finished runs — which is what
#: :func:`trust_score_05.common.io.s3.ready_exists` does — does not find this
#: one at all.
_MARKER_INCOMPLETE = "INCOMPLETE"

_CONFIG_METRICS_FILE = "metrics/all_scenario_metrics.csv"
_CONFIG_K_METRICS_FILE = "metrics/all_scenario_k_metrics.csv"


class SweepError(RuntimeError):
    """The sweep could not proceed. Distinct from one configuration failing."""


# --------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SweepArm:
    """One ``(model, method, feature count, preprocessing)`` combination.

    Frozen and hashable so a caller can hold the set of arms it has already
    dispatched. It deliberately does not carry the run id: the run id comes from
    :class:`~trust_score_05.ml.config.MLConfig` and is the same for every arm of
    one sweep, which is what makes the arms' outputs siblings under one prefix
    rather than a scatter of unrelated runs.
    """

    model: str
    method: str
    top_n_features: int
    preprocessing_id: str

    def prefix(self, cfg: MLConfig) -> str:
        return run_prefix(
            cfg, self.model, self.method, self.top_n_features, self.preprocessing_id
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "feature_selection_method": self.method,
            "top_n_features": self.top_n_features,
            "preprocessing": self.preprocessing_id,
        }


def sweep_arms(
    cfg: MLConfig,
    models: Sequence[str],
    method: str,
    start: int = 0,
    count: int = 0,
) -> Tuple[SweepArm, ...]:
    """The arms of the sweep, in a fixed order, optionally sliced.

    The order is model, then feature count, then preprocessing variant — models
    outermost because a model is the coarsest thing an operator wants to run or
    skip on its own, and preprocessing innermost because consecutive variants
    share the same feature list and the same training frame, so a driver that
    walks them in this order reloads the training sample once per feature count
    rather than once per arm.

    ``start`` and ``count`` slice that order, which is how one sweep is fanned
    out across several jobs. The order is a pure function of the config, so two
    jobs given disjoint slices cover disjoint arms and together cover all of
    them — the property the notebook's ``CONFIG_SLICE_START`` was meant to have
    but did not, because it sliced the *hyperparameter* grid after an
    auto-selection step that could reorder it.

    Args:
        cfg: Supplies ``top_n_feature_counts`` and ``preprocessing_ids``.
        models: Model names, validated against :data:`MODEL_NAMES`.
        method: The feature-selection method whose lists to read.
        start: Index of the first arm to return.
        count: How many to return. 0 means all of them from ``start``.
    """
    unknown = [name for name in models if name not in MODEL_NAMES]
    if unknown:
        raise SweepError(f"unknown model names {unknown}; expected a subset of {MODEL_NAMES}")
    if not models:
        raise SweepError("no models given; there would be nothing to sweep")

    arms = [
        SweepArm(model=model, method=str(method), top_n_features=int(top_n), preprocessing_id=prep)
        for model in models
        for top_n in cfg.top_n_feature_counts
        for prep in cfg.preprocessing_ids
    ]
    if start < 0:
        raise SweepError(f"start must be non-negative, got {start}")
    sliced = arms[start:] if count <= 0 else arms[start : start + count]
    return tuple(sliced)


# --------------------------------------------------------------------------
# outcomes
# --------------------------------------------------------------------------

@dataclass
class ConfigOutcome:
    """What one hyperparameter configuration produced.

    ``metrics`` and ``k_metrics`` are empty frames for anything other than
    ``status="ready"``, and a caller must not read a status off their emptiness:
    a configuration whose every scenario evaluated to nothing is ``ready`` with
    empty tables, and a failed one is ``failed`` with the same. That conflation
    is why ``status`` exists as a field.
    """

    hyperparam_id: str
    params: Dict[str, Any]
    prefix: str
    status: str
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    k_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    fit_seconds: float = 0.0
    error: str = ""

    def __post_init__(self) -> None:
        if self.status not in CONFIG_STATUSES:
            raise SweepError(f"status must be one of {CONFIG_STATUSES}, got {self.status!r}")


@dataclass
class RunOutcome:
    """What one arm produced, and enough counts to know whether to trust it.

    ``status`` is the run marker that was written: ``READY``, ``FAILED`` or
    ``INCOMPLETE``. ``n_unattempted`` is non-zero only for ``INCOMPLETE``, and it
    is the number that makes a truncated run recognisable — the notebook wrote a
    ``READY`` marker with a ``configs_attempted`` count and no total, so nothing
    downstream could tell 3 of 500 from 500 of 500.

    ``marker`` is the marker payload for an arm that was skipped because a
    previous process had already finished it. Such an arm has no
    :class:`ConfigOutcome` objects — nothing ran here to produce them — so
    :meth:`counts` reads its tally out of the marker instead. Without that, a
    caller summarising a resumed sweep sees zeros for every arm it did not
    execute, which reads as "nothing was done" for the arms where in fact
    everything was.
    """

    arm: SweepArm
    prefix: str
    status: str
    configs: List[ConfigOutcome] = field(default_factory=list)
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    k_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    best: Optional[Dict[str, Any]] = None
    n_unattempted: int = 0
    n_features: int = 0
    error: str = ""
    marker: Optional[Dict[str, Any]] = None

    def counts(self) -> Dict[str, int]:
        if not self.configs and self.marker:
            tally = {
                status: int(self.marker.get(f"n_configs_{status}", 0) or 0)
                for status in CONFIG_STATUSES
            }
            tally["unattempted"] = int(self.marker.get("n_configs_unattempted", 0) or 0)
            tally["total"] = int(self.marker.get("n_configs_total", 0) or 0)
            return tally
        tally = {status: 0 for status in CONFIG_STATUSES}
        for outcome in self.configs:
            tally[outcome.status] += 1
        tally["unattempted"] = self.n_unattempted
        tally["total"] = len(self.configs) + self.n_unattempted
        return tally


# --------------------------------------------------------------------------
# one configuration
# --------------------------------------------------------------------------

def _impute_rules(preprocessor: Any) -> Dict[str, float]:
    """The learned fill value per input feature, for the bundle.

    Read from the domain imputer when the variant has one and from the sklearn
    imputer's ``statistics_`` otherwise, so that every variant records the values
    it actually filled with. This is the record that makes scoring reproducible:
    the notebook stored the rules for the domain variants and left the dict empty
    for the median and zero ones, then recomputed the fill values at scoring time
    from the rows being scored — which imputed the fraud population from the
    fraud population, making the imputed value itself label-correlated. An empty
    dict here would silently permit that again.
    """
    imputer = getattr(preprocessor, "domain_imputer", None)
    if imputer is not None:
        return {str(k): float(v) for k, v in imputer.fill_values().items()}
    pipeline = getattr(preprocessor, "pipeline", None)
    step = getattr(pipeline, "named_steps", {}).get("imputer") if pipeline else None
    statistics = getattr(step, "statistics_", None)
    features = list(getattr(preprocessor, "features", []))
    if statistics is None:
        return {}
    values = np.asarray(statistics, dtype=float).ravel()
    return {
        name: float(value)
        for name, value in zip(features, values)
        if np.isfinite(value)
    }


def run_config(
    cfg: MLConfig,
    arm: SweepArm,
    params: Dict[str, Any],
    train_matrix: np.ndarray,
    features: Sequence[str],
    preprocessor: Any,
    scenarios: Sequence[str],
    scenario_loader: Callable[[str], pd.DataFrame],
    run: str,
    thresholds: Optional[DriftThresholds] = None,
    force: bool = False,
    write_scores: bool = True,
) -> ConfigOutcome:
    """Fit one configuration, score every scenario, and write its artifacts.

    Returns rather than raises on failure: a configuration that cannot fit is a
    fact about that arm of the grid, not a reason to abandon the other several
    hundred. The exception is recorded in ``_FAILED.json`` with its traceback and
    in the returned :class:`ConfigOutcome`.

    The order of operations matters in two places. The bundle is saved *before*
    scoring, so a configuration that fits expensively and then dies in evaluation
    leaves the fitted model behind to be inspected; the notebook saved it before
    scoring too, and that is one of the things it got right. And drift is
    computed from the same adapter and the same matrices that produced the
    scores, in the same call, rather than by re-reading the scored parquet back
    out of object storage as the notebook's drift step did — which meant a
    configuration whose score write had been skipped produced no drift output and
    no complaint.

    Args:
        cfg: Supplies the seed and the score-writing behaviour.
        arm: Which run prefix this configuration belongs to.
        params: The hyperparameters, exactly as the grid produced them. Not
            normalised, because the identifier is computed from them and must
            match the grid entry the driver iterated over.
        train_matrix: The preprocessed training matrix, fitted once per arm.
        features: The input feature names, in preprocessor order.
        preprocessor: The fitted :class:`~trust_score_05.ml.preprocessing.Preprocessor`.
        scenarios: Evaluation window identifiers.
        scenario_loader: Returns the raw evaluation frame for one scenario.
        run: The arm's run prefix.
        thresholds: Drift thresholds. Defaults to the config's.
        force: Refit even if ``_READY.json`` is already there.
        write_scores: Write the scored records out — parquet always, CSV as well
            when ``cfg.write_scores_csv``. The metrics do not depend on either,
            so a smoke run can turn the whole thing off.
    """
    thresholds = thresholds or DriftThresholds.from_config(cfg)
    hp_id = hyperparam_id(arm.model, params)
    prefix = config_prefix(run, hp_id)

    if ready_exists(prefix) and not force:
        LOGGER.info("skipping ready configuration %s", hp_id)
        return ConfigOutcome(hyperparam_id=hp_id, params=dict(params), prefix=prefix,
                             status="skipped")

    write_marker(prefix, "STARTED", {"hyperparam_id": hp_id, "params": params,
                                     **arm.as_dict()})
    try:
        started = time.time()
        adapter = build_model(arm.model, params, seed=cfg.seed).fit(train_matrix)
        fit_seconds = time.time() - started
    except ModelUnavailableError as exc:
        # A missing optional package is a property of the environment, not of
        # the configuration. Recorded as its own status so a leaderboard missing
        # a model reads as "not installed here" rather than "this model fails".
        LOGGER.warning("configuration %s unavailable: %s", hp_id, exc)
        write_marker(prefix, "FAILED", {"hyperparam_id": hp_id, "params": params,
                                        "reason": "model_unavailable", "error": str(exc)})
        return ConfigOutcome(hyperparam_id=hp_id, params=dict(params), prefix=prefix,
                             status="unavailable", error=str(exc))
    except Exception as exc:  # noqa: BLE001 - one arm of the grid, not the run
        LOGGER.exception("configuration %s failed to fit", hp_id)
        write_marker(prefix, "FAILED", {"hyperparam_id": hp_id, "params": params,
                                        "error": repr(exc),
                                        "traceback": traceback.format_exc()})
        return ConfigOutcome(hyperparam_id=hp_id, params=dict(params), prefix=prefix,
                             status="failed", error=repr(exc))

    try:
        bundle = bundle_from_parts(
            model_name=arm.model,
            features=features,
            preprocessing_id=arm.preprocessing_id,
            impute_rules=_impute_rules(preprocessor),
            preprocessor=preprocessor,
            model=adapter,
            params=params,
            extra=adapter.state(),
        )
        save_bundle(bundle, join_uri(prefix, "model_artifacts"))
        write_json(join_uri(prefix, "config", "diagnostics.json"), adapter.diagnostics())
        for name, frame in adapter.training_frames().items():
            write_pandas(frame, join_uri(prefix, "metrics", f"{name}.csv"))

        training_scores = adapter.training_scores(train_matrix)
        evaluations = _evaluate_scenarios(
            cfg=cfg,
            arm=arm,
            adapter=adapter,
            preprocessor=preprocessor,
            params=params,
            hp_id=hp_id,
            prefix=prefix,
            scenarios=scenarios,
            scenario_loader=scenario_loader,
            train_matrix=train_matrix,
            training_scores=training_scores,
            thresholds=thresholds,
            write_scores=write_scores,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("configuration %s failed during evaluation", hp_id)
        write_marker(prefix, "FAILED", {"hyperparam_id": hp_id, "params": params,
                                        "fit_seconds": fit_seconds, "error": repr(exc),
                                        "traceback": traceback.format_exc()})
        return ConfigOutcome(hyperparam_id=hp_id, params=dict(params), prefix=prefix,
                             status="failed", fit_seconds=fit_seconds, error=repr(exc))

    metrics = _concat([ev.metrics for ev in evaluations])
    k_metrics = _concat([ev.k_metrics for ev in evaluations])
    if not metrics.empty:
        write_pandas(metrics, join_uri(prefix, _CONFIG_METRICS_FILE))
    if not k_metrics.empty:
        write_pandas(k_metrics, join_uri(prefix, _CONFIG_K_METRICS_FILE))

    write_marker(prefix, "READY", {
        "hyperparam_id": hp_id,
        "params": params,
        "fit_seconds": fit_seconds,
        "scenarios": list(scenarios),
        "n_scenarios_evaluated": int(sum(not ev.is_empty() for ev in evaluations)),
        "n_metric_rows": int(len(metrics)),
        **arm.as_dict(),
    })
    return ConfigOutcome(hyperparam_id=hp_id, params=dict(params), prefix=prefix,
                         status="ready", metrics=metrics, k_metrics=k_metrics,
                         fit_seconds=fit_seconds)


def _evaluate_scenarios(
    cfg: MLConfig,
    arm: SweepArm,
    adapter: Any,
    preprocessor: Any,
    params: Dict[str, Any],
    hp_id: str,
    prefix: str,
    scenarios: Sequence[str],
    scenario_loader: Callable[[str], pd.DataFrame],
    train_matrix: np.ndarray,
    training_scores: np.ndarray,
    thresholds: DriftThresholds,
    write_scores: bool,
) -> List[ScenarioEvaluation]:
    """Score and evaluate each scenario in turn, holding one frame at a time.

    One scenario's frame is loaded, scored, evaluated, written and dropped before
    the next is loaded. Holding them all would multiply the memory footprint by
    the number of windows for no benefit — the metrics are per-scenario and the
    aggregation across scenarios happens on the metric tables, which are tiny.

    The preprocessed matrix is computed once per scenario and passed to both the
    scorer and the drift report, so drift is measured on the array the adapter
    scored rather than on a second transform of the same frame.

    The identity columns are stamped onto every metric row here rather than in
    :mod:`trust_score_05.ml.evaluation`, because they identify the *arm* and the
    evaluation module has no idea which arm called it. Without them the
    concatenated run-level table is a stack of anonymous rows, which is precisely
    what the notebook's cross-model comparison produced.
    """
    identity = {"hyperparam_id": hp_id, "params": str(params), **arm.as_dict()}
    evaluations: List[ScenarioEvaluation] = []
    for scenario in scenarios:
        frame = scenario_loader(scenario)
        # Transformed here, once, and handed to both the scorer and the drift
        # report. Doing it inside `score_population` instead would leave drift
        # with no matrix at all, because the scored frame drops the feature
        # columns by design, and recomputing it would only produce an array that
        # is *probably* the one that was scored.
        eval_matrix = preprocessor.transform(frame)
        scored = score_population(frame, preprocessor, adapter, matrix=eval_matrix)
        del frame
        evaluation = evaluate_population(scored, scenario, keep_scored=True)
        for column, value in identity.items():
            if not evaluation.metrics.empty:
                evaluation.metrics[column] = value
            if not evaluation.k_metrics.empty:
                evaluation.k_metrics[column] = value

        scenario_dir = join_uri(prefix, "metrics", f"scenario={scenario}")
        if not evaluation.metrics.empty:
            write_pandas(evaluation.metrics, join_uri(scenario_dir, "metrics.csv"))
        if not evaluation.k_metrics.empty:
            write_pandas(evaluation.k_metrics, join_uri(scenario_dir, "k_metrics.csv"))
        write_json(join_uri(scenario_dir, "corrections.json"), evaluation.corrections)

        try:
            report = drift_report(
                adapter,
                train_matrix,
                scored,
                eval_matrix,
                training_scores,
                thresholds=thresholds,
            )
            write_json(join_uri(scenario_dir, "drift.json"), report)
        except Exception as exc:  # noqa: BLE001 - drift is a diagnostic
            # Downgraded to a warning because drift is a report *about* the
            # metrics rather than an input to them, and a run whose metrics are
            # sound should not be discarded because a diagnostic failed. The
            # reason is written down, which is more than the notebook's bare
            # `except Exception: pass` around the same step did.
            LOGGER.warning("drift for %s/%s failed: %s", hp_id, scenario, exc)
            write_json(join_uri(scenario_dir, "drift_failed.json"), {"error": repr(exc)})

        if write_scores and evaluation.scored is not None:
            # Parquet under the config's own `scores/` prefix, not under
            # `metrics/`, because these are records rather than a metric table
            # and a consumer reading the metric prefix as one partitioned table
            # would otherwise find a schema per configuration. CSV is the opt-in
            # extra: the scored frame is the largest thing a configuration
            # writes, and `cfg.write_scores_csv` exists so a human can ask for a
            # readable copy of a single run without doubling every run's output.
            write_pandas(
                evaluation.scored,
                join_uri(scenario_scores_prefix(prefix, scenario), "scored_records.parquet"),
            )
            if cfg.write_scores_csv:
                write_pandas(evaluation.scored, join_uri(scenario_dir, "scored_records.csv"))
        evaluation.scored = None
        del scored, eval_matrix
        evaluations.append(evaluation)
    return evaluations


# --------------------------------------------------------------------------
# one arm
# --------------------------------------------------------------------------

def run_arm(
    cfg: MLConfig,
    arm: SweepArm,
    features: Sequence[str],
    train_frame: pd.DataFrame,
    scenarios: Sequence[str],
    scenario_loader: Callable[[str], pd.DataFrame],
    thresholds: Optional[DriftThresholds] = None,
    force: bool = False,
    max_minutes: float = 0.0,
) -> RunOutcome:
    """Fit and evaluate every hyperparameter configuration of one arm.

    Fits the preprocessor once, builds the grid from the resulting matrix's
    actual shape, walks it, and then writes the run-level metric tables, the
    leaderboard and the best configuration.

    The grid is built from the matrix rather than from the requested feature
    count, because the two differ: the requested count is a ceiling on the
    selection, some selected features are absent from the training months, and a
    variant with missing indicators produces *more* matrix columns than input
    features. Several grid dimensions are functions of the row and column counts
    (see :func:`trust_score_05.ml.grids.config_grid_for_model`), so building from
    the request rather than the reality is how a 25-feature arm comes to sweep a
    PCA rank of 40.

    Args:
        cfg: The run's configuration.
        arm: The combination to run.
        features: The selected feature names for this arm.
        train_frame: The raw training sample, with the feature columns.
        scenarios: Evaluation window identifiers.
        scenario_loader: Returns the raw evaluation frame for one scenario.
        thresholds: Drift thresholds. Defaults to the config's.
        force: Refit configurations that are already ``_READY``.
        max_minutes: Wall-clock budget. 0 disables it, per the config
            convention. When it is exceeded the loop stops and the run is marked
            ``INCOMPLETE`` rather than ``READY``.
    """
    thresholds = thresholds or DriftThresholds.from_config(cfg)
    run = arm.prefix(cfg)
    write_marker(run, "STARTED", {**arm.as_dict(), "run_id": cfg.run_id, "mode": cfg.mode})

    try:
        train_matrix, preprocessor = fit_preprocessor(
            train_frame, features, arm.preprocessing_id, seed=cfg.seed
        )
        grid = config_grid_for_model(
            arm.model,
            cfg,
            n_features=int(train_matrix.shape[1]),
            n_rows=int(train_matrix.shape[0]),
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("arm %s could not be prepared", arm)
        write_marker(run, "FAILED", {**arm.as_dict(), "stage": "preprocessing_or_grid",
                                     "error": repr(exc),
                                     "traceback": traceback.format_exc()})
        return RunOutcome(arm=arm, prefix=run, status="FAILED", error=repr(exc))

    write_json(join_uri(run, "config", "run_config.json"), {
        **arm.as_dict(),
        "run_id": cfg.run_id,
        "requested_feature_count": arm.top_n_features,
        "selected_feature_count": len(features),
        "matrix_columns": int(train_matrix.shape[1]),
        "matrix_rows": int(train_matrix.shape[0]),
        "n_hyperparam_configs": len(grid),
        "scenarios": list(scenarios),
        "features": list(features),
        "preprocessing": preprocessor.metadata(),
        "config": cfg.as_dict(),
        "drift_thresholds": thresholds.as_dict(),
    })

    outcomes: List[ConfigOutcome] = []
    started = time.time()
    unattempted = 0
    for index, params in enumerate(grid):
        if max_minutes and (time.time() - started) / 60.0 > max_minutes:
            unattempted = len(grid) - index
            LOGGER.warning(
                "wall-clock guard stopped arm %s after %d of %d configurations",
                arm, index, len(grid),
            )
            break
        outcomes.append(
            run_config(
                cfg=cfg,
                arm=arm,
                params=params,
                train_matrix=train_matrix,
                features=preprocessor.features,
                preprocessor=preprocessor,
                scenarios=scenarios,
                scenario_loader=scenario_loader,
                run=run,
                thresholds=thresholds,
                force=force,
            )
        )

    metrics, k_metrics = collect_metrics(outcomes)
    if not metrics.empty:
        write_pandas(metrics, join_uri(run, "metrics", "all_hyperparams_all_scenario_metrics.csv"))
    if not k_metrics.empty:
        write_pandas(
            k_metrics, join_uri(run, "metrics", "all_hyperparams_all_scenario_k_metrics.csv")
        )

    best = _write_run_reports(cfg, run, metrics, k_metrics)

    tally = {status: sum(o.status == status for o in outcomes) for status in CONFIG_STATUSES}
    if unattempted:
        status = _MARKER_INCOMPLETE
    elif tally["ready"] or tally["skipped"]:
        status = "READY"
    else:
        status = "FAILED"
    payload = {
        **arm.as_dict(),
        "run_id": cfg.run_id,
        "n_configs_total": len(grid),
        "n_configs_unattempted": unattempted,
        **{f"n_configs_{name}": count for name, count in tally.items()},
        "n_metric_rows": int(len(metrics)),
        # Repeated here as well as in config/run_config.json because this marker
        # is the only file a resuming process reads for an arm it is skipping,
        # and `_ready_arm_outcome` has to be able to report the same numbers as
        # a fresh run without opening a second object.
        "matrix_columns": int(train_matrix.shape[1]),
        "matrix_rows": int(train_matrix.shape[0]),
        "best_config": best,
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    write_marker(run, status, payload)
    return RunOutcome(
        arm=arm,
        prefix=run,
        status=status,
        configs=outcomes,
        metrics=metrics,
        k_metrics=k_metrics,
        best=best,
        n_unattempted=unattempted,
        n_features=int(train_matrix.shape[1]),
        marker=payload,
    )


def _write_run_reports(
    cfg: MLConfig,
    run: str,
    metrics: pd.DataFrame,
    k_metrics: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    """The within-arm leaderboard and best configuration.

    Both are best-effort: an arm whose configurations all failed has nothing to
    rank, and that is already recorded in the run marker's counts. What is *not*
    best-effort is that a failure here cannot look like an absence of a winner —
    the returned ``None`` is accompanied by a written reason.
    """
    if k_metrics.empty:
        write_json(join_uri(run, "reports", "best_config.json"),
                   {"selected": False, "reason": "no configuration produced top-k metrics"})
        return None
    try:
        table, summary = leaderboard(
            k_metrics,
            metric=cfg.best_config_metric,
            level=cfg.best_config_level,
            aggregate=cfg.best_config_aggregate,
            k_percent=cfg.best_config_k_percent,
        )
        write_pandas(table, join_uri(run, "reports", "hyperparam_leaderboard.csv"))
        write_json(join_uri(run, "reports", "hyperparam_leaderboard.json"), summary)
        best = select_best_config(
            k_metrics,
            metric=cfg.best_config_metric,
            level=cfg.best_config_level,
            k_percent=cfg.best_config_k_percent,
            aggregate=cfg.best_config_aggregate,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("best-configuration selection failed for %s: %s", run, exc)
        write_json(join_uri(run, "reports", "best_config.json"),
                   {"selected": False, "reason": repr(exc)})
        return None

    payload = {"selected": True, **best.as_dict()}
    write_json(join_uri(run, "reports", "best_config.json"), payload)
    if not metrics.empty:
        write_pandas(
            metrics.loc[metrics["hyperparam_id"] == best.hyperparam_id],
            join_uri(run, "reports", "best_config_scenario_metrics.csv"),
        )
    return payload


def collect_metrics(
    outcomes: Iterable[ConfigOutcome],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The run-level metric tables, including the configurations this job skipped.

    A configuration skipped because it was already ``_READY`` carries no frames
    in memory, so its tables are read back from its prefix. Without that step the
    run-level leaderboard of a resumed job contains only the configurations the
    resume happened to execute, and the best configuration is then selected from
    a subset determined by the restart history rather than by the grid — two
    resumes of one sweep naming different winners from identical data. This is
    the notebook's behaviour, and it is silent: the truncated table looks exactly
    like a complete one.

    A skipped configuration whose tables cannot be read is logged and omitted,
    because the alternative — failing the run — would make one corrupt prefix
    from an old sweep permanently block the current one.
    """
    metric_frames: List[pd.DataFrame] = []
    k_frames: List[pd.DataFrame] = []
    for outcome in outcomes:
        if outcome.status == "ready":
            metric_frames.append(outcome.metrics)
            k_frames.append(outcome.k_metrics)
            continue
        if outcome.status != "skipped":
            continue
        for filename, sink in (
            (_CONFIG_METRICS_FILE, metric_frames),
            (_CONFIG_K_METRICS_FILE, k_frames),
        ):
            uri = join_uri(outcome.prefix, filename)
            if not s3_exists(uri):
                LOGGER.warning(
                    "configuration %s is READY but %s is missing; it will be absent "
                    "from the run-level table",
                    outcome.hyperparam_id, filename,
                )
                continue
            try:
                sink.append(pd.read_csv(uri))
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("could not read %s: %s", uri, exc)
    return _concat(metric_frames), _concat(k_frames)


def _concat(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate, dropping empties, and return an empty frame for nothing.

    The empties are dropped before the concatenation rather than after because
    ``pd.concat`` on a list containing an all-NA or zero-column frame emits a
    FutureWarning and, in some pandas versions, upcasts the dtypes of the
    non-empty frames to object — which turns every metric column into strings
    and every subsequent comparison into a lexicographic one.
    """
    usable = [frame for frame in frames if frame is not None and not frame.empty]
    if not usable:
        return pd.DataFrame()
    return pd.concat(usable, ignore_index=True)


# --------------------------------------------------------------------------
# the whole sweep
# --------------------------------------------------------------------------

def run_sweep(
    cfg: MLConfig,
    arms: Sequence[SweepArm],
    feature_loader: Callable[[SweepArm], Sequence[str]],
    training_loader: Callable[[Sequence[str]], pd.DataFrame],
    scenarios: Sequence[str],
    scenario_loader: Callable[[SweepArm, str], pd.DataFrame],
    force: bool = False,
    max_minutes_per_arm: float = 0.0,
    skip_ready_arms: bool = True,
) -> List[RunOutcome]:
    """Run every arm, and keep going when one of them fails.

    An arm that raises is recorded and the sweep continues, for the same reason a
    failed configuration does not fail its arm: the arms are independent, and one
    model's missing optional package should not cost the other nine their
    results. The returned list has one entry per arm regardless of outcome, so a
    caller can count what happened without re-listing object storage.

    ``skip_ready_arms`` skips an arm whose run prefix already has a ``_READY``
    marker without loading its training sample. That is the cheap outer half of
    resumption — it avoids a six-figure-row Spark read per already-finished arm —
    and it is safe precisely because the marker is only written when the arm
    genuinely completed, per :func:`run_arm`. It does *not* substitute for the
    per-configuration check, since an arm can be ``INCOMPLETE``.

    Args:
        cfg: The run's configuration.
        arms: What to run, typically from :func:`sweep_arms`.
        feature_loader: The selected feature names for an arm.
        training_loader: The raw training sample given a feature list. Called
            once per arm; a caller that walks arms in :func:`sweep_arms` order
            can memoise on the feature list, since consecutive preprocessing
            variants share one. It may return fewer columns than were asked for
            — :func:`trust_score_05.ml.datasets.load_training_matrix` drops a
            feature absent from every sampled month — and this function narrows
            the list to what came back before fitting anything on it.
        scenarios: Evaluation window identifiers.
        scenario_loader: The raw evaluation frame for an ``(arm, scenario)``
            pair. Takes the arm because the frame is projected onto the arm's
            feature list.
        force: Refit configurations that are already ``_READY``.
        max_minutes_per_arm: Per-arm wall-clock budget. 0 disables it.
        skip_ready_arms: Skip arms whose run prefix is already ``_READY``.
    """
    if not arms:
        raise SweepError("no arms to run")
    thresholds = DriftThresholds.from_config(cfg)
    results: List[RunOutcome] = []

    for arm in arms:
        run = arm.prefix(cfg)
        if skip_ready_arms and not force and ready_exists(run):
            LOGGER.info("skipping ready arm %s", arm)
            results.append(_ready_arm_outcome(arm, run))
            continue
        try:
            features = list(feature_loader(arm))
            if not features:
                raise SweepError(f"no selected features for {arm}")
            train_frame = training_loader(features)
            # Narrowed to what the loader actually returned, in selection order.
            # A feature that no sampled month carries is dropped by
            # `load_training_matrix`, and passing the unnarrowed list on would
            # fail the arm inside `fit_preprocessor` with a missing-column error
            # naming a feature the operator can see in the selection report and
            # cannot see in the data.
            present = [name for name in features if name in train_frame.columns]
            if not present:
                raise SweepError(
                    f"none of the {len(features)} selected features for {arm} are "
                    "columns of the training sample"
                )
            if len(present) != len(features):
                LOGGER.warning(
                    "arm %s: %d of %d selected features are absent from the training "
                    "sample and were dropped",
                    arm, len(features) - len(present), len(features),
                )
            outcome = run_arm(
                cfg=cfg,
                arm=arm,
                features=present,
                train_frame=train_frame,
                scenarios=scenarios,
                scenario_loader=lambda scenario, _arm=arm: scenario_loader(_arm, scenario),
                thresholds=thresholds,
                force=force,
                max_minutes=max_minutes_per_arm,
            )
        except Exception as exc:  # noqa: BLE001 - one arm, not the sweep
            LOGGER.exception("arm %s failed", arm)
            write_marker(run, "FAILED", {**arm.as_dict(), "stage": "arm_setup",
                                         "error": repr(exc),
                                         "traceback": traceback.format_exc()})
            outcome = RunOutcome(arm=arm, prefix=run, status="FAILED", error=repr(exc))
        results.append(outcome)

    return results


def _ready_arm_outcome(arm: SweepArm, run: str) -> RunOutcome:
    """A :class:`RunOutcome` for an arm skipped because it was already ready.

    The counts come from the existing ``_READY.json`` rather than being left at
    zero, so a caller summarising the sweep sees the same numbers whether an arm
    ran in this process or a previous one.
    """
    outcome = RunOutcome(arm=arm, prefix=run, status="READY")
    try:
        marker = read_json(join_uri(run, "_READY.json"))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("could not read the READY marker for %s: %s", run, exc)
        return outcome
    if not isinstance(marker, dict):
        LOGGER.warning("the READY marker for %s is not an object; ignoring it", run)
        return outcome
    outcome.marker = marker
    outcome.best = marker.get("best_config")
    outcome.n_features = int(marker.get("matrix_columns", 0) or 0)
    return outcome
