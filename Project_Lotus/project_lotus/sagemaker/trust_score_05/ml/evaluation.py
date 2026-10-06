"""Scoring an evaluation population, and choosing between configurations.

Three stages, and the boundaries between them are the point.

:func:`score_population` turns a scenario frame into a scored frame. It applies a
fitted :class:`~trust_score_05.ml.preprocessing.Preprocessor` and a fitted
:class:`~trust_score_05.ml.models.ModelAdapter` and computes nothing else. No
statistic is estimated here, because a statistic estimated at scoring time is a
statistic that depends on which rows happened to be scored together — the defect
that ran through four of the notebook's k-means variants and is documented at
:class:`~trust_score_05.ml.models._ClusterScoreStats`.

:func:`evaluate_population` turns a scored frame into metric rows. It does the
two population corrections first — de-duplicating fraud cases and removing fraud
customers from the negative class — then reports at all three grains in
:data:`~trust_score_05.common.splitting.AGGREGATION_LEVELS`. The corrections
belong here rather than in :mod:`trust_score_05.ml.datasets` because they change
the *denominators* of the metrics, so they have to happen in the same place that
records what those denominators were: every returned row carries the row counts
the corrections produced, which is the only way a later reader can tell whether
a recall figure was against 400 cases or 400 case-transactions.

:func:`select_best_config` reads the metric rows back and picks one
configuration. This is a separate function taking a frame rather than a step
inside the sweep, because best-configuration selection has to be re-runnable
against a finished run's artifacts — a run whose selection criterion changed
should not need re-fitting.

What the notebook got wrong here, beyond the population corrections:

* Its selection sorted the metric table by ``average_precision`` across all
  ``(config, scenario)`` rows at once. Average precision is base-rate sensitive
  (see :mod:`trust_score_05.common.metrics.classification`), the four scenarios
  had materially different fraud prevalence, and the sort therefore largely
  recovered *which scenario a row came from* rather than which configuration was
  better. Selection here aggregates across scenarios before comparing, and the
  default criterion is a top-*k* metric at a fixed queue length.
* It aggregated across scenarios by taking the maximum. A configuration that is
  excellent on one month and useless on the other three then outranks one that
  is good on all four, which is the opposite of what a deployment wants. The
  default here is the mean, with the minimum and the spread reported alongside
  so that an unstable configuration is visible rather than merely averaged.
* It selected on ``level="record"`` while reporting the customer level as the
  business number, so the configuration deployed was not the one the report
  argued for. The level is a config field
  (:attr:`~trust_score_05.ml.config.MLConfig.best_config_level`) and is recorded
  in the selection output.
* Configurations that produced no metrics at all — a failed fit, an empty
  scenario — were dropped by a ``dropna`` before the sort, so a sweep in which
  most arms failed reported a winner with no indication of how few arms it beat.
  :class:`BestConfig` carries ``n_configs_considered`` and
  ``n_configs_incomplete``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from trust_score_05.common.metrics.classification import (
    BINARY_METRIC_COLUMNS,
    compute_binary_metrics,
    score_summary,
)
from trust_score_05.common.metrics.ranking import compute_k_metrics, make_k_grid
from trust_score_05.common.splitting import (
    AGGREGATION_LEVELS,
    CASE_KEY_COLUMN,
    CUSTOMER_KEY_COLUMN,
    LABEL_COLUMN,
    aggregate_level,
    deduplicate_fraud,
    remove_fraud_customers,
)
from trust_score_05.ml.models import ModelAdapter
from trust_score_05.ml.preprocessing import Preprocessor

__all__ = [
    "AGGREGATION_LEVELS",
    "SCORE_COLUMN",
    "BestConfig",
    "EvaluationError",
    "ScenarioEvaluation",
    "evaluate_population",
    "leaderboard",
    "score_population",
    "select_best_config",
]

#: The column :func:`score_population` writes and everything downstream reads.
#: Named once here rather than spelled as a literal at each of the dozen call
#: sites, which is how the notebook came to have both ``anomaly_score`` and
#: ``score`` in its scored parquet.
SCORE_COLUMN = "anomaly_score"

#: Columns copied from the scenario frame onto the scored frame, when present.
#: Deliberately not the feature columns: the scored output is joined back to the
#: features by row identity when anybody needs them, and carrying two hundred
#: float columns through a parquet write per configuration per scenario is what
#: made the notebook's score output larger than its input.
_CARRIED_COLUMNS = (
    LABEL_COLUMN,
    CUSTOMER_KEY_COLUMN,
    CASE_KEY_COLUMN,
    "scenario",
    "fraud_source",
    "fraud_type",
    "fraud_category",
    "fraud_timestamp",
    "reference_id",
    "reference_timestamp",
)


class EvaluationError(ValueError):
    """A population could not be scored or evaluated as configured."""


@dataclass
class ScenarioEvaluation:
    """Everything one ``(configuration, scenario)`` pair produced.

    Held together rather than returned as a tuple because the three parts are
    written to three different places under the same prefix and a caller that
    receives them separately has to keep them associated by convention.

    Attributes:
        scenario: The evaluation window's identifier.
        metrics: One row per aggregation level: the binary metrics, the score
            summary, and the population counts.
        k_metrics: One row per ``(level, k)``.
        scored: The scored frame, or ``None`` when the caller asked not to keep
            it. Kept optional because the sweep writes it to parquet and then has
            no further use for it, and holding it would mean holding one frame
            per scenario for the life of a configuration.
        corrections: What the two population corrections did, for the manifest.
    """

    scenario: str
    metrics: pd.DataFrame
    k_metrics: pd.DataFrame
    scored: Optional[pd.DataFrame] = None
    corrections: Dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return self.metrics.empty


def score_population(
    frame: pd.DataFrame,
    preprocessor: Preprocessor,
    adapter: ModelAdapter,
    keep_columns: Sequence[str] = _CARRIED_COLUMNS,
    matrix: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Score ``frame`` with a fitted preprocessor and a fitted model.

    Returns a new frame carrying the identifier and label columns from
    ``keep_columns`` that were present, plus :data:`SCORE_COLUMN`. The row order
    is the input's, so the caller can align the result back to the input by
    position.

    ``matrix`` lets a caller that has already transformed ``frame`` hand the
    result in rather than have it recomputed, and that is not merely an economy.
    :func:`~trust_score_05.ml.drift.drift_report` needs the preprocessed
    evaluation matrix, and the frame returned here deliberately does not carry
    the feature columns it could be rebuilt from, so a caller without this
    argument must either transform a second time — with nothing tying the second
    matrix to the one that was actually scored — or hold the raw frame alive
    beside the scores for the rest of the configuration. Passing the matrix in
    makes the array drift measures the same array the adapter scored.

    Both fitted arguments must already be fitted. ``adapter`` is asked for
    :meth:`~trust_score_05.ml.models.ModelAdapter.score`, never ``fit``, and the
    preprocessor for ``transform``, never ``fit_transform`` — the distinction
    that keeps evaluation statistics out of the training set. The notebook's
    scoring path called ``fit_transform`` on the scaler for the evaluation
    population, so the standardisation applied at evaluation time was fitted on
    the evaluation rows, including the fraud rows; a model trained to find
    outliers in the training distribution was then shown a population re-centred
    on its own mean, which flattened exactly the deviations it was looking for.

    Raises:
        EvaluationError: If the frame is empty, if the model is not fitted, if a
            supplied ``matrix`` has a different number of rows than ``frame``, or
            if scoring produces a length other than one score per row.
    """
    if frame is None or len(frame) == 0:
        raise EvaluationError("cannot score an empty population")
    if not adapter.fitted:
        raise EvaluationError(
            f"the {adapter.name!r} adapter is not fitted; score_population never fits"
        )

    if matrix is None:
        matrix = preprocessor.transform(frame)
    elif len(matrix) != len(frame):
        # A supplied matrix that does not line up row-for-row with the frame
        # would attach every score to the wrong record, silently, and every
        # metric downstream would be computed over a shuffled population. Cheap
        # to check, impossible to notice later.
        raise EvaluationError(
            f"the supplied matrix has {len(matrix)} rows and the frame {len(frame)}; "
            "they must be the same population in the same order"
        )
    scores = np.asarray(adapter.score(matrix), dtype=float)
    if scores.shape != (len(frame),):
        raise EvaluationError(
            f"{adapter.name} returned {scores.shape} scores for {len(frame)} rows"
        )

    carried = [name for name in keep_columns if name in frame.columns]
    scored = frame.loc[:, carried].copy()
    scored.reset_index(drop=True, inplace=True)
    scored[SCORE_COLUMN] = scores
    if LABEL_COLUMN not in scored.columns:
        # A population with no label column is scorable but not evaluable, and
        # the distinction is worth keeping: this function is also used to score
        # an unlabelled production population.
        scored[LABEL_COLUMN] = 0
    scored[LABEL_COLUMN] = scored[LABEL_COLUMN].fillna(0).astype(float).astype(int)
    return scored


def _apply_corrections(
    scored: pd.DataFrame,
    deduplicate: bool,
    decontaminate: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """De-duplicate fraud cases and drop fraud customers from the negative class.

    Order matters and this is the only correct one. De-duplication runs first, on
    the positive rows alone, so the set of fraud customer identifiers used for
    de-contamination is taken from the *retained* cases. Reversing the two would
    de-contaminate against the identifiers of duplicate rows that are about to be
    dropped, removing negatives that correspond to no surviving positive — which
    shrinks the negative class without shrinking the positive one and inflates
    every precision figure.
    """
    summary: Dict[str, Any] = {"rows_in": int(len(scored))}
    labels = scored[LABEL_COLUMN].to_numpy()
    positives = scored.loc[labels == 1]
    negatives = scored.loc[labels != 1]

    if deduplicate and not positives.empty:
        positives, dedup_summary = deduplicate_fraud(positives)
        summary["deduplicate_fraud"] = dedup_summary
    else:
        summary["deduplicate_fraud"] = {"skipped": True, "rows_out": int(len(positives))}

    if decontaminate and not positives.empty and not negatives.empty:
        if CUSTOMER_KEY_COLUMN not in negatives.columns:
            # Not an error: a scored population assembled without customer
            # identifiers cannot be de-contaminated, and saying so in the
            # manifest is better than raising and losing the whole scenario.
            # Recorded rather than logged because it changes how the precision
            # figures in this scenario should be read.
            summary["remove_fraud_customers"] = {
                "skipped": True,
                "reason": f"no {CUSTOMER_KEY_COLUMN!r} column on the negative class",
                "rows_out": int(len(negatives)),
            }
        else:
            fraud_ids = (
                positives[CUSTOMER_KEY_COLUMN].dropna().unique()
                if CUSTOMER_KEY_COLUMN in positives.columns
                else []
            )
            negatives, decon_summary = remove_fraud_customers(negatives, fraud_ids)
            summary["remove_fraud_customers"] = decon_summary
    else:
        summary["remove_fraud_customers"] = {
            "skipped": True,
            "rows_out": int(len(negatives)),
        }

    corrected = pd.concat([negatives, positives], ignore_index=True)
    summary["rows_out"] = int(len(corrected))
    summary["n_fraud"] = int((corrected[LABEL_COLUMN] == 1).sum())
    summary["n_nonfraud"] = summary["rows_out"] - summary["n_fraud"]
    return corrected, summary


def evaluate_population(
    scored: pd.DataFrame,
    scenario: str,
    levels: Sequence[str] = AGGREGATION_LEVELS,
    k_values: Optional[Sequence[int]] = None,
    deduplicate: bool = True,
    decontaminate: bool = True,
    keep_scored: bool = False,
) -> ScenarioEvaluation:
    """Metric tables for one scored population, at every requested grain.

    Args:
        scored: Output of :func:`score_population`.
        scenario: The window identifier, written into every row.
        levels: A subset of :data:`~trust_score_05.common.splitting.AGGREGATION_LEVELS`.
            A level whose key column is absent is skipped and recorded in
            ``corrections["levels_skipped"]`` rather than raising, so a fraud
            feed that lost its case key still yields record and customer
            metrics — but unlike the notebook, the omission is written down.
        k_values: Queue lengths. Defaults to
            :func:`~trust_score_05.common.metrics.ranking.make_k_grid` computed
            *per level*, which is the correction: a grid computed once on the
            record count and reused at the customer level asks for a top-100,000
            queue from a population of 30,000 customers, and every such *k* is
            dropped, so the customer-level table silently reported fewer rows
            than the record-level one.
        deduplicate: Collapse the fraud population to one row per case.
        decontaminate: Remove fraud customers from the negative class.
        keep_scored: Attach the corrected scored frame to the result.

    Returns:
        A :class:`ScenarioEvaluation`. Both frames are empty when the population
        has no rows after the corrections; that is a valid outcome for a scenario
        whose window contained no data, and the caller distinguishes it from a
        failure by ``is_empty()``.
    """
    if scored is None or len(scored) == 0:
        return ScenarioEvaluation(
            scenario=scenario,
            metrics=pd.DataFrame(),
            k_metrics=pd.DataFrame(),
            corrections={"rows_in": 0, "rows_out": 0},
        )
    if SCORE_COLUMN not in scored.columns:
        raise EvaluationError(
            f"scored frame has no {SCORE_COLUMN!r} column; it has {sorted(scored.columns)[:12]}…"
        )

    corrected, corrections = _apply_corrections(scored, deduplicate, decontaminate)
    if corrected.empty:
        return ScenarioEvaluation(
            scenario=scenario,
            metrics=pd.DataFrame(),
            k_metrics=pd.DataFrame(),
            corrections=corrections,
        )

    metric_rows: List[Dict[str, Any]] = []
    k_rows: List[Dict[str, Any]] = []
    skipped: Dict[str, str] = {}

    for level in levels:
        try:
            entities = aggregate_level(corrected, level, score_col=SCORE_COLUMN)
        except KeyError as exc:
            skipped[level] = str(exc)
            continue
        if entities.empty:
            skipped[level] = "no rows after aggregation"
            continue

        labels = entities["label"].to_numpy()
        scores = entities[SCORE_COLUMN].to_numpy(dtype=float)
        binary = compute_binary_metrics(labels, scores)
        grid = (
            list(k_values)
            if k_values is not None
            else make_k_grid(len(entities), int(labels.sum()))
        )

        row: Dict[str, Any] = {"scenario": scenario, "level": level}
        row.update(binary)
        row.update(score_summary(scores))
        row["n_records_before_aggregation"] = int(len(corrected))
        metric_rows.append(row)
        k_rows.extend(
            compute_k_metrics(
                labels,
                scores,
                entities["entity_id"].to_numpy(),
                scenario=scenario,
                level=level,
                k_values=grid,
            )
        )

    if skipped:
        corrections["levels_skipped"] = skipped

    metrics = pd.DataFrame(metric_rows)
    if not metrics.empty:
        leading = ["scenario", "level", *BINARY_METRIC_COLUMNS]
        metrics = metrics.loc[:, leading + [c for c in metrics.columns if c not in leading]]

    return ScenarioEvaluation(
        scenario=scenario,
        metrics=metrics,
        k_metrics=pd.DataFrame(k_rows),
        scored=corrected if keep_scored else None,
        corrections=corrections,
    )


# --------------------------------------------------------------------------
# choosing between configurations
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BestConfig:
    """The chosen configuration, and enough context to argue with the choice.

    Attributes:
        hyperparam_id: The identifier of the winning configuration.
        metric: The metric it was chosen on.
        level: The aggregation level it was chosen at.
        k: The queue length the metric was read at, or ``None`` for a
            threshold-free metric.
        k_percent: The requested queue length as a percentage of the population.
        value: The aggregated metric value.
        worst_scenario_value: The same metric on the configuration's *worst*
            scenario. Reported beside the aggregate because a mean hides
            instability, and instability is the thing that makes a sweep winner
            fail in production.
        scenario_spread: Max minus min across scenarios.
        n_scenarios: How many scenarios contributed.
        n_configs_considered: How many configurations had usable metrics.
        n_configs_incomplete: How many were excluded for missing a scenario or
            producing a non-finite value. Non-zero means the leaderboard is over
            a subset, which is a fact about the run, not a detail.
        table: The full aggregated comparison, best first.
    """

    hyperparam_id: str
    metric: str
    level: str
    k: Optional[int]
    k_percent: Optional[float]
    value: float
    worst_scenario_value: float
    scenario_spread: float
    n_scenarios: int
    n_configs_considered: int
    n_configs_incomplete: int
    table: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def as_dict(self) -> Dict[str, Any]:
        """A JSON-serialisable record for the run manifest, without the table."""
        return {
            "hyperparam_id": self.hyperparam_id,
            "metric": self.metric,
            "level": self.level,
            "k": self.k,
            "k_percent": self.k_percent,
            "value": float(self.value),
            "worst_scenario_value": float(self.worst_scenario_value),
            "scenario_spread": float(self.scenario_spread),
            "n_scenarios": int(self.n_scenarios),
            "n_configs_considered": int(self.n_configs_considered),
            "n_configs_incomplete": int(self.n_configs_incomplete),
        }


#: Metrics that live in the per-*k* table rather than the per-level one. Any
#: other metric name is looked up in the per-level table and ``k`` is ignored.
_K_METRICS = (
    "precision_at_k",
    "recall_at_k",
    "fraud_rate_in_top_k",
    "lift_at_k",
    "true_positive_count_at_k",
)

#: How scenario values are combined. ``max`` is offered and is not the default,
#: for the reason in the module docstring.
_AGGREGATORS = ("mean", "median", "min", "max")


def _resolve_k(frame: pd.DataFrame, k_percent: float) -> Tuple[Optional[int], pd.DataFrame]:
    """The row subset at the queue length nearest ``k_percent``, per scenario.

    Nearest rather than exact, because ``k`` is an integer count and the *k*
    grid is built from rounded fractions of each scenario's population — so an
    exact match on ``k_percent == 1.0`` exists in no scenario whose population
    is not a multiple of 100. The notebook filtered on equality against a
    hardcoded ``k_percent`` and got an empty frame whenever the population size
    was not round, then fell back to the whole table, which selected on the
    metric at ``k = n`` where recall is 1.0 for every configuration.

    Resolved per scenario, since two scenarios of different size have different
    integer *k* at the same percentage.
    """
    if frame.empty:
        return None, frame
    target = float(k_percent)
    scenarios = (
        frame["scenario"]
        if "scenario" in frame.columns
        else pd.Series("", index=frame.index)
    )
    picked: List[pd.DataFrame] = []
    for _, group in frame.groupby(scenarios, dropna=False):
        distance = (group["k_percent"].astype(float) - target).abs()
        near = group.loc[distance == distance.min()]
        # Ties on distance resolve to the smaller queue: of two equally close
        # queue lengths, the shorter one is the more conservative claim.
        #
        # Every row at the chosen k is kept, not one of them. Keeping one would
        # collapse the frame to a single configuration per scenario, and the
        # leaderboard built from it would rank whichever configuration happened
        # to sort first against nothing at all.
        picked.append(group.loc[group["k"] == near["k"].min()])
    subset = pd.concat(picked, ignore_index=True)
    k = int(subset["k"].iloc[0]) if len(subset) else None
    return k, subset


def leaderboard(
    metrics: pd.DataFrame,
    metric: str,
    level: str,
    aggregate: str = "mean",
    k_percent: Optional[float] = None,
    require_all_scenarios: bool = True,
    group_columns: Sequence[str] = ("hyperparam_id",),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Aggregate a metric across scenarios and rank the groups, best first.

    ``group_columns`` is what makes this reusable for both jobs it has to do.
    Grouping by ``hyperparam_id`` compares configurations within one run;
    grouping by the segments of
    :data:`~trust_score_05.ml.paths.RUN_PREFIX_SEGMENTS` compares across models,
    feature-selection methods and preprocessing variants. The notebook had two
    separate implementations of this, and the cross-model one concatenated
    metrics CSVs found by a recursive listing *without parsing their paths*, so
    its rows carried no model identity and its leaderboard could not say which
    model a row belonged to (see
    :func:`~trust_score_05.ml.paths.parse_run_prefix`).

    Returns ``(table, summary)``. The table has one row per group with the
    aggregated value plus ``value_min``, ``value_max``, ``value_spread`` and
    ``n_scenarios``. Higher is better for every metric this function accepts, so
    the sort is always descending.
    """
    if aggregate not in _AGGREGATORS:
        raise EvaluationError(
            f"unknown aggregator {aggregate!r}; expected one of {list(_AGGREGATORS)}"
        )
    if metrics is None or metrics.empty:
        return pd.DataFrame(), {"n_configs_considered": 0, "n_configs_incomplete": 0}

    keys = [name for name in group_columns if name in metrics.columns]
    if not keys:
        raise EvaluationError(
            f"none of the grouping columns {list(group_columns)} are present; "
            f"the frame has {sorted(metrics.columns)[:12]}…"
        )
    if "level" not in metrics.columns or metric not in metrics.columns:
        raise EvaluationError(
            f"the frame has no {metric!r} column at all; it has "
            f"{sorted(metrics.columns)[:12]}…"
        )

    frame = metrics.loc[metrics["level"].astype(str) == str(level)].copy()
    if frame.empty:
        return pd.DataFrame(), {
            "n_configs_considered": 0,
            "n_configs_incomplete": 0,
            "note": f"no rows at level={level!r}",
        }

    resolved_k: Optional[int] = None
    if metric in _K_METRICS:
        if "k" not in frame.columns:
            raise EvaluationError(
                f"{metric!r} is a top-k metric but the frame has no 'k' column; "
                "pass the k_metrics table, not the per-level metrics table"
            )
        resolved_k, frame = _resolve_k(frame, 1.0 if k_percent is None else k_percent)

    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    n_scenarios_total = int(frame["scenario"].nunique()) if "scenario" in frame else 1

    grouped = (
        frame.groupby(keys, dropna=False)[metric]
        .agg(["mean", "median", "min", "max", "count"])
        .reset_index()
    )
    grouped = grouped.rename(
        columns={
            aggregate: "value",
            "min": "value_min",
            "max": "value_max",
            "count": "n_scenarios",
        }
    )
    if "value" not in grouped.columns:
        # `aggregate` was "min" or "max", which the rename above also consumed
        # as the spread columns. Copy rather than rename in that case.
        grouped["value"] = grouped[f"value_{aggregate}"]
    grouped["value_spread"] = grouped["value_max"] - grouped["value_min"]

    considered = int(len(grouped))
    incomplete_mask = ~np.isfinite(grouped["value"].to_numpy(dtype=float))
    if require_all_scenarios and n_scenarios_total > 1:
        incomplete_mask = incomplete_mask | (
            grouped["n_scenarios"].to_numpy() < n_scenarios_total
        )
    complete = grouped.loc[~incomplete_mask].copy()

    # Sort descending on the value, then ascending on the spread, then on the
    # group key. The spread is a genuine tie-break rather than decoration: two
    # configurations with the same mean recall are not equally good, and the
    # steadier one is the one to deploy. The final key makes the order
    # reproducible when both are equal.
    complete = complete.sort_values(
        by=["value", "value_spread", *keys],
        ascending=[False, True, *[True] * len(keys)],
        kind="mergesort",
    ).reset_index(drop=True)

    summary = {
        "metric": metric,
        "level": str(level),
        "aggregate": aggregate,
        "k": resolved_k,
        "k_percent": None if metric not in _K_METRICS else float(k_percent or 1.0),
        "n_scenarios": n_scenarios_total,
        "n_configs_considered": considered,
        "n_configs_incomplete": int(incomplete_mask.sum()),
        "group_columns": keys,
    }
    return complete, summary


def select_best_config(
    metrics: pd.DataFrame,
    metric: str = "recall_at_k",
    level: str = "customer",
    k_percent: float = 1.0,
    aggregate: str = "mean",
    require_all_scenarios: bool = True,
) -> BestConfig:
    """Pick the configuration with the best aggregated metric across scenarios.

    The defaults match :class:`~trust_score_05.ml.config.MLConfig`'s
    ``best_config_*`` fields — recall at a 1% queue, at the customer grain,
    averaged over scenarios — and a caller should pass the config's values rather
    than relying on these, so that the criterion a run used is recorded in its
    config rather than in this signature.

    Raises:
        EvaluationError: If no configuration has a usable value. Raising is
            deliberate: a sweep that finished with nothing selectable is a failed
            sweep, and the notebook's behaviour of returning the first row of an
            empty-after-``dropna`` frame produced a "best configuration" that was
            whichever arm sorted first alphabetically.
    """
    table, summary = leaderboard(
        metrics,
        metric=metric,
        level=level,
        aggregate=aggregate,
        k_percent=k_percent,
        require_all_scenarios=require_all_scenarios,
        group_columns=("hyperparam_id",),
    )
    if table.empty:
        raise EvaluationError(
            f"no configuration has a usable {metric!r} at level={level!r}: "
            f"{summary.get('n_configs_considered', 0)} considered, "
            f"{summary.get('n_configs_incomplete', 0)} incomplete. "
            f"{summary.get('note', '')}".strip()
        )

    best = table.iloc[0]
    return BestConfig(
        hyperparam_id=str(best["hyperparam_id"]),
        metric=metric,
        level=str(level),
        k=summary.get("k"),
        k_percent=summary.get("k_percent"),
        value=float(best["value"]),
        worst_scenario_value=float(best["value_min"]),
        scenario_spread=float(best["value_spread"]),
        n_scenarios=int(best["n_scenarios"]),
        n_configs_considered=int(summary["n_configs_considered"]),
        n_configs_incomplete=int(summary["n_configs_incomplete"]),
        table=table,
    )
