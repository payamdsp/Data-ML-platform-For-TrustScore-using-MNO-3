"""Top-*k* metrics: what an analyst actually sees.

Nobody reviews a hundred thousand scored records. An investigations team works a
queue, and the queue has a length. So the metric that decides whether a model is
useful is not its AUC but its recall at the *k* the team can actually get
through — and that *k* is a business constraint, not a model parameter. Hence the
grid: every metric here is reported at many *k* values so that whoever owns the
queue can read off the row that matches their capacity.

Four quantities per *k*, and they answer different questions:

``precision_at_k``
    Of the *k* records reviewed, what fraction were fraud. This is the analyst's
    hit rate, and it is what determines whether the queue feels worth working.
``recall_at_k``
    Of all the fraud there was, what fraction the top *k* caught. This is the
    business outcome.
``fraud_rate_in_top_k``
    Numerically identical to precision at *k*. Kept because the two names are
    read by different audiences and the "rate" framing is what sits next to the
    base rate in a report.
``lift_at_k``
    ``fraud_rate_in_top_k / base_rate``. How many times better than random. The
    only one of the four that is comparable *across populations*, because the
    base rate is divided out — which makes it the right metric for a cross-model
    leaderboard whose rows come from scenarios with different fraud prevalence.

The notebook also emitted ``enrichment_at_k``, computed by the same expression as
``lift_at_k`` from the same inputs. Two columns holding the same float in every
row is not redundancy, it is a trap: a reader comparing "lift" and "enrichment"
concludes they measure different things. There is one column here.

Ties matter more than they look. Several models produce heavily tied scores —
HDBSCAN gives every core point an outlier score of 0.0, and a k-means cluster
percentile variant produces exact ties by construction — so the tie-break decides
which records land inside the top *k* and therefore decides the metric. The
tie-break here is by ``entity_id`` and it is applied to a *string* column, which
is the only stable choice when the identifier is a customer ID. The notebook
built record-level entity IDs with ``np.arange(n).astype(str)`` and then sorted
them as strings, which orders ``'10'`` before ``'2'`` — so record-level ties were
broken in lexicographic order over decimal representations, an ordering with no
meaning at all and one that changes when the population size crosses a power of
ten.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "K_GRID_FIXED",
    "K_GRID_PERCENTAGES",
    "K_METRIC_COLUMNS",
    "make_k_grid",
    "compute_k_metrics",
]

#: Absolute queue lengths. These are the sizes a review team plausibly has:
#: one alert, a shift's worth, a week's worth, a whole backlog. Reported even
#: when they are a vanishing fraction of the population, because "how good is
#: the very top of the list" is the question a pilot asks.
K_GRID_FIXED = (
    1,
    5,
    10,
    25,
    50,
    100,
    250,
    500,
    1_000,
    2_500,
    5_000,
    10_000,
    25_000,
    50_000,
    100_000,
)

#: Population fractions, as proportions rather than percentages. Dense at the
#: sharp end — the first six are all under 1% — because that is where an anomaly
#: detector either works or does not, and coarse above it.
K_GRID_PERCENTAGES = (
    0.001,
    0.002,
    0.003,
    0.004,
    0.005,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.75,
    0.80,
    0.95,
    1.0,
)

#: The schema of the frame :func:`compute_k_metrics` returns, in order.
K_METRIC_COLUMNS = (
    "scenario",
    "level",
    "k",
    "k_percent",
    "total_records",
    "total_fraud",
    "base_rate_percent",
    "true_positive_count_at_k",
    "false_positive_count_at_k",
    "precision_at_k",
    "recall_at_k",
    "fraud_rate_in_top_k",
    "lift_at_k",
    "score_at_k",
)


def make_k_grid(
    n_records: int,
    n_fraud: int = 0,
    fixed: Optional[Sequence[int]] = None,
    percentages: Optional[Sequence[float]] = None,
) -> List[int]:
    """The sorted, de-duplicated *k* values to report for a population.

    The union of :data:`K_GRID_FIXED`, the rounded fractions in
    :data:`K_GRID_PERCENTAGES`, and ``n_fraud`` itself — that last one because
    ``k == n_fraud`` is where precision and recall are equal, which makes it the
    single most quotable point on the curve.

    Values outside ``[1, n_records]`` are dropped rather than clamped. Clamping
    would collapse every fixed *k* above the population size onto ``n_records``,
    producing a dozen identical rows claiming to be different queue lengths.
    """
    total = int(n_records)
    if total <= 0:
        return []

    candidates: List[int] = list(fixed if fixed is not None else K_GRID_FIXED)
    if n_fraud:
        candidates.append(int(n_fraud))
    for fraction in percentages if percentages is not None else K_GRID_PERCENTAGES:
        candidates.append(max(1, int(round(total * float(fraction)))))

    return sorted({k for k in (int(c) for c in candidates) if 1 <= k <= total})


def _ranking_order(scores: np.ndarray, entity_ids: Sequence[Any]) -> np.ndarray:
    """Indices that sort by score descending, breaking ties by ``entity_id``.

    ``np.lexsort`` sorts by its *last* key first, so the keys are given in
    reverse priority: the entity ID is the minor key and the negated score the
    major one. Negating rather than reversing keeps the tie-break ascending while
    the score descends, which is the intended combination — reversing the whole
    result would put the tie-break in descending order too, and then the
    tie-break would depend on the direction the scores happened to run.

    Entity IDs are compared as strings. They arrive as strings from every
    aggregation level (see :func:`~trust_score_05.common.splitting.aggregate_level`),
    and the caller is responsible for making them meaningful — at record level
    that means zero-padding, which is why that function pads.
    """
    ids = np.asarray([str(value) for value in entity_ids], dtype=object)
    return np.lexsort((ids, -scores))


def compute_k_metrics(
    labels: Sequence[Any],
    scores: Sequence[float],
    entity_ids: Sequence[Any],
    scenario: str,
    level: str,
    k_values: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """Top-*k* metrics for one scored population, one row per *k*.

    Returns a list of dicts rather than a DataFrame so that the caller can
    concatenate across scenarios and levels once, instead of building and
    discarding a frame per call. Every row carries the keys in
    :data:`K_METRIC_COLUMNS`.

    ``score_at_k`` is the score of the *k*-th record — the cutoff that would
    produce this queue. Reported because it is what a consumer needs in order to
    turn a chosen operating point into a deployable threshold, and because
    comparing it across scenarios is the cheapest possible drift check: if the
    score at 1% has doubled between two months, the model's calibration has
    moved even if its recall has not.

    Raises ``ValueError`` on a length mismatch or on non-finite scores, for the
    reason given in :func:`~trust_score_05.common.metrics.classification.compute_binary_metrics`.
    """
    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(
        [0 if value is None or value != value else int(value) for value in labels],
        dtype=np.int64,
    )

    if not (len(label_array) == len(score_array) == len(entity_ids)):
        raise ValueError(
            "labels, scores and entity_ids must be the same length; got "
            f"{len(label_array)}, {len(score_array)}, {len(entity_ids)}"
        )

    total = int(label_array.shape[0])
    if total == 0:
        return []

    non_finite = int(np.sum(~np.isfinite(score_array)))
    if non_finite:
        raise ValueError(
            f"{non_finite} of {total} scores are NaN or infinite; a population "
            "cannot be ranked."
        )

    order = _ranking_order(score_array, entity_ids)
    ranked_labels = label_array[order]
    ranked_scores = score_array[order]

    total_fraud = int(ranked_labels.sum())
    base_rate = total_fraud / total
    # One cumulative sum serves every k, rather than re-summing a prefix per k.
    # With a 24-point percentage grid on a million rows the difference is a
    # second against several minutes.
    cumulative_tp = np.cumsum(ranked_labels)

    grid = list(k_values) if k_values is not None else make_k_grid(total, total_fraud)
    rows: List[Dict[str, Any]] = []
    for k in grid:
        k = int(k)
        if not 1 <= k <= total:
            continue
        true_positives = int(cumulative_tp[k - 1])
        precision = true_positives / k
        rows.append(
            {
                "scenario": scenario,
                "level": level,
                "k": k,
                "k_percent": 100.0 * k / total,
                "total_records": total,
                "total_fraud": total_fraud,
                "base_rate_percent": 100.0 * base_rate,
                "true_positive_count_at_k": true_positives,
                "false_positive_count_at_k": k - true_positives,
                "precision_at_k": precision,
                "recall_at_k": (true_positives / total_fraud) if total_fraud else float("nan"),
                "fraud_rate_in_top_k": precision,
                "lift_at_k": (precision / base_rate) if base_rate > 0 else float("nan"),
                "score_at_k": float(ranked_scores[k - 1]),
            }
        )
    return rows
