"""Threshold-free classification metrics for a scored population.

Two numbers, plus the counts that make them interpretable: ROC AUC and average
precision. Both are computed from the ranking alone, with no threshold, which is
the right shape for this problem — nothing in the pipeline ever converts an
anomaly score into a decision. (The notebook tuned a ``contamination``
hyperparameter on every pyod model and then never used the resulting threshold
anywhere, which is why seven distinct ECOD configurations produced seven
byte-identical rankings.)

The convention throughout: ``label`` is 1 for fraud, ``score`` is higher for more
anomalous. Every model adapter is responsible for orienting its score that way
before it gets here — Isolation Forest's ``score_samples`` is *lower* for more
anomalous and is negated in its adapter, not here.

Why average precision needs care: it is sensitive to the base rate, so an average
precision of 0.02 on a population with a 0.1% fraud rate is far better than 0.02
on one with a 5% rate. That makes it a reasonable metric for comparing models on
*the same* population and a bad one for comparing across populations. The
notebook's cross-model leaderboard sorted every ``(config, scenario)`` row in one
table by average precision, and since the four scenarios had different base
rates, the ranking largely recovered which scenario a row came from. Anything
comparing across populations should use :func:`lift_at_k` from
:mod:`trust_score_05.common.metrics.ranking`, which divides the base rate out.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

__all__ = [
    "BINARY_METRIC_COLUMNS",
    "compute_binary_metrics",
    "score_summary",
]

#: The columns :func:`compute_binary_metrics` always returns, in order. Fixed so
#: that a frame built from many calls has a stable schema even when some calls
#: hit the single-class case and return NaN for the two AUC-style metrics.
BINARY_METRIC_COLUMNS = (
    "n_records",
    "n_fraud",
    "fraud_rate",
    "roc_auc",
    "average_precision",
)


def _as_label_array(labels: Sequence[Any]) -> np.ndarray:
    """Coerce labels to an integer 0/1 array, treating null as 0.

    Null-as-0 is correct here and is not a convenience. The evaluation population
    is a union of a labelled fraud set and an unlabelled non-fraud sample; the
    non-fraud rows have no label column of their own and are assigned 0 when the
    two are unioned. A null that survives to this point is a non-fraud row whose
    label was never set, not a fraud row whose label was lost.
    """
    array = np.asarray(labels)
    if array.dtype == object or np.issubdtype(array.dtype, np.floating):
        array = np.where(_isnull(array), 0, array)
    return np.asarray(array, dtype=np.int64)


def _isnull(array: np.ndarray) -> np.ndarray:
    """Elementwise null test that works on object and float arrays alike."""
    if array.dtype == object:
        return np.array([value is None or value != value for value in array], dtype=bool)
    if np.issubdtype(array.dtype, np.floating):
        return np.isnan(array)
    return np.zeros(array.shape, dtype=bool)


def compute_binary_metrics(
    labels: Sequence[Any],
    scores: Sequence[float],
) -> Dict[str, Any]:
    """Ranking quality of ``scores`` against ``labels``.

    Returns the keys in :data:`BINARY_METRIC_COLUMNS`. ``roc_auc`` and
    ``average_precision`` are ``nan`` when the population has only one class,
    because both are undefined there — not zero, and not one. A scenario whose
    fraud rows all got filtered out is a broken scenario, and it must not appear
    on a leaderboard with a defensible-looking 0.5.

    Raises ``ValueError`` on a length mismatch or on non-finite scores. The
    notebook wrapped both metric calls in bare ``except: nan``, which meant a
    model that emitted NaN scores for every row — the autoencoder does this when
    its learning rate diverges, and 0.1 was in the grid — reported ``nan`` for its
    metrics and was then dropped from the leaderboard by a ``dropna``, so a
    catastrophically broken configuration was indistinguishable from one that had
    not finished.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    label_array = _as_label_array(labels)
    score_array = np.asarray(scores, dtype=float)

    if label_array.shape[0] != score_array.shape[0]:
        raise ValueError(
            f"labels and scores have different lengths: "
            f"{label_array.shape[0]} vs {score_array.shape[0]}"
        )

    n_records = int(label_array.shape[0])
    n_fraud = int(np.sum(label_array == 1))
    out: Dict[str, Any] = {
        "n_records": n_records,
        "n_fraud": n_fraud,
        "fraud_rate": float(n_fraud / n_records) if n_records else 0.0,
        "roc_auc": float("nan"),
        "average_precision": float("nan"),
    }

    if n_records == 0:
        return out

    non_finite = int(np.sum(~np.isfinite(score_array)))
    if non_finite:
        raise ValueError(
            f"{non_finite} of {n_records} scores are NaN or infinite. A model that "
            "cannot score its own input has failed; it must not be reported as "
            "having produced an undefined metric."
        )

    if np.unique(label_array).size < 2:
        return out

    out["roc_auc"] = float(roc_auc_score(label_array, score_array))
    out["average_precision"] = float(average_precision_score(label_array, score_array))
    return out


def score_summary(
    scores: Sequence[float],
    quantiles: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Distributional summary of a score column.

    Recorded per scenario alongside the metrics, and the reason is drift: two
    scenarios can have near-identical ROC AUC while the score *scale* has moved
    by an order of magnitude, and a downstream consumer that has chosen a cutoff
    needs to know. This is the cheap half of what
    :mod:`trust_score_05.ml.drift` does properly.
    """
    array = np.asarray(scores, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "score_count": 0.0,
            "score_non_finite": float(array.size),
            "score_mean": float("nan"),
            "score_std": float("nan"),
        }

    grid = tuple(quantiles) if quantiles is not None else (0.01, 0.25, 0.5, 0.75, 0.95, 0.99)
    out: Dict[str, float] = {
        "score_count": float(finite.size),
        "score_non_finite": float(array.size - finite.size),
        "score_mean": float(np.mean(finite)),
        "score_std": float(np.std(finite)),
        "score_min": float(np.min(finite)),
        "score_max": float(np.max(finite)),
    }
    for quantile in grid:
        # Formatted rather than interpolated raw so that 0.5 becomes `p50` and
        # 0.995 becomes `p99_5`, both of which are legal column names.
        label = f"score_p{quantile * 100:g}".replace(".", "_")
        out[label] = float(np.quantile(finite, quantile))
    return out
