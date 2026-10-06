"""Has the population moved since the model was trained?

Two different questions, and the notebook only asked the narrower one.

**Cluster drift** compares the shape of the population as the model partitions
it: what share of rows falls in each cluster, and how far from its centroid the
average row now sits. It applies only to models that partition — k-means,
weighted k-means, HDBSCAN — and :func:`cluster_drift` computes it from the
adapter's own assignments.

**Score drift** compares the distribution of the anomaly score itself, and it
applies to every model. This is the one the notebook never computed. Its drift
step was guarded by ``if model_name not in {'kmeans','weighted_kmeans','hdbscan'}:
return``, so seven of the ten models had no drift output of any kind — including
every model a consumer would actually deploy behind a fixed threshold, which is
the case where a score-scale shift is the failure that matters. A model whose
ROC AUC is unchanged but whose 99th percentile score has doubled will send twice
the intended volume to a review queue, and nothing in the notebook's output said
so.

Why relative rather than absolute thresholds, and why a distribution-level
statistic exists at all: the notebook flagged a cluster when its share of the
population changed by more than a fixed 0.05 in absolute terms, over a grid in
which ``k`` ran from 2 to 200. At ``k=2`` the average cluster holds half the
population and 0.05 is a 10% move — the flag fires on noise. At ``k=200`` the
average cluster holds 0.005 and *no* cluster can move by 0.05 without emptying
several others — the flag cannot fire. So the meaning of "drift detected"
depended on a hyperparameter, which makes the flag uncomparable across exactly
the dimension the sweep varies. :func:`cluster_drift` reports the absolute
difference too, because it is what a human reads, but flags on the ratio; and
:func:`population_stability_index` gives one number per population that is
comparable across ``k`` and across models.

The other corrections to the notebook's version:

* Its outer merge filled the *count* columns for a cluster missing from one side
  but not the *distance* columns, so a cluster that existed in training and was
  empty in production got ``prod_mean_distance = NaN``, hence
  ``distance_ratio = NaN``, and ``NaN > threshold`` is ``False``. A cluster that
  had vanished entirely was therefore reported as not drifting. Same for a
  training cluster whose mean distance was 0: the code replaced the zero with
  NaN to avoid a division error and inherited the same silent ``False``.
  :func:`cluster_drift` returns an explicit ``status`` per cluster and never
  reports ``drift=False`` for a cluster it could not evaluate.
* Its thresholds came from environment variables read at the drift call site.
  They are config fields here, so a run's thresholds are recoverable from its
  manifest.
* It re-read the scored population back from object storage to compute drift
  from data it had just held in memory, and returned silently when that read
  found nothing — so a run whose score write had been skipped produced no drift
  output and no complaint. These functions take arrays.
* It computed production cluster shares over the whole evaluation population,
  fraud rows included. The fraud rows are the ones expected to sit in unusual
  clusters, so including them measures "is there fraud in this window" and
  reports it as "has the population moved". :func:`cluster_drift` takes the
  non-fraud rows; :func:`drift_report` selects them from the label column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from trust_score_05.common.splitting import LABEL_COLUMN

__all__ = [
    "CLUSTER_DRIFT_COLUMNS",
    "DRIFT_STATUSES",
    "DriftError",
    "DriftThresholds",
    "cluster_drift",
    "drift_report",
    "population_stability_index",
    "score_drift",
    "total_variation_distance",
]

#: The schema of :func:`cluster_drift`'s frame, in order.
CLUSTER_DRIFT_COLUMNS = (
    "cluster_label",
    "train_count",
    "train_proportion",
    "train_mean_distance",
    "eval_count",
    "eval_proportion",
    "eval_mean_distance",
    "proportion_abs_diff",
    "proportion_ratio",
    "distance_ratio",
    "proportion_drift_flag",
    "distance_drift_flag",
    "status",
)

#: What :func:`cluster_drift` can conclude about one cluster. ``unevaluable``
#: exists so that "we could not tell" is never written down as "no drift" — the
#: substitution the notebook made by relying on ``NaN > threshold`` being false.
DRIFT_STATUSES = ("ok", "drifted", "vanished", "emerged", "unevaluable")

_EPS = 1e-12


class DriftError(ValueError):
    """Drift could not be computed from the inputs given."""


@dataclass(frozen=True)
class DriftThresholds:
    """The three thresholds, together, so they travel as one thing.

    Built from :class:`~trust_score_05.ml.config.MLConfig` by
    :meth:`from_config` and written into the run manifest by
    :meth:`as_dict`. Kept as a separate object rather than three loose floats
    because every drift artifact has to record the thresholds it was judged
    against — a drift flag without its threshold is not interpretable next year,
    and the notebook wrote the flags and the thresholds to two different files
    with no key linking them.
    """

    proportion_ratio: float = 0.25
    distance_ratio: float = 1.25
    psi: float = 0.25

    @classmethod
    def from_config(cls, cfg: Any) -> "DriftThresholds":
        return cls(
            proportion_ratio=float(cfg.cluster_drift_proportion_ratio),
            distance_ratio=float(cfg.cluster_drift_distance_ratio),
            psi=float(cfg.score_drift_psi_threshold),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cluster_proportion_ratio_threshold": self.proportion_ratio,
            "cluster_mean_distance_ratio_threshold": self.distance_ratio,
            "score_psi_threshold": self.psi,
            "acceptance_criteria": (
                "A cluster is flagged when its share of the population changes by "
                "more than proportion_ratio relative to its training share, or when "
                "its mean distance to centroid exceeds distance_ratio times the "
                "training mean. The population is flagged when the anomaly score's "
                "stability index exceeds score_psi_threshold."
            ),
        }


# --------------------------------------------------------------------------
# distribution-level statistics
# --------------------------------------------------------------------------

def population_stability_index(
    reference: Sequence[float],
    current: Sequence[float],
    n_bins: int = 10,
    edges: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """The population stability index of ``current`` against ``reference``.

    ``sum((c - r) * log(c / r))`` over bins, the standard credit-risk statistic,
    with the conventional reading: below 0.1 is stable, 0.1 to 0.25 is a moderate
    shift, above 0.25 is a significant one.

    Bin edges come from the *reference* distribution's quantiles, and from the
    reference alone. Quantile bins of the reference hold a known equal share of
    it — one tenth each at the default — so the current distribution's departure
    from equal shares *is* the signal, and the ``edges`` argument lets a caller
    reuse one training-time cut across many evaluation windows so their indices
    are mutually comparable.

    The error this rules out is binning each distribution by *its own*
    quantiles, which is what applying :func:`pandas.qcut` separately to the two
    samples gives: both sides then hold an equal share in every bin by
    construction, and the index is exactly zero however far apart the
    distributions are. Binning the pooled sample is a milder version of the same
    mistake — it does detect a location shift, but the edges move with whatever
    is being measured, so two evaluation windows judged against the same
    training population get different bins and their indices cannot be compared
    to each other.

    Empty bins are given a floor rather than dropped. Dropping them — which is
    what a naïve implementation does to avoid ``log(0)`` — discards precisely the
    bins where the shift is total, so the most severe drift contributes least to
    the index. The floor makes an emptied bin contribute a large finite amount,
    and ``n_current_empty_bins`` is returned so a reader can see how much of the
    index came from that.

    Returns a dict rather than a float because the index alone is not actionable:
    the caller needs the bin count and the empty-bin count to know how much to
    trust it, and the per-bin shares to see *where* the mass moved.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        raise DriftError(
            f"cannot compute a stability index from {ref.size} reference and "
            f"{cur.size} current finite values"
        )

    if edges is None:
        n_bins = max(2, int(n_bins))
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        cut = np.unique(np.quantile(ref, quantiles))
        if cut.size < 2:
            # A constant reference distribution has no quantile structure. Two
            # bins around the single value is the most that can be said, and it
            # still distinguishes "current is also constant" from "current has
            # spread out", which is the question.
            cut = np.array([ref[0] - _EPS, ref[0] + _EPS])
    else:
        cut = np.unique(np.asarray(edges, dtype=float))
        if cut.size < 2:
            raise DriftError("edges must contain at least two distinct values")

    # Open the outer bins so that current values beyond the reference range are
    # counted in the extreme bins rather than falling outside the histogram. A
    # value more extreme than anything seen in training is drift, not an absence
    # of data.
    interior = cut[1:-1]
    if interior.size == 0:
        # Two distinct edges leave no interior edge, so opening both outer ones
        # would collapse the histogram to the single bin ``(-inf, inf)``, in
        # which every distribution has a share of 1 and the index is
        # unconditionally zero. Splitting at the midpoint keeps two bins, which
        # is the least that can distinguish one distribution from another. This
        # is reachable from a reference that is constant, or one whose values
        # are so tied that its quantiles collapse to two — a score column that
        # is mostly zeros, which several of the detectors produce.
        interior = np.array([0.5 * (cut[0] + cut[-1])])
    bounded = np.concatenate(([-np.inf], interior, [np.inf]))
    ref_counts, _ = np.histogram(ref, bins=bounded)
    cur_counts, _ = np.histogram(cur, bins=bounded)

    ref_share = ref_counts / max(1, ref.size)
    cur_share = cur_counts / max(1, cur.size)
    # The floor is one tenth of a row's worth of share, so an empty bin behaves
    # like "fewer than one row would have landed here" rather than like zero.
    floor = 0.1 / max(ref.size, cur.size)
    ref_floored = np.maximum(ref_share, floor)
    cur_floored = np.maximum(cur_share, floor)
    contributions = (cur_floored - ref_floored) * np.log(cur_floored / ref_floored)
    psi = float(contributions.sum())

    return {
        "psi": psi,
        "n_bins": int(ref_share.size),
        "n_reference": int(ref.size),
        "n_current": int(cur.size),
        "n_current_empty_bins": int(np.sum(cur_counts == 0)),
        "n_reference_empty_bins": int(np.sum(ref_counts == 0)),
        "max_bin_contribution": float(np.max(np.abs(contributions))),
        "bin_edges": [float(value) for value in cut],
        "reference_share": [float(value) for value in ref_share],
        "current_share": [float(value) for value in cur_share],
    }


def total_variation_distance(
    reference_counts: Sequence[float],
    current_counts: Sequence[float],
) -> float:
    """Half the L1 distance between two count vectors read as distributions.

    In ``[0, 1]``, and comparable across cluster counts, which is what makes it
    the right summary for a partition whose ``k`` is swept: 0 means the two
    populations distribute identically over the clusters and 1 means they share
    no cluster at all, at every ``k``. The per-cluster absolute differences that
    the notebook thresholded do not have that property.
    """
    ref = np.asarray(reference_counts, dtype=float)
    cur = np.asarray(current_counts, dtype=float)
    if ref.shape != cur.shape:
        raise DriftError(f"count vectors have different lengths: {ref.shape} vs {cur.shape}")
    ref_total = ref.sum()
    cur_total = cur.sum()
    if ref_total <= 0 or cur_total <= 0:
        raise DriftError("a count vector sums to zero; there is no distribution to compare")
    return float(0.5 * np.abs(ref / ref_total - cur / cur_total).sum())


# --------------------------------------------------------------------------
# cluster drift
# --------------------------------------------------------------------------

def cluster_drift(
    train_labels: Sequence[int],
    train_distances: Sequence[float],
    eval_labels: Sequence[int],
    eval_distances: Sequence[float],
    thresholds: Optional[DriftThresholds] = None,
    n_clusters: Optional[int] = None,
) -> pd.DataFrame:
    """Per-cluster share and distance drift between two populations.

    Every cluster present in either population gets a row, and a cluster present
    in only one gets ``status`` of ``vanished`` or ``emerged`` rather than a
    ``NaN`` ratio and a ``False`` flag. HDBSCAN's noise label ``-1`` is a cluster
    here like any other; a population whose noise share has tripled has drifted,
    and excluding the label would hide the clearest signal HDBSCAN produces.

    Args:
        train_labels: Cluster assignment per training row.
        train_distances: Distance to the assigned centroid per training row.
        eval_labels: Cluster assignment per evaluation row. Non-fraud rows only —
            see the module docstring.
        eval_distances: Distance per evaluation row.
        thresholds: Defaults to :class:`DriftThresholds`'s own defaults.
        n_clusters: Include empty clusters up to this count. Passing it means a
            cluster that was empty in training *and* in evaluation still appears,
            which is how a reader sees that the model has dead clusters at all.

    Returns:
        A frame with :data:`CLUSTER_DRIFT_COLUMNS`, one row per cluster, ordered
        by label.
    """
    thresholds = thresholds or DriftThresholds()
    train_l = np.asarray(train_labels, dtype=np.int64)
    eval_l = np.asarray(eval_labels, dtype=np.int64)
    train_d = np.asarray(train_distances, dtype=float)
    eval_d = np.asarray(eval_distances, dtype=float)

    if train_l.shape != train_d.shape:
        raise DriftError(
            f"training labels and distances differ in length: {train_l.shape} vs {train_d.shape}"
        )
    if eval_l.shape != eval_d.shape:
        raise DriftError(
            f"evaluation labels and distances differ in length: {eval_l.shape} vs {eval_d.shape}"
        )
    if train_l.size == 0 or eval_l.size == 0:
        raise DriftError(
            f"cannot compare a {train_l.size}-row training population with a "
            f"{eval_l.size}-row evaluation population"
        )

    labels = sorted(set(train_l.tolist()) | set(eval_l.tolist()))
    if n_clusters is not None:
        labels = sorted(set(labels) | set(range(int(n_clusters))))

    rows = []
    for label in labels:
        train_mask = train_l == label
        eval_mask = eval_l == label
        train_count = int(train_mask.sum())
        eval_count = int(eval_mask.sum())
        train_share = train_count / train_l.size
        eval_share = eval_count / eval_l.size
        # `nan` for an empty cluster's mean distance, which is honest: there is
        # no distance to report. What must not happen is that nan flowing into a
        # comparison and coming out as "not drifted", which is why `status` is
        # decided before the flags and overrides them.
        train_mean = float(train_d[train_mask].mean()) if train_count else float("nan")
        eval_mean = float(eval_d[eval_mask].mean()) if eval_count else float("nan")

        proportion_ratio = (
            abs(eval_share - train_share) / train_share if train_share > 0 else float("nan")
        )
        distance_ratio = (
            eval_mean / train_mean
            if train_count and eval_count and train_mean > _EPS
            else float("nan")
        )

        if train_count == 0 and eval_count == 0:
            status, prop_flag, dist_flag = "unevaluable", False, False
        elif train_count == 0:
            status, prop_flag, dist_flag = "emerged", True, False
        elif eval_count == 0:
            status, prop_flag, dist_flag = "vanished", True, False
        else:
            prop_flag = bool(
                np.isfinite(proportion_ratio) and proportion_ratio > thresholds.proportion_ratio
            )
            # A training cluster with a zero mean distance -- every member
            # exactly on the centroid, which happens for a singleton -- has no
            # ratio to take. It is `unevaluable` on distance rather than
            # silently unflagged.
            if not np.isfinite(distance_ratio):
                dist_flag = False
                status = "drifted" if prop_flag else "unevaluable"
            else:
                dist_flag = bool(distance_ratio > thresholds.distance_ratio)
                status = "drifted" if (prop_flag or dist_flag) else "ok"

        rows.append(
            {
                "cluster_label": int(label),
                "train_count": train_count,
                "train_proportion": train_share,
                "train_mean_distance": train_mean,
                "eval_count": eval_count,
                "eval_proportion": eval_share,
                "eval_mean_distance": eval_mean,
                "proportion_abs_diff": abs(eval_share - train_share),
                "proportion_ratio": proportion_ratio,
                "distance_ratio": distance_ratio,
                "proportion_drift_flag": prop_flag,
                "distance_drift_flag": dist_flag,
                "status": status,
            }
        )

    return pd.DataFrame(rows, columns=list(CLUSTER_DRIFT_COLUMNS))


def score_drift(
    train_scores: Sequence[float],
    eval_scores: Sequence[float],
    thresholds: Optional[DriftThresholds] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Distribution drift of the anomaly score itself, for any model.

    Returns the stability index and its diagnostics, the shift in each reported
    quantile, and the fraction of the evaluation population scoring above the
    training 99th percentile. That last number is the operationally important
    one: it is the factor by which a queue fed by a fixed threshold will grow.
    A model can hold its ROC AUC exactly while that fraction triples, and the
    notebook — which computed no score drift at all — had no way to see it.
    """
    thresholds = thresholds or DriftThresholds()
    train = np.asarray(train_scores, dtype=float)
    evaluate = np.asarray(eval_scores, dtype=float)
    train = train[np.isfinite(train)]
    evaluate = evaluate[np.isfinite(evaluate)]
    if train.size == 0 or evaluate.size == 0:
        raise DriftError(
            f"cannot compare {train.size} finite training scores with "
            f"{evaluate.size} finite evaluation scores"
        )

    index = population_stability_index(train, evaluate, n_bins=n_bins)
    out: Dict[str, Any] = {
        "psi": index["psi"],
        "psi_threshold": thresholds.psi,
        "psi_drift_flag": bool(index["psi"] > thresholds.psi),
        "psi_n_bins": index["n_bins"],
        "psi_max_bin_contribution": index["max_bin_contribution"],
        "psi_current_empty_bins": index["n_current_empty_bins"],
        "n_train_scores": int(train.size),
        "n_eval_scores": int(evaluate.size),
    }

    for quantile in (0.5, 0.9, 0.99, 0.999):
        label = f"p{quantile * 100:g}".replace(".", "_")
        train_value = float(np.quantile(train, quantile))
        eval_value = float(np.quantile(evaluate, quantile))
        out[f"train_{label}"] = train_value
        out[f"eval_{label}"] = eval_value
        out[f"{label}_ratio"] = (
            eval_value / train_value if abs(train_value) > _EPS else float("nan")
        )

    cutoff = float(np.quantile(train, 0.99))
    exceed = float(np.mean(evaluate > cutoff))
    out["train_p99_cutoff"] = cutoff
    out["eval_fraction_above_train_p99"] = exceed
    # 0.01 by construction on the training population, so the ratio is the
    # multiple a fixed-threshold queue would grow by.
    out["queue_growth_at_train_p99"] = exceed / 0.01
    return out


def drift_report(
    adapter: Any,
    train_matrix: np.ndarray,
    scored: pd.DataFrame,
    eval_matrix: np.ndarray,
    train_scores: Sequence[float],
    thresholds: Optional[DriftThresholds] = None,
    score_column: str = "anomaly_score",
) -> Dict[str, Any]:
    """Both kinds of drift for one fitted model against one evaluation window.

    Score drift is always computed. Cluster drift is computed when the adapter
    exposes ``cluster_assignments`` — which the clustering adapters do and the
    others do not — so the capability is discovered from the object rather than
    from a hardcoded set of model names. The notebook's
    ``if model_name not in {'kmeans','weighted_kmeans','hdbscan'}`` meant adding
    a clustering model required remembering to edit the drift step, and adding a
    non-clustering one under a name in that set would have crashed it.

    The evaluation side uses the *non-fraud* rows only, selected from
    ``scored[LABEL_COLUMN]``. Drift asks whether the background population has
    moved; the fraud rows are the ones expected to be unusual, and including them
    reports the presence of fraud as a change in the population.

    Returns a dict with ``score`` (always), ``cluster`` (a list of records, when
    applicable), ``cluster_summary`` and ``thresholds``.
    """
    thresholds = thresholds or DriftThresholds()
    if score_column not in scored.columns:
        raise DriftError(f"scored frame has no {score_column!r} column")
    if len(scored) != len(eval_matrix):
        raise DriftError(
            f"the scored frame has {len(scored)} rows and the evaluation matrix "
            f"{len(eval_matrix)}; they must be the same population in the same order"
        )

    if LABEL_COLUMN in scored.columns:
        background = scored[LABEL_COLUMN].fillna(0).astype(float).to_numpy() != 1
    else:
        background = np.ones(len(scored), dtype=bool)
    if not background.any():
        raise DriftError(
            "the evaluation population is entirely fraud rows; there is no "
            "background population to measure drift against"
        )

    report: Dict[str, Any] = {
        "thresholds": thresholds.as_dict(),
        "n_eval_rows": int(len(scored)),
        "n_eval_background_rows": int(background.sum()),
        "score": score_drift(
            train_scores,
            scored.loc[background, score_column].to_numpy(dtype=float),
            thresholds=thresholds,
        ),
    }

    assign = getattr(adapter, "cluster_assignments", None)
    if not callable(assign):
        report["cluster"] = None
        report["cluster_summary"] = {
            "applicable": False,
            "reason": f"the {getattr(adapter, 'name', '?')!r} adapter does not partition",
        }
        return report

    try:
        train_labels, train_distances = assign(train_matrix)
        eval_labels, eval_distances = assign(np.asarray(eval_matrix)[background])
    except ValueError as exc:
        # An adapter that partitions in principle but has no partition for *this*
        # fit -- HDBSCAN with the optional clusterer unimportable, which
        # :class:`~trust_score_05.ml.models.HDBSCANAdapter` downgrades to a
        # warning because the clusterer takes no part in scoring. The arm's
        # scores and metrics are unaffected, so the reason is recorded and the
        # score drift already computed above is still returned. Failing here
        # would let a missing optional package decide whether a model that does
        # not use it gets evaluated.
        report["cluster"] = None
        report["cluster_summary"] = {"applicable": False, "reason": str(exc)}
        return report

    frame = cluster_drift(
        train_labels,
        train_distances,
        eval_labels,
        eval_distances,
        thresholds=thresholds,
    )
    report["cluster"] = frame.to_dict(orient="records")
    report["cluster_summary"] = {
        "applicable": True,
        "n_clusters": int(len(frame)),
        "n_drifted": int((frame["status"] == "drifted").sum()),
        "n_vanished": int((frame["status"] == "vanished").sum()),
        "n_emerged": int((frame["status"] == "emerged").sum()),
        "n_unevaluable": int((frame["status"] == "unevaluable").sum()),
        # One k-comparable number for the whole partition, which is what a
        # cross-configuration comparison needs and what a table of per-cluster
        # flags cannot provide.
        "total_variation_distance": total_variation_distance(
            frame["train_count"].to_numpy(), frame["eval_count"].to_numpy()
        ),
        "max_distance_ratio": float(np.nanmax(frame["distance_ratio"].to_numpy()))
        if np.isfinite(frame["distance_ratio"].to_numpy()).any()
        else float("nan"),
    }
    return report
