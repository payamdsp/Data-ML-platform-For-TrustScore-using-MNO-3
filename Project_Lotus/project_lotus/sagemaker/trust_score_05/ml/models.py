"""Ten anomaly-detection models behind one interface.

Every adapter in here answers the same two questions — fit on this matrix, score
that matrix — and answers them with the same orientation: **a higher score is
more anomalous**. Nothing downstream of this module knows which model produced a
score column, and that is deliberate, because the evaluation stage ranks a
population and a ranking is the only thing the ten models have in common. An
isolation forest returns a signed path-length average where *low* means
anomalous; a k-means distance is unbounded above; a PyOD detector returns a
decision function calibrated around its own threshold. The sign flip and the
orientation are each adapter's business, settled once at fit time, so that
:mod:`trust_score_05.ml.evaluation` never asks.

The adapter *is* the model artifact. :class:`ModelAdapter` instances are what
:class:`~trust_score_05.common.io.serialization.ModelBundle` stores in its
``model`` field, and that is the fix for the worst of the notebook's modelling
defects. The notebook represented a fitted model as a dict tagged with a
``kind`` string and dispatched scoring on that string. The weighted k-means fit
returned ``{"kind": "kmeans", ...}`` in one of the two code paths that produced
it, so the weighted model's centroids — which live in the space stretched by
``sqrt(weights)`` — were scored against unweighted rows. Every weighted k-means
number in the notebook's leaderboard was produced that way. When the object that
knows how to score is the object that was fitted, the two cannot disagree.

The second systemic defect was scoring-time refitting. Four of the notebook's
score paths computed a statistic from the frame being scored: the k-means
``cluster_z_distance`` and ``cluster_size_adjusted`` variants took the mean and
standard deviation of the distances *of the rows in front of them*, the
``cluster_percentile`` variant took each row's rank among them, and HDBSCAN
returned the clusterer's training GLOSH scores whenever the frame it was handed
happened to have the same row count as the training matrix. All four make a
row's score depend on which other rows were scored alongside it, which means the
fraud population's own concentration in the tail is what defines the tail. Each
adapter here freezes those statistics at fit time; see
:class:`_ClusterScoreStats` and :class:`HDBSCANAdapter`.

The third was silent substitution. A missing ``pyod`` turned an ECOD run into a
robust z-score over medians and absolute deviations, still labelled ``ecod``,
still written to the ``model=ecod`` prefix, still compared against real ECOD
runs on the leaderboard. Fallbacks here compute *the same statistic* by another
route where that is possible — see :class:`_TailProbabilityScorer` — and every
adapter reports an ``implementation`` string in its
:meth:`~ModelAdapter.diagnostics`, which the metrics rows carry, so a fallback
run is identifiable after the fact rather than indistinguishable.

The families, and what each is for:

``kmeans``, ``weighted_kmeans``
    Distance to the assigned centroid of a partition fitted on the non-fraud
    population. Cheap, and interpretable in the one way that matters
    operationally: an analyst can be shown the cluster a flagged account was
    assigned to and what that cluster looks like.
``hdbscan``
    Distance to the k-th nearest non-fraud neighbour. Density rather than
    partition, so it does not force every row into a cluster.
``isolation_forest``, ``imf``
    Average isolation depth. ``imf`` is an iMondrian forest — the same idea with
    a split process that can be extended by new points without a refit, which is
    the property that makes it a candidate for a model kept current between
    retrains.
``copod``, ``ecod``, ``hbos``
    Coordinate-wise tail probability and histogram density. No pairwise
    distances, so they scale to the full population and are indifferent to
    feature scaling.
``pca_reconstruction``, ``autoencoder``
    Reconstruction error under a bottleneck fitted on non-fraud. Linear and
    non-linear versions of one hypothesis: fraud is what the normal population's
    own low-dimensional structure cannot express.

Hyperparameter grids are not here. They are in :mod:`trust_score_05.ml.grids`,
because a grid is a statement about the search budget and depends on the shape
of the data, while an adapter is a statement about an algorithm and does not.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

__all__ = [
    "EULER_MASCHERONI",
    "KMEANS_SCORE_VARIANTS",
    "MODEL_NAMES",
    "WEIGHT_STRATEGIES",
    "AutoencoderAdapter",
    "COPODAdapter",
    "ECODAdapter",
    "HBOSAdapter",
    "HDBSCANAdapter",
    "IMondrianForest",
    "IMondrianForestAdapter",
    "IsolationForestAdapter",
    "KMeansAdapter",
    "ModelAdapter",
    "ModelError",
    "ModelUnavailableError",
    "PCAReconstructionAdapter",
    "WeightedKMeansAdapter",
    "adapter_class",
    "average_path_length",
    "build_model",
    "fit_model",
    "resolve_pca_components",
]

LOGGER = logging.getLogger(__name__)

#: The ten adapters, in the order they are reported. Fixed so that a leaderboard
#: built from a directory listing has a stable row order regardless of which
#: models a partial sweep happened to finish.
MODEL_NAMES: Tuple[str, ...] = (
    "kmeans",
    "weighted_kmeans",
    "hdbscan",
    "isolation_forest",
    "imf",
    "copod",
    "ecod",
    "hbos",
    "pca_reconstruction",
    "autoencoder",
)

#: How a distance to the assigned centroid becomes an anomaly score.
#:
#: ``raw_distance`` is comparable across clusters only if the clusters have
#: comparable spread, which they do not. The other three each correct for that
#: differently, and which correction is right is an empirical question the sweep
#: answers — hence a grid dimension rather than a decision.
KMEANS_SCORE_VARIANTS: Tuple[str, ...] = (
    "raw_distance",
    "cluster_z_distance",
    "cluster_percentile",
    "cluster_size_adjusted",
)

#: How weighted k-means derives a per-feature weight from a provisional
#: partition. ``inverse_dispersion`` rewards features that are tight within
#: clusters; ``variance_stability`` rewards features that vary overall;
#: ``hybrid`` wants both.
WEIGHT_STRATEGIES: Tuple[str, ...] = (
    "inverse_dispersion",
    "variance_stability",
    "hybrid",
)

#: Euler–Mascheroni constant, for the harmonic-number approximation in
#: :func:`average_path_length`.
EULER_MASCHERONI = 0.5772156649015329

#: Floor on a weighted k-means feature weight, as a multiple of the uniform
#: weight ``1/n_features``. A weight of zero deletes the feature, which makes a
#: 200-feature arm silently equivalent to a smaller one and breaks the
#: comparison the ``top_n_features`` dimension exists to make.
_MIN_RELATIVE_WEIGHT = 1e-3

#: Points in a per-cluster training-distance ladder. 512 is enough to resolve a
#: percentile to better than 0.2% and small enough that a 200-cluster model's
#: ladders stay under a megabyte in the bundle.
_LADDER_POINTS = 512

_EPS = 1e-12


class ModelError(ValueError):
    """Raised when a model is misconfigured or used out of order.

    A ``ValueError`` because every case is a caller mistake that the caller can
    correct: a ``k`` larger than the row count, a score variant that is not in
    :data:`KMEANS_SCORE_VARIANTS`, a ``score`` call before ``fit``.
    """


class ModelUnavailableError(ModelError):
    """Raised when a model needs an optional dependency that is not installed.

    Separate from :class:`ModelError` so the sweep driver can tell "this arm is
    impossible in this environment" from "this arm is wrong". The first is a
    reason to skip and record; the second is a reason to stop.
    """


# --------------------------------------------------------------------------
# matrix and score hygiene
# --------------------------------------------------------------------------

def _as_matrix(matrix: Any, context: str) -> np.ndarray:
    """Coerce ``matrix`` to a finite 2-D float64 array, or raise.

    Finiteness is checked rather than repaired. Every matrix reaching an adapter
    came from :meth:`trust_score_05.ml.preprocessing.Preprocessor.transform`,
    which already guarantees it, so a NaN here means a caller assembled a matrix
    by hand and skipped the preprocessor — in which case the imputation and
    scaling are also missing and the fitted model does not apply to the rows.
    Silently imputing at this depth would hide that.

    float64 rather than the preprocessor's float32 because the covariance in
    :class:`PCAReconstructionAdapter` and the per-cluster variances in
    :class:`WeightedKMeansAdapter` accumulate over the whole population, and
    float32 accumulation over a hundred thousand rows loses digits that matter
    to a ranking near the top.
    """
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ModelError(f"{context}: expected a 2-D matrix, got shape {array.shape}")
    if array.shape[0] == 0:
        raise ModelError(f"{context}: matrix has no rows")
    if array.shape[1] == 0:
        raise ModelError(f"{context}: matrix has no columns")
    if not np.isfinite(array).all():
        bad = int((~np.isfinite(array)).sum())
        raise ModelError(
            f"{context}: matrix has {bad} non-finite values; it did not come from a "
            "fitted Preprocessor"
        )
    return np.ascontiguousarray(array)


def _check_scores(scores: Any, n_rows: int, context: str) -> np.ndarray:
    """Coerce an adapter's raw output to a finite 1-D float64 array of length ``n_rows``.

    Raises rather than substituting, for the same reason as :func:`_as_matrix`:
    a non-finite score sorts unpredictably. ``numpy`` places NaN *last* in an
    ascending sort, so a NaN score becomes the most anomalous row in a
    descending ranking, and one degenerate cluster is then enough to fill the
    top of the review queue with rows the model had no opinion about. The sweep
    driver catches this and records the arm as failed.
    """
    array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if array.shape[0] != n_rows:
        raise ModelError(
            f"{context}: produced {array.shape[0]} scores for {n_rows} rows"
        )
    if not np.isfinite(array).all():
        bad = int((~np.isfinite(array)).sum())
        raise ModelError(f"{context}: produced {bad} non-finite scores")
    return array


def resolve_pca_components(value: Any, n_features: int, n_rows: int) -> Optional[Any]:
    """Interpret a ``pca_components`` grid value, or return ``None`` for "no PCA".

    ``None``, ``0``, and the strings ``"none"``/``"null"``/``"0"`` all disable
    PCA, following the repository's convention that zero disables a limit. A
    fraction strictly between 0 and 1 is a variance target and passes through. An
    integer is a component count and is **clamped** to ``min(n_features,
    n_rows)``.

    The clamp is a fix. The notebook built its component grid with
    ``min(100, n_features)`` where ``n_features`` was the *configured* top-N,
    not the number of features selection actually returned, so an arm asking
    for 100 components of a 74-feature matrix reached scikit-learn and raised
    mid-sweep. Clamping means such an arm becomes the full-rank arm rather than
    a crash, and because the resolved value is what goes into the arm's
    diagnostics, two arms that clamp to the same rank are visible as duplicates
    instead of being mistaken for independent evidence.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null", "0"}:
            return None
        try:
            value = float(text) if "." in text else int(text)
        except ValueError as exc:
            raise ModelError(f"pca_components is not a number: {value!r}") from exc
    if isinstance(value, bool):
        raise ModelError(f"pca_components must be a number or None, got {value!r}")
    limit = max(1, min(int(n_features), int(n_rows)))
    if isinstance(value, float):
        if value <= 0.0:
            return None
        if value < 1.0:
            return float(value)
        value = int(round(value))
    value = int(value)
    if value <= 0:
        return None
    return min(value, limit)


def average_path_length(n: Any) -> np.ndarray:
    """Expected path length of an unsuccessful search in a binary search tree.

    ``c(n) = 2 * H(n - 1) - 2 * (n - 1) / n``, with ``H`` the harmonic number
    approximated as ``ln(n) + gamma``. This is the normalising constant of the
    isolation-forest score family: it is how deep a tree built on ``n`` points
    would place an *average* point, so dividing an observed depth by it turns
    "shallow" into "shallow for a tree this size".

    Vectorised, and defined at the boundaries — ``c(0) = c(1) = 0`` and
    ``c(2) = 1`` — because a Mondrian leaf can hold a single point and the
    notebook's scalar version returned ``c(2)`` for those cases by clamping
    ``n`` to 2, which credited a singleton leaf with a full extra level of
    depth it had not earned.
    """
    array = np.asarray(n, dtype=np.float64)
    out = np.zeros_like(array)
    two = array == 2.0
    many = array > 2.0
    out[two] = 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        big = array[many]
        out[many] = 2.0 * (np.log(big - 1.0) + EULER_MASCHERONI) - 2.0 * (big - 1.0) / big
    return out


# --------------------------------------------------------------------------
# the interface
# --------------------------------------------------------------------------

class ModelAdapter:
    """One anomaly-detection model, fitted on non-fraud and scoring anything.

    Subclasses implement :meth:`_fit` and :meth:`_score` and may override
    :meth:`_diagnostics`, :meth:`training_scores` and :meth:`training_frames`.
    The public :meth:`fit` and :meth:`score` are templates that validate their
    inputs, so a subclass never has to, and every subclass therefore fails the
    same way on the same bad input.

    Attributes:
        name: The model identifier, one of :data:`MODEL_NAMES`. A class
            attribute, not a constructor argument, because it is the identity of
            the algorithm rather than a property of an instance — and because
            the notebook's version was a constructor-set string that one code
            path set wrongly.
        params: The hyperparameters as given. Retained verbatim, not normalised,
            so that the arm's identifier — computed from this dict by
            :func:`trust_score_05.ml.paths.hyperparam_id` — matches the grid
            entry the sweep driver iterated over. Resolved and clamped values go
            into :meth:`diagnostics` instead.
        seed: The random seed. Every adapter that draws is seeded from this and
            nothing else, so a rerun of one arm reproduces its scores exactly.
    """

    name: ClassVar[str] = ""
    #: Set on subclasses whose fit needs an optional package. Used only to give
    #: :class:`ModelUnavailableError` a useful message.
    requires: ClassVar[Tuple[str, ...]] = ()

    def __init__(self, params: Optional[Dict[str, Any]] = None, seed: int = 42) -> None:
        self.params: Dict[str, Any] = dict(params or {})
        self.seed = int(seed)
        self._fitted = False
        self._n_train_rows = 0
        self._n_train_features = 0
        self._implementation = ""

    # -- lifecycle ---------------------------------------------------------

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def implementation(self) -> str:
        """Which code actually produced the fit, e.g. ``sklearn.MiniBatchKMeans``.

        Recorded because two runs of the same ``model=`` prefix can be produced
        by different code — a PyOD detector or its numpy stand-in, a torch
        autoencoder or an ``MLPRegressor`` — and a leaderboard that cannot tell
        them apart is comparing an algorithm against its substitute.
        """
        return self._implementation

    def fit(self, matrix: Any) -> "ModelAdapter":
        """Fit on ``matrix``, a preprocessed non-fraud training matrix. Returns self."""
        array = _as_matrix(matrix, f"{self.name}.fit")
        self._n_train_rows, self._n_train_features = array.shape
        self._fit(array)
        self._fitted = True
        return self

    def score(self, matrix: Any) -> np.ndarray:
        """Anomaly scores for ``matrix``, higher meaning more anomalous.

        Raises if the adapter is unfitted, or if the matrix is not as wide as the
        one it was fitted on. The width check matters more than it looks: the
        preprocessing variants that add missing-indicator columns widen the
        matrix, and a caller that fitted with indicators and scores without them
        gets a shape error here rather than a silently truncated feature space.
        """
        self._require_fitted("score")
        array = _as_matrix(matrix, f"{self.name}.score")
        if array.shape[1] != self._n_train_features:
            raise ModelError(
                f"{self.name}.score: matrix has {array.shape[1]} columns but the model "
                f"was fitted on {self._n_train_features}"
            )
        return _check_scores(self._score(array), array.shape[0], f"{self.name}.score")

    def training_scores(self, matrix: Any) -> np.ndarray:
        """Scores for the *training* matrix, for the training-distribution plot.

        Distinct from :meth:`score` because two adapters must exclude a row's own
        contribution when the row is one they were fitted on:
        :class:`HDBSCANAdapter`, whose nearest neighbour would otherwise be
        itself at distance zero, and nothing else. The default is
        :meth:`score`.
        """
        return self.score(matrix)

    def _require_fitted(self, what: str) -> None:
        if not self._fitted:
            raise ModelError(f"{self.name}.{what} called before fit")

    # -- reporting ---------------------------------------------------------

    def diagnostics(self) -> Dict[str, Any]:
        """Flat, JSON-serialisable facts about the fit, for the arm's manifest.

        Flat and scalar-valued on purpose: these become columns of the metrics
        table, so that a question like "did the arms that scored well all end up
        with few clusters" is a query rather than a re-run.
        """
        payload: Dict[str, Any] = {
            "model": self.name,
            "implementation": self._implementation,
            "n_train_rows": self._n_train_rows,
            "n_train_features": self._n_train_features,
            "seed": self.seed,
        }
        if self._fitted:
            payload.update(self._diagnostics())
        return payload

    def training_frames(self) -> Dict[str, pd.DataFrame]:
        """Named tables the training report writes, e.g. cluster sizes.

        Returned as frames rather than written here so that this module has no
        opinion about where artifacts go and no dependency on the object store.
        """
        return {}

    def state(self) -> Dict[str, Any]:
        """Extra fitted state for :attr:`ModelBundle.extra`.

        Usually empty. The adapter itself is the bundle's ``model``, so its
        fitted state is already persisted; this is for the few values a consumer
        wants without unpickling the adapter, such as the learned feature
        weights.
        """
        return {}

    # -- hooks -------------------------------------------------------------

    def _fit(self, matrix: np.ndarray) -> None:
        raise NotImplementedError

    def _score(self, matrix: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _diagnostics(self) -> Dict[str, Any]:
        return {}

    # -- shared helpers ----------------------------------------------------

    def _param(self, key: str, default: Any) -> Any:
        value = self.params.get(key, default)
        return default if value is None else value

    def _int_param(self, key: str, default: int, minimum: int = 1) -> int:
        try:
            value = int(self._param(key, default))
        except (TypeError, ValueError) as exc:
            raise ModelError(
                f"{self.name}: {key} must be an integer, got {self.params.get(key)!r}"
            ) from exc
        return max(minimum, value)

    def _float_param(self, key: str, default: float) -> float:
        try:
            return float(self._param(key, default))
        except (TypeError, ValueError) as exc:
            raise ModelError(
                f"{self.name}: {key} must be a number, got {self.params.get(key)!r}"
            ) from exc

    def _require(self, module_name: str) -> Any:
        """Import an optional dependency or raise :class:`ModelUnavailableError`."""
        try:
            return __import__(module_name, fromlist=["_"])
        except Exception as exc:  # noqa: BLE001 - a broken install is as fatal as a missing one
            raise ModelUnavailableError(
                f"{self.name} needs {module_name!r}, which could not be imported: {exc}"
            ) from exc


# --------------------------------------------------------------------------
# optional PCA, shared by the clustering and reconstruction families
# --------------------------------------------------------------------------

class _OptionalPCAMixin:
    """A fitted PCA that several adapters optionally interpose before clustering.

    PCA before k-means is not cosmetic: Euclidean distance in a 200-column space
    where most columns are near-duplicates of each other is dominated by
    whichever feature family happens to have the most members, and projecting
    first removes that. It is a grid dimension because the right rank is not
    knowable in advance.

    The projection is fitted on the training matrix only, and the *fitted*
    object is reused at scoring time. The notebook's helper pair did this
    correctly, which is worth saying because almost nothing else in its scoring
    path did.
    """

    def _fit_pca(self, matrix: np.ndarray) -> np.ndarray:
        n_rows, n_features = matrix.shape
        resolved = resolve_pca_components(
            self.params.get("pca_components"), n_features, n_rows
        )
        self._pca_requested = self.params.get("pca_components")
        self._pca_resolved = resolved
        if resolved is None:
            self._pca = None
            return matrix
        from sklearn.decomposition import PCA

        pca = PCA(n_components=resolved, random_state=self.seed)
        projected = np.ascontiguousarray(pca.fit_transform(matrix), dtype=np.float64)
        self._pca = pca
        return projected

    def _apply_pca(self, matrix: np.ndarray) -> np.ndarray:
        pca = getattr(self, "_pca", None)
        if pca is None:
            return matrix
        return np.ascontiguousarray(pca.transform(matrix), dtype=np.float64)

    def _pca_diagnostics(self) -> Dict[str, Any]:
        pca = getattr(self, "_pca", None)
        return {
            "pca_components_requested": (
                "none" if self._pca_requested is None else str(self._pca_requested)
            ),
            "pca_components_resolved": (
                0 if pca is None else int(pca.n_components_)
            ),
            "pca_explained_variance": (
                1.0 if pca is None else float(pca.explained_variance_ratio_.sum())
            ),
        }


# --------------------------------------------------------------------------
# frozen per-cluster score statistics
# --------------------------------------------------------------------------

@dataclass
class _ClusterScoreStats:
    """Per-cluster training-distance statistics, frozen at fit time.

    This class exists because of one defect, repeated in three places. The
    notebook's three non-raw k-means score variants each derived their
    correction from the distances of the rows being scored:

    - ``cluster_z_distance`` used ``dist[idx].mean()`` and ``.std()`` of the
      scored frame, so the same account scored alone and scored inside its
      month's population got different z-scores;
    - ``cluster_percentile`` used ``argsort(argsort(dist[idx]))`` over the
      scored frame, so the top row of every cluster scored exactly 1.0 whatever
      its distance, and the *number* of fraud rows in a cluster determined how
      far down the ranking they landed;
    - ``cluster_size_adjusted`` combined the leaking z-score with
      ``log(len(dist) / idx.sum())``, a cluster-share of the *scored* frame
      rather than of the training population.

    The evaluation population is a non-fraud sample unioned with the whole fraud
    set, so its cluster shares are not the training shares and its within-cluster
    distance distributions are inflated by exactly the rows the model is supposed
    to be finding. Freezing the statistics on the training population makes a
    row's score a function of the row.

    Attributes:
        counts: Training rows assigned to each cluster, indexed by label.
        means, stds: Training within-cluster distance mean and standard
            deviation. ``stds`` is floored at :data:`_EPS` so a single-member
            cluster does not divide by zero.
        ladders: Per-cluster sorted training distances, subsampled to at most
            :data:`_LADDER_POINTS` quantiles, for the percentile variant.
        n_train_rows: Total training rows, for the size adjustment.
    """

    counts: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    ladders: List[np.ndarray]
    n_train_rows: int

    @classmethod
    def fit(
        cls, labels: np.ndarray, distances: np.ndarray, n_clusters: int
    ) -> "_ClusterScoreStats":
        counts = np.zeros(n_clusters, dtype=np.int64)
        means = np.zeros(n_clusters, dtype=np.float64)
        stds = np.full(n_clusters, _EPS, dtype=np.float64)
        ladders: List[np.ndarray] = []
        probabilities = np.linspace(0.0, 1.0, _LADDER_POINTS)
        for label in range(n_clusters):
            member = distances[labels == label]
            counts[label] = member.size
            if member.size == 0:
                # An empty cluster is possible: MiniBatchKMeans can leave a
                # centroid unassigned. Its ladder is a single zero so that a
                # scored row assigned to it lands at percentile 0 plus the tail
                # term, i.e. is ranked by raw distance, which is the only
                # defensible thing to do with no training evidence.
                ladders.append(np.zeros(1, dtype=np.float64))
                continue
            means[label] = float(member.mean())
            stds[label] = max(float(member.std()), _EPS)
            ladders.append(np.quantile(member, probabilities).astype(np.float64))
        return cls(
            counts=counts,
            means=means,
            stds=stds,
            ladders=ladders,
            n_train_rows=int(distances.size),
        )

    def transform(self, labels: np.ndarray, distances: np.ndarray, variant: str) -> np.ndarray:
        if variant == "raw_distance":
            return distances
        if variant == "cluster_z_distance":
            return (distances - self.means[labels]) / self.stds[labels]
        if variant == "cluster_percentile":
            return self._percentile(labels, distances)
        if variant == "cluster_size_adjusted":
            z = (distances - self.means[labels]) / self.stds[labels]
            share = np.maximum(self.counts[labels], 1).astype(np.float64)
            return z + np.log(max(self.n_train_rows, 1) / share)
        raise ModelError(
            f"unknown score_variant {variant!r}; expected one of {KMEANS_SCORE_VARIANTS}"
        )

    def _percentile(self, labels: np.ndarray, distances: np.ndarray) -> np.ndarray:
        """Where each distance falls in its cluster's *training* distance ladder.

        In ``[0, 1]`` for a distance within the training range, and above 1 for
        one beyond it, by a term proportional to how far beyond. The tail term is
        the second half of the fix: interpolation alone saturates at 1.0, so
        every evaluation row further from its centroid than any training row
        would tie at the top of the ranking with no ordering between them — and
        those are precisely the rows a fraud model is looking for. Extending
        past 1 keeps them ordered, and keeps clusters comparable because the
        extension is scaled by the cluster's own training range.
        """
        levels = np.linspace(0.0, 1.0, _LADDER_POINTS)
        out = np.zeros(distances.shape[0], dtype=np.float64)
        for label in np.unique(labels):
            index = labels == label
            ladder = self.ladders[int(label)]
            member = distances[index]
            if ladder.size < 2:
                out[index] = member / (float(ladder[-1]) + _EPS)
                continue
            interpolated = np.interp(member, ladder, levels[: ladder.size])
            top = float(ladder[-1])
            spread = max(top - float(ladder[0]), _EPS)
            out[index] = interpolated + np.maximum(member - top, 0.0) / spread
        return out

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cluster": np.arange(self.counts.size, dtype=np.int64),
                "n_train_rows": self.counts,
                "train_share": self.counts / max(self.n_train_rows, 1),
                "distance_mean": self.means,
                "distance_std": self.stds,
                "distance_p50": [
                    float(ladder[ladder.size // 2]) for ladder in self.ladders
                ],
                "distance_max": [float(ladder[-1]) for ladder in self.ladders],
            }
        )


# --------------------------------------------------------------------------
# k-means
# --------------------------------------------------------------------------

class KMeansAdapter(_OptionalPCAMixin, ModelAdapter):
    """Distance to the assigned centroid of a mini-batch k-means partition.

    Mini-batch rather than full Lloyd's iteration because the training
    population is six figures of rows by up to 200 columns and the sweep fits
    thousands of arms over it; the partition quality difference is immaterial
    next to the arm count it buys.

    ``k`` is validated against the row count rather than clamped. scikit-learn
    reduces an oversized ``n_clusters`` with a warning, which in a sweep means a
    warning nobody reads and two arms that report different ``k`` values while
    holding the same partition. The notebook went further and derived its ``k``
    grid from the *feature* count — ``range(2, n_features + 1)`` — so a
    25-feature arm could never try more than 25 clusters and a 200-feature arm
    tried 200 regardless of whether the population supported them. The number of
    groups in a population is not a function of how many columns describe it;
    see :func:`trust_score_05.ml.grids.k_values`.
    """

    name = "kmeans"

    def _space(self, matrix: np.ndarray) -> np.ndarray:
        """The space the partition lives in. Overridden by the weighted variant."""
        return matrix

    def _fit(self, matrix: np.ndarray) -> None:
        from sklearn.cluster import MiniBatchKMeans

        self._variant = str(self._param("score_variant", "raw_distance"))
        if self._variant not in KMEANS_SCORE_VARIANTS:
            raise ModelError(
                f"{self.name}: score_variant {self._variant!r} is not one of "
                f"{KMEANS_SCORE_VARIANTS}"
            )
        projected = self._fit_pca(matrix)
        self._k = self._int_param("k", 8, minimum=2)
        if self._k > projected.shape[0]:
            raise ModelError(
                f"{self.name}: k={self._k} exceeds the {projected.shape[0]} training rows"
            )
        space = self._fit_space(projected)
        self._model = MiniBatchKMeans(
            n_clusters=self._k,
            batch_size=self._int_param("batch_size", 8192, minimum=32),
            max_iter=self._int_param("max_iter", 700, minimum=1),
            reassignment_ratio=self._float_param("reassignment_ratio", 0.01),
            n_init="auto",
            random_state=self.seed,
        )
        self._model.fit(space)
        self._implementation = "sklearn.MiniBatchKMeans"
        labels, distances = self._assign(space)
        self._stats = _ClusterScoreStats.fit(labels, distances, self._k)
        self._inertia = float(self._model.inertia_)

    def _fit_space(self, projected: np.ndarray) -> np.ndarray:
        """Hook for a subclass that must learn something before clustering."""
        return self._space(projected)

    def _assign(self, space: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        labels = self._model.predict(space)
        centroids = self._model.cluster_centers_[labels]
        return labels, np.linalg.norm(space - centroids, axis=1)

    def _score(self, matrix: np.ndarray) -> np.ndarray:
        space = self._space(self._apply_pca(matrix))
        labels, distances = self._assign(space)
        return self._stats.transform(labels, distances, self._variant)

    def cluster_assignments(self, matrix: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster label and distance to that cluster, per row of ``matrix``.

        The presence of this method is what
        :func:`trust_score_05.ml.drift.drift_report` uses to decide whether
        cluster drift applies, so the capability is a property of the object
        rather than of a hardcoded list of model names kept in the drift step.

        It goes through the same :meth:`_apply_pca` and :meth:`_space` as
        :meth:`_score`, which is the whole point of routing it through the
        adapter: the notebook's free function ``_v31_cluster_assignments``
        rebuilt the weighted space from a ``weights`` key it read out of the
        model dict, so the drift table for a weighted arm was computed in
        whichever space that key described rather than in the space the arm
        scored in — the same fit/score space mismatch this class exists to
        prevent, reintroduced one module later.

        Returns:
            ``(labels, distances)``, both of length ``len(matrix)``.
        """
        self._require_fitted("cluster_assignments")
        array = _as_matrix(matrix, f"{self.name}.cluster_assignments")
        if array.shape[1] != self._n_train_features:
            raise ModelError(
                f"{self.name}.cluster_assignments: matrix has {array.shape[1]} columns "
                f"but the model was fitted on {self._n_train_features}"
            )
        labels, distances = self._assign(self._space(self._apply_pca(array)))
        return labels.astype(np.int64), distances.astype(np.float64)

    def _diagnostics(self) -> Dict[str, Any]:
        occupied = int((self._stats.counts > 0).sum())
        payload: Dict[str, Any] = {
            "k": self._k,
            "score_variant": self._variant,
            "inertia": self._inertia,
            "n_clusters_occupied": occupied,
            "n_clusters_empty": self._k - occupied,
            "smallest_cluster_rows": int(self._stats.counts.min()),
            "largest_cluster_share": float(
                self._stats.counts.max() / max(self._stats.n_train_rows, 1)
            ),
        }
        payload.update(self._pca_diagnostics())
        return payload

    def training_frames(self) -> Dict[str, pd.DataFrame]:
        self._require_fitted("training_frames")
        return {"cluster_score_statistics": self._stats.as_frame()}


class WeightedKMeansAdapter(KMeansAdapter):
    """k-means in a space stretched by iteratively learned per-feature weights.

    The idea: cluster once with uniform weights, measure how tightly each
    feature holds within the resulting clusters, up-weight the features that
    hold tightly, and repeat. A feature that is near-constant inside every
    cluster is carrying the partition; one that is as spread inside a cluster as
    across the population is noise as far as this partition is concerned.

    Two defects fixed.

    The first is the one described in the module docstring: the notebook's fit
    returned a dict tagged ``"kind": "weighted_kmeans"`` in the primary path, but
    the ``save_model_diagnostics`` path and one of the three redefinitions of
    ``fit_model`` produced centroids in the weighted space while the scorer's
    ``kind`` dispatch sent them through the unweighted branch. Here the space is
    a method — :meth:`_space` — so fitting and scoring cannot use different
    ones.

    The second is the ``hybrid`` strategy. It multiplied inverse within-cluster
    dispersion by total variance without normalising either factor first. Those
    two quantities have different units and wildly different dynamic ranges
    across a mixed feature set — a count feature's variance can exceed a
    ratio feature's by ten orders of magnitude — so the product was dominated by
    whichever factor happened to have the larger spread, and after the
    ``raw / raw.sum()`` normalisation a single feature routinely took upward of
    99% of the weight. That arm was then a one-dimensional model wearing a
    200-feature label. :meth:`_weights` normalises each factor to a probability
    vector before combining them geometrically, which makes the combination
    scale-free in each factor, and floors every weight at
    :data:`_MIN_RELATIVE_WEIGHT` of uniform so no feature is deleted outright.
    """

    name = "weighted_kmeans"

    def _space(self, matrix: np.ndarray) -> np.ndarray:
        return matrix * np.sqrt(self._weights)

    def _fit_space(self, projected: np.ndarray) -> np.ndarray:
        from sklearn.cluster import MiniBatchKMeans

        strategy = str(self._param("weight_strategy", "inverse_dispersion"))
        if strategy not in WEIGHT_STRATEGIES:
            raise ModelError(
                f"{self.name}: weight_strategy {strategy!r} is not one of {WEIGHT_STRATEGIES}"
            )
        self._strategy = strategy
        iterations = self._int_param("weight_update_iterations", 1, minimum=1)
        n_features = projected.shape[1]
        self._weights = np.full(n_features, 1.0 / n_features, dtype=np.float64)
        total_variance = np.nan_to_num(projected.var(axis=0), nan=0.0, posinf=0.0)
        self._weight_history: List[Dict[str, Any]] = []
        k = self._int_param("k", 8, minimum=2)
        batch = self._int_param("batch_size", 8192, minimum=32)
        for iteration in range(iterations):
            provisional = MiniBatchKMeans(
                n_clusters=k,
                batch_size=batch,
                max_iter=self._int_param("max_iter", 500, minimum=1),
                n_init="auto",
                # Offset the seed per iteration so successive provisional
                # partitions are not the identical local optimum, which would
                # make every update after the first a no-op.
                random_state=self.seed + iteration,
            ).fit(self._space(projected))
            labels = provisional.predict(self._space(projected))
            self._weights = self._weights_from(projected, labels, total_variance)
            self._weight_history.append(
                {
                    "iteration": iteration + 1,
                    "weight_entropy": float(
                        -(self._weights * np.log(self._weights + _EPS)).sum()
                    ),
                    "max_weight": float(self._weights.max()),
                    "min_weight": float(self._weights.min()),
                    "effective_features": float(
                        np.exp(-(self._weights * np.log(self._weights + _EPS)).sum())
                    ),
                }
            )
        return self._space(projected)

    def _weights_from(
        self, projected: np.ndarray, labels: np.ndarray, total_variance: np.ndarray
    ) -> np.ndarray:
        """Per-feature weights from a provisional partition, as a probability vector."""
        within = self._within_cluster_variance(projected, labels, total_variance)
        inverse_dispersion = 1.0 / (within + 1e-6)
        if self._strategy == "inverse_dispersion":
            raw = inverse_dispersion
        elif self._strategy == "variance_stability":
            raw = total_variance
        else:
            # Geometric mean of the two factors *after* each has been made a
            # probability vector, so neither factor's units decide the outcome.
            raw = np.sqrt(_normalise(inverse_dispersion) * _normalise(total_variance))
        weights = _normalise(raw)
        floor = _MIN_RELATIVE_WEIGHT / weights.size
        return _normalise(np.maximum(weights, floor))

    @staticmethod
    def _within_cluster_variance(
        projected: np.ndarray, labels: np.ndarray, total_variance: np.ndarray
    ) -> np.ndarray:
        """Mean within-cluster variance per feature, vectorised over clusters.

        The notebook computed this with a Python loop over ``(cluster, feature)``
        pairs, which for 200 clusters and 200 features is 40,000 boolean masks
        over the whole training matrix per weight-update iteration — the single
        slowest thing in its training path. This is two ``np.add.at``
        accumulations regardless of ``k``.

        Clusters with fewer than two members are excluded from the mean, and a
        feature with no qualifying cluster falls back to its total variance,
        which is what "no within-cluster evidence" should mean.
        """
        n_clusters = int(labels.max()) + 1
        counts = np.bincount(labels, minlength=n_clusters).astype(np.float64)
        sums = np.zeros((n_clusters, projected.shape[1]), dtype=np.float64)
        squares = np.zeros_like(sums)
        np.add.at(sums, labels, projected)
        np.add.at(squares, labels, projected * projected)
        usable = counts >= 2
        safe = np.where(counts > 0, counts, 1.0)[:, None]
        variance = np.maximum(squares / safe - (sums / safe) ** 2, 0.0)
        if not usable.any():
            return total_variance
        mean_within = variance[usable].mean(axis=0)
        # A feature that is exactly constant inside every cluster would give an
        # unbounded inverse dispersion and take the whole weight budget. Falling
        # back to its total variance keeps it merely dominant rather than
        # infinite; the weight floor and the probability normalisation handle
        # the rest.
        return np.where(mean_within > 0.0, mean_within, total_variance)

    def _diagnostics(self) -> Dict[str, Any]:
        payload = super()._diagnostics()
        last = self._weight_history[-1] if self._weight_history else {}
        payload.update(
            {
                "weight_strategy": self._strategy,
                "weight_update_iterations": len(self._weight_history),
                "weight_entropy": last.get("weight_entropy", float("nan")),
                "max_weight": last.get("max_weight", float("nan")),
                "effective_features": last.get("effective_features", float("nan")),
            }
        )
        return payload

    def training_frames(self) -> Dict[str, pd.DataFrame]:
        frames = super().training_frames()
        pca = getattr(self, "_pca", None)
        names = (
            [f"pc_{i + 1}" for i in range(self._weights.size)]
            if pca is not None
            else [f"feature_{i}" for i in range(self._weights.size)]
        )
        frames["learned_feature_weights"] = pd.DataFrame(
            {"position": np.arange(self._weights.size), "name": names, "weight": self._weights}
        ).sort_values("weight", ascending=False, ignore_index=True)
        frames["weight_update_history"] = pd.DataFrame(self._weight_history)
        return frames

    def state(self) -> Dict[str, Any]:
        return {"feature_weights": self._weights.tolist()}


def _normalise(values: np.ndarray) -> np.ndarray:
    """Turn a non-negative vector into a probability vector, tolerating a zero sum."""
    clipped = np.maximum(np.nan_to_num(values, nan=0.0, posinf=0.0), 0.0)
    total = float(clipped.sum())
    if total <= 0.0:
        return np.full(clipped.size, 1.0 / max(clipped.size, 1), dtype=np.float64)
    return clipped / total


# --------------------------------------------------------------------------
# HDBSCAN
# --------------------------------------------------------------------------

class HDBSCANAdapter(ModelAdapter):
    """Distance to the k-th nearest training row, with HDBSCAN as a diagnostic.

    HDBSCAN is a clustering algorithm, not an anomaly detector, and it does not
    have an ``predict``. It labels the rows it was fitted on — including a noise
    label of ``-1`` — and its GLOSH ``outlier_scores_`` are defined for those
    rows only. There is a ``approximate_predict`` for new points, but it needs
    ``prediction_data=True`` at fit time, costs a copy of the condensed tree per
    arm, and returns a cluster label rather than an outlier score. So the thing
    that actually scores a new row here is the distance to its k-th nearest
    training neighbour, which is the quantity HDBSCAN's own mutual-reachability
    core distance is built from, and the clusterer is fitted for its
    diagnostics: how many clusters the non-fraud population falls into, and what
    fraction of it HDBSCAN considers noise. Those two numbers are the reason to
    run this arm at all — a population that HDBSCAN says is 60% noise is telling
    you something about the feature space that no metric will.

    The notebook's version returned ``clusterer.outlier_scores_`` whenever
    ``len(outlier_scores_) == len(X)``. That condition is a coincidence, not a
    check: it is true when scoring the training matrix, and true again for any
    evaluation frame that happens to have the same row count — which the
    training-score plot and the capped evaluation frames made likelier than it
    sounds. When it fired on an evaluation frame it returned *training* scores,
    positionally aligned to evaluation rows. The metrics from those arms are
    noise. Here the two are separate methods: :meth:`score` is always the kNN
    distance, and the GLOSH scores appear only in
    :meth:`training_frames`, labelled as training-row diagnostics.

    ``min_samples`` doubles as the ``k`` of the kNN score, defaulting to
    ``min(50, max(5, sqrt(n_train)))`` as in the notebook.
    """

    name = "hdbscan"
    requires = ("hdbscan",)

    def _fit(self, matrix: np.ndarray) -> None:
        from sklearn.neighbors import NearestNeighbors

        self._metric = str(self._param("metric", "euclidean"))
        configured = self.params.get("min_samples")
        default_k = min(50, max(5, int(math.sqrt(matrix.shape[0]))))
        self._k = int(configured) if configured else default_k
        # +1 so training_scores can drop the self-match without a second query.
        self._k = max(1, min(self._k, matrix.shape[0] - 1))
        self._neighbours = NearestNeighbors(
            n_neighbors=self._k + 1, metric=self._metric, n_jobs=-1
        ).fit(matrix)
        self._implementation = "sklearn.NearestNeighbors"
        self._cluster_diagnostics: Dict[str, Any] = {"hdbscan_fitted": False}
        self._labels: Optional[np.ndarray] = None
        self._glosh: Optional[np.ndarray] = None
        self._fit_clusterer(matrix)

    def _fit_clusterer(self, matrix: np.ndarray) -> None:
        """Fit the clusterer for diagnostics. A failure here is recorded, not raised.

        Downgraded to a warning — unlike the PyOD family's missing dependency,
        which raises — because the clusterer does not participate in scoring.
        The arm is still exactly the kNN-distance model it would have been, and
        ``hdbscan_fitted: False`` in the diagnostics says so. Raising would make
        the availability of an optional package decide whether a scoring model
        that does not use it can run.
        """
        allow = bool(self._param("require_hdbscan", False))
        try:
            import hdbscan as hdbscan_module
        except Exception as exc:  # noqa: BLE001 - optional, and only a diagnostic
            if allow:
                raise ModelUnavailableError(
                    f"{self.name}: require_hdbscan is set but hdbscan is unimportable: {exc}"
                ) from exc
            self._cluster_diagnostics["hdbscan_error"] = f"import failed: {exc}"
            LOGGER.warning("hdbscan unavailable; scoring by kNN distance only: %s", exc)
            return
        try:
            clusterer = hdbscan_module.HDBSCAN(
                min_cluster_size=self._int_param("min_cluster_size", 500, minimum=2),
                min_samples=self.params.get("min_samples") or None,
                metric=self._metric,
                cluster_selection_method=str(
                    self._param("cluster_selection_method", "eom")
                ),
                prediction_data=False,
            )
            clusterer.fit(matrix)
        except Exception as exc:  # noqa: BLE001 - a degenerate arm, not a broken run
            if allow:
                raise ModelError(f"{self.name}: HDBSCAN fit failed: {exc}") from exc
            self._cluster_diagnostics["hdbscan_error"] = f"fit failed: {exc}"
            LOGGER.warning("HDBSCAN fit failed; scoring by kNN distance only: %s", exc)
            return
        labels = np.asarray(clusterer.labels_)
        glosh = np.asarray(getattr(clusterer, "outlier_scores_", []), dtype=np.float64)
        self._labels = labels
        self._glosh = glosh if glosh.size == labels.size else None
        self._fit_centroids(matrix, labels)
        self._cluster_diagnostics = {
            "hdbscan_fitted": True,
            "n_clusters": int(len(set(labels.tolist())) - (1 if -1 in labels else 0)),
            "noise_fraction": float(np.mean(labels == -1)),
            "largest_cluster_share": float(
                np.bincount(labels[labels >= 0]).max() / max(labels.size, 1)
            )
            if (labels >= 0).any()
            else 0.0,
        }

    def _fit_centroids(self, matrix: np.ndarray, labels: np.ndarray) -> None:
        """The mean position of each labelled group, noise included.

        Noise gets a centroid like any other label. It is a diffuse cloud rather
        than a mode, so the *distance* to it is a weaker statistic than for a
        real cluster — but it is still the mean radius of the noise cloud, and a
        noise cloud that has doubled its radius is worth a flag. Dropping the
        label instead, as the notebook did when it built its centroid list with
        ``if lab == -1: continue``, discards the one number HDBSCAN produces that
        is unambiguously about the population rather than the model.
        """
        present = np.unique(labels)
        self._centroid_labels = present.astype(np.int64)
        self._centroids = np.vstack(
            [np.nanmean(matrix[labels == label], axis=0) for label in present]
        )

    def _score(self, matrix: np.ndarray) -> np.ndarray:
        distances, _ = self._neighbours.kneighbors(matrix, n_neighbors=self._k)
        return distances[:, -1]

    def cluster_assignments(self, matrix: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster label by nearest-training-row transfer, and distance to its centroid.

        HDBSCAN has no ``predict``: its labels are defined for the rows it was
        fitted on. To place an evaluation row in the partition, the label of its
        nearest *training* row is transferred to it. The notebook instead
        assigned every row to its nearest cluster *centroid*, which is a Voronoi
        rule — and a Voronoi rule is exactly the assumption HDBSCAN exists to
        avoid. Its clusters are density-connected and can be crescents, shells
        or filaments, whose centroids may sit in empty space closer to a
        different cluster's members than to their own. Nearest-neighbour transfer
        respects whatever shape the clusters actually have, and costs nothing
        because the neighbour index is already fitted for scoring. It also
        transfers the noise label honestly: a row whose nearest training
        neighbour is noise is itself in a sparse region.

        The reported distance is to the assigned label's centroid rather than to
        that nearest neighbour, because a training row's nearest neighbour is
        itself at distance zero, which would make every training distance zero
        and every drift ratio unevaluable.

        Raises:
            ModelError: If the clusterer did not fit, so there is no partition to
                assign into. :func:`trust_score_05.ml.drift.drift_report` catches
                this and records the reason rather than failing the run — the
                scoring model is unaffected, since it never used the clusterer.
        """
        self._require_fitted("cluster_assignments")
        if self._labels is None:
            raise ModelError(
                f"{self.name}.cluster_assignments: no partition is available because the "
                f"clusterer did not fit "
                f"({self._cluster_diagnostics.get('hdbscan_error', 'unknown reason')})"
            )
        array = _as_matrix(matrix, f"{self.name}.cluster_assignments")
        if array.shape[1] != self._n_train_features:
            raise ModelError(
                f"{self.name}.cluster_assignments: matrix has {array.shape[1]} columns "
                f"but the model was fitted on {self._n_train_features}"
            )
        _, indices = self._neighbours.kneighbors(array, n_neighbors=1)
        labels = self._labels[indices[:, 0]].astype(np.int64)
        positions = np.searchsorted(self._centroid_labels, labels)
        distances = np.linalg.norm(array - self._centroids[positions], axis=1)
        return labels, distances.astype(np.float64)

    def training_scores(self, matrix: Any) -> np.ndarray:
        """kNN distance with the self-match removed.

        Every training row is its own nearest neighbour at distance zero, so
        asking for ``k`` neighbours of a training row returns the ``k-1``-th
        true neighbour. Left uncorrected, the training score distribution is
        shifted low relative to the evaluation distribution, and the
        training-score histogram — which exists to show whether the evaluation
        scores fall where the model expected — compares two different
        statistics.
        """
        self._require_fitted("training_scores")
        array = _as_matrix(matrix, f"{self.name}.training_scores")
        distances, _ = self._neighbours.kneighbors(array, n_neighbors=self._k + 1)
        return _check_scores(
            distances[:, -1], array.shape[0], f"{self.name}.training_scores"
        )

    def _diagnostics(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "knn_k": self._k,
            "metric": self._metric,
            "min_cluster_size": self._int_param("min_cluster_size", 500, minimum=2),
            "cluster_selection_method": str(self._param("cluster_selection_method", "eom")),
        }
        payload.update(self._cluster_diagnostics)
        return payload

    def training_frames(self) -> Dict[str, pd.DataFrame]:
        self._require_fitted("training_frames")
        frames: Dict[str, pd.DataFrame] = {}
        if self._labels is not None:
            counts = np.bincount(self._labels[self._labels >= 0])
            frames["hdbscan_cluster_sizes"] = pd.DataFrame(
                {
                    "cluster": np.arange(counts.size, dtype=np.int64),
                    "n_train_rows": counts,
                }
            )
        if self._glosh is not None:
            frames["hdbscan_training_glosh_scores"] = pd.DataFrame(
                {"glosh_score": self._glosh}
            ).describe().reset_index(names="statistic")
        return frames


# --------------------------------------------------------------------------
# isolation forest
# --------------------------------------------------------------------------

class IsolationForestAdapter(ModelAdapter):
    """scikit-learn's isolation forest, sign-flipped to the shared orientation.

    ``score_samples`` returns the *negated* anomaly score — low values are
    anomalous — so the adapter negates it once. The notebook did this correctly;
    it is spelled out here because it is the single most common way an anomaly
    pipeline inverts its own ranking, and the negation is easy to lose in a
    refactor.

    ``contamination`` is accepted and passed through, but it does not affect the
    ranking: it sets ``offset_``, which only matters to ``predict``. The grid in
    :mod:`trust_score_05.ml.grids` therefore does not sweep it — see there for
    why that removes several thousand duplicate arms.
    """

    name = "isolation_forest"

    def _fit(self, matrix: np.ndarray) -> None:
        from sklearn.ensemble import IsolationForest

        max_samples = self._param("max_samples", "auto")
        if isinstance(max_samples, str) and max_samples != "auto":
            raise ModelError(
                f"{self.name}: max_samples must be a number or 'auto', got {max_samples!r}"
            )
        if not isinstance(max_samples, str):
            # An integer larger than the row count makes scikit-learn warn and
            # clamp. Clamping here instead keeps the diagnostics honest about
            # what the trees actually saw, which is what the score normalisation
            # depends on.
            max_samples = min(int(max_samples), matrix.shape[0])
        self._resolved_max_samples = max_samples
        self._model = IsolationForest(
            n_estimators=self._int_param("n_estimators", 500, minimum=1),
            max_samples=max_samples,
            contamination=self._param("contamination", "auto"),
            max_features=self._float_param("max_features", 1.0),
            bootstrap=bool(self._param("bootstrap", False)),
            random_state=self.seed,
            n_jobs=-1,
        )
        self._model.fit(matrix)
        self._implementation = "sklearn.IsolationForest"

    def _score(self, matrix: np.ndarray) -> np.ndarray:
        return -np.asarray(self._model.score_samples(matrix), dtype=np.float64)

    def _diagnostics(self) -> Dict[str, Any]:
        return {
            "n_estimators": int(self._model.n_estimators),
            "max_samples_resolved": int(self._model.max_samples_),
            "max_features": float(self._model.max_features),
            "bootstrap": bool(self._model.bootstrap),
            "offset": float(getattr(self._model, "offset_", float("nan"))),
        }


# --------------------------------------------------------------------------
# iMondrian forest
# --------------------------------------------------------------------------

class _MondrianNode:
    """One node of a Mondrian tree, during construction and online extension.

    ``__slots__`` because a forest of 100 trees over a 4096-row subsample holds
    on the order of a million of these, and a per-node ``__dict__`` is the
    difference between a bundle that fits in memory and one that does not.

    ``birth_time`` is the Mondrian process's split time. It is what makes online
    extension well defined: a new point that falls outside a node's bounding box
    can be split off *above* that node if the exponential clock for the box's
    expansion rings before the node's own split time, and the resulting tree is
    distributed identically to one built in batch from all the points seen so
    far. That is the whole reason to prefer this over an isolation forest for a
    model kept current between retrains, and it is why the times are stored
    rather than recomputed.
    """

    __slots__ = (
        "is_leaf",
        "split_dim",
        "split_val",
        "birth_time",
        "lower",
        "upper",
        "population",
        "left",
        "right",
    )

    def __init__(
        self,
        is_leaf: bool,
        lower: np.ndarray,
        upper: np.ndarray,
        population: int,
        split_dim: int = -1,
        split_val: float = 0.0,
        birth_time: float = float("inf"),
        left: Optional["_MondrianNode"] = None,
        right: Optional["_MondrianNode"] = None,
    ) -> None:
        self.is_leaf = bool(is_leaf)
        self.lower = np.asarray(lower, dtype=np.float32)
        self.upper = np.asarray(upper, dtype=np.float32)
        self.population = int(population)
        self.split_dim = int(split_dim)
        self.split_val = float(split_val)
        self.birth_time = float(birth_time)
        self.left = left
        self.right = right


class _MondrianTree:
    """A single Mondrian tree: batch construction plus online extension.

    Scoring is vectorised. The tree is built and extended as linked
    :class:`_MondrianNode` objects, then flattened into parallel arrays the
    first time a score is asked for, and descended with one boolean pass per
    level. The notebook descended recursively, one Python call per node per
    row: a hundred trees by a hundred thousand evaluation rows by a depth of
    twenty is two hundred million calls, and it was the reason the ``imf`` arms
    were the slowest in the sweep by an order of magnitude.

    The flattened form is invalidated by :meth:`partial_fit`, so an online
    update cannot be scored against a stale layout.
    """

    def __init__(self, max_depth: int = 64, min_leaf_size: int = 1, seed: int = 42) -> None:
        self.max_depth = int(max_depth)
        self.min_leaf_size = max(1, int(min_leaf_size))
        self.seed = int(seed)
        self.root: Optional[_MondrianNode] = None
        #: Rows this tree was actually built or extended on. The normalising
        #: constant is derived from this, per tree.
        self.n_fitted = 0
        self._flat: Optional[Tuple[np.ndarray, ...]] = None

    # -- construction ------------------------------------------------------

    def fit(self, matrix: np.ndarray) -> "_MondrianTree":
        rng = np.random.default_rng(self.seed)
        self.root = self._build(np.asarray(matrix, dtype=np.float32), 0.0, 0, rng)
        self.n_fitted = int(matrix.shape[0])
        self._flat = None
        return self

    def _build(
        self, block: np.ndarray, parent_time: float, depth: int, rng: np.random.Generator
    ) -> _MondrianNode:
        n_rows = block.shape[0]
        lower = block.min(axis=0)
        upper = block.max(axis=0)
        widths = np.maximum(upper - lower, 0.0).astype(np.float64)
        rate = float(widths.sum())
        if (
            n_rows <= self.min_leaf_size
            or depth >= self.max_depth
            or rate <= 0.0
            or not np.isfinite(rate)
        ):
            return _MondrianNode(True, lower, upper, n_rows)
        split_time = parent_time + float(rng.exponential(1.0 / rate))
        dimension = int(rng.choice(widths.size, p=widths / rate))
        low = float(lower[dimension])
        high = float(upper[dimension])
        if not (high > low):
            return _MondrianNode(True, lower, upper, n_rows)
        threshold = float(rng.uniform(low, high))
        mask = block[:, dimension] < threshold
        taken = int(mask.sum())
        if taken == 0 or taken == n_rows:
            # A uniform draw inside the box can still fall outside every
            # observed value when the dimension is heavily tied. Splitting at
            # the median keeps the node productive instead of degenerating into
            # a leaf, which is what would otherwise happen to every
            # near-constant dimension in a sparse count feature.
            order = np.argsort(block[:, dimension], kind="stable")
            cut = n_rows // 2
            mask = np.zeros(n_rows, dtype=bool)
            mask[order[:cut]] = True
            threshold = float(
                (block[order[cut - 1], dimension] + block[order[cut], dimension]) / 2.0
            )
        return _MondrianNode(
            False,
            lower,
            upper,
            n_rows,
            split_dim=dimension,
            split_val=threshold,
            birth_time=split_time,
            left=self._build(block[mask], split_time, depth + 1, rng),
            right=self._build(block[~mask], split_time, depth + 1, rng),
        )

    # -- online extension --------------------------------------------------

    def partial_fit(self, matrix: np.ndarray) -> "_MondrianTree":
        block = np.asarray(matrix, dtype=np.float32)
        if self.root is None:
            return self.fit(block)
        rng = np.random.default_rng(self.seed + self.n_fitted)
        for row in block:
            self.root = self._extend(self.root, row, 0.0, rng)
        self.n_fitted += int(block.shape[0])
        self._flat = None
        return self

    def _extend(
        self,
        node: _MondrianNode,
        row: np.ndarray,
        parent_time: float,
        rng: np.random.Generator,
    ) -> _MondrianNode:
        below = np.maximum(node.lower - row, 0.0).astype(np.float64)
        above = np.maximum(row - node.upper, 0.0).astype(np.float64)
        widths = below + above
        rate = float(widths.sum())
        if rate > 0.0:
            elapsed = float(rng.exponential(1.0 / rate))
            if parent_time + elapsed < node.birth_time:
                dimension = int(rng.choice(widths.size, p=widths / rate))
                if row[dimension] > node.upper[dimension]:
                    low, high = float(node.upper[dimension]), float(row[dimension])
                else:
                    low, high = float(row[dimension]), float(node.lower[dimension])
                threshold = float(rng.uniform(low, high)) if high > low else float(row[dimension])
                leaf = _MondrianNode(True, row.copy(), row.copy(), 1)
                left, right = (
                    (node, leaf) if row[dimension] > threshold else (leaf, node)
                )
                return _MondrianNode(
                    False,
                    np.minimum(node.lower, row),
                    np.maximum(node.upper, row),
                    node.population + 1,
                    split_dim=dimension,
                    split_val=threshold,
                    birth_time=parent_time + elapsed,
                    left=left,
                    right=right,
                )
        node.lower = np.minimum(node.lower, row)
        node.upper = np.maximum(node.upper, row)
        node.population += 1
        if node.is_leaf:
            return node
        if row[node.split_dim] < node.split_val:
            node.left = self._extend(node.left, row, node.birth_time, rng)
        else:
            node.right = self._extend(node.right, row, node.birth_time, rng)
        return node

    # -- scoring -----------------------------------------------------------

    def _flatten(self) -> Tuple[np.ndarray, ...]:
        if self._flat is not None:
            return self._flat
        if self.root is None:
            raise ModelError("Mondrian tree is not fitted")
        split_dim: List[int] = []
        split_val: List[float] = []
        left: List[int] = []
        right: List[int] = []
        leaf_length: List[float] = []
        stack: List[Tuple[_MondrianNode, int]] = [(self.root, 0)]
        indices: Dict[int, int] = {id(self.root): 0}
        split_dim.append(-1)
        split_val.append(0.0)
        left.append(-1)
        right.append(-1)
        leaf_length.append(0.0)
        while stack:
            node, depth = stack.pop()
            position = indices[id(node)]
            if node.is_leaf or node.left is None or node.right is None:
                # The path length of a row landing in a leaf is the depth it
                # descended *plus* the expected further depth of the points
                # still bundled together there. The notebook returned the raw
                # depth, so a leaf holding two hundred indistinguishable rows
                # scored them as isolated as a singleton at the same depth --
                # which inverts the entire premise of the score, since a large
                # leaf is evidence of density, not of isolation.
                leaf_length[position] = float(depth) + float(
                    average_path_length(node.population)
                )
                continue
            split_dim[position] = node.split_dim
            split_val[position] = node.split_val
            for child, slot in ((node.left, "left"), (node.right, "right")):
                child_position = len(split_dim)
                indices[id(child)] = child_position
                split_dim.append(-1)
                split_val.append(0.0)
                left.append(-1)
                right.append(-1)
                leaf_length.append(0.0)
                if slot == "left":
                    left[position] = child_position
                else:
                    right[position] = child_position
                stack.append((child, depth + 1))
        self._flat = (
            np.asarray(split_dim, dtype=np.int64),
            np.asarray(split_val, dtype=np.float64),
            np.asarray(left, dtype=np.int64),
            np.asarray(right, dtype=np.int64),
            np.asarray(leaf_length, dtype=np.float64),
        )
        return self._flat

    def path_lengths(self, matrix: np.ndarray) -> np.ndarray:
        """Path length of every row, one vectorised pass per tree level."""
        split_dim, split_val, left, right, leaf_length = self._flatten()
        n_rows = matrix.shape[0]
        node = np.zeros(n_rows, dtype=np.int64)
        out = np.zeros(n_rows, dtype=np.float64)
        active = np.arange(n_rows, dtype=np.int64)
        while active.size:
            current = node[active]
            is_leaf = split_dim[current] < 0
            if is_leaf.any():
                landed = active[is_leaf]
                out[landed] = leaf_length[node[landed]]
                active = active[~is_leaf]
                if active.size == 0:
                    break
                current = node[active]
            go_left = matrix[active, split_dim[current]] < split_val[current]
            node[active] = np.where(go_left, left[current], right[current])
        return out


class IMondrianForest:
    """An ensemble of Mondrian trees scoring by isolation depth.

    Standalone rather than nested inside the adapter because it is a genuine
    estimator with a ``fit``/``partial_fit``/``score_samples`` surface, and the
    thing that distinguishes it from an isolation forest — that ``partial_fit``
    is exact rather than an approximation — is worth being able to exercise
    without a sweep around it.

    Attributes:
        subsample_size: Rows each tree is built on. 0 means the whole matrix.
            Subsampling is what makes isolation-style scores work at all: a tree
            built on the full population isolates nothing, because every point
            eventually gets its own leaf at similar depth.
        training_history: One record per batch fit and per online update, so the
            score drift an online update causes is attributable to the update
            that caused it.
    """

    def __init__(
        self,
        n_trees: int = 100,
        subsample_size: int = 4096,
        max_depth: int = 64,
        min_leaf_size: int = 1,
        seed: int = 42,
    ) -> None:
        self.n_trees = max(1, int(n_trees))
        self.subsample_size = max(0, int(subsample_size))
        self.max_depth = max(1, int(max_depth))
        self.min_leaf_size = max(1, int(min_leaf_size))
        self.seed = int(seed)
        self.trees: List[_MondrianTree] = []
        self.training_history: List[Dict[str, Any]] = []

    def _subsample(self, matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n_rows = matrix.shape[0]
        if not self.subsample_size or n_rows <= self.subsample_size:
            return matrix
        return matrix[rng.choice(n_rows, size=self.subsample_size, replace=False)]

    def fit(self, matrix: np.ndarray) -> "IMondrianForest":
        rng = np.random.default_rng(self.seed)
        self.trees = []
        for index in range(self.n_trees):
            tree = _MondrianTree(self.max_depth, self.min_leaf_size, self.seed + index * 9973)
            tree.fit(self._subsample(matrix, rng))
            self.trees.append(tree)
        self.training_history.append(
            {
                "stage": "batch_fit",
                "rows": int(matrix.shape[0]),
                "trees": self.n_trees,
                "rows_per_tree": int(self.trees[0].n_fitted),
            }
        )
        return self

    def partial_fit(self, matrix: np.ndarray) -> "IMondrianForest":
        """Extend every tree with a *different* subsample of ``matrix``.

        The per-tree subsample is the fix. The notebook's online update handed
        every tree the identical batch, so after a few updates the trees had
        converged on the same structure over the same points and the ensemble
        average was the same as any single tree — the variance reduction that is
        the only reason to have a hundred of them was gone. Since the batch fit
        *did* subsample per tree, this also meant a forest's diversity decayed
        monotonically with the number of updates applied to it, which is exactly
        backwards for a model whose selling point is staying current.
        """
        if not self.trees:
            return self.fit(matrix)
        rng = np.random.default_rng(self.seed + 1 + len(self.training_history))
        for tree in self.trees:
            tree.partial_fit(self._subsample(matrix, rng))
        self.training_history.append(
            {
                "stage": "online_extend",
                "rows": int(matrix.shape[0]),
                "trees": len(self.trees),
                "rows_per_tree": int(min(matrix.shape[0], self.subsample_size or matrix.shape[0])),
            }
        )
        return self

    def score_samples(self, matrix: np.ndarray) -> np.ndarray:
        """``2 ** -mean_t(h_t / c(n_t))``, in ``(0, 1]``, higher being more anomalous.

        The normalising constant is per tree, taken from the rows *that tree*
        was built on. The notebook divided by ``c(n_seen)`` where ``n_seen`` was
        the size of the whole training population — 130,000 rows against a
        4,096-row subsample. ``c(130000)`` is 22.71 and ``c(4096)`` is 15.79, so
        every observed path length was divided by a constant 44% too large, the
        exponent shrank toward zero, and the entire score distribution
        collapsed into a narrow band just above 0.5 where a float comparison in
        the eleventh decimal place decides the ranking. Worse,
        ``n_seen`` grew with every online update while the trees' subsample size
        did not, so the same forest's scores drifted with the number of updates
        applied to it and the online-update history was not comparable across
        batches.
        """
        if not self.trees:
            raise ModelError("iMondrian forest is not fitted")
        block = np.asarray(matrix, dtype=np.float32)
        ratios = np.zeros(block.shape[0], dtype=np.float64)
        for tree in self.trees:
            constant = float(average_path_length(max(tree.n_fitted, 2)))
            ratios += tree.path_lengths(block) / max(constant, _EPS)
        return np.power(2.0, -ratios / len(self.trees))


class IMondrianForestAdapter(ModelAdapter):
    """The iMondrian forest, exposed as one of the ten models.

    Registered as its own adapter, which is the fix for a naming defect with
    real consequences. The notebook's base ``fit_model`` handled ``imf`` in the
    same branch as ``isolation_forest`` and returned a scikit-learn
    ``IsolationForest``; a later cell in the ``imf`` notebook *only* rebound
    ``fit_model`` to a Mondrian implementation. So the ``model=imf`` prefix
    contained iMondrian results when written by the ``imf`` notebook and
    isolation-forest results when written by any other, under the same
    parameter identifiers, and the leaderboard aggregated them together. Two
    names cannot resolve to one estimator here: :data:`_ADAPTERS` maps each
    name to exactly one class.

    Online extension is available but not used by the sweep, which fits each arm
    once. It is exercised by :meth:`extend`, so that the property can be
    measured against the batch fit rather than assumed.
    """

    name = "imf"

    def _fit(self, matrix: np.ndarray) -> None:
        subsample = self.params.get("subsample_size", self.params.get("max_samples", 4096))
        if isinstance(subsample, str):
            # "auto" is in the grid for isolation forest and leaked into the imf
            # grid in the notebook, where int("auto") would have raised had the
            # branch ever been reached. Treated as the default rather than an
            # error because the two grids legitimately share dimension names.
            subsample = 4096
        self._model = IMondrianForest(
            n_trees=self._int_param("n_trees", self._int_param("n_estimators", 100)),
            subsample_size=max(0, int(subsample or 0)),
            max_depth=self._int_param("max_depth", 64),
            min_leaf_size=self._int_param("min_leaf_size", 1),
            seed=self.seed,
        )
        self._model.fit(matrix)
        self._implementation = "trust_score_05.ml.models.IMondrianForest"

    def _score(self, matrix: np.ndarray) -> np.ndarray:
        return self._model.score_samples(matrix)

    def extend(self, matrix: Any) -> "IMondrianForestAdapter":
        """Extend the fitted forest with new rows. Returns self."""
        self._require_fitted("extend")
        array = _as_matrix(matrix, f"{self.name}.extend")
        if array.shape[1] != self._n_train_features:
            raise ModelError(
                f"{self.name}.extend: matrix has {array.shape[1]} columns but the forest "
                f"was fitted on {self._n_train_features}"
            )
        self._model.partial_fit(array)
        return self

    def _diagnostics(self) -> Dict[str, Any]:
        return {
            "n_trees": len(self._model.trees),
            "subsample_size": self._model.subsample_size,
            "max_depth": self._model.max_depth,
            "min_leaf_size": self._model.min_leaf_size,
            "rows_per_tree": int(self._model.trees[0].n_fitted),
            "n_online_updates": sum(
                1 for row in self._model.training_history if row["stage"] == "online_extend"
            ),
        }

    def training_frames(self) -> Dict[str, pd.DataFrame]:
        self._require_fitted("training_frames")
        return {"imf_training_history": pd.DataFrame(self._model.training_history)}


# --------------------------------------------------------------------------
# coordinate-wise tail probability: ECOD and COPOD
# --------------------------------------------------------------------------

@dataclass
class _TailProbabilityScorer:
    """Coordinate-wise empirical tail probabilities, fitted on the training columns.

    This is the shared statistic behind ECOD and COPOD. For each column it holds
    the sorted training values, which is enough to evaluate the left tail
    probability ``P(X <= x)`` and the right tail ``P(X >= x)`` of any new value
    by binary search. The score of a row is the sum over columns of
    ``-log(tail)``, under three tail choices — all-left, all-right, and
    per-column by the sign of the training skewness — and the maximum of the
    three. A row that is extreme in either direction on several columns
    accumulates a large sum; a row in the bulk on every column accumulates
    close to zero.

    It exists as the *fallback* for the PyOD detectors, and its existence is
    the point. The notebook's fallback, when ``pyod`` could not be imported, was
    a mean absolute robust z-score over medians and median absolute deviations —
    a different statistic entirely, with a different sensitivity to heavy tails
    and no tail-direction handling — returned under the same ``kind`` and
    written to the same ``model=ecod`` prefix. Runs from environments with and
    without PyOD were averaged together on the leaderboard. Computing the
    *specified* statistic in numpy means the fallback is the same model, just
    another implementation of it, and the implementation is named in the
    diagnostics either way.

    The tail probabilities are floored at ``1 / n_train`` before the logarithm.
    A value beyond every training value has an empirical tail probability of
    exactly zero, whose logarithm is infinite, and one such column would make
    the row's score infinite and its rank arbitrary among all other such rows.
    The floor caps a single column's contribution at ``log(n_train)``, so a row
    extreme on five columns still outranks one extreme on one.
    """

    sorted_columns: np.ndarray
    skewness: np.ndarray
    n_train: int

    @classmethod
    def fit(cls, matrix: np.ndarray) -> "_TailProbabilityScorer":
        sorted_columns = np.sort(matrix, axis=0)
        centred = matrix - matrix.mean(axis=0)
        deviation = np.sqrt(np.maximum((centred**2).mean(axis=0), _EPS))
        skewness = (centred**3).mean(axis=0) / deviation**3
        return cls(
            sorted_columns=np.ascontiguousarray(sorted_columns),
            skewness=np.nan_to_num(skewness, nan=0.0, posinf=0.0, neginf=0.0),
            n_train=int(matrix.shape[0]),
        )

    def score(self, matrix: np.ndarray) -> np.ndarray:
        floor = 1.0 / max(self.n_train, 1)
        n_columns = matrix.shape[1]
        left = np.empty_like(matrix)
        right = np.empty_like(matrix)
        for column in range(n_columns):
            reference = self.sorted_columns[:, column]
            # 'right' side of searchsorted counts values <= x; the complement
            # counts values > x, and the right tail wants values >= x, so it is
            # taken from the 'left' side. Getting this backwards shifts both
            # tails by the number of ties, which for a sparse count column is
            # most of the population.
            at_or_below = np.searchsorted(reference, matrix[:, column], side="right")
            strictly_below = np.searchsorted(reference, matrix[:, column], side="left")
            left[:, column] = np.maximum(at_or_below / self.n_train, floor)
            right[:, column] = np.maximum(
                (self.n_train - strictly_below) / self.n_train, floor
            )
        negative_log_left = -np.log(left)
        negative_log_right = -np.log(right)
        skewed = np.where(
            self.skewness[None, :] < 0.0, negative_log_left, negative_log_right
        )
        return np.maximum(
            negative_log_left.sum(axis=1),
            np.maximum(negative_log_right.sum(axis=1), skewed.sum(axis=1)),
        )


class _PyODAdapter(ModelAdapter):
    """Base for the three PyOD detectors, with a named numpy fallback.

    ``contamination`` is accepted because the grids and the notebook both carry
    it, and because PyOD's constructors require a valid value. It has **no
    effect on the ranking**: PyOD uses it only to place ``threshold_``, which
    ``decision_function`` does not consult. Every metric this pipeline reports
    is a ranking metric, so sweeping contamination over seven values produced
    seven arms with byte-identical score columns, seven identical metrics rows,
    and seven copies of every plot — and then the best-configuration selection
    broke the seven-way tie arbitrarily and reported one of them as the winner.
    :mod:`trust_score_05.ml.grids` does not sweep it for this family.
    """

    #: ``module:attribute`` of the PyOD class this adapter prefers.
    pyod_target: ClassVar[Tuple[str, str]] = ("", "")

    def _pyod_kwargs(self) -> Dict[str, Any]:
        return {"contamination": self._contamination()}

    def _contamination(self) -> float:
        value = self._float_param("contamination", 0.01)
        # PyOD rejects anything outside (0, 0.5]. Clamping rather than raising
        # because the value provably cannot change the ranking, so refusing the
        # arm over it would drop a valid model for a parameter that does nothing.
        return float(min(max(value, 1e-6), 0.5))

    def _fit(self, matrix: np.ndarray) -> None:
        module_name, attribute = self.pyod_target
        try:
            module = __import__(module_name, fromlist=[attribute])
            factory = getattr(module, attribute)
        except Exception as exc:  # noqa: BLE001 - absent or broken, same outcome
            if bool(self._param("require_pyod", False)):
                raise ModelUnavailableError(
                    f"{self.name}: require_pyod is set but {module_name}.{attribute} is "
                    f"unimportable: {exc}"
                ) from exc
            LOGGER.warning(
                "%s: %s unavailable (%s); using the numpy implementation",
                self.name,
                module_name,
                exc,
            )
            self._fit_fallback(matrix, reason=str(exc))
            return
        kwargs = self._pyod_kwargs()
        try:
            self._model = factory(**kwargs)
        except TypeError:
            # Older PyOD releases do not accept every keyword. Retry with the
            # keywords the installed signature admits, and record which were
            # dropped -- the notebook retried with a hardcoded shorter kwargs
            # list and recorded nothing, so an arm whose n_bins was silently
            # ignored was indistinguishable from one where it took effect.
            import inspect

            accepted = set(inspect.signature(factory).parameters)
            self._dropped_kwargs = sorted(set(kwargs) - accepted)
            self._model = factory(**{k: v for k, v in kwargs.items() if k in accepted})
        self._model.fit(matrix)
        self._implementation = f"{module_name}.{attribute}"
        self._fallback = None

    def _fit_fallback(self, matrix: np.ndarray, reason: str) -> None:
        self._model = _TailProbabilityScorer.fit(matrix)
        self._implementation = "trust_score_05.ml.models._TailProbabilityScorer"
        self._fallback = reason

    def _score(self, matrix: np.ndarray) -> np.ndarray:
        if isinstance(self._model, _TailProbabilityScorer):
            return self._model.score(matrix)
        return np.asarray(self._model.decision_function(matrix), dtype=np.float64)

    def _diagnostics(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "contamination": self._contamination(),
            "used_fallback": self._fallback is not None,
        }
        if self._fallback:
            payload["fallback_reason"] = self._fallback[:200]
        dropped = getattr(self, "_dropped_kwargs", ())
        if dropped:
            payload["dropped_kwargs"] = ",".join(dropped)
        return payload


class ECODAdapter(_PyODAdapter):
    """Empirical-cumulative-distribution outlier detection.

    Coordinate-wise tail probabilities, aggregated by summing their negative
    logarithms. No pairwise distances and no parameters that affect the ranking,
    which makes it the cheapest useful arm in the sweep and the natural baseline
    for the rest: a model that cannot beat ECOD is not finding structure, it is
    finding marginal extremity.
    """

    name = "ecod"
    pyod_target = ("pyod.models.ecod", "ECOD")


class COPODAdapter(_PyODAdapter):
    """Copula-based outlier detection.

    Fits an empirical copula and reads each row's tail probability from it. The
    numpy fallback computes the coordinate-wise tail probabilities that the
    empirical copula reduces to under independent margins, which is the same
    quantity ECOD computes — so **the fallback scores for COPOD and ECOD are
    identical**, and only their PyOD implementations differ. That is stated
    plainly rather than papered over, because two leaderboard rows with
    identical metrics are otherwise a mystery worth a day of someone's time. The
    ``implementation`` field distinguishes them.
    """

    name = "copod"
    pyod_target = ("pyod.models.copod", "COPOD")


class HBOSAdapter(_PyODAdapter):
    """Histogram-based outlier score: per-column density, multiplied across columns.

    Each column gets an equal-width histogram of the training values; a row's
    score is the sum over columns of ``log2(1 / density)`` at the bin it falls
    into. Assumes the columns are independent, which they are not, and works
    anyway — because for fraud detection the interesting rows are usually
    extreme on a *marginal*, and a model that only reads marginals cannot be
    fooled by a correlation structure it has learned from a population that no
    longer holds.

    ``n_bins``, ``alpha`` and ``tol`` do affect the ranking, unlike
    ``contamination``, so they are the grid dimensions for this family.
    """

    name = "hbos"
    pyod_target = ("pyod.models.hbos", "HBOS")

    def _pyod_kwargs(self) -> Dict[str, Any]:
        return {
            "contamination": self._contamination(),
            "n_bins": self._int_param("n_bins", 20, minimum=2),
            "alpha": self._float_param("alpha", 0.1),
            "tol": self._float_param("tol", 0.5),
        }

    def _fit_fallback(self, matrix: np.ndarray, reason: str) -> None:
        self._model = _HistogramDensityScorer.fit(
            matrix,
            n_bins=self._int_param("n_bins", 20, minimum=2),
            alpha=self._float_param("alpha", 0.1),
            tol=self._float_param("tol", 0.5),
        )
        self._implementation = "trust_score_05.ml.models._HistogramDensityScorer"
        self._fallback = reason

    def _score(self, matrix: np.ndarray) -> np.ndarray:
        if isinstance(self._model, _HistogramDensityScorer):
            return self._model.score(matrix)
        return super()._score(matrix)

    def _diagnostics(self) -> Dict[str, Any]:
        payload = super()._diagnostics()
        payload.update(
            {
                "n_bins": self._int_param("n_bins", 20, minimum=2),
                "alpha": self._float_param("alpha", 0.1),
                "tol": self._float_param("tol", 0.5),
            }
        )
        return payload


@dataclass
class _HistogramDensityScorer:
    """Equal-width histogram densities per column, fitted on training values.

    The numpy stand-in for HBOS. Densities are normalised per column by that
    column's maximum, so a column's contribution is ``log2(1 / (relative
    density + alpha))`` — zero for the modal bin and at most ``log2(1 / alpha)``
    for an empty one. ``tol`` is how far outside the outermost bin edge, as a
    fraction of the bin width, a value may fall and still be assigned to that
    outermost bin; beyond that it takes the maximum contribution.

    A column whose training values are constant has no width to bin. Its
    histogram is a single bin holding everything, so it contributes zero to
    every in-range row and the maximum to any row that differs from the
    constant — which is the correct reading of "this column has one value and
    yours is not it".
    """

    edges: List[np.ndarray]
    densities: List[np.ndarray]
    alpha: float
    tol: float

    @classmethod
    def fit(
        cls, matrix: np.ndarray, n_bins: int, alpha: float, tol: float
    ) -> "_HistogramDensityScorer":
        edges: List[np.ndarray] = []
        densities: List[np.ndarray] = []
        for column in range(matrix.shape[1]):
            values = matrix[:, column]
            low = float(values.min())
            high = float(values.max())
            if not (high > low):
                edges.append(np.asarray([low, low], dtype=np.float64))
                densities.append(np.asarray([1.0], dtype=np.float64))
                continue
            counts, boundaries = np.histogram(values, bins=n_bins, range=(low, high))
            peak = float(counts.max())
            densities.append(counts / peak if peak > 0 else np.zeros_like(counts, float))
            edges.append(boundaries.astype(np.float64))
        return cls(edges=edges, densities=densities, alpha=float(alpha), tol=float(tol))

    def score(self, matrix: np.ndarray) -> np.ndarray:
        alpha = max(self.alpha, _EPS)
        worst = math.log2(1.0 / alpha)
        out = np.zeros(matrix.shape[0], dtype=np.float64)
        for column, (boundaries, density) in enumerate(zip(self.edges, self.densities)):
            values = matrix[:, column]
            n_bins = density.size
            width = (
                (float(boundaries[-1]) - float(boundaries[0])) / n_bins
                if boundaries[-1] > boundaries[0]
                else 0.0
            )
            slack = width * self.tol
            index = np.clip(
                np.searchsorted(boundaries, values, side="right") - 1, 0, n_bins - 1
            )
            contribution = np.log2(1.0 / (density[index] + alpha))
            outside = (values < float(boundaries[0]) - slack) | (
                values > float(boundaries[-1]) + slack
            )
            out += np.where(outside, worst, contribution)
        return out


# --------------------------------------------------------------------------
# reconstruction error
# --------------------------------------------------------------------------

class PCAReconstructionAdapter(ModelAdapter):
    """Mean squared error of a linear reconstruction through a fitted bottleneck.

    Fit PCA on the non-fraud population, project a row down and back up, and
    measure how much of it did not survive. A row that lies in the subspace the
    normal population occupies reconstructs exactly; one that does not, does
    not.

    ``n_components`` is resolved through :func:`resolve_pca_components`, so an
    integer larger than the available rank becomes the full-rank arm rather than
    a mid-sweep exception. That arm is worth keeping in the grid despite being
    degenerate — full rank reconstructs everything perfectly, giving errors at
    machine epsilon and a ranking that is pure floating-point noise — because
    seeing it score at chance is the confirmation that the metric is measuring
    reconstruction and not something else. Its ``explained_variance`` of 1.0
    makes it identifiable in the leaderboard.
    """

    name = "pca_reconstruction"

    def _fit(self, matrix: np.ndarray) -> None:
        from sklearn.decomposition import PCA

        resolved = resolve_pca_components(
            self.params.get("n_components", 0.9), matrix.shape[1], matrix.shape[0]
        )
        if resolved is None:
            raise ModelError(
                f"{self.name}: n_components must be a positive count or a variance "
                f"fraction, got {self.params.get('n_components')!r}"
            )
        self._model = PCA(
            n_components=resolved,
            whiten=bool(self._param("whiten", False)),
            random_state=self.seed,
        )
        self._model.fit(matrix)
        self._implementation = "sklearn.PCA"

    def _score(self, matrix: np.ndarray) -> np.ndarray:
        reconstructed = self._model.inverse_transform(self._model.transform(matrix))
        return np.mean((matrix - reconstructed) ** 2, axis=1)

    def _diagnostics(self) -> Dict[str, Any]:
        ratios = self._model.explained_variance_ratio_
        return {
            "n_components_requested": str(self.params.get("n_components", 0.9)),
            "n_components_resolved": int(self._model.n_components_),
            "whiten": bool(self._model.whiten),
            "explained_variance": float(ratios.sum()),
        }

    def training_frames(self) -> Dict[str, pd.DataFrame]:
        self._require_fitted("training_frames")
        ratios = self._model.explained_variance_ratio_
        return {
            "pca_explained_variance": pd.DataFrame(
                {
                    "component": np.arange(1, ratios.size + 1, dtype=np.int64),
                    "explained_variance_ratio": ratios,
                    "cumulative_explained_variance": np.cumsum(ratios),
                }
            )
        }


class AutoencoderAdapter(ModelAdapter):
    """Reconstruction error through a non-linear bottleneck.

    Prefers PyOD's torch-backed ``AutoEncoder``. Falls back to a scikit-learn
    ``MLPRegressor`` trained to predict its own input, which is an autoencoder
    in every respect except that the bottleneck is not enforced by the
    architecture alone — the hidden layer sizes still narrow, so it is.

    Two things about the fallback that the notebook's version obscured. Its
    ``max_iter`` is scikit-learn's *iteration* cap, not an epoch count, and with
    ``early_stopping=True`` the fit usually stops well before it; passing an
    ``epochs`` value there and reporting it as epochs trained overstated the
    training by however much early stopping saved. And ``early_stopping`` holds
    out 10% of the training rows to decide when to stop, so the fallback is
    fitted on 90% of the population the PyOD path uses all of. Both are recorded
    in the diagnostics: ``n_iter`` is what happened, ``epochs`` is what was
    asked for.
    """

    name = "autoencoder"

    def _fit(self, matrix: np.ndarray) -> None:
        hidden = [int(v) for v in self._param("hidden_neurons", [128, 64, 32, 64, 128])]
        if not hidden:
            raise ModelError(f"{self.name}: hidden_neurons is empty")
        self._hidden = hidden
        if bool(self._param("force_fallback", False)):
            self._fit_fallback(matrix, reason="force_fallback")
            return
        try:
            from pyod.models.auto_encoder import AutoEncoder
        except Exception as exc:  # noqa: BLE001 - torch or pyod absent
            if bool(self._param("require_pyod", False)):
                raise ModelUnavailableError(
                    f"{self.name}: require_pyod is set but pyod's AutoEncoder is "
                    f"unimportable: {exc}"
                ) from exc
            LOGGER.warning("%s: PyOD AutoEncoder unavailable (%s); using MLPRegressor",
                           self.name, exc)
            self._fit_fallback(matrix, reason=str(exc))
            return
        self._model = AutoEncoder(
            hidden_neuron_list=hidden[: max(1, len(hidden) // 2 + 1)],
            epoch_num=self._int_param("epochs", 20),
            batch_size=self._int_param("batch_size", 1024, minimum=8),
            lr=self._float_param("learning_rate", 0.001),
            contamination=min(max(self._float_param("contamination", 0.01), 1e-6), 0.5),
            random_state=self.seed,
            verbose=0,
        )
        self._model.fit(matrix)
        self._implementation = "pyod.models.auto_encoder.AutoEncoder"
        self._fallback = None

    def _fit_fallback(self, matrix: np.ndarray, reason: str) -> None:
        from sklearn.neural_network import MLPRegressor

        self._model = MLPRegressor(
            hidden_layer_sizes=tuple(self._hidden),
            activation="relu",
            solver="adam",
            learning_rate_init=self._float_param("learning_rate", 0.001),
            batch_size=min(self._int_param("batch_size", 1024, minimum=8), matrix.shape[0]),
            max_iter=self._int_param("epochs", 20),
            early_stopping=matrix.shape[0] >= 20,
            random_state=self.seed,
        )
        self._model.fit(matrix, matrix)
        self._implementation = "sklearn.MLPRegressor"
        self._fallback = reason

    def _score(self, matrix: np.ndarray) -> np.ndarray:
        if self._implementation == "sklearn.MLPRegressor":
            predicted = np.asarray(self._model.predict(matrix), dtype=np.float64)
            return np.mean((matrix - predicted.reshape(matrix.shape)) ** 2, axis=1)
        return np.asarray(self._model.decision_function(matrix), dtype=np.float64)

    def _diagnostics(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "hidden_neurons": "x".join(str(v) for v in self._hidden),
            "epochs": self._int_param("epochs", 20),
            "batch_size": self._int_param("batch_size", 1024, minimum=8),
            "learning_rate": self._float_param("learning_rate", 0.001),
            "used_fallback": self._fallback is not None,
            "n_iter": int(getattr(self._model, "n_iter_", 0) or 0),
        }
        if self._fallback:
            payload["fallback_reason"] = self._fallback[:200]
        return payload


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

_ADAPTERS: Dict[str, Type[ModelAdapter]] = {
    "kmeans": KMeansAdapter,
    "weighted_kmeans": WeightedKMeansAdapter,
    "hdbscan": HDBSCANAdapter,
    "isolation_forest": IsolationForestAdapter,
    "imf": IMondrianForestAdapter,
    "copod": COPODAdapter,
    "ecod": ECODAdapter,
    "hbos": HBOSAdapter,
    "pca_reconstruction": PCAReconstructionAdapter,
    "autoencoder": AutoencoderAdapter,
}


def adapter_class(model: str) -> Type[ModelAdapter]:
    """The adapter class for ``model``, or raise :class:`ModelError`.

    One name, one class. The lookup exists so that no caller has to hold a
    mapping of its own — the notebook had four, in four cells, and they
    disagreed about ``imf``.
    """
    try:
        return _ADAPTERS[model]
    except KeyError as exc:
        raise ModelError(
            f"unknown model {model!r}; expected one of {MODEL_NAMES}"
        ) from exc


def build_model(
    model: str, params: Optional[Dict[str, Any]] = None, seed: int = 42
) -> ModelAdapter:
    """Construct an unfitted adapter."""
    return adapter_class(model)(params=params, seed=seed)


def fit_model(
    model: str,
    matrix: Any,
    params: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> ModelAdapter:
    """Construct and fit an adapter in one call. Returns the fitted adapter.

    The signature is ``(model, matrix, params, seed)`` with ``seed`` an integer.
    The notebook's ``fit_model`` was redefined five times across its cells, and
    the fourth parameter was a seed in two of them and the whole
    ``PipelineConfig`` in the other three, so which one a call site got depended
    on cell execution order. An adapter never receives a config object: a model
    that can read the pipeline configuration is a model that can read a
    threshold it should not know about.
    """
    return build_model(model, params=params, seed=seed).fit(matrix)


def assert_model_names_registered() -> None:
    """Check the registry covers :data:`MODEL_NAMES` exactly. Called by the tests."""
    missing = sorted(set(MODEL_NAMES) - set(_ADAPTERS))
    extra = sorted(set(_ADAPTERS) - set(MODEL_NAMES))
    if missing or extra:
        raise ModelError(
            f"model registry disagrees with MODEL_NAMES: missing={missing} extra={extra}"
        )
    for name, cls in _ADAPTERS.items():
        if cls.name != name:
            raise ModelError(
                f"adapter {cls.__name__} is registered as {name!r} but names itself "
                f"{cls.name!r}"
            )
