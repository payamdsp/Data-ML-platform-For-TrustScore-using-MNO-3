"""Hyperparameter grids: what the sweep searches, and what it stops searching.

A grid is separate from its model because the two answer different questions.
:mod:`trust_score_05.ml.models` says what an algorithm does with a parameter;
this module says which values are worth spending a fit on. The second answer
depends on the shape of the data and on the search budget, and it changes
without the first changing at all.

Every dimension in here earns its place by changing the *ranking* a model
produces. That test removed most of the notebook's grid. Three of its models
swept ``contamination`` over six or seven values —
``[0.0005, 0.001, 0.003, 0.005, 0.01, 0.02, 0.05]`` for ECOD and COPOD — and
``contamination`` sets a PyOD detector's ``threshold_``, which
``decision_function`` never reads. Every metric this pipeline reports is a
ranking metric, so those arms produced byte-identical score columns. ECOD's
entire grid was that one dimension: seven arms, one model, seven copies of every
plot and metrics row, and a best-configuration step that broke the seven-way tie
by whichever happened to sort first. The same applies to the isolation forest's
seven contamination values, which multiplied its grid by seven for nothing, and
to HBOS's six — although HBOS's ``n_bins``, ``alpha`` and ``tol`` do change its
ranking, so it keeps those. On a 100-feature, 130,000-row arm the ten grids now
total 5,619 configurations; carrying ``contamination`` at the notebook's
cardinalities would have made that 6,396, of which 777 were duplicate fits —
every one of ECOD's and COPOD's beyond the first, six sevenths of the isolation
forest's, and five sixths of HBOS's.

The second correction is to ``k``. The notebook derived the number of k-means
clusters from the number of *features*: ``range(2, n_features + 1)`` below 75
features, and a ladder capped at ``n_features`` above it. The number of groups a
population falls into is not a function of how many columns describe it. The
practical effect was that a 25-feature arm could never try more than 25
clusters while a 200-feature arm tried 200, so the ``top_n_features`` dimension
and the ``k`` dimension were confounded: an experiment that looked like it was
measuring the effect of feature count was also, invisibly, changing the model
class. :func:`k_values` uses a fixed ladder bounded by the *row* count, which is
the quantity that actually constrains how many clusters can be estimated.

Grids are deterministic and ordered. :func:`limit_grid` truncates from the
front, so a budgeted run is a prefix of the full run and its results are a
subset of what the full run would have produced. Truncating randomly — or
shuffling first — makes a budgeted sweep unreproducible from the configuration
alone, and makes two budgeted sweeps of the same config incomparable.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from trust_score_05.ml.config import MLConfig
from trust_score_05.ml.models import MODEL_NAMES, ModelError

__all__ = [
    "AUTOENCODER_ARCHITECTURES",
    "GRID_DIMENSIONS",
    "K_LADDER",
    "PCA_VARIANCE_TARGETS",
    "config_grid_for_model",
    "grid_shape",
    "k_values",
    "limit_grid",
    "pca_component_values",
]

LOGGER = logging.getLogger(__name__)

#: Candidate cluster counts, bounded at use time by the row count. A ladder
#: rather than a range: the difference between k=101 and k=102 is not something
#: a fraud metric can resolve, while the difference between k=8 and k=64 is, so
#: dense sampling in the low range and geometric spacing above it spends the
#: budget where the answer changes.
K_LADDER: Tuple[int, ...] = (
    2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 28, 32,
    40, 48, 64, 80, 100, 128, 160, 200,
)

#: Variance targets for a PCA rank chosen by explained variance rather than a
#: component count. Useful precisely because they are invariant to the feature
#: count, so the same value means the same thing on a 25-feature and a
#: 200-feature arm — which is what the ``top_n_features`` dimension needs in
#: order to be interpretable.
PCA_VARIANCE_TARGETS: Tuple[float, ...] = (0.80, 0.90, 0.95)

#: Encoder/decoder widths. Symmetric, and narrowing to a bottleneck well below
#: the input width, because a bottleneck wider than the input reconstructs the
#: identity and scores everything at zero.
AUTOENCODER_ARCHITECTURES: Tuple[Tuple[int, ...], ...] = (
    (128, 64, 32, 64, 128),
    (256, 128, 64, 128, 256),
    (512, 256, 128, 64, 128, 256, 512),
)

#: Every parameter each model's grid emits, whether or not it varies. Published
#: as data so that the sweep manifest can record the search space it walked
#: without reconstructing it, and so a reader can see at a glance that
#: ``contamination`` appears nowhere.
#:
#: A dimension pinned to a single value is listed rather than omitted. The
#: weighted k-means grid emits ``score_variant`` fixed at
#: ``cluster_z_distance``, and leaving it out of this table meant
#: :func:`grid_shape` reported a search space that did not mention it at all —
#: so a reader of the manifest, knowing the unweighted grid sweeps four
#: variants, had no way to tell which one the weighted arms used and every
#: reason to assume the same four. ``grid_shape`` now reports it with a count of
#: one, which says plainly that it is pinned. The completeness of this table
#: against what the builders actually emit is asserted in
#: ``tests/ml/test_grids.py``.
GRID_DIMENSIONS: Dict[str, Tuple[str, ...]] = {
    "kmeans": ("k", "batch_size", "score_variant", "pca_components"),
    "weighted_kmeans": (
        "k",
        "batch_size",
        "weight_strategy",
        "weight_update_iterations",
        "pca_components",
        "score_variant",
    ),
    "hdbscan": ("min_cluster_size", "min_samples", "metric", "cluster_selection_method"),
    "isolation_forest": ("n_estimators", "max_samples", "max_features", "bootstrap"),
    "imf": ("n_trees", "subsample_size", "max_depth", "min_leaf_size"),
    "copod": (),
    "ecod": (),
    "hbos": ("n_bins", "alpha", "tol"),
    "pca_reconstruction": ("n_components", "whiten"),
    "autoencoder": ("epochs", "hidden_neurons", "batch_size", "learning_rate"),
}


def k_values(n_rows: int, pilot: bool = False) -> List[int]:
    """Cluster counts to try against a training population of ``n_rows`` rows.

    Bounded above by ``n_rows // _MIN_ROWS_PER_CLUSTER`` as well as by
    ``n_rows`` itself. A partition whose clusters average a handful of members
    has per-cluster distance statistics estimated from a handful of members, and
    :class:`~trust_score_05.ml.models._ClusterScoreStats` freezes those
    statistics and divides by them — so an under-populated cluster does not
    merely add noise, it adds a divisor estimated from five points that every
    row assigned to it is then scored against.

    Deliberately *not* a function of the feature count; see the module
    docstring.
    """
    n_rows = max(2, int(n_rows))
    ceiling = max(2, min(n_rows, n_rows // _MIN_ROWS_PER_CLUSTER))
    ladder = _PILOT_K_LADDER if pilot else K_LADDER
    return sorted({k for k in ladder if 2 <= k <= ceiling})


#: Minimum average cluster occupancy. 25 is the point below which a
#: within-cluster standard deviation stops being an estimate and starts being an
#: artifact of the sample.
_MIN_ROWS_PER_CLUSTER = 25

_PILOT_K_LADDER: Tuple[int, ...] = (2, 4, 8, 16, 32, 64)


def pca_component_values(
    n_features: int, pilot: bool = False, include_none: bool = True
) -> List[Any]:
    """PCA ranks to try, as ``None``, integer counts, and variance fractions.

    Integer counts above ``n_features`` are dropped rather than clamped.
    :func:`~trust_score_05.ml.models.resolve_pca_components` clamps, which keeps
    a stray value from crashing a sweep, but a grid that *generates* two values
    clamping to the same rank generates two arms that fit the same model and
    report the same metrics — and the notebook's ``min(100, n_features)``
    construction did exactly that, producing up to four duplicate full-rank arms
    on a narrow feature set. Dropping here and clamping there means duplicates
    are prevented at the source and tolerated at the boundary.
    """
    n_features = max(1, int(n_features))
    counts = (10, 25) if pilot else (10, 25, 50, 100)
    targets = (0.90,) if pilot else PCA_VARIANCE_TARGETS
    values: List[Any] = [None] if include_none else []
    values.extend(count for count in counts if count < n_features)
    values.extend(targets)
    return values


def _hdbscan_min_cluster_sizes(n_rows: int, pilot: bool) -> List[int]:
    """Minimum cluster sizes, as absolute counts and as shares of the population.

    Both, because ``min_cluster_size`` is the parameter that decides what
    HDBSCAN calls noise, and the answer is sometimes absolute — "a group of
    fewer than fifty accounts is not a segment" — and sometimes relative — "a
    group under a thousandth of the population is not a segment". The notebook
    mixed absolute values with multiples of the *feature* count, which is
    neither.
    """
    shares = (0.001, 0.005) if pilot else (0.0005, 0.001, 0.005, 0.01, 0.05)
    absolute = (50, 500) if pilot else (25, 50, 100, 250, 500, 2500)
    candidates = set(absolute)
    candidates.update(int(share * n_rows) for share in shares)
    return sorted(size for size in candidates if 5 <= size <= max(5, n_rows // 2))


def _row_ladder(ladder: Sequence[int], n_rows: int) -> Tuple[int, ...]:
    """The rungs of ``ladder`` that fit inside ``n_rows``, or the population itself.

    For any dimension whose values are counts of rows — a bagging subsample, a
    mini-batch size. Both halves matter, and each fixes a different way the
    notebook's ladders misbehaved on a small population.

    Dropping the rungs that do not fit is what stops two arms being the same
    fitted model. Every such parameter in scikit-learn silently clamps to the
    population rather than failing: ``IsolationForest.max_samples`` warns and
    clamps, and ``MiniBatchKMeans`` and ``MLPRegressor`` take
    ``min(batch_size, n_samples)``. So on a 600-row matrix a ladder of
    ``(1024, 4096)`` is two arms that fit *identical* models, write two
    hyperparameter prefixes, and land two indistinguishable rows on the
    leaderboard — where the best-config step then breaks a tie between one model
    and itself. The notebook's pilot ladders were unfiltered and did this every
    time the pilot row cap fell below the smallest rung, which for its 25,000-row
    pilot was most of them.

    Falling back to ``(n_rows,)`` when nothing fits is what stops the grid
    emptying. An empty dimension makes the whole cross-product empty, and
    :func:`config_grid_for_model` reads an empty grid as "this model cannot run
    on this data" — which is wrong here, because none of these parameters has a
    lower bound worth enforcing. A forest on 600 rows is a forest.
    """
    n_rows = max(1, int(n_rows))
    fitted = tuple(int(value) for value in ladder if int(value) <= n_rows)
    return fitted if fitted else (n_rows,)


def config_grid_for_model(
    model: str,
    cfg: MLConfig,
    n_features: int,
    n_rows: int,
) -> List[Dict[str, Any]]:
    """The hyperparameter configurations to try for ``model``, in a fixed order.

    ``n_features`` is the width of the *selected* feature set for this arm of
    the sweep and ``n_rows`` the height of the training matrix. Both are passed
    in rather than read from ``cfg`` because they are properties of the data
    that reached this arm — the configured ``top_n_feature_counts`` value is an
    upper bound on the first, not its value, since selection returns fewer
    features when fewer survive the null-rate and variance gates.

    The returned dicts contain only the dimensions in
    :data:`GRID_DIMENSIONS`. Nothing that does not affect the ranking is
    included, so no two entries ever describe the same fitted model, and
    :func:`~trust_score_05.ml.paths.hyperparam_id` over these dicts is injective
    on the grid.
    """
    if model not in MODEL_NAMES:
        raise ModelError(f"unknown model {model!r}; expected one of {MODEL_NAMES}")
    pilot = cfg.mode == "pilot"
    builder = _BUILDERS[model]
    grid = builder(pilot=pilot, n_features=int(n_features), n_rows=int(n_rows))
    if not grid:
        raise ModelError(
            f"the grid for {model!r} is empty at n_features={n_features} "
            f"n_rows={n_rows}; the training matrix is too small for this model"
        )
    return limit_grid(grid, cfg.max_hyperparam_configs)


def limit_grid(grid: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """The first ``limit`` entries of ``grid``, or all of them if ``limit`` is 0.

    Zero disables the limit, per the repository convention. Truncation is from
    the front and the grids are ordered with their cheapest and most-likely
    dimensions first, so a budgeted run is a genuine prefix of the full run:
    rerunning with a larger budget adds arms without invalidating the ones
    already computed, which is what makes the sweep's resume markers meaningful
    across a budget change.

    The notebook took its bounds from four environment variables consulted in
    two different orders — ``CONFIG_SLICE_SIZE``,
    ``MAX_HYPERPARAM_CONFIGS_PER_NOTEBOOK``, ``CONFIG_SLICE_START``, and a
    config field of the same name — so the size of the grid a run had walked was
    not recoverable from its artifacts. Here it is one config field, recorded in
    the run manifest.
    """
    if limit and limit > 0:
        return [dict(entry) for entry in grid[: int(limit)]]
    return [dict(entry) for entry in grid]


def grid_shape(model: str, cfg: MLConfig, n_features: int, n_rows: int) -> Dict[str, Any]:
    """A summary of a model's grid, for planning a sweep without building it.

    Returns the dimension names, the number of values each takes, the full
    cross-product size and the size after the configured limit. Worth having as
    its own call because the useful question before starting a sweep is "how
    many fits is this", and answering it by building the grid and taking its
    length is fine for one model and misleading for ten: the product is what
    tells you which dimension to cut.
    """
    if model not in MODEL_NAMES:
        raise ModelError(f"unknown model {model!r}; expected one of {MODEL_NAMES}")
    full = _BUILDERS[model](
        pilot=cfg.mode == "pilot", n_features=int(n_features), n_rows=int(n_rows)
    )
    counts: Dict[str, int] = {}
    for dimension in GRID_DIMENSIONS[model]:
        counts[dimension] = len({str(entry.get(dimension)) for entry in full})
    return {
        "model": model,
        "dimensions": counts,
        "n_configs_full": len(full),
        "n_configs_after_limit": len(limit_grid(full, cfg.max_hyperparam_configs)),
        "limit": int(cfg.max_hyperparam_configs),
    }


# --------------------------------------------------------------------------
# per-model builders
# --------------------------------------------------------------------------

def _product(**dimensions: Iterable[Any]) -> List[Dict[str, Any]]:
    """Cross-product of named dimensions, as a list of dicts in a fixed order.

    Ordered by the *last* dimension varying fastest, which is
    ``itertools.product``'s own order, so the caller controls the truncation
    order by the order it passes the keyword arguments — earliest arguments vary
    slowest and are therefore the dimensions a truncated grid explores least.
    Put the dimension you most want covered last.
    """
    names = list(dimensions)
    values = [list(dimensions[name]) for name in names]
    if any(not value for value in values):
        return []
    return [dict(zip(names, combination)) for combination in itertools.product(*values)]


def _kmeans_grid(pilot: bool, n_features: int, n_rows: int) -> List[Dict[str, Any]]:
    return _product(
        pca_components=pca_component_values(n_features, pilot),
        batch_size=_row_ladder((8192,) if pilot else (4096, 8192, 32768), n_rows),
        k=k_values(n_rows, pilot),
        score_variant=(
            ("cluster_z_distance",)
            if pilot
            else (
                "raw_distance",
                "cluster_z_distance",
                "cluster_percentile",
                "cluster_size_adjusted",
            )
        ),
    )


def _weighted_kmeans_grid(pilot: bool, n_features: int, n_rows: int) -> List[Dict[str, Any]]:
    return _product(
        pca_components=pca_component_values(n_features, pilot),
        batch_size=_row_ladder((8192,) if pilot else (8192, 32768), n_rows),
        weight_update_iterations=(1,) if pilot else (1, 3, 5),
        # score_variant is fixed rather than swept. The weighted arm exists to
        # test whether learned feature weights help; sweeping the score variant
        # here as well would multiply the grid by four to answer a question the
        # unweighted arm already answers, and the interaction between the two is
        # not what the experiment is for.
        score_variant=("cluster_z_distance",),
        weight_strategy=(
            ("inverse_dispersion",)
            if pilot
            else ("inverse_dispersion", "variance_stability", "hybrid")
        ),
        k=k_values(n_rows, pilot),
    )


def _hdbscan_grid(pilot: bool, n_features: int, n_rows: int) -> List[Dict[str, Any]]:
    return _product(
        # No PCA dimension, by design: HDBSCAN's density estimate is what the
        # arm is testing, and projecting first replaces the density of the
        # feature space with the density of a rotation of it.
        metric=("euclidean",) if pilot else ("euclidean", "manhattan"),
        cluster_selection_method=("eom",) if pilot else ("eom", "leaf"),
        min_cluster_size=_hdbscan_min_cluster_sizes(n_rows, pilot),
        min_samples=(None, 25) if pilot else (None, 5, 10, 25, 50, 100),
    )


def _isolation_forest_grid(pilot: bool, n_features: int, n_rows: int) -> List[Dict[str, Any]]:
    """Isolation forest arms. ``max_samples`` is always an explicit row count.

    Never ``"auto"``, even though that is scikit-learn's default and the
    notebook swept it alongside the integers. ``max_samples="auto"`` means
    ``min(256, n_samples)``, so on any training matrix of 256 rows or more it is
    the same draw as ``max_samples=256`` — the notebook's ladder contained both,
    which made two arms out of one fitted forest and left the best-config step
    choosing between two identical score columns. Stating the count also puts it
    in the hyperparameter identifier, so the subsample a run used is readable
    from its prefix instead of depending on what the row cap happened to be.
    """
    return _product(
        bootstrap=(False,) if pilot else (False, True),
        max_features=(1.0,) if pilot else (0.4, 0.75, 1.0),
        n_estimators=(200,) if pilot else (300, 800, 1600),
        max_samples=_row_ladder(
            (1024, 4096) if pilot else (256, 1024, 4096, 16384, 65536), n_rows
        ),
    )


def _imf_grid(pilot: bool, n_features: int, n_rows: int) -> List[Dict[str, Any]]:
    return _product(
        min_leaf_size=(1,) if pilot else (1, 5, 25),
        max_depth=(32,) if pilot else (16, 32, 64),
        n_trees=(64,) if pilot else (64, 128, 256),
        subsample_size=_row_ladder(
            (1024, 4096) if pilot else (256, 1024, 4096, 16384), n_rows
        ),
    )


def _tail_probability_grid(pilot: bool, n_features: int, n_rows: int) -> List[Dict[str, Any]]:
    """ECOD and COPOD have no ranking-relevant parameters, so the grid is one arm.

    Returning a single empty-parameter configuration rather than an empty list,
    because an empty list would mean "this model cannot run here" and these two
    can always run. The single arm gets the hyperparameter identifier of a
    parameterless model, which reads oddly in a path and is correct.
    """
    return [{}]


def _hbos_grid(pilot: bool, n_features: int, n_rows: int) -> List[Dict[str, Any]]:
    return _product(
        tol=(0.5,) if pilot else (0.1, 0.5, 1.0),
        alpha=(0.1,) if pilot else (0.01, 0.1, 0.2),
        n_bins=(20,) if pilot else (5, 10, 20, 50, 100),
    )


def _pca_reconstruction_grid(pilot: bool, n_features: int, n_rows: int) -> List[Dict[str, Any]]:
    counts = (5, 10) if pilot else (2, 5, 10, 20, 50, 100)
    return _product(
        whiten=(False,) if pilot else (False, True),
        # Full rank is excluded: it reconstructs every row exactly and ranks by
        # floating-point residue. The notebook's 0.99 target came close enough
        # to the same thing to produce a chance-level arm at the top of the
        # grid, which is why 0.95 is the highest target here.
        n_components=tuple(value for value in counts if value < n_features)
        + ((0.90,) if pilot else PCA_VARIANCE_TARGETS),
    )


def _autoencoder_grid(pilot: bool, n_features: int, n_rows: int) -> List[Dict[str, Any]]:
    architectures = (
        (AUTOENCODER_ARCHITECTURES[0],)
        if pilot
        else tuple(
            architecture
            for architecture in AUTOENCODER_ARCHITECTURES
            # An encoder whose first layer is narrower than the input is a
            # bottleneck the arm did not ask for, and one whose bottleneck
            # exceeds the input width is no bottleneck at all.
            if min(architecture) < n_features
        )
        or (AUTOENCODER_ARCHITECTURES[0],)
    )
    return _product(
        learning_rate=(0.001,) if pilot else (0.0005, 0.001, 0.003),
        batch_size=_row_ladder((1024,) if pilot else (512, 2048), n_rows),
        epochs=(10,) if pilot else (20, 50, 100),
        hidden_neurons=tuple(list(architecture) for architecture in architectures),
    )


_BUILDERS = {
    "kmeans": _kmeans_grid,
    "weighted_kmeans": _weighted_kmeans_grid,
    "hdbscan": _hdbscan_grid,
    "isolation_forest": _isolation_forest_grid,
    "imf": _imf_grid,
    "copod": _tail_probability_grid,
    "ecod": _tail_probability_grid,
    "hbos": _hbos_grid,
    "pca_reconstruction": _pca_reconstruction_grid,
    "autoencoder": _autoencoder_grid,
}
