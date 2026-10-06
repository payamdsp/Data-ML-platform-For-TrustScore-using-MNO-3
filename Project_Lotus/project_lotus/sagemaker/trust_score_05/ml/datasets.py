"""Finding the parquet, reading it safely, and assembling the two populations.

Everything downstream of here sees pandas frames with a known schema. This
module is where the object store's messiness is absorbed: two different monthly
layouts, feed-partitioned fraud prefixes whose partition values are not
enumerable from config, feature columns that exist in one month and not the
next, and a fraud feed whose timestamps are strings of inconsistent quality.

Three things happen in a fixed order, and the order is the point.

*Discovery* resolves each configured time window to exactly one ``/data/``
prefix by probing candidates (:mod:`trust_score_05.ml.paths` builds them) and
records the windows that resolved to nothing. The notebook logged those as
warnings, which meant a month silently missing from the training set looked
identical in the output to a month that was never configured.

*Reading* goes one prefix at a time. Handing Spark several
``…/time_window=…/data`` paths in one ``spark.read.parquet`` call goes wrong in
two different ways, and which one you get depends on the layouts the months
happened to resolve to. With a ``basePath`` spanning them — or reading their
common ancestor recursively — Spark's partition inference sees two directory
depths under one root and fails the read outright: "Conflicting directory
structures detected", whose own remedy is "please load them separately and then
union them". Without a ``basePath`` it does not fail at all; each path becomes
its own root, nothing beneath it is a partition directory, and the read succeeds
having *silently dropped every partition column*, so a training frame comes back
with no ``time_window`` and no record of which month a row belongs to. Reading
each prefix on its own, with its own inferred root, and combining with
``unionByName(allowMissingColumns=True)`` avoids both, and the provenance column
:data:`SOURCE_PATH_COLUMN` keeps the origin of every row recoverable after the
union has flattened the plan.

*Assembly* builds the training matrix and the per-scenario evaluation frames.
The training matrix is non-fraud only, and it has fraud customers removed from
it by an anti-join before sampling: without that, a customer who later turned
out to be fraudulent is in the population the model learns "normal" from, and
the model is being taught that fraud is normal. The notebook performed this join
but downgraded a missing fraud feed to a warning and continued, so a listing
failure produced a contaminated training set and a run that looked successful.
Here the absence of fraud data raises.

The evaluation fraud population is *fixed* across scenarios: every evaluation
window is scored against the same fraud rows rather than only the fraud that
occurred in that window. That is deliberate and was the fraud team's call —
there are too few labelled cases per month for a per-window fraud set to give a
stable metric — but it means a scenario's metrics answer "can this model
separate this month's normal traffic from known fraud", not "would it have
caught this month's fraud". :func:`load_scenario_frame` writes the scenario name
onto every row so a reader can never confuse the two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from trust_score_05.common.io.s3 import join_uri, list_common_prefixes, s3_exists, write_pandas
from trust_score_05.common.splitting import (
    CASE_KEY_COLUMN,
    CUSTOMER_KEY_COLUMN,
    LABEL_COLUMN,
)
from trust_score_05.ml.config import MLConfig
from trust_score_05.ml.paths import (
    fraud_source_root,
    testing_nonfraud_candidates,
    training_nonfraud_candidates,
)
from trust_score_05.ml.taxonomy import FRAUD_SOURCES, resolve_fraud_source

__all__ = [
    "CASE_KEY_COLUMN",
    "CUSTOMER_KEY_COLUMN",
    "FRAUD_META_COLUMNS",
    "FRAUD_TIMESTAMP_COLUMN",
    "LABEL_COLUMN",
    "SOURCE_PATH_COLUMN",
    "DatasetError",
    "DiscoveredPaths",
    "add_missing_and_select",
    "bounded_to_pandas",
    "discover_fraud_paths",
    "discover_testing_nonfraud_paths",
    "discover_training_nonfraud_paths",
    "fraud_source_from_path",
    "load_fraud_customer_ids",
    "load_fraud_frame",
    "load_scenario_frame",
    "load_training_matrix",
    "read_parquet_path",
    "read_parquet_paths",
]

LOGGER = logging.getLogger(__name__)

# The join key, the fraud case key and the label are re-exported from
# trust_score_05.common.splitting rather than redefined. They are the same
# strings that the aggregation levels group by, and two definitions of
# "customer_id" that could drift apart is exactly the kind of duplication that
# makes a customer-level metric quietly become a record-level one.

FRAUD_TIMESTAMP_COLUMN = "fraud_timestamp"

#: Provenance. Added by :func:`read_parquet_paths` to every row, because after a
#: union the partition values are no longer visible anywhere else and the fraud
#: feed's identity has to be recoverable from something.
SOURCE_PATH_COLUMN = "__source_parquet_path"

#: Carried alongside the features through scoring so a scored record can be
#: traced back to a customer, a case and a feed. Not features: nothing in this
#: list is ever handed to a model.
FRAUD_META_COLUMNS: Tuple[str, ...] = (
    CUSTOMER_KEY_COLUMN,
    "reference_id",
    "industry",
    "partner",
    "fraud_type",
    FRAUD_TIMESTAMP_COLUMN,
    "fraud_source",
    "source",
    "source_name",
    "fraud_event_key",
    CASE_KEY_COLUMN,
)


class DatasetError(RuntimeError):
    """A dataset could not be discovered, read, or assembled as configured."""


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DiscoveredPaths:
    """The outcome of resolving configured time windows to actual prefixes.

    ``found`` maps time window to the one prefix that exists; ``missing`` lists
    the windows that matched no candidate, and ``probed`` records every candidate
    tried per window. Missing windows are returned rather than logged so the
    caller can put them in its manifest — an evaluation that silently covered
    four of six months is the failure this type exists to make visible.
    """

    found: Dict[str, str] = field(default_factory=dict)
    missing: Tuple[str, ...] = ()
    probed: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    def paths(self) -> List[str]:
        """The resolved prefixes, in configured window order."""
        return [self.found[window] for window in self.found]

    def require(self, what: str) -> "DiscoveredPaths":
        """Raise unless at least one window resolved."""
        if not self.found:
            raise DatasetError(
                f"no {what} prefixes exist for any configured window; probed "
                f"{sum(len(v) for v in self.probed.values())} candidates across "
                f"{len(self.probed)} windows"
            )
        return self


def _discover(windows: Sequence[str], candidates_for) -> DiscoveredPaths:
    found: Dict[str, str] = {}
    missing: List[str] = []
    probed: Dict[str, Tuple[str, ...]] = {}
    for window in windows:
        candidates = tuple(candidates_for(window))
        probed[window] = candidates
        for candidate in candidates:
            if s3_exists(candidate):
                found[window] = candidate
                break
        else:
            missing.append(window)
            LOGGER.warning("no data prefix for window %s; probed %s", window, list(candidates))
    return DiscoveredPaths(found=found, missing=tuple(missing), probed=probed)


def discover_training_nonfraud_paths(cfg: MLConfig) -> DiscoveredPaths:
    """Resolve every configured training month to its ``/data/`` prefix."""
    return _discover(
        cfg.training_months,
        lambda window: training_nonfraud_candidates(cfg, window),
    )


def discover_testing_nonfraud_paths(cfg: MLConfig) -> DiscoveredPaths:
    """Resolve every configured evaluation window to its ``/data/`` prefix.

    Keyed by the full time window, not by the scenario name, because two windows
    inside one month would collide on ``window[:7]``. Use
    :meth:`MLConfig.scenario_names` to label the results.
    """
    return _discover(
        cfg.testing_windows,
        lambda window: testing_nonfraud_candidates(cfg, window),
    )


def discover_fraud_paths(cfg: MLConfig) -> Dict[str, List[str]]:
    """The ``/data/`` prefixes under each configured fraud feed.

    The feed's date partitions are not in config — the feed adds them — so they
    are listed. A listing failure propagates as
    :class:`~trust_score_05.common.io.s3.ObjectStoreError` rather than being
    caught: the notebook's ``except Exception`` here turned a throttle or an
    expired token into "this feed has no fraud", which then removed no customers
    from the training set and produced no error anywhere.
    """
    out: Dict[str, List[str]] = {}
    for source in cfg.fraud_sources:
        root = fraud_source_root(cfg, source)
        data_prefixes = []
        for partition in list_common_prefixes(root):
            candidate = join_uri(partition, "data") + "/"
            if s3_exists(candidate):
                data_prefixes.append(candidate)
            else:
                LOGGER.info("fraud partition has no data/ prefix, skipping: %s", partition)
        if not data_prefixes:
            LOGGER.warning("fraud feed %s has no partitions with data: %s", source, root)
        out[source] = data_prefixes
    return out


#: The feed directory in a provenance path. The value is the feed's *own* name,
#: version suffix included (``Phase1-v1.0``), which is why it is resolved rather
#: than compared — see :func:`trust_score_05.ml.taxonomy.resolve_fraud_source`.
_FRAUD_SEGMENT_PATTERN = re.compile(r"/fraud/([^/]+)/")


def fraud_source_from_path(path: str) -> Optional[str]:
    """Which feed a fraud row came from, recovered from its provenance path.

    The fraud rows themselves are not guaranteed to carry a ``fraud_source``
    column, but they are always read from ``…/fraud/<feed>/…``, so the feed
    identity is in the path even when it is absent from the data. This matters
    because :func:`trust_score_05.ml.taxonomy.category_for` refuses an unknown
    source and would otherwise raise on every row of a feed that omitted it.

    The path segment is the feed's own directory name — ``Phase1-v1.0``, not
    ``phase1`` — so it is put through
    :func:`~trust_score_05.ml.taxonomy.resolve_fraud_source` rather than
    compared against the taxonomy keys. Matching the keys directly, which is
    what the notebook did, recovered the source from no real path at all.

    :func:`load_fraud_frame` does the same fill-in inside Spark, via
    :func:`_source_from_path_expression`. This function is the pandas-side
    equivalent, for code reading scored records back out of the store.
    """
    match = _FRAUD_SEGMENT_PATTERN.search(str(path))
    if match is None:
        return None
    return resolve_fraud_source(match.group(1))


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

_PARTITION_MARKERS = ("/time_window=", "/dt=")


def _partition_base_path(path: str) -> Optional[str]:
    """The partition root above the outermost ``key=value`` segment, or ``None``.

    Passed to Spark as ``basePath`` so it reads the partition values as columns
    instead of guessing a root from the path it was handed.

    "Outermost" means the marker that occurs earliest in the path, not the first
    entry of :data:`_PARTITION_MARKERS` that happens to be present. The two
    differ when a path carries both — a ``dt=`` fraud partition under a root that
    itself sits below a ``time_window=`` segment — and taking the marker-list
    order there would name a root *below* a partition column, which is the
    condition ``basePath`` exists to prevent: Spark then reports a column the
    data does not contain.

    The markers carry their own leading separator, so the slice normally stops
    cleanly. The ``rstrip`` covers the one case where it does not: a prefix
    assembled by concatenation rather than by
    :func:`~trust_score_05.common.io.s3.join_uri` can contain a doubled
    separator, and ``s3://b/version=v3/`` and ``s3://b/version=v3`` are two
    different ``basePath`` values to Spark.
    """
    text = str(path)
    positions = [text.index(marker) for marker in _PARTITION_MARKERS if marker in text]
    if not positions:
        return None
    return text[: min(positions)].rstrip("/")


def read_parquet_path(spark: SparkSession, path: str) -> DataFrame:
    """Read one parquet prefix, with ``basePath`` set when it can be inferred.

    Retries without ``basePath`` if the first attempt fails, since an inferred
    root that is wrong — a hand-assembled path, a prefix that happens to contain
    ``dt=`` in a customer identifier — is worse than no root at all.
    """
    base_path = _partition_base_path(path)
    if not base_path:
        return spark.read.parquet(path)
    try:
        return spark.read.option("basePath", base_path).parquet(path)
    except Exception as first_error:  # noqa: BLE001 - retried below, then re-raised
        LOGGER.warning(
            "parquet read failed with basePath=%s, retrying without it: path=%s error=%s",
            base_path,
            path,
            first_error,
        )
        return spark.read.parquet(path)


def read_parquet_paths(
    spark: SparkSession,
    paths: Sequence[str],
    source_column: str = SOURCE_PATH_COLUMN,
) -> DataFrame:
    """Read several parquet prefixes and union them by name.

    Reads one prefix at a time, because there is no option that makes
    ``spark.read.parquet(*paths)`` safe over monthly partition prefixes — the
    loop is the fix rather than a workaround. Given a ``basePath`` that spans
    two layouts it raises "Conflicting directory structures detected"; given no
    ``basePath`` it succeeds and silently drops the partition columns, since
    each supplied path then acts as its own table root. The module docstring has
    the detail. Spark's own advice in the first case is to load the paths
    separately and union them, which is what this does.

    ``allowMissingColumns=True`` is what lets a month that lacks a feature union
    with one that has it — the missing side becomes null, which is exactly what
    the imputation step is for. It also means a *typo* in a feature name unions
    silently as all-null, which is why
    :func:`load_training_matrix` counts the missing features per month and
    :attr:`MLConfig.strict_missing_feature_check` can make that fatal.
    """
    unique = list(dict.fromkeys([p for p in paths if p]))
    if not unique:
        raise DatasetError("no parquet paths supplied")
    frames = [
        read_parquet_path(spark, path).withColumn(source_column, F.lit(path)) for path in unique
    ]
    out = frames[0]
    for frame in frames[1:]:
        out = out.unionByName(frame, allowMissingColumns=True)
    return out


def add_missing_and_select(
    frame: DataFrame,
    features: Sequence[str],
    meta_columns: Sequence[str] = (),
) -> DataFrame:
    """Project to ``meta + features``, creating absent columns as typed nulls.

    Features are cast to ``double`` — every model here consumes a numeric matrix,
    and a feature stored as a string in one month and a long in the next must not
    become two columns after a union. Meta columns are cast to ``string``.

    The cast on *absent* columns is the part that matters. The notebook filled
    absent meta columns with a bare ``F.lit(None)``, which has Spark's ``void``
    type: it survives the union and the ``toPandas`` collection, then fails the
    scored-records write with "Parquet data source does not support void data
    type" — at the end of a training run, after the model was fitted. Every
    column this function produces has a concrete type.

    Meta columns that are absent are still created, so both populations come out
    of here with identical schemas and can be unioned without an alignment pass.
    """
    out = frame
    existing = set(out.columns)
    projection = []
    for column in meta_columns:
        if column in existing:
            projection.append(F.col(column).cast("string").alias(column))
        else:
            projection.append(F.lit(None).cast("string").alias(column))
    for column in features:
        if column in existing:
            projection.append(F.col(column).cast("double").alias(column))
        else:
            projection.append(F.lit(None).cast("double").alias(column))
    return out.select(*projection)


def bounded_to_pandas(
    frame: DataFrame,
    max_rows: int = 0,
    seed: int = 42,
    context: str = "data",
) -> pd.DataFrame:
    """Collect a Spark frame to pandas, uniformly downsampled to ``max_rows``.

    ``max_rows=0`` means no cap, per the repo convention.

    The sample is uniform and the collection is bounded, which are separate
    requirements. Uniformity rules out ``limit`` alone, which takes whichever
    partitions the scheduler reaches first. The notebook got uniformity from
    ``orderBy(F.rand(seed)).limit(n)``, which is correct but sorts the entire
    month across the cluster to keep 130,000 rows of it — the single most
    expensive operation in the training load.

    So this samples first at a fraction slightly above the target, then does the
    random ordering over that much smaller frame. A uniform sample of a uniform
    sample is uniform, so the result is distributed identically to the notebook's
    while the sort input is roughly ``max_rows`` rows instead of all of them. The
    headroom covers the binomial variance of the fraction sample; if it
    undershoots anyway the result is slightly smaller than ``max_rows``, which is
    reported rather than corrected.
    """
    total = frame.count()
    if not max_rows or total <= max_rows:
        LOGGER.info("%s: collecting %d rows", context, total)
        return frame.toPandas()

    fraction = min(1.0, (max_rows / total) * 1.10 + (10.0 / max(total, 1)))
    LOGGER.info(
        "%s: %d rows, sampling fraction %.6f then taking %d",
        context,
        total,
        fraction,
        max_rows,
    )
    sampled = frame.sample(withReplacement=False, fraction=fraction, seed=seed)
    return sampled.orderBy(F.rand(seed)).limit(int(max_rows)).toPandas()


# --------------------------------------------------------------------------
# the fraud population
# --------------------------------------------------------------------------

def _source_from_path_expression():
    """A Spark expression mapping the provenance path to a feed identifier.

    The Spark-side counterpart of :func:`fraud_source_from_path`, and it has to
    agree with it row for row: the two fill in the same column, one during the
    read and one when scored records are read back. So it applies the same rule
    — extract the ``/fraud/<feed>/`` segment, and attribute it to the taxonomy
    key it *uniquely* contains. The ``& ~contains(other)`` guard is what makes
    the uniqueness part exact rather than "whichever key this loop reached
    first", which would silently disagree with the pandas side on a feed
    directory naming both.

    A chain of ``when`` clauses rather than a Python UDF, so the fill-in costs
    nothing and stays inside the JVM. A path matching no known feed yields null,
    which :func:`trust_score_05.ml.taxonomy.category_for` will then reject
    loudly — the right outcome, since it means rows were read from a prefix this
    pipeline does not recognise.
    """
    segment = F.lower(
        F.regexp_extract(F.col(SOURCE_PATH_COLUMN), _FRAUD_SEGMENT_PATTERN.pattern, 1)
    )
    expression = None
    for source in FRAUD_SOURCES:
        condition = segment.contains(source)
        for other in FRAUD_SOURCES:
            if other != source:
                condition = condition & ~segment.contains(other)
        if expression is None:
            expression = F.when(condition, F.lit(source))
        else:
            expression = expression.when(condition, F.lit(source))
    return expression


def load_fraud_customer_ids(spark: SparkSession, cfg: MLConfig) -> DataFrame:
    """Every customer that appears anywhere in the fraud feeds, distinct.

    One column, ``customer_id``, cast to string so the anti-join in
    :func:`load_training_matrix` cannot miss on a type difference between the
    feeds and the non-fraud tables.

    Raises if no fraud data exists. This function's only caller uses it to keep
    fraud out of the training population, so an empty result is not "nothing to
    remove" — it is "the check could not be performed", and continuing produces a
    model trained on the thing it is supposed to detect.
    """
    paths = [p for prefixes in discover_fraud_paths(cfg).values() for p in prefixes]
    if not paths:
        raise DatasetError(
            "no fraud data prefixes were discovered, so fraud customers cannot be "
            "removed from the training population; refusing to train on a "
            "possibly contaminated non-fraud sample"
        )
    frame = read_parquet_paths(spark, paths)
    if CUSTOMER_KEY_COLUMN not in frame.columns:
        raise DatasetError(
            f"fraud data has no {CUSTOMER_KEY_COLUMN} column; cannot enforce "
            "no-leakage training"
        )
    return (
        frame.select(F.col(CUSTOMER_KEY_COLUMN).cast("string").alias(CUSTOMER_KEY_COLUMN))
        .where(F.col(CUSTOMER_KEY_COLUMN).isNotNull())
        .distinct()
    )


def load_fraud_frame(
    spark: SparkSession,
    cfg: MLConfig,
    features: Sequence[str],
    run_prefix: str,
) -> Tuple[DataFrame, Dict[str, Any]]:
    """The labelled fraud population, filtered by timestamp, with ``label=1``.

    Returns the frame and a manifest dict. The manifest is not optional
    bookkeeping: the timestamp filter is the one place in the pipeline where
    labelled rows are discarded, and three different things cause a discard —
    an unparseable timestamp, an absent timestamp, and a timestamp before
    :attr:`MLConfig.fraud_min_timestamp`. The notebook's filter expressed all
    three as one ``where`` clause, so a feed that shipped a month of nulls looked
    the same as a feed that shipped a month of old fraud, and both looked like a
    feed that shipped less fraud.

    Rows whose timestamp is present but unparseable are written to
    ``reports/quarantined_bad_fraud_timestamps.parquet`` and dropped when
    :attr:`MLConfig.quarantine_bad_fraud_timestamps` is set, and raise otherwise.
    Rows with no timestamp at all are dropped and counted — they cannot be placed
    relative to the cutoff, and treating them as recent would put pre-cutoff
    fraud into the evaluation set.
    """
    paths = [p for prefixes in discover_fraud_paths(cfg).values() for p in prefixes]
    if not paths:
        raise DatasetError("no fraud data prefixes were discovered for evaluation")

    frame = read_parquet_paths(spark, paths)
    if FRAUD_TIMESTAMP_COLUMN not in frame.columns:
        raise DatasetError(
            f"fraud data has no {FRAUD_TIMESTAMP_COLUMN} column; the evaluation "
            "population cannot be time-bounded without it"
        )
    if CUSTOMER_KEY_COLUMN not in frame.columns:
        raise DatasetError(f"fraud data has no {CUSTOMER_KEY_COLUMN} column")

    from_path = _source_from_path_expression()
    if "fraud_source" in frame.columns:
        source_column = F.coalesce(F.col("fraud_source").cast("string"), from_path)
    else:
        source_column = from_path
    frame = frame.withColumn("fraud_source", source_column)

    parsed = frame.withColumn(
        "_fraud_timestamp_parsed", F.to_timestamp(F.col(FRAUD_TIMESTAMP_COLUMN))
    )
    manifest: Dict[str, Any] = {"rows_in": parsed.count()}

    unparseable = parsed.where(
        F.col(FRAUD_TIMESTAMP_COLUMN).isNotNull() & F.col("_fraud_timestamp_parsed").isNull()
    )
    bad_count = unparseable.count()
    manifest["rows_unparseable_timestamp"] = bad_count
    if bad_count:
        report_uri = join_uri(run_prefix, "reports", "quarantined_bad_fraud_timestamps.parquet")
        unparseable.drop("_fraud_timestamp_parsed").write.mode("overwrite").parquet(report_uri)
        message = (
            f"{bad_count:,} fraud rows have an unparseable {FRAUD_TIMESTAMP_COLUMN}; "
            f"quarantined to {report_uri}"
        )
        if not cfg.quarantine_bad_fraud_timestamps:
            raise DatasetError(message)
        LOGGER.warning(message)
        manifest["quarantine_uri"] = report_uri

    manifest["rows_null_timestamp"] = parsed.where(
        F.col(FRAUD_TIMESTAMP_COLUMN).isNull()
    ).count()

    kept = parsed.where(
        F.col("_fraud_timestamp_parsed") >= F.to_timestamp(F.lit(cfg.fraud_min_timestamp))
    )
    manifest["fraud_min_timestamp"] = cfg.fraud_min_timestamp
    manifest["rows_kept"] = kept.count()
    manifest["rows_before_min_timestamp"] = (
        manifest["rows_in"]
        - manifest["rows_kept"]
        - manifest["rows_unparseable_timestamp"]
        - manifest["rows_null_timestamp"]
    )
    if not manifest["rows_kept"]:
        raise DatasetError(
            f"no fraud rows have a {FRAUD_TIMESTAMP_COLUMN} at or after "
            f"{cfg.fraud_min_timestamp}; there is nothing to evaluate against"
        )

    meta = list(FRAUD_META_COLUMNS) + [SOURCE_PATH_COLUMN]
    out = add_missing_and_select(kept, features, meta_columns=meta).withColumn(
        LABEL_COLUMN, F.lit(1).cast("int")
    )
    return out, manifest


# --------------------------------------------------------------------------
# the training population
# --------------------------------------------------------------------------

def load_training_matrix(
    spark: SparkSession,
    cfg: MLConfig,
    features: Sequence[str],
    run_prefix: str,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """The non-fraud training sample: one bounded draw per configured month.

    Returns the concatenated pandas frame, the feature names that were actually
    present in at least one month, and the per-month manifest.

    Sampling per month rather than from the union is what keeps the months
    balanced. A union-then-sample draws in proportion to each month's size, so a
    month with three times the traffic contributes three times the rows and the
    model's notion of normal drifts towards it.

    Fraud customers are removed before sampling, not after: removing after would
    make the retained row count depend on how much fraud happened to land in the
    sample, and would leave fewer than ``rows_per_month`` rows for no visible
    reason.
    """
    discovered = discover_training_nonfraud_paths(cfg).require("training non-fraud")
    rows_per_month = cfg.train_rows_per_month
    fraud_ids = load_fraud_customer_ids(spark, cfg)

    frames: List[pd.DataFrame] = []
    manifest_rows: List[Dict[str, Any]] = []
    for window, path in discovered.found.items():
        frame = read_parquet_path(spark, path)
        if CUSTOMER_KEY_COLUMN not in frame.columns:
            raise DatasetError(f"training prefix has no {CUSTOMER_KEY_COLUMN}: {path}")

        before = frame.count()
        deduped = frame.withColumn(
            CUSTOMER_KEY_COLUMN, F.col(CUSTOMER_KEY_COLUMN).cast("string")
        ).join(fraud_ids, on=CUSTOMER_KEY_COLUMN, how="left_anti")
        after = deduped.count()

        missing = [name for name in features if name not in frame.columns]
        if missing and cfg.strict_missing_feature_check:
            raise DatasetError(
                f"training prefix {path} is missing {len(missing)} requested "
                f"features, first 20: {missing[:20]}"
            )
        if missing:
            LOGGER.warning(
                "training prefix %s is missing %d requested features; they will be "
                "all-null for this month",
                path,
                len(missing),
            )

        projected = add_missing_and_select(
            deduped, features, meta_columns=[CUSTOMER_KEY_COLUMN]
        )
        month = bounded_to_pandas(
            projected,
            max_rows=rows_per_month,
            seed=cfg.seed,
            context=f"training sample {window}",
        )
        month["time_window"] = window
        frames.append(month)
        manifest_rows.append(
            {
                "time_window": window,
                "path": path,
                "rows_before_fraud_removal": before,
                "rows_after_fraud_removal": after,
                "fraud_customers_removed": before - after,
                "rows_sampled": len(month),
                "rows_requested": rows_per_month,
                "missing_feature_count": len(missing),
            }
        )

    for window in discovered.missing:
        manifest_rows.append(
            {
                "time_window": window,
                "path": "",
                "rows_before_fraud_removal": 0,
                "rows_after_fraud_removal": 0,
                "fraud_customers_removed": 0,
                "rows_sampled": 0,
                "rows_requested": rows_per_month,
                "missing_feature_count": len(features),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    write_pandas(manifest, join_uri(run_prefix, "reports", "training_manifest.csv"))

    training = pd.concat(frames, ignore_index=True)
    present = [
        name
        for name in features
        if name in training.columns and training[name].notna().any()
    ]
    if not present:
        raise DatasetError(
            f"none of the {len(features)} requested features has a non-null value "
            "anywhere in the training sample"
        )
    return training, present, manifest


# --------------------------------------------------------------------------
# the evaluation populations
# --------------------------------------------------------------------------

def load_scenario_frame(
    spark: SparkSession,
    cfg: MLConfig,
    scenario: str,
    nonfraud_path: str,
    fraud_frame: Any,
    features: Sequence[str],
) -> pd.DataFrame:
    """One evaluation population: a window's non-fraud plus the fixed fraud set.

    ``fraud_frame`` may be the Spark frame from :func:`load_fraud_frame` or an
    already-collected pandas frame. Pass pandas when scoring several scenarios:
    the fraud population is identical for all of them, and collecting it once
    instead of once per scenario removes a full re-read of every fraud partition
    per window.

    Both sides are projected through :func:`add_missing_and_select` with the same
    meta list, so they share a schema and the union needs no alignment pass. The
    notebook projected them with two *different* meta lists and then aligned with
    untyped ``F.lit(None)`` columns, which is where the void-typed columns
    described in :func:`add_missing_and_select` came from.

    ``label`` is 0 for the whole non-fraud side. That is an assumption, not a
    measurement — the non-fraud population is "customers with no fraud record",
    and undetected fraud in it is counted as a false positive. Every precision
    figure the pipeline reports is therefore a lower bound.

    :attr:`MLConfig.max_eval_rows_per_scenario`, when non-zero, caps the
    *non-fraud* side only. The notebook capped the union, which sampled fraud
    rows away at the same rate as everything else: with a few hundred labelled
    cases against a million negatives, a cap of 200,000 discarded roughly four
    fifths of the positives, and ``recall_at_k`` — the metric the best
    configuration is chosen on — was then computed over whichever fifth
    survived. Capping the negatives keeps every positive, at the cost of raising
    the base rate, so precision and lift under a cap are comparable *between
    configurations at the same cap* but are not the absolute numbers a queue
    would see. The default of ``0`` leaves the population whole.
    """
    meta = list(FRAUD_META_COLUMNS) + [SOURCE_PATH_COLUMN]
    nonfraud = read_parquet_path(spark, nonfraud_path).withColumn(
        SOURCE_PATH_COLUMN, F.lit(nonfraud_path)
    )
    nonfraud = add_missing_and_select(nonfraud, features, meta_columns=meta).withColumn(
        LABEL_COLUMN, F.lit(0).cast("int")
    )

    negatives = bounded_to_pandas(
        nonfraud,
        max_rows=cfg.max_eval_rows_per_scenario,
        seed=cfg.seed,
        context=f"scenario {scenario} non-fraud",
    )
    if isinstance(fraud_frame, pd.DataFrame):
        positives = fraud_frame.copy()
    else:
        positives = bounded_to_pandas(
            fraud_frame, max_rows=0, seed=cfg.seed, context=f"scenario {scenario} fraud"
        )
    frame = pd.concat([negatives, positives], ignore_index=True)
    frame["scenario"] = scenario
    return frame
