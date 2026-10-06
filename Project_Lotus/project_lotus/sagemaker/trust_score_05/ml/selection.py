"""Which columns are features, and in what order.

Two jobs, and they are worth keeping apart. *Exclusion* decides whether a column
is eligible to be a feature at all; it is a correctness question and a wrong
answer is a leak or a crash. *Ranking* decides the order eligible features are
offered in, so that ``top_n_features`` means something; it is a heuristic and a
wrong answer costs some accuracy.

Exclusion drops four kinds of column. The join key, because it is an identifier.
Anything in :data:`LABEL_LEAKAGE_COLUMNS`, which is every column that exists only
because a row came from the fraud feed — a model given ``fraud_type`` will
"detect" fraud perfectly and detect nothing at all. Timestamps and dates,
because an absolute instant is not behaviour: a model trained on one month and
scored on the next sees every value shifted by a month and calls the entire
population anomalous. And identifiers and hashes, because a customer number is a
high-cardinality integer that a tree will happily split on.

The identifier and timestamp rules are pattern-based, and patterns
over-match: ``msisdn_device_change_cnt_7d`` contains ``msisdn``,
``cust_days_since_account_created`` contains ``created``. So a carve-out
readmits a column whose name is scope-prefixed *and* contains a behavioural
token. The notebook applied that carve-out to the identifier patterns only, so a
behavioural feature whose name happened to contain ``created`` or ``timestamp``
was dropped unconditionally, and no report said so — the exclusion reports the
notebook wrote were seven empty CSVs. :func:`classify_column` returns a reason
for every column and :func:`classify_columns` assembles them into the report.

Ranking scores a feature by how much of it is present and how spread out it is,
both measured scale-free. The notebook scored ``log1p(variance) * (1 -
null_rate)`` on raw unscaled columns, which ranks by *unit*: the same quantity
recorded in seconds outranks itself recorded in days by a factor of log(3600^2),
and a currency column outranks every ratio in the table. Since the scaler is
chosen downstream, per preprocessing variant, the selection step has no business
being scale-sensitive at all. Here spread is the fraction of a feature's 21
computed quantiles that are distinct, which is invariant under any monotone
transform, so seconds and days rank identically and a column that is 95% zeros
ranks below one that is not.

The notebook called this ``clusterability_stability``, which it never computed —
there is no clustering and no stability anywhere in it. The method's name here is
:data:`METHOD_NAME`, which is what it does. It remains a cheap pre-filter whose
real job is to drop degenerate columns and give a stable ordering; whether the
ordering matters is what the sweep's ``top_n_features`` dimension measures.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "BEHAVIOURAL_TOKENS",
    "EXCLUSION_REASONS",
    "FAMILY_BOOST_TOKENS",
    "FAMILY_BOOST_WEIGHT",
    "ID_PATTERNS",
    "LABEL_LEAKAGE_COLUMNS",
    "METHOD_NAME",
    "NUMERIC_DTYPE_PATTERN",
    "PROFILE_QUANTILES",
    "RANKED_FEATURE_COLUMNS",
    "SCOPE_PREFIXES",
    "TS_PATTERNS",
    "ColumnClassification",
    "SelectionError",
    "candidate_features",
    "classify_column",
    "classify_columns",
    "family_boost",
    "is_numeric_dtype",
    "profile_features",
    "rank_features",
    "select_top_n",
]

LOGGER = logging.getLogger(__name__)

#: The method identifier written into the output path. Named for what the
#: ranking measures, so a reader of ``feature_selection_method=…`` is not misled
#: about what produced the ordering.
METHOD_NAME = "null_rate_and_quantile_spread"


class SelectionError(ValueError):
    """Feature selection could not produce a usable candidate set."""


# --------------------------------------------------------------------------
# exclusion
# --------------------------------------------------------------------------

#: Columns that exist only because a row came from the fraud feed, or that
#: describe the sampling rather than the customer. Present in the fraud
#: population and absent (hence all-null after the union) in the non-fraud one,
#: which makes every one of them a perfect label proxy.
LABEL_LEAKAGE_COLUMNS = frozenset(
    """
    testing_label testing_label_name testing_split testing_version
    testing_input_feature_set testing_gather_pipeline_version testing_gathered_at_utc
    fraud_type fraud_category fraud_category_short_label fraud_timestamp
    fraud_event_key fraud_case_natural_key fraud_case_natural_key_columns
    fraud_event_occurrence_seq fraud_case_transaction_count is_duplicate_fraud_case
    deduplication_policy transaction_level_rows_represented
    source source_name source_version fraud_source fraud_source_key
    fraud_source_name fraud_source_version partner industry phone_type
    label scenario hyperparam_id anomaly_score rank
    _sample_source_prefix _sample_source_files_used _sample_label _sample_source_root
    """.split()
)

#: Identifier-shaped names. ``hash`` and the network identifiers are here
#: because they are high-cardinality integers that a tree will split on happily
#: and that mean nothing outside the row they came from.
ID_PATTERNS: Tuple[str, ...] = (
    r"(^|_)id$",
    r"(^|_)key$",
    r"hash",
    r"reference_id",
    r"phone",
    r"msisdn",
    r"imsi",
    r"imei",
)

#: Absolute-time names. Excluded because the value shifts with the calendar:
#: a model fitted on one month sees the next month's timestamps as out of range
#: and scores the whole population as anomalous. Elapsed-time features
#: (``days_since_…``) are relative and are kept by the carve-out below.
TS_PATTERNS: Tuple[str, ...] = (
    r"_ts$",
    r"timestamp",
    r"created",
    r"updated",
    r"_date$",
    r"(^|_)dt$",
)

#: Tokens that mark a name as describing behaviour over a window rather than
#: identifying or timestamping anything.
BEHAVIOURAL_TOKENS: Tuple[str, ...] = (
    "cnt",
    "days",
    "entropy",
    "velocity",
    "interval",
    "distinct",
    "ratio",
    "frac",
    "stddev",
    "slope",
    "burst",
)

#: Name prefixes that mark the scope a feature was computed at.
SCOPE_PREFIXES: Tuple[str, ...] = ("msisdn_", "cust_", "customer_", "acct_", "account_")

#: Every value :attr:`ColumnClassification.reason` can take. Fixed so the
#: exclusion report has a stable set of groups even when a group is empty.
EXCLUSION_REASONS: Tuple[str, ...] = (
    "",
    "join_key",
    "label_leakage",
    "timestamp_pattern",
    "id_pattern",
    "not_numeric",
    "provenance",
)

#: Spark dtype strings that denote a scalar number.
#:
#: Anchored at both ends, with an optional parenthesised tail for the one
#: parameterised type that reaches here (``decimal(18,4)``). A substring match
#: would be simpler and is what the notebook used, but Spark spells its complex
#: types by naming their element type — ``array<int>``, ``map<string,bigint>``,
#: ``struct<retries:int>`` — so ``int`` as a substring admits all three. Each
#: then reaches ``F.col(name).cast("double")`` in
#: :func:`profile_features`, which yields all nulls for an array or a struct
#: rather than raising, so the column profiles as 100% null, is dropped for its
#: null rate, and is reported as a data-quality problem in a feature that never
#: existed.
#:
#: The trailing ``\d*`` accepts the numpy/pandas spellings — ``float64``,
#: ``int32``, ``uint8`` — as well as Spark's. Spark never emits them, but the
#: schema inventory :mod:`trust_score_05.ml.discovery` writes is a CSV that a
#: notebook reads back with pandas, and a dtype column round-tripped through a
#: pandas frame comes back in pandas' spelling.
NUMERIC_DTYPE_PATTERN = re.compile(
    r"^(u?tinyint|u?smallint|u?int|u?integer|u?bigint|long|short|float|double|real"
    r"|decimal|numeric)\d*(\s*\([^)]*\))?$",
    re.IGNORECASE,
)

_ID_RE = tuple(re.compile(pattern) for pattern in ID_PATTERNS)
_TS_RE = tuple(re.compile(pattern) for pattern in TS_PATTERNS)


def is_numeric_dtype(dtype: Any) -> bool:
    """Whether a dtype string denotes a scalar number.

    A one-line wrapper over :data:`NUMERIC_DTYPE_PATTERN` so that every caller
    coerces and strips the same way. Three modules ask this question — this one,
    :mod:`trust_score_05.ml.discovery` and
    :mod:`trust_score_05.ml.jobs.selection_job` — and when the pattern was
    matched as a substring they could each get away with matching it slightly
    differently. Now that it is anchored they cannot: a caller that forgets to
    strip excludes a numeric column, and one that forgets to coerce raises on a
    non-string dtype object.
    """
    return bool(NUMERIC_DTYPE_PATTERN.search(str(dtype).strip()))


@dataclass(frozen=True)
class ColumnClassification:
    """Why one column is or is not eligible to be a feature.

    ``reason`` is ``""`` when the column is eligible. ``readmitted`` records that
    a pattern matched but the behavioural carve-out kept the column anyway, which
    is the decision most worth auditing: it is the only place a name-based
    exclusion is overridden.
    """

    name: str
    dtype: str
    eligible: bool
    reason: str
    readmitted: bool = False


def _is_behavioural(lowered: str) -> bool:
    """Scope-prefixed and containing a behavioural token.

    Both conditions are required. A bare ``distinct_cnt`` with no scope prefix is
    not a feature this pipeline produces, so requiring the prefix costs nothing
    and stops the carve-out from readmitting arbitrary upstream columns whose
    names happen to contain ``ratio``.
    """
    if not lowered.startswith(SCOPE_PREFIXES):
        return False
    return any(token in lowered for token in BEHAVIOURAL_TOKENS)


def classify_column(
    name: str,
    dtype: str = "double",
    join_key: str = "customer_id",
    provenance_columns: Iterable[str] = (),
) -> ColumnClassification:
    """Decide one column's eligibility, and say why.

    Checked in a fixed order, most decisive first: the join key, then the
    leakage list, then provenance, then the name patterns, then the dtype. The
    order matters for the report rather than the outcome — a column excluded for
    several reasons is attributed to the first, so ``fraud_timestamp`` is
    reported as leakage rather than as a timestamp, which is the more useful
    thing to know about it.

    The behavioural carve-out applies to the timestamp patterns as well as the
    identifier patterns. In the notebook it applied only to the identifiers,
    which meant ``cust_days_since_account_created`` — a relative, perfectly
    usable feature — was dropped for containing ``created``, along with every
    other elapsed-time feature naming the event it measured from.
    """
    text = str(name)
    lowered = text.lower()

    if lowered == str(join_key).lower():
        return ColumnClassification(text, dtype, False, "join_key")
    if text in LABEL_LEAKAGE_COLUMNS or lowered in LABEL_LEAKAGE_COLUMNS:
        return ColumnClassification(text, dtype, False, "label_leakage")
    if text in set(provenance_columns) or lowered.startswith("__"):
        return ColumnClassification(text, dtype, False, "provenance")

    behavioural = _is_behavioural(lowered)
    if any(pattern.search(lowered) for pattern in _TS_RE):
        if not behavioural:
            return ColumnClassification(text, dtype, False, "timestamp_pattern")
        readmitted = True
    elif any(pattern.search(lowered) for pattern in _ID_RE):
        if not behavioural:
            return ColumnClassification(text, dtype, False, "id_pattern")
        readmitted = True
    else:
        readmitted = False

    if not is_numeric_dtype(dtype):
        return ColumnClassification(text, dtype, False, "not_numeric", readmitted)
    return ColumnClassification(text, dtype, True, "", readmitted)


def classify_columns(
    dtypes: Sequence[Tuple[str, str]],
    join_key: str = "customer_id",
    provenance_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Classify a Spark ``df.dtypes`` list into a report frame.

    Columns: ``column``, ``dtype``, ``eligible``, ``reason``, ``readmitted``.
    This frame is the source of the per-reason CSVs that
    :mod:`trust_score_05.ml.discovery` writes, replacing the notebook's seven
    empty placeholder files.
    """
    rows = [
        {
            "column": item.name,
            "dtype": item.dtype,
            "eligible": item.eligible,
            "reason": item.reason,
            "readmitted": item.readmitted,
        }
        for item in (
            classify_column(name, dtype, join_key, provenance_columns)
            for name, dtype in dtypes
        )
    ]
    frame = pd.DataFrame(rows, columns=["column", "dtype", "eligible", "reason", "readmitted"])
    return frame.sort_values(["eligible", "reason", "column"]).reset_index(drop=True)


def candidate_features(
    dtypes: Sequence[Tuple[str, str]],
    join_key: str = "customer_id",
    provenance_columns: Iterable[str] = (),
) -> Tuple[List[str], pd.DataFrame]:
    """The eligible feature names, and the full classification report.

    Raises when nothing is eligible. That is not a degenerate case to be logged:
    it means the schema is not the one this pipeline was written against, and
    every downstream stage would produce an empty matrix and a meaningless
    metric.
    """
    report = classify_columns(dtypes, join_key, provenance_columns)
    features = report.loc[report["eligible"], "column"].tolist()
    if not features:
        raise SelectionError(
            f"no eligible numeric features among {len(report)} columns; "
            f"exclusion reasons: {report['reason'].value_counts().to_dict()}"
        )
    return features, report


# --------------------------------------------------------------------------
# profiling
# --------------------------------------------------------------------------

#: The quantile probabilities used for the spread measure. Twenty-one points,
#: evenly spaced, so ties among them are a direct reading of how concentrated a
#: feature's mass is: a column that is 95% zeros has its first twenty quantiles
#: all equal to zero and scores 1/20.
PROFILE_QUANTILES: Tuple[float, ...] = tuple(round(i / 20.0, 3) for i in range(21))

PROFILE_COLUMNS: Tuple[str, ...] = (
    "feature",
    "n_rows",
    "null_rate",
    "variance",
    "quantile_spread",
    "n_distinct_quantiles",
)


def profile_features(
    frame: Any,
    features: Sequence[str],
    sample_rows: int = 0,
    seed: int = 42,
    relative_error: float = 0.01,
) -> pd.DataFrame:
    """Null rate, variance and quantile spread for each candidate feature.

    ``frame`` is a Spark DataFrame. Two passes: one aggregate for the null rates
    and variances, and one ``approxQuantile`` over every feature at once for the
    spread. ``approxQuantile`` ignores nulls, which is what we want — the spread
    should describe the values that exist, with their scarcity accounted for
    separately by the null rate.

    Profiling runs on the *union* of the training months, deliberately. A feature
    that only exists in the most recent month is null for the rest of the
    training population, and that high null rate is the honest description of
    what the model will be fitted on, not an artifact.

    ``sample_rows`` bounds the profiling cost; ``0`` profiles everything. The
    sample is taken with :func:`~trust_score_05.ml.datasets.bounded_to_pandas`'s
    strategy — a uniform fraction sample rather than a global sort — for the
    reason given there.
    """
    from pyspark.sql import functions as F

    feature_list = [str(name) for name in features]
    if not feature_list:
        raise SelectionError("no features to profile")

    working = frame
    total = working.count()
    if not total:
        raise SelectionError("the profiling population has no rows")
    if sample_rows and total > sample_rows:
        fraction = min(1.0, (sample_rows / total) * 1.10 + (10.0 / total))
        working = working.sample(withReplacement=False, fraction=fraction, seed=seed)
        LOGGER.info(
            "profiling on a sample: %d of %d rows (fraction %.6f)", sample_rows, total, fraction
        )

    working = working.select(
        *[F.col(name).cast("double").alias(name) for name in feature_list]
    ).cache()
    try:
        profiled_rows = working.count()
        aggregates = []
        for name in feature_list:
            column = F.col(name)
            aggregates.append(
                F.mean(
                    F.when(column.isNull() | F.isnan(column), 1.0).otherwise(0.0)
                ).alias(f"{name}__null_rate")
            )
            aggregates.append(F.variance(column).alias(f"{name}__variance"))
        summary = working.agg(*aggregates).toPandas().iloc[0].to_dict()

        quantiles = working.approxQuantile(
            feature_list, list(PROFILE_QUANTILES), relative_error
        )
    finally:
        working.unpersist()

    rows = []
    for index, name in enumerate(feature_list):
        null_rate = _finite(summary.get(f"{name}__null_rate"), default=1.0)
        variance = _finite(summary.get(f"{name}__variance"), default=0.0)
        observed = quantiles[index] if index < len(quantiles) else []
        distinct = len({round(float(value), 12) for value in observed if value is not None})
        spread = (distinct - 1) / (len(PROFILE_QUANTILES) - 1) if distinct else 0.0
        rows.append(
            {
                "feature": name,
                "n_rows": profiled_rows,
                "null_rate": null_rate,
                "variance": variance,
                "quantile_spread": spread,
                "n_distinct_quantiles": distinct,
            }
        )
    return pd.DataFrame(rows, columns=list(PROFILE_COLUMNS))


def _finite(value: Any, default: float) -> float:
    """A float, with ``None`` and non-finite values replaced by ``default``.

    ``default`` differs by statistic and the difference matters. An absent null
    rate defaults to 1.0 — "assume the worst", so an unmeasurable column is
    dropped — while an absent variance defaults to 0.0, which also drops it. The
    notebook used ``float(x or 0.0)`` for both, so an unmeasurable null rate
    became "no nulls at all" and the column was kept.
    """
    if value is None:
        return float(default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------

#: Per-model token preferences. A model whose notion of an outlier is a distance
#: in a shared space benefits from features that count and compare; a model
#: whose notion is a univariate tail benefits from features that record extremes.
#: The weight is small on purpose — it reorders near-ties, it does not overrule
#: a real difference in spread or completeness.
FAMILY_BOOST_TOKENS: Mapping[str, Tuple[str, ...]] = {
    "kmeans": ("cnt", "velocity", "ratio", "distinct"),
    "weighted_kmeans": ("cnt", "velocity", "ratio", "distinct"),
    "hdbscan": ("cnt", "velocity", "ratio", "distinct"),
    "ecod": ("max", "peak", "velocity"),
    "copod": ("max", "peak", "velocity"),
    "hbos": ("max", "peak", "velocity"),
}

FAMILY_BOOST_WEIGHT = 1.10

RANKED_FEATURE_COLUMNS: Tuple[str, ...] = (
    "rank",
    "feature",
    "null_rate",
    "variance",
    "quantile_spread",
    "n_distinct_quantiles",
    "family_boost",
    "score",
    "keep",
    "drop_reason",
)

_BOOST_CACHE: Dict[str, Tuple[Any, ...]] = {}


def family_boost(feature: str, model: str) -> float:
    """The multiplier a model's token preferences give one feature name.

    Tokens are matched on word boundaries against the lowercased name. The
    notebook used ``token in feature_name`` on the raw name, so the ``ecod`` and
    ``hbos`` preference for ``over`` — meant for ``…_over_…`` ratio features —
    also fired on ``overall``, ``recovery`` and ``leftover``, and ``max`` fired
    inside ``max_gap_days`` and ``proximax_score`` alike. ``over`` is not in the
    token list here; the ratio features it was aimed at are matched by ``ratio``.
    """
    tokens = FAMILY_BOOST_TOKENS.get(str(model).lower())
    if not tokens:
        return 1.0
    key = str(model).lower()
    if key not in _BOOST_CACHE:
        _BOOST_CACHE[key] = tuple(
            re.compile(rf"(^|_){re.escape(token)}(_|$)") for token in tokens
        )
    lowered = str(feature).lower()
    return FAMILY_BOOST_WEIGHT if any(p.search(lowered) for p in _BOOST_CACHE[key]) else 1.0


def rank_features(
    profile: pd.DataFrame,
    model: str,
    max_null_rate: float = 0.999,
    min_variance: float = 1e-12,
) -> pd.DataFrame:
    """Order the profiled features, with a keep flag and a drop reason.

    ``score = (1 - null_rate) * quantile_spread * family_boost``. Every factor is
    in [0, 1] except the boost, so the score is comparable across features,
    across models and across runs — the notebook's ``log1p(variance)`` term was
    none of those things, since its magnitude was set by each column's unit.

    ``variance`` is still computed and still gates: a variance at or below
    ``min_variance`` means the column is constant, which is a fact about the
    column and not about its scale. The gate is expressed as "constant", and the
    scale-sensitive part of the old score is gone from the ordering.

    Ties break on feature name, so two runs over the same data produce the same
    order. The notebook broke ties on variance, which reintroduced the unit
    dependence it was trying to leave behind and left genuinely identical
    features in arbitrary order.
    """
    if profile.empty:
        raise SelectionError("the feature profile is empty")

    frame = profile.copy()
    for column in ("null_rate", "variance", "quantile_spread"):
        if column not in frame.columns:
            raise SelectionError(f"the feature profile has no {column!r} column")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["family_boost"] = [family_boost(name, model) for name in frame["feature"]]
    frame["score"] = (
        (1.0 - frame["null_rate"].clip(0.0, 1.0))
        * frame["quantile_spread"].clip(0.0, 1.0)
        * frame["family_boost"]
    )

    reasons = []
    for _, row in frame.iterrows():
        if not np.isfinite(row["null_rate"]) or row["null_rate"] > max_null_rate:
            reasons.append("null_rate_above_max")
        elif not np.isfinite(row["variance"]) or row["variance"] <= min_variance:
            reasons.append("constant")
        elif row["quantile_spread"] <= 0.0:
            reasons.append("no_quantile_spread")
        else:
            reasons.append("")
    frame["drop_reason"] = reasons
    frame["keep"] = frame["drop_reason"].eq("")

    frame = frame.sort_values(
        ["keep", "score", "feature"], ascending=[False, False, True]
    ).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    return frame[[column for column in RANKED_FEATURE_COLUMNS if column in frame.columns]]


def select_top_n(
    ranked: pd.DataFrame,
    counts: Sequence[int],
    min_features: int = 2,
) -> Dict[int, List[str]]:
    """The kept features truncated to each requested count.

    A requested count larger than the number of kept features is *dropped*, not
    clamped. Clamping is what the notebook did, by slicing a shorter list: with
    120 kept features, ``top_n_features`` of 125, 150 and 200 all produced the
    identical 120-feature list, and the sweep then trained three identical models
    under three separate run prefixes and listed them as three distinct
    configurations on the leaderboard. The largest requested count that the data
    supports is kept, so the "all available features" arm still exists.

    Counts below ``min_features`` are refused: a one-feature anomaly model is a
    threshold, and several of the estimators here cannot be fitted at all on a
    single column.
    """
    kept = ranked.loc[ranked["keep"], "feature"].tolist() if "keep" in ranked else []
    if len(kept) < min_features:
        raise SelectionError(
            f"only {len(kept)} features survived profiling, fewer than the "
            f"minimum of {min_features}"
        )

    requested = sorted({int(count) for count in counts})
    too_small = [count for count in requested if count < min_features]
    if too_small:
        LOGGER.warning("ignoring top_n_features values below %d: %s", min_features, too_small)

    usable = [count for count in requested if min_features <= count <= len(kept)]
    oversized = [count for count in requested if count > len(kept)]
    if oversized:
        LOGGER.warning(
            "%d requested feature counts exceed the %d available and were dropped: %s",
            len(oversized),
            len(kept),
            oversized,
        )
        if len(kept) not in usable:
            usable.append(len(kept))

    if not usable:
        raise SelectionError(
            f"none of the requested feature counts {requested} is usable against "
            f"{len(kept)} kept features"
        )
    return {count: kept[:count] for count in sorted(usable)}


def selection_manifest(
    model: str,
    method: str,
    report: pd.DataFrame,
    ranked: pd.DataFrame,
    selections: Mapping[int, Sequence[str]],
    profiled_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """The JSON summary written next to the selected feature lists.

    Records the exclusion reasons and the drop reasons as counts, so a later
    reader can tell a schema change ("forty columns became ``not_numeric``") from
    a data change ("forty features became ``constant``") without re-reading the
    reports.
    """
    return {
        "model": str(model),
        "method": str(method),
        "columns_seen": int(len(report)),
        "columns_eligible": int(report["eligible"].sum()) if len(report) else 0,
        "columns_readmitted_by_carve_out": (
            int(report["readmitted"].sum()) if len(report) else 0
        ),
        "exclusion_reasons": (
            report.loc[~report["eligible"], "reason"].value_counts().to_dict()
            if len(report)
            else {}
        ),
        "features_profiled": int(len(ranked)),
        "features_kept": int(ranked["keep"].sum()) if len(ranked) else 0,
        "drop_reasons": (
            ranked.loc[~ranked["keep"], "drop_reason"].value_counts().to_dict()
            if len(ranked)
            else {}
        ),
        "profiled_rows": profiled_rows,
        "selected_counts": {int(count): len(names) for count, names in selections.items()},
    }
