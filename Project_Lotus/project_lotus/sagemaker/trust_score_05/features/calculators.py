"""One function per feature family.

Every calculator has the same signature — a scoped event frame, the entity
columns, an event family, a name prefix, a config — and returns a frame keyed on
the entity columns with one column per feature. They are independent: none reads
another's output, so a family can be dropped from the assembly without
disturbing the rest.

The families, and what each is trying to see:

``frequency``
    How much happened, in each window. The plain count, a partner-weighted
    count, and a binary "any at all" flag. The flag matters separately from the
    count because for rare families the distinction between zero and one is the
    whole signal.
``recency``
    How long ago the last one was, and the first. A device change yesterday and
    a device change last year are the same count and very different facts.
``periodicity``
    The rhythm of the gaps between events — mean, median, spread, and the
    coefficient of variation. Regular activity and bursty activity can produce
    identical counts.
``burstness``
    Concentration in time: the peak month against the average month, and the
    short-window counts as a fraction of the long ones. A ratio near one means
    everything happened recently.
``temporal entropy``
    Whether activity is spread across the hours, days and months or concentrated
    in a few. Automated traffic is low-entropy in a way human traffic is not.
``velocity``
    Two views of speed. How long the last *k* events took, and the reciprocal of
    the mean gap.
``recency-weighted intensity``
    A single number that counts every event but discounts old ones
    exponentially, so a burst last week outweighs the same burst last quarter
    without either being windowed away.
``cumulative slope``
    Is the rate of events rising or falling across the observed period.
``state transitions``
    For each ordered pair of event types, how long after one comes the other.
    This is where the sequences live — a SIM change shortly after a device
    change is a different story than either alone.
``stability``
    Tenure and breadth from the snapshot alone, with no events involved: how
    long we have known this entity, how long since its current assignment
    began, how many accounts and numbers it holds.

Nulls are meaningful throughout and are never filled here. An entity with one
event has no inter-arrival gap, and a null in ``interval_mean`` says exactly
that. Deciding what a model should do about it is the preprocessing stage's job,
and it can only decide if the null reaches it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from trust_score_05.features.config import FeatureConfig, scope_prefix
from trust_score_05.features.expressions import (
    entropy_over_dimension,
    feature_base,
    feature_columns,
    feature_token,
    filtered_events,
    has_col,
    lookback_condition,
    next_event_window,
    safe_divide,
    safe_join,
)

__all__ = [
    "calculate_burstness_features",
    "calculate_cumulative_event_slope",
    "calculate_event_diversity_features",
    "calculate_event_frequency_features",
    "calculate_event_recency_features",
    "calculate_periodicity_features",
    "calculate_recency_weighted_intensity",
    "calculate_stability_features",
    "calculate_state_transition_features",
    "calculate_temporal_entropy_features",
    "calculate_time_to_k_events",
]


def calculate_event_frequency_features(
    scoped_events_df: DataFrame,
    entity_cols: Sequence[str],
    event_family: str,
    prefix: str,
    cfg: FeatureConfig = FeatureConfig(),
) -> DataFrame:
    """Counts, weighted counts and any-flags per lookback window.

    The weight column is used when it is present and falls back to 1.0 when it
    is not, so the same function serves both the carrier families (where every
    event weighs the same) and the EnStream family (where a partner that calls
    constantly is down-weighted so it cannot drown out the rest).
    """
    base = feature_base(scoped_events_df, entity_cols)
    filtered = filtered_events(scoped_events_df, event_family, cfg)

    weight_expr = (
        F.coalesce(F.col(cfg.event_weight_col).cast("double"), F.lit(1.0))
        if has_col(filtered, cfg.event_weight_col)
        else F.lit(1.0)
    )

    agg_exprs: List[Column] = []
    for days in cfg.lookback_days:
        condition = lookback_condition(days, cfg)
        agg_exprs.extend(
            [
                F.sum(F.when(condition, F.lit(1)).otherwise(F.lit(0))).alias(
                    f"{prefix}_{event_family}_cnt_{days}d"
                ),
                F.sum(F.when(condition, weight_expr).otherwise(F.lit(0.0))).alias(
                    f"{prefix}_{event_family}_weighted_cnt_{days}d"
                ),
                F.max(F.when(condition, F.lit(1)).otherwise(F.lit(0))).alias(
                    f"{prefix}_{event_family}_any_{days}d"
                ),
            ]
        )

    features = filtered.groupBy(*list(entity_cols)).agg(*agg_exprs)
    return safe_join(base, [features], entity_cols)


def calculate_event_recency_features(
    scoped_events_df: DataFrame,
    entity_cols: Sequence[str],
    event_family: str,
    prefix: str,
    cfg: FeatureConfig = FeatureConfig(),
) -> DataFrame:
    """Days since the most recent and the earliest event, and the total count.

    The total count is unwindowed on purpose. Every other count in the package
    is bounded by a lookback, and a lifetime count is the one thing those cannot
    express — though note that when the event scan is pruned (the default) the
    "lifetime" it sees is the pruned window plus the buffer, not the entity's
    whole history. That is a deliberate trade of completeness for a bounded
    scan; ``prune_event_window=False`` buys the true lifetime at the cost of
    reading everything.
    """
    base = feature_base(scoped_events_df, entity_cols)
    filtered = filtered_events(scoped_events_df, event_family, cfg)

    features = (
        filtered.groupBy(*list(entity_cols))
        .agg(
            F.max(F.col(cfg.reference_ts_col)).alias("__reference_ts"),
            F.max(F.col(cfg.event_ts_col)).alias("__last_event_ts"),
            F.min(F.col(cfg.event_ts_col)).alias("__first_event_ts"),
            F.count(F.col(cfg.event_ts_col)).alias(
                f"{prefix}_{event_family}_total_cnt"
            ),
        )
        .withColumn(
            f"{prefix}_{event_family}_days_since_last",
            F.datediff(
                F.to_date(F.col("__reference_ts")), F.to_date(F.col("__last_event_ts"))
            ),
        )
        .withColumn(
            f"{prefix}_{event_family}_days_since_first",
            F.datediff(
                F.to_date(F.col("__reference_ts")), F.to_date(F.col("__first_event_ts"))
            ),
        )
        .drop("__reference_ts", "__last_event_ts", "__first_event_ts")
    )

    return safe_join(base, [features], entity_cols)


def calculate_event_diversity_features(
    scoped_events_df: DataFrame,
    entity_cols: Sequence[str],
    event_family: str,
    prefix: str,
    dimension_cols: Sequence[str],
    cfg: FeatureConfig = FeatureConfig(),
) -> DataFrame:
    """Distinct and non-null counts of each metadata dimension.

    Only meaningful for a family whose events carry metadata, which in practice
    means EnStream. A dimension absent from the frame is skipped rather than
    producing a column of nulls, because the absence is a property of the feed
    and not of the entity.
    """
    base = feature_base(scoped_events_df, entity_cols)
    filtered = filtered_events(scoped_events_df, event_family, cfg)

    feature_frames: List[DataFrame] = []
    for dimension in dimension_cols:
        if not has_col(filtered, dimension):
            continue
        feature_frames.append(
            filtered.groupBy(*list(entity_cols)).agg(
                F.countDistinct(F.col(dimension)).alias(
                    f"{prefix}_{event_family}_{dimension}_distinct_cnt"
                ),
                F.count(F.col(dimension)).alias(
                    f"{prefix}_{event_family}_{dimension}_non_null_cnt"
                ),
            )
        )

    return safe_join(base, feature_frames, entity_cols)


def calculate_periodicity_features(
    scoped_events_df: DataFrame,
    entity_cols: Sequence[str],
    event_family: str,
    prefix: str,
    cfg: FeatureConfig = FeatureConfig(),
) -> DataFrame:
    """Statistics of the gaps between consecutive events.

    The coefficient of variation is the useful one: the standard deviation of
    the gaps divided by their mean, which is scale-free, so a customer who acts
    monthly and one who acts daily are comparable on how *regular* they are
    rather than on how often.

    Entities with fewer than two events are absent from the result and pick up
    nulls at assembly, which is the honest answer — one event has no gap.
    """
    filtered = filtered_events(scoped_events_df, event_family, cfg)
    order_window = Window.partitionBy(*list(entity_cols)).orderBy(
        F.col(cfg.event_ts_col)
    )

    intervals = (
        filtered.withColumn("__prev_event_ts", F.lag(cfg.event_ts_col).over(order_window))
        .withColumn(
            "__interval_days",
            F.datediff(F.col(cfg.event_ts_col), F.col("__prev_event_ts")),
        )
        .filter(F.col("__interval_days").isNotNull())
    )

    stub = f"{prefix}_{event_family}"
    return (
        intervals.groupBy(*list(entity_cols))
        .agg(
            F.avg("__interval_days").alias(f"{stub}_interval_mean"),
            F.expr("percentile_approx(__interval_days, 0.5)").alias(
                f"{stub}_interval_median"
            ),
            F.min("__interval_days").alias(f"{stub}_interval_min"),
            F.max("__interval_days").alias(f"{stub}_interval_max"),
            F.stddev_samp("__interval_days").alias(f"{stub}_interval_stddev"),
        )
        .withColumn(
            f"{stub}_interval_cv",
            safe_divide(
                F.col(f"{stub}_interval_stddev"), F.col(f"{stub}_interval_mean")
            ),
        )
        .withColumn(
            f"{stub}_inverse_mean_interarrival_velocity",
            safe_divide(F.lit(1.0), F.col(f"{stub}_interval_mean")),
        )
    )


def calculate_burstness_features(
    scoped_events_df: DataFrame,
    entity_cols: Sequence[str],
    event_family: str,
    prefix: str,
    cfg: FeatureConfig = FeatureConfig(),
) -> DataFrame:
    """How concentrated the activity is, monthly and across windows.

    Two independent views. The monthly view compares the busiest calendar month
    to the average one, so an entity whose year of activity all happened in
    March scores high however low its total is. The window view is a set of
    short-over-long count ratios — one day over three, seven over thirty, thirty
    over ninety — where a value near one means the long window contains nothing
    the short window does not, which is to say everything happened just now.

    The ratios need the frequency counts, so this calls
    :func:`calculate_event_frequency_features` and drops the borrowed columns
    afterwards rather than returning them twice.

    Which ratios are computed is ``cfg.ratio_windows``, and ``FeatureConfig``
    validates that every window named there is also in ``lookback_days``. The
    notebook version hardcoded 1/3, 7/30 and 30/90 while taking the windows from
    config, so a caller who narrowed ``lookback_days`` got an unresolved-column
    failure from inside a helper rather than an error about their own argument.
    """
    filtered = filtered_events(scoped_events_df, event_family, cfg).withColumn(
        "__event_month", F.date_trunc("month", F.col(cfg.event_ts_col))
    )
    stub = f"{prefix}_{event_family}"

    monthly_counts = filtered.groupBy(*list(entity_cols), "__event_month").agg(
        F.count("*").alias("__monthly_cnt")
    )
    burstness = (
        monthly_counts.groupBy(*list(entity_cols))
        .agg(
            F.max("__monthly_cnt").alias(f"{stub}_max_monthly_cnt"),
            F.avg("__monthly_cnt").alias(f"{stub}_avg_monthly_cnt"),
            F.stddev_samp("__monthly_cnt").alias(f"{stub}_monthly_cnt_std"),
            F.count("*").alias(f"{stub}_active_months"),
        )
        .withColumn(
            f"{stub}_monthly_peak_to_avg",
            safe_divide(
                F.col(f"{stub}_max_monthly_cnt"), F.col(f"{stub}_avg_monthly_cnt")
            ),
        )
    )

    frequency = calculate_event_frequency_features(
        scoped_events_df,
        entity_cols=entity_cols,
        event_family=event_family,
        prefix=prefix,
        cfg=cfg,
    )
    borrowed_cols = feature_columns(frequency, entity_cols)

    result = burstness.join(frequency, list(entity_cols), "left")
    for numerator, denominator in cfg.ratio_windows:
        result = result.withColumn(
            f"{stub}_cnt_{numerator}d_over_{denominator}d",
            safe_divide(
                F.col(f"{stub}_cnt_{numerator}d"), F.col(f"{stub}_cnt_{denominator}d")
            ),
        )
    return result.drop(*borrowed_cols)


def calculate_temporal_entropy_features(
    scoped_events_df: DataFrame,
    entity_cols: Sequence[str],
    event_family: str,
    prefix: str,
    cfg: FeatureConfig = FeatureConfig(),
) -> DataFrame:
    """Entropy of the hour, weekday and month an event happened in.

    Low hour-entropy means the events all happen at the same time of day, which
    is what a scheduled job looks like and not what a person looks like. Low
    month-entropy means the activity is confined to part of the year, which is
    the burstness signal seen a different way.
    """
    filtered = filtered_events(scoped_events_df, event_family, cfg)
    enriched = (
        filtered.withColumn("__event_hour", F.hour(F.col(cfg.event_ts_col)))
        .withColumn("__event_day_of_week", F.dayofweek(F.col(cfg.event_ts_col)))
        .withColumn("__event_month", F.month(F.col(cfg.event_ts_col)))
    )

    feature_frames = [
        entropy_over_dimension(
            enriched,
            entity_cols,
            f"__{dimension}",
            f"{prefix}_{event_family}_{dimension}_entropy",
        )
        for dimension in ("event_hour", "event_day_of_week", "event_month")
    ]
    return safe_join(feature_base(filtered, entity_cols), feature_frames, entity_cols)


def calculate_time_to_k_events(
    scoped_events_df: DataFrame,
    entity_cols: Sequence[str],
    event_family: str,
    prefix: str,
    k: int,
    cfg: FeatureConfig = FeatureConfig(),
) -> DataFrame:
    """How many days the most recent ``k`` events spanned, and the rate.

    Deliberately null for an entity with fewer than ``k`` events rather than
    computed over however many there are. "Three device changes in two days" and
    "one device change" are not the same shape of fact, and averaging the second
    into the first scale would be an invention.
    """
    filtered = filtered_events(scoped_events_df, event_family, cfg)
    ordered = Window.partitionBy(*list(entity_cols)).orderBy(
        F.col(cfg.event_ts_col).desc()
    )
    latest_k = filtered.withColumn("__rn", F.row_number().over(ordered)).filter(
        F.col("__rn") <= k
    )

    stub = f"{prefix}_{event_family}"
    span_col = f"{stub}_days_for_last_{k}_events"
    return (
        latest_k.groupBy(*list(entity_cols))
        .agg(
            F.count("*").alias("__k_event_cnt"),
            F.max(cfg.event_ts_col).alias("__latest_event_ts"),
            F.min(cfg.event_ts_col).alias("__kth_event_ts"),
        )
        .withColumn(
            span_col,
            F.when(
                F.col("__k_event_cnt") >= F.lit(k),
                F.datediff(
                    F.to_date(F.col("__latest_event_ts")),
                    F.to_date(F.col("__kth_event_ts")),
                ),
            ),
        )
        .withColumn(
            f"{stub}_k_event_velocity_{k}",
            safe_divide(F.lit(float(k)), F.col(span_col)),
        )
        .drop("__k_event_cnt", "__latest_event_ts", "__kth_event_ts")
    )


def calculate_recency_weighted_intensity(
    scoped_events_df: DataFrame,
    entity_cols: Sequence[str],
    event_family: str,
    prefix: str,
    cfg: FeatureConfig = FeatureConfig(),
    half_life_days: Optional[float] = None,
) -> DataFrame:
    """Every event counted once, discounted exponentially by its age.

    With the default seven-day half life, an event today contributes 1, one a
    week ago contributes a half, one a month ago about a sixteenth. This is the
    windowed counts and the recency features folded into a single number, and it
    has no cliff: an event does not stop mattering the day it leaves the
    ninety-day window.
    """
    half_life = (
        cfg.recency_half_life_days if half_life_days is None else float(half_life_days)
    )
    if half_life <= 0:
        raise ValueError(f"half_life_days must be positive, got {half_life}.")

    filtered = filtered_events(scoped_events_df, event_family, cfg)
    decay = F.log(F.lit(2.0)) / F.lit(half_life)
    weighted = filtered.withColumn(
        "__event_age_days",
        F.abs(F.datediff(F.col(cfg.reference_ts_col), F.col(cfg.event_ts_col))),
    ).withColumn("__recency_weight", F.exp(-decay * F.col("__event_age_days")))

    return weighted.groupBy(*list(entity_cols)).agg(
        F.sum("__recency_weight").alias(
            f"{prefix}_{event_family}_recency_weighted_intensity"
        )
    )


def calculate_cumulative_event_slope(
    scoped_events_df: DataFrame,
    entity_cols: Sequence[str],
    event_family: str,
    prefix: str,
    cfg: FeatureConfig = FeatureConfig(),
) -> DataFrame:
    """The slope of cumulative event count against elapsed days.

    A straight line fitted through the running total. Above the entity's average
    rate means accelerating, below means slowing. Two events give a slope with
    no residual and one gives none at all, so this is a feature that only says
    something for entities with a history.
    """
    filtered = filtered_events(scoped_events_df, event_family, cfg)
    ordered = Window.partitionBy(*list(entity_cols)).orderBy(F.col(cfg.event_ts_col))
    partitioned = Window.partitionBy(*list(entity_cols))

    cumulative = (
        filtered.withColumn(
            "__first_event_ts", F.min(cfg.event_ts_col).over(partitioned)
        )
        .withColumn(
            "__elapsed_days",
            F.abs(F.datediff(F.col(cfg.event_ts_col), F.col("__first_event_ts"))),
        )
        .withColumn("__cum_event_cnt", F.row_number().over(ordered).cast("double"))
    )

    return cumulative.groupBy(*list(entity_cols)).agg(
        F.regr_slope("__cum_event_cnt", "__elapsed_days").alias(
            f"{prefix}_{event_family}_cumulative_event_slope"
        )
    )


def calculate_state_transition_features(
    scoped_events_df: DataFrame,
    entity_cols: Sequence[str],
    state_transitions: Sequence[Tuple[str, str]],
    prefix: str,
    cfg: FeatureConfig = FeatureConfig(),
) -> DataFrame:
    """For each ordered pair of event types, how long until the second follows.

    Three columns per pair: the mean gap, the shortest gap, and the reciprocal of
    the mean as a velocity. The shortest gap is the one to watch — the mean over
    a long history dilutes a single fast pair, and it is the single fast pair
    that a takeover produces.

    Only the types named in ``state_transitions`` are read, and a source event
    with no subsequent target contributes nothing rather than a sentinel, so the
    mean is over the transitions that actually completed.

    See :func:`~trust_score_05.features.expressions.next_event_window` for the
    ordering, including why a target event sharing a timestamp with its source
    has to sort *before* it.
    """
    if not state_transitions:
        return feature_base(scoped_events_df, entity_cols)

    relevant_types = sorted(
        {event_type for transition in state_transitions for event_type in transition}
    )
    filtered = scoped_events_df.filter(
        F.col(cfg.event_type_col).isin(*relevant_types)
    ).select(*list(entity_cols), cfg.event_type_col, cfg.event_ts_col)

    next_target_cols: Dict[str, str] = {}
    enriched = filtered
    for target_type in relevant_types:
        next_col = f"__next_{feature_token(target_type)}_ts"
        next_target_cols[target_type] = next_col
        enriched = enriched.withColumn(
            next_col,
            F.min(
                F.when(
                    F.col(cfg.event_type_col) == target_type, F.col(cfg.event_ts_col)
                )
            ).over(next_event_window(entity_cols, target_type, cfg)),
        )

    agg_exprs: List[Column] = []
    velocity_sources: List[Tuple[str, str]] = []
    for source_type, target_type in state_transitions:
        stub = (
            f"{prefix}_{feature_token(source_type)}_to_{feature_token(target_type)}"
        )
        next_col = next_target_cols[target_type]
        transition_days = F.when(
            (F.col(cfg.event_type_col) == source_type) & F.col(next_col).isNotNull(),
            F.datediff(F.to_date(F.col(next_col)), F.to_date(F.col(cfg.event_ts_col))),
        )
        avg_col = f"{stub}_avg_transition_days"
        agg_exprs.extend(
            [
                F.avg(transition_days).alias(avg_col),
                F.min(transition_days).alias(f"{stub}_min_transition_days"),
            ]
        )
        velocity_sources.append((stub, avg_col))

    result = enriched.groupBy(*list(entity_cols)).agg(*agg_exprs)
    for stub, avg_col in velocity_sources:
        result = result.withColumn(
            f"{stub}_transition_velocity", safe_divide(F.lit(1.0), F.col(avg_col))
        )
    return result


def calculate_stability_features(
    snapshot_scope: Dict[str, DataFrame],
    scope: str,
    cfg: FeatureConfig = FeatureConfig(),
) -> DataFrame:
    """Tenure and breadth, from the snapshot alone.

    No events are involved, which is what makes these the most robust features
    in the set: they survive a carrier feed going quiet. How long we have known
    the entity, how long since its current assignment started, and how many
    accounts and numbers it holds.

    Every feature here is computed from ``snapshot_scope["base"]``, the
    un-aggregated scope-to-lineage join, and ``snapshot_scope[scope]`` is used
    only for the entity skeleton the results are attached to. That is a
    deliberate correction rather than an arbitrary choice: per
    ``snapshots.build_scope_snapshots``, each scope entry is the base frame
    grouped down to one row per entity with ``min(from_ts)`` and
    ``max(to_ts)``, so the individual lineage intervals — the very thing this
    family measures — survive only in ``base``.

    Two drafts of this function read the scope entry instead, and both produced
    columns that were plausible and wrong with nothing raised — see finding 4.4.
    The breadth counts were either typed nulls or the constant 1, because at the
    customer scope the aggregated snapshot has no ``acct_id`` or ``msisdn``
    column at all, and at the account and MSISDN scopes those columns are
    grouping keys, so counting distinct values of them within a group
    necessarily gives one. "How many accounts does this customer hold" is the
    most useful thing in this family and it was unavailable at the only scope
    where it means anything. The second draft fixed breadth but left tenure on
    the scope entry, where ``max(from_ts)`` can only re-derive the single
    surviving ``from_ts``, which is already the minimum: every entity's
    ``{prefix}_days_since_current_assignment_start`` came back exactly equal to
    its ``{prefix}_days_since_first_seen``, so a customer of three years who
    added a line last week was indistinguishable from one who had held the same
    line for three years. Both halves therefore read ``base``, and a missing
    ``base`` entry raises rather than falling back — a fallback here is how the
    defect got in twice.
    """
    for required in (scope, "base"):
        if required not in snapshot_scope:
            raise KeyError(
                f"snapshot_scope has no {required!r} entry; it has "
                f"{sorted(snapshot_scope)}. Build it with build_scope_snapshots."
            )

    entity_cols = list(cfg.entity_cols_for(scope))
    base = snapshot_scope[scope].select(*entity_cols).distinct()
    prefix = scope_prefix(scope)

    intervals = snapshot_scope["base"]
    missing_keys = [key for key in entity_cols if not has_col(intervals, key)]
    if missing_keys:
        raise ValueError(
            f"snapshot_scope['base'] is missing {missing_keys}, which the "
            f"{scope!r} stability features are grouped by; it has "
            f"{intervals.columns}."
        )

    work = intervals
    for col_name in (cfg.reference_ts_col, cfg.from_ts_col, cfg.to_ts_col):
        if not has_col(work, col_name):
            work = work.withColumn(col_name, F.lit(None).cast("timestamp"))

    tenure = (
        work.groupBy(*entity_cols)
        .agg(
            F.max(F.col(cfg.reference_ts_col)).alias("__reference_ts"),
            F.min(F.col(cfg.from_ts_col)).alias("__min_from_ts"),
            F.max(F.col(cfg.from_ts_col)).alias("__max_from_ts"),
        )
        .withColumn(
            f"{prefix}_days_since_first_seen",
            F.datediff(
                F.to_date(F.col("__reference_ts")), F.to_date(F.col("__min_from_ts"))
            ),
        )
        .withColumn(
            f"{prefix}_days_since_current_assignment_start",
            F.datediff(
                F.to_date(F.col("__reference_ts")), F.to_date(F.col("__max_from_ts"))
            ),
        )
        .drop("__reference_ts", "__min_from_ts", "__max_from_ts")
    )

    def _distinct_or_null(frame: DataFrame, col_name: str, alias: str) -> Column:
        """Count the distinct values, or a typed null if the column is absent.

        Typed rather than ``F.lit(None)`` because an untyped null column cannot
        be written to Parquet, so the failure would land in the write stage
        rather than here.
        """
        if has_col(frame, col_name):
            return F.countDistinct(F.col(col_name)).alias(alias)
        return F.lit(None).cast("long").alias(alias)

    breadth = intervals.groupBy(*entity_cols).agg(
        _distinct_or_null(intervals, cfg.account_col, f"{prefix}_distinct_account_cnt"),
        _distinct_or_null(intervals, cfg.msisdn_col, f"{prefix}_distinct_msisdn_cnt"),
    )

    return safe_join(base, [tenure, breadth], entity_cols)
