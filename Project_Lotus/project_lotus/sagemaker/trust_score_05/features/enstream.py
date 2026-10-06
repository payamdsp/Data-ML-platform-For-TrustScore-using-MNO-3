"""EnStream partner traffic, which needs its own treatment.

An EnStream event is a partner asking us about a phone number. Unlike a carrier
change record it is our own traffic, it carries metadata about who asked and
why, and its volume says as much about the partner's business as about the
number. That last point is the reason this module exists.

**Why the counts are weighted.** A single high-volume partner — a bank that
checks every number on every login — will dominate a raw count of API calls for
every number it touches. The number is not unusual; the partner is. So each
event is weighted by how a partner's volume for *this entity* compares to the
average partner's volume for the same entity: a partner that asks far more than
its peers is discounted, one that asks less is amplified, and the weight is
clamped so neither can run away. The result is a count in which a first-ever
call from an unusual partner is worth more than the thousandth call from the
usual one.

**Why diversity and novelty are here and not in the generic calculators.** Both
are computed over partner metadata — industry, service provider, use case — and
only EnStream events have any. Breadth of partners asking about one number, and
how recently the breadth widened, are among the strongest signals in the set:
a number that has been asked about by one bank for two years and then by four
lenders in a week is describing an application spree.

The weighted frequency features produced here use the *same column names* as
:func:`~trust_score_05.features.calculators.calculate_event_frequency_features`
would. That is intentional — the generic feature set skips frequency for this
family precisely so that these fill the slot — and it is why the prefix has to
match. In the notebook library it did not: the wrapper passed the scope name
("customer") where every other calculator passed the prefix ("cust"), so the
EnStream family ended up split across two namespaces and the generic frequency
columns for it were never filled by anything.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from trust_score_05.features.calculators import calculate_event_diversity_features
from trust_score_05.features.config import (
    CANONICAL_EVENT_FAMILY,
    ENSTREAM_DIMENSION_COLS,
    FeatureConfig,
)
from trust_score_05.features.expressions import (
    feature_base,
    filtered_events,
    has_col,
    lookback_condition,
    safe_divide,
    safe_join,
)
from trust_score_05.features.snapshots import attach_events_to_scope

__all__ = [
    "calculate_enstream_event_diversity_features",
    "calculate_enstream_event_frequency_features",
    "calculate_enstream_event_novelty_features",
    "calculate_enstream_partner_only_features",
    "calculate_enstream_scoped_events",
    "with_partner_volume_weights",
]

ENSTREAM_FAMILY = CANONICAL_EVENT_FAMILY["ENSTREAM_API_CALL"]


def calculate_enstream_scoped_events(
    snapshot_scope: Dict[str, DataFrame],
    scope: str,
    normalized_enstream_events_df: DataFrame,
    cfg: FeatureConfig = FeatureConfig(),
) -> DataFrame:
    """Attach normalised EnStream events to a scope snapshot.

    The EnStream feed is joined to the scopes separately from the carrier feeds
    rather than after being unioned into the canonical table, because it is the
    only feed whose events need to be re-read at partner grain to compute the
    weights. Joining it once here and once inside
    :func:`~trust_score_05.features.events.build_changes_table` costs a second
    pass over the feed and keeps the weighting out of the shared table, where it
    would mean nothing for the eight carrier families.
    """
    return attach_events_to_scope(
        snapshot_scope, normalized_enstream_events_df, scope, cfg
    )


def _window_weight_col(days: int, cfg: FeatureConfig) -> str:
    return f"{cfg.event_weight_col}_{days}d"


def with_partner_volume_weights(
    scoped_enstream_events_df: DataFrame,
    entity_cols: Sequence[str],
    cfg: FeatureConfig = FeatureConfig(),
) -> DataFrame:
    """Add one volume-normalising weight column per lookback window.

    For each window and each (entity, partner) pair, count the partner's events
    in the window. Compare that to the mean across the entity's partners. The
    weight is the square root of the ratio, clamped to
    ``[enstream_min_weight, enstream_max_weight]`` — the square root because the
    raw ratio between a bank checking daily and a lender checking once is three
    orders of magnitude, and a weight that spans three orders of magnitude is a
    filter rather than a normaliser.

    The join back onto the events is on the entity columns *and* the partner
    columns, which is the whole grain the weights were computed at. The notebook
    library dropped the partner columns from the projection and joined on the
    entity alone, which was not merely a wrong weight but a fan-out: each event
    row was duplicated once per (service provider, use case) pair the entity had
    ever produced, each copy carrying a different weight. Because the same frame
    is the source of the *unweighted* counts too, every EnStream count feature
    was multiplied by the entity's partner breadth — so the feature that was
    supposed to correct for partner volume was instead the largest source of
    partner-volume bias in the table.

    One frame is returned carrying every window's weight, rather than one frame
    per window. Same arithmetic, one shuffle of the event table instead of one
    per window.
    """
    reference_cols = list(entity_cols)
    partner_cols = [
        col_name
        for col_name in cfg.enstream_partner_cols
        if has_col(scoped_enstream_events_df, col_name)
    ]

    if not partner_cols:
        # No partner metadata means there is no partner volume to normalise
        # against. Weight of 1.0 makes the weighted counts equal the plain
        # counts, which is the right degenerate answer.
        weighted = scoped_enstream_events_df
        for days in cfg.lookback_days:
            weighted = weighted.withColumn(
                _window_weight_col(days, cfg), F.lit(1.0)
            )
        return weighted

    partner_group_cols = [*reference_cols, *partner_cols]
    weighted = scoped_enstream_events_df

    for days in cfg.lookback_days:
        condition = lookback_condition(days, cfg)
        volume_col = f"__partner_volume_{days}d"
        avg_col = f"__avg_partner_volume_{days}d"
        weight_col = _window_weight_col(days, cfg)

        partner_stats = scoped_enstream_events_df.groupBy(*partner_group_cols).agg(
            F.sum(F.when(condition, F.lit(1)).otherwise(F.lit(0))).alias(volume_col)
        )
        entity_stats = partner_stats.groupBy(*reference_cols).agg(
            F.avg(volume_col).alias(avg_col)
        )
        partner_weights = partner_stats.join(
            entity_stats, reference_cols, "left"
        ).withColumn(
            weight_col,
            F.least(
                F.lit(float(cfg.enstream_max_weight)),
                F.greatest(
                    F.lit(float(cfg.enstream_min_weight)),
                    F.sqrt(
                        F.col(avg_col) / F.greatest(F.col(volume_col), F.lit(1.0))
                    ),
                ),
            ),
        )

        weighted = weighted.join(
            partner_weights.select(*partner_group_cols, weight_col),
            partner_group_cols,
            "left",
        ).withColumn(weight_col, F.coalesce(F.col(weight_col), F.lit(1.0)))

    return weighted


def calculate_enstream_event_frequency_features(
    scoped_enstream_events_df: DataFrame,
    entity_cols: Sequence[str],
    prefix: str,
    cfg: FeatureConfig = FeatureConfig(),
) -> DataFrame:
    """Counts, partner-weighted counts and any-flags for the EnStream family.

    The weight applied in each window is that window's own weight, because a
    partner's relative volume over a day and over a quarter are different
    facts: a partner that made one call today and a thousand last quarter should
    be amplified in the one-day window and discounted in the ninety-day one.
    """
    weighted = with_partner_volume_weights(
        scoped_enstream_events_df, entity_cols, cfg
    )
    filtered = filtered_events(weighted, ENSTREAM_FAMILY, cfg)
    base = feature_base(scoped_enstream_events_df, entity_cols)

    agg_exprs: List[Column] = []
    for days in cfg.lookback_days:
        condition = lookback_condition(days, cfg)
        weight_expr = F.coalesce(
            F.col(_window_weight_col(days, cfg)).cast("double"), F.lit(1.0)
        )
        agg_exprs.extend(
            [
                F.sum(F.when(condition, F.lit(1)).otherwise(F.lit(0))).alias(
                    f"{prefix}_{ENSTREAM_FAMILY}_cnt_{days}d"
                ),
                F.sum(F.when(condition, weight_expr).otherwise(F.lit(0.0))).alias(
                    f"{prefix}_{ENSTREAM_FAMILY}_weighted_cnt_{days}d"
                ),
                F.max(F.when(condition, F.lit(1)).otherwise(F.lit(0))).alias(
                    f"{prefix}_{ENSTREAM_FAMILY}_any_{days}d"
                ),
            ]
        )

    features = filtered.groupBy(*list(entity_cols)).agg(*agg_exprs)
    return safe_join(base, [features], entity_cols)


def calculate_enstream_event_diversity_features(
    scoped_enstream_events_df: DataFrame,
    entity_cols: Sequence[str],
    prefix: str,
    dimension_cols: Sequence[str] = ENSTREAM_DIMENSION_COLS,
    cfg: FeatureConfig = FeatureConfig(),
) -> DataFrame:
    """Breadth of partners, industries and use cases asking about the entity."""
    return calculate_event_diversity_features(
        scoped_events_df=scoped_enstream_events_df,
        entity_cols=entity_cols,
        event_family=ENSTREAM_FAMILY,
        prefix=prefix,
        dimension_cols=dimension_cols,
        cfg=cfg,
    )


def calculate_enstream_event_novelty_features(
    scoped_enstream_events_df: DataFrame,
    entity_cols: Sequence[str],
    prefix: str,
    dimension_cols: Sequence[str] = ENSTREAM_DIMENSION_COLS,
    cfg: FeatureConfig = FeatureConfig(),
) -> DataFrame:
    """When the entity's partner mix last widened, and by how much.

    Diversity says how many distinct industries have asked. Novelty says
    *when* each of them first did, which separates two entities with identical
    diversity: one accumulated four industries over three years, the other over
    three days.

    Three groups of column per dimension. The spread between the earliest and
    latest first-sighting, in days — small means the breadth arrived all at
    once. Days since the most recent first-sighting — small means it is still
    arriving. And the share of the entity's events in that dimension that were
    first-sightings — high means almost every call is from somebody new.
    """
    filtered = filtered_events(scoped_enstream_events_df, ENSTREAM_FAMILY, cfg)
    base = feature_base(scoped_enstream_events_df, entity_cols)
    reference_keys = [*list(entity_cols), cfg.reference_ts_col]
    feature_frames: List[Optional[DataFrame]] = []

    for dimension in dimension_cols:
        if not has_col(filtered, dimension):
            continue

        first_seen = filtered.groupBy(*reference_keys, dimension).agg(
            F.min(cfg.event_ts_col).alias("__first_seen_ts")
        )

        novelty = first_seen.groupBy(*reference_keys).agg(
            F.datediff(F.max("__first_seen_ts"), F.min("__first_seen_ts")).alias(
                f"{prefix}_new_{dimension}_spread_days"
            ),
            F.datediff(F.max(cfg.reference_ts_col), F.max("__first_seen_ts")).alias(
                f"{prefix}_days_since_last_new_{dimension}"
            ),
            F.countDistinct(F.col(dimension)).alias(
                f"{prefix}_{dimension}_distinct_cnt"
            ),
        )

        event_stats = (
            filtered.join(first_seen, [*reference_keys, dimension], "inner")
            .withColumn(
                "__is_first_seen_event",
                F.when(
                    F.col(cfg.event_ts_col) == F.col("__first_seen_ts"), F.lit(1)
                ).otherwise(F.lit(0)),
            )
            .groupBy(*reference_keys)
            .agg(
                F.sum("__is_first_seen_event").alias(
                    f"{prefix}_new_{dimension}_event_cnt"
                ),
                F.count("*").alias(f"{prefix}_{dimension}_event_cnt"),
            )
            .withColumn(
                f"{prefix}_new_{dimension}_event_frac",
                safe_divide(
                    F.col(f"{prefix}_new_{dimension}_event_cnt"),
                    F.col(f"{prefix}_{dimension}_event_cnt"),
                ),
            )
        )

        # The reference timestamp is carried through the grouping because the
        # "days since" columns need it, but it is not part of the entity key, so
        # it is dropped before the join back onto the base.
        feature_frames.extend(
            [novelty.drop(cfg.reference_ts_col), event_stats.drop(cfg.reference_ts_col)]
        )

    return safe_join(base, feature_frames, entity_cols)


def calculate_enstream_partner_only_features(
    snapshot_scope: Dict[str, DataFrame],
    scope: str,
    normalized_enstream_events_df: DataFrame,
    entity_cols: Sequence[str],
    prefix: str,
    cfg: FeatureConfig = FeatureConfig(),
) -> DataFrame:
    """The three EnStream-only feature groups, joined onto one frame.

    ``prefix`` is used, not the scope name. That sounds too obvious to state,
    and it is stated because the previous implementation took this argument and
    then passed ``scope`` to all three calculators underneath — see the module
    docstring.
    """
    scoped = calculate_enstream_scoped_events(
        snapshot_scope, scope, normalized_enstream_events_df, cfg
    )
    base = feature_base(scoped, entity_cols)

    feature_dfs: List[Optional[DataFrame]] = [
        calculate_enstream_event_frequency_features(
            scoped, entity_cols=entity_cols, prefix=prefix, cfg=cfg
        ),
        calculate_enstream_event_diversity_features(
            scoped,
            entity_cols=entity_cols,
            prefix=prefix,
            dimension_cols=ENSTREAM_DIMENSION_COLS,
            cfg=cfg,
        ),
        calculate_enstream_event_novelty_features(
            scoped,
            entity_cols=entity_cols,
            prefix=prefix,
            dimension_cols=ENSTREAM_DIMENSION_COLS,
            cfg=cfg,
        ),
    ]
    return safe_join(base, feature_dfs, entity_cols)
