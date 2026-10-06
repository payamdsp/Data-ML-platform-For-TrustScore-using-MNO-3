"""References, scopes, and the join that attaches events to them.

A *reference* is one question: describe this entity as it stood at this moment.
Every feature in the package is computed per reference, which is why
``reference_id`` leads every grouping key. Two references for the same customer
a month apart are two independent rows, and that is the property that makes the
feature table usable as training data: a fraud that happened in March and a
non-fraud sampled in June are different rows even when they name the same
customer.

References come from one of two places.

*A single moment across the whole population.* Used to build the non-fraud
sample: pick a date, describe every customer as of that date.

*A list of phone numbers, each with its own moment.* Used to build the fraud
sample: each confirmed fraud names a number and a time, and the reference is
resolved through the lineage to whichever customer held that number then.

A *scope snapshot* then rolls the reference out to the grain features are built
at. The customer scope is one row per reference; the account scope is one row
per reference per account the customer held; the MSISDN scope is one row per
reference per number. Coarser scopes are prefixes of finer ones, which is what
lets the same event join serve all three.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from trust_score_05.features.config import SCOPES, FeatureConfig
from trust_score_05.features.expressions import (
    closed_lineage_intervals,
    combine_conditions,
    dedupe_columns,
    has_col,
)

__all__ = [
    "attach_events_to_scope",
    "build_reference_df",
    "build_reference_df_at_timestamp",
    "build_reference_df_from_msisdns",
    "build_scope_snapshots",
]


def _reference_ts_expr(reference_ts) -> Column:
    """Accept either a Spark column or something castable to a timestamp."""
    if hasattr(reference_ts, "cast"):
        return reference_ts.cast("timestamp")
    return F.to_timestamp(F.lit(reference_ts))


def _reference_keys(df: DataFrame, cfg: FeatureConfig) -> List[str]:
    """The reference identity columns that ``df`` actually carries."""
    ordered = (cfg.reference_id_col, cfg.customer_col, cfg.reference_ts_col)
    return [col_name for col_name in ordered if has_col(df, col_name)]


def _synthetic_reference_id(entity_col: str, cfg: FeatureConfig) -> Column:
    """A reference id from the entity and the moment, when none was supplied.

    Two references are the same reference exactly when they name the same entity
    at the same second, so that is what the id is built from. It is readable on
    purpose — a reference id that can be eyeballed back to a customer and a date
    saves a join every time somebody debugs a feature row.
    """
    return F.concat_ws(
        "_",
        F.col(entity_col).cast("string"),
        F.date_format(F.col(cfg.reference_ts_col), "yyyyMMddHHmmss"),
    )


def build_reference_df_at_timestamp(
    lineage_df: DataFrame,
    reference_ts,
    cfg: FeatureConfig = FeatureConfig(),
) -> DataFrame:
    """One reference per customer, all at the same moment.

    This is the non-fraud population. Every customer the lineage knows about is
    described as of ``reference_ts``, which is what makes the resulting sample
    comparable to a fraud sample drawn at the same date.
    """
    lineage = closed_lineage_intervals(lineage_df, cfg)
    return (
        lineage.select(cfg.customer_col)
        .distinct()
        .withColumn(cfg.reference_ts_col, _reference_ts_expr(reference_ts))
        .withColumn(
            cfg.reference_id_col, _synthetic_reference_id(cfg.customer_col, cfg)
        )
    )


def build_reference_df_from_msisdns(
    lineage_df: DataFrame,
    msisdn_reference_df: DataFrame,
    reference_ts=None,
    cfg: FeatureConfig = FeatureConfig(),
) -> DataFrame:
    """Resolve (phone number, moment) pairs to the customer who held the number.

    This is the fraud population: a confirmed fraud names a number and a time,
    and the customer is whoever the lineage says held that number then.

    Two details are worth stating because both are choices.

    The interval test is on dates, not timestamps, and it is closed at both
    ends. A fraud report carries the date reliably and the time of day much less
    so, so an interval that opened at nine in the morning should still be
    considered for a fraud recorded as having happened that day. Closing the
    upper bound has the same motivation from the other side, and means a number
    that changed hands on the day of the fraud matches both intervals — which is
    why the result is then ranked.

    Where both intervals match, the one that started later wins. A number moving
    to a new customer on the day of the fraud is far more likely to be the
    fraud's context than the assignment it replaced; that is precisely the
    pattern a takeover produces.
    """
    lineage = closed_lineage_intervals(lineage_df, cfg)
    lookups = dedupe_columns(msisdn_reference_df)

    if reference_ts is not None:
        lookups = lookups.withColumn(
            cfg.reference_ts_col, _reference_ts_expr(reference_ts)
        )
    elif not has_col(lookups, cfg.reference_ts_col):
        raise ValueError(
            f"`{cfg.reference_ts_col}` must be a column of msisdn_reference_df when "
            "reference_ts is not given, because each reference needs its own moment."
        )

    if not has_col(lookups, cfg.reference_id_col):
        lookups = lookups.withColumn(
            cfg.reference_id_col, _synthetic_reference_id(cfg.msisdn_col, cfg)
        )

    join_conditions = [F.col(f"ref.{cfg.msisdn_col}") == F.col(f"lin.{cfg.msisdn_col}")]
    active_at_reference = [
        F.to_date(F.col(f"lin.{cfg.from_ts_col}"))
        <= F.to_date(F.col(f"ref.{cfg.reference_ts_col}")),
        F.to_date(F.col(f"lin.{cfg.to_ts_col}"))
        >= F.to_date(F.col(f"ref.{cfg.reference_ts_col}")),
    ]

    matched = (
        lookups.alias("ref")
        .join(lineage.alias("lin"), combine_conditions(join_conditions), "inner")
        .filter(combine_conditions(active_at_reference))
    )

    partition_cols = [
        F.col(f"ref.{col_name}")
        for col_name in (cfg.reference_id_col, cfg.msisdn_col, cfg.reference_ts_col)
        if has_col(lookups, col_name)
    ]
    ranked = matched.withColumn(
        "__rn",
        F.row_number().over(
            Window.partitionBy(*partition_cols).orderBy(
                F.col(f"lin.{cfg.from_ts_col}").desc(),
                F.col(f"lin.{cfg.to_ts_col}").desc(),
            )
        ),
    ).filter(F.col("__rn") == 1)

    return ranked.select(
        F.col(f"ref.{cfg.reference_id_col}").alias(cfg.reference_id_col),
        F.col(f"ref.{cfg.reference_ts_col}").alias(cfg.reference_ts_col),
        F.col(f"ref.{cfg.msisdn_col}").alias(cfg.msisdn_col),
        F.col(f"lin.{cfg.customer_col}").alias(cfg.customer_col),
        F.col(f"lin.{cfg.account_col}").alias(cfg.account_col),
        F.col(f"lin.{cfg.from_ts_col}").alias(cfg.from_ts_col),
        F.col(f"lin.{cfg.to_ts_col}").alias(cfg.to_ts_col),
        F.col(f"lin.{cfg.mno_col}").alias(cfg.mno_col),
    )


def build_reference_df(
    lineage_df: DataFrame,
    msisdn_reference_df: Optional[DataFrame] = None,
    reference_ts=None,
    cfg: FeatureConfig = FeatureConfig(),
) -> DataFrame:
    """Build references either way, dispatching on which inputs were given.

    Exactly one of the two routes has to be chosen, and the error when neither
    or both is given is explicit. The notebook version of this had an
    ``if``/``elif`` with no ``else``, so calling it with both arguments — or
    with neither — raised ``UnboundLocalError`` from the return statement,
    several frames away from the mistake.
    """
    has_lookups = msisdn_reference_df is not None
    has_timestamp = reference_ts is not None

    if has_lookups and not has_timestamp:
        return build_reference_df_from_msisdns(
            lineage_df, msisdn_reference_df, None, cfg
        )
    if has_lookups and has_timestamp:
        return build_reference_df_from_msisdns(
            lineage_df, msisdn_reference_df, reference_ts, cfg
        )
    if has_timestamp:
        return build_reference_df_at_timestamp(lineage_df, reference_ts, cfg)
    raise ValueError(
        "build_reference_df needs either `reference_ts` (one moment for every "
        "customer) or `msisdn_reference_df` (a phone number and a moment per "
        "row), and was given neither."
    )


def build_scope_snapshots(
    lineage_df: DataFrame,
    reference_df: DataFrame,
    cfg: FeatureConfig = FeatureConfig(),
) -> Dict[str, DataFrame]:
    """Roll each reference out to the customer, account and MSISDN grains.

    Returns a dict with four entries: the three scopes, plus ``"base"`` — the
    un-aggregated join of references to every lineage row active at the
    reference moment. ``base`` is not a scope and is not fed to the calculators;
    it is kept because it is what a data-quality check or an investigation of a
    surprising feature value needs, and recomputing it costs the same join
    again.

    Each scope carries the widest interval its rows span: the earliest
    ``from_ts`` and the latest ``to_ts`` of the lineage rows underneath it. That
    is what the stability features measure tenure against, and it is also the
    lower bound for the event scan.
    """
    lineage = closed_lineage_intervals(lineage_df, cfg)
    references = dedupe_columns(reference_df)

    if not has_col(references, cfg.reference_id_col):
        references = references.withColumn(
            cfg.reference_id_col, _synthetic_reference_id(cfg.customer_col, cfg)
        )

    join_conditions = [
        F.col(f"lin.{cfg.customer_col}") == F.col(f"ref.{cfg.customer_col}")
    ]
    # Dates rather than timestamps, and closed at both ends, for the reason
    # given in build_reference_df_from_msisdns: a reference moment is trusted to
    # the day and not to the second.
    active_at_reference = [
        F.to_date(F.col(f"lin.{cfg.from_ts_col}"))
        <= F.to_date(F.col(f"ref.{cfg.reference_ts_col}")),
        F.to_date(F.col(f"lin.{cfg.to_ts_col}"))
        >= F.to_date(F.col(f"ref.{cfg.reference_ts_col}")),
    ]

    base = (
        lineage.alias("lin")
        .join(references.alias("ref"), combine_conditions(join_conditions), "inner")
        .filter(combine_conditions(active_at_reference))
        .select(
            F.col(f"ref.{cfg.reference_id_col}").alias(cfg.reference_id_col),
            F.col(f"lin.{cfg.customer_col}").alias(cfg.customer_col),
            F.col(f"ref.{cfg.reference_ts_col}").alias(cfg.reference_ts_col),
            F.col(f"lin.{cfg.account_col}").alias(cfg.account_col),
            F.col(f"lin.{cfg.msisdn_col}").alias(cfg.msisdn_col),
            F.col(f"lin.{cfg.mno_col}").alias(cfg.mno_col),
            F.col(f"lin.{cfg.from_ts_col}").alias(cfg.from_ts_col),
            F.col(f"lin.{cfg.to_ts_col}").alias(cfg.to_ts_col),
        )
        .dropDuplicates()
    )

    reference_keys = _reference_keys(base, cfg)
    snapshots: Dict[str, DataFrame] = {"base": base}

    for scope in SCOPES:
        extra_cols = [
            col_name
            for col_name in cfg.group_cols_for(scope)
            if col_name not in reference_keys
        ]
        # The carrier is part of the account and MSISDN grains but not of the
        # customer grain: one customer can hold accounts on two carriers, and
        # collapsing them would attribute one carrier's events to the other's
        # account.
        group_cols = [*reference_keys, *extra_cols]
        if scope in ("account", "msisdn") and has_col(base, cfg.mno_col):
            group_cols = [*group_cols, cfg.mno_col]

        snapshots[scope] = (
            base.groupBy(*group_cols)
            .agg(
                F.min(cfg.from_ts_col).alias(cfg.from_ts_col),
                F.max(cfg.to_ts_col).alias(cfg.to_ts_col),
            )
            .dropDuplicates(group_cols)
        )

    return snapshots


def attach_events_to_scope(
    snapshot_scope: Dict[str, DataFrame],
    changes_df: DataFrame,
    scope: str,
    cfg: FeatureConfig = FeatureConfig(),
) -> DataFrame:
    """Join the canonical event table to a scope snapshot.

    The result is one row per (reference, event) pair, plus one row per
    reference that had no events at all — the join is a left join for the same
    reason :func:`~trust_score_05.features.expressions.safe_join` is: a customer
    with a silent history is a feature row of zeros and nulls, not an absent
    row, and silence is itself informative.

    Two performance decisions are in here, both switchable in
    :class:`~trust_score_05.features.config.FeatureConfig`.

    The time predicate is *in* the join condition rather than in a filter after
    it. Joining on identity first and filtering afterwards materialises every
    event a customer ever produced before discarding almost all of them, and for
    a customer with years of history that intermediate is the largest thing the
    job builds. Pushed into the condition, the lower bound is the later of the
    scope's own interval start and the widest lookback window, so the scan is
    bounded by the window rather than by the customer's tenure.

    The buffer on that bound (``event_read_buffer_days``) exists because some
    features are not windowed. Periodicity needs the interval between the last
    two events, which can straddle the ninety-day edge, and reading a week
    either side of the widest window keeps those intervals intact where they
    matter most — nearest the reference.

    The snapshot is broadcast. It is one row per reference and the event table
    is many rows per reference, so it is the small side by construction.
    """
    if scope not in SCOPES:
        raise ValueError(
            f"Unsupported scope {scope!r}. Expected one of {', '.join(SCOPES)}."
        )
    if scope not in snapshot_scope:
        raise KeyError(
            f"snapshot_scope has no {scope!r} entry; it has "
            f"{sorted(snapshot_scope)}. Build it with build_scope_snapshots."
        )

    join_cols = list(cfg.group_cols_for(scope))
    scope_snapshot_df = dedupe_columns(snapshot_scope[scope])
    events_df = dedupe_columns(changes_df)

    scope_alias, event_alias = "scope", "evt"

    entity_conditions = [
        F.col(f"{scope_alias}.{col_name}") == F.col(f"{event_alias}.{col_name}")
        for col_name in join_cols
    ]

    event_ts = F.col(f"{event_alias}.{cfg.event_ts_col}")
    reference_ts = F.col(f"{scope_alias}.{cfg.reference_ts_col}")
    interval_start = F.col(f"{scope_alias}.{cfg.from_ts_col}")

    if cfg.prune_event_window:
        window_days = cfg.max_lookback_days + int(cfg.event_read_buffer_days)
        event_lower_bound = F.greatest(
            interval_start,
            reference_ts - F.expr(f"INTERVAL {window_days} DAYS"),
        )
    else:
        event_lower_bound = interval_start

    # `event_ts.isNull()` keeps the left-join misses. Without it the null side
    # of the outer join fails the range test and the reference is dropped,
    # turning the left join back into an inner one.
    time_condition = event_ts.isNull() | (
        (event_ts <= reference_ts) & (event_ts >= event_lower_bound)
    )

    scope_cols = list(scope_snapshot_df.columns)
    event_cols = [
        col_name for col_name in events_df.columns if col_name not in scope_cols
    ]

    left_df = scope_snapshot_df.alias(scope_alias)
    if cfg.broadcast_scope:
        left_df = F.broadcast(left_df)

    return left_df.join(
        events_df.alias(event_alias),
        combine_conditions([*entity_conditions, time_condition]),
        "left",
    ).select(
        *[
            F.col(f"{scope_alias}.{col_name}").alias(col_name)
            for col_name in scope_cols
        ],
        *[
            F.col(f"{event_alias}.{col_name}").alias(col_name)
            for col_name in event_cols
        ],
    )
