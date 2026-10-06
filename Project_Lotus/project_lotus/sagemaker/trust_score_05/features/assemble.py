"""The entry points: turn lineage plus events into one wide row per reference.

Everything below this module is a calculator that answers one narrow question.
This module decides which questions get asked, at which scope, and how the
answers are glued into a single frame.

There are three levels:

:func:`calculate_generic_event_feature_set`
    Every family-agnostic feature for one event family at one scope.
:func:`assemble_scope_features`
    Every family, plus the EnStream-only features, plus stability, plus state
    transitions, for one scope.
:func:`assemble_all_scope_features`
    All three scopes, returned separately rather than joined, because they have
    different grains and joining them is a modelling decision rather than a
    feature-engineering one.

The one asymmetry worth knowing about before reading the code: the
``enstream_api_call`` family is deliberately *excluded* from the generic
frequency and velocity calculators. Its counts come from
:func:`~trust_score_05.features.enstream.calculate_enstream_partner_only_features`
instead, which produces the same column names but weights each event by the
inverse of its partner's call volume. Running both would produce two frames
claiming the same columns, and :func:`join_feature_frames` would keep the first
and silently drop the second — so which one runs is a correctness question, not
a performance one.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pyspark.sql import DataFrame

from trust_score_05.features.calculators import (
    calculate_burstness_features,
    calculate_cumulative_event_slope,
    calculate_event_frequency_features,
    calculate_event_recency_features,
    calculate_periodicity_features,
    calculate_recency_weighted_intensity,
    calculate_stability_features,
    calculate_state_transition_features,
    calculate_temporal_entropy_features,
    calculate_time_to_k_events,
)
from trust_score_05.features.config import (
    CANONICAL_EVENT_FAMILY,
    DEFAULT_STATE_TRANSITIONS,
    GENERIC_EVENT_FAMILIES,
    SCOPES,
    FeatureConfig,
    scope_prefix,
)
from trust_score_05.features.enstream import (
    calculate_enstream_partner_only_features,
)
from trust_score_05.features.expressions import (
    feature_base,
    feature_columns,
    safe_join,
)
from trust_score_05.features.snapshots import attach_events_to_scope

__all__ = [
    "assemble_all_scope_features",
    "assemble_scope_features",
    "calculate_generic_event_feature_set",
    "feature_column_names",
    "join_feature_frames",
]

ENSTREAM_FAMILY = CANONICAL_EVENT_FAMILY["ENSTREAM_API_CALL"]


def join_feature_frames(
    base_df: DataFrame,
    feature_dfs: Iterable[Optional[DataFrame]],
    join_keys: Sequence[str],
) -> DataFrame:
    """Left-join every feature frame onto ``base_df`` on ``join_keys``.

    A thin public alias for :func:`~trust_score_05.features.expressions.safe_join`.
    It exists under this name because it is the one function in the package that
    a caller assembling their own subset of features will need, and pointing
    them at a module called ``expressions`` to find it reads badly.

    The semantics that matter: the join is a left join from the base, so an
    entity with no events of a family keeps its row with nulls rather than
    disappearing; and a column already present on the accumulated result is
    skipped rather than suffixed, so two frames claiming the same feature name
    resolve to the first.
    """
    return safe_join(base_df, feature_dfs, join_keys)


def calculate_generic_event_feature_set(
    scoped_events_df: DataFrame,
    entity_cols: Sequence[str],
    event_family: str,
    prefix: str,
    cfg: FeatureConfig = FeatureConfig(),
) -> DataFrame:
    """Every family-agnostic feature for one event family.

    Six calculators run for every family: recency, periodicity, burstness,
    temporal entropy, recency-weighted intensity and cumulative slope. Two more
    — frequency and time-to-*k* — run for every family except
    ``enstream_api_call``, for the reason in the module docstring.

    ``k`` comes from ``cfg.k_event_values`` rather than from a default argument.
    The notebook version took ``k_values=(3, 5)`` as a parameter of this function
    while ``cfg`` also carried the windows, so the two could disagree; and it
    then called ``calculate_time_to_k_events`` without forwarding ``cfg`` at all,
    which meant a caller who narrowed ``lookback_days`` got narrowed windows
    everywhere except the velocity features.
    """
    base = feature_base(scoped_events_df, entity_cols)

    shared = dict(
        entity_cols=entity_cols,
        event_family=event_family,
        prefix=prefix,
        cfg=cfg,
    )
    feature_dfs: List[Optional[DataFrame]] = [
        calculate_event_recency_features(scoped_events_df, **shared),
        calculate_periodicity_features(scoped_events_df, **shared),
        calculate_burstness_features(scoped_events_df, **shared),
        calculate_temporal_entropy_features(scoped_events_df, **shared),
        calculate_recency_weighted_intensity(
            scoped_events_df,
            **shared,
            half_life_days=cfg.recency_half_life_days,
        ),
        calculate_cumulative_event_slope(scoped_events_df, **shared),
    ]

    if event_family != ENSTREAM_FAMILY:
        feature_dfs.append(
            calculate_event_frequency_features(scoped_events_df, **shared)
        )
        for k in cfg.k_event_values:
            feature_dfs.append(
                calculate_time_to_k_events(scoped_events_df, **shared, k=k)
            )

    return join_feature_frames(
        base_df=base, feature_dfs=feature_dfs, join_keys=entity_cols
    )


def assemble_scope_features(
    snapshot_scope: Dict[str, DataFrame],
    changes_df: DataFrame,
    normalized_enstream_events_df: Optional[DataFrame],
    scope: str,
    state_transitions: Sequence[Tuple[str, str]] = DEFAULT_STATE_TRANSITIONS,
    event_families: Sequence[str] = GENERIC_EVENT_FAMILIES,
    cfg: FeatureConfig = FeatureConfig(),
) -> DataFrame:
    """The full feature row for one scope.

    Args:
        snapshot_scope: The output of
            :func:`~trust_score_05.features.snapshots.build_scope_snapshots` —
            a mapping with ``base``, ``customer``, ``account`` and ``msisdn``
            entries.
        changes_df: The canonical event table from
            :func:`~trust_score_05.features.events.build_changes_table`.
        normalized_enstream_events_df: The EnStream feed with partner metadata
            still attached, which ``changes_df`` no longer carries at full
            fidelity. ``None`` skips the EnStream-only features, which is what
            a run against carrier data alone wants.
        scope: One of ``customer``, ``account``, ``msisdn``.
        state_transitions: Ordered ``(source, target)`` event type pairs.
        event_families: Which families get the generic feature set. Defaults to
            every family in the canonical mapping; narrowing it is how a caller
            builds a cheaper subset.
        cfg: Windows and column names.

    Returns:
        One row per reference at this scope: the snapshot columns, then every
        feature, left-joined so that a reference with no events survives.

    The scope's events are attached once and then filtered per family, rather
    than joined per family. The join is the expensive step — it is a range join
    between the snapshot and the event table — and doing it nine times to
    produce nine disjoint subsets of the same result was the single largest cost
    in the notebook pipeline.
    """
    if scope not in snapshot_scope:
        raise KeyError(
            f"snapshot_scope has no {scope!r} entry; it has "
            f"{sorted(snapshot_scope)}. Build it with build_scope_snapshots."
        )

    scope_snapshot_df = snapshot_scope[scope]
    entity_cols = list(cfg.entity_cols_for(scope))
    prefix = scope_prefix(scope)

    scoped_events = attach_events_to_scope(snapshot_scope, changes_df, scope, cfg)

    feature_frames: List[Optional[DataFrame]] = [
        calculate_generic_event_feature_set(
            scoped_events,
            entity_cols=entity_cols,
            event_family=event_family,
            prefix=prefix,
            cfg=cfg,
        )
        for event_family in event_families
    ]

    if normalized_enstream_events_df is not None:
        feature_frames.append(
            calculate_enstream_partner_only_features(
                snapshot_scope,
                scope,
                normalized_enstream_events_df,
                entity_cols=entity_cols,
                prefix=prefix,
                cfg=cfg,
            )
        )

    feature_frames.append(calculate_stability_features(snapshot_scope, scope, cfg))

    if state_transitions:
        feature_frames.append(
            calculate_state_transition_features(
                scoped_events,
                entity_cols=entity_cols,
                state_transitions=state_transitions,
                prefix=prefix,
                cfg=cfg,
            )
        )

    return join_feature_frames(
        base_df=scope_snapshot_df,
        feature_dfs=feature_frames,
        join_keys=entity_cols,
    )


def assemble_all_scope_features(
    snapshot_scope: Dict[str, DataFrame],
    changes_df: DataFrame,
    normalized_enstream_events_df: Optional[DataFrame],
    scopes: Sequence[str] = SCOPES,
    state_transitions: Sequence[Tuple[str, str]] = DEFAULT_STATE_TRANSITIONS,
    event_families: Sequence[str] = GENERIC_EVENT_FAMILIES,
    cfg: FeatureConfig = FeatureConfig(),
) -> Dict[str, DataFrame]:
    """:func:`assemble_scope_features` for each scope, keyed by scope name.

    The three frames are returned separately and deliberately not joined. They
    have different grains — one row per customer, per account, per number — and
    a customer-scope feature broadcast across that customer's four numbers is
    the same value four times, which is a choice about what a training row
    represents. That choice belongs to whoever builds the training table, and it
    is made in :mod:`trust_score_05.preprocessing`.
    """
    return {
        scope: assemble_scope_features(
            snapshot_scope,
            changes_df,
            normalized_enstream_events_df,
            scope,
            state_transitions=state_transitions,
            event_families=event_families,
            cfg=cfg,
        )
        for scope in scopes
    }


def feature_column_names(
    features_df: DataFrame,
    scope: str,
    cfg: FeatureConfig = FeatureConfig(),
) -> List[str]:
    """The feature columns of an assembled frame, excluding the entity keys.

    The model stages need this constantly — a feature matrix is the frame minus
    its keys — and computing it from ``cfg`` rather than by pattern-matching the
    prefix means a snapshot column that happens to start with ``cust_`` is not
    mistaken for a feature.
    """
    return feature_columns(features_df, cfg.entity_cols_for(scope))
