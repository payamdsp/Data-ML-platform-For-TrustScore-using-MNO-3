"""Behavioural feature engineering for Trust Score 0.5.

This package turns the Gold lineage tables and the raw event feeds into one wide
row per *reference* — an entity observed at a moment in time — at three scopes:
customer, account and MSISDN.

The layout follows the order the work happens in:

``config``
    Every constant and every column name. Nothing here computes anything.
``expressions``
    Small Spark column and DataFrame helpers used by everything below.
``events``
    Normalisation. Carrier change records, derived lineage events and EnStream
    API calls all become rows in one canonical event table.
``snapshots``
    References and scopes. Who are we describing, as of when, and which events
    belong to them.
``calculators``
    One function per feature family — frequency, recency, periodicity,
    burstness, entropy, velocity, state transitions, stability.
``enstream``
    The EnStream partner feed, which earns its own module because its events
    carry partner metadata and a volume-normalising weight that no other feed
    has.
``assemble``
    The entry points. :func:`assemble_scope_features` builds one scope;
    :func:`assemble_all_scope_features` builds all three.

The public names are re-exported here, so ``from trust_score_05.features import
assemble_scope_features`` is the intended way in and the module layout above
stays an implementation detail.
"""

from trust_score_05.features.assemble import (
    assemble_all_scope_features,
    assemble_scope_features,
    calculate_generic_event_feature_set,
    join_feature_frames,
)
from trust_score_05.features.calculators import (
    calculate_burstness_features,
    calculate_cumulative_event_slope,
    calculate_event_diversity_features,
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
    EVENT_METADATA_COLS,
    GENERIC_EVENT_FAMILIES,
    SCOPES,
    STATE_TRANSITION_EVENT_TYPES,
    SUPPORTED_CHANGE_TYPES,
    FeatureConfig,
    scope_prefix,
)
from trust_score_05.features.enstream import (
    calculate_enstream_event_diversity_features,
    calculate_enstream_event_frequency_features,
    calculate_enstream_event_novelty_features,
    calculate_enstream_partner_only_features,
    calculate_enstream_scoped_events,
)
from trust_score_05.features.events import (
    build_changes_table,
    build_mno_port_events,
    build_phone_number_change_events,
    normalize_account_change_events,
    normalize_enstream_events,
    preprocess_enstream_events,
)
from trust_score_05.features.snapshots import (
    attach_events_to_scope,
    build_reference_df,
    build_scope_snapshots,
)

__all__ = [
    "CANONICAL_EVENT_FAMILY",
    "DEFAULT_STATE_TRANSITIONS",
    "EVENT_METADATA_COLS",
    "GENERIC_EVENT_FAMILIES",
    "SCOPES",
    "STATE_TRANSITION_EVENT_TYPES",
    "SUPPORTED_CHANGE_TYPES",
    "FeatureConfig",
    "assemble_all_scope_features",
    "assemble_scope_features",
    "attach_events_to_scope",
    "build_changes_table",
    "build_mno_port_events",
    "build_phone_number_change_events",
    "build_reference_df",
    "build_scope_snapshots",
    "calculate_burstness_features",
    "calculate_cumulative_event_slope",
    "calculate_enstream_event_diversity_features",
    "calculate_enstream_event_frequency_features",
    "calculate_enstream_event_novelty_features",
    "calculate_enstream_partner_only_features",
    "calculate_enstream_scoped_events",
    "calculate_event_diversity_features",
    "calculate_event_frequency_features",
    "calculate_event_recency_features",
    "calculate_generic_event_feature_set",
    "calculate_periodicity_features",
    "calculate_recency_weighted_intensity",
    "calculate_stability_features",
    "calculate_state_transition_features",
    "calculate_temporal_entropy_features",
    "calculate_time_to_k_events",
    "join_feature_frames",
    "normalize_account_change_events",
    "normalize_enstream_events",
    "preprocess_enstream_events",
    "scope_prefix",
]
