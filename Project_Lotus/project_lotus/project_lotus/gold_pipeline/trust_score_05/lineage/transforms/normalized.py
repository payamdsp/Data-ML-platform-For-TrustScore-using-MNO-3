"""Stage 5: canonical events flattened onto the account and customer they fall in.

The serving shape. Everything above this stage is structure - which events form a
lifecycle, which lifecycles form an account, which accounts form a customer - and
this stage turns that structure back into one row per event with the structure
attached, which is what the scoring API reads.

It is a pure function of the four tables above it, restricted to the slice, so
incrementally it is the simplest of the five: rebuild for the slice, merge like
the rest. Nothing in the audit found anything wrong with it.

The linkage is an interval join. An event belongs to the customer mapping row
whose ``[from_ts, to_ts]`` contains it, on the same phone number and carrier, and
where two intervals contain the same instant the later one wins. Events that fall
inside no interval are kept with a null ``customer_id`` and ``acct_id`` rather
than dropped: the event happened, and a Gold table that silently loses events is
worse than one that admits it could not place them.

**Both bounds are closed, and that is a correction.** The interval itself is a
statement about *state*, and for state the half-open ``[from_ts, to_ts)`` is
right - it is what stops two adjacent accounts from both claiming the instant one
ends and the next begins. But this join places *events*, and the event that sits
exactly on ``to_ts`` is the event that closed the interval. Excluding it means
every cancellation and every outgoing number change - the end of every account
that ever ended - arrives in the serving table with a null ``customer_id``, so
"when did this customer cancel?" cannot be answered from the table whose entire
purpose is to answer it.

Closing the upper bound does not reintroduce the double-claim, because the
deduplication window below already resolves it: an instant claimed by both the
interval that ends there and the interval that begins there goes to the later
one, which is the correct answer, since an event at that instant with a following
interval is the event that *opened* it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Mapping

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from ..logging_utils import get_logger
from ..schemas import (
    EVENT_FAMILY_BY_TYPE,
    NORMALIZED_EVENTS_SCHEMA,
    SLICE_KEY,
)
from .common import align_to_schema, with_audit_columns

__all__ = [
    "DEFAULT_EVENT_WEIGHT",
    "DERIVED_EVENT_TYPES",
    "EVENT_SOURCE",
    "build_normalized_events",
    "weights_from_config",
]

LOGGER = get_logger(__name__)

#: The feed these events came from. The column exists so that API-sourced events
#: can be told apart once TransUnion and Port PS are in scope; today every row
#: this pipeline writes comes from the carrier account-changes feed.
EVENT_SOURCE = "account_changes_batch"

#: Canonical types that no carrier sends - Gold worked them out from the shape of
#: the surrounding history. They are called out in ``updated_source`` so a
#: consumer can weight inferred evidence differently from stated evidence.
#:
#: Two of the five derived types are not visible here, and it is worth being
#: plain about it. The ROGERS activations and number changes inferred from the
#: device feed are emitted as ordinary ``account_activation`` and
#: ``phone_number_change_*`` events, and the published canonical-events table
#: keeps no column recording that they were derived, so they are reported as
#: carrier-sent. Fixing that means widening the canonical-events schema, which is
#: a schema change and therefore out of scope for this version.
DERIVED_EVENT_TYPES = (
    "account_activation_mno_port",
    "account_activation_number_recycle",
    "plan_change",
)

#: Scoring weights are owned by the model, not by this pipeline. Until that
#: mapping is handed over, every event weighs the same and the value is
#: overridable per event type from ``normalized.event_weights`` in config, so a
#: change of weights is a config change and not a code change.
DEFAULT_EVENT_WEIGHT = 1.0


def weights_from_config(cfg) -> dict[str, float]:
    """Read ``normalized.event_weights`` as a plain ``{event_type: weight}`` dict."""
    section = (cfg.get("normalized.event_weights", {}) or {}) if cfg is not None else {}
    return {str(k): float(v) for k, v in section.items()}


def _event_weight(weights: Mapping[str, float]) -> Column:
    if not weights:
        return F.lit(DEFAULT_EVENT_WEIGHT).cast("double")
    mapping = F.create_map(
        *[x for kv in weights.items() for x in (F.lit(kv[0]), F.lit(float(kv[1])))]
    )
    return F.coalesce(
        mapping[F.col("ac_event_type")], F.lit(DEFAULT_EVENT_WEIGHT)
    ).cast("double")


def _event_family() -> Column:
    mapping = F.create_map(
        *[x for kv in EVENT_FAMILY_BY_TYPE.items() for x in (F.lit(kv[0]), F.lit(kv[1]))]
    )
    return F.coalesce(mapping[F.col("ac_event_type")], F.lit("other"))


def build_normalized_events(
    canonical_events: DataFrame,
    customers: DataFrame,
    device_history: DataFrame | None,
    run_ts: datetime,
    cfg=None,
    with_counts: bool = False,
) -> DataFrame:
    """Attach the account and customer to every canonical event in the slice.

    ``customers`` is the work frame from :mod:`customers` - lifecycle grain, with
    ``from_ts`` / ``to_ts`` already clipped to the customer's view of the
    account, which is what makes an event that arrived after the subscriber
    ported land on the right side of the boundary.

    ``device_history`` supplies ``phone_number_AT_hash``, which the
    account-changes feed does not carry. It is optional; without it the column is
    null, which is a legitimate state for a phone number Gold has only ever seen
    on the account feed.

    ``with_counts`` asks for the unlinked-event warning, which costs a full pass
    over the result. Off by default so that building the plan is free.
    """
    weights = weights_from_config(cfg)

    intervals = customers.select(
        F.col("customer_id"),
        F.col("acct_id"),
        F.col(SLICE_KEY).alias("_link_phone"),
        F.col("mno").alias("_link_mno"),
        F.col("from_ts").alias("_link_from"),
        F.col("to_ts").alias("_link_to"),
    )

    # Both bounds closed. See the module docstring: the event on ``to_ts`` is the
    # event that closed the interval, and it belongs to it. A zero-length
    # interval - a lone cancellation, whose start had to be inferred at the same
    # instant - contains exactly one instant and would contain nothing at all
    # under a half-open upper bound.
    contains = (
        (F.col(SLICE_KEY) == F.col("_link_phone"))
        & (F.col("mno") == F.col("_link_mno"))
        & (F.col("ac_event_ts") >= F.col("_link_from"))
        & (F.col("_link_to").isNull() | (F.col("ac_event_ts") <= F.col("_link_to")))
    )

    linked = canonical_events.join(intervals, contains, "left")

    # An event can only belong to one interval. Two cases reach here with more
    # than one candidate. The first is genuine overlap, which should not happen
    # and which DQ checks for. The second is the boundary the closed upper bound
    # creates: one interval ends where the next begins, so both contain that
    # instant. The later interval wins in both cases - it is the more recent
    # statement about the same phone number, and at a boundary it is the one the
    # event opened.
    # `customer_id` is the last tiebreak and it is not decoration. `customers` is
    # lifecycle grain, so one account contributes one row per lifecycle, and two
    # lifecycles of the same account can share a `from_ts` when one of them is
    # zero-duration. On that tie the first two keys are equal and `row_number`
    # picks whichever the shuffle happened to order first. The two rows agree on
    # every column this stage publishes, so the *output* was never wrong - but
    # the plan was non-deterministic, and a non-deterministic plan is one
    # optimiser change away from being wrong. Total order, no drift.
    pick = Window.partitionBy("ac_event_id").orderBy(
        F.col("_link_from").desc_nulls_last(),
        F.col("acct_id").asc_nulls_last(),
        F.col("customer_id").asc_nulls_last(),
    )
    best = (
        linked.withColumn("_rn", F.row_number().over(pick))
        .where(F.col("_rn") == 1)
        .drop("_rn")
    )

    if device_history is not None:
        at_hash = (
            device_history.where(F.col("phone_number_AT_hash").isNotNull())
            .groupBy(SLICE_KEY)
            .agg(F.max("phone_number_AT_hash").alias("phone_number_AT_hash"))
        )
        # No broadcast hint. `at_hash` is one row per phone number carrying a
        # 64-character hash, so at national scale it is on the order of a
        # gigabyte - not a lookup table. Hinting it killed the 24 August 2026 run
        # of notebook 05 after eight minutes with
        #
        #   Total size of serialized results of 235 tasks (1026.3 MiB) is bigger
        #   than spark.driver.maxResultSize (1024.0 MiB)
        #
        # raised from `BroadcastExchangeExec`, which is the driver collecting the
        # whole frame in order to ship it. The hint overrides the planner's own
        # size estimate, so no amount of statistics would have saved it; the only
        # fix is not to ask. Left as a plain join, AQE still broadcasts it on a
        # slice small enough to be worth broadcasting and shuffles it when it is
        # not, which is the behaviour that was wanted in the first place.
        best = best.join(at_hash, SLICE_KEY, "left")
    else:
        best = best.withColumn(
            "phone_number_AT_hash", F.lit(None).cast("string")
        )

    out = (
        best.withColumn("event_id", F.col("ac_event_id"))
        .withColumn("event_ts", F.col("ac_event_ts"))
        .withColumn("event_type", F.col("ac_event_type"))
        .withColumn("event_family", _event_family())
        .withColumn("event_source", F.lit(EVENT_SOURCE))
        .withColumn("event_weight", _event_weight(weights))
        # API-sourced context. Null until TransUnion and Port PS are in scope;
        # the columns exist so that adding them is a backfill and not a schema
        # migration.
        .withColumn("industry", F.lit(None).cast("string"))
        .withColumn("service_provider_name", F.lit(None).cast("string"))
        .withColumn("use_case", F.lit(None).cast("string"))
        .withColumn(
            "updated_source",
            F.when(
                F.col("ac_event_type").isin(list(DERIVED_EVENT_TYPES)),
                F.lit("enstream_event_derived"),
            ).otherwise(F.lit("enstream_event")),
        )
    )

    stamped = with_audit_columns(out, F.lit(run_ts).cast("timestamp"))
    result = align_to_schema(stamped, NORMALIZED_EVENTS_SCHEMA)

    # Off by default, and that is a change. This is a *builder*: it is supposed
    # to return a plan, and the caller decides when to pay for it. The count that
    # used to be unconditional here was a full pass over every canonical event,
    # taken before the caller had a chance to cache anything, and notebook 05
    # then took the same count again two lines later - so the warning cost a
    # whole extra scan of the largest table in the estate to say something the
    # notebook was about to print anyway.
    if with_counts:
        unlinked = result.where(F.col("acct_id").isNull()).count()
        if unlinked:
            LOGGER.warning(
                "normalized: %d event(s) fell inside no account interval and were "
                "written with a null acct_id. This is usually a phone number whose "
                "lifecycles were all filtered upstream; it is never dropped data.",
                unlinked,
            )
    return result
