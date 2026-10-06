"""Stage 1: Silver events -> canonical events.

Silver speaks each carrier's vocabulary. Gold speaks one. Eight of the eleven
canonical types are a direct rename of something a carrier sent; the other
three - ports, recycles and plan changes - plus two Rogers-only substitutes are
*derived*, because no carrier sends them and they have to be inferred from the
shape of the surrounding history.

Every derived rule is a window over one phone number's own history, or over a
pair of phone numbers that shared a SIM card. That is the whole reason slice
recomputation is worth its cost: the slice already holds the complete history of
every phone number it names, and the slice closure already put both sides of
every SIM-inferred pair in it, so none of these windows can straddle the edge of
the data being read. There is no lookback to configure and no special case at
the start of a batch.
"""

from __future__ import annotations

from datetime import datetime
from typing import Mapping

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from ..logging_utils import get_logger
from ..schemas import CANONICAL_EVENTS_SCHEMA, SLICE_KEY
from .common import (
    ac_event_id,
    align_to_schema,
    days_between,
    hours_between,
    join_sorted_ids,
    windows_from_config,
    with_audit_columns,
)

__all__ = [
    "DIRECT_TYPE_MAP",
    "build_canonical_events",
    "classify_direct",
    "resolve_rogers_status_change",
    "collapse_telus_plan_changes",
    "derive_rogers_activations",
    "rogers_imsi_pairs",
    "derive_rogers_number_changes",
    "mark_ports",
    "port_gaps",
    "mark_recycles",
]

LOGGER = get_logger(__name__)

#: The events a carrier actually sends, renamed. No inference involved.
DIRECT_TYPE_MAP: Mapping[str, str] = {
    "STATUS_CHANGE-AA": "account_activation",
    "STATUS_CHANGE-A": "account_reactivation",
    "STATUS_CHANGE-S": "account_suspension",
    "STATUS_CHANGE-C": "account_cancellation",
    "MSISDN_CHANGE_FROM": "phone_number_change_from",
    "MSISDN_CHANGE": "phone_number_change_to",
    "DEVICE_CHANGE": "device_change",
    "SIM_CHANGE": "sim_change",
}

#: Types that can open a lifecycle. Used here to decide what a port or a recycle
#: is allowed to reclassify.
_EXPLICIT_STARTS = ("account_activation", "account_reactivation")

#: Column set carried through the intermediate stages before the final projection.
_WORK_COLUMNS = (
    SLICE_KEY,
    "mno",
    "ac_event_type",
    "ac_event_ts",
    "correlation_id",
    "source_record_id",
)


def _empty_work(spark) -> DataFrame:
    return spark.createDataFrame(
        [],
        "phone_number_AC_hash string, mno string, ac_event_type string, "
        "ac_event_ts timestamp, correlation_id string, source_record_id string",
    )


# ==========================================================================
# 1. direct mapping
# ==========================================================================


def classify_direct(account_changes: DataFrame) -> DataFrame:
    """Rename carrier event types to canonical ones; drop what Gold does not model.

    ``ADDRESS_CHANGE`` and ``OWNERSHIP_CHANGE`` (Bell) are dropped here. They are
    real events and they are kept in Silver, but nothing in this lineage
    consumes them: they neither open, close nor evidence a lifecycle. Dropping
    them at the boundary is better than carrying them through four stages and
    filtering them at the end.
    """
    mapping = F.create_map(
        *[x for kv in DIRECT_TYPE_MAP.items() for x in (F.lit(kv[0]), F.lit(kv[1]))]
    )
    return (
        account_changes.withColumn("ac_event_type", mapping[F.col("event_type")])
        .where(F.col("ac_event_type").isNotNull())
        .select(
            F.col(SLICE_KEY),
            F.upper(F.col("mno")).alias("mno"),
            F.col("ac_event_type"),
            F.col("event_timestamp").alias("ac_event_ts"),
            F.col("correlationId").alias("correlation_id"),
            F.col("record_id").alias("source_record_id"),
        )
    )


# ==========================================================================
# 2. ROGERS: is a STATUS_CHANGE-C a cancellation or a suspension?
# ==========================================================================


def resolve_rogers_status_change(
    events: DataFrame, lookahead_days: int | None = None
) -> DataFrame:
    """Split Rogers' single ``STATUS_CHANGE-C`` into cancellation or suspension.

    Rogers sends one status event for two different things. The specification
    settles it by what follows: a ``STATUS_CHANGE-C`` is a **suspension** when
    other activity follows it at the same MNO, and a **cancellation** when
    nothing does, or when it coincides with a port.

    The previous code added two conditions the specification does not have. It
    required the following activity to fall inside a ten-day lookahead, and it
    counted other ``STATUS_CHANGE-C`` rows as "following activity". Together
    those meant a number with two cancellations eleven days apart read as one
    suspension followed by one cancellation - exactly backwards. Here the
    lookahead is unbounded by default and only events *of a different type*
    count as evidence of life.

    ``lookahead_days`` is kept configurable because an unbounded lookahead over
    a very long history is the one part of this rule that could get expensive,
    and an operator may want to bound it.
    """
    later = events.select(
        F.col(SLICE_KEY).alias("_p"),
        F.col("mno").alias("_m"),
        F.col("ac_event_type").alias("_t"),
        F.col("ac_event_ts").alias("_ts"),
    )
    rogers_c = events.where(
        (F.col("mno") == F.lit("ROGERS"))
        & (F.col("ac_event_type") == F.lit("account_cancellation"))
    )
    others = events.where(
        (F.col("mno") != F.lit("ROGERS"))
        | (F.col("ac_event_type") != F.lit("account_cancellation"))
    )

    cond = (
        (F.col(SLICE_KEY) == F.col("_p"))
        & (F.col("mno") == F.col("_m"))
        & (F.col("_ts") > F.col("ac_event_ts"))
        & (F.col("_t") != F.lit("account_cancellation"))
    )
    if lookahead_days is not None:
        cond = cond & (days_between("_ts", "ac_event_ts") <= F.lit(float(lookahead_days)))

    with_evidence = (
        rogers_c.join(later, cond, "left")
        .groupBy(*[F.col(c) for c in _WORK_COLUMNS])
        .agg(F.count("_t").alias("_followers"))
        .withColumn(
            "ac_event_type",
            F.when(F.col("_followers") > 0, F.lit("account_suspension")).otherwise(
                F.lit("account_cancellation")
            ),
        )
        .drop("_followers")
    )
    return others.unionByName(with_evidence.select(*_WORK_COLUMNS))


# ==========================================================================
# 3. TELUS: cancel-then-activate inside 24h is one plan change
# ==========================================================================


def collapse_telus_plan_changes(events: DataFrame, window_hours: int) -> DataFrame:
    """Replace a TELUS ``C`` -> ``AA`` pair inside the window with one ``plan_change``.

    The order matters and is now enforced. A cancellation followed by an
    activation is a subscriber moving to a different plan; an activation
    followed by a cancellation is a subscriber signing up and immediately
    leaving. The previous code matched the pair without regard to order and
    turned the second case into a plan change too, which erased a real
    cancellation.

    The synthesised event is timestamped at the cancellation - the moment the
    pair begins. Since ``plan_change`` is neutral it moves no lifecycle
    boundary, so the choice affects only where the event sorts; the beginning of
    the pair is the more useful of the two for anyone reading a timeline.
    """
    telus = events.where(F.col("mno") == F.lit("TELUS"))
    rest = events.where(F.col("mno") != F.lit("TELUS"))

    order = Window.partitionBy(SLICE_KEY, "mno").orderBy(
        F.col("ac_event_ts").asc(), F.col("source_record_id").asc()
    )
    flagged = (
        telus.withColumn("_next_type", F.lead("ac_event_type").over(order))
        .withColumn("_next_ts", F.lead("ac_event_ts").over(order))
        .withColumn("_next_rid", F.lead("source_record_id").over(order))
        .withColumn("_prev_type", F.lag("ac_event_type").over(order))
        .withColumn("_prev_ts", F.lag("ac_event_ts").over(order))
    )

    # ``lead`` and ``lag`` are null at the two ends of every partition, and a
    # comparison against null is null rather than false. Without the coalesce,
    # ``~(head | tail)`` is null for the first and last event of every phone
    # number, ``where`` drops them, and the stage quietly loses one event at each
    # end of every history. Each predicate is forced to a boolean where it is
    # built, so no caller can reintroduce the problem by negating it.
    is_pair_head = F.coalesce(
        (F.col("ac_event_type") == F.lit("account_cancellation"))
        & (F.col("_next_type") == F.lit("account_activation"))
        & (hours_between("_next_ts", "ac_event_ts") <= F.lit(float(window_hours)))
        & (hours_between("_next_ts", "ac_event_ts") >= F.lit(0.0)),
        F.lit(False),
    )
    is_pair_tail = F.coalesce(
        (F.col("ac_event_type") == F.lit("account_activation"))
        & (F.col("_prev_type") == F.lit("account_cancellation"))
        & (hours_between("ac_event_ts", "_prev_ts") <= F.lit(float(window_hours)))
        & (hours_between("ac_event_ts", "_prev_ts") >= F.lit(0.0)),
        F.lit(False),
    )

    collapsed = (
        flagged.where(is_pair_head)
        .withColumn("ac_event_type", F.lit("plan_change"))
        # Both source rows go into the seed: the synthesised event exists
        # because of the pair, and tracing it back to one half would be a lie.
        .withColumn(
            "source_record_id",
            F.concat_ws(",", F.array_sort(F.array(F.col("source_record_id"), F.col("_next_rid")))),
        )
        .select(*_WORK_COLUMNS)
    )
    untouched = flagged.where(~(is_pair_head | is_pair_tail)).select(*_WORK_COLUMNS)
    return rest.unionByName(untouched).unionByName(collapsed)


# ==========================================================================
# 4. ROGERS: activation inferred from the device feed
# ==========================================================================


def derive_rogers_activations(device_history: DataFrame) -> DataFrame:
    """Rogers sends no activation, so infer it from the first unmatched IMSI start.

    "Unmatched" means no ``IMSI_END`` has occurred for that phone number before
    the start - the subscriber was not previously on this network under a
    different SIM.

    The partition is ``phone_number_AC_hash`` **only**, and that is the entire
    correction. The previous code partitioned by ``(phone_number_AC_hash, imsi)``,
    so every SIM swap landed in a brand-new partition whose prior window was
    necessarily empty, and every SIM swap was therefore read as an activation.
    A Rogers subscriber who replaced a lost SIM three times acquired four
    account activations and, downstream, four lifecycles.

    One ``IMSI_START`` is evidence for two derivations and only one of them can
    be true of it. This function reads it as an activation; the number-change
    route reads it as the arrival of a number that used to be somebody's
    something else. For the phone number that *receives* a new number both fire
    on the same instant, so the SIM settles it: a SIM that has never carried
    another phone number is a subscriber joining the network, and a SIM that was
    already carrying one is that same subscriber renumbering. The second is not
    an activation and is excluded here.

    Nothing is lost by the exclusion. ``phone_number_change_to`` opens a
    lifecycle in the walk, so the receiving number still gets one; what goes is a
    duplicate claim about *why* it opened. The consequence worth naming is that
    ``recycle_ind`` and ``mno_port_ind`` are read off activation events, so a
    number change can no longer be tested for a recycle or a port. That is
    correct: it is neither. The number did not lie dormant and change hands, and
    it did not move between carriers.
    """
    imsi_events = device_history.where(
        (F.upper(F.col("mno")) == F.lit("ROGERS"))
        & F.col("event_type").startswith("IMSI_")
    ).select(
        F.col(SLICE_KEY),
        F.col("imsi"),
        F.col("event_type"),
        F.col("event_timestamp"),
        F.col("record_id"),
    )

    first_end = imsi_events.where(F.col("event_type").startswith("IMSI_END")).groupBy(
        SLICE_KEY
    ).agg(F.min("event_timestamp").alias("_first_end_ts"))

    starts = imsi_events.where(F.col("event_type").startswith("IMSI_START"))
    order = Window.partitionBy(SLICE_KEY).orderBy(
        F.col("event_timestamp").asc(), F.col("record_id").asc()
    )
    candidates = (
        starts.join(first_end, SLICE_KEY, "left")
        .where(
            F.col("_first_end_ts").isNull()
            | (F.col("event_timestamp") < F.col("_first_end_ts"))
        )
        .withColumn("_rn", F.row_number().over(order))
        .where(F.col("_rn") == 1)
    )

    # The SIM's earlier life, if it had one, under a different phone number.
    # "Earlier" is what makes this the *arrival* half and not both halves: the
    # number the SIM started on has nothing before it, so it keeps its
    # activation, and only the number that inherited the SIM loses one.
    #
    # A null ``imsi`` never matches, so a feed that omits the column leaves the
    # activation alone rather than deriving a number change it cannot evidence.
    elsewhere = imsi_events.select(
        F.col("imsi").alias("_imsi"),
        F.col(SLICE_KEY).alias("_phone"),
        F.col("event_timestamp").alias("_other_ts"),
    )
    genuine = candidates.join(
        elsewhere,
        (F.col("imsi") == F.col("_imsi"))
        & (F.col(SLICE_KEY) != F.col("_phone"))
        & (F.col("_other_ts") < F.col("event_timestamp")),
        "left_anti",
    )

    return genuine.select(
        F.col(SLICE_KEY),
        F.lit("ROGERS").alias("mno"),
        F.lit("account_activation").alias("ac_event_type"),
        F.col("event_timestamp").alias("ac_event_ts"),
        F.lit(None).cast("string").alias("correlation_id"),
        F.col("record_id").alias("source_record_id"),
    )


# ==========================================================================
# 5. ROGERS: number change inferred from one SIM card carrying two numbers
# ==========================================================================


def rogers_imsi_pairs(
    device_history: DataFrame, max_gap_days: int | None = None
) -> DataFrame:
    """The pairing itself: one SIM card, two phone numbers, in that order.

    Rogers sends no ``MSISDN_CHANGE`` at all, which is why Rogers produced no
    account chains whatsoever under the previous code that prepared the current
    tables: chains were built by matching correlation ids, and Rogers sends no
    correlation id either. The only remaining evidence is the device feed.

    The evidence is the **SIM card**, not the handset. A handset is a possession -
    sold, lent, traded in, repaired - so one IMEI carrying two phone numbers is
    as consistent with two unrelated people as with one person renumbering, and
    the earlier IMEI version of this rule needed a five-day window to be usable
    at all: the window was doing the work the evidence should have been doing. A
    SIM card *is* the subscription. A subscriber who changes their phone number
    keeps their SIM, so the IMSI is the thing that stays put across the change,
    and the same IMSI on two phone numbers is close to a statement that the
    subscription continued. (The handset is still the right evidence elsewhere,
    for the opposite reason: a change of carrier means a new SIM and therefore a
    new IMSI, while the handset survives. The IMSI carries identity within a
    carrier and the IMEI carries it across one.)

    Because the evidence is stronger the temporal filter stops being the test.
    Two phone numbers eleven months apart on one SIM are one account, and that is
    the case this route exists to recover - the window used to exclude it
    silently. ``max_gap_days`` is an optional outer bound for the day SIM cards
    turn out to be reissued to different subscribers; ``None``, the default,
    applies no bound at all.

    What replaces the window is ``overlap_days``: how long both phone numbers
    were live on the one SIM. Nothing is filtered on it here, because an overlap
    is usually the feed's clocks rather than the subscription's - the ``IMSI_END``
    for the old number lands after the ``IMSI_START`` for the new one - and the
    account stage marks the link down instead of throwing the pair away.

    The ``from`` timestamp is the **end** of A's association with the SIM: the
    ``IMSI_END`` for ``(A, imsi)`` if there is one, and otherwise the moment B
    appeared. The previous code timestamped it at the moment A *started* on the
    SIM, which put the end of A's lifecycle before its beginning whenever A had
    held the number for any length of time.
    """
    imsi_events = device_history.where(
        (F.upper(F.col("mno")) == F.lit("ROGERS"))
        & F.col("imsi").isNotNull()
        & F.col("event_type").startswith("IMSI_")
    ).select(
        F.col(SLICE_KEY),
        F.col("imsi"),
        F.col("event_type"),
        F.col("event_timestamp"),
        F.col("record_id"),
    )

    # One row per (phone, imsi) association: when it started, when it ended.
    assoc = imsi_events.groupBy(SLICE_KEY, "imsi").agg(
        F.min("event_timestamp").alias("start_ts"),
        F.max(
            F.when(
                F.col("event_type") == F.lit("IMSI_END"), F.col("event_timestamp")
            )
        ).alias("end_ts"),
        F.min("record_id").alias("record_id"),
    )

    a = assoc.select(
        F.col("imsi"),
        F.col(SLICE_KEY).alias("a_phone"),
        F.col("start_ts").alias("a_start"),
        F.col("end_ts").alias("a_end"),
        F.col("record_id").alias("a_rid"),
    )
    b = assoc.select(
        F.col("imsi"),
        F.col(SLICE_KEY).alias("b_phone"),
        F.col("start_ts").alias("b_start"),
        F.col("record_id").alias("b_rid"),
    )

    pairs = (
        a.join(b, "imsi")
        .where(F.col("a_phone") != F.col("b_phone"))
        .where(F.col("b_start") > F.col("a_start"))
        # The gap that matters runs from the *end* of A's association to the
        # start of B's. Measuring it from A's start instead means a subscriber
        # who held A for a year and then changed their number is never detected,
        # because a year is not a few days - and that is the common case, which
        # is the case this route exists to recover.
        #
        # Where the feed sent no end for A, A ran up to B by assumption and the
        # only measurable gap is from A's start. So the fallback carries its own
        # weight rather than being a default.
        .withColumn("_transition_ts", F.coalesce(F.col("a_end"), F.col("a_start")))
        # How long both numbers were live on the one SIM. Signed the other way
        # from the gap, so it is only non-zero where ``a_end`` lands after B
        # began; where the feed sent no end for A there is no measurable overlap
        # and the answer is zero rather than null, because a null here would
        # propagate into a link confidence and read as "we did not look".
        .withColumn(
            "overlap_days",
            F.when(
                F.col("a_end").isNotNull() & (F.col("a_end") > F.col("b_start")),
                days_between("a_end", "b_start"),
            ).otherwise(F.lit(0.0)),
        )
        # The transition timestamp: A's association ended here, B's began here.
        .withColumn("from_ts", F.coalesce(F.col("a_end"), F.col("b_start")))
        .withColumn(
            "source_record_id",
            F.concat_ws(",", F.array_sort(F.array(F.col("a_rid"), F.col("b_rid")))),
        )
    )

    if max_gap_days is not None:
        pairs = pairs.where(
            days_between("b_start", "_transition_ts") <= F.lit(float(max_gap_days))
        )

    # A SIM that carried many numbers would otherwise produce a change for every
    # pair. Keep only each number's nearest successor.
    #
    # The partition is the phone number and *not* ``(a_phone, imsi)``, and the
    # two are not interchangeable. A number change is a fact about the
    # subscriber: A became exactly one other number, once. Partitioning by the
    # SIM as well would let a phone number that has sat on two SIMs emit two
    # successors, and the account walk would then throw one of them away as a
    # conflicting edge (``chains.build_chains`` keeps the earliest and reports
    # the rest), so the answer would be the same edge arrived at through a
    # reported repair instead of a clean selection - a worse route to the same
    # place, and one that would inflate ``edges_dropped`` for no reason.
    #
    # ``imsi`` goes into the ordering instead, as the last tiebreak, and it is
    # there because without it the ordering was not total. Two IMSIs carrying the
    # same ``(a_phone, b_phone)`` at the same ``b_start`` - eSIM re-provisioning,
    # where the subscriber's SIM profile is reissued and both profiles are on
    # file - tied on every key in this window, and whichever row the shuffle
    # happened to leave at rank 1 decided ``overlap_days``, which decides HIGH
    # against LOW in the account stage. Same input, same code, potentially
    # different published confidence. Smallest IMSI wins: the choice is arbitrary
    # but it has to be *made*, and the value is a stable property of the row
    # rather than of the run.
    nearest = Window.partitionBy("a_phone").orderBy(
        F.col("b_start").asc(), F.col("b_phone").asc(), F.col("imsi").asc()
    )
    return pairs.withColumn("_rn", F.row_number().over(nearest)).where(
        F.col("_rn") == 1
    )


def derive_rogers_number_changes(
    device_history: DataFrame, max_gap_days: int | None = None
) -> DataFrame:
    """The canonical event pair projected out of :func:`rogers_imsi_pairs`.

    Two events, because a number change is two things happening: one phone
    number stopped being the subscriber's and another started. The walk reads
    ``phone_number_change_from`` as closing a lifecycle and
    ``phone_number_change_to`` as opening one, so the pair is what turns one
    subscriber's two numbers into two lifecycles of one account.
    """
    pairs = rogers_imsi_pairs(device_history, max_gap_days)

    out_from = pairs.select(
        F.col("a_phone").alias(SLICE_KEY),
        F.lit("ROGERS").alias("mno"),
        F.lit("phone_number_change_from").alias("ac_event_type"),
        F.col("from_ts").alias("ac_event_ts"),
        F.lit(None).cast("string").alias("correlation_id"),
        F.col("source_record_id"),
    )
    out_to = pairs.select(
        F.col("b_phone").alias(SLICE_KEY),
        F.lit("ROGERS").alias("mno"),
        F.lit("phone_number_change_to").alias("ac_event_type"),
        F.col("b_start").alias("ac_event_ts"),
        F.lit(None).cast("string").alias("correlation_id"),
        F.col("source_record_id"),
    )
    return out_from.unionByName(out_to)


# ==========================================================================
# 6. ports
# ==========================================================================


def _mno_segments(events: DataFrame, tie_break_col: str) -> DataFrame:
    """Number each phone number's contiguous runs of events at one carrier."""
    order = Window.partitionBy(SLICE_KEY).orderBy(
        F.col("ac_event_ts").asc(), F.col(tie_break_col).asc()
    )
    tagged = events.withColumn(
        "_prev_mno", F.lag("mno").over(order)
    ).withColumn(
        "_seg_start",
        F.when(F.col("_prev_mno").isNull() | (F.col("_prev_mno") != F.col("mno")), 1).otherwise(0),
    )
    return tagged.withColumn(
        "_seg_id", F.sum("_seg_start").over(order.rowsBetween(Window.unboundedPreceding, 0))
    )


def _segment_handovers(seg: DataFrame, window_days: int) -> DataFrame:
    """One row per MNO segment, with the handover that opened it.

    ``_gap_days`` is the elapsed time between the previous carrier's last event
    and this carrier's first - the handover. It is null for a phone number's
    first segment, and null is not zero: one means "there was no previous
    carrier", the other means "the handover was instant".

    ``_is_port`` is the same test :func:`mark_ports` has always applied, kept
    here so that the classification and the measurement of a port cannot come
    apart. The gap is the measurement; ``window_days`` is the outer bound on
    calling it a port at all. How *sure* of the port we are is a tighter bound
    applied later, in the customer stage - two different quantities, and
    conflating them would either refuse real ports or vouch for guessed ones.
    """
    bounds = seg.groupBy(SLICE_KEY, "_seg_id", "mno").agg(
        F.min("ac_event_ts").alias("_seg_from"),
        F.max("ac_event_ts").alias("_seg_to"),
    )
    seg_order = Window.partitionBy(SLICE_KEY).orderBy(
        F.col("_seg_from").asc(), F.col("_seg_id").asc()
    )
    return (
        bounds.withColumn("_prev_to", F.lag("_seg_to").over(seg_order))
        .withColumn("_prev_mno", F.lag("mno").over(seg_order))
        .withColumn("_gap_days", days_between("_seg_from", "_prev_to"))
        .withColumn(
            "_is_port",
            F.col("_prev_to").isNotNull()
            & (F.col("_prev_mno") != F.col("mno"))
            & (F.col("_gap_days") <= F.lit(float(window_days)))
            & (F.col("_gap_days") >= F.lit(0.0)),
        )
        .select(SLICE_KEY, "_seg_id", "_seg_from", "_gap_days", "_is_port")
    )


def mark_ports(events: DataFrame, window_days: int) -> DataFrame:
    """Reclassify the opening event of a new MNO segment as a port, where one fits.

    A port is visible only as a *shape*: the same phone number stops being
    active at one carrier and starts being active at another, close together in
    time. Neither carrier says "port"; the receiving one, at best, sends an
    ordinary activation.

    So MNO segments are built per phone number - a contiguous run of events at
    one carrier - and where segment *k+1* begins within the window of segment
    *k* ending, segment *k+1*'s first event becomes
    ``account_activation_mno_port``. Emitted on the receiving side, which is
    where the new lifecycle opens.

    When that first event is not itself a start (a device change, say, because
    the receiving carrier sent nothing else), the type is still rewritten: the
    port is the better description of what happened, and losing a device change
    to gain a port is the right trade. The device change is not lost from
    Silver, only from this lineage's view of that instant.
    """
    seg = _mno_segments(events, "source_record_id")
    bounds = _segment_handovers(seg, window_days).select(
        SLICE_KEY, "_seg_id", "_is_port"
    )

    first_in_seg = Window.partitionBy(SLICE_KEY, "_seg_id").orderBy(
        F.col("ac_event_ts").asc(), F.col("source_record_id").asc()
    )
    return (
        seg.join(bounds, [SLICE_KEY, "_seg_id"], "left")
        .withColumn("_rn", F.row_number().over(first_in_seg))
        .withColumn(
            "ac_event_type",
            F.when(
                F.col("_is_port") & (F.col("_rn") == 1),
                F.lit("account_activation_mno_port"),
            ).otherwise(F.col("ac_event_type")),
        )
        .select(*_WORK_COLUMNS)
    )


def port_gaps(events: DataFrame, window_days: int) -> DataFrame:
    """How tight each port's handover was: ``(phone, port_ts, port_gap_days)``.

    The customer stage needs this and cannot compute it for itself. By the time
    the lifecycle walk has run, the losing lifecycle has been closed *at the
    arriving lifecycle's first event* - carriers do not announce the subscriber
    leaving - so the gap between the two lifecycle boundaries is zero by
    construction. The real handover is only visible in the events, which is why
    it is measured here.

    ``events`` is the published canonical-event frame, so the segments are
    recomputed rather than carried through from :func:`mark_ports`. That is
    deliberate and it is the safer of the two. Those events are the ones the
    lifecycle walk saw, so a segment boundary found here lines up exactly with
    the lifecycle a port edge will connect, whereas :func:`mark_ports` runs
    before the dedupe and its segments are one step older than the lifecycles.

    ``port_ts`` is the arriving segment's first event, which is the arriving
    lifecycle's ``lifecycle_start_ts`` - that is the join key back.
    """
    seg = _mno_segments(events, "ac_event_id")
    return (
        _segment_handovers(seg, window_days)
        .where(F.col("_is_port"))
        .select(
            F.col(SLICE_KEY),
            F.col("_seg_from").alias("port_ts"),
            F.col("_gap_days").alias("port_gap_days"),
        )
    )


# ==========================================================================
# 7. recycles
# ==========================================================================


def mark_recycles(events: DataFrame, threshold_days: int) -> DataFrame:
    """Reclassify an explicit start after a long total silence as a recycle.

    A phone number that goes completely quiet for three months and then starts
    again has, in all likelihood, been reassigned to a different subscriber.
    That is a different person, so it is a different lifecycle and it must not
    be joined to what came before.

    "Completely quiet" is the correction. The previous code treated any event
    91 days after a cancellation as a recycle, without checking whether the
    number had been active in between - so a number cancelled in January,
    reactivated in February and still in daily use acquired a spurious recycle
    in April. Here the test is against the *immediately preceding* event of any
    kind, which is the same thing as requiring the gap to be empty.

    Ports are excluded. A port already explains why activity resumed elsewhere,
    and it explains it better.
    """
    order = Window.partitionBy(SLICE_KEY).orderBy(
        F.col("ac_event_ts").asc(), F.col("source_record_id").asc()
    )
    return (
        events.withColumn("_prev_ts", F.lag("ac_event_ts").over(order))
        .withColumn(
            "ac_event_type",
            F.when(
                F.col("ac_event_type").isin(list(_EXPLICIT_STARTS))
                & F.col("_prev_ts").isNotNull()
                & (days_between("ac_event_ts", "_prev_ts") >= F.lit(float(threshold_days))),
                F.lit("account_activation_number_recycle"),
            ).otherwise(F.col("ac_event_type")),
        )
        .select(*_WORK_COLUMNS)
    )


# ==========================================================================
# the stage
# ==========================================================================


def build_canonical_events(
    account_changes: DataFrame,
    device_history: DataFrame,
    run_ts: datetime,
    windows: Mapping[str, int | None] | None = None,
    cfg=None,
) -> DataFrame:
    """Produce the complete set of canonical events for the slice.

    ``account_changes`` and ``device_history`` are the *full history* of the
    slice's phone numbers, not the batch. Order of application matters:

    1. direct renames
    2. Rogers ``STATUS_CHANGE-C`` resolved, before anything counts cancellations
    3. Rogers derived activations and number changes folded in, so the port and
       recycle rules can see them
    4. TELUS plan-change pairs collapsed, before ports, so a plan change is
       never mistaken for a cancellation followed by a fresh activation
    5. ports, then recycles - in that order, because a port is the better
       explanation whenever both would fit
    """
    win = dict(windows) if windows is not None else windows_from_config(cfg)

    events = classify_direct(account_changes)
    events = resolve_rogers_status_change(events, win["rogers_status_lookahead_days"])
    events = events.unionByName(derive_rogers_activations(device_history))
    events = events.unionByName(
        derive_rogers_number_changes(
            device_history, win["rogers_imsi_reuse_max_gap_days"]
        )
    )
    events = collapse_telus_plan_changes(
        events, int(win["telus_plan_change_window_hours"])
    )
    events = mark_ports(events, int(win["mno_port_window_days"]))
    events = mark_recycles(events, int(win["number_recycle_threshold_days"]))

    # Two Silver rows describing the same instant in the same way are one
    # canonical event. This is not the Silver replay guard - it is a genuine
    # collapse, and the surviving row carries both record ids so the seed
    # (and therefore the id) is stable no matter which physical row arrived
    # first.
    deduped = events.groupBy(
        SLICE_KEY, "mno", "ac_event_type", "ac_event_ts"
    ).agg(
        F.min("correlation_id").alias("correlation_id"),
        join_sorted_ids(F.col("source_record_id")).alias("source_record_id"),
    )

    keyed = deduped.withColumn(
        "ac_event_id",
        ac_event_id(
            F.col(SLICE_KEY),
            F.col("mno"),
            F.col("ac_event_type"),
            F.col("ac_event_ts"),
            F.col("source_record_id"),
        ),
    )
    stamped = with_audit_columns(keyed, F.lit(run_ts).cast("timestamp"))
    return align_to_schema(stamped, CANONICAL_EVENTS_SCHEMA)
