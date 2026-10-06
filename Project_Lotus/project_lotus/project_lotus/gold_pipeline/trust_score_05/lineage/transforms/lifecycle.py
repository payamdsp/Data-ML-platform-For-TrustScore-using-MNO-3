"""Stage 2: canonical events -> lifecycles.

A lifecycle is one continuous stretch of a phone number being alive at one
carrier. Building them is the only genuinely sequential step in the pipeline:
whether an event opens, closes or merely decorates a lifecycle depends on
whether one is currently open, which depends on every event before it.

So the walk is written as a plain Python function over one phone number's
events, and Spark's only job is to group the events and collect the results.
That is a deliberate choice. The same logic expressed as a chain of window
functions would be perhaps forty lines of ``lag``/``lead``/``sum-over`` that
nobody could check against the decision table in the design document, and the
edge cases - two cancellations in a row, a reactivation with nothing open, a
plan change as the only evidence a number exists - are exactly where such a
chain would quietly go wrong. Here each of those is a branch you can read, and
:func:`walk_phone` can be called from a test with four dictionaries and no
SparkSession at all.

The cost is a Python round trip per slice. The slice is bounded by construction,
so it is affordable, and it is the right trade against logic this fiddly.

Boundary confidence is computed here too, in the same pass, for the same reason
the walk is one function: ``start_confidence`` and ``end_confidence`` say how
firm each edge of the lifecycle is, and the only place that is known is the
branch that decided the edge. Computing it anywhere else would mean
reconstructing which branch ran, from flags that describe it only approximately.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from ..logging_utils import get_logger
from ..spark import shuffle_partitions
from ..schemas import (
    END_EVENTS,
    FORCE_NEW_EVENTS,
    LIFECYCLE_SCHEMA,
    LIFECYCLE_START_FLOOR,
    MNO_DOMAIN,
    NEUTRAL_EVENTS,
    SCHEMA_VERSION,
    SLICE_KEY,
)
from .common import align_to_schema, with_audit_columns

__all__ = [
    "LIFECYCLE_WORK_SCHEMA",
    "walk_phone",
    "build_lifecycles",
    "sha256_join",
    "START_TOKENS",
    "END_TOKENS",
    "LOW_BOUNDARY_MNOS",
    "event_rank",
    "DUPLICATE_END_WINDOW",
]

LOGGER = get_logger(__name__)


def sha256_join(*parts: Any) -> str:
    """The Python twin of ``sha2(concat_ws("|", ...), 256)``.

    Identical output to the Spark expression for the same inputs, which is what
    lets the walk compute ``lifecycle_uid`` in Python without the keys drifting
    from the ones the SQL-side helpers produce. Nulls become empty strings, same
    as ``coalesce(c, "")``.
    """
    joined = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


#: canonical event type -> the token that describes how the lifecycle opened.
START_TOKENS: Mapping[str, str] = {
    "account_activation": "explicit_activation",
    "account_reactivation": "explicit_activation",
    "account_activation_mno_port": "explicit_mno_port",
    # 90, not 91: the rule stage 1 applies is ``days_between(...) >= 90.0`` on
    # elapsed days (``canonical_events.py:702``, ``common.NUMBER_RECYCLE_
    # THRESHOLD_DAYS``), so the whole ``[90, 91)`` band is a recycle too. The
    # 91 this token used to carry was left over from the previous
    # implementation; ``docs/confidence_design.md`` section 7 records 90 as the
    # deliberate choice, paired with the ``<= 90`` in its section F.
    "account_activation_number_recycle": (
        "number_recycle_after_gap_greater_or_equal_90_days"
    ),
    "phone_number_change_to": "explicit_phone_number_change",
    "plan_change": "inferred_start_from_plan_change_only",
}

_DEFAULT_START_TOKEN = "inferred_start_from_first_observed_activity"

#: how the lifecycle closed -> token.
END_TOKENS: Mapping[str, str] = {
    "account_cancellation": "_to_explicit_cancellation",
    "phone_number_change_from": "_to_explicit_phone_number_change",
    "__inferred__": "_to_inferred_end_from_next_start",
    "__open__": "_open_no_cancellation",
}

#: What the walk does with two events that landed in the same instant.
#:
#: ``canonical_events.py:734`` dedups on
#: ``(phone, mno, ac_event_type, ac_event_ts)``, so two events surviving at one
#: instant necessarily carry *different types* - which is precisely the case
#: where the order changes the answer rather than only the order of the rows. The
#: August 2026 audit found 1,658,031 zero-duration lifecycles out of 77,172,342,
#: and in every one of them the tiebreak deciding the outcome was
#: ``ac_event_id``, a SHA-256 hash. Reproducible, and arbitrary: a cancellation
#: and an activation in the same second published the number as dead if the
#: activation's hash sorted first and as live and open if the cancellation's did.
#:
#: The rule, decided as policy rather than derived from the code: at one instant,
#: **end-like events are applied before start-like ones, and neutral events
#: last.**
#:
#: Closing before opening is what preserves the invariant the rest of the walk is
#: built on - at most one lifecycle is open at any instant. Applying the start
#: first forces the walk to *infer* a close for a lifecycle whose real, observed
#: close is the very next event in the list, and to write ``LOW`` on a boundary a
#: carrier had actually told us about.
#:
#: Neutral events go last because a plan change landing in the same second as an
#: activation belongs to the lifecycle that activation has just opened, not to
#: the one the cancellation has just closed. Ranked with the starts it would
#: instead open an inferred lifecycle of its own in the instant between the two.
_RANK_END_LIKE = 0
_RANK_START_LIKE = 1
_RANK_NEUTRAL = 2

#: The two ranked sets, derived from the vocabularies the walk's own branches
#: read rather than restated as a third and fourth list. A new event type given a
#: rank here and a different meaning in the branches below would be a silent
#: reordering of somebody's history, and deriving them is the only way the two
#: cannot drift apart.
#:
#: ``plan_change`` is subtracted deliberately even though ``START_TOKENS`` names
#: it. It has a start token because a plan change with nothing open is the only
#: evidence that a lifecycle exists at all - but it is a neutral event, it opens
#: a lifecycle only in the neutral branch, and at a shared instant it is the plan
#: change of whatever is already running.
_END_LIKE_EVENTS = frozenset(END_EVENTS)
_START_LIKE_EVENTS = (
    frozenset(START_TOKENS) - _END_LIKE_EVENTS - frozenset(NEUTRAL_EVENTS)
)


def event_rank(event_type: str) -> int:
    """The same-instant priority of one canonical event type.

    Exposed rather than kept private because it is a statement about the *data*
    and not only about this function: anything that reasons about two events in
    one second - a test, a future incremental corrector - has to order them the
    same way the walk does or it will describe a different table.
    """
    if event_type in _END_LIKE_EVENTS:
        return _RANK_END_LIKE
    if event_type in _START_LIKE_EVENTS:
        return _RANK_START_LIKE
    return _RANK_NEUTRAL


#: How close a second end with nothing open has to sit to the close it follows
#: before the walk reads the two as one end arriving twice.
#:
#: The walk swallows an end it has nothing to attach to when it has already
#: closed a lifecycle in this carrier segment, on the reading that a carrier
#: re-sending or correcting an end it has already sent is noise rather than
#: evidence of a second lifecycle. That reading is right for a correction landing
#: in the same daily batch and wrong for anything further away: the August 2026
#: audit's example is a cancellation in March and another in November at one
#: carrier, where the number was demonstrably resurrected in between and the
#: November lifecycle was being discarded on the strength of a flag set eight
#: months earlier.
#:
#: Twenty-four hours, because it has to be wide enough to cover the case the
#: suite already pins - a cancellation at 10:00 and a second one at 11:00, which
#: is one carrier correcting itself - and narrow enough that no plausible
#: reactivate-and-recancel cycle fits inside it. Anything outside the window is
#: treated exactly as the first end of a segment is: as a lifecycle we know ended
#: and never saw begin, starting at the previous lifecycle's end.
DUPLICATE_END_WINDOW = timedelta(hours=24)

#: Carriers whose boundaries are never firm, whatever the events say.
#:
#: ROGERS is the whole set today, and the reason is a property of the feed rather
#: than of any row. ROGERS sends no activation and no cancellation this pipeline
#: can trust: a start is inferred from a SIM appearing, which may be days after
#: the account opened, and ``STATUS_CHANGE-C`` is sent for a cancellation and for
#: a suspension alike, so seeing one does not say the account ended. Both
#: boundaries are therefore ``LOW`` *before* the inferred flags are consulted -
#: written this way so that nobody later reads a ROGERS ``is_start_inferred = 0``
#: and concludes the start is firm.
LOW_BOUNDARY_MNOS = frozenset({"ROGERS"})

HIGH = "HIGH"
LOW = "LOW"

#: Extra columns the walk emits for downstream stages. They are not part of the
#: published table - the account stage needs the boundary event ids to build its
#: edges and to fill ``source_event_id``, and re-deriving them from the events
#: would mean re-running the walk.
LIFECYCLE_WORK_SCHEMA = StructType(
    [
        StructField("lifecycle_uid", StringType(), False),
        StructField("phone_number_AC_hash", StringType(), False),
        StructField("mno", StringType(), False),
        StructField("mno_segment_id", IntegerType(), False),
        StructField("lifecycle_id", IntegerType(), False),
        StructField("lifecycle_start_ts", TimestampType(), False),
        StructField("lifecycle_end_ts", TimestampType(), True),
        StructField("lifecycle_is_open", IntegerType(), False),
        StructField("lifecycle_infer_type", StringType(), False),
        StructField("is_start_inferred", IntegerType(), False),
        StructField("is_end_inferred", IntegerType(), False),
        StructField("recycle_ind", IntegerType(), False),
        StructField("mno_port_ind", IntegerType(), False),
        StructField("temporary_pn_ind", IntegerType(), False),
        StructField("plan_change_ind", IntegerType(), False),
        StructField("plan_change_cnt", IntegerType(), False),
        StructField("start_confidence", StringType(), False),
        StructField("end_confidence", StringType(), False),
        StructField("confidence", StringType(), False),
        StructField("mno_is_supported", IntegerType(), False),
        StructField("tu_enriched", IntegerType(), False),
        # downstream-only
        StructField("anchor_ac_event_id", StringType(), False),
        StructField("start_event_id", StringType(), False),
        StructField("start_event_type", StringType(), False),
        StructField("end_event_id", StringType(), True),
        StructField("end_event_type", StringType(), True),
        StructField("phone_ignored_end_cnt", IntegerType(), False),
    ]
)


# ==========================================================================
# the walk
# ==========================================================================


def walk_phone(events: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Build every lifecycle for one phone number from its complete event history.

    ``events`` are that phone number's canonical events, in any order; they are
    sorted here so a caller cannot get it wrong. The sort is by timestamp, then
    by :func:`event_rank`, then by ``ac_event_id``.

    The rank is the middle key and not the last one because it is the only one of
    the three that carries meaning: it says that at a single instant an end is
    applied before a start and a neutral event after both, for the reasons set
    out on :data:`_RANK_END_LIKE`. ``ac_event_id`` stays as the final tiebreak
    because it is what makes the order total - without it, two events of the same
    rank in the same second would order differently on different runs and the
    same input would produce different Gold, which would make the convergence
    property false.

    Returns one dict per lifecycle, in start order.
    """
    ordered = sorted(
        events,
        key=lambda e: (
            e["ac_event_ts"],
            event_rank(e["ac_event_type"]),
            e["ac_event_id"],
        ),
    )

    out: list[dict] = []
    open_lc: dict | None = None
    prev_mno: str | None = None
    mno_segment_id = 0
    # When this carrier segment last closed a lifecycle, or ``None`` if it never
    # has. This replaces the ``closed_in_segment`` flag it used to be, because a
    # flag can only say *whether* a lifecycle was closed and the question the end
    # branch below actually asks is *how long ago* - see
    # :data:`DUPLICATE_END_WINDOW`.
    last_close_ts: datetime | None = None
    ignored_end_cnt = 0

    def start(event: Mapping[str, Any], inferred: bool) -> dict:
        phone = event[SLICE_KEY]
        mno = event["mno"]
        anchor = event["ac_event_id"]
        etype = event["ac_event_type"]
        return {
            "lifecycle_uid": sha256_join(
                "lifecycle", SCHEMA_VERSION, phone, mno, anchor
            ),
            SLICE_KEY: phone,
            "mno": mno,
            "mno_segment_id": mno_segment_id,
            "lifecycle_id": 0,  # assigned at the end, once the order is known
            "lifecycle_start_ts": event["ac_event_ts"],
            "lifecycle_end_ts": None,
            "lifecycle_is_open": 1,
            "lifecycle_infer_type": "",  # assembled at close
            "is_start_inferred": 1 if inferred else 0,
            "is_end_inferred": 1,
            "recycle_ind": 1
            if etype == "account_activation_number_recycle"
            else 0,
            "mno_port_ind": 1 if etype == "account_activation_mno_port" else 0,
            "temporary_pn_ind": 0,
            "plan_change_ind": 0,
            "plan_change_cnt": 1 if etype == "plan_change" else 0,
            # Boundary confidence is a naming of a fact this walk already
            # establishes, not a new derivation: ``inferred`` here is the same
            # value that becomes ``is_start_inferred``. Computing it in the same
            # pass is what makes it impossible for the confidence to drift away
            # from the boundary it describes.
            "start_confidence": LOW
            if (mno in LOW_BOUNDARY_MNOS or inferred)
            else HIGH,
            "end_confidence": LOW,  # replaced at close, or when the walk ends
            "confidence": LOW,  # assembled at the end, once both are known
            # A function of ``mno`` and nothing else, which is the whole point:
            # a TU-derived lifecycle constructed at a carrier we have no feed
            # from gets ``0`` without anybody remembering to set it. Every
            # lifecycle this walk builds gets ``1``, because this walk only ever
            # sees canonical events and those only come from supported feeds.
            "mno_is_supported": 1 if mno in MNO_DOMAIN else 0,
            # Set by the TU stage, which knows whether TU was asked about this
            # phone number. ``0`` here means "not asked", not "asked and nothing
            # came back" - the distinction the flag exists for.
            "tu_enriched": 0,
            "anchor_ac_event_id": anchor,
            "start_event_id": anchor,
            "start_event_type": etype,
            "end_event_id": None,
            "end_event_type": None,
            "_start_token": START_TOKENS.get(etype, _DEFAULT_START_TOKEN),
        }

    def close(lc: dict, ts, event_id, reason: str, inferred: bool) -> None:
        lc["lifecycle_end_ts"] = ts
        lc["lifecycle_is_open"] = 0
        lc["is_end_inferred"] = 1 if inferred else 0
        # An inferred close has no witnessing event, and the id passed in is the
        # event that opened the *next* lifecycle - a port, a fresh activation, a
        # carrier change. Writing it here used to give the row a non-null
        # ``end_event_id`` beside a null ``end_event_type``, naming an event that
        # belongs to a different lifecycle. ``schemas.py:576`` says the account
        # stage matches an edge by boundary event id *and* type, so that pairing
        # is a trap for anything reading the four boundary columns: the id
        # resolves, it resolves to a real event, and the event is not this
        # lifecycle's end. Admitting there is no witness is the honest answer, and
        # ``is_end_inferred`` and the ``_to_inferred_end_from_next_start`` token
        # already say where the boundary came from.
        lc["end_event_id"] = None if inferred else event_id
        lc["end_event_type"] = None if inferred else reason
        lc["end_confidence"] = (
            LOW if (lc["mno"] in LOW_BOUNDARY_MNOS or inferred) else HIGH
        )
        lc["lifecycle_infer_type"] = lc["_start_token"] + END_TOKENS[reason]
        out.append(lc)

    for event in ordered:
        etype = event["ac_event_type"]
        mno = event["mno"]

        # A change of carrier always starts a new MNO segment, and therefore a
        # new lifecycle. Anything still open at the old carrier is closed here,
        # not left running: leaving it open would put two overlapping
        # lifecycles on one phone number, which is the one thing the uniqueness
        # checks say cannot happen. The carrier that lost the subscriber rarely
        # says so, which is exactly why this has to be inferred.
        if mno != prev_mno:
            if open_lc is not None:
                close(
                    open_lc,
                    event["ac_event_ts"],
                    event["ac_event_id"],
                    "__inferred__",
                    inferred=True,
                )
                open_lc = None
            prev_mno = mno
            mno_segment_id += 1
            last_close_ts = None

        if etype in FORCE_NEW_EVENTS or etype == "account_activation":
            # A port, a recycle, an incoming number change or a fresh activation
            # is a statement that whatever was live before has ended - whether
            # or not the carrier bothered to send an end.
            if open_lc is not None:
                close(
                    open_lc,
                    event["ac_event_ts"],
                    event["ac_event_id"],
                    "__inferred__",
                    inferred=True,
                )
            open_lc = start(event, inferred=False)

        elif etype == "account_reactivation":
            # Coming back from a suspension. Suspension never closed the
            # lifecycle, so there is no boundary to move when one is open.
            if open_lc is None:
                open_lc = start(event, inferred=False)
            elif open_lc["_start_token"] == "inferred_start_from_plan_change_only":
                # The token claims the segment holds nothing but a plan change,
                # and a reactivation makes that false in exactly the way the
                # neutral branch below already handles. The two branches used to
                # disagree, so a phone number whose history was a plan change
                # followed by a reactivation published the narrower token and
                # said the reactivation had never happened.
                open_lc["_start_token"] = _DEFAULT_START_TOKEN

        elif etype in END_EVENTS:
            if open_lc is not None:
                close(
                    open_lc,
                    event["ac_event_ts"],
                    event["ac_event_id"],
                    etype,
                    inferred=False,
                )
                open_lc = None
                last_close_ts = event["ac_event_ts"]
            elif (
                last_close_ts is not None
                and event["ac_event_ts"] - last_close_ts <= DUPLICATE_END_WINDOW
            ):
                # A second end with nothing open, close behind the close it
                # follows: a late or duplicated end for a lifecycle we have
                # already closed. Emitting a near-zero-length lifecycle for it
                # would be noise, so it is ignored and counted; a rising count
                # means the feed's starts and ends disagree.
                #
                # The proximity test is the fix for the August 2026 audit's
                # finding 13, and it is deliberately *not* the fix that finding
                # proposed. Clearing the flag when a new lifecycle opens reads
                # like the obvious repair and is provably a no-op: this branch is
                # only reachable with nothing open, and the only way back to
                # nothing-open inside one carrier segment is through the close
                # above, which sets the state again on its way past. The
                # discarded November cancellation in the audit's example had no
                # intervening open lifecycle at all - that is why it was
                # discarded - so distance, not the flag, is what has to decide.
                ignored_end_cnt += 1
            else:
                # The first end we have seen in this carrier segment, with
                # nothing open to attach it to. The lifecycle certainly existed
                # - somebody cancelled something - and we never saw it begin.
                #
                # It is recorded with a start it did not observe, because the
                # alternative is worse in both directions. Starting it at the
                # cancellation makes a half-open ``[T, T)``: live at no moment,
                # so the row that records the one thing we know about this
                # number also says the number was never there. Discarding it
                # throws away the only evidence we have.
                #
                # Where the number has earlier history the start comes from it:
                # the previous lifecycle's end is the last instant the number
                # was demonstrably somewhere else, so the stretch between that
                # and the cancellation is the whole of what this lifecycle could
                # have been. It is the same rule the carrier-change branch above
                # applies - lifecycles abut, gaps are not left - stated
                # backwards, and it is what keeps the start from reaching back
                # across history the feeds actually witnessed.
                #
                # Where it does not, there is nothing to reach back to and the
                # start is the floor. Both are ``is_start_inferred = 1`` and
                # ``start_confidence = LOW`` already, from ``inferred=True``.
                lc = start(event, inferred=True)
                if out:
                    lc["lifecycle_start_ts"] = out[-1]["lifecycle_end_ts"]
                    lc["_start_token"] = "inferred_start_from_previous_lifecycle_end"
                else:
                    lc["lifecycle_start_ts"] = LIFECYCLE_START_FLOOR
                    lc["_start_token"] = "inferred_start_from_floor"
                close(
                    lc,
                    event["ac_event_ts"],
                    event["ac_event_id"],
                    etype,
                    inferred=False,
                )
                last_close_ts = event["ac_event_ts"]

        else:  # neutral: suspension, device_change, sim_change, plan_change
            if open_lc is None:
                open_lc = start(event, inferred=True)
            else:
                if etype == "plan_change":
                    open_lc["plan_change_cnt"] += 1
                elif open_lc["_start_token"] == "inferred_start_from_plan_change_only":
                    # The segment turned out to contain more than a plan change,
                    # so the narrower token no longer describes it.
                    open_lc["_start_token"] = _DEFAULT_START_TOKEN

    if open_lc is not None:
        open_lc["lifecycle_infer_type"] = (
            open_lc["_start_token"] + END_TOKENS["__open__"]
        )
        # is_end_inferred stays 1 on an open lifecycle. That reads oddly until
        # you say it out loud: the flag means "we never saw an end", which is
        # precisely the state of a lifecycle that is still running. It is not a
        # claim that an end was computed. The base table already behaves this
        # way.
        #
        # end_confidence goes the other way, and for the same reason. All three
        # carriers send a cancellation, so the *absence* of one is itself
        # evidence, and the evidence says the number is still live. A lifecycle
        # with no end is not one whose end we missed; it is one that has not
        # ended. That holds for ROGERS too - its end is low only because
        # STATUS_CHANGE-C is ambiguous between a cancellation and a suspension,
        # and where no such event was sent there is nothing to be ambiguous
        # about.
        open_lc["end_confidence"] = HIGH
        out.append(open_lc)

    # Start, then end, then uid. The end is in the key because starts tie: a
    # zero-duration lifecycle shares its start with the lifecycle that follows
    # it, and with only ``(start, uid)`` the hash decided which of the two came
    # first. ``mno_segment_id`` is assigned during the walk in true chronological
    # order while ``lifecycle_id`` comes from this sort, so half the time the
    # hash produced a row carrying ``lifecycle_id = 1, mno_segment_id = 2`` - two
    # ordinals on one row disagreeing about which lifecycle came first.
    #
    # ``end_ts is None`` sorts before ``end_ts`` for the same reason it appears
    # at all: an open lifecycle reaches forward without limit, so among
    # lifecycles sharing a start it is the last one, and the boolean also keeps
    # the tuple comparison from ever reaching a ``None`` against a ``datetime``.
    out.sort(
        key=lambda lc: (
            lc["lifecycle_start_ts"],
            lc["lifecycle_end_ts"] is None,
            lc["lifecycle_end_ts"],
            lc["lifecycle_uid"],
        )
    )
    for index, lc in enumerate(out, start=1):
        lc["lifecycle_id"] = index
        lc["plan_change_ind"] = 1 if lc["plan_change_cnt"] > 0 else 0
        # Named for what it is: a phone-number total written onto every lifecycle
        # of that phone number, not a count of the ends this particular lifecycle
        # swallowed. It was ``ignored_end_cnt``, which read as per-row and made a
        # consumer summing it over a phone number's lifecycles multiply the
        # figure by the number of rows. Renaming it rather than making it
        # per-row was the smaller change and the truer one - the walk genuinely
        # does not know which lifecycle a discarded end belonged to, which is why
        # it was discarded - and the column is not in ``LIFECYCLE_SCHEMA``, so
        # nothing published moves.
        lc["phone_ignored_end_cnt"] = ignored_end_cnt
        # A lifecycle is only as good as its weaker edge. Both boundaries have
        # to be firm for a point-in-time question about this stretch of life to
        # have a firm answer, so the two are combined with AND rather than
        # averaged - there is no half-answer to "was this number theirs on the
        # 3rd of June".
        lc["confidence"] = (
            HIGH
            if lc["start_confidence"] == HIGH and lc["end_confidence"] == HIGH
            else LOW
        )
        lc.pop("_start_token", None)
    return out


# ==========================================================================
# the stage
# ==========================================================================

_WALK_INPUT_COLUMNS = (SLICE_KEY, "mno", "ac_event_type", "ac_event_ts", "ac_event_id")


def _walk_group(item) -> Iterable[tuple]:
    _, rows = item
    events = [dict(zip(_WALK_INPUT_COLUMNS, row)) for row in rows]
    fields = [f.name for f in LIFECYCLE_WORK_SCHEMA.fields]
    for lc in walk_phone(events):
        yield tuple(lc[name] for name in fields)


def build_lifecycles(
    spark: SparkSession, canonical_events: DataFrame
) -> DataFrame:
    """Run :func:`walk_phone` for every phone number in the slice.

    Returns the work frame - the published columns plus the boundary event ids
    the account stage needs. :func:`to_gold` projects it onto the table schema.

    The partition count is passed rather than left to Spark: see
    :func:`~trust_score_05.lineage.spark.shuffle_partitions`. This is the widest
    shuffle in the pipeline - every canonical event in the slice crosses it -
    and without the argument it was also the one the operator could not tune.
    """
    rows = canonical_events.select(*_WALK_INPUT_COLUMNS).rdd.map(tuple)
    walked = rows.groupBy(lambda r: r[0], shuffle_partitions(spark)).flatMap(
        _walk_group
    )
    return spark.createDataFrame(walked, LIFECYCLE_WORK_SCHEMA)


def to_gold(lifecycles: DataFrame, run_ts: datetime) -> DataFrame:
    """Project the work frame onto ``msisdn_lifecycle``."""
    stamped = with_audit_columns(lifecycles, F.lit(run_ts).cast("timestamp"))
    return align_to_schema(stamped, LIFECYCLE_SCHEMA)
