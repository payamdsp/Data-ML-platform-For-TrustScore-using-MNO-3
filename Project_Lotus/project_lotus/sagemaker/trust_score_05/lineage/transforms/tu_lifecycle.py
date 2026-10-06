"""Stage 2b: the missing middle, from TransUnion's porting history.

Our own feeds see a phone number leave Bell and reappear at Telus eight months
later. They cannot see where it was in between, because it was at a carrier we
have no feed from, and so the gap sits in Gold as an absence with nothing to say
about it. TU sees that stretch: its porting history names every carrier the
number moved to, whether or not we have a relationship with that carrier.

This stage turns that history into lifecycles and adds them to the ones stage 2
built. Three properties of the result are worth stating before the code, because
each is a decision rather than a consequence.

**It adds, it never merges.** The two lifecycles our feeds built either side of
an unsupported stretch are factually correct - the number really was not at
either carrier in between - so nothing here rewrites them. Merging them into one
would assert continuity that demonstrably did not happen, and it would move
``lifecycle_uid``, which ``bad_actor_gold`` publishes; a published row would
dangle the first time TU arrived for its number. What TU contributes is the
missing middle and the links across it, and section 3 of the enrichment design
is explicit that this is load-bearing rather than stylistic.

The one exception is a *floored* start, and it is an exception because a floor
is not something our feeds observed. A phone number whose first event in its
whole history is a cancellation gets a lifecycle at
:data:`~..schemas.LIFECYCLE_START_FLOOR`, which is how stage 2 writes "it ended,
we never saw it begin". Where TU shows a port into that same carrier, before
that same cancellation, TU has watched the thing stage 2 missed and the start
moves onto it. Nothing about the row's identity moves with it:
``anchor_ac_event_id`` and therefore ``lifecycle_uid`` are still the ones stage 2
derived from the cancellation, so a published ``bad_actor_gold`` row still
resolves. Only the boundary and its provenance change.

Where TU shows no such port, the floor stays. TU returning a complete porting
history that does not include an arrival at the carrier we watched a
cancellation at is TU and our feed contradicting each other, and there is no
evidence here that could settle it - so it is left, and the stays TU did report
below the cancellation are absorbed and counted rather than built into
lifecycles that would overlap the floored one.

**It never overlaps.** A phone number has one carrier at a time, and the
uniqueness checks on ``msisdn_lifecycle`` say so. A TU stay is therefore clipped
to whatever part of it our own lifecycles do not already occupy, and a stay that
is entirely occupied produces no row at all. Trusting TU over our own feeds in
that overlap would be the wrong way round: we watched those events arrive.

**It renumbers.** Inserting a lifecycle into the middle of a phone number's
history shifts ``lifecycle_id`` and ``mno_segment_id`` for everything after it.
That is expected - the design says so in as many words - and it is why neither
column is a stable identifier and nothing downstream may key on them. The
renumbering happens only for phone numbers that actually gained a lifecycle, so
an unenriched number's ordinals do not churn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..logging_utils import get_logger
from ..spark import shuffle_partitions
from ..schemas import (
    LIFECYCLE_FLOOR_START_TOKEN,
    MNO_DOMAIN,
    SCHEMA_VERSION,
    SLICE_KEY,
)
from .customers import tu_latest_response
from .lifecycle import HIGH, LIFECYCLE_WORK_SCHEMA, LOW, sha256_join

__all__ = [
    "TuLifecycleReport",
    "carrier_stays",
    "walk_tu_phone",
    "build_tu_lifecycles",
    "TU_START_TOKEN",
    "TU_END_TOKEN",
]

LOGGER = get_logger(__name__)

#: How a TU-derived lifecycle opened and closed. The vocabulary is the same shape
#: as ``lifecycle.START_TOKENS`` / ``END_TOKENS`` - a start token and an end token
#: concatenated - because ``lifecycle_infer_type`` is one column and a reader
#: parsing it should not need to know which stage wrote the row.
TU_START_TOKEN = "explicit_tu_port_in"
TU_START_CLIPPED_TOKEN = "inferred_start_from_previous_lifecycle_end"
TU_END_TOKEN = "_to_explicit_tu_port_out"
TU_END_CLIPPED_TOKEN = "_to_inferred_end_from_next_start"
TU_END_OPEN_TOKEN = "_open_no_cancellation"

#: The event type recorded on a TU-derived lifecycle's boundary. There is no
#: canonical event behind it - the boundary is a TU port row - and this is the
#: string that says so anywhere ``start_event_type`` surfaces.
TU_EVENT_TYPE = "tu_inter_spid_port"


@dataclass
class TuLifecycleReport:
    """What the stage did, in numbers nothing downstream could recover.

    A stay TU witnessed that our own lifecycles already covered leaves no row
    behind, so ``stays_absorbed`` exists here or nowhere. It is not an error -
    the usual cause is TU and a carrier feed describing the same port, which is
    the two sources agreeing - but a run where it is large and
    ``lifecycles_built`` is zero means TU is telling us only things we already
    knew, and that is worth being able to see.
    """

    phones_enriched: int = 0
    lifecycles_built: int = 0
    stays_absorbed: int = 0
    stays_split: int = 0
    floored_starts_recovered: int = 0
    unsupported_carriers: dict[str, int] = field(default_factory=dict)

    def format_line(self) -> str:
        carriers = ", ".join(
            f"{name}={count}" for name, count in sorted(self.unsupported_carriers.items())
        )
        return (
            f"tu lifecycles: phones={self.phones_enriched} "
            f"built={self.lifecycles_built} absorbed={self.stays_absorbed} "
            f"split={self.stays_split} "
            f"floors_recovered={self.floored_starts_recovered}"
            + (f" | {carriers}" if carriers else "")
        )


# ==========================================================================
# the walk
# ==========================================================================


def carrier_name(to_mno_std: str | None, to_spid: str | None) -> str:
    """The name a TU-derived lifecycle carries in ``mno``.

    The standardised name when Silver could resolve the SPID, and the SPID
    itself, marked as one, when it could not. Never null and never empty:
    ``mno`` is not nullable and ``consistency_mno_domain`` fails an empty one, so
    an unresolved SPID has to arrive as something a human can look up rather than
    as a hole. ``SPID_328F`` is not a carrier name and is not meant to read like
    one - it is a pointer at the row that needs a mapping.
    """
    if to_mno_std and to_mno_std.strip():
        return to_mno_std.strip().upper()
    if to_spid and str(to_spid).strip():
        return f"SPID_{str(to_spid).strip().upper()}"
    return "UNKNOWN_CARRIER"


def carrier_stays(port_events: Sequence[Mapping[str, Any]]) -> list[dict]:
    """One phone number's ports, as the stretches of time they imply.

    A port event says the number moved to a carrier at an instant. The stay that
    follows runs from that instant until the next port, or forever if there is no
    next one. Turning events into stays here rather than in the caller is what
    lets the overlap arithmetic below work on intervals, which is the only shape
    in which "clip this against what we already know" is expressible without a
    special case per boundary.

    Only inter-carrier ports delimit a stay. An intra-SPID port is a number
    moving between two switches of one carrier: real history, kept in Silver, and
    not a change of who the number is with. Letting one end a stay would split a
    single stretch at that carrier into two abutting lifecycles that say nothing
    different from each other.

    Sorted by timestamp, then ``event_seq``, then ``record_id``. The two
    tiebreaks are not decoration: two ports in the same second would otherwise
    order differently on different runs, and a table rebuilt from scratch every
    run would stop converging. ``event_seq`` is TU's own ordering where it sent
    one; ``record_id`` is the deterministic fallback where it did not.
    """
    ordered = sorted(
        (e for e in port_events if e.get("is_inter_carrier_port")),
        key=lambda e: (
            e["event_timestamp"],
            -1 if e.get("event_seq") is None else e["event_seq"],
            e["record_id"],
        ),
    )
    stays: list[dict] = []
    for index, event in enumerate(ordered):
        following = ordered[index + 1] if index + 1 < len(ordered) else None
        stays.append(
            {
                "mno": carrier_name(event.get("to_mno_std"), event.get("to_spid")),
                "start_ts": event["event_timestamp"],
                "end_ts": None if following is None else following["event_timestamp"],
                "start_record_id": event["record_id"],
                "end_record_id": None if following is None else following["record_id"],
            }
        )
    return stays


def _is_floored(lifecycle: Mapping[str, Any]) -> bool:
    """Whether stage 2 gave this lifecycle a floor instead of a start.

    Read off ``lifecycle_infer_type`` rather than off ``lifecycle_start_ts``,
    because the floor is a value some other lifecycle could legitimately hold -
    a real activation on 1 January 1970 is absurd but not impossible to write -
    and because the token is the column that is supposed to say where a boundary
    came from.
    """
    return str(lifecycle.get("lifecycle_infer_type") or "").startswith(
        LIFECYCLE_FLOOR_START_TOKEN
    )


def _stay_live_at(stays: Sequence[Mapping[str, Any]], mno: str, end) -> dict | None:
    """The TU stay at ``mno`` that was still running at ``end``, if there is one.

    At most one can match: stays partition the number's history and do not
    overlap. ``end`` is a floored lifecycle's cancellation, so a stay qualifies
    when it began strictly before it - a stay beginning at the cancellation is
    the number's *next* carrier - and had not ended before it. A stay ending
    exactly at the cancellation qualifies: that is TU and our feed describing the
    same departure, which is the two sources agreeing rather than disagreeing.

    ``end`` of ``None`` is an open lifecycle, matched by the stay that is itself
    still open. Stage 2 never floors an open lifecycle today - it floors only in
    the branch that closes the row on the spot - and this is handled anyway
    because the cost is one comparison and the alternative is a silent wrong
    answer if that ever changes.
    """
    for stay in stays:
        if stay["mno"] != mno:
            continue
        if end is None:
            if stay["end_ts"] is None:
                return dict(stay)
            continue
        if stay["start_ts"] < end and (stay["end_ts"] is None or stay["end_ts"] >= end):
            return dict(stay)
    return None


def _recover_floored_starts(
    kept: list[dict], stays: Sequence[Mapping[str, Any]]
) -> int:
    """Move floored starts onto the TU port that opened them. Mutates ``kept``.

    Done before the subtraction below rather than after it, so that everything
    downstream sees one consistent picture: the recovered lifecycle occupies its
    real stretch, TU's own stays are clipped against that stretch and not against
    the floor, and the missing middles TU reports *below* the recovered start are
    built instead of being swallowed by an interval reaching back to 1970.

    Returns how many starts moved, for the walk's counters.
    """
    moved = 0
    for lifecycle in kept:
        if not _is_floored(lifecycle):
            continue
        stay = _stay_live_at(stays, lifecycle["mno"], lifecycle["lifecycle_end_ts"])
        if stay is None:
            continue

        end_token = lifecycle["lifecycle_infer_type"][len(LIFECYCLE_FLOOR_START_TOKEN):]
        lifecycle["lifecycle_start_ts"] = stay["start_ts"]
        lifecycle["lifecycle_infer_type"] = TU_START_TOKEN + end_token
        lifecycle["is_start_inferred"] = 0
        lifecycle["start_confidence"] = HIGH
        lifecycle["confidence"] = (
            HIGH if lifecycle["end_confidence"] == HIGH else LOW
        )
        # The number demonstrably arrived here by porting; TU watched it happen.
        lifecycle["mno_port_ind"] = 1
        # The boundary's provenance, which is now a TU port row rather than the
        # cancellation stage 2 had to stand the start on. ``anchor_ac_event_id``
        # is deliberately left alone: it seeds ``lifecycle_uid``, and moving a
        # boundary is not allowed to move the row's identity.
        lifecycle["start_event_id"] = stay["start_record_id"]
        lifecycle["start_event_type"] = TU_EVENT_TYPE
        moved += 1
    return moved


def _occupied(existing: Sequence[Mapping[str, Any]]) -> list[tuple]:
    """Existing lifecycles as (start, end) intervals, open ones ending at None.

    The sort key is spelled out rather than left to the tuples themselves. A
    plain ``sorted`` on ``(start, end)`` compares element 0 and then element 1,
    and element 1 is ``None`` on an open lifecycle: a zero-duration lifecycle
    ``(T, T)`` beside an open one starting in the same instant ``(T, None)``
    compares a ``datetime`` against ``None`` and raises ``TypeError`` in the
    executor. That is not a hypothetical shape - it is the shape that produced
    the 1,658,031 zero-duration rows the August 2026 audit counted, and it was a
    live blocker on running this stage over full history.

    ``None`` sorts last, which is also what it means: an open lifecycle reaches
    forward without limit, so among intervals sharing a start it is the widest
    and the one :func:`_subtract` should apply after the others.
    """
    return sorted(
        ((lc["lifecycle_start_ts"], lc["lifecycle_end_ts"]) for lc in existing),
        key=lambda interval: (interval[0], interval[1] is None, interval[1]),
    )


def _subtract(start, end, occupied: Sequence[tuple]) -> list[tuple]:
    """The parts of ``[start, end)`` no existing lifecycle covers.

    ``None`` at either end means unbounded, which is how an open lifecycle and an
    ongoing TU stay both arrive here. Comparisons are written against explicit
    ``None`` checks rather than against a sentinel date, because a sentinel that
    is far enough in the future to be safe is also far enough to be wrong when it
    reaches a timestamp column.

    Returns the remainders in order. More than one means an existing lifecycle
    sits *inside* the stay, which is our feeds and TU contradicting each other
    about where the number was; the caller counts that rather than resolving it,
    because there is no evidence here that could.
    """
    pieces = [(start, end)]
    for occ_start, occ_end in occupied:
        # A zero-length occupier covers no time, so it subtracts nothing. Left
        # in, one falling strictly inside a stay matches neither no-overlap
        # guard below - the piece does not end before it starts and does not
        # start after it ends - and the two remainder clauses then emit ``(S, T)``
        # and ``(T, E)``: two contiguous TU lifecycles where one belongs, an
        # extra ``#1``-suffixed anchor and ``lifecycle_uid``, and an inflated
        # ``stays_split``. This is the same fix ``rules_gold.py:179`` already
        # carries for the overlap check - *"An interval that ends where it starts
        # is never live, so it takes no part in this"* - which ``_subtract``
        # never got.
        if occ_end is not None and occ_end <= occ_start:
            continue
        nxt: list[tuple] = []
        for piece_start, piece_end in pieces:
            # No overlap: the piece ends before the occupier starts, or starts
            # after it ends.
            if piece_end is not None and piece_end <= occ_start:
                nxt.append((piece_start, piece_end))
                continue
            if occ_end is not None and piece_start >= occ_end:
                nxt.append((piece_start, piece_end))
                continue
            # Left remainder.
            if piece_start < occ_start:
                nxt.append((piece_start, occ_start))
            # Right remainder.
            if occ_end is not None and (piece_end is None or piece_end > occ_end):
                nxt.append((occ_end, piece_end))
        pieces = nxt
    return pieces


def _new_lifecycle(
    phone: str,
    stay: Mapping[str, Any],
    piece: tuple,
    suffix: int,
) -> dict:
    """One TU-derived lifecycle row, in the work schema's shape."""
    piece_start, piece_end = piece
    start_is_observed = piece_start == stay["start_ts"]
    end_is_observed = stay["end_ts"] is not None and piece_end == stay["end_ts"]
    is_open = piece_end is None
    mno = stay["mno"]

    anchor = stay["start_record_id"] if suffix == 0 else f"{stay['start_record_id']}#{suffix}"

    if start_is_observed:
        start_token = TU_START_TOKEN
    else:
        start_token = TU_START_CLIPPED_TOKEN
    if is_open:
        end_token = TU_END_OPEN_TOKEN
    elif end_is_observed:
        end_token = TU_END_TOKEN
    else:
        end_token = TU_END_CLIPPED_TOKEN

    # An observed port timestamp is the firmest boundary evidence this pipeline
    # ever has: TU watched the handover and recorded when it happened. A clipped
    # boundary is the opposite - it is our own lifecycle's edge standing in for a
    # boundary nobody observed - so it is LOW whatever TU said about the stay.
    #
    # An open stay's end is HIGH for the same reason an open lifecycle's is in
    # stage 2: TU returns a complete history, so the absence of a port out is
    # evidence that the number has not left, not evidence that we missed it.
    start_confidence = HIGH if start_is_observed else LOW
    end_confidence = HIGH if (is_open or end_is_observed) else LOW

    return {
        "lifecycle_uid": sha256_join("lifecycle", SCHEMA_VERSION, phone, mno, anchor),
        SLICE_KEY: phone,
        "mno": mno,
        "mno_segment_id": 0,  # assigned by the renumbering below
        "lifecycle_id": 0,
        "lifecycle_start_ts": piece_start,
        "lifecycle_end_ts": piece_end,
        "lifecycle_is_open": 1 if is_open else 0,
        "lifecycle_infer_type": start_token + end_token,
        "is_start_inferred": 0 if start_is_observed else 1,
        # 1 on an open lifecycle, exactly as stage 2 writes it: the flag means
        # "we never saw an end", which is the state of a stay that is still
        # running, and not a claim that an end was computed.
        "is_end_inferred": 0 if end_is_observed else 1,
        "recycle_ind": 0,
        # A TU-derived lifecycle exists *because* the number ported into it.
        # This is the one flag on the row that needs no qualification.
        "mno_port_ind": 1,
        "temporary_pn_ind": 0,
        "plan_change_ind": 0,
        "plan_change_cnt": 0,
        "start_confidence": start_confidence,
        "end_confidence": end_confidence,
        "confidence": HIGH
        if start_confidence == HIGH and end_confidence == HIGH
        else LOW,
        # Zero by construction and asserted downstream: this stage is the only
        # thing in the pipeline that can produce a lifecycle at a carrier outside
        # MNO_DOMAIN, and `consistency_unsupported_lifecycle_is_tu_derived`
        # aborts a run where one appears with tu_enriched = 0.
        "mno_is_supported": 1 if mno in MNO_DOMAIN else 0,
        "tu_enriched": 1,
        # There is no canonical event behind this row, so the TU record id stands
        # in everywhere an event id would. That is traceable in the direction
        # that matters - the id resolves to exactly one row of `tu_portps` - and
        # the alternative, a null, would fail the work schema and lose the
        # provenance of the only lifecycles nobody can re-derive from our feeds.
        "anchor_ac_event_id": anchor,
        "start_event_id": anchor,
        "start_event_type": TU_EVENT_TYPE,
        "end_event_id": stay["end_record_id"] if end_is_observed else None,
        "end_event_type": TU_EVENT_TYPE if end_is_observed else None,
        "phone_ignored_end_cnt": 0,
    }


def _renumber(lifecycles: list[dict]) -> list[dict]:
    """Reassign ``lifecycle_id`` and ``mno_segment_id`` over the merged history.

    Same rule stage 2 applies, restated over lifecycles instead of over events:
    ``lifecycle_id`` is the 1-based position in start order, and
    ``mno_segment_id`` advances every time the carrier changes from one lifecycle
    to the next. Two lifecycles at one carrier with nothing in between - a
    cancellation and a later reactivation - stay in the same segment, which is
    what the segment means.

    The sort key is (start, end, uid) and not (start) alone, for the reason every
    sort in this pipeline carries a tiebreak: two lifecycles starting in the same
    second would otherwise be numbered by whatever order the shuffle produced,
    and the same input would stop producing the same table.

    The end sits between the two because ``uid`` alone is not a tiebreak, it is a
    coin toss. Starts tie whenever one of the two members is zero-duration - a
    stay clipped down to nothing, or a stage 2 lifecycle a port closed in the
    second it opened - and a hash deciding which of them is ``lifecycle_id = 1``
    puts the shorter interval after the longer one about half the time, which
    then also walks ``mno_segment_id`` through the carriers in the wrong order.
    Ordering by the end resolves it the way the intervals themselves do: the one
    that finishes first came first.

    ``end_ts is None`` sorts last within a tied start, because an open lifecycle
    reaches forward without limit, and the boolean also keeps the tuple
    comparison from reaching a ``None`` against a ``datetime``.
    """
    ordered = sorted(
        lifecycles,
        key=lambda lc: (
            lc["lifecycle_start_ts"],
            lc["lifecycle_end_ts"] is None,
            lc["lifecycle_end_ts"],
            lc["lifecycle_uid"],
        ),
    )
    segment = 0
    previous_mno: str | None = None
    for index, lc in enumerate(ordered, start=1):
        if lc["mno"] != previous_mno:
            segment += 1
            previous_mno = lc["mno"]
        lc["mno_segment_id"] = segment
        lc["lifecycle_id"] = index
    return ordered


def walk_tu_phone(
    existing: Sequence[Mapping[str, Any]],
    port_events: Sequence[Mapping[str, Any]],
) -> tuple[list[dict], dict]:
    """Every lifecycle for one phone number, once TU has been read.

    Returns the complete list - the stage 2 lifecycles, with ``tu_enriched`` set
    and their ordinals possibly shifted, plus whatever TU added - and a small
    dict of counters for the report. Returning the complete list rather than only
    the additions is deliberate: the ordinals of the existing rows are part of
    what this function decides, so handing back only the new rows would leave the
    caller to redo the renumbering and get a second chance to disagree.

    Callable with two lists of dictionaries and no SparkSession, like
    :func:`~.lifecycle.walk_phone` and for the same reason - the interesting
    cases here are arithmetic on intervals, and they should be testable as
    arithmetic on intervals.
    """
    kept = [dict(lc) for lc in existing]
    for lc in kept:
        # Set for every lifecycle of a phone number TU answered about, including
        # the ones TU had nothing to add to. The flag means "TU was asked", which
        # is what a reader needs in order to tell "TU had nothing" from "nobody
        # ever asked" - two states that justify opposite actions.
        lc["tu_enriched"] = 1

    counters = {"built": 0, "absorbed": 0, "split": 0, "floors_recovered": 0, "carriers": {}}
    if not port_events:
        return _renumber(kept), counters

    # Borrowed from an existing lifecycle where there is one, and read off the TU
    # response where there is not. Both say the same thing - the walk is grouped
    # by phone number and every row in the group carries it - and the fallback
    # exists because a number can have a porting history and no event either
    # carrier feed ever sent.
    phone = kept[0][SLICE_KEY] if kept else port_events[0][SLICE_KEY]

    stays = carrier_stays(port_events)
    counters["floors_recovered"] = _recover_floored_starts(kept, stays)

    occupied = _occupied(kept)
    added: list[dict] = []
    for stay in stays:
        if stay["mno"] in MNO_DOMAIN:
            # A stay at a carrier we do have a feed from is not a missing middle.
            # Our own events are the better witness to it - they carry the
            # activations, cancellations and plan changes TU cannot see - so the
            # stage leaves it alone rather than building a thinner duplicate
            # beside it.
            continue
        pieces = [
            (start, end)
            for start, end in _subtract(stay["start_ts"], stay["end_ts"], occupied)
            if end is None or end > start
        ]
        if not pieces:
            counters["absorbed"] += 1
            continue
        if len(pieces) > 1:
            counters["split"] += 1
        for suffix, piece in enumerate(pieces):
            added.append(_new_lifecycle(phone, stay, piece, suffix))
            counters["built"] += 1
            counters["carriers"][stay["mno"]] = counters["carriers"].get(stay["mno"], 0) + 1

    return _renumber(kept + added), counters


# ==========================================================================
# the stage
# ==========================================================================

_TU_COLUMNS = (
    SLICE_KEY,
    "record_id",
    "event_timestamp",
    "event_seq",
    "is_inter_carrier_port",
    "to_mno_std",
    "to_spid",
)

_LC_COLUMNS = tuple(f.name for f in LIFECYCLE_WORK_SCHEMA.fields)


def _walk_group(item) -> Iterable[tuple]:
    _, rows = item
    existing = [dict(zip(_LC_COLUMNS, row[1])) for row in rows if row[0] == "lc"]
    ports = [dict(zip(_TU_COLUMNS, row[1])) for row in rows if row[0] == "tu"]
    lifecycles, _ = walk_tu_phone(existing, ports)
    for lc in lifecycles:
        yield tuple(lc[name] for name in _LC_COLUMNS)


def build_tu_lifecycles(
    spark: SparkSession, lifecycles: DataFrame, tu_ports: DataFrame | None
) -> tuple[DataFrame, TuLifecycleReport]:
    """Add TU's missing middles to the slice's lifecycles.

    ``tu_ports`` is the complete TU history for the slice, unwatermarked, or
    ``None`` when the feed is disabled or has never landed. A ``None`` returns
    the input frame untouched, which is the pipeline behaving exactly as it did
    before enrichment existed - the property that lets this stage be switched on
    without a migration.

    Only phone numbers TU answered about go through the Python walk. The rest are
    passed through with an anti-join, which is worth the two extra joins: the
    walk is a shuffle to Python and back, and on a nightly batch the enriched
    numbers are a small minority of the slice - TU is fetched per fraud record,
    and most phone numbers in a slice arrived because a carrier sent an event
    about them.
    """
    if tu_ports is None:
        return lifecycles, TuLifecycleReport()

    latest = tu_latest_response(tu_ports).select(*_TU_COLUMNS)
    keys = latest.select(SLICE_KEY).distinct().persist()
    phones_enriched = keys.count()
    if phones_enriched == 0:
        keys.unpersist()
        return lifecycles, TuLifecycleReport()

    # Re-selected in schema order after each join: a join on a named key puts
    # that key first in the result, and the anti-joined half would otherwise
    # reach ``unionByName`` in a different column order from the walked half.
    # The union would still line the two up by name - and the frame handed back
    # would no longer match the work schema it claims to be, which every caller
    # downstream is entitled to assume.
    touched = lifecycles.join(keys, SLICE_KEY, "left_semi").select(*_LC_COLUMNS)
    untouched = lifecycles.join(keys, SLICE_KEY, "left_anti").select(*_LC_COLUMNS)

    lc_rdd = touched.rdd.map(lambda r: (r[SLICE_KEY], ("lc", tuple(r))))
    tu_rdd = latest.rdd.map(lambda r: (r[0], ("tu", tuple(r))))
    # Partition count passed for the reason in
    # :func:`~trust_score_05.lineage.spark.shuffle_partitions`: ``groupByKey`` on
    # an RDD does not read ``spark.sql.shuffle.partitions``, and a union of two
    # RDDs has no partitioner of its own for Spark to inherit a sensible number
    # from either.
    walked = (
        lc_rdd.union(tu_rdd)
        .groupByKey(shuffle_partitions(spark))
        .flatMap(_walk_group)
    )
    rebuilt = spark.createDataFrame(walked, LIFECYCLE_WORK_SCHEMA)

    out = untouched.unionByName(rebuilt)
    report = _report(rebuilt, phones_enriched)
    keys.unpersist()
    LOGGER.info(report.format_line())
    return out, report


def _report(rebuilt: DataFrame, phones_enriched: int) -> TuLifecycleReport:
    """Count what was built, from the frame rather than from the walk.

    The absorbed and split counts live inside the walk and would have to be
    carried out of an RDD to reach here, which means a second pass or an
    accumulator. Neither is worth it: what an operator needs from this stage is
    how many lifecycles it built and at which carriers, and both are one
    aggregation over a frame that is already small. The two counters the walk
    keeps are reachable from its unit tests, which is where the interesting
    cases are checked anyway.
    """
    from_tu = rebuilt.where(F.col("tu_enriched") == F.lit(1)).where(
        F.col("start_event_type") == F.lit(TU_EVENT_TYPE)
    )
    # A lifecycle this stage *built* seeds its own uid on the TU record that
    # opened it, so its anchor and its start event are the same id. A stage 2
    # lifecycle whose floored start was recovered also carries a TU port as its
    # start event, but keeps the anchor stage 2 gave it - which is exactly what
    # tells the two apart, and what keeps a recovered start out of a count that
    # is meant to say how much history TU supplied that we did not have.
    recovered = F.col("anchor_ac_event_id") != F.col("start_event_id")
    built = from_tu.where(~recovered)
    by_carrier = {
        row["mno"]: row["count"]
        for row in built.groupBy("mno").agg(F.count(F.lit(1)).alias("count")).collect()
    }
    return TuLifecycleReport(
        phones_enriched=phones_enriched,
        lifecycles_built=sum(by_carrier.values()),
        floored_starts_recovered=from_tu.where(recovered).count(),
        unsupported_carriers=by_carrier,
    )
