"""Lineage 2: device events -> the phone-number <-> device map.

One row per *contiguous* association between a phone number and a device, valid
``[from_ts, to_ts)``. A phone number that used device A, moved to device B and
later came back to A gets three rows, not two - the third is a new association,
not a resumption of the first.

That is the whole point of the table and it is what the nearest existing
artefact got wrong. Grouping by ``(msisdn, imei)`` and taking ``min(start)`` and
``max(end)`` collapses A -> B -> A into one row for A whose interval swallows the
entire period the phone number spent on B, so the table answers "which device was
this number on last March?" with two devices at once.

The rule that makes this build deterministic is the cardinality the business
confirmed: **a phone number is attached to exactly one device at a time**, while
a device may carry several phone numbers at once. One to many, in that
direction. It follows that a device start is by itself sufficient evidence that
the previous association ended, whether or not the carrier sent an end - and
that single consequence is what produces the three rows for A -> B -> A with no
special handling anywhere.

One consequence of that rule is worth stating on its own, because it is where
the table first went wrong in production. A device start is evidence the previous
association ended *at that instant*, so a feed that announces A, then B, then A
again all inside one timestamp closes A and B at the moment they opened. Those
are half-open intervals containing no moment, and they are not published: an
association nobody could ever observe is not one this table has an opinion about.
Publishing them put two rows for A on the same start, which is the one shape the
key ``(phone number, device, start)`` may not take.

Like the lifecycle walk, this is written as a plain Python function over one
phone number's events. The reasons are the same: the branch table is short but
its edge cases - a duplicate start, an end for a device that is no longer the
open one, an end with nothing open - are precisely where a chain of window
functions would go quietly wrong, and :func:`walk_device_history` can be called
from a test with a handful of dictionaries and no SparkSession.
"""

from __future__ import annotations

from datetime import datetime
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
from ..schemas import IMEI_MAP_SCHEMA, SLICE_KEY
from .common import align_to_schema

__all__ = [
    "DEVICE_START_EVENTS",
    "DEVICE_END_EVENTS",
    "IMEI_WORK_SCHEMA",
    "walk_device_history",
    "build_imei_map",
    "close_on_lifecycle_end",
    "to_gold",
]

LOGGER = get_logger(__name__)

#: Events that attach a phone number to a device. Three spellings of the same
#: thing: a plain start, a start that accompanied an activation (``_A``), and a
#: start that accompanied a number change (``_NC``). The distinction matters
#: upstream, not here.
DEVICE_START_EVENTS = ("IMEI_START", "IMEI_START_A", "IMEI_START_NC")

#: Events that detach it. ``IMSI_END_C`` is TELUS's cancellation, which ends the
#: SIM's association and therefore the handset's; it carries no IMEI of its own,
#: so it closes whatever is open rather than being matched to a device.
DEVICE_END_EVENTS = ("IMEI_END",)
SIM_END_EVENTS = ("IMSI_END_C",)

#: The published columns plus the counters the DQ checks read.
IMEI_WORK_SCHEMA = StructType(
    [
        StructField("phone_number_AC_hash", StringType(), False),
        StructField("phone_number_AT_hash", StringType(), True),
        StructField("imei", StringType(), False),
        StructField("mno", StringType(), False),
        StructField("tac", StringType(), True),
        StructField("oldIMEI", StringType(), True),
        StructField("from_ts", TimestampType(), False),
        StructField("to_ts", TimestampType(), True),
        StructField("segment_id", IntegerType(), False),
        StructField("is_open", IntegerType(), False),
        StructField("duplicate_start_cnt", IntegerType(), False),
        StructField("orphan_end_cnt", IntegerType(), False),
        StructField("superseded_start_cnt", IntegerType(), False),
    ]
)


# ==========================================================================
# the walk
# ==========================================================================


def walk_device_history(events: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Build every device segment for one phone number from its full history.

    ``events`` may arrive in any order; they are sorted here by timestamp with
    ``record_id`` as the tie-break, so two events in the same second cannot order
    differently on different runs. Without that the same input would produce
    different Gold, and the convergence property the whole pipeline rests on
    would be false.

    Returns one dict per segment, in start order.
    """
    ordered = sorted(events, key=lambda e: (e["event_timestamp"], e["record_id"]))

    out: list[dict] = []
    open_seg: dict | None = None
    duplicate_start_cnt = 0
    orphan_end_cnt = 0
    superseded_start_cnt = 0

    def close(seg: dict, ts) -> bool:
        """Close ``seg`` at ``ts``, and say whether it was worth publishing.

        A segment closed at the instant it opened is not a short association, it
        is no association at all: ``[T, T)`` is half-open and contains no moment,
        so there is no point in time at which the number was on the device. It is
        a start the feed superseded before any time passed - a second start in
        the same second, or an end delivered alongside its own start.

        Publishing them was the fault behind ``uniqueness_imei_map_key``. A
        number that moves A -> B -> A within one timestamp produces an empty
        segment for A, an empty one for B and then the real, open one for A, and
        the two A rows share a start - which is exactly the key the check
        forbids. Nothing downstream wanted the empty rows either: they were also
        the whole of the notebook's 195,883 apparent overlaps, and
        :func:`~trust_score_05.lineage.dq.rules_gold._overlap_check` already
        discards them for the same reason before it counts anything.

        Dropping them here rather than filtering afterwards keeps ``segment_id``
        contiguous and lets the displaced device carry forward, so the table
        still reads backwards as a lineage over the rows it actually contains.
        """
        nonlocal superseded_start_cnt
        if ts <= seg["from_ts"]:
            superseded_start_cnt += 1
            return False
        seg["to_ts"] = ts
        seg["is_open"] = 0
        out.append(seg)
        return True

    for event in ordered:
        etype = event["event_type"]
        ts = event["event_timestamp"]

        if etype in DEVICE_START_EVENTS:
            imei = event.get("imei")
            if imei is None:
                # A start with no device names nothing. It cannot open a
                # segment and it is not evidence that the open one ended.
                continue

            if open_seg is not None:
                if open_seg["imei"] == imei and open_seg["from_ts"] == ts:
                    # The same start delivered twice. Not a re-attachment, and
                    # emitting a zero-length segment for it would be noise.
                    duplicate_start_cnt += 1
                    continue
                if open_seg["imei"] == imei:
                    # Same handset, later timestamp. The carrier re-announced an
                    # association that never lapsed - a refresh, a snapshot, a
                    # replayed row. Extending the open segment is right; closing
                    # and reopening would split one association into two rows and
                    # make the table claim a device change that never happened.
                    duplicate_start_cnt += 1
                    continue
                if close(open_seg, ts):
                    displaced = open_seg["imei"]
                else:
                    # The segment just closed was never a period, so it is not in
                    # the table and cannot be what this one displaced. What this
                    # one displaces, as the published rows read, is whatever that
                    # segment displaced - which is how A -> B -> A in a single
                    # instant comes out as one row for A naming the device that
                    # really did precede it.
                    displaced = open_seg["oldIMEI"]
            else:
                displaced = None

            open_seg = {
                SLICE_KEY: event[SLICE_KEY],
                "phone_number_AT_hash": event.get("phone_number_AT_hash"),
                "imei": imei,
                "mno": event["mno"],
                "tac": event.get("tac"),
                # The device this one displaced, which makes the table readable
                # backwards as a device lineage and not merely as a set of
                # intervals. ``oldIMEI`` as the carrier sent it is only sometimes
                # populated; this is derived and always right.
                #
                # Because it is a lineage it names a row that is in the table.
                # Where the displaced segment was superseded in the same instant
                # and so never published, the name skips past it to whatever it
                # displaced in turn - which may be nothing, and then this is null
                # and the segment is where the number's device history begins.
                "oldIMEI": displaced if displaced is not None else event.get("oldIMEI"),
                "from_ts": ts,
                "to_ts": None,
                "segment_id": 0,
                "is_open": 1,
            }

        elif etype in DEVICE_END_EVENTS:
            imei = event.get("imei")
            if open_seg is None:
                # An end with nothing open, or an end for a device a later start
                # already displaced. Either way the segment it refers to is
                # already closed at the right moment; counting it is the useful
                # response, and a rising count means the feed's starts and ends
                # disagree.
                orphan_end_cnt += 1
            elif imei is not None and open_seg["imei"] != imei:
                orphan_end_cnt += 1
            else:
                close(open_seg, ts)
                open_seg = None

        elif etype in SIM_END_EVENTS:
            # The SIM was cancelled, so the handset is no longer carrying this
            # phone number - whatever the device feed does or does not say next.
            if open_seg is not None:
                close(open_seg, ts)
                open_seg = None

    if open_seg is not None:
        out.append(open_seg)

    # Start order. Two segments can no longer share a start - the only way that
    # happened was one device displacing another in the same second, and the
    # displaced one is now dropped rather than published empty - but the rest of
    # the key is kept so the sort stays total and cannot reorder between runs.
    out.sort(key=lambda s: (s["from_ts"], s["to_ts"] is None, s["imei"]))
    for index, seg in enumerate(out, start=1):
        seg["segment_id"] = index
        seg["duplicate_start_cnt"] = duplicate_start_cnt
        seg["orphan_end_cnt"] = orphan_end_cnt
        seg["superseded_start_cnt"] = superseded_start_cnt
    return out


# ==========================================================================
# the stage
# ==========================================================================

_WALK_INPUT_COLUMNS = (
    SLICE_KEY,
    "phone_number_AT_hash",
    "mno",
    "imei",
    "tac",
    "oldIMEI",
    "event_type",
    "event_timestamp",
    "record_id",
)

_RELEVANT_EVENTS = DEVICE_START_EVENTS + DEVICE_END_EVENTS + SIM_END_EVENTS


def _walk_group(item) -> Iterable[tuple]:
    _, rows = item
    events = [dict(zip(_WALK_INPUT_COLUMNS, row)) for row in rows]
    fields = [f.name for f in IMEI_WORK_SCHEMA.fields]
    for seg in walk_device_history(events):
        yield tuple(seg[name] for name in fields)


def build_imei_map(spark: SparkSession, device_history: DataFrame) -> DataFrame:
    """Run :func:`walk_device_history` for every phone number in the slice.

    ``device_history`` is the *complete* device history of the slice's phone
    numbers, not the batch. A segment's start can be months before the event that
    caused this run, and a walk over the batch alone would invent a start at the
    batch boundary.
    """
    relevant = device_history.where(
        F.col("event_type").isin(list(_RELEVANT_EVENTS))
    ).select(
        F.col(SLICE_KEY),
        F.col("phone_number_AT_hash"),
        F.upper(F.col("mno")).alias("mno"),
        F.col("imei"),
        F.col("tac"),
        F.col("oldIMEI"),
        F.col("event_type"),
        F.col("event_timestamp"),
        F.col("record_id"),
    )
    # The partition count is passed rather than left to Spark, for the reason
    # in :func:`~trust_score_05.lineage.spark.shuffle_partitions`: an RDD
    # ``groupBy`` does not read ``spark.sql.shuffle.partitions``, so without
    # this the widest shuffle in the stage is the one nobody can tune. The whole
    # device feed crosses it on a rebuild.
    walked = (
        relevant.rdd.map(tuple)
        .groupBy(lambda r: r[0], shuffle_partitions(spark))
        .flatMap(_walk_group)
    )
    return spark.createDataFrame(walked, IMEI_WORK_SCHEMA)


def close_on_lifecycle_end(segments: DataFrame, lifecycles: DataFrame) -> DataFrame:
    """Stop an open device segment at the end of the lifecycle it belongs to.

    A phone number that was cancelled, or ported away, is not still carrying a
    handset. If the carrier's own ``IMEI_END`` closed the segment, that wins and
    nothing here applies. If nothing closed it and the phone number's last
    lifecycle has ended, the segment is closed there instead of running on
    forever.

    A segment that would be closed at the very moment it opened is dropped
    instead, on the same reasoning as :func:`walk_device_history`: the lifecycle
    ended in the instant the handset was attached, so there is no moment at which
    the number was on it, and a published ``[T, T)`` would be a row every
    point-in-time query has to be taught to skip.

    This is behind ``imei_map.close_on_lifecycle_end`` because it couples two
    jobs that are otherwise independent - the IMEI map would need the account
    lineage's output to build - and somebody will eventually want it off.
    """
    last_end = (
        lifecycles.where(F.col("lifecycle_is_open") == F.lit(0))
        .groupBy(SLICE_KEY)
        .agg(F.max("lifecycle_end_ts").alias("_lc_end"))
    )
    still_open = (
        lifecycles.groupBy(SLICE_KEY)
        .agg(F.max("lifecycle_is_open").alias("_any_open"))
        .where(F.col("_any_open") == F.lit(1))
        .select(SLICE_KEY)
    )
    return (
        segments.join(last_end, SLICE_KEY, "left")
        .join(still_open.withColumn("_open", F.lit(1)), SLICE_KEY, "left")
        .withColumn(
            "to_ts",
            F.when(
                F.col("to_ts").isNull()
                & F.col("_open").isNull()
                & F.col("_lc_end").isNotNull()
                & (F.col("_lc_end") >= F.col("from_ts")),
                F.col("_lc_end"),
            ).otherwise(F.col("to_ts")),
        )
        .withColumn(
            "is_open",
            F.when(F.col("to_ts").isNotNull(), F.lit(0)).otherwise(F.col("is_open")),
        )
        .where(F.col("to_ts").isNull() | (F.col("from_ts") < F.col("to_ts")))
        .drop("_lc_end", "_any_open", "_open")
    )


def to_gold(segments: DataFrame, run_ts: datetime) -> DataFrame:
    """Project the work frame onto ``msisdn_imei_map``.

    This table's bookkeeping is its own: one ``last_updated_ts`` rather than the
    ``created_ts`` / ``updated_ts`` pair, and ``event_date`` derived from
    ``from_ts`` as the partition. That is the shape the table already has and the
    shape its consumers already read.
    """
    stamped = segments.withColumn(
        "last_updated_ts", F.lit(run_ts).cast("timestamp")
    ).withColumn("event_date", F.to_date(F.col("from_ts")))
    return align_to_schema(stamped, IMEI_MAP_SCHEMA)
