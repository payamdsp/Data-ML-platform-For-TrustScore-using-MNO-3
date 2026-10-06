"""Data quality on the enrichment *inputs*, before they are believed.

Every other rule module in this package checks something the pipeline built.
These check something the pipeline was given, which is a different job and is
why they live apart.

The reason they exist at all is section 1 of the enrichment design: enrichment
is an input to the rebuild rather than an update on top of it, so a defect in
the feed does not sit in one column of one row - it propagates through the walk,
moves lifecycle boundaries, raises confidence values and changes which
subscriber a fraud is attributed to. By the time it reaches Gold it looks like a
conclusion the pipeline drew rather than a number somebody sent us. Catching it
at the door is the only place the evidence is still recognisable as evidence.

The three things worth catching
-------------------------------
**A timezone that was not converted.** TU timestamps are UTC in bronze and EST
in silver. A missed conversion shifts every port by five hours, which is enough
to move a same-day handover across ``windows.port_high_confidence_days`` and
flip a customer link from ``HIGH`` to ``LOW``. Five hours is a perfectly
plausible time for a port, so no single row looks wrong; the shift is visible
only in the aggregate. It is therefore checked both ways - per row against the
response that carried it, and per batch against the distribution of the history
it is being added to.

**A hash that means two different things.** The plaintext phone number is
encrypted at rest, so the obvious check - that a row's ``phone_number_AC_hash``
is the hash of its own ``msisdn`` - is not available to us and is deliberately
not written. What replaces it is agreement between the two hash columns across
every feed that carries both: they are the only identity we hold, so the thing
worth asserting is that they tell the same story everywhere.

**A number we have never heard of.** Not an error. ``never_seen`` is a real and
expected state - TU's history reaches back further than ours, which is most of
why we buy it - but a sudden rise in it means the fraud feed has started sending
numbers from carriers we hold no feed for, and nothing else in the run would say
so.

Thresholds arrive as arguments rather than being read from ``Config`` here. They
are check thresholds and not business rules - nothing about what a Gold column
*means* depends on them - but the module still refuses to hold the numbers
itself, because a threshold in a code literal is a threshold nobody can move
during an incident.
"""

from __future__ import annotations

import datetime as _dt
from typing import Sequence

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ..schemas import HEX64_REGEX, IDV_PRIMARY_KEY, SLICE_KEY
from .framework import (
    ABORT,
    INFO,
    WARN,
    Check,
    counter,
    matches_when_present,
    not_null,
    unique_by,
)

__all__ = [
    "TU_PORTPS_INPUT",
    "MNO_ACTIVATION_INPUT",
    "tu_portps_checks",
    "mno_activation_checks",
    "hash_agreement_checks",
    "modal_hour_checks",
    "first_seen_check",
    "never_seen_check",
    "modal_hour",
    "hour_drift",
]

#: Dataset labels the metrics rows are written under. They are not Gold tables
#: and must not read as though they were, so they carry the feed's own name with
#: an ``_input`` suffix - a reader scanning the metrics table can then tell at a
#: glance whether a row is about what we were sent or about what we built.
TU_PORTPS_INPUT = "tu_portps_input"
MNO_ACTIVATION_INPUT = "mno_activation_input"


# --------------------------------------------------------------------------
# tu_portps
# --------------------------------------------------------------------------


def tu_portps_checks(
    max_days_after_ingestion: int,
    run_ts: _dt.datetime,
) -> list[Check]:
    """Row-level checks on a TransUnion porting delivery.

    ``max_days_after_ingestion`` is small on purpose. Ingestion follows the
    event: we are told about a port after it happened, never before, so a row
    stamped meaningfully ahead of the response that carried it is a clock defect
    or a parse defect rather than an unusual port. It is the tightest of the
    three timestamp checks and the only one that aborts on its own.
    """
    return [
        not_null(
            SLICE_KEY,
            dimension="Completeness",
            severity=ABORT,
            description=(
                "every join in this pipeline is on the phone-number hash, so a "
                "null one is a row that can never be matched to anything. It "
                "aborts rather than warns because the row would otherwise be "
                "silently dropped by the join and counted nowhere"
            ),
        ),
        not_null(
            "event_timestamp",
            dimension="Completeness",
            severity=ABORT,
            description=(
                "the port's own time is what the stay walk orders lifecycles by; "
                "without it the row cannot take part in the walk at all"
            ),
        ),
        matches_when_present(
            SLICE_KEY,
            HEX64_REGEX,
            name=f"validity_tu_{SLICE_KEY.lower()}_format",
            severity=WARN,
        ),
        matches_when_present(
            "phone_number_AT_hash",
            HEX64_REGEX,
            name="validity_tu_phone_number_at_hash_format",
            severity=WARN,
        ),
        Check(
            name="timeliness_tu_event_after_ingestion",
            dimension="Timeliness",
            severity=ABORT,
            checked_field="event_timestamp, ingestion_ts",
            description=(
                "a port stamped further than the configured margin ahead of the "
                "response that delivered it. Ingestion follows the event, so "
                "this is a clock or a parse defect - most likely a timestamp "
                "read in the wrong zone, which is the failure this whole group "
                "of checks exists for"
            ),
            violation=(
                F.col("event_timestamp").isNotNull()
                & F.col("ingestion_ts").isNotNull()
                & (
                    F.col("event_timestamp")
                    > F.col("ingestion_ts")
                    + F.expr(f"INTERVAL {int(max_days_after_ingestion)} DAYS")
                )
            ),
        ),
        Check(
            name="timeliness_tu_event_in_the_future",
            dimension="Timeliness",
            severity=WARN,
            checked_field="event_timestamp",
            description=(
                "a port stamped after the moment this run started. Weaker than "
                "the ingestion check and kept separately: a run replaying an old "
                "window will not trip this, and a run whose own clock is wrong "
                "will trip only this, so which of the two fires narrows the "
                "cause considerably"
            ),
            violation=F.col("event_timestamp") > F.lit(run_ts),
        ),
        Check(
            name="consistency_tu_event_date_matches_timestamp",
            dimension="Consistency",
            severity=WARN,
            checked_field="event_timestamp, event_date",
            description=(
                "``event_date`` is the date part of ``event_timestamp``, so the "
                "two disagree only when one of them was derived in a different "
                "zone from the other. A handful of rows disagreeing means ports "
                "near midnight; a large share disagreeing is a zone bug, and "
                "this is the cheapest place it shows up"
            ),
            violation=(
                F.col("event_timestamp").isNotNull()
                & F.col("event_date").isNotNull()
                & (F.to_date(F.col("event_timestamp")) != F.col("event_date"))
            ),
        ),
    ]


# --------------------------------------------------------------------------
# mno_activation
# --------------------------------------------------------------------------


def mno_activation_checks(run_ts: _dt.datetime) -> list[Check]:
    """Row-level checks on the IDV activation feed.

    ``activation_date`` is a date and not a timestamp - the API returns
    ``MM/DD/YYYY`` with no time of day - so every comparison here treats it as
    midnight, the same reading the boundary stage uses when it moves a start.
    """
    return [
        not_null(
            SLICE_KEY,
            dimension="Completeness",
            severity=ABORT,
            description=(
                "the answer is about a phone number; without the hash there is "
                "no lifecycle it could be applied to"
            ),
        ),
        not_null(
            "response_code",
            dimension="Completeness",
            severity=ABORT,
            description=(
                "the boundary stage reads only successful answers, so a null "
                "response code is an answer we cannot classify as either. "
                "Treating it as a failure would silently discard evidence and "
                "treating it as a success would apply an unverified date"
            ),
        ),
        unique_by(
            list(IDV_PRIMARY_KEY),
            name="uniqueness_idv_primary_key",
            severity=WARN,
            description=(
                "the request id comes from outside our control, so a replay "
                "arrives as a second row rather than as an overwrite. Warn "
                "rather than abort: duplicates of an identical answer change "
                "nothing, and the stage takes the earliest successful answer per "
                "phone number regardless"
            ),
        ),
        Check(
            name="timeliness_idv_activation_date_in_the_future",
            dimension="Timeliness",
            severity=WARN,
            checked_field="activation_date",
            description=(
                "an activation date after the moment this run started. The "
                "boundary stage refuses these anyway - a start it cannot place "
                "is counted as a conflict rather than applied - so this is the "
                "metric that says how much of the conflict count is the feed "
                "rather than a genuine disagreement with a carrier"
            ),
            violation=F.col("activation_date") > F.lit(run_ts.date()),
        ),
        Check(
            name="timeliness_idv_api_ts_in_the_future",
            dimension="Timeliness",
            severity=WARN,
            checked_field="api_ts",
            description="the call itself is stamped after this run started",
            violation=F.col("api_ts") > F.lit(run_ts),
        ),
    ]


# --------------------------------------------------------------------------
# cross-feed hash agreement
# --------------------------------------------------------------------------


def _pairs(frames: Sequence[DataFrame]) -> DataFrame | None:
    """The distinct ``(AC, AT)`` pairs carried by every frame that has both."""
    usable = [
        f.select(SLICE_KEY, "phone_number_AT_hash")
        for f in frames
        if f is not None
        and SLICE_KEY in f.columns
        and "phone_number_AT_hash" in f.columns
    ]
    if not usable:
        return None
    out = usable[0]
    for frame in usable[1:]:
        out = out.unionByName(frame)
    return out.where(
        F.col(SLICE_KEY).isNotNull() & F.col("phone_number_AT_hash").isNotNull()
    ).distinct()


def _fan_out(pairs: DataFrame | None, key: str, other: str):
    """How many ``key`` values carry more than one distinct ``other``."""

    def _fn(_df: DataFrame) -> int:
        if pairs is None:
            return 0
        return (
            pairs.groupBy(key)
            .agg(F.countDistinct(other).alias("__n"))
            .where(F.col("__n") > 1)
            .count()
        )

    return _fn


def hash_agreement_checks(frames: Sequence[DataFrame]) -> list[Check]:
    """The two hash columns must tell the same story in every feed.

    This is what stands in for the hash-consistency check we cannot write. Both
    directions are asserted because they fail for different reasons and want
    different responses: one AT hash under two AC hashes means two phone numbers
    were conflated, and one AC hash under two AT hashes means one phone number
    was split in half. Either way an analysis keyed on AC that arrived through an
    AT-only table is quietly wrong.

    Warn rather than abort. The damage is historical by the time we see it, and
    refusing to run does not undo it - it only stops us learning anything else
    about the batch.
    """
    pairs = _pairs(frames)
    return [
        Check(
            name="consistency_at_hash_maps_to_one_ac_hash",
            dimension="Consistency",
            severity=WARN,
            checked_field=f"phone_number_AT_hash, {SLICE_KEY}",
            description=(
                "an AT hash appearing under two different AC hashes: two phone "
                "numbers conflated into one identity across the feeds"
            ),
            frame_fn=_fan_out(pairs, "phone_number_AT_hash", SLICE_KEY),
        ),
        Check(
            name="consistency_ac_hash_maps_to_one_at_hash",
            dimension="Consistency",
            severity=WARN,
            checked_field=f"{SLICE_KEY}, phone_number_AT_hash",
            description=(
                "an AC hash appearing under two different AT hashes: one phone "
                "number split into two identities across the feeds"
            ),
            frame_fn=_fan_out(pairs, SLICE_KEY, "phone_number_AT_hash"),
        ),
    ]


# --------------------------------------------------------------------------
# the aggregate timezone check
# --------------------------------------------------------------------------


def modal_hour(frame: DataFrame | None, column: str = "event_timestamp") -> int | None:
    """The most common hour of day in ``frame``, or ``None`` if there is none.

    Ties are broken by taking the lower hour, which matters only for making the
    function deterministic: a tie means the distribution is flat enough that no
    drift measured against it would be meaningful anyway.
    """
    if frame is None or column not in frame.columns:
        return None
    rows = (
        frame.where(F.col(column).isNotNull())
        .groupBy(F.hour(F.col(column)).alias("__hour"))
        .count()
        .orderBy(F.col("count").desc(), F.col("__hour").asc())
        .limit(1)
        .collect()
    )
    return int(rows[0]["__hour"]) if rows else None


def hour_drift(a: int | None, b: int | None) -> int:
    """Distance between two hours of day, the short way round the clock.

    23 and 0 are one hour apart, not twenty-three. Getting this wrong would make
    the check scream every time a feed's busiest hour sat near midnight, which
    is exactly where a five-hour shift is most likely to put it.
    """
    if a is None or b is None:
        return 0
    gap = abs(int(a) - int(b)) % 24
    return min(gap, 24 - gap)


def modal_hour_checks(
    batch_modal_hour: int | None,
    baseline_modal_hour: int | None,
    max_drift_hours: int,
) -> list[Check]:
    """Compare the delivery's busiest hour against the history it joins.

    The baseline is the TransUnion history we already hold for this slice with
    the current delivery removed, not the previous run's number. That is a
    deliberate departure from the first sketch of this check, and it is worth
    saying why: reading the previous run's value back out of the metrics table
    would make an abort depend on a table that can be disabled, can be
    partitioned differently per environment, and is empty on a first run. The
    history is already in the frame we are about to walk, it is a far larger
    sample than one batch, and comparing against it answers the same question -
    "is today's delivery stamped like every delivery before it".

    The delivery is excluded from its own baseline for the obvious reason: a
    shifted batch left in the baseline drags the baseline towards itself and the
    measured drift shrinks by exactly the amount that matters.

    Silent when either side is missing, which covers the first delivery for a
    slice, a delivery for phone numbers we have never seen before, and a feed
    that is switched off. A check with no baseline has nothing to say, and
    saying it loudly would train an operator to ignore it.
    """
    drift = hour_drift(batch_modal_hour, baseline_modal_hour)
    return [
        counter(
            "info_tu_batch_modal_hour",
            -1 if batch_modal_hour is None else batch_modal_hour,
            "Timeliness",
            description=(
                "busiest hour of day in this delivery, or -1 where the delivery "
                "is empty. Recorded on every run so the trend is readable even "
                "on runs where the drift check found nothing to say"
            ),
            checked_field="event_timestamp",
        ),
        counter(
            "info_tu_baseline_modal_hour",
            -1 if baseline_modal_hour is None else baseline_modal_hour,
            "Timeliness",
            description=(
                "busiest hour of day in the history this delivery joins, with "
                "the delivery itself excluded, or -1 where there is no history"
            ),
            checked_field="event_timestamp",
        ),
        Check(
            name="consistency_tu_modal_hour_drift",
            dimension="Consistency",
            severity=ABORT,
            checked_field="event_timestamp",
            description=(
                f"the delivery's busiest hour moved more than "
                f"{int(max_drift_hours)} hour(s) from the history it joins. A "
                "whole feed read in the wrong timezone shifts every row by the "
                "same amount, which no single row reveals and this does. It "
                "aborts because the rows would otherwise be believed: a "
                "five-hour shift is enough to move a same-day handover across "
                "the port window and flip a customer link"
            ),
            frame_fn=lambda _df: 1 if drift > int(max_drift_hours) else 0,
        ),
    ]


# --------------------------------------------------------------------------
# how far back the feed is allowed to reach
# --------------------------------------------------------------------------


def first_seen_check(
    first_seen: DataFrame | None,
    max_days_before_first_seen: int,
) -> Check:
    """A port may precede our own first sight of the number, but only so far.

    TU's history reaches back further than ours does, which is the entire reason
    we buy it, so a port before our first event for a number is normal and
    expected. What is not normal is a port before our first event by decades:
    that is a hash collision or a misparsed year, and applying it would open a
    lifecycle in the wrong century and push every later boundary around it.

    ``first_seen`` is one row per phone number with the earliest timestamp any
    of our own feeds holds for it. A number absent from it is not checked here -
    that is the ``never_seen`` case, which has its own counter and is a
    different thing entirely.
    """
    limit = int(max_days_before_first_seen)

    def _fn(df: DataFrame) -> int:
        if first_seen is None or "event_timestamp" not in df.columns:
            return 0
        joined = df.join(first_seen, SLICE_KEY, "inner")
        return joined.where(
            F.col("event_timestamp").isNotNull()
            & F.col("first_seen_ts").isNotNull()
            & (
                F.col("event_timestamp")
                < F.col("first_seen_ts") - F.expr(f"INTERVAL {limit} DAYS")
            )
        ).count()

    return Check(
        name="timeliness_tu_event_before_first_seen",
        dimension="Timeliness",
        severity=ABORT,
        checked_field="event_timestamp",
        description=(
            f"a port more than {limit} days before our own first sight of the "
            "phone number. Reaching further back than we do is the point of the "
            "feed; reaching this much further is a collision or a misparsed year"
        ),
        frame_fn=_fn,
    )


def never_seen_check(first_seen: DataFrame | None) -> Check:
    """Phone numbers TransUnion knows about and our own feeds do not.

    An expected state, not a failure, and so ``info``. It is worth counting
    because a rise in it is the earliest visible sign that the fraud feed has
    started sending numbers from carriers we hold no feed for - which changes
    what a ``LOW`` confidence on those numbers means, and which nothing else in
    the run reports.
    """

    def _fn(df: DataFrame) -> int:
        numbers = df.select(SLICE_KEY).distinct()
        if first_seen is None:
            return numbers.count()
        return numbers.join(first_seen, SLICE_KEY, "left_anti").count()

    return Check(
        name="info_tu_phones_never_seen",
        dimension="Completeness",
        severity=INFO,
        checked_field=SLICE_KEY,
        description=(
            "phone numbers in the delivery with no row in any feed of our own. "
            "Expected and non-zero: TransUnion is the only witness we have for "
            "carriers we hold no feed for. A rise means the fraud feed's mix of "
            "carriers has moved"
        ),
        frame_fn=_fn,
    )
