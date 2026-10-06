"""Data-quality rules for the six Gold tables.

Silver's rules ask whether a carrier sent well-formed data. Gold's ask whether
*this pipeline* built a coherent structure out of it, and that is a different
question with a different answer to "who do I call about this?".

That difference drives the tiering, and it is more aggressive here than in
Silver. A null ``imei`` in Silver is a carrier problem, warned about and shipped.
A lifecycle whose end precedes its start is a Gold problem, and there is no
carrier to raise it with - it means the walk is wrong. Structural invariants are
therefore ``abort``:

    every key present and 64-char hex
    every interval ordered, ``from_ts <= to_ts``
    no two lifecycles of one phone number overlapping in time
    every ``lifecycle_uid`` in the account mapping is a lifecycle this run built
    every ``acct_id`` in the customer mapping is an account this run built

``warn`` covers the vocabularies - segment types, account types, infer tokens -
which are open enough that a new value is news rather than a catastrophe. ``info``
covers the numbers that are *expected* to be non-zero and are worth trending:
inferred starts, open lifecycles, chains broken or truncated, and what each merge
inserted, updated and deleted.

One deliberate omission: there is no freshness or row-count check. A Gold run
processes a slice, and the slice is as large as the batch made it. A run that
rebuilds forty rows is not a failure, it is a quiet day.
"""

from __future__ import annotations

from typing import Sequence

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ..schemas import (
    AC_EVENT_TYPE_DOMAIN,
    ACCOUNT_TYPE_DOMAIN,
    CONFIDENCE_DOMAIN,
    HEX64_REGEX,
    LIFECYCLE_END_TOKENS,
    LIFECYCLE_START_TOKENS,
    LINK_ROUTE_DOMAIN,
    MNO_DOMAIN,
    SEGMENT_TYPE_DOMAIN,
    SLICE_KEY,
    UPDATED_SOURCE_MAPPING_DOMAIN,
    UPDATED_SOURCE_NORMALIZED_DOMAIN,
)
from ..transforms.accounts import (
    TRUNCATED_ACCOUNT_TYPES,
    TRUNCATED_FRAGMENT,
    TRUNCATED_SEGMENT_TYPES,
)
from ..transforms.boundaries import IDV_START_TOKEN
from ..transforms.lifecycle import LOW_BOUNDARY_MNOS
from .framework import (
    ABORT,
    INFO,
    WARN,
    Check,
    counter,
    in_domain,
    matches_when_present,
    not_null,
    references,
    unique,
    unique_by,
)

__all__ = [
    "canonical_events_checks",
    "lifecycle_checks",
    "account_mapping_checks",
    "customer_mapping_checks",
    "normalized_events_checks",
    "imei_map_checks",
    "merge_stat_checks",
]


# ==========================================================================
# shared pieces
# ==========================================================================


def _slice_key_checks(prefix: str) -> list[Check]:
    """Every Gold table carries the slice key, so every table checks it the same.

    ``abort`` rather than ``warn``, and that is not excessive. The slice key is
    what scopes the merge and what scopes the delete branch. A null or malformed
    value in it does not produce a slightly wrong row - it produces a row the
    next run cannot find, and therefore cannot correct, forever.
    """
    return [
        not_null(
            SLICE_KEY,
            dimension="Completeness",
            severity=ABORT,
            description=(
                f"{prefix}: {SLICE_KEY} is null - the row cannot be scoped to a "
                "slice and no later run will be able to correct it"
            ),
        ),
        matches_when_present(
            SLICE_KEY,
            HEX64_REGEX,
            name="validity_phone_number_ac_hash_format",
            severity=ABORT,
            description=f"{prefix}: {SLICE_KEY} is not 64-character hex",
        ),
    ]


def _interval_ordered(
    from_col: str, to_col: str, name: str, description: str
) -> Check:
    """``from_ts <= to_ts`` where an end exists.

    An interval that ends before it starts is not a data-quality nuance, it is a
    contradiction, and every consumer that does a point-in-time lookup will
    quietly return nothing for it rather than fail. Hence ``abort``.
    """
    return Check(
        name=name,
        dimension="Consistency",
        severity=ABORT,
        checked_field=f"{from_col}, {to_col}",
        description=description,
        violation=F.col(to_col).isNotNull() & (F.col(to_col) < F.col(from_col)),
    )


def _overlap_check(
    partition: Sequence[str],
    from_col: str,
    to_col: str,
    order_col: str,
    name: str,
    description: str,
    severity: str = ABORT,
) -> Check:
    """Count intervals in the same partition that overlap in time.

    Whether an overlap is a fault depends entirely on what the partition is, so
    the severity is the caller's to choose and both answers are in use.

    On a **phone number** it aborts. "Which device was this number on last
    March?" and "which carrier held it?" both assume one answer, and both
    silently return two if this fails.

    On an **account or a customer** it warns. Those partitions hold several phone
    numbers on purpose, and two of them being live at once is what a number
    change looks like from the inside rather than a contradiction; see the
    reasoning at each of those call sites.

    It is a frame-level check because it needs a window: it counts rows that
    start before the furthest point any earlier row in the same partition
    reaches.

    Two things about how the comparison is made, both of which this check used
    to get wrong and both of which hid the failure it exists to find.

    **A missing end means the interval is still running, not that it has none.**
    An open interval reaches forward without limit, so everything after it
    overlaps it. Comparing against a null end and discarding the null - which is
    what a plain ``lag`` on the end column does - means a phone number with two
    live lifecycles, or one sitting on two handsets right now, counts zero
    violations. That is the single state this check most needs to catch, because
    it is the one a point-in-time question is most likely to be asked about.
    Nulls are therefore read as the far future.

    **The overlap need not be with the row immediately before.** One long
    interval can contain several short ones, and the short one after the first
    contained interval compares clean against its immediate predecessor while
    still sitting inside the long one. The comparison is against a running
    maximum of every earlier end in the partition rather than against the
    previous row alone.

    Intervals are half-open, so a row starting exactly where the previous one
    ended is not an overlap.

    **An interval that ends where it starts is never live, so it takes no part
    in this.** The walk emits those: a port arriving in the same second as the
    activation it supersedes closes one lifecycle and opens the next at a single
    instant, and a carrier change at a single instant does the same. A half-open
    ``[T, T)`` contains no moment, so it cannot be one of two lifecycles live at
    the same moment - which is the sentence this check reports when it fires.

    Left in, they were reported anyway, and reported at random. The window
    breaks ties on ``order_col``, a hash, so an empty interval sharing its start
    with the real one that follows it sorted before that interval about half the
    time and after it the other half. Sorted after, the running maximum already
    held the real interval's end, the empty interval's start was below it, and
    the row was counted. The same data, with a different hash, counted zero.
    That is how the first full rebuild aborted on 1,658,031 of 77,172,342
    lifecycles, none of which overlapped anything.

    Dropping them before the window is safe in both directions. They cannot be
    counted, which is the point. They also cannot be missed from the running
    maximum: rows arrive in start order, so every later row starts at or after
    an empty interval's start, which is the whole of its reach. Inverted
    intervals go with them for the same reason and are the business of
    ``_interval_ordered``, which names the fault properly rather than describing
    it as an overlap.
    """
    from pyspark.sql import Window

    cols = list(partition)
    # Later than any timestamp the carriers can produce, and representable in a
    # Spark timestamp. Only ever compared against, never written.
    forever = F.lit("9999-12-31 23:59:59").cast("timestamp")

    def _fn(df: DataFrame) -> int:
        order = (
            Window.partitionBy(*cols)
            .orderBy(F.col(from_col).asc(), F.col(order_col).asc())
            .rowsBetween(Window.unboundedPreceding, -1)
        )
        # Null where the row has no predecessor, which is the one case that
        # cannot be an overlap and the only reason this is allowed to be null.
        reach = F.max(F.coalesce(F.col(to_col), forever)).over(order)
        live = F.col(to_col).isNull() | (F.col(from_col) < F.col(to_col))
        return (
            df.where(live)
            .withColumn("_reach", reach)
            .where(F.col("_reach").isNotNull() & (F.col(from_col) < F.col("_reach")))
            .count()
        )

    return Check(
        name=name,
        dimension="Consistency",
        severity=severity,
        checked_field=f"{from_col}, {to_col}",
        description=description,
        frame_fn=_fn,
    )


def _at_most_one_open(
    partition: Sequence[str],
    to_col: str,
    name: str,
    description: str,
    severity: str = ABORT,
) -> Check:
    """At most one interval in the partition may be missing its end.

    The overlap check above would also catch this, now that it reads a missing
    end as the far future. This one is kept beside it because the two failures
    have different causes and want different answers. Overlapping closed
    intervals mean the boundaries were computed wrongly and the question is which
    of the two is right. Two open intervals mean nothing closed the first one -
    the close step did not run, or ran and did not match - and the question is
    what happened to the end, not where it should have been. A report that says
    only "these overlap" sends whoever reads it looking in the wrong place.

    It is also the check ``data_schema/gold/gold_msisdn_imei_map.md`` has always
    said this table carries.
    """
    cols = list(partition)

    def _fn(df: DataFrame) -> int:
        offenders = (
            df.where(F.col(to_col).isNull())
            .groupBy(*cols)
            .count()
            .where(F.col("count") > 1)
        )
        # The rows at fault, not the phone numbers, so the number is comparable
        # with every other violation count in the report.
        return int(offenders.agg(F.coalesce(F.sum("count"), F.lit(0))).collect()[0][0])

    return Check(
        name=name,
        dimension="Consistency",
        severity=severity,
        checked_field=to_col,
        description=description,
        frame_fn=_fn,
    )


def _confidence_columns(prefix: str, columns: Sequence[str]) -> list[Check]:
    """Every confidence column is present and says one of two things.

    ``abort`` on both, which is stricter than the ``warn`` the other vocabularies
    get, and the reason is what a consumer does with the value. ``segment_type``
    is read by a human; a new token there is news. Confidence is read by the
    ownership resolver as a branch - ``HIGH`` takes one path, anything else takes
    the other - so a null or an unexpected third value does not surface as an
    odd-looking row, it silently sends every affected subscriber down the wrong
    branch. There is no partial-credit reading of a two-valued column.
    """
    checks: list[Check] = []
    for column in columns:
        checks.append(
            not_null(
                column,
                dimension="Completeness",
                severity=ABORT,
                description=(
                    f"{prefix}: {column} is null. The resolver branches on this "
                    "value, so a null is not a missing annotation - it is an "
                    "unmade decision that reads as 'not HIGH'"
                ),
            )
        )
        checks.append(
            in_domain(
                column,
                CONFIDENCE_DOMAIN,
                name=f"validity_{column}_domain",
                severity=ABORT,
                description=(
                    f"{prefix}: {column} is outside HIGH/LOW. There is no MEDIUM "
                    "by design - a routing decision is binary"
                ),
            )
        )
    return checks


def _rollup_check(
    name: str, link_col: str, rollup_col: str, group_col: str, description: str
) -> Check:
    """A chain cannot claim to be sound while holding a link that is not.

    The rollup is a minimum over the whole chain, so this is the arithmetic
    restated as an assertion. It earns its place because the two columns are
    computed in different ways - the link from one edge, the rollup from a window
    over the chain plus two flags about how the chain was built - and a join that
    lost rows would leave the rollup looking better than its parts.
    """

    def _fn(df: DataFrame) -> int:
        return (
            df.groupBy(group_col)
            .agg(
                F.max(F.when(F.col(link_col) == F.lit("LOW"), 1).otherwise(0)).alias(
                    "_has_low"
                ),
                F.min(
                    F.when(F.col(rollup_col) == F.lit("HIGH"), 1).otherwise(0)
                ).alias("_all_high"),
            )
            .where((F.col("_has_low") == F.lit(1)) & (F.col("_all_high") == F.lit(1)))
            .count()
        )

    return Check(
        name=name,
        dimension="Consistency",
        severity=ABORT,
        checked_field=f"{link_col}, {rollup_col}",
        description=description,
        frame_fn=_fn,
    )


def _share(name: str, condition, description: str) -> Check:
    """An ``info`` metric: how many rows are in a state that is normal but worth
    watching. Never a failure - the ``info`` tier exists so that a number which
    is *supposed* to be non-zero does not read as one.
    """
    return Check(
        name=name,
        dimension="Completeness",
        severity=INFO,
        description=description,
        violation=condition,
    )


# ==========================================================================
# account_changes_canonical_events
# ==========================================================================


def canonical_events_checks() -> list[Check]:
    return [
        not_null("ac_event_id", severity=ABORT, description="canonical event has no id"),
        matches_when_present(
            "ac_event_id",
            HEX64_REGEX,
            name="validity_ac_event_id_format",
            severity=ABORT,
        ),
        unique("ac_event_id", severity=ABORT, description="two canonical events share an id"),
        *_slice_key_checks("canonical_events"),
        not_null("ac_event_ts", severity=ABORT, description="canonical event has no timestamp"),
        in_domain("mno", MNO_DOMAIN, severity=WARN),
        in_domain(
            "ac_event_type",
            AC_EVENT_TYPE_DOMAIN,
            name="consistency_ac_event_type_domain",
            severity=WARN,
            description=(
                "canonical event type outside the eleven Gold models. A new value "
                "means a carrier feed changed and stage 1 has not caught up"
            ),
        ),
        _share(
            "info_derived_event_share",
            F.col("ac_event_type").isin(
                [
                    "account_activation_mno_port",
                    "account_activation_number_recycle",
                    "plan_change",
                ]
            ),
            "events Gold inferred rather than a carrier sending them",
        ),
    ]


# ==========================================================================
# msisdn_lifecycle
# ==========================================================================


def lifecycle_checks() -> list[Check]:
    return [
        not_null("lifecycle_uid", severity=ABORT),
        matches_when_present(
            "lifecycle_uid", HEX64_REGEX, name="validity_lifecycle_uid_format", severity=ABORT
        ),
        unique(
            "lifecycle_uid",
            severity=ABORT,
            description=(
                "two lifecycles share a uid. The seed is (phone, mno, anchor "
                "event), so a collision means two lifecycles claim the same "
                "anchor - which the walk cannot produce and a bad merge can"
            ),
        ),
        *_slice_key_checks("lifecycle"),
        not_null("lifecycle_start_ts", severity=ABORT),
        _interval_ordered(
            "lifecycle_start_ts",
            "lifecycle_end_ts",
            name="consistency_lifecycle_interval_ordered",
            description="lifecycle ends before it starts",
        ),
        _overlap_check(
            partition=[SLICE_KEY],
            from_col="lifecycle_start_ts",
            to_col="lifecycle_end_ts",
            order_col="lifecycle_uid",
            name="consistency_lifecycle_no_overlap",
            description=(
                "one phone number has two lifecycles live at the same moment. "
                "Every point-in-time question about that number now has two "
                "answers"
            ),
        ),
        _at_most_one_open(
            partition=[SLICE_KEY],
            to_col="lifecycle_end_ts",
            name="consistency_one_open_lifecycle_per_number",
            description=(
                "a phone number has two lifecycles with no end. A number is with "
                "one carrier at a time, so the earlier one should have been closed "
                "when the later one opened and was not"
            ),
        ),
        Check(
            name="consistency_open_lifecycle_has_no_end",
            dimension="Consistency",
            severity=ABORT,
            checked_field="lifecycle_is_open, lifecycle_end_ts",
            description="lifecycle_is_open disagrees with lifecycle_end_ts",
            violation=(
                (F.col("lifecycle_is_open") == F.lit(1))
                & F.col("lifecycle_end_ts").isNotNull()
            )
            | (
                (F.col("lifecycle_is_open") == F.lit(0))
                & F.col("lifecycle_end_ts").isNull()
            ),
        ),
        Check(
            name="consistency_open_lifecycle_end_inferred",
            dimension="Consistency",
            severity=WARN,
            checked_field="is_end_inferred",
            description=(
                "an open lifecycle must carry is_end_inferred = 1. The flag means "
                "'we never saw an end', which is exactly the state of a lifecycle "
                "still running - it is not a claim that an end was computed"
            ),
            violation=(F.col("lifecycle_is_open") == F.lit(1))
            & (F.col("is_end_inferred") != F.lit(1)),
        ),
        Check(
            name="consistency_lifecycle_id_positive",
            dimension="Validity",
            severity=WARN,
            checked_field="lifecycle_id",
            description="lifecycle_id and mno_segment_id are 1-based ordinals",
            violation=(F.col("lifecycle_id") < F.lit(1))
            | (F.col("mno_segment_id") < F.lit(1)),
        ),
        # Not `in_domain("mno", MNO_DOMAIN)`. This is the one table that can
        # legitimately hold a carrier outside the domain: a TU-derived lifecycle
        # sits at a carrier we have no feed from, and its `mno` holds that
        # carrier's resolved name. So the rule relaxes to *non-empty, and in the
        # domain if supported*, which is strictly more information than the old
        # one carried - it now catches a supported lifecycle at a bogus carrier
        # AND an unsupported one that forgot to say so.
        Check(
            name="consistency_mno_domain",
            dimension="Consistency",
            severity=WARN,
            checked_field="mno, mno_is_supported",
            description=(
                "mno is empty, or a supported lifecycle names a carrier outside "
                f"{sorted(MNO_DOMAIN)}, or an unsupported one names a carrier "
                "inside it. The third case is the interesting one: it means a "
                "TU-derived lifecycle was built at a carrier we do have a feed "
                "from, which is a stage that should have merged and did not"
            ),
            violation=(
                F.col("mno").isNull()
                | (F.trim(F.col("mno")) == F.lit(""))
                | (
                    (F.col("mno_is_supported") == F.lit(1))
                    & ~F.col("mno").isin(sorted(MNO_DOMAIN))
                )
                | (
                    (F.col("mno_is_supported") == F.lit(0))
                    & F.col("mno").isin(sorted(MNO_DOMAIN))
                )
            ),
        ),
        Check(
            name="validity_mno_is_supported_flag",
            dimension="Validity",
            severity=ABORT,
            checked_field="mno_is_supported, tu_enriched",
            description="mno_is_supported and tu_enriched are 0/1 flags",
            violation=~F.col("mno_is_supported").isin([0, 1])
            | ~F.col("tu_enriched").isin([0, 1])
            | F.col("mno_is_supported").isNull()
            | F.col("tu_enriched").isNull(),
        ),
        Check(
            name="consistency_unsupported_lifecycle_is_tu_derived",
            dimension="Consistency",
            severity=ABORT,
            checked_field="mno_is_supported, tu_enriched",
            description=(
                "a lifecycle at an unsupported carrier can only have come from "
                "TU - nothing else in this pipeline can see one. An unsupported "
                "lifecycle with tu_enriched = 0 means a stage invented a carrier"
            ),
            violation=(F.col("mno_is_supported") == F.lit(0))
            & (F.col("tu_enriched") == F.lit(0)),
        ),
        _share(
            "info_unsupported_carrier_share",
            F.col("mno_is_supported") == F.lit(0),
            "lifecycles at carriers we have no feed from - the missing middle "
            "TU is the only witness to",
        ),
        _share(
            "info_tu_enriched_share",
            F.col("tu_enriched") == F.lit(1),
            "lifecycles on phone numbers TU was asked about, whatever it said",
        ),
        Check(
            name="consistency_lifecycle_infer_type_shape",
            dimension="Consistency",
            severity=WARN,
            checked_field="lifecycle_infer_type",
            description=(
                "lifecycle_infer_type is a start token plus an end token, so it "
                "always contains the '_to_' or '_open_' separator"
            ),
            violation=~(
                F.col("lifecycle_infer_type").contains("_to_")
                | F.col("lifecycle_infer_type").contains("_open_")
            ),
        ),
        Check(
            name="validity_lifecycle_infer_type_vocabulary",
            dimension="Validity",
            severity=WARN,
            checked_field="lifecycle_infer_type",
            description=(
                "the column is published as a closed vocabulary - every value is "
                "one of the start tokens followed by one of the end tokens - and "
                "consumers read it by matching against that list. The walk is no "
                "longer its only writer, so this check reads the published "
                "vocabulary and warns the moment a stage emits a token the "
                "schema module has not been told about. It is a WARN rather than "
                "an ABORT because an unpublished token is a documentation defect: "
                "the row itself is still correct, and refusing the whole run "
                "would be a heavier penalty than the mistake deserves"
            ),
            violation=~F.col("lifecycle_infer_type").isin(
                sorted(
                    start + end
                    for start in LIFECYCLE_START_TOKENS
                    for end in LIFECYCLE_END_TOKENS
                )
            ),
        ),
        _share(
            "info_inferred_start_share",
            F.col("is_start_inferred") == F.lit(1),
            "lifecycles whose start we never saw - the number was already live "
            "when the feed began, or the carrier sent no activation",
        ),
        _share(
            "info_open_lifecycle_share",
            F.col("lifecycle_is_open") == F.lit(1),
            "lifecycles still running",
        ),
        _share(
            "info_port_lifecycle_share",
            F.col("mno_port_ind") == F.lit(1),
            "lifecycles opened by a port",
        ),
        _share(
            "info_recycle_lifecycle_share",
            F.col("recycle_ind") == F.lit(1),
            "lifecycles opened by a number recycle - a different subscriber on a "
            "reassigned number",
        ),
        *_confidence_columns(
            "lifecycle", ("start_confidence", "end_confidence", "confidence")
        ),
        Check(
            name="consistency_confidence_is_the_weaker_boundary",
            dimension="Consistency",
            severity=ABORT,
            checked_field="start_confidence, end_confidence, confidence",
            description=(
                "confidence must be HIGH exactly when both boundaries are. A "
                "lifecycle is only as good as its weaker edge, and there is no "
                "half-answer to 'was this number theirs on the 3rd of June'"
            ),
            violation=(
                (F.col("confidence") == F.lit("HIGH"))
                != (
                    (F.col("start_confidence") == F.lit("HIGH"))
                    & (F.col("end_confidence") == F.lit("HIGH"))
                )
            ),
        ),
        Check(
            name="consistency_low_boundary_carrier_start_is_low",
            dimension="Consistency",
            severity=ABORT,
            checked_field="mno, start_confidence, lifecycle_infer_type",
            description=(
                "a carrier that sends no trustworthy activation cannot produce a "
                "HIGH start on its own evidence. The carrier list lives with the "
                "walk that applies it, so this check reads the same constant the "
                "walk does and starts failing the moment the two drift apart. A "
                "start firmed by an observed activation date is exempt, and says "
                "so on the row: the enrichment stage writes its own start token, "
                "so the exemption is the evidence rather than a hole in the rule"
            ),
            violation=F.col("mno").isin(sorted(LOW_BOUNDARY_MNOS))
            & (F.col("start_confidence") == F.lit("HIGH"))
            & ~F.col("lifecycle_infer_type").startswith(F.lit(IDV_START_TOKEN)),
        ),
        Check(
            name="consistency_low_boundary_carrier_closed_end_is_low",
            dimension="Consistency",
            severity=ABORT,
            checked_field="mno, lifecycle_is_open, end_confidence, tu_enriched",
            description=(
                "the same carriers cannot produce a HIGH end on a *closed* "
                "lifecycle on their own evidence, because the event that closed "
                "it is ambiguous between a cancellation and a suspension. An "
                "open lifecycle is deliberately exempt: where no such event was "
                "sent there is nothing to be ambiguous about, and the absence of "
                "a cancellation is itself evidence the number is still live. A "
                "lifecycle on a phone number a second feed answered about is "
                "exempt too, and says so on the row: the ambiguity is settled by "
                "evidence from outside the carrier, and the timestamp the "
                "carrier reported is kept exactly as it was reported"
            ),
            violation=F.col("mno").isin(sorted(LOW_BOUNDARY_MNOS))
            & (F.col("lifecycle_is_open") == F.lit(0))
            & (F.col("end_confidence") == F.lit("HIGH"))
            & (F.coalesce(F.col("tu_enriched"), F.lit(0)) == F.lit(0)),
        ),
        *[
            _share(
                f"info_low_confidence_share_{carrier.lower()}",
                (F.col("mno") == F.lit(carrier))
                & (F.col("confidence") == F.lit("LOW")),
                f"{carrier} lifecycles whose boundaries are not both firm",
            )
            for carrier in MNO_DOMAIN
        ],
    ]


# ==========================================================================
# msisdn_lifecycle_account_mapping
# ==========================================================================


def account_mapping_checks(lifecycles: DataFrame | None = None) -> list[Check]:
    """``lifecycles`` is this run's lifecycle output, for the referential check.

    It is passed in rather than read back from the table because the question is
    whether the *run* is internally consistent. Reading the table would test the
    merge as well and blur which of the two went wrong.
    """
    checks = [
        not_null("acct_id", severity=ABORT),
        not_null("lifecycle_uid", severity=ABORT),
        matches_when_present(
            "acct_id", HEX64_REGEX, name="validity_acct_id_format", severity=ABORT
        ),
        unique_by(
            ["acct_id", "lifecycle_uid"],
            name="uniqueness_account_mapping_key",
            severity=ABORT,
            description="the same lifecycle appears twice in one account",
        ),
        unique(
            "lifecycle_uid",
            severity=ABORT,
            description=(
                "a lifecycle belongs to exactly one account. Two rows means the "
                "chain traversal placed it in two chains, which a path graph "
                "cannot do and a conflicting edge can"
            ),
        ),
        *_slice_key_checks("account_mapping"),
        not_null("from_ts", severity=ABORT),
        _interval_ordered(
            "from_ts",
            "to_ts",
            name="consistency_account_interval_ordered",
            description="account segment ends before it starts",
        ),
        # A warning, not an abort, and the difference is the whole point of the
        # partition. On a phone number the same check is an abort, because a
        # number is with one carrier at a time and two live lifecycles means
        # every point-in-time question about it has two answers. An account is
        # not a phone number. It is a subscriber, and a subscriber holding two
        # numbers at once is what a number change looks like from the inside: the
        # new number is live before the old one is released.
        #
        # The pipeline does not merely tolerate that, it measures it. The SIM
        # route carries ``overlap_days`` precisely because the two numbers were
        # both live on one SIM, and
        # :func:`~trust_score_05.lineage.transforms.accounts.link_confidence_for`
        # marks the link down when the overlap runs past the tolerance rather
        # than refusing the link. Aborting the table for the same overlap the
        # link confidence has already graded is the pipeline contradicting
        # itself, and it did: the first full account rebuild refused to publish
        # 842,918 of 77,172,342 segments against 971,371 SIM links already
        # published as LOW for that exact reason.
        #
        # Left as a warning it still says something worth hearing. A share that
        # climbs is either the carrier feeds drifting apart or the SIM route
        # pairing two subscriptions rather than one renumbering, and both are
        # worth a look - neither is worth throwing the rebuild away over.
        _overlap_check(
            partition=["acct_id"],
            from_col="from_ts",
            to_col="to_ts",
            order_col="lifecycle_uid",
            name="consistency_account_segments_no_overlap",
            severity=WARN,
            description=(
                "two lifecycles of one account were live at the same moment. A "
                "number change routinely produces this - the new number opens "
                "before the old one is released - and the SIM route measures the "
                "overlap in overlap_days and grades the link on it. Read it "
                "alongside info_low_confidence_account_link_share rather than on "
                "its own"
            ),
        ),
        # The account mapping publishes two values that the customer mapping
        # never does, so the two domains are extended here rather than in
        # :mod:`~trust_score_05.lineage.schemas`. ``truncated_fragment`` and
        # ``final_truncated`` exist only where ``windows.max_chain_depth`` cut a
        # chain in half, and only the account stage walks a chain deep enough to
        # be cut - the customer stage chains accounts, of which the longest real
        # component is a handful.
        #
        # They were added because of what the August 2026 audit found on the
        # 24 August rebuild: the 1,596 lifecycles hanging off the far side of the
        # 38 depth-cap cuts published as ``segment_type = 'single'``,
        # ``account_type = 'single'``, and every one of those was a lifecycle the
        # traversal knew to be mid-chain. They passed this very check, because
        # 'single' is in the domain. A vocabulary that has no word for "we did
        # not finish resolving this" forces the code to pick a word that is
        # wrong, and every check downstream then agrees with it.
        in_domain(
            "segment_type",
            (*SEGMENT_TYPE_DOMAIN, *TRUNCATED_SEGMENT_TYPES),
            severity=WARN,
        ),
        in_domain(
            "account_type",
            (*ACCOUNT_TYPE_DOMAIN, *TRUNCATED_ACCOUNT_TYPES),
            severity=WARN,
        ),
        # How much of the table the depth cap gave up on. Zero on every slice
        # small enough that no chain reaches ``windows.max_chain_depth``, which
        # is every nightly run; 1,596 of 77,172,342 on the 24 August 2026 full
        # rebuild, a share small enough that nobody would ever have gone looking
        # for it and large enough to matter to the 38 accounts it dismembered.
        #
        # INFO rather than WARN on purpose. A truncated fragment is not a defect
        # in the row - the row is as honest as it can be about what is known
        # about it - it is a measure of how often the cap is being hit, and the
        # question it should prompt is whether ``max_chain_depth`` is set for the
        # data or the data has a component that is not a subscriber at all.
        _share(
            "info_truncated_fragment_share",
            F.col("segment_type") == F.lit(TRUNCATED_FRAGMENT),
            "segments past a chain the depth cap stopped resolving. Their "
            "position in their chain, and which account they belong to, are "
            "unknown rather than absent",
        ),
        Check(
            name="consistency_single_segment_agrees_with_single_account",
            dimension="Consistency",
            severity=WARN,
            checked_field="segment_type, account_type",
            description=(
                "segment_type = 'single' and account_type = 'single' must agree. "
                "The previous code had a branch that could emit 'single' inside a "
                "chain; it was dead code and this is the check that keeps it dead"
            ),
            violation=(F.col("segment_type") == F.lit("single"))
            != (F.col("account_type") == F.lit("single")),
        ),
        Check(
            name="consistency_seq_order_zero_based",
            dimension="Validity",
            severity=WARN,
            checked_field="seq_order",
            description="seq_order is a 0-based position along the chain",
            violation=F.col("seq_order") < F.lit(0),
        ),
        Check(
            name="completeness_source_event_id_bounded",
            dimension="Completeness",
            severity=WARN,
            checked_field="source_event_id",
            description=(
                "source_event_id holds the events that decided this segment's "
                "boundaries - at most two. The previous code put every event in "
                "the lifecycle's window here, seventeen in one sampled row, which "
                "made the column useless for tracing"
            ),
            violation=F.size(F.split(F.col("source_event_id"), ",")) > F.lit(2),
        ),
        _share(
            "info_chained_account_share",
            F.col("account_type") == F.lit("chain"),
            "segments belonging to a multi-lifecycle account - a subscriber who "
            "changed their phone number at least once",
        ),
        _share(
            "info_origin_unmatched_share",
            F.col("segment_type") == F.lit("origin_unmatched"),
            "chains whose first link has an inferred start, so the chain probably "
            "runs back further than the evidence does",
        ),
        in_domain(
            "link_route",
            LINK_ROUTE_DOMAIN,
            severity=WARN,
            description=(
                "link_route names the evidence that joined this lifecycle to the "
                "one before it. A new value means a new route was added to the "
                "account stage and this vocabulary was not told about it"
            ),
        ),
        *_confidence_columns(
            "account_mapping", ("link_confidence", "acct_link_confidence_min")
        ),
        Check(
            name="consistency_chain_head_has_no_route",
            dimension="Consistency",
            severity=ABORT,
            checked_field="seq_order, link_route",
            description=(
                "the first lifecycle in an account has no incoming link, and "
                "every later one does. 'none' anywhere but position zero means a "
                "row lost its edge in the join; a real route at position zero "
                "means the chain starts somewhere the traversal did not say"
            ),
            violation=(F.col("seq_order") == F.lit(0))
            != (F.col("link_route") == F.lit("none")),
        ),
        Check(
            name="consistency_chain_head_link_is_high",
            dimension="Consistency",
            severity=ABORT,
            checked_field="link_route, link_confidence",
            description=(
                "a chain head makes no claim that two lifecycles are one account, "
                "so there is nothing there to be wrong about and it is HIGH. A "
                "LOW head is a coalesce that fired on the wrong side of a join"
            ),
            violation=(F.col("link_route") == F.lit("none"))
            & (F.col("link_confidence") != F.lit("HIGH")),
        ),
        _rollup_check(
            "consistency_acct_rollup_not_better_than_its_links",
            link_col="link_confidence",
            rollup_col="acct_link_confidence_min",
            group_col="acct_id",
            description=(
                "an account claims acct_link_confidence_min = HIGH while holding "
                "a LOW link. The rollup is a minimum over the chain, so this is "
                "the arithmetic failing, most likely a lost join row"
            ),
        ),
        _share(
            "info_low_confidence_account_link_share",
            F.col("link_confidence") == F.lit("LOW"),
            "lifecycles joined into an account on evidence worth confirming - "
            "today that means a SIM overlap wider than the tolerance",
        ),
        _share(
            "info_low_confidence_account_share",
            F.col("acct_link_confidence_min") == F.lit("LOW"),
            "segments of an account whose chain has at least one soft link, was "
            "cut at the depth cap, or was repaired around a cycle",
        ),
    ]
    if lifecycles is not None:
        checks.append(
            references(
                "lifecycle_uid",
                lifecycles,
                "lifecycle_uid",
                name="referential_account_lifecycle_exists",
                severity=ABORT,
                description=(
                    "the account mapping names a lifecycle this run did not "
                    "build - the two stages disagree about what exists"
                ),
            )
        )
    return checks


# ==========================================================================
# msisdn_lifecycle_account_customer_mapping
# ==========================================================================


def customer_mapping_checks(accounts: DataFrame | None = None) -> list[Check]:
    checks = [
        not_null("customer_id", severity=ABORT),
        not_null("acct_id", severity=ABORT),
        matches_when_present(
            "customer_id", HEX64_REGEX, name="validity_customer_id_format", severity=ABORT
        ),
        unique_by(
            ["customer_id", "acct_id", "lifecycle_uid"],
            name="uniqueness_customer_mapping_key",
            severity=ABORT,
        ),
        unique(
            "lifecycle_uid",
            severity=ABORT,
            description="a lifecycle belongs to exactly one customer",
        ),
        *_slice_key_checks("customer_mapping"),
        not_null("from_ts", severity=ABORT),
        _interval_ordered(
            "from_ts",
            "to_ts",
            name="consistency_customer_interval_ordered",
            description=(
                "customer segment ends before it starts. The likely cause is the "
                "port clip: an outgoing port timestamped before the account "
                "segment it is clipping began"
            ),
        ),
        # A warning for the reason the account one is, and more so. A customer
        # holds accounts, an account holds phone numbers, and a customer running
        # two numbers at once is not an anomaly at all - it is a person with a
        # work phone. Everything the account check tolerates arrives here as
        # well, on top of that.
        _overlap_check(
            partition=["customer_id"],
            from_col="from_ts",
            to_col="to_ts",
            order_col="lifecycle_uid",
            name="consistency_customer_segments_no_overlap",
            severity=WARN,
            description=(
                "two lifecycles of one customer were live at the same moment. A "
                "customer may hold several numbers at once, so this counts how "
                "often it happens rather than forbidding it"
            ),
        ),
        # Extended for this table's *own* truncation, not the account stage's.
        # ``assemble_customers`` recomputes both columns from scratch at account
        # grain - the ``withColumn`` calls overwrite whatever the account
        # mapping carried in on the join - so an account-level
        # 'truncated_fragment' can never reach this check. What can reach it is
        # a customer-level one: the customer stage runs the same depth-capped
        # walk over port edges, and an account stranded past a cut in *that*
        # walk is labelled with the same two words.
        #
        # Latent rather than live. ``max_chain_depth`` is 50 and the largest
        # port component ever observed is 38, so these tokens appear on no row
        # published to date. The domain is widened by exactly the two the
        # transform can emit and no further: a check that tolerates a word the
        # code cannot produce is a check that has stopped describing the code.
        in_domain(
            "segment_type",
            (*SEGMENT_TYPE_DOMAIN, *TRUNCATED_SEGMENT_TYPES),
            severity=WARN,
        ),
        in_domain(
            "account_type",
            (*ACCOUNT_TYPE_DOMAIN, *TRUNCATED_ACCOUNT_TYPES),
            severity=WARN,
        ),
        in_domain(
            "updated_source",
            UPDATED_SOURCE_MAPPING_DOMAIN,
            severity=WARN,
            description=(
                "only 'enstream' is written today. TransUnion and Port PS are out "
                "of scope for this version"
            ),
        ),
        _share(
            "info_multi_account_customer_share",
            F.col("account_type") == F.lit("chain"),
            "segments belonging to a customer with more than one account - a "
            "subscriber who ported between carriers",
        ),
        *_confidence_columns(
            "customer_mapping", ("link_confidence", "customer_link_confidence_min")
        ),
        Check(
            name="consistency_customer_head_link_is_high",
            dimension="Consistency",
            severity=ABORT,
            checked_field="seq_order, link_confidence",
            description=(
                "the first account in a customer was not ported into, so it makes "
                "no claim of sameness and cannot be soft. Only the forward "
                "direction is asserted - a later account may perfectly well be "
                "HIGH, and usually is"
            ),
            violation=(F.col("seq_order") == F.lit(0))
            & (F.col("link_confidence") != F.lit("HIGH")),
        ),
        _rollup_check(
            "consistency_customer_rollup_not_better_than_its_links",
            link_col="link_confidence",
            rollup_col="customer_link_confidence_min",
            group_col="customer_id",
            description=(
                "a customer claims customer_link_confidence_min = HIGH while "
                "holding a LOW port link"
            ),
        ),
        _share(
            "info_low_confidence_port_link_share",
            F.col("link_confidence") == F.lit("LOW"),
            "accounts joined to the one before them on a loose handover - the "
            "number may have sat in quarantine and been reassigned rather than "
            "ported, so these are the identities worth confirming against TU",
        ),
        _share(
            "info_low_confidence_customer_share",
            F.col("customer_link_confidence_min") == F.lit("LOW"),
            "segments of a customer whose chain has at least one loose handover, "
            "was cut at the depth cap, or was repaired around a cycle",
        ),
    ]
    if accounts is not None:
        checks.append(
            references(
                "acct_id",
                accounts,
                "acct_id",
                name="referential_customer_account_exists",
                severity=ABORT,
                description=(
                    "the customer mapping names an account this run did not build"
                ),
            )
        )
    return checks


# ==========================================================================
# normalized_canonical_events
# ==========================================================================


def normalized_events_checks(accounts: DataFrame | None = None) -> list[Check]:
    checks = [
        not_null("event_id", severity=ABORT),
        unique("event_id", severity=ABORT, description="two normalized rows share an event id"),
        # This table is merged and delete-scoped by the slice key like every
        # other Gold table, so it gets the same guard. It is easy to think
        # otherwise here, because ``acct_id`` and ``customer_id`` are legitimately
        # nullable on this table - an event that fell inside no account interval
        # is kept rather than dropped - but that is about the *links*, not about
        # the key the merge is scoped by.
        *_slice_key_checks("normalized_events"),
        not_null("event_ts", severity=ABORT),
        not_null("event_family", severity=WARN),
        in_domain("updated_source", UPDATED_SOURCE_NORMALIZED_DOMAIN, severity=WARN),
        Check(
            name="validity_event_weight_non_negative",
            dimension="Validity",
            severity=WARN,
            checked_field="event_weight",
            description="event_weight is a scoring weight and cannot be negative",
            violation=F.col("event_weight").isNull() | (F.col("event_weight") < F.lit(0)),
        ),
        Check(
            name="consistency_event_family_known",
            dimension="Consistency",
            severity=WARN,
            checked_field="event_family",
            description=(
                "event_family fell through to 'other', which means an event type "
                "was added without adding it to EVENT_FAMILY_BY_TYPE"
            ),
            violation=F.col("event_family") == F.lit("other"),
        ),
        _share(
            "info_unlinked_event_share",
            F.col("acct_id").isNull(),
            "events that fell inside no account interval. They are kept rather "
            "than dropped - the event happened, and a table that silently loses "
            "events is worse than one that admits it could not place them",
        ),
        _share(
            "info_derived_normalized_share",
            F.col("updated_source") == F.lit("enstream_event_derived"),
            "events Gold inferred rather than a carrier sending them",
        ),
    ]
    if accounts is not None:
        checks.append(
            references(
                "acct_id",
                accounts,
                "acct_id",
                name="referential_normalized_account_exists",
                severity=ABORT,
                description=(
                    "a normalized event points at an account this run did not build"
                ),
            )
        )
    return checks


# ==========================================================================
# msisdn_imei_map
# ==========================================================================


def imei_map_checks() -> list[Check]:
    return [
        not_null("imei", severity=ABORT, description="device segment with no device"),
        *_slice_key_checks("imei_map"),
        not_null("from_ts", severity=ABORT),
        unique_by(
            [SLICE_KEY, "imei", "from_ts"],
            name="uniqueness_imei_map_key",
            severity=ABORT,
            description="two segments claim the same phone number, device and start",
        ),
        _interval_ordered(
            "from_ts",
            "to_ts",
            name="consistency_imei_interval_ordered",
            description="device segment ends before it starts",
        ),
        _overlap_check(
            partition=[SLICE_KEY],
            from_col="from_ts",
            to_col="to_ts",
            order_col="imei",
            name="consistency_one_device_at_a_time",
            description=(
                "a phone number is attached to two devices at the same moment. "
                "This is the cardinality rule the business confirmed and the whole "
                "walk depends on it: one device at a time, many phone numbers per "
                "device"
            ),
        ),
        _at_most_one_open(
            partition=[SLICE_KEY],
            to_col="to_ts",
            name="consistency_one_open_segment_per_number",
            description=(
                "a phone number has two device segments with no end, so it is "
                "recorded as being on two handsets now. The displacing segment "
                "did not close the one it displaced"
            ),
        ),
        in_domain("mno", MNO_DOMAIN, severity=WARN),
        Check(
            name="consistency_event_date_matches_from_ts",
            dimension="Consistency",
            severity=ABORT,
            checked_field="event_date",
            description=(
                "event_date is the partition and is derived from from_ts. A "
                "mismatch puts the row in a partition no slice-scoped read will "
                "look in"
            ),
            violation=F.col("event_date") != F.to_date(F.col("from_ts")),
        ),
        Check(
            name="consistency_oldimei_differs",
            dimension="Consistency",
            severity=WARN,
            checked_field="oldIMEI",
            description=(
                "oldIMEI is the device this segment displaced, so it cannot be "
                "the same device - that would be a segment that displaced itself"
            ),
            violation=F.col("oldIMEI").isNotNull() & (F.col("oldIMEI") == F.col("imei")),
        ),
        _share(
            "info_open_segment_share",
            F.col("to_ts").isNull(),
            "device segments still open - the phone number is on that handset now",
        ),
        _share(
            "info_first_segment_share",
            F.col("oldIMEI").isNull(),
            "segments that displaced nothing, so the first device we ever saw for "
            "that phone number",
        ),
    ]


# ==========================================================================
# what the run itself did
# ==========================================================================


def merge_stat_checks(stats, prefix: str) -> list[Check]:
    """Turn one table's :class:`~..io.merge_sink.MergeStats` into ``info`` metrics.

    A merge is the only part of the run whose effect is invisible in the output
    frame - the frame says what Gold should contain, not what changed to get
    there. Recording inserts, updates and deletes per table is what makes "this
    run deleted eleven thousand rows" a question somebody can ask afterwards
    instead of a surprise found weeks later.
    """
    return [
        counter(
            f"info_{prefix}_rows_inserted",
            stats.inserted,
            dimension="Lineage",
            description=f"rows inserted into {stats.table}",
        ),
        counter(
            f"info_{prefix}_rows_updated",
            stats.updated,
            dimension="Lineage",
            description=f"rows updated in {stats.table}",
        ),
        counter(
            f"info_{prefix}_rows_deleted",
            stats.deleted,
            dimension="Lineage",
            description=(
                f"rows deleted from {stats.table}. Legitimate deletions are a "
                "trickle - a spurious lifecycle dissolved by a correction, a row "
                "whose key migrated. A flood means a recomputation input was empty"
            ),
        ),
    ]
