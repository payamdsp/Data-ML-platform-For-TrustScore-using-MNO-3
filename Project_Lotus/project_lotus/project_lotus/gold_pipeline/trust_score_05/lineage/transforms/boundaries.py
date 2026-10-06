"""Stage 2b: boundaries the carrier feeds guessed, read off the enrichment feeds.

Section 7 of the enrichment design is emphatic that enrichment adds no confidence
rule. It adds evidence to rules that already exist, and the existing rules
produce a higher value from it. This module is that sentence in code: two joins
onto the lifecycle work frame, each supplying one piece of evidence the carrier
feeds could not, and the same ``HIGH``/``LOW`` arithmetic stage 2 already runs.

The two cases are not symmetric, and the asymmetry is the whole design.

**A start that was a guess is replaced.** A ROGERS lifecycle drawn from an
``IMSI_START`` starts when the SIM appeared, which may be days after the account
opened. There is nothing in that timestamp worth preserving, so an observed
``activationDate`` replaces it outright and the start stops being inferred.

**An end that was ambiguous is only confirmed.** A ROGERS ``STATUS_CHANGE-C``
arrives at an observed instant; what is unclear is whether it means a
cancellation or a suspension. TU's port out of that carrier settles the meaning
and says nothing better about the instant, so the timestamp is left alone and
only ``end_confidence`` moves. Taking TU's port timestamp instead would open a
gap where stage 4 currently reads a zero-length handover, and would be trading a
boundary our own lineage is built from for one that is not more accurate.

This stage runs **before** the TU-derived lifecycles of
:mod:`~.tu_lifecycle`, because that stage clips TU's carrier stays against the
intervals these lifecycles occupy. Firming afterwards would clip against
boundaries this stage is about to move.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from ..logging_utils import get_logger
from ..schemas import (
    IDV_SUCCESS_RESPONSE_CODE,
    LIFECYCLE_END_TOKENS,
    LIFECYCLE_FLOOR_START_TOKEN,
    SLICE_KEY,
)
from .customers import tu_latest_response
from .lifecycle import HIGH, LIFECYCLE_WORK_SCHEMA, LOW

__all__ = [
    "BoundaryReport",
    "IDV_START_TOKEN",
    "FLOOR_START_TOKEN",
    "latest_activation",
    "firm_starts_from_activation",
    "firm_ends_from_tu",
    "firm_boundaries",
]

LOGGER = get_logger(__name__)

#: The start token a lifecycle carries once IDV has supplied its start. It
#: replaces whichever of :data:`~.lifecycle.START_TOKENS` the walk chose, so that
#: ``lifecycle_infer_type`` keeps its one job - saying where this lifecycle's
#: boundaries came from - without a second column to say that IDV was involved.
IDV_START_TOKEN = "observed_start_from_idv_activation_date"

#: The token the walk writes on a lifecycle it knows ended and never saw begin,
#: on a phone number with no earlier history to carry the start back to. Aliased
#: here under the name this stage reads it by, because this is one of the two
#: stages that treat such a start differently from every other inferred one -
#: see ``_floored`` below.
FLOOR_START_TOKEN = LIFECYCLE_FLOOR_START_TOKEN

#: Every end token any stage can publish, not just the four the stage-2 walk
#: writes. :data:`~.lifecycle.END_TOKENS` is the walk's own subset and would
#: leave ``_to_explicit_tu_port_out`` - written by :mod:`~.tu_lifecycle` - out;
#: :data:`~..schemas.LIFECYCLE_END_TOKENS` is the published vocabulary and is the
#: right thing for a parser of ``lifecycle_infer_type`` to match against. See
#: :func:`_end_token_of` for why the difference used to matter.
_END_TOKEN_VALUES = tuple(LIFECYCLE_END_TOKENS)


@dataclass
class BoundaryReport:
    """What the evidence changed, and what it contradicted.

    ``start_conflicts`` is the number worth watching. It counts phone numbers
    where IDV named an activation date our own lifecycle cannot accommodate -
    later than the first activity we saw, or earlier than the end of the previous
    lifecycle - which is IDV and a carrier feed disagreeing about when the
    current subscriber arrived. The stage applies neither reading and counts the
    disagreement, because there is no third source here that could break the tie.
    """

    phones_with_activation: int = 0
    starts_firmed: int = 0
    starts_moved: int = 0
    start_conflicts: int = 0
    ends_firmed: int = 0

    def format_line(self) -> str:
        return (
            f"boundary evidence: activations={self.phones_with_activation} "
            f"starts_firmed={self.starts_firmed} (moved={self.starts_moved}) "
            f"start_conflicts={self.start_conflicts} "
            f"ends_firmed={self.ends_firmed}"
        )


#: What a ``lifecycle_infer_type`` this module cannot parse costs the row, said
#: out loud. See :func:`_end_token_of`.
_UNKNOWN_END_TOKEN_ERROR = (
    "boundaries._end_token_of: lifecycle_infer_type ends in none of the known "
    "end tokens, so the start half cannot be replaced without inventing one"
)


def _end_token_of(infer_type):
    """The end half of a ``lifecycle_infer_type``, so the start half can change.

    ``lifecycle_infer_type`` is one string built as start token + end token, and
    the end tokens are a closed set that all begin with ``_``. Matching against
    that set is safer than splitting on the underscore: two of the start tokens
    contain underscores of their own, and a split would cut one of them in half.

    The set matched is :data:`~..schemas.LIFECYCLE_END_TOKENS`, the vocabulary
    the whole estate publishes. This module keeps **no** copy of its own; it
    reads the tuple at import (see ``_END_TOKEN_VALUES`` above), so a token added
    to the vocabulary is a token this function can already find.

    That used to be :data:`~.lifecycle.END_TOKENS`, which is only the four the
    stage-2 walk can write, and it left out ``_to_explicit_tu_port_out`` from
    :data:`~.tu_lifecycle.TU_END_TOKEN`. **That gap was survivable only because
    of call ordering:** ``lineage_job.py:281-286`` runs :func:`firm_boundaries`
    before :func:`~.tu_lifecycle.build_tu_lifecycles`, so no TU-ended lifecycle
    has ever reached here. Nothing in the code enforces that ordering, and the
    failure it was hiding is an ABORT-severity ``raise_error`` in the middle of a
    production job, on rows the stage has just declared firm. Widening the set
    fixes it at the source and costs one extra ``endswith``; a loud import-time
    assertion on the ordering would only convert a wrong answer into a different
    refusal, and would still be wrong on the day the ordering legitimately
    changes. So the ordering stays what it is - for the reason this module's own
    header gives, that firming after the TU stage would clip against boundaries
    this stage is about to move - but it is no longer load-bearing *here*.

    An unmatched value still raises, and now means what it says: a
    ``lifecycle_infer_type`` outside the published vocabulary. Falling back to an
    empty string would publish an ``observed_start_from_idv_activation_date``
    with no end token at all - half a value, with nothing anywhere to say a token
    went missing, since the DQ vocabulary check is a warning and would not stop
    the run.
    """
    out = F.raise_error(F.lit(_UNKNOWN_END_TOKEN_ERROR)).cast("string")
    # Built in reverse: each round wraps the previous expression in ``.otherwise``,
    # so the LAST element of ``_END_TOKEN_VALUES`` is the first branch tested.
    #
    # That is immaterial *only* because the end tokens are pairwise non-suffixes -
    # no token ends with another - so at most one ``endswith`` can ever match and
    # the order the branches are tried in cannot change the answer. Anyone adding a
    # token has to preserve that property (``test_schemas.py`` is where to assert
    # it); a token that is a suffix of another would silently resolve to whichever
    # of the two sits later in the tuple.
    for token in _END_TOKEN_VALUES:
        out = F.when(infer_type.endswith(F.lit(token)), F.lit(token)).otherwise(out)
    return out


# ==========================================================================
# starts, from mno_activation
# ==========================================================================


def latest_activation(activations: DataFrame) -> DataFrame:
    """The newest usable ``activationDate`` answer per phone number.

    Usable means the call succeeded *and* returned a date. The two are separate
    conditions because they fail separately: a ``NOT_FOUND`` is an answer with no
    date, and a success with a null date is a provider gap. Neither is evidence,
    and both would otherwise arrive here as a null that silently loses every
    comparison.

    Newest by ``api_ts``, with ``request_id`` breaking a tie, for the reason
    every tiebreak in this pipeline exists: without it two calls landing in the
    same second would both survive, the rebuild would keep whichever the shuffle
    handed it first, and a table rebuilt from scratch every run would stop
    converging.
    """
    newest = F.max(F.struct(F.col("api_ts"), F.col("request_id"))).over(
        Window.partitionBy(SLICE_KEY)
    )
    return (
        activations.where(
            (F.col("response_code") == F.lit(IDV_SUCCESS_RESPONSE_CODE))
            & F.col("activation_date").isNotNull()
        )
        .withColumn("_newest", newest)
        .where(
            (F.col("api_ts") == F.col("_newest.api_ts"))
            & (F.col("request_id") == F.col("_newest.request_id"))
        )
        .select(
            F.col(SLICE_KEY),
            F.col("api_ts").alias("_act_api_ts"),
            F.upper(F.col("mno")).alias("_act_mno"),
            F.col("activation_date").alias("_act_date"),
            F.col("activation_date").cast("timestamp").alias("_act_ts"),
        )
    )


def firm_starts_from_activation(
    lifecycles: DataFrame, activations: DataFrame | None
) -> tuple[DataFrame, dict]:
    """Replace guessed starts with the activation date IDV observed.

    ``activationDate`` says when the number was activated **to its current
    subscriber**, and current means current as of the call. So the lifecycle it
    describes is the one that was live at ``api_ts`` - not the earliest, not the
    latest, and not whichever one the date happens to land in. A number cancelled
    before IDV was asked has no lifecycle the answer belongs to, and gets none.

    Three further conditions, each closing off a way of being wrong:

    * the answer's carrier, when it names one, has to be the lifecycle's carrier;
    * an activation *after* the first activity we saw is a contradiction, not a
      correction, and is counted rather than applied - unless the start is the
      floor, which is not an activity we saw and which any real date improves
      on;
    * a start moved back past the previous lifecycle's end would overlap it, and
      overlapping lifecycles are the one thing the uniqueness checks say cannot
      happen. That case is refused whole rather than clamped: a start clamped to
      the previous end is not the activation date and not the original guess, and
      nothing downstream could tell it from either.

    An **inferred** start is replaced by the activation date. It is taken as
    midnight, because the API returns ``MM/DD/YYYY`` with no time of day, and the
    consequence is stated in the schema and in the resolution design: a fraud at
    09:00 on the activation date itself falls on the later side of that boundary.

    An **observed** start is confirmed but never moved. When our feed recorded an
    actual activation or port-in event, that event is the better witness to the
    time of day, so IDV only has to agree with it - same calendar day in EST -
    for the start to stop being low.
    """
    if activations is None:
        return lifecycles, {"phones": 0, "firmed": 0, "moved": 0, "conflicts": 0}

    latest = latest_activation(activations)
    phones = latest.count()
    if phones == 0:
        return lifecycles, {"phones": 0, "firmed": 0, "moved": 0, "conflicts": 0}

    # The previous lifecycle's end, in start order. Lifecycles for one phone
    # number never overlap, so the one immediately before has the latest end of
    # everything before it and a single ``lag`` is the whole floor.
    order = Window.partitionBy(SLICE_KEY).orderBy(
        F.col("lifecycle_start_ts").asc(), F.col("lifecycle_uid").asc()
    )
    joined = (
        lifecycles.withColumn("_prev_end", F.lag("lifecycle_end_ts").over(order))
        .join(latest, SLICE_KEY, "left")
        .withColumn(
            "_live_at_call",
            F.col("_act_ts").isNotNull()
            & (F.col("lifecycle_start_ts") <= F.col("_act_api_ts"))
            & (
                F.col("lifecycle_end_ts").isNull()
                | (F.col("_act_api_ts") < F.col("lifecycle_end_ts"))
            ),
        )
        .withColumn(
            "_carrier_agrees",
            F.col("_act_mno").isNull() | (F.col("_act_mno") == F.col("mno")),
        )
        .withColumn(
            "_eligible",
            F.col("_live_at_call")
            & F.col("_carrier_agrees")
            & (F.col("start_confidence") == F.lit(LOW)),
        )
        .withColumn(
            "_room",
            F.col("_prev_end").isNull() | (F.col("_act_ts") >= F.col("_prev_end")),
        )
        .withColumn(
            "_same_day",
            F.to_date(F.col("lifecycle_start_ts")) == F.col("_act_date"),
        )
        # A floored start is the one inferred start an activation date is
        # allowed to move *forward*. Everywhere else, an activation later than
        # the start contradicts the first activity we saw and is counted rather
        # than applied - but a floor is not an activity we saw, it is the
        # absence of one, and it sits below every real date by construction.
        # Refusing to move it would make every such lifecycle a permanent
        # conflict and leave the floor in place with evidence sitting next to it
        # saying what the start was.
        .withColumn(
            "_floored",
            F.col("lifecycle_infer_type").startswith(F.lit(FLOOR_START_TOKEN)),
        )
        .withColumn(
            "_moves",
            F.col("_eligible")
            & (F.col("is_start_inferred") == F.lit(1))
            & F.col("_room")
            & (
                (F.col("_act_ts") <= F.col("lifecycle_start_ts"))
                | (
                    F.col("_floored")
                    & (
                        F.col("lifecycle_end_ts").isNull()
                        | (F.col("_act_ts") < F.col("lifecycle_end_ts"))
                    )
                )
            ),
        )
        .withColumn(
            "_confirms",
            F.col("_eligible")
            & (F.col("is_start_inferred") == F.lit(0))
            & F.col("_same_day"),
        )
        .withColumn(
            "_conflicts",
            F.col("_eligible") & ~F.col("_moves") & ~F.col("_confirms"),
        )
    ).persist()

    counts = joined.agg(
        F.sum(F.col("_moves").cast("int")).alias("moved"),
        F.sum(F.col("_confirms").cast("int")).alias("confirmed"),
        F.sum(F.col("_conflicts").cast("int")).alias("conflicts"),
    ).collect()[0]

    firmed = F.col("_moves") | F.col("_confirms")
    out = (
        joined.withColumn(
            "lifecycle_start_ts",
            F.when(F.col("_moves"), F.col("_act_ts")).otherwise(
                F.col("lifecycle_start_ts")
            ),
        )
        .withColumn(
            "is_start_inferred",
            F.when(firmed, F.lit(0)).otherwise(F.col("is_start_inferred")),
        )
        .withColumn(
            "start_confidence",
            F.when(firmed, F.lit(HIGH)).otherwise(F.col("start_confidence")),
        )
        .withColumn(
            "lifecycle_infer_type",
            F.when(
                firmed,
                F.concat(
                    F.lit(IDV_START_TOKEN),
                    _end_token_of(F.col("lifecycle_infer_type")),
                ),
            ).otherwise(F.col("lifecycle_infer_type")),
        )
        .select(*[f.name for f in LIFECYCLE_WORK_SCHEMA.fields])
    )
    stats = {
        "phones": phones,
        "firmed": int(counts["moved"] or 0) + int(counts["confirmed"] or 0),
        "moved": int(counts["moved"] or 0),
        "conflicts": int(counts["conflicts"] or 0),
    }
    joined.unpersist()
    return out, stats


# ==========================================================================
# ends, from tu_portps
# ==========================================================================


def firm_ends_from_tu(
    lifecycles: DataFrame, tu_ports: DataFrame | None
) -> tuple[DataFrame, int]:
    """Settle an ambiguous end with the port TU witnessed.

    The target is narrow and named exactly: a lifecycle that our feed **closed on
    an observed event** - ``is_end_inferred = 0`` - whose ``end_confidence`` is
    nevertheless ``LOW``. Today that is only ROGERS, and only because
    ``STATUS_CHANGE-C`` is sent for a cancellation and for a suspension alike, so
    seeing one does not say the account ended. A TU port out of that carrier
    inside the lifecycle says it did.

    Ends that our own walk *inferred* are deliberately left alone. Their
    timestamps are guesses, and a port confirming that the number departed says
    nothing about whether the guessed instant is the right one; firming them
    would put ``HIGH`` on a boundary we made up.

    A port counts when it is an inter-carrier port away from this lifecycle's
    carrier and falls inside the lifecycle's own interval. Both halves matter: an
    intra-SPID port is one carrier's internal move and is not a departure, and a
    port outside the interval belongs to a different stretch of the number's
    life.

    "Inside" here is ``(start, end]`` and not the estate's usual ``[start, end)``,
    and the asymmetry is the point rather than an oversight - see the join below.
    """
    if tu_ports is None:
        return lifecycles, 0

    ports = (
        tu_latest_response(tu_ports)
        .where(F.col("is_inter_carrier_port") & F.col("from_mno_std").isNotNull())
        .select(
            F.col(SLICE_KEY).alias("_p_phone"),
            F.upper(F.col("from_mno_std")).alias("_p_from"),
            F.col("event_timestamp").alias("_p_ts"),
        )
    )

    candidate = (
        (F.col("lifecycle_is_open") == F.lit(0))
        & (F.col("is_end_inferred") == F.lit(0))
        & (F.col("end_confidence") == F.lit(LOW))
    )
    # Both bounds used to be inclusive, which the August 2026 audit's finding 12
    # caught: where two lifecycles at one carrier abut at instant ``T``, a port at
    # ``T`` matched both. It confirmed the earlier one's end, correctly, and it
    # also put ``end_confidence = HIGH`` on the later one - on evidence sitting at
    # that lifecycle's *start*, describing a boundary nothing had witnessed.
    #
    # The interval that fixes it is ``(start, end]``, not the ``[start, end)`` the
    # rest of the estate uses, and the difference is not carelessness. A lifecycle
    # occupies ``[start, end)``; what this join asks is a different question -
    # which lifecycle's *departure* could this port be - and a departure is
    # witnessed at the instant the interval closes, not at the last instant inside
    # it. A port at exactly ``end`` is TU and the carrier feed describing the same
    # handover, which is the two sources agreeing and the single commonest thing
    # this stage exists to record; excluding it would switch off the primary case.
    # A port at exactly ``start`` is the arrival, not the departure, and belongs
    # to the lifecycle before this one.
    #
    # ``lifecycle_end_ts`` cannot be null here - ``candidate`` above requires
    # ``lifecycle_is_open = 0`` - but the null branch is written anyway, for the
    # reason ``tu_lifecycle._stay_live_at`` gives for the same shape: the cost is
    # one comparison and the alternative is a silent wrong answer if that filter
    # is ever loosened. An open lifecycle's end reaches forward without limit, so
    # every port after its start falls inside it.
    matched = (
        lifecycles.where(candidate)
        .join(
            ports,
            (F.col(SLICE_KEY) == F.col("_p_phone"))
            & (F.col("mno") == F.col("_p_from"))
            & (F.col("_p_ts") > F.col("lifecycle_start_ts"))
            & (
                F.col("lifecycle_end_ts").isNull()
                | (F.col("_p_ts") <= F.col("lifecycle_end_ts"))
            ),
            "left_semi",
        )
        .select(F.col("lifecycle_uid").alias("_firm_uid"))
    ).persist()
    ends_firmed = matched.count()
    if ends_firmed == 0:
        matched.unpersist()
        return lifecycles, 0

    out = (
        lifecycles.join(
            matched, F.col("lifecycle_uid") == F.col("_firm_uid"), "left"
        )
        .withColumn(
            "end_confidence",
            F.when(F.col("_firm_uid").isNotNull(), F.lit(HIGH)).otherwise(
                F.col("end_confidence")
            ),
        )
        .select(*[f.name for f in LIFECYCLE_WORK_SCHEMA.fields])
    )
    matched.unpersist()
    return out, ends_firmed


# ==========================================================================
# the stage
# ==========================================================================


def firm_boundaries(
    lifecycles: DataFrame,
    activations: DataFrame | None = None,
    tu_ports: DataFrame | None = None,
) -> tuple[DataFrame, BoundaryReport]:
    """Run both firmings and recombine ``confidence``.

    ``None`` for either feed leaves that half untouched, which is the pipeline
    behaving exactly as it did before enrichment existed - the property that lets
    this stage be switched on without a migration.

    ``confidence`` is recomputed here rather than in each half, with the same AND
    stage 2 uses: a lifecycle is only as good as its weaker edge, and there is no
    half-answer to "was this number theirs on the 3rd of June". Recomputing in one
    place is what stops a start firmed by IDV and an end firmed by TU from
    disagreeing about which of them was supposed to combine the two.
    """
    out, start_stats = firm_starts_from_activation(lifecycles, activations)
    out, ends_firmed = firm_ends_from_tu(out, tu_ports)

    report = BoundaryReport(
        phones_with_activation=start_stats["phones"],
        starts_firmed=start_stats["firmed"],
        starts_moved=start_stats["moved"],
        start_conflicts=start_stats["conflicts"],
        ends_firmed=ends_firmed,
    )
    if start_stats["firmed"] or ends_firmed or start_stats["conflicts"]:
        out = out.withColumn(
            "confidence",
            F.when(
                (F.col("start_confidence") == F.lit(HIGH))
                & (F.col("end_confidence") == F.lit(HIGH)),
                F.lit(HIGH),
            ).otherwise(F.lit(LOW)),
        )
    LOGGER.info(report.format_line())
    return out, report
