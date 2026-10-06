"""Stage 4: accounts -> customers.

A customer is a chain of accounts joined by ports. The subscriber kept their
phone number and changed carrier, so the account they left and the account they
arrived at are one person.

The rule from the specification is deliberately narrow: **a customer is created
from porting events only.** No name, no address, no device fingerprint - just
the same phone number leaving one carrier and appearing at another. An account
with no port on either side is its own customer, and that is the common case.

The traversal is the account stage one level up, and it is the same code:
:func:`~.chains.build_chains` over accounts instead of lifecycles. What differs
is where the edges come from and how the intervals are clipped. A customer's
view of an account stops when the subscriber ported away, even if that account's
own rows run on past the port - the losing carrier's feed does not always stop
when the subscriber does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from ..logging_utils import get_logger
from ..progress import note
from ..schemas import CUSTOMER_MAPPING_SCHEMA, SLICE_KEY
from ..spark import shuffle_partitions
# The two words for a chain the depth cap cut. They are defined at the account
# stage because that is where the cap has always actually bitten, but they name
# a property of the traversal rather than of accounts, and the traversal is the
# same code here. Imported rather than restated so the two stages and the DQ
# vocabularies cannot drift apart - a drift would abort a check on a token the
# transform legitimately produces.
from .accounts import TRUNCATED_FINAL, TRUNCATED_FRAGMENT
from .canonical_events import port_gaps
from .chains import build_chains
from .common import (
    MAX_COLLECTED_EDGES,
    align_to_schema,
    collect_edge_rows,
    component_labels,
    describe_components,
    customer_id,
    days_between,
    windows_from_config,
    with_audit_columns,
)

__all__ = [
    "CustomerBuildReport",
    "build_port_edges",
    "build_customers",
    "count_quiet_port_candidates",
    "port_link_confidence_for",
    "to_gold",
    "tu_latest_response",
    "tu_port_edges",
    # the four steps `build_customers` is made of, exposed so a rebuild can run
    # them as four statements with the intermediate results staged on S3. See
    # :mod:`~trust_score_05.lineage.staging` for why that matters at full scale.
    "customer_port_edges",
    "label_port_components",
    "walk_port_components",
    "positions_from_walk",
    "assemble_customers",
]

LOGGER = get_logger(__name__)

#: The only value this pipeline writes to ``updated_source`` on the mapping
#: tables. TransUnion and Port PS are out of scope for this version.
UPDATED_SOURCE = "enstream"

_HIGH = "HIGH"
_LOW = "LOW"


def port_link_confidence_for(
    gap_days: float | None, high_confidence_days: float
) -> str:
    """How sound the claim is that two accounts are one subscriber.

    A customer link is a port and nothing else, so the only question left is how
    tight the handover was. A same-day handover is a port. Three weeks between
    one carrier losing the number and another gaining it is more likely the
    number sitting in quarantine and being reassigned, and calling that one
    person is a guess.

    The link is still made either way - ``mno_port_window_days`` decides
    *whether* to make it - but a loose one is made at ``LOW``, which is the flag
    that tells the resolver this is an identity worth confirming against TU's
    porting history.

    The comparison is strict, so a handover of exactly ``high_confidence_days``
    is ``LOW``. That is the opposite convention from the SIM-overlap tolerance in
    the account stage, and the difference is not an oversight in either place:
    the overlap branch is a tolerance and takes its boundary, this one is a
    threshold and does not.

    A null gap means the handover could not be measured, and an unmeasured link
    is not one to vouch for.
    """
    if gap_days is None:
        return _LOW
    return _HIGH if gap_days < high_confidence_days else _LOW


@dataclass
class CustomerBuildReport:
    """What the port traversal did."""

    account_count: int = 0
    customer_count: int = 0
    port_edge_count: int = 0
    quiet_port_candidates: int = 0
    broken_cycle_edges: list[tuple[Any, Any]] = field(default_factory=list)
    truncated_roots: list[Any] = field(default_factory=list)
    dropped_edges: list[tuple[Any, Any]] = field(default_factory=list)

    #: Whether the three distinct-counts above were actually taken.
    #:
    #: They are three full shuffles over every lifecycle in the estate and
    #: ``assemble_customers`` skips them when asked to, so their fields are left
    #: at the dataclass default. A default of zero rendered as ``0 account(s) ->
    #: 0 customer(s)`` on the 24 August 2026 rebuild, which reads as a run that
    #: built nothing rather than a run that was not asked to count - the table
    #: it wrote a minute later had 77 million rows in it. A number that was
    #: never measured has to say so.
    counts_taken: bool = False

    @property
    def clean(self) -> bool:
        return not (
            self.broken_cycle_edges or self.truncated_roots or self.dropped_edges
        )

    def format_line(self) -> str:
        if self.counts_taken:
            scale = (
                f"{self.account_count:,} account(s) -> "
                f"{self.customer_count:,} customer(s) over "
                f"{self.port_edge_count:,} port edge(s); "
                f"quiet_port_candidates={self.quiet_port_candidates:,}"
            )
        else:
            scale = (
                f"{self.port_edge_count:,} port edge(s); accounts, customers and "
                "quiet_port_candidates not counted (with_counts=False)"
            )
        return (
            f"customers: {scale} "
            f"cycles_broken={len(self.broken_cycle_edges)} "
            f"chains_truncated={len(self.truncated_roots)} "
            f"edges_dropped={len(self.dropped_edges)}"
        )


# ==========================================================================
# port edges
# ==========================================================================


def _lifecycles_with_accounts(lifecycles: DataFrame, accounts: DataFrame) -> DataFrame:
    """Lifecycle facts plus the account each lifecycle was placed in."""
    return lifecycles.select(
        F.col("lifecycle_uid"),
        F.col(SLICE_KEY),
        F.col("mno"),
        F.col("lifecycle_start_ts"),
        F.col("lifecycle_end_ts"),
        F.col("lifecycle_is_open"),
        F.col("is_start_inferred"),
        F.col("mno_port_ind"),
    ).join(accounts.select("lifecycle_uid", "acct_id"), "lifecycle_uid")


def build_port_edges(
    lifecycles: DataFrame, accounts: DataFrame, gaps: DataFrame | None = None
) -> DataFrame:
    """One edge per port: the account left, the account arrived at, the moment.

    A port shows up as a phone number whose next lifecycle is at a different
    carrier and whose opening event stage 1 already classified as
    ``account_activation_mno_port`` - which is what ``mno_port_ind`` records.

    The ten-day window is **not** re-tested here, and that is on purpose. Stage 1
    applied it against the raw event timestamps, where the losing carrier's last
    real activity is still visible. By the time the lifecycle walk has run, the
    losing lifecycle has been closed *at the arriving lifecycle's first event* -
    carriers do not announce the subscriber leaving - so the gap between the two
    boundaries is zero by construction. Re-testing the window here would measure
    the inferred boundary against itself and pass every time, which is worse than
    not testing it: it would look like a check.

    That same collapse is why the handover has to be handed in rather than
    measured here. ``gaps`` is :func:`~.canonical_events.port_gaps`, which reads
    it off the events where it is still visible, joined on the arriving
    lifecycle's start. Without it the edge still exists - the port is not in
    doubt, only how tight it was - and ``port_gap_days`` is null, which the
    confidence rule reads as unmeasured.
    """
    lc = _lifecycles_with_accounts(lifecycles, accounts)

    # Lifecycles with no slice key are dropped before anything is ordered, and
    # this line is load-bearing rather than defensive housekeeping.
    #
    # ``Window.partitionBy`` treats null as an ordinary value, so every
    # null-phone lifecycle in the estate - every carrier, every year, no
    # relationship of any kind between them - would be gathered into one
    # partition and sorted against each other by start timestamp. The ``lag``
    # below reads the row before as the account the subscriber left, so each of
    # those lifecycles would be recorded as the port destination of whichever
    # unrelated lifecycle happened to sort in front of it, and the port graph
    # would acquire a single path threaded through all of them.
    # ``build_chains`` would then resolve that path into one customer holding
    # every account on it.
    #
    # A silent giant component is the worst thing that can happen at this stage,
    # and it is worse than an outright crash for three separate reasons. It
    # raises nothing: the walk is perfectly happy to order a million-node chain
    # and would only report it as a truncation at ``max_chain_depth``, which
    # reads as a long customer rather than as a fabricated one. It passes DQ:
    # the shared identity rules do check the slice key ``not_null`` at ABORT
    # severity, but they run on the *published* frames in
    # :mod:`~..jobs.lineage_job` after this transform has already built the
    # component, and they fire on the null-keyed rows themselves - while an
    # account that holds one null-phone lifecycle also holds its other,
    # perfectly well-formed lifecycles, and those rows publish the shared
    # ``customer_id`` while breaching no rule at all. And it converges: a
    # rebuild from scratch produces the same wrong customer every time, so
    # re-running is not a way of finding out.
    #
    # Note that this is the one null in the edge rule that has to be excluded
    # explicitly. ``mno`` and ``acct_id`` are compared with ``!=``, which is
    # null-rejecting, so a null in either of those drops the edge on its own and
    # errs conservatively; the slice key is never compared, only partitioned on,
    # so nothing about the predicates below would ever catch it.
    #
    # The 24 August 2026 rebuild did not realise this - ``describe_components``
    # printed a largest port component of 38 accounts, which is a chain of
    # ported accounts and nothing more. That is a fact about that day's feed,
    # not a property of this code, and the component-size line is the only
    # signal that would ever betray it: by the time it reads in the millions the
    # wrong answer has already been published.
    lc = lc.where(F.col(SLICE_KEY).isNotNull())

    order = Window.partitionBy(SLICE_KEY).orderBy(
        F.col("lifecycle_start_ts").asc(), F.col("lifecycle_uid").asc()
    )
    edges = (
        lc.withColumn("_prev_mno", F.lag("mno").over(order))
        .withColumn("_prev_acct", F.lag("acct_id").over(order))
        .where(
            (F.col("mno_port_ind") == F.lit(1))
            & F.col("_prev_acct").isNotNull()
            & (F.col("_prev_mno") != F.col("mno"))
            & (F.col("_prev_acct") != F.col("acct_id"))
        )
        .select(
            F.col("_prev_acct").alias("from_acct_id"),
            F.col("acct_id").alias("to_acct_id"),
            F.col("lifecycle_start_ts").alias("port_ts"),
            F.col(SLICE_KEY),
        )
        .distinct()
    )

    if gaps is None:
        return edges.withColumn(
            "port_gap_days", F.lit(None).cast("double")
        )
    return edges.join(
        gaps.select(
            F.col(SLICE_KEY).alias("_g_phone"),
            F.col("port_ts").alias("_g_ts"),
            F.col("port_gap_days").cast("double").alias("port_gap_days"),
        ),
        (F.col(SLICE_KEY) == F.col("_g_phone")) & (F.col("port_ts") == F.col("_g_ts")),
        "left",
    ).select("from_acct_id", "to_acct_id", "port_ts", SLICE_KEY, "port_gap_days")


# ==========================================================================
# port edges TransUnion witnessed
# ==========================================================================


def tu_latest_response(tu_ports: DataFrame) -> DataFrame:
    """The newest TU response for each phone number, whole.

    TU returns a complete porting history on every call, so two responses for
    one phone number are two pictures of the same thing rather than two halves
    of it. The newest picture is the one to believe, and it is believed *whole*:
    a port the latest history no longer contains has been withdrawn, and merging
    the two responses event by event would keep it alive forever. That is the
    rule in section 3 of the enrichment design and this is the only place it is
    applied.

    Newest is ``ingestion_ts``, with ``request_id`` breaking a tie. The tiebreak
    is arbitrary and it has to exist: two responses landing in the same second
    would otherwise both survive, the rebuild would pick whichever the shuffle
    handed it first, and a table that is rebuilt from scratch every run would
    stop converging. An arbitrary rule applied identically on every run is a
    stable answer; no rule at all is not.
    """
    newest = F.max(F.struct(F.col("ingestion_ts"), F.col("request_id"))).over(
        Window.partitionBy(SLICE_KEY)
    )
    return (
        tu_ports.withColumn("_newest", newest)
        .where(
            (F.col("ingestion_ts") == F.col("_newest.ingestion_ts"))
            & (F.col("request_id") == F.col("_newest.request_id"))
        )
        .drop("_newest")
    )


def tu_port_edges(tu_ports: DataFrame) -> DataFrame:
    """Every port TU witnessed, as the pair of phone numbers it binds.

    This is the customer stage's TU pairing, and it exists as a function of its
    own because two callers need it: the slice closure, which has to know which
    phone numbers a batch drags in through a TU edge before it reads anything,
    and the TU-derived lifecycle build, which turns the same edges into the
    missing middle of a number's history. Restating the rule in both places is
    how a run comes to build an edge to a phone number whose history it never
    read - and rebuild the customer from half its evidence, which converges on a
    *stable wrong answer*, the one failure mode re-running never reveals. The
    argument is the same one that gives :func:`~..slice._l1_rogers_imsi_pairs`
    its reason to exist.

    Two ends come back rather than one, and in the TU feed as specified they are
    always the same phone number: porting history is per number, TU carries no
    counterparty column, and ``correlation_id`` is a null placeholder held for
    schema alignment. So the closure built on this pairing adds no phone number
    the slice does not already have, exactly as the port level of the closure
    adds none today.

    That is a fact about the feed and not a reason to collapse the pair into one
    column. The unsupported-stretch case - a number leaving for a carrier we have
    no feed from and *a different number* arriving from it - is a real
    relationship that TU is the only witness to, and the day TU carries the
    evidence for it, this function is the one place that has to change and the
    closure is not. Deciding whether two ends of an unsupported stretch are one
    subscriber is section F's question, which belongs to the resolver because its
    answer depends on ``fraud_event_ts``; it is deliberately not asked here.

    Only inter-carrier ports are edges. An intra-SPID port is a number moving
    between two switches of the same carrier, which is real history and is kept
    in Silver, but it binds nothing to anything and would put a self-edge in the
    closure for every one of them.
    """
    latest = tu_latest_response(tu_ports)
    return (
        latest.where(F.col("is_inter_carrier_port"))
        .select(
            F.col(SLICE_KEY).alias("a_phone"),
            F.col(SLICE_KEY).alias("b_phone"),
            F.col("event_timestamp").alias("port_ts"),
            F.col("from_spid"),
            F.col("to_spid"),
            F.col("from_mno_raw"),
            F.col("to_mno_raw"),
            F.col("from_mno_std"),
            F.col("to_mno_std"),
        )
        .distinct()
    )


def count_quiet_port_candidates(
    lifecycles: DataFrame, port_window_days: int, recycle_threshold_days: int
) -> int:
    """Cross-carrier reappearances too far apart to be called a port.

    The known gap in the port rule. A subscriber who leaves ROGERS and turns up
    at TELUS six weeks later is the same person, but six weeks is well outside
    the ten-day window and the two accounts stay separate customers. Widening the
    window would trade these false negatives for false positives, because past
    some gap a phone number genuinely does belong to somebody else - that is what
    number recycling is.

    So the window stays where it is and the size of the problem is measured
    instead: reappearances between the port window and the recycle threshold,
    reported as a DQ ``info`` metric. Section 12 of the design document puts the
    decision to the business, and this is the number that decision needs.

    **The band is exact; the ruler is not the classifier's.** Read the number
    that comes out of here as an indication of scale and not as a population
    that could be reconciled against stage 1, because it is measured with a
    different instrument and the difference is systematic rather than noise.

    The band itself is beyond argument. ``gap > port_window_days`` is the
    precise complement of :func:`~.canonical_events._segment_handovers`, which
    calls a handover a port at ``<= window_days``, and
    ``gap < recycle_threshold_days`` is the precise complement of
    :func:`~.canonical_events.mark_recycles`, which calls a gap a recycle at
    ``>= threshold_days``. No overlap with either neighbour and no hole between
    them; a reappearance falls in exactly one of the three.

    What differs is the quantity the thresholds are applied to. Stage 1 measures
    ``_seg_from - _prev_to`` (``canonical_events.py:551-560``): the elapsed time
    between the last canonical event the losing carrier sent and the first the
    gaining carrier sent, straight off the events. This measures
    ``lifecycle_start_ts - previous lifecycle_end_ts``, and a lifecycle boundary
    is not an event timestamp whenever the walk placed it by inference. Two
    consequences, both one-directional:

    * When the previous lifecycle was still open as the new carrier's first
      event arrived, :func:`~.lifecycle.walk_phone` closes it *at that event*
      (``lifecycle.py:259-268``) - the losing carrier does not announce the
      subscriber leaving, so there is nothing else to close it at. The gap
      measured here is then exactly zero however long the true silence was, the
      row falls below ``port_window_days``, and it is never counted. Every
      cross-carrier reappearance of that shape is a quiet-port candidate this
      number does not contain, so the count is a **floor** and not an estimate.
    * When the previous lifecycle *was* closed at a real end event - a
      cancellation, say - that event is the end of the lifecycle but not
      necessarily the end of the carrier's segment: ``_prev_to`` is the maximum
      event timestamp in the segment and picks up any device or SIM change the
      carrier sent afterwards. The gap measured here is then longer than stage
      1's, which can carry a pair across the top of the band and out of the
      count, or into the band from below it.

    Measuring the same quantity here is not possible from these inputs. Doing it
    needs ``_seg_to``, which lives on the canonical events and is not carried on
    any lifecycle column - ``LIFECYCLE_SCHEMA`` has ``is_end_inferred``, which
    says the boundary was placed by inference, but not the event timestamp the
    boundary would otherwise have had - and this function is called from
    :func:`assemble_customers`, which is handed the lifecycles and never the
    events. Widening the signature to take a 353-million-row frame in order to
    sharpen one ``info`` metric is not a trade worth making; saying plainly what
    the number is, is.
    """
    order = Window.partitionBy(SLICE_KEY).orderBy(
        F.col("lifecycle_start_ts").asc(), F.col("lifecycle_uid").asc()
    )
    # Lifecycle boundaries, not event timestamps - see the docstring above for
    # exactly where and in which direction this parts company with the port
    # classifier's own measurement. Anybody reconciling this count against a
    # stage-1 figure should expect them to disagree, and should not read the
    # disagreement as a defect in either.
    gap = days_between("lifecycle_start_ts", "_prev_end")
    return (
        lifecycles.withColumn("_prev_mno", F.lag("mno").over(order))
        .withColumn("_prev_end", F.lag("lifecycle_end_ts").over(order))
        .where(
            (F.col("mno_port_ind") == F.lit(0))
            & F.col("_prev_end").isNotNull()
            & (F.col("_prev_mno") != F.col("mno"))
            & (gap > F.lit(float(port_window_days)))
            & (gap < F.lit(float(recycle_threshold_days)))
        )
        .count()
    )


# ==========================================================================
# traversal
# ==========================================================================

_EDGE_COLUMNS = ("from_acct_id", "to_acct_id", "port_ts", "port_gap_days")

_POSITION = "position"
_CYCLE = "cycle"
_TRUNCATED = "truncated"
_DROPPED = "dropped"

#: Everything one component's walk has to say, in one schema.
#:
#: Four kinds of record share it, told apart by ``kind``. Three of them - the
#: broken cycle edges, the truncated roots and the edges that lost a conflict -
#: are the repairs, and they are here rather than in frames of their own because
#: they come out of the same pass. A ``groupBy`` yields one frame; asking for
#: four would mean four shuffles of the same edges, and the shuffle is the
#: expensive part.
#:
#: For a ``position`` record, ``node`` is the account and ``other`` is the root
#: of its customer chain. For a repair, ``node`` and ``other`` are the two ends
#: of the edge - or, for a truncation, the root that was cut and nothing.
#:
#: The last two columns belong to the ``position`` record and to the account it
#: names: the confidence of the port that *arrived* at it, and the timestamp of
#: the port it left by. Both were separate frames when the traversal ran on the
#: driver and both are one value per account, so they ride along rather than
#: costing a join.
#:
#: ``port_link_confidence`` and not ``link_confidence`` because the account frame
#: this joins onto already carries a ``link_confidence`` of its own, describing
#: the link that joined two *lifecycles*. Two different links, one name, and
#: Spark would refuse the reference as ambiguous. It is renamed on the way out,
#: where there is only one link left to describe.
_WALK_SCHEMA = StructType(
    [
        StructField("kind", StringType(), False),
        StructField("node", StringType(), False),
        StructField("other", StringType(), True),
        StructField("acct_seq_order", IntegerType(), True),
        StructField("customer_length", IntegerType(), True),
        StructField("customer_truncated", IntegerType(), True),
        # Not the same fact as ``customer_truncated``, which is true of every
        # account in a component the cap cut - including the ones the walk did
        # resolve. This one is true only of the accounts on the far side of the
        # cut, whose position and root are unknown rather than merely suspect.
        # The account stage carries the same distinction as
        # ``chain_truncated_tail`` and for the same reason; see
        # :attr:`~.chains.ChainPosition.truncated_tail`.
        StructField("customer_truncated_tail", IntegerType(), True),
        StructField("customer_repaired", IntegerType(), True),
        StructField("port_link_confidence", StringType(), True),
        StructField("outgoing_port_ts", TimestampType(), True),
    ]
)


def _worst_gap_by_edge(edge_rows) -> dict[tuple[Any, Any], float | None]:
    """One handover per account pair: the loosest of however many were observed.

    Two accounts can be joined by more than one port row, and until the audit of
    August 2026 this was collapsed with a dict comprehension over the rows -
    ``{(src, dst): gap for src, dst, _ts, gap in edge_rows}`` - which keeps
    whichever row the comprehension happened to see last. That is the arrival
    order of a Spark shuffle, which is not a property of the input, so where two
    parallel edges carried different handovers the published
    ``link_confidence`` could come out HIGH on one run and LOW on the next from
    byte-identical Silver. The metric it lands on,
    ``info_low_confidence_port_link_share``, read 400,530 on the 24 August 2026
    rebuild, and none of that number was reproducible.

    Parallel edges are real rather than a duplication defect.
    :func:`build_port_edges` de-duplicates on
    ``(from_acct_id, to_acct_id, port_ts, slice_key)``, and
    :func:`customer_port_edges` then projects the slice key away without
    re-de-duplicating, so two rows survive whenever one pair of accounts is
    joined by two genuinely distinct ports - two phone numbers held by the same
    two accounts porting between them, or the same number porting away and back
    and away again. Both are histories the feed can legitimately contain, and
    collapsing them any earlier would lose the second port rather than resolve
    it.

    The worst gap wins, and worst is defined so that the answer is a pure
    function of the *set* of rows. ``max`` is commutative and associative, so no
    arrival order can change it; an unmeasured handover - a null gap, which
    :func:`port_link_confidence_for` already reads as LOW - sorts above every
    measured one, because "we could not measure this handover" is weaker
    evidence than any handover we did measure. So the pair's confidence is HIGH
    only when *every* port joining those two accounts was tight.

    Note that ``build_chains`` keeps only the earliest of a set of parallel rows
    as the structural edge and reports the rest through ``dropped_edges``
    (``chains.py:186-190``) - which is most of what the 10,477 dropped port
    edges of the last full run were. The dropped rows are still evidence about
    the same pair of accounts, though: the subscriber did port between them
    twice, and the second handover being loose is a fact about the claim that
    those two accounts are one person. Keying the map on the pair rather than on
    the individual row is what lets that evidence count, and it is why the
    reconciliation has to happen here rather than being left to whichever row
    the walk kept.

    Taking the worst rather than the best is the conservative direction and it
    is a deliberate call. ``link_confidence`` is a claim that two accounts are
    one subscriber; publishing HIGH for a pair that also carries weak evidence
    overstates how sound that claim is, and the whole purpose of the LOW
    annotation is to mark identities worth confirming against TU's porting
    history. An identity marked for confirmation that did not need it costs a
    lookup. One that needed it and was not marked costs the wrong answer.
    """
    worst: dict[tuple[Any, Any], float | None] = {}
    for src, dst, _ts, gap in edge_rows:
        key = (src, dst)
        if key not in worst:
            worst[key] = gap
        elif worst[key] is None or gap is None:
            worst[key] = None
        elif gap > worst[key]:
            worst[key] = gap
    return worst


def _resolve_component(item, max_depth: int, high_confidence_days: float):
    """Walk one connected component of the port graph.

    This is the body the driver used to run over every edge in the estate,
    unchanged except for what it is given: one component's edges instead of all
    of them. It is the same body because the walk never read across components -
    which of two conflicting edges to drop depends only on the edges touching
    those two accounts, and where to break a cycle depends only on that cycle -
    so a component is the largest thing the answer can depend on, and the
    smallest thing that has to be in one place.

    The rows arrive in whatever order the shuffle produced them.
    :func:`~.chains.build_chains` sorts what it is given and breaks every tie on
    the values themselves, so that is not a source of drift; there is a test
    that runs it twice on shuffled input and compares.
    """
    _component, rows = item
    edge_rows = [tuple(row[1:]) for row in rows]

    edge_list = [(src, dst, ts) for src, dst, ts, _gap in edge_rows]
    gap_by_edge = _worst_gap_by_edge(edge_rows)
    del edge_rows
    nodes = (n for edge in edge_list for n in edge[:2])
    result = build_chains(nodes, edge_list, max_depth=max_depth)

    # A port cycle is a phone number that appears to have ported back to a
    # carrier it had already left. Breaking it publishes a customer the data did
    # not describe, so the whole repaired chain is marked down - see the account
    # stage for why the mark applies to the chain and not just the broken edge.
    repaired_roots = {
        result.positions[node].root
        for pair in result.broken_cycle_edges
        for node in pair
        if node in result.positions
    }

    # Keyed on the arriving account, which has at most one incoming edge left
    # once the conflicts are resolved, so this is one entry per account and not
    # one per edge.
    kept = set(result.kept_edges)
    link_by_node = {
        dst: port_link_confidence_for(
            gap_by_edge.get((src, dst)), high_confidence_days
        )
        for src, dst in kept
    }
    # An account can only leave once, so the earliest surviving departure is the
    # one that ends the customer's view of it.
    outgoing: dict[Any, Any] = {}
    for src, dst, ts in edge_list:
        if (src, dst) not in kept or ts is None:
            continue
        if src not in outgoing or ts < outgoing[src]:
            outgoing[src] = ts

    for acct, pos in result.positions.items():
        yield (
            _POSITION,
            acct,
            pos.root,
            pos.seq_order,
            pos.chain_length,
            1 if pos.truncated else 0,
            1 if pos.truncated_tail else 0,
            1 if pos.root in repaired_roots else 0,
            link_by_node.get(acct),
            outgoing.get(acct),
        )
    for src, dst in result.broken_cycle_edges:
        yield (_CYCLE, src, dst, None, None, None, None, None, None, None)
    for root in result.truncated_roots:
        yield (_TRUNCATED, root, None, None, None, None, None, None, None, None)
    for src, dst in result.dropped_edges:
        yield (_DROPPED, src, dst, None, None, None, None, None, None, None)


# ==========================================================================
# the four steps, each small enough to be one statement
# ==========================================================================
#
# ``build_customers`` below is these four called in order, and on a slice that
# is the right way to run them: the whole stage is a few seconds and splitting
# it would only add writes.
#
# On a full-history rebuild it is the wrong way, and the reason is not
# performance. A Glue interactive statement returns its stdout when it reaches a
# final state, so a statement whose session dies returns nothing at all - not
# the counts, not the round lines, not the exception. The run of 24 August 2026
# spent twenty-one minutes in ``build_customers`` and came back with
# ``Failed to cancel the statement 8 as it has been cancelled`` and no other
# word about it, having printed nine lines that never left the driver.
#
# Four statements, each staging its result to S3, turn that into four answers:
# the ones that finished printed what they found, the one that died is named by
# the fact that it is the first one with nothing staged, and the next attempt
# starts there instead of at the beginning. See
# :mod:`~trust_score_05.lineage.staging`.


def customer_port_edges(
    lifecycles: DataFrame,
    accounts: DataFrame,
    canonical_events: DataFrame | None = None,
    windows: Mapping[str, int | None] | None = None,
    cfg=None,
) -> DataFrame:
    """Step 1 of 4: the port graph, as one row per port.

    The expensive half of this is ``canonical_events``: measuring how tight each
    handover was means a pass over every canonical event in the estate, and at
    full history that is 353 million rows. Everything after it works on the
    edges alone, which are smaller than the input by four orders of magnitude -
    so staging the output of *this* step is what makes the other three cheap.

    The projection below drops the slice key that
    :func:`build_port_edges` de-duplicated on, and it does not de-duplicate
    again. That is deliberate: two accounts joined by two distinct ports are two
    edges, and collapsing them here would silently discard the second port
    rather than resolve it. ``port_edge_count`` in the report is therefore a
    count of port *rows* and not of distinct account pairs - it read 8,988,570
    on the 24 August 2026 rebuild and should be read that way. Where the
    parallel rows disagree about how tight the handover was,
    :func:`_worst_gap_by_edge` is what reconciles them, deterministically and in
    the conservative direction.
    """
    win = dict(windows) if windows is not None else windows_from_config(cfg)
    gaps = (
        None
        if canonical_events is None
        else port_gaps(canonical_events, int(win["mno_port_window_days"]))
    )
    return build_port_edges(lifecycles, accounts, gaps).select(*_EDGE_COLUMNS)


def label_port_components(edges: DataFrame) -> DataFrame:
    """Step 2 of 4: which accounts belong to the same customer, as ``(node, component)``.

    Not which position they hold in it - only which pile they are in. The
    labelling is undirected and the walk that follows is where direction starts
    to matter.

    This is the step that has taken the longest and failed the most: it is
    iterative, so it is the only part of the stage whose cost is not one pass
    over something. ``edges`` should be cached or read off staged files before
    it gets here, because the loop reads it twice per round.
    """
    return component_labels(edges, "from_acct_id", "to_acct_id", what="port edges")


def walk_port_components(
    spark: SparkSession,
    edges: DataFrame,
    labels: DataFrame,
    max_depth: int,
    high_confidence_days: float,
) -> DataFrame:
    """Step 3 of 4: order each component into a chain. Returns the raw walk frame.

    One component to a group, :func:`_resolve_component` per group, and every
    kind of record the walk produces - positions and repairs alike - in the one
    frame described by :data:`_WALK_SCHEMA`.
    """
    # Either end of an edge would do: both are in the same component by
    # definition, which is the whole point of having labelled them.
    keyed = edges.join(
        labels.withColumnRenamed("node", "from_acct_id"), "from_acct_id"
    ).select("component", *_EDGE_COLUMNS)

    # One component to a group. The partition count is passed for the reason
    # given in `spark.shuffle_partitions`: `rdd.groupBy` does not read
    # `spark.sql.shuffle.partitions` on its own, so without it this shuffle is
    # the one the operator cannot tune.
    #
    # A single enormous component would land on one task and one executor, which
    # is the skew this shape can still suffer from. It is strictly better than
    # what it replaces - that component was on the driver before, along with
    # every other - and if it ever bites, the thing to look at is why the port
    # graph has a hub in it, because a customer is a chain and a chain does not.
    # :func:`~.common.describe_components` is what makes that visible; run it on
    # the labels before this and the largest component is a printed number.
    return spark.createDataFrame(
        keyed.rdd.map(tuple)
        .groupBy(lambda row: row[0], shuffle_partitions(spark))
        .flatMap(
            lambda item: _resolve_component(item, max_depth, high_confidence_days)
        ),
        _WALK_SCHEMA,
    )


def positions_from_walk(
    walked: DataFrame,
    report: CustomerBuildReport | None = None,
    max_collected_edges: int | None = MAX_COLLECTED_EDGES,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Split the walk frame into the three frames the assembly joins on.

    Also drains the repairs into ``report``, which is the one thing here that
    comes back to the driver. ``max_collected_edges`` bounds it: these are
    pathologies rather than population, and a run where they are not is a run
    that should refuse rather than fill the driver's heap with the evidence.
    """
    report = CustomerBuildReport() if report is None else report
    is_position = F.col("kind") == F.lit(_POSITION)

    repairs = collect_edge_rows(
        walked.where(~is_position),
        ("kind", "node", "other"),
        limit=max_collected_edges,
        what="port chain repairs",
    )
    for kind, node, other in repairs:
        if kind == _CYCLE:
            report.broken_cycle_edges.append((node, other))
        elif kind == _TRUNCATED:
            report.truncated_roots.append(node)
        elif kind == _DROPPED:
            report.dropped_edges.append((node, other))

    positions = walked.where(is_position).select(
        F.col("node").alias("acct_id"),
        F.col("other").alias("root_acct_id"),
        "acct_seq_order",
        "customer_length",
        "customer_truncated",
        "customer_truncated_tail",
        "customer_repaired",
    )
    # Split off rather than joined on: an account with no incoming port has no
    # confidence to publish and an account that never left has no departure, and
    # a null in a frame that is about to be left-joined would say the same thing
    # as a missing row while costing a row to say it.
    links = walked.where(is_position & F.col("port_link_confidence").isNotNull()).select(
        F.col("node").alias("acct_id"), "port_link_confidence"
    )
    outgoing = walked.where(is_position & F.col("outgoing_port_ts").isNotNull()).select(
        F.col("node").alias("acct_id"), "outgoing_port_ts"
    )
    return positions, outgoing, links


def _resolve_positions(
    spark: SparkSession,
    edges: DataFrame,
    max_depth: int,
    high_confidence_days: float,
    max_collected_edges: int | None = MAX_COLLECTED_EDGES,
) -> tuple[DataFrame, DataFrame, DataFrame, CustomerBuildReport]:
    """Walk the port graph one component at a time; hand the positions back.

    Only accounts that take part in a port are walked. Every other account is a
    customer of one, resolved with a left join rather than a row in the
    traversal, so the work is bounded by how many ports the history contains.

    The kept edges matter here in a way they did not for accounts: a port edge
    that lost a conflict, or fell past the depth cap, must not clip the losing
    account's ``to_ts``. Only edges the traversal actually used may do that.

    **This ran on the driver until 20 August 2026, and the driver is where it
    died.** The edge list was collected whole and the walk was one Python call
    over all of it, guarded by :data:`~.common.MAX_COLLECTED_EDGES` against
    exactly the outcome that then happened: a full-history rebuild refused in
    notebook 04 with more than five million port edges, four minutes in. The
    comment that guard replaced said ports were rarer than number changes and so
    the count here should be smaller than the account stage's. It is not.
    Porting is something a subscriber does on purpose and often; changing your
    number is something almost nobody does twice.

    So the walk moved instead of the limit moving. The edges are labelled by
    connected component in Spark - :func:`~.common.component_labels` - grouped on
    that label, and :func:`_resolve_component` runs on an executor once per
    component. This is the shape :mod:`.chains` describes as the way to scale
    itself, and the shape the lifecycle and device-segment walks have always had:
    Python over one group, distributed by the group key. Nothing about the answer
    changes, because nothing in the walk ever read across components.

    What still comes to the driver is the repairs - the broken cycles, the
    truncated roots and the edges that lost a conflict - because the report
    quotes them and consumers of the report expect the pairs and not a count.
    Those are the pathologies rather than the population: 217 broken cycles and
    6,268 dropped edges in the account stage's last full run, against four
    million edges. ``max_collected_edges`` now bounds that list, which is the
    only list left that a driver has to hold.
    """
    report = CustomerBuildReport()
    # Cached because it is read four times from here - once to count, twice per
    # labelling round, once to group - and rebuilding it means rebuilding the
    # lifecycle join and the gap join underneath it every time.
    edges = edges.select(*_EDGE_COLUMNS).cache()
    report.port_edge_count = edges.count()

    note(f"  port edges: {report.port_edge_count:,}")

    labels = describe_components(label_port_components(edges), what="port components")
    walked = walk_port_components(
        spark, edges, labels, max_depth, high_confidence_days
    ).cache()
    positions, outgoing, links = positions_from_walk(
        walked, report, max_collected_edges=max_collected_edges
    )

    # Read for the last time above: `walked` is cached and materialised by the
    # collect, so the three frames returned no longer need what built it. The
    # labels are not released alongside it because they are checkpointed rather
    # than cached - their blocks go when the frame itself does.
    edges.unpersist()

    return positions, outgoing, links, report


# ==========================================================================
# the stage
# ==========================================================================


def build_customers(
    spark: SparkSession,
    accounts: DataFrame,
    lifecycles: DataFrame,
    canonical_events: DataFrame | None = None,
    windows: Mapping[str, int | None] | None = None,
    cfg=None,
) -> tuple[DataFrame, CustomerBuildReport]:
    """Group the slice's accounts into customers.

    The published grain is one row per lifecycle, same as the account mapping -
    the customer is carried down to the lifecycle rather than the account so that
    a customer can be read off a phone number without a join. ``seq_order`` is
    therefore the *account's* position in the customer chain, repeated on each of
    that account's lifecycles, and ``segment_type`` and ``account_type`` describe
    the account's place in the customer, not the lifecycle's place in the
    account.

    ``canonical_events`` is read for one thing only: how tight each port's
    handover was, which the lifecycles no longer show because the walk closed the
    losing lifecycle at the arriving one's first event. Omitting it leaves the
    customers unchanged and every port link at ``LOW``, since an unmeasured
    handover is not one to vouch for.
    """
    win = dict(windows) if windows is not None else windows_from_config(cfg)

    edges = customer_port_edges(lifecycles, accounts, canonical_events, windows=win)
    positions, outgoing, links, report = _resolve_positions(
        spark,
        edges,
        int(win["max_chain_depth"]),
        float(win["port_high_confidence_days"]),
    )
    return assemble_customers(
        accounts,
        lifecycles,
        positions,
        outgoing,
        links,
        report=report,
        windows=win,
    )


def assemble_customers(
    accounts: DataFrame,
    lifecycles: DataFrame,
    positions: DataFrame,
    outgoing: DataFrame,
    links: DataFrame,
    report: CustomerBuildReport | None = None,
    windows: Mapping[str, int | None] | None = None,
    cfg=None,
    with_counts: bool = True,
) -> tuple[DataFrame, CustomerBuildReport]:
    """Step 4 of 4: join the walk's answer back onto every lifecycle.

    All joins and window functions - no traversal, no iteration, nothing that
    can fail in a way the previous three steps have not already ruled out.

    ``with_counts`` is the escape hatch for a full-history rebuild. The three
    report numbers below - the distinct accounts, the distinct customers, and
    the quiet port candidates - are each a full shuffle over every lifecycle in
    the estate, and all three exist to fill in a log line. On a slice they are
    free. On 77 million lifecycles they are three passes bought with the run's
    last headroom, to print something the published table can be asked for
    afterwards at leisure. The notebook turns them off; the incremental job,
    which runs on a slice and whose report is read, leaves them on.
    """
    report = CustomerBuildReport() if report is None else report
    win = dict(windows) if windows is not None else windows_from_config(cfg)

    # Whether an account is open, and whether its first lifecycle had a real
    # start, are properties of the account as a whole - so they are computed
    # once per account and broadcast back down to its lifecycles.
    facts = (
        accounts.select("acct_id", "lifecycle_uid", "seq_order")
        .join(
            lifecycles.select(
                "lifecycle_uid", "lifecycle_is_open", "is_start_inferred"
            ),
            "lifecycle_uid",
        )
        .groupBy("acct_id")
        .agg(
            F.max(
                F.when(F.col("seq_order") == F.lit(0), F.col("is_start_inferred"))
            ).alias("acct_start_inferred"),
            F.max("lifecycle_is_open").alias("acct_is_open"),
        )
    )

    # No broadcast hints - see the account stage for the general reason. The
    # fourth one was the worst of the four: ``facts`` is not a port-sized frame
    # at all, it is a ``groupBy`` over every lifecycle in the input, so hinting
    # it asked the driver to hold one row per account in the estate. A hint that
    # is right on a slice and fatal on a rebuild is not a hint worth keeping.
    joined = (
        accounts.join(positions, "acct_id", "left")
        .join(outgoing, "acct_id", "left")
        .join(links, "acct_id", "left")
        .join(facts, "acct_id", "left")
    )

    resolved = (
        joined.withColumn(
            "root_acct_id", F.coalesce(F.col("root_acct_id"), F.col("acct_id"))
        )
        .withColumn("acct_seq_order", F.coalesce(F.col("acct_seq_order"), F.lit(0)))
        .withColumn(
            "customer_length", F.coalesce(F.col("customer_length"), F.lit(1))
        )
        .withColumn(
            "customer_truncated",
            F.coalesce(F.col("customer_truncated"), F.lit(0)),
        )
        # An account that never reached the walk has no port on either side, so
        # it is a customer of one for the honest reason rather than a piece of a
        # component the cap cut, and the coalesce says zero.
        .withColumn(
            "customer_truncated_tail",
            F.coalesce(F.col("customer_truncated_tail"), F.lit(0)),
        )
        .withColumn(
            "customer_repaired", F.coalesce(F.col("customer_repaired"), F.lit(0))
        )
        # An account with no incoming port is the head of its customer, and a head
        # makes no claim that two accounts are one subscriber. There is nothing
        # there to be wrong about, so it is ``HIGH`` rather than unknown.
        .withColumn(
            "port_link_confidence",
            F.coalesce(F.col("port_link_confidence"), F.lit(_HIGH)),
        )
        # The rollup answers a different question from the link: not "is this
        # port sound" but "is this whole customer sound". One loose handover
        # anywhere in the chain puts every account in it in doubt, because they
        # are only one customer by way of that link. A chain cut at the depth cap
        # or repaired around a cycle is in the same position - the shape that was
        # published is not the shape the data described.
        .withColumn(
            "_chain_has_low",
            F.max(
                F.when(F.col("port_link_confidence") == F.lit(_LOW), 1).otherwise(0)
            ).over(Window.partitionBy("root_acct_id")),
        )
        .withColumn(
            "customer_link_confidence_min",
            F.when(
                (F.col("_chain_has_low") == F.lit(1))
                | (F.col("customer_truncated") == F.lit(1))
                | (F.col("customer_repaired") == F.lit(1)),
                F.lit(_LOW),
            ).otherwise(F.lit(_HIGH)),
        )
    )

    # The same qualification the account stage makes one level down, for the
    # same reason. A component the depth cap cut has no known end, and the
    # accounts stranded past the cut have no known beginning, so neither "this
    # is the whole customer" nor "this is where the customer finishes" is a
    # claim either row is entitled to make.
    #
    # Nothing published today is wrong here: ``max_chain_depth`` is 50 and the
    # largest port component ever observed is 38, so ``customer_truncated_tail``
    # is zero on every row of every run so far. That is exactly why it is worth
    # writing down now - the first component that exceeds the cap would
    # otherwise publish a stranded account as a subscriber who never ported,
    # with ``customer_truncated = 1`` on the same row saying the opposite.
    is_truncated = F.col("customer_truncated") == F.lit(1)
    is_fragment = F.col("customer_truncated_tail") == F.lit(1)
    is_single = (F.col("customer_length") == F.lit(1)) & ~is_truncated & ~is_fragment
    is_first = (F.col("acct_seq_order") == F.lit(0)) & ~is_fragment
    is_last = (
        F.col("acct_seq_order") == (F.col("customer_length") - F.lit(1))
    ) & ~is_fragment

    out = (
        resolved.withColumn("customer_id", customer_id(F.col("root_acct_id")))
        # The customer's view of this account stops when the subscriber ported
        # away. ``least`` ignores nulls, so an account still open at the losing
        # carrier is clipped to the port and one with no outgoing port keeps its
        # own end.
        #
        # Only a row that was already running when the subscriber left may be
        # clipped, though. An account holds several lifecycles, and the losing
        # carrier sometimes opens another one after the port - it kept the number
        # alive on its own books. Clipping that row too would set its ``to_ts``
        # to a moment before its ``from_ts`` and publish an interval that ends
        # before it starts, which no consumer can read and which reads as data
        # corruption rather than as the carrier disagreement it actually is.
        # Left alone, the row overlaps the arriving account and the overlap is
        # visible, which is the honest rendering of two carriers both claiming
        # the number.
        #
        # A null ``outgoing_port_ts`` makes the condition null, so the row falls
        # to ``otherwise`` and keeps its own end - the same answer ``least``
        # gave, reached the same way.
        .withColumn(
            "to_ts",
            F.when(
                F.col("from_ts") < F.col("outgoing_port_ts"),
                F.least(F.col("to_ts"), F.col("outgoing_port_ts")),
            ).otherwise(F.col("to_ts")),
        )
        .withColumn(
            "segment_type",
            # The truncation tokens are tested first because they describe the
            # traversal and the rest describe the subscriber, and where both
            # have something to say the traversal's admission is the one a
            # consumer needs.
            F.when(is_fragment, F.lit(TRUNCATED_FRAGMENT))
            .when(is_last & is_truncated, F.lit(TRUNCATED_FINAL))
            .when(is_single, F.lit("single"))
            .when(
                is_first & (F.col("acct_start_inferred") == F.lit(1)),
                F.lit("origin_unmatched"),
            )
            .when(is_first, F.lit("origin"))
            .when(
                is_last & (F.col("acct_is_open") == F.lit(1)),
                F.lit("final_unmatched"),
            )
            .when(is_last, F.lit("final"))
            .otherwise(F.lit("intermediate")),
        )
        .withColumn(
            "account_type",
            # The two values a consumer filters on are 'single' for a subscriber
            # who never ported and 'chain' for one who did. A stranded fragment
            # is not known to be either, so it is kept out of both populations
            # rather than silently counted in one.
            F.when(is_fragment, F.lit(TRUNCATED_FRAGMENT))
            .when(is_single, F.lit("single"))
            .otherwise(F.lit("chain")),
        )
        .withColumn("seq_order", F.col("acct_seq_order"))
        .withColumn("updated_source", F.lit(UPDATED_SOURCE))
        .select(
            "customer_id",
            "acct_id",
            "mno",
            SLICE_KEY,
            "lifecycle_uid",
            "seq_order",
            "source_event_id",
            "from_ts",
            "to_ts",
            "segment_type",
            "account_type",
            "updated_source",
            F.col("port_link_confidence").alias("link_confidence"),
            "customer_link_confidence_min",
        )
    )

    # Persisted only when something here is about to read it twice.
    #
    # ``with_counts`` reads ``out`` for the distinct customer count and then
    # hands it to the caller, so without a persist the whole assembly runs
    # twice. With the counts off there is exactly one reader - whoever called
    # this - and persisting for them is a decision that is not this function's
    # to make: the rebuild notebook stages its result to parquet instead,
    # precisely because ``persist`` is memory *and* local disk and the disk half
    # lands on the same executor volume the write's shuffle spills to.
    if with_counts:
        out = out.persist()
        report.counts_taken = True
        report.account_count = accounts.select("acct_id").distinct().count()
        report.customer_count = out.select("customer_id").distinct().count()
        report.quiet_port_candidates = count_quiet_port_candidates(
            lifecycles,
            int(win["mno_port_window_days"]),
            int(win["number_recycle_threshold_days"]),
        )
    LOGGER.info(report.format_line())
    if report.quiet_port_candidates:
        LOGGER.info(
            "customers: %d cross-carrier reappearance(s) fell outside the "
            "%d-day port window but inside the %d-day recycle threshold; these "
            "are separate customers today and may be the same subscriber",
            report.quiet_port_candidates,
            int(win["mno_port_window_days"]),
            int(win["number_recycle_threshold_days"]),
        )
    if report.dropped_edges:
        LOGGER.warning(
            "customers: %d port edge(s) dropped as conflicting", len(report.dropped_edges)
        )
    if report.broken_cycle_edges:
        LOGGER.warning(
            "customers: %d port cycle(s) broken - a phone number that appears to "
            "have ported back to a carrier it had already left",
            len(report.broken_cycle_edges),
        )
    if report.truncated_roots:
        LOGGER.warning(
            "customers: %d port chain(s) hit max_chain_depth=%d",
            len(report.truncated_roots),
            int(win["max_chain_depth"]),
        )
    return out, report


def to_gold(customers: DataFrame, run_ts: datetime) -> DataFrame:
    """Project onto ``msisdn_lifecycle_account_customer_mapping``."""
    stamped = with_audit_columns(customers, F.lit(run_ts).cast("timestamp"))
    return align_to_schema(stamped, CUSTOMER_MAPPING_SCHEMA)
