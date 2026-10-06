"""Helpers shared by every Gold stage: surrogate keys, business windows,
bookkeeping columns, and the small ordering utilities the walks depend on.

Split rule, inherited from Bronze -> Silver: anything where two stages
legitimately differ gets its own module; anything where a difference would be a
bug lives here. Key construction is the clearest example - if two stages hashed
a lifecycle differently, every merge downstream would silently duplicate.
"""

from __future__ import annotations

import time
from typing import Iterable, Sequence

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from ..progress import note
from ..schemas import SCHEMA_VERSION

__all__ = [
    # business windows
    "MNO_PORT_WINDOW_DAYS",
    "PORT_HIGH_CONFIDENCE_DAYS",
    "NUMBER_RECYCLE_THRESHOLD_DAYS",
    "ROGERS_MSISDN_OVERLAP_TOLERANCE_DAYS",
    "ROGERS_IMSI_REUSE_MAX_GAP_DAYS",
    "ROGERS_STATUS_LOOKAHEAD_DAYS",
    "NUMBER_CHANGE_MATCH_WINDOW_DAYS",
    "TELUS_PLAN_CHANGE_WINDOW_HOURS",
    "TEMPORARY_PN_MAX_DAYS",
    "MAX_CHAIN_DEPTH",
    "WINDOW_DEFAULTS",
    "windows_from_config",
    # the driver-side traversals
    "MAX_COLLECTED_EDGES",
    "EdgeListTooLarge",
    "collect_edge_rows",
    # the distributed traversals
    "forget_statistics",
    "truncate_lineage",
    "MAX_COMPONENT_ROUNDS",
    "ComponentsDidNotConverge",
    "component_labels",
    "describe_components",
    # keys
    "sha2_cols",
    "ac_event_id",
    "lifecycle_uid",
    "acct_id",
    "customer_id",
    # frame helpers
    "with_audit_columns",
    "align_to_schema",
    "event_order_window",
    "join_sorted_ids",
    "days_between",
    "hours_between",
]


# ==========================================================================
# business windows
# ==========================================================================
# These are business rules, not tuning parameters. They are module constants so
# they can be imported by tests and by the rebuild code, and they are also every
# one of them overridable from config - see ``windows_from_config`` - because
# section 12 of the design document puts two of them to the business as open
# questions.

#: A phone number ending at one MNO and starting at another inside this many
#: days is a port. Beyond it, the two are unrelated as far as Gold can tell.
#: This window decides whether the reappearance is a port; it says nothing about
#: how sure of it we are, which is :data:`PORT_HIGH_CONFIDENCE_DAYS`.
MNO_PORT_WINDOW_DAYS = 10

#: A handover this tight is a port we can act on without confirming the
#: subscriber elsewhere. Beyond it the port link is still made - that is
#: :data:`MNO_PORT_WINDOW_DAYS` - but a number that sat idle for weeks between
#: two carriers is as likely to have been quarantined and reassigned, so the
#: claim that it is one subscriber is recorded as a guess. Two different
#: quantities, and conflating them would either refuse real ports or vouch for
#: guessed ones.
PORT_HIGH_CONFIDENCE_DAYS = 3

#: An explicit start after at least this much complete silence is a recycle -
#: the number has been reassigned to a different subscriber. Compared
#: inclusively, so a gap of exactly this many days is a recycle.
NUMBER_RECYCLE_THRESHOLD_DAYS = 90

#: ROGERS sends no number-change event, so the change is inferred from one SIM
#: card carrying two phone numbers. The feed is not simultaneous - the old
#: number's ``IMSI_END`` routinely lands after the new number's ``IMSI_START`` -
#: so an overlap up to this long is that lag and the two numbers are still one
#: account. Longer than this is two subscriptions on one SIM, or a feed defect,
#: and the link is marked down rather than made silently.
ROGERS_MSISDN_OVERLAP_TOLERANCE_DAYS = 1

#: Outer bound on how long a SIM card may sit between its two phone numbers and
#: still be read as one subscriber changing their number. ``None`` means no
#: bound, which is the rule as specified: the subscriber keeps their SIM across
#: the change however long they waited, and those long gaps are the common case
#: this inference exists to recover. It is a lever for the day SIM cards turn out
#: to be reissued to different subscribers, which nothing in the feed otherwise
#: distinguishes from a number change.
ROGERS_IMSI_REUSE_MAX_GAP_DAYS = None

#: How far the ROGERS suspension/cancellation lookahead reads. ``None`` means
#: unbounded within the phone number's own history, which is what the
#: specification describes; the previous code capped it at 10 days and that cap
#: is the source of the mis-classification recorded in the audit document.
ROGERS_STATUS_LOOKAHEAD_DAYS = None

#: TELUS emits a cancel then an activate for a plan change. Inside this many
#: hours, in that order, the pair is one ``plan_change`` event.
TELUS_PLAN_CHANGE_WINDOW_HOURS = 24

#: How far apart the two halves of one number change may be timestamped and
#: still be matched into an account edge. The carrier sends the ``from`` and the
#: ``to`` as separate rows, sometimes hours apart and occasionally on either side
#: of midnight; the counterparty route in the account stage needs a tolerance to
#: pair them. Kept small on purpose - a wide tolerance would let an unrelated
#: later number change be pulled in as the counterparty.
NUMBER_CHANGE_MATCH_WINDOW_DAYS = 5

#: Reserved for the temporary-phone-number rule that was never specified.
TEMPORARY_PN_MAX_DAYS = 10

#: Traversal guard on number-change and port chains. A component longer than
#: this is emitted up to the cap and flagged, never silently truncated.
MAX_CHAIN_DEPTH = 50

WINDOW_DEFAULTS: dict[str, int | None] = {
    "mno_port_window_days": MNO_PORT_WINDOW_DAYS,
    "port_high_confidence_days": PORT_HIGH_CONFIDENCE_DAYS,
    "number_recycle_threshold_days": NUMBER_RECYCLE_THRESHOLD_DAYS,
    "rogers_msisdn_overlap_tolerance_days": ROGERS_MSISDN_OVERLAP_TOLERANCE_DAYS,
    "rogers_imsi_reuse_max_gap_days": ROGERS_IMSI_REUSE_MAX_GAP_DAYS,
    "rogers_status_lookahead_days": ROGERS_STATUS_LOOKAHEAD_DAYS,
    "number_change_match_window_days": NUMBER_CHANGE_MATCH_WINDOW_DAYS,
    "telus_plan_change_window_hours": TELUS_PLAN_CHANGE_WINDOW_HOURS,
    "temporary_pn_max_days": TEMPORARY_PN_MAX_DAYS,
    "max_chain_depth": MAX_CHAIN_DEPTH,
}


def windows_from_config(cfg) -> dict[str, int | None]:
    """Merge the ``windows`` config section over :data:`WINDOW_DEFAULTS`.

    Returns a plain dict so transforms can be unit-tested without a Config
    object, and so a test can override one window without touching the others.
    """
    resolved = dict(WINDOW_DEFAULTS)
    section = (cfg.get("windows", {}) or {}) if cfg is not None else {}
    for key, value in section.items():
        if key not in resolved:
            raise KeyError(
                f"unknown window {key!r} in config; known windows: "
                f"{', '.join(sorted(resolved))}"
            )
        resolved[key] = None if value is None else int(value)
    return resolved


# ==========================================================================
# the driver-side traversals
# ==========================================================================
# Two stages resolve chains by pulling an edge list to the driver and walking it
# in Python - accounts through number changes, customers through ports. Both are
# written on the same assumption, stated in ``chains.py``: edges exist only where
# a number changed or a port happened, which is a small fraction of any slice.
#
# That assumption is about a *slice*. A rebuild is not a slice, and the first
# full-history run of the account stage killed the Glue session outright: no
# Spark error anywhere in the traceback, only the session gone. That is the
# driver's JVM being killed, and the two things that kill it here are below.

#: The most edges either traversal will pull to the driver before refusing.
#:
#: Not a tuning knob so much as a tripwire. The driver holds roughly a kilobyte
#: per edge once the Python edge list, the evidence map, the node set and one
#: position object per node are all live at the same time, so five million edges
#: is already several gigabytes and a ``G.1X`` driver has sixteen in total. Past
#: this the run is going to die; the only question is whether it dies here, in
#: one line that says what to do, or ten minutes later in a botocore traceback
#: about a session that no longer exists.
MAX_COLLECTED_EDGES = 5_000_000


class EdgeListTooLarge(RuntimeError):
    """The edge list is past what the driver-side traversal will attempt.

    It says *more than* the limit rather than an exact count, deliberately.
    Counting the rest would mean holding the rest, which is the thing being
    refused; and the exact number changes nothing about what to do next.
    """

    def __init__(self, what: str, limit: int) -> None:
        self.what = what
        self.limit = limit
        super().__init__(
            f"{what}: more than {limit} edges, which is past the driver-side "
            "limit. This traversal runs in Python on the driver, so the edge "
            "list has to fit in the driver's memory. Either give the driver "
            "more of it - on Glue that is --worker-type G.4X or G.8X, which "
            "take the driver to 64GB and 128GB from G.1X's 16GB - and raise "
            "max_collected_edges to match, or narrow the window being rebuilt "
            "and run it in parts."
        )


def collect_edge_rows(
    frame: DataFrame,
    columns: Sequence[str],
    limit: int | None = MAX_COLLECTED_EDGES,
    what: str = "edges",
) -> list[tuple]:
    """Bring ``frame``'s edges to the driver as plain tuples, one partition at a time.

    Three things about this that ``.collect()`` does not do, each of which cost
    a rebuild:

    ``toLocalIterator`` fetches one partition at a time. ``.collect()`` builds
    the entire result as an array in the driver's JVM *and then* ships it to
    Python, so both copies are live at the peak - which is the copy that killed
    the account stage, because a JVM driver being killed takes the whole session
    with it and leaves no Spark error behind to read.

    The rows come back as tuples, not ``Row`` objects. A ``Row`` is a tuple with
    a schema attached and costs a few hundred bytes more than the values it
    holds; at these counts that difference is measured in gigabytes.

    And it counts as it goes, so passing ``limit`` turns running out of driver
    memory into :class:`EdgeListTooLarge`, raised at the point where the number
    is known and while there is still memory left to raise it with.
    """
    rows: list[tuple] = []
    for row in frame.select(*columns).toLocalIterator():
        rows.append(tuple(row))
        if limit is not None and len(rows) > limit:
            raise EdgeListTooLarge(what, limit)
    return rows


# ==========================================================================
# the distributed traversals
# ==========================================================================
# The way past :class:`EdgeListTooLarge`, and the way the traversals should have
# been written for a full history in the first place.
#
# Every decision a chain walk makes is local to a connected component: which of
# two conflicting edges to drop depends only on the other edges touching those
# nodes, and where to break a cycle depends only on that cycle. Nothing reads
# across components. So the walk does not need the whole edge list in one
# process - it needs each component in one process, which is a ``groupBy`` and
# not a ``collect``. What is missing to do that is a key to group on, and that is
# what this section produces.
#
# The label is the least node id in the component, which makes it stable across
# runs: it is a property of the graph and not of the order Spark happened to
# read it in.

#: How many rounds of label propagation to allow before giving up.
#:
#: This was thirty, on the strength of a claim that each round doubles the
#: distance a label travels and that thirty rounds therefore covers a component a
#: billion nodes deep. The doubling is a best case and not a guarantee - see
#: :func:`component_labels` for why - and the guaranteed rate is one hop per
#: round, so the rounds a component needs are bounded by its diameter and not by
#: the logarithm of it. A chain of *n* nodes can take *n - 1* rounds, and does
#: whenever the node ids run along it in an unhelpful order, which is the normal
#: case here because every node id is a SHA-256 hash and hashes sort at random
#: with respect to the chain.
#:
#: Thirty therefore had *negative* headroom against the graph it was actually
#: run on. The 24 August 2026 rebuild's largest port component was 38 accounts
#: and the loop took 15 rounds, which is the median for a 38-node chain under
#: randomly ordered ids; the worst case for that same component is 37 rounds,
#: seven past the cap. The run converged because it was lucky, not because the
#: cap allowed for it.
#:
#: Two hundred is the honest number: it covers the worst case for any component
#: up to 200 nodes end to end, which is four times the ``MAX_CHAIN_DEPTH`` the
#: walk downstream is willing to emit and five times the largest component this
#: estate has produced. It is a tripwire and not a budget - a converging graph
#: never comes near it, and a component past 200 accounts is not a long customer
#: but a hub, which is what :func:`describe_components` prints the largest
#: component in order to reveal.
#:
#: Hitting the cap still raises rather than carrying on with labels that have not
#: settled, and that has not changed and must not. A half-propagated label splits
#: one component into several, the traversal resolves each half separately, and
#: the run publishes chains nobody described - quietly, and identically on every
#: re-run.
MAX_COMPONENT_ROUNDS = 200


def forget_statistics(frame: DataFrame) -> DataFrame:
    """Rebuild a checkpointed frame so its plan carries no inherited statistics.

    A checkpoint cuts the operator tree down to one leaf, but it hands that leaf
    the *statistics* the tree had - ``sizeInBytes`` above all. Spark's
    ``SizeInBytesOnlyStatsPlanVisitor`` has no join cardinality model, so at a
    binary node it returns the **product** of its children's sizes. The pointer
    jump in :func:`component_labels` self-joins ``stepped`` against a projection
    of ``stepped``, which makes that product a square: the ``BigInt`` statistic
    is squared once per round and its bit length therefore *doubles* per round,
    however small the data actually is.

    Nothing notices for the first twenty rounds. Then multiplying two numbers
    thousands of bits wide starts to cost real driver time - the dev bootstrap of
    1 September 2026 went 20 s, 41 s, 74 s, 157 s, 467 s, 1202 s over rounds 20
    to 25 while the graph itself never changed size - and in round 26 the value
    passed ``java.math.BigInteger``'s magnitude limit and the run died with
    ``BigInteger would overflow supported range``. Both checkpoint variants
    behave this way; ``localCheckpoint`` propagates statistics exactly as the
    reliable form does, so this is latent in any long propagation and is not a
    property of where the checkpoint is written.

    The cure is to hand Spark a leaf it knows nothing about. Round-tripping the
    checkpointed frame's ``javaRDD`` back through ``createDataFrame`` produces a
    fresh ``LogicalRDD`` with no statistics attached, so the estimate falls back
    to the default ``Long.MaxValue`` and stays there - flat, round after round.
    It is all done on the JVM, so no rows come to the driver; the cost is one
    ``InternalRow`` -> ``Row`` -> ``InternalRow`` pass on the executors. The
    checkpoint itself is untouched, which is what keeps the reliable/local choice
    above and the ``ContextCleaner``'s cleanup of those blocks working as before.

    The one side effect worth naming: a ``Long.MaxValue`` estimate is above every
    broadcast threshold, so these frames will not be broadcast. In this loop that
    changes nothing - the edge and label frames run to millions of rows and were
    never broadcast candidates - but it is the reason this is applied at the
    checkpoint boundary of an iterative stage rather than anywhere a plan looks
    large.

    Falls back to the frame it was given. Spark Connect has no ``_jdf``, and a
    flat statistic is a performance property rather than a correctness one, so a
    session that cannot do this should still get its answer.
    """
    if not hasattr(frame, "_jdf"):
        return frame
    try:
        jdf = frame._jdf
        reset = jdf.sparkSession().createDataFrame(jdf.javaRDD(), jdf.schema())
        return DataFrame(reset, frame.sparkSession)
    except Exception:  # noqa: BLE001 - the answer matters, the estimate does not
        return frame


def truncate_lineage(frame: DataFrame) -> DataFrame:
    """Materialise ``frame`` and forget how it was computed.

    ``cache`` remembers the *result* but keeps the recipe: the logical plan still
    describes every operator that produced it, because Spark has to be able to
    recompute a lost partition. That is the right trade for a frame read twice,
    and the wrong one inside a loop that feeds each round into the next - the
    plan then carries one copy of itself per round, and a round that reads its
    input twice carries two. Analysing and optimising that tree is driver work
    that doubles per round while the data stays the same size, which is how an
    iteration that should take minutes spends an hour printing plans instead.

    Checkpointing is what actually cuts the recipe. The frame is written out,
    and what comes back is a plan with a single leaf. Reliable checkpointing
    (``DataFrame.checkpoint``) writes to the checkpoint directory, so a lost
    partition is re-read rather than recomputed; it needs somebody to have
    called ``setCheckpointDir``. Without one, the local variant keeps the blocks
    on the executors, which is faster and enough for a session that holds its
    executors, but a lost executor takes the only copy with it and the query
    fails rather than recovering. Preferring the reliable form when a directory
    exists means a caller who cares can have durability by setting one, and
    everybody else still gets the truncation, which is the part that was
    missing.

    Cutting the plan is necessary and, on its own, not sufficient - see
    :func:`forget_statistics`, which every branch below routes through. A
    checkpoint replaces the operator tree with a single leaf but hands that leaf
    the *statistics* the tree had, and the statistic is the thing that was
    actually doubling.

    ``cache`` is the last resort and not the intended path. A Spark Connect
    session has no ``SparkContext`` to ask about a checkpoint directory, and
    before Spark 4 it has no ``checkpoint`` on the DataFrame either; caching
    there is slower to plan but correct, where raising would take out a stage
    that has nothing wrong with it. It is the one branch that leaves the plan
    growing, so it says so on the way past rather than degrading in silence.
    """
    directory = None
    try:
        directory = frame.sparkSession.sparkContext.getCheckpointDir()
    except Exception:  # noqa: BLE001 - Spark Connect has no SparkContext
        pass

    if directory and hasattr(frame, "checkpoint"):
        return forget_statistics(frame.checkpoint(eager=True))
    if hasattr(frame, "localCheckpoint"):
        return forget_statistics(frame.localCheckpoint(eager=True))

    print(
        "WARNING: this session offers no checkpoint, so an iterative stage is "
        "falling back to cache(). The answer is the same; the logical plan will "
        "grow with every round and the driver pays for it."
    )
    return frame.cache()


def _release(frame: DataFrame) -> None:
    """Drop a frame's materialised blocks, and never fail for doing so.

    The counterpart to :func:`truncate_lineage`. Every caller of this is saying
    "I am finished with this intermediate", which is a hint and not a
    correctness requirement - so a session that has no ``unpersist``, or a frame
    that was never persisted in the first place, is not an error worth ending a
    forty-minute run over.

    Non-blocking on purpose: the driver has no reason to wait for executors to
    confirm, and the point is to stop *tracking* the blocks, which happens
    immediately either way.
    """
    try:
        frame.unpersist(blocking=False)
    except Exception:  # noqa: BLE001 - releasing memory must never fail a run
        pass


class ComponentsDidNotConverge(RuntimeError):
    """Label propagation ran out of rounds with labels still moving."""

    def __init__(self, what: str, rounds: int) -> None:
        self.what = what
        self.rounds = rounds
        super().__init__(
            f"{what}: connected components did not settle in {rounds} rounds of "
            "label propagation. There are two causes and the round lines printed "
            "above tell them apart. If the number of labels still moving has been "
            "falling steadily, this is a depth problem: the pointer jump only "
            "doubles a label's reach when the node ids happen to run with the "
            "graph, and the guaranteed rate is one hop per round, so a component "
            f"more than about {rounds} nodes end to end can genuinely need more "
            "rounds than this. Look at the largest component printed by "
            "describe_components - if it is in the thousands the graph has a hub "
            "in it and the answer is to find the hub, not to raise the cap. If "
            "instead the count is not falling, or is moving erratically, the edge "
            "frame is changing between rounds, which happens when it is built "
            "from a non-deterministic expression rather than cached. Cache the "
            "edges before labelling them."
        )


def component_labels(
    frame: DataFrame,
    src_column: str,
    dst_column: str,
    max_rounds: int = MAX_COMPONENT_ROUNDS,
    what: str = "edges",
) -> DataFrame:
    """Label each node with its connected component, as ``(node, component)``.

    Undirected: the direction of an edge decides where a chain begins, which is
    the traversal's business, but it has nothing to do with which nodes belong
    together. ``A -> B`` and ``B -> A`` are the same component either way.

    The algorithm is label propagation with pointer jumping. Every node starts
    labelled with itself; each round takes the least label among its neighbours,
    then replaces its label with *that label's* label. The first half moves a
    label one hop. The second half is the one that used to be described here as
    doubling the distance covered, so that a chain of length n settled in about
    log2(n) rounds - and that claim is wrong, in a way worth spelling out
    because the round cap was set on the strength of it.

    Write ``stepped(v)`` for the label after the one-hop half. It is the least
    label in the ball of radius one around ``v``, so if labels already covered
    radius ``r``, ``stepped`` covers ``r + 1``. The jump then replaces it with
    ``stepped(stepped(v))``. Let ``w = stepped(v)``: ``w`` sits within ``r + 1``
    of ``v``, and ``stepped(w)`` is the least label within ``r + 1`` of ``w``,
    so the result is the least label over one ball of radius ``r + 1`` centred
    somewhere in ``v``'s ball - a *subset* of ``v``'s ball of radius
    ``2r + 2``, not the whole of it. Doubling is therefore an upper bound on how
    far a label can have travelled, never a floor. The only guaranteed progress
    in a round is the single hop, so the worst case is the component's diameter.

    That is not a theoretical worst case reached by an adversary; it is the
    ordinary case here. Simulating this exact loop on a 38-node chain - the
    largest port component of the 24 August 2026 rebuild - over random
    orderings of the node ids gives a median of 15 rounds, a 95th percentile of
    27 and a maximum of 37, against the 6 that doubling would promise. The
    rebuild took 15. Node ids in this pipeline are SHA-256 hashes, so a random
    ordering is precisely what a chain of them has.

    The jump is still worth its cost: when the ids do run with the graph it
    collapses a fifty-long chain into seven rounds instead of fifty, and it
    never makes a round worse. What it cannot be is the basis for a round
    budget - see :data:`MAX_COMPONENT_ROUNDS`, which is sized on the diameter.

    Each round's labels are checkpointed, not cached, so the next round starts
    from a plan with one leaf. Caching was not enough and the difference is not
    academic: a full-history rebuild on 20 August 2026 lost its Glue session
    after fifty-six minutes in this function, and the driver log held one
    logical plan carrying dozens of copies of the same
    ``Exchange hashpartitioning(node, 2000) ... plan_id=11374`` at every
    indentation depth. Cached data is still described by the operators that
    produced it, and this loop reads its input twice per round - once in
    ``proposed`` and once in the self-join - so the tree multiplies rather than
    growing. See :func:`truncate_lineage`.

    ``frame`` should be cached before it gets here. It is read twice per round,
    and if recomputing it is expensive then labelling costs more than the
    traversal it exists to enable.

    It prints a line per round. This function is where the 20 August rebuild
    spent fifty-six minutes before its session went away, and what came back to
    the notebook was the connection's own "the statement was cancelled" wrapper,
    which names no cause and no location. A round line costs nothing - ``moved``
    is counted anyway, because the loop stops on it - and it turns that wrapper
    into "it died in round nine of label propagation with four million labels
    still moving", which is a different conversation. Silence during a
    forty-minute call is indistinguishable from a hang.
    """
    started = time.monotonic()
    peers = truncate_lineage(
        frame.select(
            F.col(src_column).alias("node"), F.col(dst_column).alias("peer")
        )
        .union(
            frame.select(
                F.col(dst_column).alias("node"), F.col(src_column).alias("peer")
            )
        )
        .distinct()
    )
    labels = peers.select("node").distinct().withColumn("component", F.col("node"))

    # ``peers`` is checkpointed, so this count reads the checkpoint rather than
    # replaying the union - it is the one free measurement of the graph's size,
    # and a wildly larger number than expected is itself the diagnosis.
    note(
        f"  {what}: label propagation over {peers.count():,} directed pairs "
        f"({time.monotonic() - started:.0f}s to build them)"
    )

    for _round in range(max_rounds):
        round_started = time.monotonic()
        proposed = (
            peers.join(labels, "node")
            .groupBy("peer")
            .agg(F.min("component").alias("_proposed"))
            .withColumnRenamed("peer", "node")
        )
        # Checkpointed here as well as at the end of the round, because the jump
        # below reads it twice and the whole point of this loop's shape is that
        # nothing gets read twice through a plan.
        stepped = truncate_lineage(
            labels.join(proposed, "node", "left").select(
                "node",
                F.least(
                    F.col("component"),
                    F.coalesce(F.col("_proposed"), F.col("component")),
                ).alias("component"),
            )
        )
        # The jump. `stepped` is joined to itself on "the node my label names",
        # so a node whose label is two hops away comes out labelled four hops
        # away. A label that names a node with no label of its own cannot happen
        # - every label is a node in this frame - and the left join is there for
        # the empty-graph case rather than for that.
        #
        # The other side is renamed rather than aliased. `df.alias("x")` on both
        # sides of a self-join leaves two branches carrying the same attribute
        # ids, and the analyser's deduplication rewrites one of them - which is
        # why `mine.component` came back unresolvable while being listed as a
        # candidate. Distinct column names have no such ambiguity to resolve.
        other = stepped.select(
            F.col("node").alias("_their_node"),
            F.col("component").alias("_their_component"),
        )
        jumped = truncate_lineage(
            stepped.join(
                other, F.col("component") == F.col("_their_node"), "left"
            ).select(
                "node",
                F.coalesce(
                    F.col("_their_component"), F.col("component")
                ).alias("component"),
            )
        )
        # Counting the movers against the labels we came in with is the only
        # reason the previous round is still referenced here, and it reads a
        # checkpoint on both sides rather than replaying the round.
        moved = (
            jumped.join(
                labels.withColumnRenamed("component", "_was"), "node"
            )
            .where(F.col("component") != F.col("_was"))
            .count()
        )
        # Nothing needs `stepped` or the incoming `labels` past this line, and
        # saying so is what keeps the round count off the driver's heap.
        #
        # A checkpoint is backed by cached blocks, and the block *metadata* -
        # one entry per partition per checkpoint - lives on the driver for as
        # long as something references the RDD. Three checkpoints a round at 512
        # shuffle partitions is 1,536 entries per round, and the loop is allowed
        # 200 rounds. `ContextCleaner` would eventually collect them, because
        # both names are rebound on the next pass, but it only looks when a JVM
        # GC has run and `spark.cleaner.periodicGC.interval` defaults to thirty
        # minutes - longer than the account stage of a full-history rebuild had
        # been alive when the 1 September 2026 dev run died with
        # `java.lang.OutOfMemoryError: Java heap space` in a 24 GB driver, three
        # lines after `account edges: correlation_id=3,390,953, ...`.
        #
        # Releasing them here is not an optimisation of that default; it removes
        # the dependence on it. `jumped` was materialised eagerly above, so it
        # does not need `stepped` to stay around, and `moved` has already been
        # counted, so nothing needs the old `labels` either.
        _release(stepped)
        _release(labels)
        labels = jumped
        note(
            f"  {what}: round {_round + 1} of {max_rounds}, "
            f"{moved:,} labels moved, {time.monotonic() - round_started:.0f}s "
            f"({time.monotonic() - started:.0f}s total)"
        )
        if moved == 0:
            # `peers` was read twice a round and is read no more; the caller
            # wants the labels, not the graph they were derived from.
            _release(peers)
            return labels

    _release(peers)
    raise ComponentsDidNotConverge(what, max_rounds)


def describe_components(labels: DataFrame, what: str = "components") -> DataFrame:
    """Print how the nodes are spread across components; return ``labels``.

    Both walks group on the component label and run Python over one group per
    task, and both carry the same comment about the one thing that shape can
    still suffer from: a single enormous component lands on a single task on a
    single executor, and no amount of shuffle width helps. That comment ends
    "if it ever bites, the thing to look at is why the graph has a hub in it" -
    which needs somebody to be able to see the hub. This is how they see it.

    The largest component's size is the whole diagnosis. A customer is a chain
    of ported accounts; the longest honest chain is tens of accounts, so a
    component of tens of thousands is not a long customer, it is a hub - one
    account both ends of the graph link to, usually a placeholder identifier
    that survived Silver. Against that, a run that dies with the largest
    component at forty is not skewed and the cause is elsewhere.

    The cost is one aggregation over a checkpointed two-column frame, plus a
    top-five sort of the group counts. That is the cheapest job either stage
    runs, and it buys the difference between a diagnosis and a guess.
    """
    # Cached because it is read twice - once for the count and once for the top
    # five - and the second read would otherwise redo the aggregation. Released
    # before returning: nothing downstream reads it, and holding a block per
    # component for the rest of the stage would be paying for a printed line.
    sizes = labels.groupBy("component").agg(F.count(F.lit(1)).alias("size")).cache()
    try:
        component_count = sizes.count()
        if component_count == 0:
            note(f"  {what}: no components - the graph has no edges")
            return labels
        biggest = sizes.orderBy(F.col("size").desc()).limit(5).collect()
        note(
            f"  {what}: {component_count:,} components, largest "
            + ", ".join(f"{row['size']:,}" for row in biggest)
            + f" (top five hold {sum(r['size'] for r in biggest):,} nodes)"
        )
    finally:
        sizes.unpersist()
    return labels


# ==========================================================================
# surrogate keys
# ==========================================================================
# Every key is a hash of facts that hold for the entire life of the thing it
# identifies. The moment a mutable attribute enters a seed, an update becomes
# indistinguishable from an insert - the key changes, the merge does not match,
# and the table grows a duplicate. Section 5 of the design document explains
# which attributes were removed from the previous seeds and why.


def sha2_cols(*cols: Column | str) -> Column:
    """``sha2(concat_ws("|", coalesce(c, "")), 256)`` over the given columns.

    ``coalesce`` to empty string rather than letting a null poison the whole
    hash, and ``concat_ws`` with an explicit separator so that ("ab", "c") and
    ("a", "bc") cannot collide.
    """
    if not cols:
        raise ValueError("sha2_cols() needs at least one column")
    parts = [
        F.coalesce(F.col(c) if isinstance(c, str) else c, F.lit("")).cast("string")
        for c in cols
    ]
    return F.sha2(F.concat_ws("|", *parts), 256)


def ac_event_id(
    phone_hash: Column | str,
    mno: Column | str,
    event_type: Column | str,
    event_ts: Column | str,
    source_record_id: Column | str,
) -> Column:
    """Key for one canonical event.

    A canonical event is immutable, so every component is safe. ``source_record_id``
    is what keeps two events that look identical - same phone number, same type,
    same second - but came from distinct Silver rows from collapsing into one.
    Derived events, which have no single source row, pass the sorted, joined
    list of the record ids they were derived from.
    """
    return sha2_cols(
        F.lit("ac_event"),
        F.lit(SCHEMA_VERSION),
        phone_hash,
        mno,
        event_type,
        F.date_format(
            F.col(event_ts) if isinstance(event_ts, str) else event_ts,
            "yyyy-MM-dd HH:mm:ss",
        ),
        source_record_id,
    )


def lifecycle_uid(
    phone_hash: Column | str,
    mno: Column | str,
    anchor_ac_event_id: Column | str,
) -> Column:
    """Key for one lifecycle, seeded on its **anchor**.

    The anchor is the earliest canonical event belonging to the lifecycle.
    Closing the lifecycle, adding events to it, and flipping its inferred flags
    all leave the anchor alone, which is exactly the property the previous
    seed - which included ``lifecycle_infer_type`` - did not have.

    The anchor does move when an event arrives that predates everything we had.
    That is a genuine change of fact and is handled as a key migration; see
    section 5.3 of the design document.
    """
    return sha2_cols(
        F.lit("lifecycle"),
        F.lit(SCHEMA_VERSION),
        phone_hash,
        mno,
        anchor_ac_event_id,
    )


def acct_id(root_lifecycle_uid: Column | str) -> Column:
    """Key for one account, seeded on the root of its number-change chain.

    One recipe for chains and singles, because a single account is a chain of
    length one. The previous code used a different literal prefix for each, so
    an account that gained a second lifecycle changed identity and reappeared as
    a duplicate.

    **This seed is not stable under new evidence, and that is the open problem
    the incremental work has to solve.** Read this note and the matching one on
    :func:`customer_id` together; they are one problem at two levels, and the
    customer level inherits everything that happens at this one.

    The section comment above says a key must be a hash of facts that hold for
    the entire life of the thing it identifies. ``root_lifecycle_uid`` is not
    such a fact. It is the head of the number-change chain *as the walk resolved
    it on the data available at that moment*, and there are exactly three ways
    it moves:

    * **An earlier lifecycle arrives.** A number change we had only the ``to``
      half of gains its ``from`` half, or a carrier backfills a stretch of
      history that predates everything we held. The chain grows a new head, and
      every account row in it re-keys.
    * **Two chains merge.** Two accounts we believed separate turn out to be
      joined - the counterparty of a number change lands, or a Rogers IMSI pair
      becomes visible once the device history catches up. One of the two roots
      wins, so at least one of the two chains re-keys, and which one wins is
      decided by the chain, not by either half on its own.
    * **The cycle break moves.** A component containing a cycle is repaired by
      dropping one edge (``chains.py:156-164``). That choice is deterministic
      for a given edge set, but it is a function of the *whole* set: adding or
      dropping any edge in the cycle can move the break, which moves the root,
      which re-keys the chain.

    On a full rebuild none of this is visible, because every row is recomputed
    from the same evidence in the same run and the old keys are never consulted.
    Incremental is the opposite case. The merge key for the account mapping is
    ``("acct_id", "lifecycle_uid")`` (``schemas.py:845``), so a re-keyed account
    does not match its predecessor: the merge takes the INSERT branch, the new
    rows land alongside the old ones, and the old rows are orphaned rather than
    updated - still present, still carrying the previous ``acct_id``, and now
    describing lifecycles that also appear under a second account. Nothing in
    the merge notices, because from its point of view a key it has never seen
    before is exactly what a new account looks like.

    Nothing here is changed by this note. A fix is a key-migration design - a
    stable surrogate with a mapping table, or a merge that resolves the
    predecessor by ``lifecycle_uid`` and retires it - and it belongs to the
    incremental redesign rather than to a hashing helper. What this note exists
    for is that the instability be a known and stated property of the key rather
    than a thing rediscovered from a duplicated table.
    """
    return sha2_cols(F.lit("acct"), F.lit(SCHEMA_VERSION), root_lifecycle_uid)


def customer_id(root_acct_id: Column | str) -> Column:
    """Key for one customer, seeded on the root of its port chain.

    **Unstable in exactly the way :func:`acct_id` is, and then again on its own
    account.** The full argument is on :func:`acct_id`; what follows is what is
    specific to this level.

    This seed inherits the whole of the instability below it. ``root_acct_id``
    is an ``acct_id``, so every one of the three ways an account re-keys - an
    earlier lifecycle arriving, two number-change chains merging, a different
    cycle-break edge - re-keys the customer whose chain that account happens to
    head, even when nothing whatever changed about the ports. A customer can
    therefore be re-keyed by evidence that has nothing to do with porting.

    And it moves again for the same three reasons one level up, with ports in
    place of number changes: an earlier port arriving gives the port chain a new
    head; two port chains merging - which is what a newly visible port between
    two accounts we held separately does - retires one of the two roots; and a
    port cycle, a number that appears to have returned to a carrier it had left,
    is repaired by dropping an edge whose choice depends on the whole component.

    The merge key is ``("customer_id", "acct_id", "lifecycle_uid")``
    (``schemas.py:852``), which contains both unstable keys, so a customer can
    be orphaned by a change at either level. As above: an INSERT rather than an
    UPDATE, the predecessor left in place, and one lifecycle published under two
    customers with no error anywhere. Full rebuilds hide it completely.
    """
    return sha2_cols(F.lit("customer"), F.lit(SCHEMA_VERSION), root_acct_id)


# ==========================================================================
# frame helpers
# ==========================================================================


def with_audit_columns(
    df: DataFrame,
    run_ts: Column,
    *,
    created: str = "created_ts",
    updated: str = "updated_ts",
    schema_version: str | None = "schema_version",
) -> DataFrame:
    """Stamp the bookkeeping columns a recomputed frame is missing.

    ``created_ts`` is set to the run timestamp here and then *preserved* by the
    merge for rows that already exist - the merge's UPDATE clause does not touch
    it. So this value only ever survives on genuinely new rows, which is what
    makes ``created_ts`` mean "first seen" rather than "last recomputed".
    """
    out = df.withColumn(created, run_ts).withColumn(updated, run_ts)
    if schema_version:
        out = out.withColumn(schema_version, F.lit(SCHEMA_VERSION).cast("int"))
    return out


def align_to_schema(df: DataFrame, schema) -> DataFrame:
    """Project ``df`` onto ``schema``'s columns, in order, with its types.

    A merge against a table whose columns are in a different order is either a
    silent mis-assignment or an error depending on the engine, so every frame
    passes through here before it reaches a sink. A missing column is an error
    rather than a null-filled column: at this point in the pipeline a missing
    column means a transform forgot something.
    """
    have = set(df.columns)
    missing = [f.name for f in schema.fields if f.name not in have]
    if missing:
        raise ValueError(
            f"frame is missing required column(s) {missing}; "
            f"has {sorted(have)}"
        )
    return df.select(
        *[F.col(f.name).cast(f.dataType).alias(f.name) for f in schema.fields]
    )


def event_order_window(
    partition_cols: Sequence[str],
    ts_col: str = "event_timestamp",
    tie_break_col: str | None = "record_id",
) -> Window:
    """The canonical event ordering: by timestamp, then by a stable tie-break.

    Every walk in this pipeline - lifecycles, accounts, device segments - orders
    events this way. Without the tie-break, two events in the same second order
    non-deterministically and the same input produces different Gold on
    different runs, which would make the convergence argument in section 10
    false.
    """
    order = [F.col(ts_col).asc()]
    if tie_break_col:
        order.append(F.col(tie_break_col).asc())
    return Window.partitionBy(*[F.col(c) for c in partition_cols]).orderBy(*order)


def join_sorted_ids(col: Column) -> Column:
    """Collect an id column into a sorted, comma-joined string.

    Sorted so the value is stable: an unsorted ``collect_list`` would produce a
    different string on a different shuffle and every row would look changed to
    the merge.
    """
    return F.concat_ws(",", F.array_sort(F.array_distinct(F.collect_list(col))))


def days_between(later: Column | str, earlier: Column | str) -> Column:
    """Whole and fractional days from ``earlier`` to ``later``.

    Computed on the timestamps rather than with ``datediff``, which counts
    calendar-day boundaries: two events 26 hours apart across midnight are
    ``datediff`` 2 and ``days_between`` 1.08. Every window in this pipeline
    means elapsed time, not calendar days.
    """
    lhs = F.col(later) if isinstance(later, str) else later
    rhs = F.col(earlier) if isinstance(earlier, str) else earlier
    return (lhs.cast("double") - rhs.cast("double")) / F.lit(86400.0)


def hours_between(later: Column | str, earlier: Column | str) -> Column:
    """Whole and fractional hours from ``earlier`` to ``later``."""
    lhs = F.col(later) if isinstance(later, str) else later
    rhs = F.col(earlier) if isinstance(earlier, str) else earlier
    return (lhs.cast("double") - rhs.cast("double")) / F.lit(3600.0)
