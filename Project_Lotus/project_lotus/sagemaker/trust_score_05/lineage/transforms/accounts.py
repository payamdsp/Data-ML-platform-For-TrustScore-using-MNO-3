"""Stage 3: lifecycles -> accounts.

An account is a chain of lifecycles at one carrier, joined by number changes.
The subscriber is the same person throughout; only their phone number changed.
Lifecycle A ends on a ``phone_number_change_from`` and lifecycle B opens on a
``phone_number_change_to``, and the job of this stage is to decide that those two
events are two halves of one transition.

Three routes propose that pairing, and this is where the largest single
correction in the pipeline lives. The previous code had only the first of them,
and it required a non-null correlation id on *both* halves - so ROGERS, which
sends neither a number-change event nor a correlation id, produced no account
chains at all. Every ROGERS subscriber who ever changed their number was
recorded as two unrelated accounts.

    correlation_id  the carrier says outright that these two rows are two
                    halves of one change. TELUS and BELL send it. Exact, no
                    tolerance.

    counterparty    the ``otherMSISDN`` column names the other phone number
                    directly. It needs no correlation id and it is the route
                    that recovers most of what the first one missed.

    imsi            for ROGERS, one SIM card carrying two phone numbers. The
                    SIM is the subscription, so a subscriber who changes their
                    number keeps it; the handset they may well have sold.
                    Derived in stage 1; re-derived here to recover which two
                    events came from which pair, and to carry how long the two
                    numbers overlapped on the one SIM.

Once the edges exist the rest is traversal, and that is :mod:`chains` - shared
with the customer stage, because a chain of lifecycles joined by number changes
and a chain of accounts joined by ports are the same problem one level apart.

That traversal runs on the executors, one connected component to a task, and it
did not always: until 20 August 2026 the driver collected every edge in the
estate and walked the lot in one Python process. What that cost is in
:func:`_resolve_positions`, which is also where the reason a component is a safe
unit is written down.
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
from ..schemas import ACCOUNT_MAPPING_SCHEMA, SLICE_KEY
from ..spark import shuffle_partitions
from .canonical_events import rogers_imsi_pairs
from .chains import build_chains
from .common import (
    MAX_COLLECTED_EDGES,
    acct_id,
    align_to_schema,
    collect_edge_rows,
    component_labels,
    describe_components,
    days_between,
    windows_from_config,
    with_audit_columns,
)

__all__ = [
    "ACCOUNT_WORK_SCHEMA",
    "AccountBuildReport",
    "EDGE_ROUTES",
    "IMSI_LINK_MIN_CONFIDENCE_KEY",
    "NO_LINK_ROUTE",
    "TRUNCATED_ACCOUNT_TYPES",
    "TRUNCATED_FRAGMENT",
    "TRUNCATED_SEGMENT_TYPES",
    "TRUNCATED_FINAL",
    "build_account_edges",
    "build_accounts",
    "imsi_link_min_confidence",
    "link_confidence_for",
    "to_gold",
]

LOGGER = get_logger(__name__)

#: Route name -> precedence. Lower wins when two routes disagree about which
#: lifecycle a number change led to. Precision is the ordering: a correlation id
#: is the carrier's own statement, a counterparty is a named phone number, and a
#: shared SIM card is an inference - a good one, but still ours rather than the
#: carrier's, so it yields to anything the carrier said outright.
EDGE_ROUTES: Mapping[str, int] = {
    "correlation_id": 1,
    "counterparty": 2,
    "imsi": 3,
}

#: ``link_route`` for the first lifecycle of a chain. It was joined to nothing,
#: so there is no route - and no claim of sameness that could be wrong, which is
#: why it is the one route that is always ``HIGH``.
NO_LINK_ROUTE = "none"

_HIGH = "HIGH"
_LOW = "LOW"

#: ``segment_type`` and ``account_type`` for a lifecycle the depth-capped walk
#: abandoned - :attr:`~.chains.ChainPosition.truncated_tail`.
#:
#: It exists because the alternative was published for real. On the 24 August
#: 2026 rebuild 1,596 such lifecycles came out as ``segment_type = 'single'`` and
#: ``account_type = 'single'``, across 38 truncated chains averaging ~92
#: lifecycles apiece, because ``chain_length == 1`` was the only thing the walk
#: said about them. Every one of those rows was demonstrably mid-chain, and a
#: consumer selecting ``account_type = 'single'`` - which is how you ask for
#: subscribers who never changed their number - picked all 1,596 of them up
#: without a hint that anything was wrong. ``acct_link_confidence_min`` was
#: correctly ``LOW`` on the same rows, so the row contradicted itself; the fix is
#: for the row to say one thing.
TRUNCATED_FRAGMENT = "truncated_fragment"

#: ``segment_type`` for the last lifecycle of a chain the depth cap cut - the
#: 50th node on the default ``max_chain_depth``.
#:
#: ``final`` says the account ends here and the walk saw it end. On 38 chains of
#: the 24 August 2026 rebuild it said that about a lifecycle whose successor the
#: walk had just refused to visit. ``final_unmatched`` is not the right word for
#: it either: that one means the lifecycle is still open, which is a statement
#: about the subscriber, whereas this is a statement about our traversal giving
#: up. Hence a third token, and it wins over both of the others below.
TRUNCATED_FINAL = "final_truncated"

#: The two additions to ``SEGMENT_TYPE_DOMAIN`` / ``ACCOUNT_TYPE_DOMAIN``, named
#: here so the DQ vocabularies can be extended from one place rather than each
#: repeating the literals. The domains themselves live in :mod:`..schemas`
#: alongside the table definitions they describe.
TRUNCATED_SEGMENT_TYPES = (TRUNCATED_FRAGMENT, TRUNCATED_FINAL)
TRUNCATED_ACCOUNT_TYPES = (TRUNCATED_FRAGMENT,)

#: Config key for the gate described in :func:`imsi_link_min_confidence`.
IMSI_LINK_MIN_CONFIDENCE_KEY = "accounts.imsi_link_min_confidence"


def imsi_link_min_confidence(cfg=None, override: str | None = None) -> str:
    """How good a SIM link has to be before it is allowed to merge two accounts.

    ``LOW`` - the default, and what every run before this key existed did - lets
    every SIM link through and leaves the grading to ``link_confidence`` and
    ``acct_link_confidence_min``. ``HIGH`` drops the SIM edges that
    :func:`link_confidence_for` grades ``LOW`` before the traversal ever sees
    them, so those lifecycles stay in separate accounts.

    The key exists because until August 2026 the choice was not a choice. Nothing
    anywhere in the transforms filtered on ``link_confidence`` - the audit
    grepped every use of it and found a ``coalesce`` and an ``isNotNull`` - so
    the ~975 K ``LOW`` SIM links of the 24 August 2026 rebuild, 90% of the
    1,082,848 the route produced, merged accounts regardless. That is the
    intended contract and it is still the default; what was missing is that
    reversing it needed a code change rather than a config line.

    Deliberately not a member of the ``windows`` block. Every window there is a
    business rule that changes what a Gold column *means*; this is a policy
    switch about which evidence is admissible, and it is the sort of thing an
    incident wants moved for one run without editing the rulebook.
    """
    value = override
    if value is None and cfg is not None:
        value = cfg.get(IMSI_LINK_MIN_CONFIDENCE_KEY, None)
    if value is None:
        return _LOW
    value = str(value).strip().upper()
    if value not in (_HIGH, _LOW):
        raise ValueError(
            f"{IMSI_LINK_MIN_CONFIDENCE_KEY} must be 'HIGH' or 'LOW', got {value!r}. "
            "There is no MEDIUM anywhere in this pipeline - a routing decision is "
            "binary"
        )
    return value


def link_confidence_for(
    route: str | None, overlap_days: float | None, overlap_tolerance_days: float
) -> str:
    """How sound the claim is that two lifecycles are one account.

    The two carrier routes are the carrier's own statement - a shared
    correlation id, or a named counterparty phone number - and there is nothing
    left to doubt, so both are ``HIGH``.

    The SIM route is ours rather than the carrier's, and it is judged on
    ``overlap_days``: how long both phone numbers were live on the one SIM. A
    small overlap is a clock artefact, because carrier feeds are not
    transactional and the ``IMSI_END`` for the old number routinely lands after
    the ``IMSI_START`` for the new one. A large one is not a number change at
    all - it is two subscriptions on one SIM, or a feed defect - and either way
    the claim should be marked down rather than made silently.

    The tolerance is inclusive: an overlap of exactly the tolerance is ``HIGH``.
    It is a tolerance rather than a threshold, and a tolerance that excluded its
    own boundary would be a strange thing to have chosen on purpose. (The port
    rule in the customer stage is strict, for the opposite reason - it is a
    threshold.)

    **A ``LOW`` returned here still merges the two accounts, and that is the
    contract rather than an oversight.** This function grades a link; it does not
    gate one. The pair selection upstream casts a wide net on purpose -
    :func:`~.canonical_events.rogers_imsi_pairs` admits a successor that started
    after A *started* rather than after A *ended*, precisely so that the overlap
    has something to measure - and this grading is what the wide net is paid for.
    On the 24 August 2026 rebuild that produced 1,082,848 SIM links of which
    ~975 K, 90%, are ``LOW``, every one of them merging two lifecycles into one
    account. A consumer who will not accept that evidence filters on
    ``acct_link_confidence_min``, which is the column that carries the answer for
    the whole chain rather than for one join; ``accounts.imsi_link_min_confidence``
    is the same decision taken at build time instead. Nothing in the transforms
    branches on the value returned here - the August 2026 audit grepped every use
    of ``link_confidence`` and found a ``coalesce`` and an ``isNotNull``.

    Pure Python, no Spark, so the rule can be read and tested on its own.
    """
    if route is None or route == NO_LINK_ROUTE:
        return _HIGH
    if route != "imsi":
        return _HIGH
    if overlap_days is None:
        # The SIM route always measures the overlap - zero when there was none.
        # A null means the measurement is missing rather than absent, and an
        # unmeasured link is not one to vouch for.
        return _LOW
    return _HIGH if overlap_days <= overlap_tolerance_days else _LOW

#: Silver event types on the account-changes feed that carry a counterparty.
_NUMBER_CHANGE_SILVER_TYPES = ("MSISDN_CHANGE", "MSISDN_CHANGE_FROM")

_EDGE_COLUMNS = (
    "from_lifecycle_uid",
    "to_lifecycle_uid",
    "from_event_id",
    "to_event_id",
    "from_event_ts",
    "to_event_ts",
    "mno",
    "route",
    # How long the two phone numbers were live at once. Only the SIM route can
    # measure it - the other two are told the pairing rather than inferring it,
    # so there is nothing to be uncertain about and the column is null for them.
    # Null and zero say different things here and neither may be read as the
    # other: zero is "we looked and they did not overlap".
    "overlap_days",
)

#: What the customer stage and the DQ checks need on top of the published
#: columns: where the account came from, and whether the traversal that produced
#: it was complete.
ACCOUNT_WORK_SCHEMA = StructType(
    [
        StructField("acct_id", StringType(), False),
        StructField("phone_number_AC_hash", StringType(), False),
        StructField("mno", StringType(), False),
        StructField("lifecycle_uid", StringType(), False),
        StructField("source_event_id", StringType(), True),
        StructField("from_ts", TimestampType(), False),
        StructField("to_ts", TimestampType(), True),
        StructField("seq_order", IntegerType(), False),
        StructField("segment_type", StringType(), False),
        StructField("account_type", StringType(), False),
        StructField("link_route", StringType(), False),
        StructField("link_confidence", StringType(), False),
        StructField("acct_link_confidence_min", StringType(), False),
        # downstream-only
        StructField("root_lifecycle_uid", StringType(), False),
        StructField("chain_length", IntegerType(), False),
        StructField("chain_truncated", IntegerType(), False),
        StructField("chain_repaired", IntegerType(), False),
    ]
)


@dataclass
class AccountBuildReport:
    """What the traversal did, in the terms an operator would ask about."""

    lifecycle_count: int = 0
    account_count: int = 0
    edges_by_route: dict[str, int] = field(default_factory=dict)
    broken_cycle_edges: list[tuple[Any, Any]] = field(default_factory=list)
    truncated_roots: list[Any] = field(default_factory=list)
    dropped_edges: list[tuple[Any, Any]] = field(default_factory=list)
    #: Lifecycles stranded past a depth-cap cut, as a count rather than a list.
    #:
    #: The other three repairs are quoted as pairs because an operator chasing
    #: one wants to know which two lifecycles it was. This one is a population,
    #: not a pathology to look at individually: 1,596 of them on the 24 August
    #: 2026 rebuild, and the useful thing about them is the arithmetic, not the
    #: identities. It is the term that was missing from the funnel - edges minus
    #: dropped minus broken minus this lands on the kept-edge count - and its
    #: absence is why the account funnel appeared not to close.
    truncated_tail_lifecycles: int = 0

    @property
    def edge_count(self) -> int:
        return sum(self.edges_by_route.values())

    @property
    def clean(self) -> bool:
        return not (
            self.broken_cycle_edges
            or self.truncated_roots
            or self.dropped_edges
            or self.truncated_tail_lifecycles
        )

    def format_line(self) -> str:
        routes = ", ".join(
            f"{name}={self.edges_by_route.get(name, 0)}" for name in EDGE_ROUTES
        )
        return (
            f"accounts: {self.lifecycle_count} lifecycle(s) -> "
            f"{self.account_count} account(s) over {self.edge_count} edge(s) "
            f"({routes}); cycles_broken={len(self.broken_cycle_edges)} "
            f"chains_truncated={len(self.truncated_roots)} "
            f"edges_dropped={len(self.dropped_edges)} "
            f"truncated_tail={self.truncated_tail_lifecycles}"
        )


# ==========================================================================
# edges
# ==========================================================================


def _transition_events(canonical_events: DataFrame) -> tuple[DataFrame, DataFrame]:
    """The two halves of every number change, as canonical events.

    Anything reclassified upstream is absent here by construction, and that is
    deliberate. If stage 1 decided a phone number's arrival at a carrier was a
    port rather than an incoming number change, there is no
    ``phone_number_change_to`` left to match and no edge is built - which is the
    right answer, because a port is the better explanation of the same instant.
    """
    froms = canonical_events.where(
        F.col("ac_event_type") == F.lit("phone_number_change_from")
    ).select(
        F.col("ac_event_id").alias("from_event_id"),
        F.col(SLICE_KEY).alias("from_phone"),
        F.col("mno").alias("from_mno"),
        F.col("ac_event_ts").alias("from_event_ts"),
        F.col("correlation_id").alias("from_corr"),
    )
    tos = canonical_events.where(
        F.col("ac_event_type") == F.lit("phone_number_change_to")
    ).select(
        F.col("ac_event_id").alias("to_event_id"),
        F.col(SLICE_KEY).alias("to_phone"),
        F.col("mno").alias("to_mno"),
        F.col("ac_event_ts").alias("to_event_ts"),
        F.col("correlation_id").alias("to_corr"),
    )
    return froms, tos


def _edges_by_correlation(froms: DataFrame, tos: DataFrame) -> DataFrame:
    """Route 1: the carrier stamped both halves with the same correlation id.

    Blank strings are excluded as well as nulls. Silver preserves what the
    carrier sent, and an empty correlation id sent on two unrelated rows would
    otherwise join them to each other and to every other empty one - a single
    blank value shared across a day's traffic is a cross join.
    """
    usable = lambda c: F.col(c).isNotNull() & (F.trim(F.col(c)) != F.lit(""))  # noqa: E731
    return (
        froms.where(usable("from_corr"))
        .join(
            tos.where(usable("to_corr")),
            (F.col("from_corr") == F.col("to_corr"))
            & (F.col("from_mno") == F.col("to_mno"))
            & (F.col("from_phone") != F.col("to_phone")),
        )
        .select(
            "from_event_id",
            "to_event_id",
            "from_event_ts",
            "to_event_ts",
            F.col("from_mno").alias("mno"),
            F.lit("correlation_id").alias("route"),
            F.lit(None).cast("double").alias("overlap_days"),
        )
    )


def _edges_by_counterparty(
    froms: DataFrame, tos: DataFrame, account_changes: DataFrame, window_days: int
) -> DataFrame:
    """Route 2: ``otherMSISDN`` names the phone number on the other side.

    The direction is read off the event type. ``MSISDN_CHANGE_FROM`` on P naming
    Q means P became Q; ``MSISDN_CHANGE`` on P naming Q means Q became P. Both
    forms appear, sometimes both for the same transition, which is harmless -
    they propose the same edge and it is deduplicated.

    The tolerance exists because the two halves are two Silver rows with two
    timestamps, and carriers do not guarantee they agree to the second. It is
    kept narrow so that a *later*, unrelated number change on the same phone
    number cannot be picked up as the counterparty.
    """
    is_from_side = F.col("event_type") == F.lit("MSISDN_CHANGE_FROM")
    pairs = (
        account_changes.where(
            F.col("event_type").isin(list(_NUMBER_CHANGE_SILVER_TYPES))
            & F.col("otherMSISDN").isNotNull()
            & (F.trim(F.col("otherMSISDN")) != F.lit(""))
        )
        .select(
            F.when(is_from_side, F.col(SLICE_KEY))
            .otherwise(F.col("otherMSISDN"))
            .alias("p_from"),
            F.when(is_from_side, F.col("otherMSISDN"))
            .otherwise(F.col(SLICE_KEY))
            .alias("p_to"),
            F.upper(F.col("mno")).alias("p_mno"),
            F.col("event_timestamp").alias("p_ts"),
        )
        .where(F.col("p_from") != F.col("p_to"))
        .distinct()
    )

    tolerance = F.lit(float(window_days))
    joined = (
        pairs.join(
            froms,
            (F.col("from_phone") == F.col("p_from"))
            & (F.col("from_mno") == F.col("p_mno"))
            & (F.abs(days_between("from_event_ts", "p_ts")) <= tolerance),
        )
        .join(
            tos,
            (F.col("to_phone") == F.col("p_to"))
            & (F.col("to_mno") == F.col("p_mno"))
            & (F.abs(days_between("to_event_ts", "p_ts")) <= tolerance),
        )
        .withColumn(
            "_distance",
            F.abs(days_between("from_event_ts", "p_ts"))
            + F.abs(days_between("to_event_ts", "p_ts")),
        )
    )

    # One counterparty row describes one transition, so keep the closest
    # candidate pair for it and discard the rest of the window's matches.
    nearest = Window.partitionBy("p_from", "p_to", "p_mno", "p_ts").orderBy(
        F.col("_distance").asc(),
        F.col("from_event_id").asc(),
        F.col("to_event_id").asc(),
    )
    return (
        joined.withColumn("_rn", F.row_number().over(nearest))
        .where(F.col("_rn") == 1)
        .select(
            "from_event_id",
            "to_event_id",
            "from_event_ts",
            "to_event_ts",
            F.col("from_mno").alias("mno"),
            F.lit("counterparty").alias("route"),
            F.lit(None).cast("double").alias("overlap_days"),
        )
    )


def _edges_by_imsi(
    froms: DataFrame,
    tos: DataFrame,
    device_history: DataFrame | None,
) -> DataFrame | None:
    """Route 3: ROGERS, from the SIM card that carried both phone numbers.

    The pair is re-derived rather than carried through stage 1. Stage 1 emits
    the two halves as ordinary canonical events and the published table has no
    column that says which two came from the same SIM, so the alternative would
    be widening the canonical-event schema with a field only this stage reads.

    It re-derives from :func:`rogers_imsi_pairs` - the pairing itself - rather
    than from the canonical events stage 1 projected out of it. Two reasons, and
    the second is the one that matters. The pairing is already one row per pair,
    so the two halves need no re-joining to each other. And it still carries
    ``overlap_days``, which the canonical events do not: the projection drops it
    because a canonical event describes one phone number and an overlap is a
    fact about two. That number is how certain this link is, and it exists
    nowhere else.

    **The re-derivation deliberately applies no ``max_gap_days``, and that is the
    fix for a silent coupling rather than a lost setting.** This function used to
    pass ``windows["rogers_imsi_reuse_max_gap_days"]`` straight through, which
    quietly assumed that the run which built the canonical events used the same
    value as the run reading them - and a rebuild reads a table stage 1 wrote
    days earlier, under whatever config it had. Where this stage's bound was the
    tighter of the two, the pairs it proposed were a subset of the events already
    in the table: lifecycles that stage 1 had *split* on a number change got no
    edge to join them back up, and they vanished through an inner join into two
    unrelated accounts with nothing raised and nothing logged.

    Passing ``None`` makes that impossible instead of merely loud. The bound is
    monotone - it only ever removes the pairs with the *largest* gap, and the
    nearest-successor window keeps the smallest ``b_start`` - so an unbounded
    re-derivation is a superset that agrees with any bounded one wherever the
    bounded one has a row at all. The join against the canonical events then does
    the filtering, which is the right authority: the events are what stage 1
    actually believed, and they are what the lifecycle boundaries were cut on.
    The published numbers do not move, because the configured default is ``null``
    and has been for every run to date.

    The pairing here is wide on purpose and graded afterwards, not filtered.
    :func:`~.canonical_events.rogers_imsi_pairs` admits a successor that began
    after A began rather than after A ended, so two numbers concurrently live on
    one SIM reach this function; ``overlap_days`` is how far they overlapped and
    :func:`link_confidence_for` is what marks the link down for it. The link is
    still made. See :func:`imsi_link_min_confidence` for the switch that refuses
    it instead.
    """
    if device_history is None:
        return None
    pairs = rogers_imsi_pairs(device_history, None)

    return (
        pairs.join(
            froms,
            (F.col("from_phone") == F.col("a_phone"))
            & (F.col("from_event_ts") == F.col("from_ts"))
            & (F.col("from_mno") == F.lit("ROGERS")),
        )
        .join(
            tos,
            (F.col("to_phone") == F.col("b_phone"))
            & (F.col("to_event_ts") == F.col("b_start"))
            & (F.col("to_mno") == F.lit("ROGERS")),
        )
        .select(
            "from_event_id",
            "to_event_id",
            "from_event_ts",
            "to_event_ts",
            F.col("from_mno").alias("mno"),
            F.lit("imsi").alias("route"),
            F.col("overlap_days").cast("double").alias("overlap_days"),
        )
    )


def build_account_edges(
    canonical_events: DataFrame,
    lifecycles: DataFrame,
    account_changes: DataFrame,
    device_history: DataFrame | None,
    windows: Mapping[str, int | None] | None = None,
    cfg=None,
) -> DataFrame:
    """All three routes, resolved down to one edge per lifecycle boundary.

    Returns lifecycle-level edges: ``(from_lifecycle_uid, to_lifecycle_uid, ...)``.
    An event-level edge only becomes a lifecycle edge if the events really are
    the boundaries of two lifecycles - the ``from`` closed one and the ``to``
    opened another. Anything else is a number-change event the lifecycle walk
    did not treat as a boundary, and joining on that would link lifecycles the
    walk never split.
    """
    win = dict(windows) if windows is not None else windows_from_config(cfg)
    froms, tos = _transition_events(canonical_events)

    candidates = [
        _edges_by_correlation(froms, tos),
        _edges_by_counterparty(
            froms, tos, account_changes, int(win["number_change_match_window_days"])
        ),
        # ``rogers_imsi_reuse_max_gap_days`` is deliberately *not* forwarded. It
        # is stage 1's setting and stage 1 has already applied it; re-applying it
        # here from a config that may have moved since is how edges disappeared
        # through an inner join with nothing raised. The reasoning in full is in
        # :func:`_edges_by_imsi`.
        _edges_by_imsi(froms, tos, device_history),
    ]
    edges = None
    for frame in candidates:
        if frame is None:
            continue
        edges = frame if edges is None else edges.unionByName(frame)

    rank = F.create_map(
        *[x for kv in EDGE_ROUTES.items() for x in (F.lit(kv[0]), F.lit(kv[1]))]
    )
    ranked = edges.withColumn("_rank", rank[F.col("route")])

    # The same transition found twice is one edge. Where two routes disagree
    # about the successor, the more precise route wins; the tie-break past that
    # is the earlier transition, so the choice does not depend on shuffle order.
    best = Window.partitionBy("from_event_id").orderBy(
        F.col("_rank").asc(), F.col("to_event_ts").asc(), F.col("to_event_id").asc()
    )
    resolved = (
        ranked.withColumn("_rn", F.row_number().over(best))
        .where(F.col("_rn") == 1)
        .drop("_rn", "_rank")
    )

    closing = lifecycles.where(
        F.col("end_event_type") == F.lit("phone_number_change_from")
    ).select(
        F.col("lifecycle_uid").alias("from_lifecycle_uid"),
        F.col("end_event_id").alias("from_event_id"),
    )
    opening = lifecycles.where(
        F.col("start_event_type") == F.lit("phone_number_change_to")
    ).select(
        F.col("lifecycle_uid").alias("to_lifecycle_uid"),
        F.col("start_event_id").alias("to_event_id"),
    )

    return (
        resolved.join(closing, "from_event_id")
        .join(opening, "to_event_id")
        .where(F.col("from_lifecycle_uid") != F.col("to_lifecycle_uid"))
        .select(*_EDGE_COLUMNS)
    )


# ==========================================================================
# traversal
# ==========================================================================

#: The five columns the walk reads, and the only ones that cross the shuffle.
#: The other four in :data:`_EDGE_COLUMNS` describe the edge for the report and
#: for the DQ checks; carrying them through a shuffle of every number change in
#: the estate would be paying to move columns nothing in the walk looks at.
_WALK_COLUMNS = (
    "from_lifecycle_uid",
    "to_lifecycle_uid",
    "to_event_ts",
    "route",
    "overlap_days",
)

_POSITION = "position"
_CYCLE = "cycle"
_TRUNCATED = "truncated"
_DROPPED = "dropped"

#: Everything one component's walk has to say, in one schema.
#:
#: The same arrangement the customer stage uses, and for the same reasons: four
#: kinds of record told apart by ``kind``, because they come out of one pass and
#: asking for them as four frames would mean four shuffles of the same edges.
#:
#: For a ``position`` record, ``node`` is the lifecycle and ``other`` is the root
#: of its account chain, and the last two columns describe the link that *arrived*
#: at it - null for the first lifecycle of a chain, which was joined to nothing.
#: For a repair, ``node`` and ``other`` are the two ends of the edge, or, for a
#: truncation, the root that was cut and nothing.
_WALK_SCHEMA = StructType(
    [
        StructField("kind", StringType(), False),
        StructField("node", StringType(), False),
        StructField("other", StringType(), True),
        StructField("seq_order", IntegerType(), True),
        StructField("chain_length", IntegerType(), True),
        StructField("chain_truncated", IntegerType(), True),
        StructField("chain_repaired", IntegerType(), True),
        # 1 where this lifecycle is not a chain of one but the debris of a chain
        # the depth cap cut - see :data:`TRUNCATED_FRAGMENT`. It rides the walk
        # schema rather than being reconstructed downstream because it is the one
        # thing only the traversal knows, and it stops at
        # :func:`build_accounts`: ``segment_type`` and ``account_type`` are where
        # it becomes visible, and widening ``ACCOUNT_WORK_SCHEMA`` for it would
        # push a column the customer stage has no use for through another shuffle.
        StructField("chain_truncated_tail", IntegerType(), True),
        StructField("link_route", StringType(), True),
        StructField("link_confidence", StringType(), True),
    ]
)


def _resolve_component(item, max_depth: int, overlap_tolerance_days: float):
    """Walk one connected component of the number-change graph.

    This is the body the driver used to run over every edge in the estate,
    unchanged except for what it is given: one component's edges instead of all
    of them. It is the same body because the walk never read across components.
    Which of two conflicting edges to drop depends only on the edges touching
    those two lifecycles; where to break a cycle depends only on that cycle; a
    root is the start of a chain and a chain lies inside one component. So a
    component is the largest thing the answer can depend on, and the smallest
    thing that has to be in one place.

    The rows arrive in whatever order the shuffle produced them.
    :func:`~.chains.build_chains` sorts what it is given and breaks every tie on
    the values themselves, so that is not a source of drift.
    """
    _component, rows = item
    edge_rows = [tuple(row[1:]) for row in rows]

    edge_list = [(src, dst, ts) for src, dst, ts, _route, _overlap in edge_rows]
    evidence = {
        (src, dst): (route, overlap)
        for src, dst, _ts, route, overlap in edge_rows
    }
    # The tuples above hold references to the same uid strings, so this frees
    # the row list itself rather than the strings in it.
    del edge_rows
    nodes = (n for edge in edge_list for n in edge[:2])
    result = build_chains(nodes, edge_list, max_depth=max_depth)

    # A cycle is a component whose structure the data contradicted, and breaking
    # it published a chain the data did not describe. That is exactly what
    # confidence exists to flag, so the whole repaired chain is marked - not just
    # the two lifecycles either side of the broken edge, because the repair
    # changed where the chain begins and therefore which account every one of
    # them belongs to.
    repaired_roots = {
        result.positions[node].root
        for pair in result.broken_cycle_edges
        for node in pair
        if node in result.positions
    }

    # Keyed on the arriving lifecycle, which has at most one incoming edge left
    # once the conflicts are resolved, so this is one entry per lifecycle and not
    # one per edge. That is also why the link can ride along on the position
    # record instead of arriving as a frame of its own to be joined: one value
    # per node either way.
    link_by_node = {}
    for src, dst in result.kept_edges:
        route, overlap = evidence.get((src, dst), (None, None))
        link_by_node[dst] = (
            route if route is not None else NO_LINK_ROUTE,
            link_confidence_for(route, overlap, overlap_tolerance_days),
        )

    for uid, pos in result.positions.items():
        route, confidence = link_by_node.get(uid, (None, None))
        yield (
            _POSITION,
            uid,
            pos.root,
            pos.seq_order,
            pos.chain_length,
            1 if pos.truncated else 0,
            1 if pos.root in repaired_roots else 0,
            1 if pos.truncated_tail else 0,
            route,
            confidence,
        )
    for src, dst in result.broken_cycle_edges:
        yield (_CYCLE, src, dst, None, None, None, None, None, None, None)
    for root in result.truncated_roots:
        yield (_TRUNCATED, root, None, None, None, None, None, None, None, None)
    for src, dst in result.dropped_edges:
        yield (_DROPPED, src, dst, None, None, None, None, None, None, None)


def _resolve_positions(
    spark: SparkSession,
    edges: DataFrame,
    max_depth: int,
    overlap_tolerance_days: float,
    max_collected_edges: int | None = MAX_COLLECTED_EDGES,
) -> tuple[DataFrame, DataFrame, AccountBuildReport]:
    """Walk the number-change graph one component at a time; hand the positions back.

    Only lifecycles that take part in an edge are walked. Everything else is a
    chain of one by definition and is resolved with a left join and a
    ``coalesce`` downstream, so the work here is bounded by how many number
    changes the history contains and not by how many phone numbers.

    The links come back alongside the positions and only for the edges the
    traversal actually kept. An edge that lost a conflict, or fell past the depth
    cap, joined nothing in the end, so publishing a confidence for it would be
    describing a link that is not there.

    **This ran on the driver until 20 August 2026, and the driver is what
    died.** Twice, in notebook 04: no Spark error, no failed stage, just
    ``Session ... unavailable, fail to call ReplServer`` - which is what a Glue
    session says when the process holding the REPL is gone. On the 20 August run
    the edge list was 3,962,773 rows, and the driver held, at the peak, the edge
    list, an evidence dict keyed on every pair, a ``Position`` object per
    lifecycle, and a list of position tuples on the way into
    ``createDataFrame`` - four copies of the same estate in Python, and then a
    local relation per frame that stayed reachable from the notebook's own
    variable until the cell that used it finished. G.1X gives that driver 16GB
    and it cannot be resized here.

    So the walk moved rather than the guard moving. The edges are labelled by
    connected component in Spark - :func:`~.common.component_labels` - grouped on
    that label, and :func:`_resolve_component` runs on an executor once per
    component. Nothing about the answer changes, because nothing in the walk ever
    read across components; there is a test that asserts the two forms agree row
    for row. It is also the shape that gets the time back: the collect it
    replaces was ``toLocalIterator`` over 2000 partitions, which is 2000
    sequential jobs and most of the twenty-seven minutes that cell spent.

    What still comes to the driver is the repairs - the broken cycles, the
    truncated roots and the edges that lost a conflict - because the report
    quotes the pairs and its consumers expect pairs rather than a count. Those
    are the pathologies rather than the population: 217 broken cycles and 6,268
    dropped edges in the last full run, against four million edges.
    ``max_collected_edges`` bounds that list, which is the only list left that
    the driver has to hold; ``None`` disables it, which is what a test with six
    edges wants.

    The truncation tail is the exception, and it is counted rather than
    collected. It is a population and not a pathology - 1,596 lifecycles behind
    38 cuts on 24 August 2026 - so the report takes the count, which is the term
    that makes ``edges - dropped - broken - tail = kept`` close in the log.
    """
    report = AccountBuildReport()
    # Cached because it is read four times from here - once for the route
    # histogram, twice per labelling round, once to group - and rebuilding it
    # means rebuilding the resolution and both lifecycle joins underneath it
    # every time.
    edges = edges.select(*_WALK_COLUMNS).cache()

    # Counted in Spark. This histogram was the reason the whole edge list came
    # to the driver first, and one row per route is not a reason to move four
    # million rows.
    for row in edges.groupBy("route").count().collect():
        report.edges_by_route[row["route"]] = row["count"]

    # Printed as it is learned rather than only in the closing report line. The
    # report is formatted after the walk, and the walk is the part that takes an
    # hour and the part that dies; a number that is already on the driver is
    # worth more now than it is in a line nobody reaches.
    note(
        "  account edges: "
        + ", ".join(
            f"{route}={count:,}"
            for route, count in sorted(report.edges_by_route.items())
        )
    )

    labels = describe_components(
        component_labels(
            edges, "from_lifecycle_uid", "to_lifecycle_uid", what="account edges"
        ),
        what="account components",
    )
    # Either end of an edge would do: both are in the same component by
    # definition, which is the whole point of having labelled them.
    keyed = edges.join(
        labels.withColumnRenamed("node", "from_lifecycle_uid"),
        "from_lifecycle_uid",
    ).select("component", *_WALK_COLUMNS)

    # One component to a group. The partition count is passed for the reason
    # given in :func:`~trust_score_05.lineage.spark.shuffle_partitions`:
    # ``rdd.groupBy`` does not read ``spark.sql.shuffle.partitions`` on its own,
    # so without it this shuffle is the one the operator cannot tune.
    #
    # A single enormous component would land on one task and one executor, which
    # is the skew this shape can still suffer from. It is strictly better than
    # what it replaces - that component was on the driver before, along with
    # every other one - and if it ever bites, the thing to look at is why the
    # number-change graph has a hub in it, because an account is a chain and a
    # chain does not.
    walked = spark.createDataFrame(
        keyed.rdd.map(tuple)
        .groupBy(lambda row: row[0], shuffle_partitions(spark))
        .flatMap(
            lambda item: _resolve_component(item, max_depth, overlap_tolerance_days)
        ),
        _WALK_SCHEMA,
    ).cache()

    is_position = F.col("kind") == F.lit(_POSITION)
    repairs = collect_edge_rows(
        walked.where(~is_position),
        ("kind", "node", "other"),
        limit=max_collected_edges,
        what="account chain repairs",
    )
    for kind, node, other in repairs:
        if kind == _CYCLE:
            report.broken_cycle_edges.append((node, other))
        elif kind == _TRUNCATED:
            report.truncated_roots.append(node)
        elif kind == _DROPPED:
            report.dropped_edges.append((node, other))

    # Counted in Spark rather than collected. The tail is a population - 1,596
    # lifecycles on the 24 August 2026 rebuild - and what the report needs from
    # it is the number that closes the funnel, not 1,596 uids on the driver. It
    # is one more pass over a frame that is cached and was just materialised by
    # the collect above, which is why it goes here rather than being folded into
    # a per-component summary record that would have to be invented, shuffled and
    # summed.
    report.truncated_tail_lifecycles = walked.where(
        is_position & (F.col("chain_truncated_tail") == F.lit(1))
    ).count()

    positions = walked.where(is_position).select(
        F.col("node").alias("lifecycle_uid"),
        F.col("other").alias("root_lifecycle_uid"),
        "seq_order",
        "chain_length",
        "chain_truncated",
        "chain_repaired",
        "chain_truncated_tail",
    )
    # Split off rather than joined on: a lifecycle with no incoming edge has no
    # link to describe, and a null in a frame that is about to be left-joined
    # would say the same thing as a missing row while costing a row to say it.
    links = walked.where(is_position & F.col("link_route").isNotNull()).select(
        F.col("node").alias("lifecycle_uid"), "link_route", "link_confidence"
    )

    # Read for the last time above: `walked` is cached and materialised by the
    # collect, so the two frames returned no longer need what built it. The
    # labels are not released alongside it because they are checkpointed rather
    # than cached - their blocks go when the frame itself does.
    edges.unpersist()

    return positions, links, report


# ==========================================================================
# the stage
# ==========================================================================


def build_accounts(
    spark: SparkSession,
    canonical_events: DataFrame,
    lifecycles: DataFrame,
    account_changes: DataFrame,
    device_history: DataFrame | None = None,
    windows: Mapping[str, int | None] | None = None,
    cfg=None,
    imsi_min_confidence: str | None = None,
) -> tuple[DataFrame, AccountBuildReport]:
    """Group the slice's lifecycles into accounts.

    ``lifecycles`` may be the work frame from :mod:`lifecycle` or the published
    ``msisdn_lifecycle`` table. Either is enough now that the table carries the
    four boundary columns this stage matches edges on - ``start_event_id``,
    ``start_event_type``, ``end_event_id``, ``end_event_type`` - which it did not
    until 20 August 2026. The incremental job passes the work frame because it
    has one in hand; a rebuild passes the table, because re-running the walk to
    recover four columns already sitting in a table is the most expensive way
    there is to read them.

    Eleven columns are read in total: the four above plus ``lifecycle_uid``, the
    slice key, ``mno``, ``lifecycle_start_ts``, ``lifecycle_end_ts``,
    ``lifecycle_is_open`` and ``is_start_inferred``. A caller passing the
    published table should project to those eleven, because everything else on
    that table shares a name with something this stage builds - ``seq_order``,
    ``link_confidence``, the audit columns - and Spark refuses an ambiguous
    reference rather than resolving it.

    On ``from_ts`` and ``to_ts``: the design document defines them as the
    incoming and outgoing transition timestamps, falling back to the lifecycle's
    own start and end. Those are the same value. An edge is only accepted when
    the ``to`` event *is* the lifecycle's start event and the ``from`` event *is*
    its end event, so the transition timestamp and the lifecycle boundary cannot
    differ. Writing it as the lifecycle boundary is the same rule stated in one
    join fewer.

    ``imsi_min_confidence`` overrides ``accounts.imsi_link_min_confidence`` from
    the config, for a caller holding no ``Config`` object. Both default to the
    behaviour every run before August 2026 had: every SIM link merges, and the
    grading is published alongside it rather than acted on. See
    :func:`imsi_link_min_confidence`.
    """
    win = dict(windows) if windows is not None else windows_from_config(cfg)
    tolerance = float(win["rogers_msisdn_overlap_tolerance_days"])
    edges = build_account_edges(
        canonical_events, lifecycles, account_changes, device_history, windows=win
    )

    # The gate the estate has never used, wired so that turning it on is a config
    # line rather than a patch. It sits here - before the walk, not after it -
    # because a link this rejects must not have decided a root, a seq_order or an
    # acct_id, and by the time the positions come back it has decided all three.
    #
    # The predicate restates :func:`link_confidence_for` rather than calling it,
    # because that one is Python and this is a filter on four million rows. A
    # null ``overlap_days`` is LOW there and is LOW here: the SIM route always
    # measures the overlap, so a null means the measurement went missing rather
    # than that there was nothing to measure.
    min_confidence = imsi_link_min_confidence(cfg, imsi_min_confidence)
    if min_confidence == _HIGH:
        is_imsi = F.col("route") == F.lit("imsi")
        imsi_is_high = F.col("overlap_days").isNotNull() & (
            F.col("overlap_days") <= F.lit(tolerance)
        )
        gated = edges.where(~is_imsi | imsi_is_high)
        LOGGER.warning(
            "accounts: %s=HIGH, so %d SIM link(s) graded LOW on an overlap wider "
            "than %s day(s) were dropped before the walk. Those lifecycles will "
            "publish as separate accounts; the default setting merges them and "
            "marks the link down instead",
            IMSI_LINK_MIN_CONFIDENCE_KEY,
            edges.count() - gated.count(),
            tolerance,
        )
        edges = gated

    positions, links, report = _resolve_positions(
        spark,
        edges,
        int(win["max_chain_depth"]),
        tolerance,
    )

    # No broadcast hint on either side, and that is the fix rather than an
    # omission. A hint is not advice: it makes Spark collect the frame to the
    # driver and push it out to every executor whatever its size, so on a slice
    # where these are a few thousand rows it did the obvious right thing, and on
    # a full history it asked the driver to hold every position in the estate a
    # second time - immediately after the traversal had just finished holding
    # them once. Left to itself Spark still broadcasts them when they are small,
    # because it can see how big a locally-built frame is, and falls back to a
    # sort-merge join when they are not.
    joined = lifecycles.join(positions, "lifecycle_uid", "left").join(
        links, "lifecycle_uid", "left"
    )
    resolved = (
        joined.withColumn(
            "root_lifecycle_uid",
            F.coalesce(F.col("root_lifecycle_uid"), F.col("lifecycle_uid")),
        )
        .withColumn("seq_order", F.coalesce(F.col("seq_order"), F.lit(0)))
        .withColumn("chain_length", F.coalesce(F.col("chain_length"), F.lit(1)))
        .withColumn(
            "chain_truncated", F.coalesce(F.col("chain_truncated"), F.lit(0))
        )
        .withColumn("chain_repaired", F.coalesce(F.col("chain_repaired"), F.lit(0)))
        # A lifecycle that never reached the walk is a chain of one for the
        # honest reason - no number change touched it - so it is not a fragment
        # of anything and the coalesce says zero.
        .withColumn(
            "chain_truncated_tail",
            F.coalesce(F.col("chain_truncated_tail"), F.lit(0)),
        )
        # No incoming edge is not a missing link, it is the start of the chain:
        # nothing was joined, so there is no claim of sameness to be wrong about.
        .withColumn(
            "link_route", F.coalesce(F.col("link_route"), F.lit(NO_LINK_ROUTE))
        )
        .withColumn(
            "link_confidence", F.coalesce(F.col("link_confidence"), F.lit(_HIGH))
        )
    )

    # The three predicates that used to be read straight off ``chain_length`` and
    # ``seq_order``, now qualified by what the traversal actually resolved. The
    # qualification is the whole of finding 2 of the August 2026 audit: a chain
    # the depth cap cut has no known end and its stranded tail has no known
    # beginning, so neither "this is the whole account" nor "this is where the
    # account finishes" is a claim either row is entitled to make.
    is_truncated = F.col("chain_truncated") == F.lit(1)
    is_fragment = F.col("chain_truncated_tail") == F.lit(1)
    is_single = (F.col("chain_length") == F.lit(1)) & ~is_truncated & ~is_fragment
    is_first = (F.col("seq_order") == F.lit(0)) & ~is_fragment
    is_last = (F.col("seq_order") == (F.col("chain_length") - F.lit(1))) & ~is_fragment

    out = (
        resolved.withColumn("acct_id", acct_id(F.col("root_lifecycle_uid")))
        .withColumn("from_ts", F.col("lifecycle_start_ts"))
        .withColumn("to_ts", F.col("lifecycle_end_ts"))
        .withColumn(
            "segment_type",
            # The two truncation tokens are tested first because they describe
            # the traversal and the rest describe the subscriber, and where both
            # have something to say the traversal's admission is the one a
            # consumer needs. A row reading 'final' that is provably not final is
            # worse than a row reading 'final_truncated' that is also open.
            F.when(is_fragment, F.lit(TRUNCATED_FRAGMENT))
            .when(is_last & is_truncated, F.lit(TRUNCATED_FINAL))
            .when(is_single, F.lit("single"))
            # An inferred start on the first link means the chain almost
            # certainly runs back further than the evidence does: something
            # opened this lifecycle and we never saw it. Saying so is more
            # useful than calling it a clean origin.
            .when(
                is_first & (F.col("is_start_inferred") == F.lit(1)),
                F.lit("origin_unmatched"),
            )
            .when(is_first, F.lit("origin"))
            .when(
                is_last & (F.col("lifecycle_is_open") == F.lit(1)),
                F.lit("final_unmatched"),
            )
            .when(is_last, F.lit("final"))
            .otherwise(F.lit("intermediate")),
        )
        .withColumn(
            "account_type",
            # 'chain' would be defensible for a fragment - it is a piece of one -
            # but it is the answer to a question nobody asked. The two values a
            # consumer filters on are 'single' for a subscriber who never changed
            # their number and 'chain' for one who did, and a fragment is neither
            # known to be the first nor known to be the second. Saying so keeps
            # the 1,596 out of both populations rather than out of one of them.
            F.when(is_fragment, F.lit(TRUNCATED_FRAGMENT))
            .when(is_single, F.lit("single"))
            .otherwise(F.lit("chain")),
        )
        # The chain is only as sound as its weakest link. The resolver asks about
        # one lifecycle but reports an account, so a consumer holding a single
        # row needs to know how firm the account it names is without fetching the
        # rest of the chain to find out.
        #
        # The rollup is over the whole chain rather than the path from this
        # lifecycle back to the root. That is deliberately conservative, and it
        # is the cheaper of the two: the traversal already hands back a root per
        # node, so a window gives the answer, whereas a per-path version would
        # need the path.
        .withColumn(
            "_chain_has_low",
            F.max(F.when(F.col("link_confidence") == F.lit(_LOW), 1).otherwise(0)).over(
                Window.partitionBy("root_lifecycle_uid")
            ),
        )
        .withColumn(
            "acct_link_confidence_min",
            # A truncated or repaired chain is LOW whatever its links say: the
            # chain we published is not the chain the data described.
            F.when(
                (F.col("_chain_has_low") == F.lit(1))
                | (F.col("chain_truncated") == F.lit(1))
                | (F.col("chain_repaired") == F.lit(1)),
                F.lit(_LOW),
            ).otherwise(F.lit(_HIGH)),
        )
        # The two events that decided this segment's boundaries, and only those.
        # The previous code wrote every event inside the lifecycle's window into
        # this column - seventeen ids in one sampled row - which made it useless
        # for the one thing it is for, which is tracing a segment back to its
        # evidence.
        .withColumn(
            "source_event_id",
            F.concat_ws(
                ",",
                F.array_sort(
                    F.array_distinct(
                        F.array_compact(
                            F.array(F.col("start_event_id"), F.col("end_event_id"))
                        )
                    )
                ),
            ),
        )
        .select(
            "acct_id",
            SLICE_KEY,
            "mno",
            "lifecycle_uid",
            "source_event_id",
            "from_ts",
            "to_ts",
            "seq_order",
            "segment_type",
            "account_type",
            "link_route",
            "link_confidence",
            "acct_link_confidence_min",
            "root_lifecycle_uid",
            "chain_length",
            "chain_truncated",
            "chain_repaired",
        )
    ).persist()

    report.lifecycle_count = out.count()
    report.account_count = out.select("acct_id").distinct().count()
    LOGGER.info(report.format_line())
    if report.dropped_edges:
        LOGGER.warning(
            "accounts: %d number-change edge(s) dropped as conflicting; "
            "the earliest transition was kept in each case",
            len(report.dropped_edges),
        )
    if report.broken_cycle_edges:
        LOGGER.warning(
            "accounts: %d number-change cycle(s) broken at their latest edge",
            len(report.broken_cycle_edges),
        )
    if report.truncated_roots:
        LOGGER.warning(
            "accounts: %d chain(s) hit max_chain_depth=%d and were emitted "
            "truncated; roots: %s",
            len(report.truncated_roots),
            int(win["max_chain_depth"]),
            ", ".join(str(r) for r in report.truncated_roots[:10]),
        )
    # Logged separately from the roots above because it is a different quantity
    # and it was the missing one. The roots say how many chains were cut; this
    # says how many lifecycles ended up on the far side of the cuts, which is the
    # number that reconciles the edge count and the number of rows that cannot be
    # read as ordinary accounts. 38 roots and 1,596 tail lifecycles on the
    # 24 August 2026 rebuild, and only the first of those two was ever printed.
    if report.truncated_tail_lifecycles:
        LOGGER.warning(
            "accounts: %d lifecycle(s) sat past a max_chain_depth cut and "
            "publish as segment_type=%s rather than as single accounts; their "
            "linkage is unresolved, not absent",
            report.truncated_tail_lifecycles,
            TRUNCATED_FRAGMENT,
        )
    return out, report


def to_gold(accounts: DataFrame, run_ts: datetime) -> DataFrame:
    """Project the work frame onto ``msisdn_lifecycle_account_mapping``."""
    stamped = with_audit_columns(accounts, F.lit(run_ts).cast("timestamp"))
    return align_to_schema(stamped, ACCOUNT_MAPPING_SCHEMA)
