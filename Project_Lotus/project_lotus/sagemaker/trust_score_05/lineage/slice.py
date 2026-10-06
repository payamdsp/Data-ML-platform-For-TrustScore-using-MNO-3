"""Building the slice: which phone numbers a batch can possibly change.

A batch of Silver rows names some phone numbers directly. It does not name the
phone numbers those drag in with them, and those are where the interesting bugs
live. A number change binds two phone numbers into one account, so touching one
end reshapes the other. A port binds two accounts into one customer. Rebuilding
only the phone numbers that literally appear in the batch would leave the other
end of every such relationship stale, and stale in a way no check downstream
would notice.

So the batch's phone numbers are *closed* under those relationships before
anything is recomputed. The closure runs in levels and then iterates to a fixed
point:

    L0   phone numbers appearing in the batch, plus the otherMSISDN
         counterparty of every number-change event in it
    L1   every phone number sharing an account with anything in the slice,
         from the existing account mapping, plus both sides of number-change
         pairs the batch is discovering right now (including Rogers pairs,
         which are inferred from the device feed)
    L1c  every phone number sharing a *correlation id* with anything in the
         slice, from the published canonical events
    L2   every phone number sharing a customer with anything in the slice
    L3   every phone number TransUnion's porting history links to anything in
         the slice

L1, L1c, L2 and L3 repeat until the slice stops growing. In practice the first
iteration finds everything and the second finds nothing; the cap exists so a
pathological component - a device shared by thousands of numbers, a mis-hashed
phone number that joins to everything - degrades into a logged failure rather
than a run that quietly reads the whole table.

Each of the three account routes has to be represented here, and until August
2026 one of them was not. The account stage joins lifecycles into a chain by
three routes - a shared ``correlationId``, a named ``otherMSISDN``
counterparty, and a shared SIM - and the closure covered the second (L0 and the
batch-edge pass) and the third (the Rogers probe) and not the first. A carrier
that stamps both halves of a number change with one correlation id and names no
counterparty therefore had a route into an account that the closure could not
follow, and the consequence is the quietest failure this pipeline has: the run
rebuilds the half it can see, finds no partner because the partner's events were
never read, and publishes two single-lifecycle accounts where one two-lifecycle
account belongs. Nothing is missing and nothing is null; the answer is simply
wrong, it is *stably* wrong, and re-running produces it again. See
:func:`_l1_correlation_partners`.

The batch is not only the two carrier feeds. TransUnion's porting history is
watermarked on ``ingestion_ts`` rather than on when the port happened, because a
port from 2019 delivered today is new evidence today, and a phone number whose
only new evidence is a TU delivery has to be rebuilt or that delivery is read by
nothing. So TU names phone numbers at L0 alongside the carrier feeds, and it is
also the only feed that does so without the number appearing in a carrier batch
at all - TU is fetched per fraud record, and the fraud records are not ours.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .config import Config
from .io.reader import SliceScope, read_gold
from .logging_utils import get_logger
from .schemas import SLICE_KEY
from .transforms.canonical_events import rogers_imsi_pairs
from .transforms.customers import tu_port_edges

__all__ = [
    "SliceResult",
    "build_slice",
    "NUMBER_CHANGE_EVENT_TYPES",
    "MIGRATING_TABLES",
]

#: Tables whose merge key can change for a row that has not changed.
#:
#: ``acct_id`` is a hash of the chain's root lifecycle and ``customer_id`` a
#: hash of the chain's root account, so both move whenever new evidence
#: lengthens or merges a chain - see ``transforms.common.acct_id``. The row is
#: the same row; its key is not. That makes the merge take the INSERT branch
#: instead of UPDATE, and the predecessor survives only because the merge's
#: delete branch is scoped to the *slice* rather than to the source: an old
#: ``acct_id`` for a phone number in the slice is ``NOT MATCHED BY SOURCE`` and
#: is deleted.
#:
#: That is why the closure in this module is load-bearing rather than an
#: optimisation, and why every phone number of an affected account and customer
#: has to be in the slice. A phone number left out keeps its rows under the old
#: key while its chain-mates move to the new one, and the account is then split
#: across two ``acct_id`` values with nothing anywhere recording that they are
#: the same account.
#:
#: Named here so the incremental design doc, the merge sink's delete ceiling and
#: the DQ orphan checks can all refer to one list.
MIGRATING_TABLES = (
    "msisdn_lifecycle_account_mapping",
    "msisdn_lifecycle_account_customer_mapping",
)

LOGGER = get_logger(__name__)

#: Silver event types on the account-changes feed that carry a counterparty.
NUMBER_CHANGE_EVENT_TYPES = ("MSISDN_CHANGE", "MSISDN_CHANGE_FROM")


@dataclass
class SliceResult:
    """The slice, plus enough detail to explain how it got that big."""

    scope: SliceScope
    l0_size: int
    iterations: int
    converged: bool
    sizes_by_iteration: list[int] = field(default_factory=list)
    #: Phone numbers L0 got from the TransUnion delivery. Reported separately
    #: because it is the one number that says whether enrichment is reaching the
    #: rebuild at all: a run whose TU feed is configured, enabled and silently
    #: empty looks exactly like a run with no TU feed except for this counter.
    tu_l0_size: int = 0

    @property
    def size(self) -> int:
        return self.scope.size

    def format_line(self) -> str:
        path = " -> ".join(str(s) for s in self.sizes_by_iteration)
        state = "converged" if self.converged else "TRUNCATED at max_iterations"
        tu = f" tu_l0={self.tu_l0_size}" if self.tu_l0_size else ""
        return (
            f"slice: L0={self.l0_size}{tu} final={self.size} "
            f"({path}) after {self.iterations} iteration(s), {state}"
        )


# ==========================================================================
# level 0
# ==========================================================================


def _l0_keys(
    account_changes_batch: DataFrame | None,
    device_lookup_batch: DataFrame | None,
    tu_batch: DataFrame | None = None,
) -> DataFrame:
    """Phone numbers named directly by the batch, plus number-change counterparties.

    ``otherMSISDN`` is the cheapest edge in the whole closure: the carrier hands
    us both ends of a number change in one row, so the counterparty needs no
    join at all. The previous code ignored this column and matched only on
    ``correlationId``, which is why Rogers - which sends neither - produced no
    account chains whatsoever.

    ``tu_batch`` is the TransUnion porting history delivered in this run's
    window, and it belongs here rather than at a later level because a TU
    delivery is *new evidence about a phone number* in exactly the way a carrier
    event is. It is also the one feed that names numbers no carrier batch
    mentions: TU is fetched per fraud record, so a number can go years without a
    carrier event and still gain a complete porting history overnight. Left out
    of L0, that history would sit in Silver unread until the number happened to
    appear in a carrier batch, which for a cancelled number is never.
    """
    frames: list[DataFrame] = []
    if account_changes_batch is not None:
        frames.append(account_changes_batch.select(F.col(SLICE_KEY)))
        frames.append(
            account_changes_batch.where(
                F.col("event_type").isin(list(NUMBER_CHANGE_EVENT_TYPES))
                & F.col("otherMSISDN").isNotNull()
                & (F.trim(F.col("otherMSISDN")) != F.lit(""))
            ).select(F.col("otherMSISDN").alias(SLICE_KEY))
        )
    if device_lookup_batch is not None:
        frames.append(device_lookup_batch.select(F.col(SLICE_KEY)))
    if tu_batch is not None:
        frames.append(tu_batch.select(F.col(SLICE_KEY)))

    if not frames:
        raise ValueError("build_slice() needs at least one batch frame")

    out = frames[0]
    for frame in frames[1:]:
        out = out.unionByName(frame)
    return out.where(F.col(SLICE_KEY).isNotNull()).distinct()


# ==========================================================================
# level 1 - number-change closure
# ==========================================================================


def _l1_existing_accounts(
    spark: SparkSession, cfg: Config, keys: DataFrame
) -> DataFrame:
    """Every phone number sharing an account with something already in the slice.

    Two joins against the account mapping rather than one: first find the
    accounts the slice's phone numbers belong to, then find every phone number
    in those accounts. That captures the chain in full, not merely the immediate
    neighbour, which matters because a chain of three number changes has an end
    two hops away from the number the batch touched.
    """
    mapping = read_gold(spark, cfg, "msisdn_lifecycle_account_mapping").select(
        "acct_id", SLICE_KEY
    )
    accounts = mapping.join(keys, SLICE_KEY, "left_semi").select("acct_id").distinct()
    return mapping.join(accounts, "acct_id", "left_semi").select(SLICE_KEY).distinct()


def _l1_batch_number_changes(account_changes: DataFrame | None) -> DataFrame | None:
    """Both sides of every number change the batch itself is discovering.

    The existing account mapping cannot help here: the pair being formed right
    now may involve a phone number Gold has never seen. Those edges come from
    the batch.
    """
    if account_changes is None:
        return None
    # The same guard L0 applies. A carrier that sends the column but leaves it
    # blank would otherwise put the empty string in the slice, and an empty
    # string is a phone number the merge would then read and delete rows for.
    pairs = account_changes.where(
        F.col("event_type").isin(list(NUMBER_CHANGE_EVENT_TYPES))
        & F.col("otherMSISDN").isNotNull()
        & (F.trim(F.col("otherMSISDN")) != F.lit(""))
    )
    return (
        pairs.select(F.col(SLICE_KEY))
        .unionByName(pairs.select(F.col("otherMSISDN").alias(SLICE_KEY)))
        .distinct()
    )


def _l1_correlation_partners(
    spark: SparkSession, cfg: Config, keys: DataFrame
) -> DataFrame:
    """Every phone number sharing a correlation id with something in the slice.

    The third of the account stage's three routes, and the last one to get a
    closure level. ``_edges_by_correlation`` joins two *different* phone numbers
    that carry the same ``correlationId`` at the same carrier
    (``accounts.py:_edges_by_correlation``), so an edge exists only where both
    ends are in the frame that stage is given - and that frame is the slice.
    A slice holding one end of a correlated pair therefore rebuilds an account
    that is missing its other half, and does so silently: the join simply
    returns nothing, no route is recorded, and both lifecycles publish as
    ``account_type = 'single'`` with ``link_route = 'none'``, which is exactly
    what a phone number that genuinely never changed its number looks like.

    Read off the **published canonical events**, not off Silver, and the two
    cases that split on are worth being explicit about.

    * The partner's event is in *this* batch. Then the partner's phone number is
      already at L0, because L0 takes every phone number the batch names. This
      level would find it too and finds nothing new.
    * The partner's event was processed by an *earlier* run. Then it is in the
      canonical events table and nowhere else this closure looks, and this is
      the case the level exists for.

    That leaves one case neither covers: an event that is in Silver, is not in
    this batch, and was never processed into Gold - which is a run that failed
    between reading Silver and writing Gold. That is a broken estate rather than
    an incremental-update concern, and the repair for it is a re-run over the
    window that failed, not a wider closure. Reading Silver here instead would
    cover it, at the price of two scans of the largest table in the estate per
    closure iteration; the canonical events table is deduplicated, nine columns
    wide, and the one this run is about to read anyway.

    Blank ids are excluded for the reason ``_edges_by_correlation`` excludes
    them: Silver preserves what the carrier sent, and one blank correlation id
    shared across a day of traffic would put the whole day in the slice.
    """
    events = read_gold(spark, cfg, "account_changes_canonical_events").select(
        SLICE_KEY, "mno", "correlation_id"
    )
    usable = events.where(
        F.col("correlation_id").isNotNull()
        & (F.trim(F.col("correlation_id")) != F.lit(""))
    )
    # Keyed on ``(mno, correlation_id)`` and not on the id alone, because the
    # edge builder requires the carrier to match too. Two carriers reusing an id
    # is not a pair, and joining on the id alone would drag an unrelated phone
    # number in on a collision - harmless to correctness, since the account
    # stage would still refuse the edge, but it is a needless way to make a
    # slice grow.
    mine = usable.join(keys, SLICE_KEY, "left_semi").select(
        "mno", "correlation_id"
    ).distinct()
    return (
        usable.join(mine, ["mno", "correlation_id"], "left_semi")
        .select(SLICE_KEY)
        .distinct()
    )


def _l1_rogers_imsi_pairs(
    device_history: DataFrame | None, max_gap_days: int | None
) -> DataFrame | None:
    """Rogers number-change pairs, inferred from one SIM card carrying two numbers.

    Rogers sends no number-change event at all, so the only evidence is the
    device feed: a SIM card that carried phone number A and then phone number B
    is one subscriber changing their number, because the subscriber keeps their
    SIM across the change. This probe is what puts B in the slice when the batch
    only mentions A - without it the pair would be discovered during
    recomputation and half of it would be out of scope, which is the worst of
    both worlds.

    It calls the account stage's own pairing rather than restating it, and that
    is the point of the function existing at all. If the closure paired on one
    thing and the account stage paired on another, a run could build an account
    edge to a phone number whose history it never read, and rebuild the account
    from half its evidence - which converges on a *stable wrong answer*, the one
    failure mode that re-running never reveals. Two definitions of one rule can
    drift; one cannot.
    """
    if device_history is None:
        return None
    paired = rogers_imsi_pairs(device_history, max_gap_days)
    return (
        paired.select(F.col("a_phone").alias(SLICE_KEY))
        .unionByName(paired.select(F.col("b_phone").alias(SLICE_KEY)))
        .distinct()
    )


# ==========================================================================
# level 2 - port closure
# ==========================================================================


def _l2_existing_customers(
    spark: SparkSession, cfg: Config, keys: DataFrame
) -> DataFrame:
    """Every phone number sharing a customer with something already in the slice."""
    mapping = read_gold(
        spark, cfg, "msisdn_lifecycle_account_customer_mapping"
    ).select("customer_id", SLICE_KEY)
    customers = (
        mapping.join(keys, SLICE_KEY, "left_semi").select("customer_id").distinct()
    )
    return mapping.join(customers, "customer_id", "left_semi").select(SLICE_KEY).distinct()


# L2b - the same phone number at a different MNO - adds no new phone hashes by
# construction, because a port is a same-number-different-MNO relationship. It
# needs no code. It is the reason the slice key is `phone_number_AC_hash` alone
# and never `(phone_number_AC_hash, mno)`: narrowing it that way would read one
# MNO's rows for a phone number and silently miss the other side of its port.


# ==========================================================================
# level 3 - the TransUnion edge
# ==========================================================================


def _l3_tu_port_partners(
    tu_ports: DataFrame | None, keys: DataFrame
) -> DataFrame | None:
    """Every phone number TransUnion's porting history links to the slice.

    L1 reads the account map and L2 reads the customer map, and both of those
    are things this pipeline wrote on a previous run. TU is the third
    relationship and it is different in kind: it exists in the Silver table
    before it exists in either map. If TU reveals that two lifecycles we thought
    were separate are one subscription that spent time at a carrier we have no
    feed from, that is a customer edge no previous run could have recorded, and
    a run that rebuilds the customer from one side of it converges on a stable
    wrong answer - the failure that re-running never reveals.

    The edges come from :func:`~.transforms.customers.tu_port_edges`, which is
    the customer stage's own pairing, called rather than restated. Two
    definitions of one rule drift; one cannot. That is the same argument as
    :func:`_l1_rogers_imsi_pairs`, and the same reason.

    In the TU feed as specified both ends of every edge are the same phone
    number, so this level adds nothing today - porting history is per number and
    TU carries no counterparty. The level is still here rather than deferred,
    for two reasons. It is where the unsupported-stretch relationship arrives
    the day TU carries the evidence for it, and having the closure already read
    the pairing means that day changes one function and not two. And it is
    cheap: a semi-join against a table the run is going to read anyway, over a
    slice that is already bounded.
    """
    if tu_ports is None:
        return None
    edges = tu_port_edges(tu_ports).select("a_phone", "b_phone")
    # Both directions. The pairing states each edge once and the closure has to
    # traverse it from whichever end the batch happened to name.
    both = edges.unionByName(
        edges.select(
            F.col("b_phone").alias("a_phone"), F.col("a_phone").alias("b_phone")
        )
    )
    return (
        both.select(
            F.col("a_phone").alias(SLICE_KEY), F.col("b_phone").alias("_partner")
        )
        .join(keys, SLICE_KEY, "left_semi")
        .select(F.col("_partner").alias(SLICE_KEY))
        .where(F.col(SLICE_KEY).isNotNull())
        .distinct()
    )


# ==========================================================================
# the closure
# ==========================================================================


def build_slice(
    spark: SparkSession,
    cfg: Config,
    account_changes_batch: DataFrame | None,
    device_lookup_batch: DataFrame | None,
    device_history: DataFrame | None = None,
    *,
    tu_batch: DataFrame | None = None,
    tu_ports: DataFrame | None = None,
    close_number_changes: bool = True,
    close_ports: bool = True,
    first_run: bool = False,
) -> SliceResult:
    """Close the batch's phone numbers under accounts, customers and TU ports.

    ``device_history`` is the unwatermarked device feed restricted to the IMEIs
    in the batch; it is optional because the IMEI-map job does not need the
    Rogers number-change probe.

    ``tu_batch`` is the TransUnion porting history delivered in this run's
    window and names phone numbers at L0. ``tu_ports`` is the same table
    unwatermarked, which the closure reads at L3 - unwatermarked because a port
    that binds a slice member to another phone number is evidence whenever it
    was delivered, not only if it was delivered today. Both are ``None`` when
    the feed is disabled or has never landed, and a ``None`` here is the
    pipeline behaving exactly as it did before enrichment existed.

    Set ``close_number_changes`` / ``close_ports`` false for the IMEI-map job,
    whose grain is one phone number and one device and which therefore needs no
    closure at all beyond L0. ``close_ports`` gates L3 as well as L2: both are
    the port relationship, one witnessed by our feeds and one by TU.

    ``first_run`` is an operator assertion, never inferred from an empty control
    table. It relaxes only ``slice.max_size`` for the legitimate all-history
    catch-up; the closure, convergence limit and every downstream guard are
    unchanged.
    """
    max_iterations = int(cfg.get("slice.max_iterations", 5))
    max_size = int(cfg.get("slice.max_size", 5_000_000))
    # Null means no outer bound on the SIM pairing, which is the rule as
    # specified, so this stays ``None`` rather than being coerced to a number.
    rogers_gap = cfg.get("windows.rogers_imsi_reuse_max_gap_days", None)
    rogers_gap = None if rogers_gap is None else int(rogers_gap)
    inline_limit = int(cfg.get("slice.inline_limit", 20_000))

    keys = _l0_keys(account_changes_batch, device_lookup_batch, tu_batch).persist()
    l0_size = keys.count()
    sizes = [l0_size]
    # Counted from the TU frame rather than differenced against the carrier
    # keys: the question this answers is "did TU deliver anything this window",
    # not "did it deliver anything nothing else mentioned". A number TU and the
    # carrier both named still proves the feed is flowing.
    tu_l0_size = (
        0
        if tu_batch is None
        else tu_batch.select(F.col(SLICE_KEY)).distinct().count()
    )
    LOGGER.info(
        "slice L0 | %d phone number(s) named by the batch, %d of them by TU",
        l0_size,
        tu_l0_size,
    )

    if l0_size == 0:
        return SliceResult(
            scope=SliceScope(keys, 0, (), inline_limit),
            l0_size=0,
            iterations=0,
            converged=True,
            sizes_by_iteration=sizes,
            tu_l0_size=tu_l0_size,
        )

    # One-off additions that do not depend on the current slice: both sides of
    # the batch's own number changes, and the Rogers SIM-inferred pairs.
    if close_number_changes:
        for extra in (
            _l1_batch_number_changes(account_changes_batch),
            _l1_rogers_imsi_pairs(device_history, rogers_gap),
        ):
            if extra is not None:
                keys = keys.unionByName(extra).distinct()
        keys = keys.persist()
        size = keys.count()
        if size != sizes[-1]:
            sizes.append(size)
            LOGGER.info("slice L1 (batch edges) | %d phone number(s)", size)

    converged = True
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        grown = keys
        if close_number_changes:
            grown = grown.unionByName(_l1_existing_accounts(spark, cfg, keys))
            grown = grown.unionByName(_l1_correlation_partners(spark, cfg, keys))
        if close_ports:
            grown = grown.unionByName(_l2_existing_customers(spark, cfg, keys))
            tu_partners = _l3_tu_port_partners(tu_ports, keys)
            if tu_partners is not None:
                grown = grown.unionByName(tu_partners)
        grown = grown.distinct().persist()
        size = grown.count()

        if size > max_size:
            if first_run:
                LOGGER.warning(
                    "FIRST-RUN MODE: slice grew to %d phone numbers, above the "
                    "steady-state slice.max_size=%d; continuing because the "
                    "operator asserted that the initial all-history slice is expected",
                    size,
                    max_size,
                )
            else:
                raise RuntimeError(
                    f"slice grew to {size} phone numbers, above slice.max_size="
                    f"{max_size}. This usually means a phone hash joined to a very "
                    "large component - check for a mis-hashed value or a device "
                    "shared by thousands of numbers. Narrow the batch with "
                    "--watermark-to, or raise the cap deliberately."
                )

        keys.unpersist()
        keys = grown
        sizes.append(size)
        LOGGER.info("slice iteration %d | %d phone number(s)", iteration, size)

        if size == sizes[-2]:
            break
    else:
        converged = False
        LOGGER.warning(
            "slice did not converge within slice.max_iterations=%d; "
            "proceeding with %d phone numbers. Rows outside this set that are "
            "linked to it will be stale until the next run picks them up.",
            max_iterations,
            sizes[-1],
        )

    scope = SliceScope.from_frame(keys, inline_limit=inline_limit)
    result = SliceResult(
        scope=scope,
        l0_size=l0_size,
        iterations=iterations,
        converged=converged,
        sizes_by_iteration=sizes,
        tu_l0_size=tu_l0_size,
    )
    LOGGER.info(result.format_line())
    return result
