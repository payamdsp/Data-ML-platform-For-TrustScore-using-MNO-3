"""Shared lineage preparation and the backward-compatible single-job driver.

The staged driver calls the same preparation and transformation functions. Its
S3 handoffs preserve the work columns, source snapshots, run clock and slice;
all five outputs still pass DQ together before the publish step starts.

What the run does, in the user's own terms: find the phone numbers with new
events, close that set under the links that could move a boundary, rebuild every
Gold row for those phone numbers from their complete history, and merge. Nothing
here reads the previous Gold rows for the slice in order to modify them; they are
recomputed from the events and replaced. That is what makes a re-run of the same
window converge rather than accumulate.

The slice is the only thing standing between "recompute the phone numbers that
changed" and "recompute the table", so the two frames handed to
:func:`~..slice.build_slice` matter: ``account_changes_batch`` and
``device_lookup_batch`` restricted to the watermark window. Everything read after
the slice is resolved is read *without* a watermark, because a lifecycle boundary
can sit years before the event that moved it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Sequence

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ..dq import (
    MNO_ACTIVATION_INPUT,
    TU_PORTPS_INPUT,
    counter,
    first_seen_check,
    hash_agreement_checks,
    mno_activation_checks,
    modal_hour,
    modal_hour_checks,
    never_seen_check,
    tu_portps_checks,
)
from ..io import (
    read_optional_silver,
    read_silver,
    restrict_to_slice,
    select_batch,
    watermark_column,
)
from ..logging_utils import get_logger
from ..schemas import SLICE_KEY
from ..slice import build_slice
from ..transforms import accounts as accounts_stage
from ..transforms import boundaries as boundary_stage
from ..transforms import customers as customers_stage
from ..transforms import lifecycle as lifecycle_stage
from ..transforms import tu_lifecycle as tu_lifecycle_stage
from ..transforms.canonical_events import build_canonical_events
from ..transforms.common import windows_from_config
from ..transforms.normalized import build_normalized_events
from .base import JobPlan, RunContext, main, read_phone_number_scope, run_gold_job

LOGGER = get_logger(__name__)

JOB_NAME = "gold_account_customer_map"

__all__ = ["JOB_NAME", "plan", "run", "main_cli"]


def _batch_frames(
    ctx: RunContext,
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
    """The batch, and the two histories the closure probes need.

    Seven frames come back: the account-changes batch, the device batch, the
    *complete* history of the SIM cards the batch touched, the TransUnion
    porting history delivered in this window, the complete TU history, and
    the new and complete activation-response frames. The
    SIM one is what lets the slice find a Rogers number change, which is visible
    only as one SIM card carrying two phone numbers - and the earlier of those
    two events is routinely months or years before the batch, because a
    subscriber keeps their SIM for as long as they keep their subscription.

    TU comes back twice for the reason every feed here does: once watermarked,
    because a TU delivery is new evidence about a phone number and that number
    has to be rebuilt, and once unwatermarked, because a port that binds a slice
    member to another number is evidence whenever it arrived. The watermark is
    on ``ingestion_ts`` and not on when the port happened - a port from 2019
    delivered today is new today - and ``watermark_column`` reads that from
    config rather than assuming it, because the IDV feeds watermark on a
    different column and guessing would silently narrow a batch.

    Scoped by ``imsi`` and not by ``imei``, which is the same rule the closure
    and the account stage now pair on. Scoping it by handset would hand the
    closure a frame that can be missing the counterpart phone number's history
    entirely - every row whose ``imei`` is null, and every subscriber who
    changed handsets - and the closure cannot find a pair in evidence it was
    never given. What comes of that is not a crash but an account rebuilt from
    half its evidence, which is the one failure re-running the job never
    reveals.
    """
    spark, cfg = ctx.spark, ctx.cfg
    account_changes = read_silver(spark, cfg, "account_changes_batch")
    device_lookup = read_silver(spark, cfg, "device_lookup_batch")
    tu_ports = read_optional_silver(spark, cfg, "tu_portps")
    activations = read_optional_silver(spark, cfg, "mno_activation")

    explicit = read_phone_number_scope(
        spark,
        ctx.args.phone_numbers,
        inline_limit=int(cfg.get("slice.inline_limit", 20_000)),
    )
    if explicit is not None:
        # Repair mode. The named phone numbers stand in for "what the batch
        # touched"; the closure then runs on top of them exactly as it would on a
        # normal run, so a repair of one phone number still pulls in the account
        # and customer it belongs to.
        LOGGER.info(
            "repair mode | %d phone number(s) named explicitly", explicit.size
        )
        ac_batch = restrict_to_slice(account_changes, explicit)
        dev_batch = restrict_to_slice(device_lookup, explicit)
        # A repair rebuilds the numbers it was given and whatever they drag in,
        # so TU's contribution to L0 is those same numbers and nothing wider.
        # Watermarking here instead would let a repair silently rebuild every
        # number TU happened to deliver for today, which is not what was asked.
        tu_batch = restrict_to_slice(tu_ports, explicit)
        activation_batch = restrict_to_slice(activations, explicit)
    else:
        ac_batch = select_batch(account_changes, ctx.watermark_from, ctx.watermark_to)
        dev_batch = select_batch(device_lookup, ctx.watermark_from, ctx.watermark_to)
        tu_batch = select_batch(
            tu_ports,
            ctx.watermark_from,
            ctx.watermark_to,
            watermark_column(cfg, "tu_portps"),
        )
        activation_batch = select_batch(
            activations, ctx.watermark_from, ctx.watermark_to,
            watermark_column(cfg, "mno_activation"),
        )

    touched_imsis = (
        dev_batch.where(F.col("imsi").isNotNull()).select("imsi").distinct()
    )
    device_by_imsi = device_lookup.join(touched_imsis, "imsi", "left_semi")
    return (
        ac_batch, dev_batch, device_by_imsi, tu_batch, tu_ports,
        activation_batch, activations,
    )


def _first_seen(*frames: DataFrame) -> DataFrame | None:
    """Earliest moment any feed of our own holds for each phone number.

    This is the baseline the "how far back may TransUnion reach" check measures
    against, and it is built from our own feeds only - deliberately. Folding TU
    into it would let a misparsed year move the very baseline that is supposed
    to catch it, and the check would then pass on exactly the row it exists for.
    """
    usable = [
        f.select(SLICE_KEY, "event_timestamp")
        for f in frames
        if f is not None and "event_timestamp" in f.columns
    ]
    if not usable:
        return None
    union = usable[0]
    for frame in usable[1:]:
        union = union.unionByName(frame)
    return (
        union.where(F.col("event_timestamp").isNotNull())
        .groupBy(SLICE_KEY)
        .agg(F.min("event_timestamp").alias("first_seen_ts"))
    )


def _enrichment_input_checks(
    ctx: RunContext,
    tu_batch: DataFrame,
    tu_history: DataFrame,
    activations: DataFrame,
    account_changes: DataFrame,
    device_history: DataFrame,
) -> dict:
    """The checks on the two enrichment feeds, ready for the driver to run.

    Built here rather than in the DQ module because only the job knows which
    frame is the delivery and which is the history, and that distinction is the
    whole content of the modal-hour check: the delivery is measured against the
    history it is joining, with itself removed from that history so it cannot
    drag the baseline towards its own defect.
    """
    cfg = ctx.cfg
    max_after = int(cfg.get("dq.enrichment.tu_event_max_days_after_ingestion", 1))
    max_before = int(
        cfg.get("dq.enrichment.tu_event_max_days_before_first_seen", 7300)
    )
    max_drift = int(cfg.get("dq.enrichment.tu_modal_hour_max_drift_hours", 1))

    first_seen = _first_seen(account_changes, device_history)
    baseline = tu_history.join(tu_batch.select("record_id"), "record_id", "left_anti")

    # The row-level rules are evaluated against the *history*, not against the
    # delivery, because the history is what the stay walk is about to believe. A
    # replay with no new delivery still re-reads every port it ever held, so
    # checking only the delivery would leave the frame the run actually consumes
    # unchecked on exactly the runs where nothing new arrived. Only the
    # modal-hour rule is split across the two, because only that one is asking a
    # question about the delivery rather than about the evidence.
    out: dict = {}
    if tu_history.take(1):
        out[TU_PORTPS_INPUT] = (
            tu_history,
            tu_portps_checks(max_after, ctx.run_ts)
            + modal_hour_checks(
                modal_hour(tu_batch), modal_hour(baseline), max_drift
            )
            + [
                first_seen_check(first_seen, max_before),
                never_seen_check(first_seen),
            ]
            + hash_agreement_checks([device_history, tu_history, activations]),
        )
    if activations.take(1):
        out[MNO_ACTIVATION_INPUT] = (
            activations,
            mno_activation_checks(ctx.run_ts),
        )
    return out


@dataclass
class PreparedLineage:
    """Work frames and evidence shared by the legacy and staged execution paths."""

    sliced: object
    events: DataFrame | None = None
    lifecycles: DataFrame | None = None
    account_changes: DataFrame | None = None
    device_history: DataFrame | None = None
    tu_batch: DataFrame | None = None
    tu_history: DataFrame | None = None
    activations: DataFrame | None = None
    boundary_report: object = None
    tu_report: object = None


def prepare_lineage(ctx: RunContext, materialize=None) -> PreparedLineage:
    """Resolve the slice and build canonical events and enriched lifecycles.

    ``materialize(name, frame)`` is optional. The staged job writes the frame
    to its own S3 attempt directory and returns a fresh Parquet read, cutting
    Spark's upstream plan. The legacy path keeps its original cache behavior.
    No filter, timestamp, event rule or graph rule depends on this callback.
    """
    spark, cfg = ctx.spark, ctx.cfg
    (
        ac_batch, dev_batch, device_by_imsi, tu_batch, tu_ports,
        activation_batch, activation_history,
    ) = _batch_frames(ctx)

    sliced = build_slice(
        spark,
        cfg,
        ac_batch,
        dev_batch,
        device_by_imsi,
        tu_batch=tu_batch,
        tu_ports=tu_ports,
        activation_batch=activation_batch,
        first_run=bool(ctx.args.first_run),
    )
    if sliced.scope.is_empty():
        return PreparedLineage(sliced=sliced)

    scope = sliced.scope
    if materialize is not None:
        from ..io.reader import SliceScope
        sliced.scope = SliceScope(
            materialize("scope", scope.frame), scope.size, scope.hashes,
            int(cfg.get("slice.inline_limit", 20_000)),
        )
        scope = sliced.scope
    account_changes = restrict_to_slice(
        read_silver(spark, cfg, "account_changes_batch"), scope
    )
    device_history = restrict_to_slice(
        read_silver(spark, cfg, "device_lookup_batch"), scope
    )
    # The two enrichment histories the lifecycle stages read, restricted to the
    # slice and never watermarked. An activation date and a port both describe a
    # boundary that can sit years before the batch that dragged its phone number
    # in, so a window on when the evidence *arrived* would hide most of it.
    #
    # New activation responses also contribute to the slice in this deployment.
    # Read their full history here, including responses received in earlier runs.
    tu_history = restrict_to_slice(tu_ports, scope)
    activations = restrict_to_slice(activation_history, scope)

    if materialize is not None:
        account_changes = materialize("account_changes", account_changes)
        device_history = materialize("device_history", device_history)
        tu_history = materialize("tu_history", tu_history)
        activations = materialize("activations", activations)
        tu_batch = materialize("tu_batch", tu_batch)

    windows = windows_from_config(cfg)

    # -- stage 1 ---------------------------------------------------------
    # Cached because four of the five stages read it, and it is the most
    # expensive frame in the run to recompute: three derivations, two window
    # passes and a dedupe.
    events = build_canonical_events(
        account_changes, device_history, ctx.run_ts, windows=windows
    )
    events = materialize("events", events) if materialize is not None else events.cache()

    # -- stage 2 ---------------------------------------------------------
    # The work frame, not the published table: the account stage needs the
    # boundary event ids, and re-deriving them would mean re-running the walk.
    lifecycles = lifecycle_stage.build_lifecycles(spark, events)
    if materialize is not None:
        # Boundary evidence can read the walk more than once. Pay for it once.
        lifecycles = materialize("carrier_lifecycles", lifecycles)

    # -- stage 2b --------------------------------------------------------
    # Boundaries the carrier feeds guessed, read off the enrichment feeds, and
    # then the stretches of the number's life only TransUnion witnessed. The
    # order is load-bearing: the second stage clips TU's carrier stays against
    # the intervals these lifecycles occupy, so firming has to have finished
    # moving those intervals before the clipping reads them.
    lifecycles, boundary_report = boundary_stage.firm_boundaries(
        lifecycles, activations, tu_history
    )
    lifecycles, tu_report = tu_lifecycle_stage.build_tu_lifecycles(
        spark, lifecycles, tu_history
    )
    lifecycles = (
        materialize("lifecycles", lifecycles)
        if materialize is not None else lifecycles.cache()
    )
    return PreparedLineage(
        sliced, events, lifecycles, account_changes, device_history,
        tu_batch, tu_history, activations, boundary_report, tu_report,
    )


def plan(ctx: RunContext) -> JobPlan:
    """Build all five Gold frames using the original single-job entry point."""
    prepared = prepare_lineage(ctx)
    sliced = prepared.sliced
    if sliced.scope.is_empty():
        return JobPlan(scope=sliced.scope, slice_result=sliced)
    spark, cfg = ctx.spark, ctx.cfg
    scope = sliced.scope
    events, lifecycles = prepared.events, prepared.lifecycles
    account_changes, device_history = prepared.account_changes, prepared.device_history
    tu_batch, tu_history, activations = prepared.tu_batch, prepared.tu_history, prepared.activations
    boundary_report, tu_report = prepared.boundary_report, prepared.tu_report
    windows = windows_from_config(cfg)

    # -- stage 3 ---------------------------------------------------------
    accounts, account_report = accounts_stage.build_accounts(
        spark,
        events,
        lifecycles,
        account_changes,
        device_history,
        windows=windows,
        # ``cfg`` is here for ``accounts.imsi_link_min_confidence``, which is not
        # a window and so cannot ride in on ``windows`` - ``windows_from_config``
        # rejects any key it does not already know. Its default preserves the
        # published behaviour exactly, so passing the config through changes
        # nothing until somebody deliberately sets the key.
        cfg=cfg,
    )
    accounts = accounts.cache()
    LOGGER.info(account_report.format_line())

    # -- stage 4 ---------------------------------------------------------
    # ``events`` is passed for one reason: how tight each port's handover was is
    # visible only on the events. The lifecycle walk closes the losing lifecycle
    # at the arriving one's first event, so by stage 4 every port gap reads zero.
    customers, customer_report = customers_stage.build_customers(
        spark, accounts, lifecycles, events, windows=windows
    )
    customers = customers.cache()
    LOGGER.info(customer_report.format_line())

    # -- stage 5 ---------------------------------------------------------
    normalized = build_normalized_events(
        events, customers, device_history, ctx.run_ts, cfg=cfg
    )

    outputs = {
        "account_changes_canonical_events": events,
        "msisdn_lifecycle": lifecycle_stage.to_gold(lifecycles, ctx.run_ts),
        "msisdn_lifecycle_account_mapping": accounts_stage.to_gold(
            accounts, ctx.run_ts
        ),
        "msisdn_lifecycle_account_customer_mapping": customers_stage.to_gold(
            customers, ctx.run_ts
        ),
        "normalized_canonical_events": normalized,
    }

    return JobPlan(
        scope=scope,
        outputs=outputs,
        slice_result=sliced,
        extra_checks=_report_checks(
            sliced, account_report, customer_report, boundary_report, tu_report
        ),
        input_checks=_enrichment_input_checks(
            ctx,
            tu_batch,
            tu_history,
            activations,
            account_changes,
            device_history,
        ),
        notes={
            "account_edges": account_report.edge_count,
            "account_edges_by_route": account_report.edges_by_route,
            "accounts": account_report.account_count,
            "customers": customer_report.customer_count,
            "port_edges": customer_report.port_edge_count,
            "quiet_port_candidates": customer_report.quiet_port_candidates,
            "tu_l0_size": sliced.tu_l0_size,
            "tu_lifecycles_built": tu_report.lifecycles_built,
            "tu_lifecycle_carriers": tu_report.unsupported_carriers,
            "starts_firmed": boundary_report.starts_firmed,
            "ends_firmed": boundary_report.ends_firmed,
        },
    )


def _report_checks(
    sliced, account_report, customer_report, boundary_report, tu_report
) -> dict[str, list]:
    """Turn the two traversal reports into ``info`` metrics.

    None of these are scans over the output - they are counts the traversal
    already made and that nothing downstream could recover. A chain that had to be
    truncated at the depth cap leaves no mark on the rows it produced; the only
    place that fact exists is the report, so it goes to the metrics table from
    here or not at all.

    The slice counters ride along on the first table so that "how big was the
    slice, and how did it get that big" is answerable from the same query as
    everything else about the run.
    """
    return {
        "account_changes_canonical_events": [
            counter(
                "info_slice_l0_size",
                sliced.l0_size,
                "Lineage",
                description="phone numbers named directly by the batch",
                checked_field=SLICE_KEY,
            ),
            counter(
                "info_slice_size",
                sliced.size,
                "Lineage",
                description="phone numbers after the account and port closure",
                checked_field=SLICE_KEY,
            ),
            counter(
                "info_slice_iterations",
                sliced.iterations,
                "Lineage",
                description="closure iterations; equal to slice.max_iterations "
                "means the slice did not converge and linked rows are stale",
            ),
            counter(
                "info_slice_l0_tu_size",
                sliced.tu_l0_size,
                "Completeness",
                description="phone numbers the TransUnion delivery named; zero "
                "on every run while the feed is enabled means enrichment is "
                "not reaching the rebuild, which no other metric distinguishes "
                "from having no feed at all",
                checked_field=SLICE_KEY,
            ),
        ],
        "msisdn_lifecycle": [
            counter(
                "info_tu_phones_enriched",
                tu_report.phones_enriched,
                "Completeness",
                description="phone numbers TransUnion answered about, whether or "
                "not it had anything to add",
                checked_field=SLICE_KEY,
            ),
            counter(
                "info_tu_lifecycles_built",
                tu_report.lifecycles_built,
                "Completeness",
                description="lifecycles at carriers we hold no feed for, which "
                "exist only because TransUnion witnessed the ports either side",
            ),
            counter(
                "info_idv_starts_firmed",
                boundary_report.starts_firmed,
                "Accuracy",
                description="lifecycle starts an observed activation date "
                "replaced or confirmed",
            ),
            counter(
                "info_idv_start_conflicts",
                boundary_report.start_conflicts,
                "Consistency",
                description="activation dates our own lifecycles cannot "
                "accommodate; a rising count means IDV and a carrier feed "
                "disagree about when the current subscriber arrived",
            ),
            counter(
                "info_tu_ends_firmed",
                boundary_report.ends_firmed,
                "Accuracy",
                description="ambiguous carrier ends a TransUnion port out "
                "settled",
            ),
        ],
        "msisdn_lifecycle_account_mapping": [
            counter(
                f"info_account_edges_route_{route}",
                count,
                "Lineage",
                description=f"number-change edges matched by the {route} route",
            )
            for route, count in sorted(account_report.edges_by_route.items())
        ]
        + [
            counter(
                "info_account_chain_cycles_broken",
                len(account_report.broken_cycle_edges),
                "Consistency",
                description="number-change edges dropped to break a cycle; "
                "non-zero means two lifecycles each claim to follow the other",
            ),
            counter(
                "info_account_chains_truncated",
                len(account_report.truncated_roots),
                "Consistency",
                description="account chains cut at windows.max_chain_depth",
            ),
            counter(
                "info_account_edges_dropped",
                len(account_report.dropped_edges),
                "Consistency",
                description="edges lost a conflict for the same lifecycle",
            ),
        ],
        "msisdn_lifecycle_account_customer_mapping": [
            counter(
                "info_customer_port_edges",
                customer_report.port_edge_count,
                "Lineage",
                description="ports that joined two accounts into one customer",
            ),
            counter(
                "info_customer_quiet_port_candidates",
                customer_report.quiet_port_candidates,
                "Completeness",
                description="cross-carrier reappearances that fell outside "
                "windows.mno_port_window_days and were therefore not ports; "
                "a rising count means the window is too narrow",
            ),
            counter(
                "info_customer_chain_cycles_broken",
                len(customer_report.broken_cycle_edges),
                "Consistency",
                description="port edges dropped to break a cycle",
            ),
            counter(
                "info_customer_chains_truncated",
                len(customer_report.truncated_roots),
                "Consistency",
                description="customer chains cut at windows.max_chain_depth",
            ),
        ],
    }


def run(argv: Sequence[str] | None = None):
    return run_gold_job(JOB_NAME, plan, argv)


def main_cli(argv: Sequence[str] | None = None) -> int:
    return main(JOB_NAME, plan, argv)


if __name__ == "__main__":  # pragma: no cover - entrypoint
    sys.exit(main_cli())
