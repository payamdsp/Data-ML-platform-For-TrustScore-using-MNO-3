"""Lineage 2: the phone number <-> device map.

A separate job from the account lineage, and separate on purpose. The two share
one Silver table and nothing else: a device segment is a fact about a handset and
a phone number, and it does not care which account or customer that phone number
belongs to. Running them together would couple the device map's schedule to the
account lineage's slice closure - which is much wider, because it follows number
changes and ports - for no benefit.

The one place they do touch is ``imei_map.close_on_lifecycle_end``: a phone number
that was cancelled or ported away is not still carrying a handset, so an open
device segment can be closed at the end of the phone number's last lifecycle. That
reads the account lineage's output, which is why it is behind a flag and why the
job runs perfectly well with it off.

The slice here is L0 only. The device map's grain is one phone number and one
device, so a phone number's segments depend on that phone number's device events
and nothing else. There is no closure to do, and running the account lineage's
closure would drag in phone numbers whose device rows this job is not going to
change - a wider slice, a wider delete branch, and no different an answer.
"""

from __future__ import annotations

import sys
from typing import Sequence

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ..dq import counter
from ..io import (
    gold_exists,
    read_gold,
    read_silver,
    restrict_to_slice,
    select_batch,
)
from ..logging_utils import get_logger
from ..schemas import SLICE_KEY
from ..slice import build_slice
from ..transforms import imei_map as imei_stage
from .base import JobPlan, RunContext, main, read_phone_number_scope, run_gold_job

LOGGER = get_logger(__name__)

JOB_NAME = "gold_msisdn_imei_map"

__all__ = ["JOB_NAME", "plan", "run", "main_cli"]


def _batch(ctx: RunContext) -> DataFrame:
    spark, cfg = ctx.spark, ctx.cfg
    device_lookup = read_silver(spark, cfg, "device_lookup_batch")

    explicit = read_phone_number_scope(
        spark,
        ctx.args.phone_numbers,
        inline_limit=int(cfg.get("slice.inline_limit", 20_000)),
    )
    if explicit is not None:
        LOGGER.info(
            "repair mode | %d phone number(s) named explicitly", explicit.size
        )
        return restrict_to_slice(device_lookup, explicit)
    return select_batch(device_lookup, ctx.watermark_from, ctx.watermark_to)


def _require_lifecycles(spark, cfg) -> None:
    """Refuse to close segments against a lifecycle table that is not there.

    ``read_gold`` answers "no such table" with an empty frame and a warning,
    which is the right default for the slice closure - a first run legitimately
    has nothing to close over, and an empty mapping is the truthful answer. It
    is the wrong default here. ``close_on_lifecycle_end`` exists because a phone
    number that was cancelled or ported away is not still carrying a handset;
    handed no lifecycles at all, it closes nothing, every segment stays open,
    and the job writes that to Gold and reports success.

    That is not hypothetical on this cluster. Both Gold jobs are queued as EMR
    steps on one cluster at a bootstrap, the account lineage runs first, and its
    step's ``ActionOnFailure`` is ``CONTINUE`` - so a lineage failure is followed
    immediately by an IMEI map run with no ``msisdn_lifecycle`` to read. The
    difference between failing here and not is the difference between one red
    step and a Gold table full of quietly wrong ``is_open`` flags.

    Existence, not freshness. A lifecycle table one day stale still closes the
    segments it knows about, and tomorrow's run closes the rest; that is the
    ordinary incremental state and not an error. What cannot be tolerated is
    *nothing*, because nothing is indistinguishable from "no phone number has
    ever ended" and the job cannot tell the two apart.

    The way past this is the flag, which is why the message names it: a run that
    genuinely has no account lineage - the device map standing alone - should
    say so with ``imei_map.close_on_lifecycle_end: false`` rather than be handed
    an empty table and left to guess.
    """
    if gold_exists(spark, cfg, "msisdn_lifecycle"):
        return
    raise RuntimeError(
        "imei_map.close_on_lifecycle_end is on, but the Gold table "
        "msisdn_lifecycle does not exist. Run the account lineage job first, "
        "or set imei_map.close_on_lifecycle_end: false to build the device map "
        "without closing open segments at the end of a phone number's last "
        "lifecycle."
    )


def plan(ctx: RunContext) -> JobPlan:
    """Rebuild every device segment for the phone numbers the batch touched."""
    spark, cfg = ctx.spark, ctx.cfg

    dev_batch = _batch(ctx)
    sliced = build_slice(
        spark,
        cfg,
        account_changes_batch=None,
        device_lookup_batch=dev_batch,
        close_number_changes=False,
        close_ports=False,
        first_run=bool(ctx.args.first_run),
    )
    if sliced.scope.is_empty():
        return JobPlan(scope=sliced.scope, slice_result=sliced)

    scope = sliced.scope

    # Complete device history, not the batch. A segment's start can be months
    # before the event that caused this run, and walking the batch alone would
    # invent a start at the batch boundary.
    device_history = restrict_to_slice(
        read_silver(spark, cfg, "device_lookup_batch"), scope
    )

    segments = imei_stage.build_imei_map(spark, device_history).cache()

    closed_on_lifecycle = 0
    if bool(cfg.get("imei_map.close_on_lifecycle_end", True)):
        _require_lifecycles(spark, cfg)
        lifecycles = restrict_to_slice(read_gold(spark, cfg, "msisdn_lifecycle"), scope)
        before = segments.where(F.col("is_open") == F.lit(1)).count()
        segments = imei_stage.close_on_lifecycle_end(segments, lifecycles).cache()
        after = segments.where(F.col("is_open") == F.lit(1)).count()
        closed_on_lifecycle = max(before - after, 0)
        LOGGER.info(
            "close_on_lifecycle_end | %d open segment(s) closed at the end of "
            "the phone number's last lifecycle",
            closed_on_lifecycle,
        )

    # The three walk counters are facts about a phone number, stamped onto every
    # segment the walk produced for it, so they have to be reduced per number
    # before they are added up. Summing the column directly multiplies each
    # number's count by how many segments it happens to have - which is how a
    # feed with one duplicated start on a number that changed handset ten times
    # reported ten duplicates. ``is_open`` is a fact about the segment and is
    # summed as it stands.
    per_number = segments.groupBy(SLICE_KEY).agg(
        F.max("duplicate_start_cnt").alias("dup"),
        F.max("orphan_end_cnt").alias("orphan"),
        F.max("superseded_start_cnt").alias("superseded"),
        F.sum("is_open").alias("open"),
    )
    counters = per_number.agg(
        F.coalesce(F.sum("dup"), F.lit(0)).alias("dup"),
        F.coalesce(F.sum("orphan"), F.lit(0)).alias("orphan"),
        F.coalesce(F.sum("superseded"), F.lit(0)).alias("superseded"),
        F.coalesce(F.sum("open"), F.lit(0)).alias("open"),
    ).collect()[0]

    return JobPlan(
        scope=scope,
        outputs={"msisdn_imei_map": imei_stage.to_gold(segments, ctx.run_ts)},
        slice_result=sliced,
        extra_checks={
            "msisdn_imei_map": [
                counter(
                    "info_slice_size",
                    scope.size,
                    "Lineage",
                    description="phone numbers rebuilt by this run",
                    checked_field=SLICE_KEY,
                ),
                counter(
                    "info_imei_open_segments",
                    int(counters["open"]),
                    "Currency",
                    description="segments still open at the end of the run; "
                    "expected to be non-zero - most phone numbers are still on "
                    "the device they are using",
                ),
                counter(
                    "info_imei_segments_closed_on_lifecycle_end",
                    closed_on_lifecycle,
                    "Consistency",
                    description="open segments closed because the phone "
                    "number's last lifecycle had ended and the carrier sent no "
                    "IMEI_END",
                ),
                counter(
                    "info_imei_duplicate_starts",
                    int(counters["dup"]),
                    "Consistency",
                    description="device starts that re-announced an association "
                    "that had not lapsed; a refresh or a replayed row, not a "
                    "device change",
                ),
                counter(
                    "info_imei_orphan_ends",
                    int(counters["orphan"]),
                    "Consistency",
                    description="device ends with nothing open, or naming a "
                    "device a later start had already displaced; a rising count "
                    "means the feed's starts and ends disagree",
                ),
                counter(
                    "info_imei_superseded_starts",
                    int(counters["superseded"]),
                    "Consistency",
                    description="device starts closed at the instant they "
                    "opened, and so not published: the feed named another device "
                    "for the number in the same timestamp, or ended the "
                    "association in it. A half-open interval that ends where it "
                    "begins holds no moment, so no query could ever have seen "
                    "them",
                ),
            ]
        },
        notes={
            "imei_open_segments": int(counters["open"]),
            "imei_duplicate_starts": int(counters["dup"]),
            "imei_orphan_ends": int(counters["orphan"]),
            "imei_superseded_starts": int(counters["superseded"]),
            "imei_closed_on_lifecycle_end": closed_on_lifecycle,
        },
    )


def run(argv: Sequence[str] | None = None):
    return run_gold_job(JOB_NAME, plan, argv)


def main_cli(argv: Sequence[str] | None = None) -> int:
    return main(JOB_NAME, plan, argv)


if __name__ == "__main__":  # pragma: no cover - entrypoint
    sys.exit(main_cli())
