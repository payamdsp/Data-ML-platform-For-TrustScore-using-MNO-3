"""The two entrypoints, and the driver they share.

``lineage_job``   the account / customer map - five tables, one recomputation
``imei_map_job``  the phone number <-> device map - one table, no closure

``base`` owns the order of operations, so neither job can skip a step and both
report the same way. A job supplies a name and a ``plan`` function; everything
from the watermark to the merge happens in ``base``.
"""

from . import base, imei_map_job, lineage_job  # noqa: F401
from .base import (
    WATERMARK_FLOOR,
    JobPlan,
    JobResult,
    RunContext,
    run_gold_job,
)

__all__ = [
    "base",
    "lineage_job",
    "imei_map_job",
    "WATERMARK_FLOOR",
    "JobPlan",
    "JobResult",
    "RunContext",
    "run_gold_job",
]
