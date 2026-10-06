"""Input and output adapters.

``reader`` distinguishes the two reads this pipeline makes - the watermarked
batch read and the unwatermarked slice read - and owns :class:`SliceScope`, the
object every other module uses to say "these phone numbers and no others".

``merge_sink`` writes. Gold is merged rather than appended, because a late event
can move a boundary written months ago.

``control`` holds the watermark and the run manifest.
"""

from .control import (
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
    RunControl,
    RunRecord,
    new_run_id,
)
from .merge_sink import (
    DeleteRatioExceeded,
    GoldSink,
    IcebergGoldSink,
    MergeStats,
    ParquetGoldSink,
    build_sink,
)
from .reader import (
    SliceScope,
    gold_exists,
    gold_location,
    optional_slice_frame,
    read_gold,
    read_optional_silver,
    read_silver,
    restrict_to_slice,
    select_batch,
    slice_frame,
    source_enabled,
    watermark_column,
)

__all__ = [
    "SliceScope",
    "read_silver",
    "read_optional_silver",
    "read_gold",
    "gold_exists",
    "gold_location",
    "select_batch",
    "restrict_to_slice",
    "slice_frame",
    "optional_slice_frame",
    "source_enabled",
    "watermark_column",
    "GoldSink",
    "IcebergGoldSink",
    "ParquetGoldSink",
    "MergeStats",
    "DeleteRatioExceeded",
    "build_sink",
    "RunControl",
    "RunRecord",
    "new_run_id",
    "STATUS_RUNNING",
    "STATUS_SUCCEEDED",
    "STATUS_FAILED",
]
