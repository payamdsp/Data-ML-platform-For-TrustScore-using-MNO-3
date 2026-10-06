"""Make the driver's Python clock agree with the Spark session's.

Why this module exists
----------------------
Two clocks decide what a row's ``ingestion_ts`` means, and until this module
existed they disagreed by five hours on every EMR run.

``spark.session_timezone`` is ``EST`` and is a correctness setting, not a
display one: every wall-clock string the estate writes or parses is read in
that zone. ``ingestion_ts`` is produced as a string from the driver's Python
clock and then parsed by Spark with ``to_timestamp``, so the instant it lands on
is *the driver's wall clock read as EST*. An EMR node's operating system runs
UTC. A job whose driver saw 10:03 therefore stamped the row 10:03 EST, which is
15:03 UTC - five hours after the run that wrote it.

Nothing inside Bronze -> Silver notices, because it only ever compares those
stamps with each other. Silver -> Gold does notice, and the way it notices is
the worst available: its watermark upper bound is a Python ``datetime`` handed
to ``F.lit``, and PySpark converts that using the *driver's* zone, giving a true
UTC instant. Every Silver row then looks five hours in the future, the window
``(from, to]`` selects nothing, and the job reports SUCCESS having built its
lineage out of an empty batch. No error, no warning, no rows - just Gold tables
that quietly stop growing.

The fix is to stop having two zones. Pinning ``TZ`` to the configured session
zone before anything reads the clock makes ``datetime.now()`` return EST wall
time, so a string rendered from it and parsed as EST round-trips to the instant
it was taken at, and a ``datetime`` handed to ``F.lit`` converts to that same
instant. One zone, one meaning.

It matters on this side for a second reason. A run's ``watermark_to`` is written
into the run-control table and read back as the next run's ``watermark_from``,
so a boundary taken in one zone and compared in another is not merely one lost
batch: it is a window edge every later run inherits.

Call this once, immediately after the config is loaded and before the first
clock read. It is idempotent, and a no-op on platforms without ``time.tzset``
(Windows), where the caller is a developer's laptop rather than a cluster.
"""

from __future__ import annotations

import os
import time

from .logging_utils import get_logger

LOGGER = get_logger(__name__)

#: The zone every wall-clock string in the estate is written and read in.
DEFAULT_SESSION_TIMEZONE = "EST"


def pin_process_timezone(zone: str | None) -> str:
    """Point the process's local zone at ``zone``. Returns the zone applied.

    ``EST`` is a fixed -05:00 offset in both the tz database and Spark, with no
    daylight saving, so the two agree on every date of the year. That is the
    reason the estate names ``EST`` rather than ``America/Toronto``: a summer
    timestamp under a DST-observing zone would be read an hour off by whichever
    of the two was configured with it.
    """
    resolved = str(zone or DEFAULT_SESSION_TIMEZONE)

    if not hasattr(time, "tzset"):  # pragma: no cover - Windows only
        LOGGER.warning(
            "cannot pin the process timezone to %s on this platform; "
            "wall-clock stamps will be taken in the host's zone",
            resolved,
        )
        return resolved

    if os.environ.get("TZ") != resolved:
        os.environ["TZ"] = resolved
        time.tzset()
        LOGGER.info("process timezone pinned to %s", resolved)

    return resolved
