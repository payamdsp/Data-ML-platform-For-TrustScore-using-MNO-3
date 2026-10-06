"""Where a stage leaves its trail when the session is about to take it away.

**A Glue interactive statement's stdout does not leave the session until the
statement finishes.** That one sentence is the reason this module exists, and it
cost a run to learn. Notebook 04 was instrumented on 23 August 2026 to print the
port-edge count, the component sizes and a line per labelling round, on the
theory that a stage which narrates itself can be diagnosed after it dies. It
died on 24 August after twenty-one minutes and returned this, and nothing else::

    Unable to run statement for connection: project.spark.compatibility.
    Error: Exception encountered while canceling statement 8: Failed to cancel
    the statement 8 as it has been cancelled

Not one of the printed lines came back. They cannot: the client polls
``GetStatement`` and collects the output when the statement reaches a final
state, and a statement whose session has gone never reaches one. The underlying
failure is visible in the job log as ``IllegalSessionStateException ... Session
<id> unavailable, fail to call ReplServer`` - the session itself is gone, taking
the buffered stdout with it.

So the trail has to leave the process while the process is still alive. Every
line goes to a small text object in S3 as it is learned, and it is there whether
or not the statement that wrote it ever returns. The next run reads it back:
:func:`progress_tail` is what turns "the connection dropped" into "it was in
round nine of label propagation with four million labels still moving".

Three properties this has to have, all of them learned the hard way:

* **It must never be the thing that fails.** A stage does not get to die because
  the progress file could not be written. Every S3 call here is wrapped, and a
  failure downgrades to printing and carries on.
* **It must be whole after every line.** The object is rewritten in full each
  time rather than appended to, because S3 objects do not append and a
  half-flushed stream that the session then kills is worth nothing. The file is
  a few kilobytes; rewriting it is cheaper than the information is.
* **It must work with no sink configured.** The transforms call :func:`note`
  unconditionally. Under pytest, under a local run, and in any job that never
  called :func:`start_progress`, ``note`` is ``print`` and nothing more.
"""

from __future__ import annotations

import datetime as _dt
import time
from typing import Any

__all__ = [
    "start_progress",
    "stop_progress",
    "note",
    "progress_path",
    "progress_tail",
]


class _Trail:
    """One run's lines, and the S3 object they are mirrored into."""

    def __init__(self, spark: Any, path: str, what: str) -> None:
        self.spark = spark
        self.path = path
        self.started = time.monotonic()
        self.lines: list[str] = [
            f"# {what}",
            f"# opened {_dt.datetime.now().isoformat(timespec='seconds')}",
        ]
        self.broken = False
        self.flush()

    def add(self, line: str) -> None:
        self.lines.append(f"[{time.monotonic() - self.started:7.0f}s] {line}")
        self.flush()

    def flush(self) -> None:
        if self.broken:
            return
        try:
            _write_text(self.spark, self.path, "\n".join(self.lines) + "\n")
        except Exception as exc:  # noqa: BLE001 - see the module docstring
            self.broken = True
            print(
                f"WARNING: the progress trail at {self.path} cannot be written "
                f"({type(exc).__name__}: {exc}). The stage carries on without "
                "it, but if this run dies there will be nothing to read.",
                flush=True,
            )


_TRAIL: _Trail | None = None


# ==========================================================================
# the Hadoop filesystem, because it is the one that already knows about s3://
# ==========================================================================


def _filesystem(spark: Any, path: str):
    """The Hadoop ``FileSystem`` for ``path``, with Spark's own credentials.

    Not boto3. The driver already holds a filesystem that resolves ``s3://``
    through exactly the configuration the rest of the run writes with - the
    endpoint, the credential chain, the connection pool sized in the notebook's
    session cell - and reaching around it for a second client with its own
    opinion about all three is how a diagnostic starts causing incidents.
    """
    jvm = spark.sparkContext._jvm  # noqa: SLF001 - the documented way in
    hadoop_path = jvm.org.apache.hadoop.fs.Path(path)
    conf = spark.sparkContext._jsc.hadoopConfiguration()  # noqa: SLF001
    return hadoop_path.getFileSystem(conf), hadoop_path


def _write_text(spark: Any, path: str, text: str) -> None:
    fs, hadoop_path = _filesystem(spark, path)
    stream = fs.create(hadoop_path, True)
    try:
        stream.write(bytearray(text, "utf-8"))
    finally:
        stream.close()


def _read_text(spark: Any, path: str) -> str:
    """Read the object back as text, a line at a time, through the JVM.

    Line by line and not into a buffer, because py4j passes a ``bytearray`` to
    ``InputStream.read(byte[])`` **by value**: the JVM fills its own copy, the
    call returns a plausible byte count, and the Python buffer it was handed
    stays exactly as zero as it started. That failure is quiet - a trail full of
    ``\\x00`` reads as a corrupt file rather than as a bad read - which is worth
    a comment, since the whole point of this module is to be trustworthy on the
    one day anybody reads it. ``readLine`` returns a Java ``String``, which py4j
    converts properly, and these files are kilobytes.
    """
    fs, hadoop_path = _filesystem(spark, path)
    if not fs.exists(hadoop_path):
        return ""
    jvm = spark.sparkContext._jvm  # noqa: SLF001
    stream = fs.open(hadoop_path)
    reader = jvm.java.io.BufferedReader(
        jvm.java.io.InputStreamReader(stream, "UTF-8")
    )
    try:
        lines = []
        while True:
            line = reader.readLine()
            if line is None:
                break
            lines.append(line)
        return "\n".join(lines)
    finally:
        reader.close()


# ==========================================================================
# the interface the stages and the notebooks use
# ==========================================================================


def progress_path(prefix: str, what: str) -> str:
    """Where ``what``'s trail lives under ``prefix``.

    One object per notebook rather than one per run, deliberately overwritten.
    The question this file answers is "how far did the run that just died get",
    and that question is always about the most recent run. A prefix full of
    timestamped files would answer it too, and would also need somebody to work
    out which of them was the last one, at the moment they are least inclined to.
    """
    return f"{prefix.rstrip('/')}/{what}.log"


def start_progress(spark: Any, path: str, what: str = "run") -> str:
    """Begin a trail at ``path``. Returns the path, for the caller to print."""
    global _TRAIL
    _TRAIL = _Trail(spark, path, what)
    return path


def stop_progress() -> None:
    """Close the trail. Later :func:`note` calls print and nothing more."""
    global _TRAIL
    _TRAIL = None


def note(line: str) -> None:
    """Say something, to stdout and to the trail if one is open.

    The stdout half is not redundant with the S3 half. A statement that finishes
    returns its stdout to the notebook, where the line is already in front of
    whoever is reading, and the file is only reached for when it is not.
    """
    print(line, flush=True)
    if _TRAIL is not None:
        _TRAIL.add(line)


def progress_tail(spark: Any, path: str, lines: int = 60) -> str:
    """The last ``lines`` of a previous run's trail, or a note that there is none.

    Called at the top of a notebook, before anything expensive. If the previous
    attempt died inside a statement, this is the only place its last words are.
    """
    try:
        text = _read_text(spark, path)
    except Exception as exc:  # noqa: BLE001
        return f"(no trail at {path}: {type(exc).__name__}: {exc})"
    if not text.strip():
        return f"(no previous trail at {path} - this is the first attempt)"
    tail = text.strip().splitlines()[-lines:]
    return "\n".join(tail)
