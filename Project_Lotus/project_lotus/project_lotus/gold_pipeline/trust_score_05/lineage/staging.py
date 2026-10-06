"""Intermediate results parked on S3 so a failure costs one step, not the walk.

The customer stage is one call that takes tens of minutes: build the port edges
off 353 million canonical events, label the graph's components, walk each
component, assemble the answer. Run as one statement it has two problems, and
the second is the one that keeps costing runs.

The first is that a failure anywhere throws away everything. The second is that
a Glue statement returns its stdout only when it finishes, so a statement that
dies returns *nothing* - not the counts, not the round lines, not the exception.
Twenty-one minutes of work and twenty-one minutes of narration come back as
``Failed to cancel the statement 8 as it has been cancelled``.

Both are the same problem wearing different hats: the statement is too big to be
the unit of anything. So the stage is run as four of them, each writing its
result here and reading its input back from here. Each one finishes, which means
each one returns its output to the notebook; each one is durable, which means the
next attempt starts where the last one stopped. What used to be one opaque
twenty-one-minute call is four observable ones, and the failed attempt names
itself.

Parquet rather than Iceberg, and a path rather than a catalog table. These are
scratch: they have no schema anybody consumes, no snapshot anybody time-travels
to, and no business in the Glue catalog next to the tables that do. The
directory is overwritten wholesale or read wholesale, which is the only access
pattern a staging file has.
"""

from __future__ import annotations

from collections.abc import Mapping

from pyspark.sql import DataFrame, SparkSession

from .progress import _read_text, _write_text, note

__all__ = [
    "staged",
    "staging_path",
    "clear_staging",
    "describe_staging",
    "match_inputs",
    "table_fingerprint",
]


def staging_path(prefix: str, name: str) -> str:
    return f"{prefix.rstrip('/')}/{name}"


def _exists(spark: SparkSession, path: str) -> bool:
    """Whether a *complete* write is sitting at ``path``.

    ``_SUCCESS`` and not the directory, because a directory is created by the
    write that then died halfway and left a third of the partitions in it.
    Reusing that is worse than recomputing it: it is a wrong answer that costs
    nothing to produce, which is the kind that gets published.
    """
    try:
        jvm = spark.sparkContext._jvm  # noqa: SLF001
        hadoop_path = jvm.org.apache.hadoop.fs.Path(f"{path.rstrip('/')}/_SUCCESS")
        conf = spark.sparkContext._jsc.hadoopConfiguration()  # noqa: SLF001
        return hadoop_path.getFileSystem(conf).exists(hadoop_path)
    except Exception:  # noqa: BLE001 - a staging check may not break the stage
        return False


def staged(
    spark: SparkSession,
    frame: DataFrame,
    path: str,
    what: str = "",
    force: bool = False,
    partitions: int | None = None,
) -> DataFrame:
    """Materialise ``frame`` at ``path`` and hand back the frame read off it.

    The returned frame is read from the files, not the one passed in. That is
    the whole point and not an implementation detail: the input's plan may be a
    join over 353 million rows, and every downstream reader of it pays that plan
    again. Reading the parquet back replaces the plan with a file scan, which is
    what makes the *next* statement short as well as this one.

    ``force`` re-does the work when the previous attempt's output is suspect.
    ``frame`` must not be derived from ``path``: the write overwrites the
    directory, and a frame reading it is then reading files that have been
    deleted out from under it. Build it from the step before, which is what a
    re-run of a notebook cell does anyway.
    ``partitions`` narrows the write - a frame that arrives on 2,000 partitions
    and holds four million rows writes 2,000 files of two thousand rows each,
    and the read after it pays a listing and 2,000 tasks for a frame that fits
    in twenty.

    It narrows with ``repartition`` and not with ``coalesce``, which is the
    opposite of the usual advice and is right here. ``coalesce`` does not
    shuffle, so the narrowing pushes back up into the stage that produces the
    frame: asking for twenty output files off ``customer_port_edges`` would run
    the join against 353 million canonical events on twenty tasks. A shuffle
    boundary is exactly what stops that, and the frames staged here are
    edge-scale and node-scale - millions of rows, not the hundreds of millions
    they were derived from - so the shuffle it costs is small and the
    parallelism it protects is not.
    """
    label = what or path.rsplit("/", 1)[-1]
    if not force and _exists(spark, path):
        out = spark.read.parquet(path)
        note(f"  {label}: reusing what a previous run already staged at {path}")
        return out

    writer = frame if partitions is None else frame.repartition(partitions)
    writer.write.mode("overwrite").parquet(path)
    out = spark.read.parquet(path)
    note(f"  {label}: staged to {path}")
    return out


def clear_staging(spark: SparkSession, prefix: str) -> None:
    """Delete everything under ``prefix``. Call it when a rebuild starts.

    A rebuild that reuses a previous rebuild's staging silently publishes the
    previous rebuild's answer, which is the one failure mode this module can
    introduce and the reason clearing is a deliberate call rather than a flag on
    each step.
    """
    try:
        jvm = spark.sparkContext._jvm  # noqa: SLF001
        hadoop_path = jvm.org.apache.hadoop.fs.Path(prefix)
        conf = spark.sparkContext._jsc.hadoopConfiguration()  # noqa: SLF001
        fs = hadoop_path.getFileSystem(conf)
        if fs.exists(hadoop_path):
            fs.delete(hadoop_path, True)
            note(f"  staging: cleared {prefix}")
        else:
            note(f"  staging: nothing to clear at {prefix}")
    except Exception as exc:  # noqa: BLE001
        note(f"  staging: could not clear {prefix} ({type(exc).__name__}: {exc})")


def table_fingerprint(spark: SparkSession, tables: Mapping[str, str]) -> str:
    """A string that changes when any of ``tables`` is written again.

    The snapshot id each Iceberg table currently points at, which is exactly the
    question :func:`match_inputs` needs answered: not "has the data changed"
    but "is this the same commit the staging was built from". Reading it is a
    metadata query and opens no data file.

    A table that cannot be asked - a metadata table an engine does not expose,
    a catalog that is not Iceberg - comes back as ``unreadable:<error>`` rather
    than raising. That is deliberate and it is a real weakening: an estate where
    every table is unreadable produces the same fingerprint on every run, so the
    staging is always reused. It is the right trade anyway, because the
    alternative is a diagnostic that fails runs, and the whole string is printed
    by the caller so a fingerprint made of nothing is visible rather than
    assumed.
    """
    parts = []
    for name in sorted(tables):
        table = tables[name]
        try:
            rows = spark.sql(
                f"SELECT snapshot_id FROM {table}.snapshots "
                "ORDER BY committed_at DESC LIMIT 1"
            ).collect()
            snapshot = str(rows[0][0]) if rows else "empty"
        except Exception as exc:  # noqa: BLE001 - a fingerprint, not a contract
            snapshot = f"unreadable:{type(exc).__name__}"
        parts.append(f"{name}={snapshot}")
    return " ".join(parts)


def match_inputs(spark: SparkSession, prefix: str, fingerprint: str) -> bool:
    """Keep what is staged under ``prefix`` only if it was built from these inputs.

    Reuse is the whole point of this module and it is also its one way of being
    catastrophically wrong. A rebuild that re-runs notebooks 01 to 03 and then
    reuses the staging from before them publishes a customer table built on the
    *previous* rebuild's accounts: no error, no empty frame, a full table of
    plausible rows that disagree with the accounts sitting next to it. Nothing
    downstream can tell. That is a worse outcome than the twenty minutes reuse
    is saving.

    So the reuse is conditional on a string the caller computes from whatever
    identifies its inputs - for the notebooks, the current Iceberg snapshot id
    of each upstream table. It is written beside the staged directories the
    first time and compared on every run after; a mismatch clears the prefix
    before anything reads it. Returns whether the staging survived.

    A fingerprint that cannot be written is reported and does not stop the run,
    for the reason :mod:`.progress` gives: a bookkeeping file does not get to
    fail a stage. It does mean the next run cannot verify the staging, which is
    why the failure is printed rather than swallowed.
    """
    marker = f"{prefix.rstrip('/')}/_inputs.txt"
    try:
        previous = _read_text(spark, marker).strip()
    except Exception as exc:  # noqa: BLE001
        note(f"  staging: could not read {marker} ({type(exc).__name__}: {exc})")
        previous = ""

    kept = True
    if previous and previous != fingerprint.strip():
        note(
            "  staging: the upstream tables have been rebuilt since this "
            "staging was written - clearing it rather than reusing it"
        )
        note(f"    was : {previous}")
        note(f"    now : {fingerprint.strip()}")
        clear_staging(spark, prefix)
        kept = False
    elif previous:
        note("  staging: the upstream tables are the ones this staging was built from")

    try:
        _write_text(spark, marker, fingerprint.strip() + "\n")
    except Exception as exc:  # noqa: BLE001
        note(
            f"  staging: could not record the inputs at {marker} "
            f"({type(exc).__name__}: {exc}). The next run will not be able to "
            "tell whether this staging matches its inputs."
        )
    return kept


def describe_staging(spark: SparkSession, prefix: str) -> list[str]:
    """The names of the completed steps sitting under ``prefix``.

    Printed at the top of a notebook so the operator can see what the next run
    will skip before it skips it, rather than inferring it from a run that
    finished suspiciously fast.
    """
    found: list[str] = []
    try:
        jvm = spark.sparkContext._jvm  # noqa: SLF001
        hadoop_path = jvm.org.apache.hadoop.fs.Path(prefix)
        conf = spark.sparkContext._jsc.hadoopConfiguration()  # noqa: SLF001
        fs = hadoop_path.getFileSystem(conf)
        if not fs.exists(hadoop_path):
            return found
        for status in fs.listStatus(hadoop_path):
            child = status.getPath()
            if status.isDirectory() and _exists(spark, child.toString()):
                found.append(child.getName())
    except Exception:  # noqa: BLE001
        return found
    return sorted(found)
