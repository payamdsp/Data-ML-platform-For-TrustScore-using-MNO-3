"""The run-control table: watermarks, run status, and the slice manifest.

A run must be able to answer two questions after the fact: *what window did I
consume*, and *which phone numbers did I intend to change*. Recomputing either
answer later is not good enough - the first would drift as new data arrives, and
the second is the only record of what a failed run was in the middle of.

So both are written down. The watermark goes to ``gold_run_control``, one row
per run per job. The slice goes to a manifest file under the run's own path,
before any Gold table is touched, which is what makes a failed run retryable on
exactly the slice it was working on.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Sequence

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import Config, ConfigError
from ..logging_utils import get_logger
from ..schemas import RUN_CONTROL_SCHEMA
from ..spark import qualified_table_name, table_exists

__all__ = [
    "RunControl",
    "RunRecord",
    "new_run_id",
    "STATUS_RUNNING",
    "STATUS_SUCCEEDED",
    "STATUS_FAILED",
]

LOGGER = get_logger(__name__)

STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"


def new_run_id() -> str:
    """A run id that sorts by time and is unique without coordination."""
    return f"{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True)
class RunRecord:
    """One row of ``gold_run_control``."""

    job_name: str
    run_id: str
    watermark_from: datetime | None
    watermark_to: datetime
    slice_size: int
    status: str
    started_ts: datetime
    finished_ts: datetime | None = None

    def as_row(self) -> tuple:
        return (
            self.job_name,
            self.run_id,
            self.watermark_from,
            self.watermark_to,
            int(self.slice_size),
            self.status,
            self.started_ts,
            self.finished_ts,
        )


class RunControl:
    """Read and write ``gold_run_control``.

    Deliberately not transactional and deliberately not clever. There are no
    transactions in this system - there are events, and a small table of facts
    about runs. Concurrency is handled by not running two instances of the same
    job at once, which the scheduler enforces; a second concurrent run would be
    visible here as two ``running`` rows for one ``job_name``, and
    :meth:`assert_no_running` is what turns that into a clear failure rather
    than two runs quietly fighting over the same slice.
    """

    def __init__(self, spark: SparkSession, cfg: Config) -> None:
        self.spark = spark
        self.cfg = cfg
        section = cfg.section("run_control")
        self.mode = str(section.get("mode", "iceberg")).lower()
        if self.mode == "iceberg":
            self.table = qualified_table_name(cfg, str(section.require("table")))
            self.path = None
        elif self.mode == "parquet":
            self.table = None
            self.path = str(section.require("path"))
        else:
            raise ConfigError(
                f"run_control.mode must be iceberg or parquet, got {self.mode!r}"
            )

    # -- existence ---------------------------------------------------------
    def exists(self) -> bool:
        if self.mode == "iceberg":
            return table_exists(self.spark, self.table)
        try:
            self.spark.read.parquet(self.path).schema
            return True
        except Exception:
            return False

    def ensure_exists(self) -> None:
        if self.mode != "iceberg":
            return
        if not self.exists():
            raise ConfigError(
                f"run control Iceberg table {self.table} is missing; provision it "
                "before submitting the Gold step"
            )

    def read(self) -> DataFrame:
        if not self.exists():
            return self.spark.createDataFrame([], RUN_CONTROL_SCHEMA)
        if self.mode == "iceberg":
            return self.spark.table(self.table)
        return self.spark.read.parquet(self.path)

    # -- watermark ---------------------------------------------------------
    def last_watermark(self, job_name: str) -> datetime | None:
        """The ``watermark_to`` of the most recent **succeeded** run.

        Succeeded only. A failed run may have written part of its slice, but
        because every write is a merge and every merge is a full recomputation
        of the slice, re-consuming its window is safe and re-consuming it is
        strictly better than skipping it. Advancing the watermark past a failed
        run is the one way to lose data permanently.
        """
        if not self.exists():
            return None
        row = (
            self.read()
            .where(
                (F.col("job_name") == F.lit(job_name))
                & (F.col("status") == F.lit(STATUS_SUCCEEDED))
            )
            .agg(F.max("watermark_to").alias("wm"))
            .collect()
        )
        return row[0]["wm"] if row else None

    def assert_no_running(self, job_name: str, *, except_run_id: str | None = None) -> None:
        """Fail loudly when a previous run of this job never finished.

        Two runs of one job on overlapping slices would merge over each other's
        output. This does not prevent that - nothing here can - but it turns a
        silent corruption into a run that refuses to start until an operator
        looks at the earlier run.

        "Unfinished" has to be evaluated **per run id**, not per row. This table
        is append-only: :meth:`start` writes a ``running`` row and :meth:`finish`
        writes a second row next to it rather than updating the first, so a
        perfectly normal completed run leaves one ``running`` row behind forever.
        Asking only "is there a ``running`` row?" would therefore be true from
        the first successful run onwards, and every run after it would refuse to
        start - a pipeline that works exactly once.

        A run is unfinished when it has a ``running`` row and no ``succeeded`` or
        ``failed`` row for the same ``run_id``.
        """
        if not self.exists():
            return
        stuck = (
            self.read()
            .where(F.col("job_name") == F.lit(job_name))
            .groupBy("run_id")
            .agg(
                F.max(
                    F.when(F.col("status") == F.lit(STATUS_RUNNING), 1).otherwise(0)
                ).alias("_started"),
                F.max(
                    F.when(
                        F.col("status").isin([STATUS_SUCCEEDED, STATUS_FAILED]), 1
                    ).otherwise(0)
                ).alias("_closed"),
                F.min(
                    F.when(
                        F.col("status") == F.lit(STATUS_RUNNING), F.col("started_ts")
                    )
                ).alias("started_ts"),
            )
            .where((F.col("_started") == F.lit(1)) & (F.col("_closed") == F.lit(0)))
            .select("run_id", "started_ts")
            .collect()
        )
        # A staged run reserves this same logical job across several EMR steps.
        # Only that exact run may resume; all other unfinished runs still block.
        if except_run_id is not None:
            stuck = [row for row in stuck if row["run_id"] != except_run_id]
        if stuck:
            ids = ", ".join(f"{r['run_id']} (started {r['started_ts']})" for r in stuck)
            raise RuntimeError(
                f"job {job_name!r} has unfinished run(s): {ids}. "
                "Mark them failed (or succeeded, if you have verified the output) "
                "before starting another run."
            )

    # -- writing -----------------------------------------------------------
    def start(
        self,
        job_name: str,
        watermark_from: datetime | None,
        watermark_to: datetime,
        run_id: str | None = None,
    ) -> RunRecord:
        """Record a run as ``running``. Called before any Gold table is read."""
        record = RunRecord(
            job_name=job_name,
            run_id=run_id or new_run_id(),
            watermark_from=watermark_from,
            watermark_to=watermark_to,
            slice_size=0,
            status=STATUS_RUNNING,
            started_ts=datetime.now(),
        )
        self.ensure_exists()
        self._append(record)
        LOGGER.info(
            "run started | job=%s | run_id=%s | window=(%s, %s]",
            job_name,
            record.run_id,
            watermark_from,
            watermark_to,
        )
        return record

    def finish(
        self, record: RunRecord, status: str, slice_size: int | None = None
    ) -> RunRecord:
        """Close out a run.

        Appends a second row rather than updating the first. The control table
        is then an append-only log of what happened, which is more useful when
        something goes wrong than a single row that was overwritten - and
        :meth:`last_watermark` takes the max over succeeded rows, so the extra
        rows cost nothing.
        """
        done = replace(
            record,
            status=status,
            slice_size=record.slice_size if slice_size is None else int(slice_size),
            finished_ts=datetime.now(),
        )
        self._append(done)
        LOGGER.info(
            "run %s | job=%s | run_id=%s | slice=%d",
            status,
            done.job_name,
            done.run_id,
            done.slice_size,
        )
        return done

    def _append(self, record: RunRecord) -> None:
        df = self.spark.createDataFrame([record.as_row()], RUN_CONTROL_SCHEMA)
        if self.mode == "iceberg":
            df.writeTo(self.table).append()
        else:
            df.write.mode("append").parquet(self.path)

    def write_summary(self, summary: dict) -> str | None:
        """Write one readable JSON object for an EMR run, including failures.

        This is diagnostic output only. It never changes a table or watermark.
        The driver already uses boto3 to load the S3 YAML configuration.
        """
        root = self.cfg.get("run_control.summary_root", None)
        if not root:
            return None
        from urllib.parse import urlsplit
        import boto3

        uri = (
            f"{str(root).rstrip('/')}/job={summary['job']}/"
            f"run_id={summary['run_id']}/summary.json"
        )
        parsed = urlsplit(uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ConfigError("run_control.summary_root must be an s3:// prefix")
        boto3.client("s3").put_object(
            Bucket=parsed.netloc, Key=parsed.path.lstrip("/"),
            Body=json.dumps(summary, indent=2, default=str).encode("utf-8"),
            ContentType="application/json",
        )
        LOGGER.info("run summary written | %s", uri)
        return uri

    # -- slice manifest ----------------------------------------------------
    def write_manifest(
        self, record: RunRecord, hashes: Sequence[str], extra: dict | None = None
    ) -> str | None:
        """Persist the slice a run intends to change, before it changes anything.

        Two uses. A failed run can be retried on exactly this slice with
        ``--phone-numbers``, which removes the risk that a retry recomputes a
        different set because more data has arrived in the meantime. And an
        operator investigating an unexpected change has the answer to "was this
        phone number in scope?" without re-deriving the closure.
        """
        root = self.cfg.get("run_control.manifest_root", None)
        if not root:
            return None
        # The path logged and returned is the *directory*, because that is what
        # exists once Spark has written it: a part file, ``_SUCCESS``, and a
        # ``.crc`` sidecar. Naming a single file here would hand an operator a
        # path that resolves to nothing - during an incident, pasted into
        # ``--phone-numbers``, on the retry of a run that has already failed.
        path = f"{str(root).rstrip('/')}/job={record.job_name}/run_id={record.run_id}"
        payload = {
            "job_name": record.job_name,
            "run_id": record.run_id,
            "watermark_from": str(record.watermark_from),
            "watermark_to": str(record.watermark_to),
            "slice_size": len(hashes),
            "phone_number_AC_hash": list(hashes),
            **(extra or {}),
        }
        body = json.dumps(payload, indent=2, default=str)
        # One small file, written through Spark so the same code path works for
        # a local directory and for s3://.
        self.spark.createDataFrame([(body,)], "body string").coalesce(1).write.mode(
            "overwrite"
        ).text(path)
        LOGGER.info("slice manifest written | %s | %d phone numbers", path, len(hashes))
        return path

    def write_manifest_frame(
        self,
        record: RunRecord,
        frame: DataFrame,
        *,
        row_count: int,
        extra: dict | None = None,
    ) -> str | None:
        """Persist a manifest without collecting its keys on the driver.

        The JSON manifest above is intentionally convenient for an ordinary
        incremental slice, whose keys are already inline. A first fraud backlog
        can contain millions of records and is deliberately represented by a
        semi-join instead; turning that frame into one Python list merely to
        write the handoff would defeat the limit that kept it off the driver.

        Parquet is self-describing, works at ``s3://`` paths through Spark, and
        preserves the record-to-phone pairing the resolver needs on a retry.
        The ``.parquet`` suffix is part of the CLI contract: the two readers use
        it to distinguish this scalable form from the human-sized text form.
        """
        root = self.cfg.get("run_control.manifest_root", None)
        if not root:
            return None
        path = (
            f"{str(root).rstrip('/')}/job={record.job_name}/"
            f"run_id={record.run_id}/scope.parquet"
        )
        metadata = {
            "job_name": record.job_name,
            "run_id": record.run_id,
            "watermark_from": None
            if record.watermark_from is None
            else str(record.watermark_from),
            "watermark_to": str(record.watermark_to),
            "slice_size": int(row_count),
            **(extra or {}),
        }
        payload = frame
        for name, value in metadata.items():
            literal = F.lit(None).cast("string") if value is None else F.lit(value)
            payload = payload.withColumn(str(name), literal)
        payload.write.mode("overwrite").parquet(path)
        LOGGER.info("distributed manifest written | %s | %d row(s)", path, row_count)
        return path
