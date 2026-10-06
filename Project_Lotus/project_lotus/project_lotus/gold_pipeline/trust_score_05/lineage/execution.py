"""Durable handoffs for sequential EMR steps; contains no business rules.

Data is immutable under a unique attempt directory. A small JSON manifest is
written only AFTER every Parquet write succeeds. Readers use that manifest,
never an S3 folder listing, so a killed writer cannot expose partial results.
Keep the whole run prefix until publishing and any investigation are finished.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from contextlib import contextmanager
from urllib.parse import urlsplit

from pyspark.sql.types import StructType

from .config import Config, ConfigError
from .logging_utils import get_logger
from .spark import qualified_table_name, table_exists

LOGGER = get_logger(__name__)
STAGES = ("prepare", "accounts", "customers", "validate", "publish")
TABLES = (
    "account_changes_canonical_events", "msisdn_lifecycle",
    "msisdn_lifecycle_account_mapping",
    "msisdn_lifecycle_account_customer_mapping", "normalized_canonical_events",
)


def digest(value) -> str:
    """Stable identity for resolved configuration and handoff manifests."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


@contextmanager
def timed(spark, label: str, timings: dict):
    """Expose each action in both the driver log and Spark's Jobs UI."""
    start = time.monotonic()
    spark.sparkContext.setJobGroup(label, label, interruptOnCancel=True)
    LOGGER.info("BEGIN %s", label)
    try:
        yield
    finally:
        elapsed = round(time.monotonic() - start, 3)
        timings[label] = timings.get(label, 0.0) + elapsed
        LOGGER.info("END %s | %.3f seconds", label, elapsed)
        spark.sparkContext.setLocalProperty("spark.jobGroup.id", None)
        spark.sparkContext.setLocalProperty("spark.job.description", None)
        spark.sparkContext.setLocalProperty("spark.job.interruptOnCancel", None)


class RunStore:
    """Small control objects in S3, large frames in distributed Parquet files.

    Steps MUST run sequentially (EMR step concurrency = 1). Conditional creation
    protects the initial run identity; it is not a distributed execution lock.
    A RUNNING row in gold_run_control reserves the logical run between steps.
    """

    def __init__(self, spark, root: str, run_id: str):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", run_id):
            raise ConfigError("--run-id must use 1-160 letters, digits, dots, underscores or hyphens")
        parsed = urlsplit(root)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
            raise ConfigError("staging.root must be an s3://bucket/prefix/, not a bucket root")
        import boto3
        self.client = boto3.client("s3")
        self.spark = spark
        self.bucket = parsed.netloc
        self.key = f"{parsed.path.strip('/')}/run_id={run_id}"
        self.uri = f"s3://{self.bucket}/{self.key}"
        self.timings = {}
        self.assets = {}
        self.attempt = uuid.uuid4().hex
        self.stage = None

    def get(self, name: str):
        """Return None only for a missing object; permissions/errors must fail."""
        from botocore.exceptions import ClientError
        try:
            result = self.client.get_object(Bucket=self.bucket, Key=f"{self.key}/{name}.json")
            return json.loads(result["Body"].read())
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("NoSuchKey", "404", "NotFound"):
                return None
            raise

    def put(self, name: str, payload: dict, *, create=False):
        """An S3 object PUT exposes either the complete JSON or its predecessor."""
        options = {"IfNoneMatch": "*"} if create else {}
        self.client.put_object(
            Bucket=self.bucket, Key=f"{self.key}/{name}.json",
            Body=json.dumps(payload, indent=2, default=str).encode("utf-8"),
            ContentType="application/json", **options,
        )

    def begin(self, stage: str):
        self.stage = stage
        self.assets = {}

    def write_frame(self, name: str, frame):
        """Cut lineage at a durable boundary without bringing rows to Python."""
        leaf = name if name.endswith(".parquet") else f"{name}.parquet"
        path = f"{self.uri}/attempts/{self.stage}/{self.attempt}/{leaf}"
        schema = frame.schema.jsonValue()
        with timed(self.spark, f"{self.stage}.write.{name}", self.timings):
            frame.write.mode("errorifexists").parquet(path)
        self.assets[name] = {"path": path, "schema": schema}
        return self.read_asset(self.assets[name])

    def read_asset(self, asset):
        # Supplying the exact schema also makes an empty Parquet dataset usable
        # and preserves private work columns needed by the next business stage.
        return self.spark.read.schema(StructType.fromJson(asset["schema"])).parquet(asset["path"])

    def read_frame(self, stage: str, name: str):
        manifest = self.require(stage)
        return self.read_asset(manifest["assets"][name])

    def require(self, stage: str):
        manifest = self.get(f"completed/{stage}")
        if manifest is None:
            raise ConfigError(f"Run {stage!r} successfully before this step: {self.uri}")
        return manifest

    def complete(self, details: dict):
        manifest = {**details, "assets": self.assets, "timings_seconds": self.timings}
        self.put(f"completed/{self.stage}", manifest, create=True)
        return manifest


def current_snapshot(spark, cfg: Config, table: str):
    """Read the current main branch, including rollback; never guess on errors."""
    name = qualified_table_name(cfg, table)
    if not table_exists(spark, name):
        return {"table": table, "missing": True, "snapshot_id": None}
    rows = spark.sql(f"SELECT snapshot_id FROM {name}.refs WHERE name = 'main'").collect()
    return {"table": table, "missing": False,
            "snapshot_id": int(rows[0][0]) if rows else None}


def pin_inputs(spark, cfg: Config) -> dict:
    """Freeze the four consumed Silver inputs and the five Gold closure inputs.

    This staged EMR path deliberately requires Iceberg inputs. An S3 path whose
    files can be overwritten has no stable snapshot to resume from.
    """
    pins = {"sources": {}, "sinks": {}}
    for dataset in ("account_changes_batch", "device_lookup_batch", "tu_portps", "mno_activation"):
        src = cfg.get(f"sources.{dataset}", {}) or {}
        optional = dataset in ("tu_portps", "mno_activation")
        if optional and (not src or not src.get("enabled", True)):
            continue
        if src.get("mode", "table") != "table":
            raise ConfigError(f"Staged execution requires an Iceberg source: {dataset}")
        pin = current_snapshot(spark, cfg, src["table"])
        if pin["missing"] and not optional:
            raise ConfigError(f"Required Silver table is missing: {src['table']}")
        pins["sources"][dataset] = pin
    for dataset in TABLES:
        sink = cfg.section(f"sinks.{dataset}")
        if sink.get("mode", "iceberg") != "iceberg":
            raise ConfigError("Staged publishing requires existing Iceberg Gold tables")
        pin = current_snapshot(spark, cfg, str(sink.require("table")))
        if pin["missing"]:
            raise ConfigError(f"Provision Gold table before the run: {pin['table']}")
        pins["sinks"][dataset] = pin
    return pins


def pinned_config(cfg: Config, pins: dict) -> Config:
    """Private read settings; sinks still publish to the current target head."""
    data = cfg.as_dict()
    for section, entries in pins.items():
        for name, pin in entries.items():
            data[section][name]["_read_snapshot"] = pin
    return Config(data, cfg.sources)
