"""SparkSession construction with pluggable Iceberg catalog profiles.

Three catalog profiles, selected by ``catalog.profile`` in config:

``glue``
    Iceberg tables registered in the Glue Data Catalog at an explicit
    ``s3://`` LOCATION. Which database that is comes from config, not from
    here: ``spark_catalog.ts05_dev_gold.<table>`` under ``conf/dev.yaml``,
    ``spark_catalog.ts05_sandbox_gold.<table>`` under ``conf/sandbox.yaml``,
    and the Silver inputs likewise from ``ts05_dev_silver`` /
    ``ts05_sandbox_silver``. (The retired POC estate,
    ``trust_score_v1_poc_*``, is what these names replaced - see the header of
    ``conf/dev.yaml``.)

``s3tables``
    Iceberg tables inside an S3 Tables table bucket, via the
    ``S3TablesCatalog`` implementation. This matches the buckets and
    ``bronze``/``silver``/``gold`` namespaces created in
    ``accounts/data-ai/s3tables.tf``.

``local``
    A Hadoop-backed Iceberg catalog on the local filesystem. Used for
    laptop / CI runs against the sample CSVs. Nothing AWS is contacted.

Switching profiles is a config change only - no code changes anywhere else in
the pipeline, because every read/write goes through a fully-qualified table
name that the profile resolves.

On a properly bootstrapped EMR cluster the catalog is usually already wired up
in ``spark-defaults``. Set ``catalog.manage_spark_conf: false`` in that case and
this module will leave catalog settings alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import Config, ConfigError
from .logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import SparkSession

LOGGER = get_logger(__name__)

ICEBERG_EXTENSIONS = "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions"

VALID_PROFILES = ("glue", "s3tables", "local")


def _catalog_conf(cfg: Config) -> dict[str, str]:
    """Translate the ``catalog`` config section into Spark conf entries."""
    profile = str(cfg.get("catalog.profile", "local")).lower()
    if profile not in VALID_PROFILES:
        raise ConfigError(
            f"catalog.profile must be one of {VALID_PROFILES}, got {profile!r}"
        )

    catalog_name = str(cfg.get("catalog.name", "spark_catalog"))
    prefix = f"spark.sql.catalog.{catalog_name}"
    conf: dict[str, str] = {"spark.sql.extensions": ICEBERG_EXTENSIONS}

    if profile == "glue":
        # `spark_catalog` must use SparkSessionCatalog so non-Iceberg tables
        # keep working; any other catalog name uses the plain SparkCatalog.
        impl = (
            "org.apache.iceberg.spark.SparkSessionCatalog"
            if catalog_name == "spark_catalog"
            else "org.apache.iceberg.spark.SparkCatalog"
        )
        conf[prefix] = impl
        conf[f"{prefix}.catalog-impl"] = "org.apache.iceberg.aws.glue.GlueCatalog"
        conf[f"{prefix}.io-impl"] = "org.apache.iceberg.aws.s3.S3FileIO"
        warehouse = cfg.get("catalog.glue.warehouse", None)
        if warehouse:
            conf[f"{prefix}.warehouse"] = str(warehouse)
        glue_id = cfg.get("catalog.glue.catalog_id", None)
        if glue_id:
            conf[f"{prefix}.glue.id"] = str(glue_id)

    elif profile == "s3tables":
        arn = cfg.require("catalog.s3tables.table_bucket_arn")
        conf[prefix] = "org.apache.iceberg.spark.SparkCatalog"
        conf[f"{prefix}.catalog-impl"] = "software.amazon.s3tables.iceberg.S3TablesCatalog"
        conf[f"{prefix}.warehouse"] = str(arn)
        conf[f"{prefix}.io-impl"] = "org.apache.iceberg.aws.s3.S3FileIO"
        region = cfg.get("catalog.s3tables.region", None)
        if region:
            conf[f"{prefix}.client.region"] = str(region)

    else:  # local
        warehouse = cfg.get("catalog.local.warehouse", "./.local_warehouse")
        conf[prefix] = "org.apache.iceberg.spark.SparkCatalog"
        conf[f"{prefix}.type"] = "hadoop"
        conf[f"{prefix}.warehouse"] = str(warehouse)

    extra = cfg.get("catalog.extra_conf", {}) or {}
    for key, value in extra.items():
        conf[str(key)] = str(value)

    return conf


def build_spark_session(cfg: Config, app_name: str) -> "SparkSession":
    """Create (or attach to) the SparkSession described by ``cfg``.

    ``spark.sql.session.timeZone`` is pinned to the configured zone (``EST`` by
    default). Silver timestamps are local wall-clock time for that zone, and
    Gold compares them against each other constantly - a lifecycle boundary, a
    port window, a device segment. A session on a different zone would shift
    every one of those comparisons by 4-5 hours.

    ``EST`` is a fixed -05:00 offset with no daylight saving, which is what the
    upstream feeds actually carry; ``America/Toronto`` observes DST and would
    read a summer timestamp an hour off. See ``conf/lineage/base.yaml``.
    """
    from pyspark.sql import SparkSession

    prefix = cfg.get("spark.app_name_prefix", "silver_to_gold")
    builder = SparkSession.builder.appName(f"{prefix}.{app_name}")

    for key, value in (cfg.get("spark.conf", {}) or {}).items():
        builder = builder.config(str(key), str(value))

    if bool(cfg.get("catalog.manage_spark_conf", True)):
        for key, value in _catalog_conf(cfg).items():
            builder = builder.config(key, value)

    for key, value in (cfg.get("spark.extra_conf", {}) or {}).items():
        builder = builder.config(str(key), str(value))

    spark = builder.getOrCreate()

    session_tz = str(cfg.get("spark.session_timezone", "EST"))
    spark.conf.set("spark.sql.session.timeZone", session_tz)

    log_level = cfg.get("spark.log_level", None)
    if log_level:
        spark.sparkContext.setLogLevel(str(log_level))

    # `transforms/common.py::truncate_lineage` prefers reliable checkpointing
    # over the local variant and falls back only when nobody has set a
    # directory. Nothing ever set one, so every iterative stage in this package
    # has been running on `localCheckpoint` - which keeps each round's blocks on
    # the executors that produced them. That is fine until an executor goes
    # away, at which point the blocks go with it and a forty-minute run fails
    # with no way to recover the round it was on; it also means the driver
    # tracks a block per partition per round for the life of the application,
    # and the account walk on the full-history slice runs up to
    # `MAX_COMPONENT_ROUNDS` = 200 rounds of three checkpoints each.
    #
    # Setting this is a config change, not a code path change: `truncate_lineage`
    # already has the branch. The directory has to be somewhere every executor
    # can write - on EMR that is S3, and the cluster role's `CheckpointReadWrite`
    # statement grants exactly that on the checkpoints bucket.
    checkpoint_dir = cfg.get("spark.checkpoint_dir", None)
    if checkpoint_dir:
        spark.sparkContext.setCheckpointDir(str(checkpoint_dir))
        LOGGER.info("checkpoint directory set | %s", checkpoint_dir)

    LOGGER.info(
        "SparkSession ready | app=%s | version=%s | catalog.profile=%s | session tz=%s",
        spark.sparkContext.appName,
        spark.version,
        cfg.get("catalog.profile", "local"),
        session_tz,
    )
    return spark


def shuffle_partitions(spark: SparkSession, default: int = 200) -> int:
    """How wide an RDD shuffle in this package should be.

    Five stages shuffle through the RDD API rather than through a DataFrame -
    the lifecycle walk, the device-segment walk, the TU corrector, and the
    account and customer chain walks - because each of them is Python that has
    to see one whole group together, and there is no DataFrame operator for
    "hand me this group and let me walk it". For the first three the group is a
    phone number; for the last two it is a connected component of the graph.

    ``rdd.groupBy`` with no partition count does not use
    ``spark.sql.shuffle.partitions``. It asks ``Partitioner.defaultPartitioner``,
    which takes the widest upstream RDD or ``spark.default.parallelism`` -
    neither of which anybody sets, and neither of which is what an operator
    reaches for when a shuffle is too coarse. So the heaviest shuffles in the
    pipeline were the ones the only knob could not reach. A full-history rebuild
    died in one of them: a shuffle fetch whose SASL handshake timed out waiting
    behind blocks that were far too large, four times over, forty-nine minutes
    in.

    Reading the SQL setting here makes the one number govern both halves, so
    ``spark.sql.shuffle.partitions`` means what it looks like it means.

    The default is Spark's own rather than the session's current value, for the
    case where the setting has been unset entirely; a local run with two rows
    pays for it in empty partitions and nothing else.
    """
    try:
        value = int(spark.conf.get("spark.sql.shuffle.partitions", str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def driver_memory_line(spark: SparkSession | None = None, what: str = "driver") -> str:
    """One line saying how much memory the driver has used and how much it has.

    Written because three full-history rebuilds died without leaving a single
    number behind. The driver process is killed, so nothing it was about to log
    gets logged, and what reaches the notebook is the session saying it can no
    longer reach its own REPL - no stage, no exception, no memory figure. The
    only way to know how close a run came is to have printed it while the run was
    still alive, at each point where a stage hands over.

    ``ru_maxrss`` is the high-water mark of the whole process, not the current
    size, which is the number that matters here: a peak that has already been
    freed is still a peak that could have been the one to cross the limit. It is
    kilobytes on Linux, which is where this runs. The ceiling comes from
    ``spark.driver.memory`` when the session will say - under a remote session it
    often will not, and then this says so rather than guessing.

    Read on the process that calls it. Under Glue's LIVY sessions that is the
    driver, which is the point. Under Spark Connect from a laptop it is the
    laptop, and the line is honest but uninteresting.
    """
    import resource

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    ceiling = "unknown"
    if spark is not None:
        try:
            ceiling = str(spark.conf.get("spark.driver.memory", None) or "unknown")
        except Exception:  # pragma: no cover - remote sessions refuse some keys
            ceiling = "unknown"
    return f"{what}: peak python rss {peak_mb:,.0f} MiB of spark.driver.memory={ceiling}"


def qualified_table_name(cfg: Config, table: str) -> str:
    """Prefix ``namespace.table`` with the configured catalog name when needed.

    ``spark_catalog`` is Spark's default, so it is left off. Any other catalog
    (for example the S3 Tables catalog) has to be named explicitly, and this is
    the single place that decision is made.
    """
    catalog_name = str(cfg.get("catalog.name", "spark_catalog"))
    if catalog_name == "spark_catalog":
        return table
    if table.startswith(f"{catalog_name}."):
        return table
    return f"{catalog_name}.{table}"


def table_exists(spark: SparkSession, table: str) -> bool:
    """Does this table exist? Asked in a way an Iceberg table survives.

    ``spark.catalog.tableExists`` is the obvious call and it is not safe against
    the catalog this pipeline actually runs on. It answers through Spark's v1
    catalog, which builds a Hive ``CatalogTable`` out of the metastore entry -
    and an Iceberg table registered in Glue has no ``InputFormat`` in its
    storage descriptor, because Iceberg does not need one. Recent Spark raises
    on that conversion rather than degrading:

        AnalysisException: org.apache.hadoop.hive.ql.metadata.HiveException:
        Unable to fetch table <name>. StorageDescriptor#InputFormat cannot be
        null for table: <name>

    The trap is that the v1 path is only reached for tables that *do* exist, so
    the check is fine on the run that creates the table and fails on every run
    afterwards. A rebuild wrote 353 million rows successfully and then died in
    the next cell reading them back.

    So: ask the cheap way, which is right for temporary views and for local
    Hadoop-catalog runs, and when the cheap way chokes resolve the name the way
    every read in this pipeline resolves it - through the v2 catalog, which on
    a session with the Iceberg extensions is the Iceberg one. That is a metadata
    load; no data is scanned.
    """
    try:
        return bool(spark.catalog.tableExists(table))
    except Exception as exc:  # noqa: BLE001 - catalogs raise assorted types
        LOGGER.debug("v1 existence check failed for %s (%s); resolving it as a "
                     "relation instead", table, exc)
    try:
        spark.table(table).schema  # noqa: B018 - resolution is the whole point
        return True
    except Exception:  # noqa: BLE001 - absent, or unreadable, which is the same
        return False
