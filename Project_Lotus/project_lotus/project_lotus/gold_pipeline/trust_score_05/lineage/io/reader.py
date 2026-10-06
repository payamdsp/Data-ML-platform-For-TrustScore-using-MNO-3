"""Reading Silver and Gold, always scoped.

Two kinds of read happen in this pipeline and they must not be confused:

**The batch read** picks up what is new - Silver rows with
``ingestion_ts`` inside the run's watermark window. It answers "what arrived?"
and it is small.

**The slice read** picks up the *complete history* of the phone numbers the
batch touches, ignoring the watermark entirely. It answers "what do I need in
order to recompute those phone numbers correctly?" and it is the read that
matters for cost.

Getting these the wrong way round is the classic incremental-pipeline bug: you
recompute a lifecycle from the three events that arrived last night, having
never read the forty that came before, and write a lifecycle that starts in the
wrong place.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Sequence

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import Config, ConfigError
from ..logging_utils import get_logger
from ..schemas import SLICE_KEY, silver_schema_for
from ..spark import is_missing_table_error, qualified_table_name, table_exists

__all__ = [
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
    "SliceScope",
]

LOGGER = get_logger(__name__)

#: Above this many phone numbers the slice stops being pushed down as a literal
#: ``IN`` list and becomes a semi-join instead. A literal list is by far the
#: better plan while it fits - Iceberg turns it into per-file min/max pruning on
#: the sort key - but a million-element list will not survive query planning.
DEFAULT_SLICE_INLINE_LIMIT = 20_000


class SliceScope:
    """The set of phone numbers a run is allowed to read and write.

    Wraps the hashes together with the decision about *how* to apply them, so
    that every read, every merge and every delete branch in the run uses one
    object and cannot disagree about the scope. That matters most for the delete
    branch: a merge that deletes on a wider scope than it recomputed would
    remove rows it never looked at.

    An empty scope is never silently allowed. Section 7.2 of the design document
    explains what an empty slice would do to a ``NOT MATCHED BY SOURCE`` delete.
    """

    __slots__ = ("_frame", "_hashes", "_inline_limit", "_size")

    def __init__(
        self,
        frame: DataFrame,
        size: int,
        hashes: Sequence[str] | None = None,
        inline_limit: int = DEFAULT_SLICE_INLINE_LIMIT,
    ) -> None:
        self._frame = frame
        self._size = int(size)
        self._hashes = tuple(hashes) if hashes is not None else None
        self._inline_limit = int(inline_limit)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_hashes(
        cls,
        spark: SparkSession,
        hashes: Iterable[str],
        inline_limit: int = DEFAULT_SLICE_INLINE_LIMIT,
    ) -> "SliceScope":
        values = sorted({h for h in hashes if h})
        frame = spark.createDataFrame(
            [(h,) for h in values], schema=f"{SLICE_KEY} string"
        )
        # The same limit :meth:`from_frame` applies, and for the same reason. A
        # scope built from an explicit list is usually small - it comes from
        # ``--phone-numbers`` or from a repair file - but "usually" is not a
        # guarantee, and a repair run over a million phone numbers would
        # otherwise inline a million literals into the merge predicate and die
        # in query planning rather than in anything a reader could diagnose.
        keep = values if len(values) <= int(inline_limit) else None
        return cls(frame, len(values), keep, inline_limit)

    @classmethod
    def from_frame(
        cls,
        frame: DataFrame,
        inline_limit: int = DEFAULT_SLICE_INLINE_LIMIT,
    ) -> "SliceScope":
        distinct = frame.select(SLICE_KEY).where(F.col(SLICE_KEY).isNotNull()).distinct()
        distinct = distinct.persist()
        size = distinct.count()
        hashes = None
        if size <= inline_limit:
            hashes = tuple(sorted(r[0] for r in distinct.collect()))
        return cls(distinct, size, hashes, inline_limit)

    # -- properties --------------------------------------------------------
    @property
    def size(self) -> int:
        return self._size

    @property
    def frame(self) -> DataFrame:
        return self._frame

    @property
    def hashes(self) -> tuple[str, ...] | None:
        """The literal hashes, or ``None`` when the slice is too large to inline."""
        return self._hashes

    @property
    def inlineable(self) -> bool:
        return self._hashes is not None

    def is_empty(self) -> bool:
        return self._size == 0

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._size

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SliceScope(size={self._size}, inlineable={self.inlineable})"

    # -- application -------------------------------------------------------
    def apply(self, df: DataFrame, column: str = SLICE_KEY) -> DataFrame:
        """Restrict ``df`` to the slice.

        Refuses an empty slice rather than returning an empty frame. An empty
        frame here would read as "this phone number has no rows", which the
        delete branch would act on.
        """
        if self.is_empty():
            raise ValueError(
                "refusing to apply an empty slice; an empty scope makes every "
                "delete branch unbounded (see incremental_design.md section 7.2)"
            )
        if self._hashes is not None:
            return df.where(F.col(column).isin(list(self._hashes)))
        keys = self._frame.withColumnRenamed(SLICE_KEY, "__slice_key")
        return df.join(
            F.broadcast(keys) if self._size <= self._inline_limit else keys,
            F.col(column) == F.col("__slice_key"),
            "left_semi",
        )

    def predicate_sql(self, column: str = SLICE_KEY) -> str:
        """The slice as a SQL predicate, for the ``MERGE ... ON`` clause.

        Only available for an inlineable slice; a larger one is applied by
        joining the staged frame instead.
        """
        if self._hashes is None:
            raise ValueError(
                f"slice of {self._size} keys is too large to inline as SQL; "
                "the merge must stage the slice frame instead"
            )
        if not self._hashes:
            raise ValueError("refusing to build a predicate from an empty slice")
        literals = ", ".join("'" + h.replace("'", "''") + "'" for h in self._hashes)
        return f"{column} IN ({literals})"


# ==========================================================================
# reading
# ==========================================================================


def _source_config(cfg: Config, dataset: str) -> Config:
    try:
        return cfg.section(f"sources.{dataset}")
    except ConfigError as exc:
        known = ", ".join(sorted((cfg.get("sources", {}) or {}).keys())) or "<none>"
        raise ConfigError(
            f"no source configuration for {dataset!r}; configured sources: {known}"
        ) from exc


def source_enabled(cfg: Config, dataset: str) -> bool:
    """Whether a configured source should be read at all.

    Only the enrichment feeds carry this. The two carrier feeds are the pipeline
    and turning one off would not produce a smaller answer, it would produce a
    wrong one; the enrichment feeds genuinely may be absent, and a run without
    them is the pipeline as it behaved before enrichment existed. That is a
    supported state rather than a degraded one, which is why it is a config key
    and not an exception.

    A source with no entry at all is disabled, not an error. That is what lets an
    overlay written before these tables existed keep working.
    """
    sources = cfg.get("sources", {}) or {}
    if dataset not in sources:
        return False
    return bool((sources.get(dataset) or {}).get("enabled", True))


def watermark_column(cfg: Config, dataset: str) -> str:
    """The column a batch read watermarks on for this dataset.

    Defaults to ``ingestion_ts``, which is what the carrier feeds carry. The IDV
    tables carry ``api_ts`` instead and TU carries ``ingestion_ts``, so the name
    is configurable rather than assumed - and it is read from config rather than
    guessed from the schema, because two of these tables have more than one
    timestamp column and picking the wrong one would silently narrow the batch.
    """
    sources = cfg.get("sources", {}) or {}
    entry = sources.get(dataset) or {}
    return str(entry.get("watermark_column", "ingestion_ts"))


def read_silver(spark: SparkSession, cfg: Config, dataset: str) -> DataFrame:
    """Read a whole Silver table (or its local sample), unscoped and unfiltered.

    Nothing calls this directly except the two scoped helpers below and the
    tests. The full table is never materialised - Spark is lazy and every caller
    immediately applies either a watermark or a slice - but it is still worth
    being explicit that this function on its own is not a bounded read.
    """
    src = _source_config(cfg, dataset)
    mode = str(src.get("mode", "table")).lower()
    schema = silver_schema_for(dataset)

    if mode == "table":
        table = qualified_table_name(cfg, str(src.require("table")))
        LOGGER.info("reading silver | dataset=%s | table=%s", dataset, table)
        pin = src.get("_read_snapshot", None)
        if pin is not None and pin["missing"]:
            # An optional feed absent when this run began stays absent on retry.
            return spark.createDataFrame([], schema)
        if pin is not None and pin["snapshot_id"] is not None:
            df = spark.read.option("snapshot-id", str(pin["snapshot_id"])).format("iceberg").load(table)
        else:
            df = spark.table(table)
            if pin is not None:
                # The table had no snapshot at run start. New arrivals belong
                # to another run; still validate this table's column contract.
                df = df.limit(0)
    elif mode in ("csv", "sample"):
        path = str(src.require("path"))
        LOGGER.info("reading silver | dataset=%s | csv=%s", dataset, path)
        # ``enforceSchema`` defaults to true in Spark, which maps the supplied
        # schema onto a header-bearing CSV *by position* and silently shifts
        # every column after a missing one. The check below only catches a
        # column that is absent from the frame, and positional mapping means it
        # never is - the name is taken from the schema, not from the file. So
        # the read itself has to be the thing that refuses: with the option off,
        # a sample that has drifted from its declared schema raises by name here
        # instead of arriving as a frame full of plausible values in the wrong
        # columns.
        options = {
            "header": "true",
            "enforceSchema": "false",
            **(src.get("options", {}) or {}),
        }
        df = spark.read.options(**{k: str(v) for k, v in options.items()}).schema(
            schema
        ).csv(path)
    elif mode == "parquet":
        path = str(src.require("path"))
        LOGGER.info("reading silver | dataset=%s | parquet=%s", dataset, path)
        df = spark.read.parquet(path)
    else:
        raise ConfigError(
            f"sources.{dataset}.mode must be one of table/csv/parquet, got {mode!r}"
        )

    # Glue may lowercase the client's camelCase names. Map by name (never by
    # position), and preserve the client's names inside the transformation code.
    # Bigint record_id values from our Silver pipeline are read as strings for
    # the client's deterministic event-ID seed; Silver itself is not altered.
    by_lower = {}
    for name in df.columns:
        if name.lower() in by_lower:
            raise ConfigError(f"silver {dataset} has ambiguous column names: {df.columns}")
        by_lower[name.lower()] = name
    aliases = src.get("column_aliases", {}) or {}
    names = {
        field.name: by_lower.get(str(aliases.get(field.name, field.name)).lower())
        for field in schema.fields
    }
    missing = [name for name, actual in names.items() if actual is None]
    if missing:
        raise ConfigError(
            f"silver {dataset} is missing expected column(s) {missing}; "
            f"found {df.columns}"
        )
    # ANSI mode is enabled in the Lotus configuration: malformed numeric/date
    # casts raise an error instead of quietly turning populated values into NULL.
    # Timestamp-NTZ Silver values are interpreted in the configured business zone.
    return df.select(*[
        F.col("`" + names[f.name].replace("`", "``") + "`")
        .cast(f.dataType).alias(f.name)
        for f in schema.fields
    ])


def read_optional_silver(
    spark: SparkSession, cfg: Config, dataset: str
) -> DataFrame:
    """:func:`read_silver`, but an absent feed reads as empty rather than failing.

    Only the enrichment feeds go through this. Three states have to be
    distinguished and only one of them is a fault:

    * **disabled**, or not configured at all - an empty frame, logged at info.
    * **configured but not there yet** - an empty frame, logged as a warning.
      This is the ordinary state of the three IDV tables before the first round
      of the handshake has completed, so it must not fail a run.
    * **there but wrong** - a missing column, a bad mode - still raises, because
      that is a contract violation and the run would otherwise produce a lineage
      built from evidence it silently failed to read.

    The empty frame carries the declared schema, so every downstream join,
    window and aggregation behaves exactly as it would against real rows. An
    empty frame with no schema would fail at the first join instead, which is a
    long way from the cause.
    """
    schema = silver_schema_for(dataset)
    empty = spark.createDataFrame([], schema)

    if not source_enabled(cfg, dataset):
        LOGGER.info("silver %s is disabled or unconfigured; reading as empty", dataset)
        return empty

    try:
        df = read_silver(spark, cfg, dataset)
    except ConfigError:
        # A contract violation. Missing columns, an unknown mode, no path.
        raise
    except Exception as exc:
        # Only an absent OPTIONAL table is equivalent to an empty feed. IAM,
        # Lake Formation, network and corrupt-metadata failures must stop the
        # run; otherwise a rebuild could erase previously used evidence.
        if not is_missing_table_error(exc):
            raise
        LOGGER.warning(
            "optional silver %s does not exist yet (%s); reading an empty frame",
            dataset,
            exc,
        )
        return empty

    # A table that exists and is genuinely empty is the same state as one that
    # is not there yet, and is equally normal. It is logged so that a run which
    # expected enrichment and got none says so once, in one place.
    return df


def read_gold(spark: SparkSession, cfg: Config, table: str) -> DataFrame:
    """Read an existing Gold table by short name.

    Used by the slice closure, which needs the current account and customer
    mappings to work out which phone numbers a batch drags in with it.
    """
    from ..schemas import gold_table

    spec = gold_table(table)
    sink_cfg = cfg.section(f"sinks.{table}")
    mode = str(sink_cfg.get("mode", "iceberg")).lower()
    if mode == "iceberg":
        name = qualified_table_name(cfg, str(sink_cfg.require("table")))
        pin = sink_cfg.get("_read_snapshot", None)
        if pin is not None:
            if pin["missing"] or pin["snapshot_id"] is None:
                return spark.createDataFrame([], spec.schema)
            return spark.read.option("snapshot-id", str(pin["snapshot_id"])).format("iceberg").load(name)
        if not table_exists(spark, name):
            LOGGER.warning(
                "gold table %s does not exist yet; treating it as empty", name
            )
            return spark.createDataFrame([], spec.schema)
        return spark.table(name)
    path = str(sink_cfg.require("path"))
    try:
        return spark.read.parquet(path)
    except Exception:  # pragma: no cover - first run against an empty warehouse
        LOGGER.warning("gold path %s not readable yet; treating it as empty", path)
        return spark.createDataFrame([], spec.schema)


def gold_exists(spark: SparkSession, cfg: Config, table: str) -> bool:
    """Has a Gold table by this short name been written yet?

    The question :func:`read_gold` answers silently - it returns an empty frame
    and logs - asked out loud, for the callers that would rather refuse than
    read nothing. The rebuild notebooks are all of them: each one opens by
    checking that the notebook before it has run, so that a missing upstream
    surfaces as "run notebook 03 first" rather than as an AnalysisException
    naming a table the reader has never heard of, forty minutes later.

    It dispatches on ``sinks.<name>.mode`` for the same reason ``read_gold``
    does, and getting that wrong is not hypothetical: the notebooks asked
    ``table_exists`` on the catalog name directly, which is right when the sink
    is Iceberg and wrong whenever it is not. A parquet-mode run - the test
    harness, and any local rehearsal - wrote the upstream table to
    ``sinks.<name>.path`` and was then told by the next notebook that it did not
    exist, naming a catalog table nothing in that run was ever going to create.
    The guard that exists to say which notebook to run was the thing stopping
    the run.

    Existence for a parquet sink means readable *and* non-empty. Two states are
    folded together there and deliberately: the directory is created by the
    write, so "not written yet" and "written empty" are not reliably
    distinguishable from outside, and both are states the caller wants to refuse
    on. An Iceberg sink keeps them apart - the table exists or it does not - and
    the notebooks check emptiness themselves on the line after, which is where
    the message can say which table was empty.
    """
    sink_cfg = cfg.section(f"sinks.{table}")
    mode = str(sink_cfg.get("mode", "iceberg")).lower()
    if mode == "iceberg":
        return table_exists(spark, qualified_table_name(cfg, str(sink_cfg.require("table"))))
    path = str(sink_cfg.require("path"))
    try:
        return not spark.read.parquet(path).limit(1).rdd.isEmpty()
    except Exception:  # noqa: BLE001 - an unreadable path is a path not written
        return False


def gold_location(cfg: Config, table: str) -> str:
    """Where a Gold table lives, named the way its own sink mode names it.

    So that a notebook refusing to start can print the thing the operator would
    go and look at - a catalog table for an Iceberg sink, a path for a parquet
    one - instead of a catalog name that a parquet run was never going to use.
    """
    sink_cfg = cfg.section(f"sinks.{table}")
    mode = str(sink_cfg.get("mode", "iceberg")).lower()
    if mode == "iceberg":
        return qualified_table_name(cfg, str(sink_cfg.require("table")))
    return str(sink_cfg.require("path"))


def select_batch(
    df: DataFrame,
    watermark_from: datetime | None,
    watermark_to: datetime,
    ts_col: str = "ingestion_ts",
) -> DataFrame:
    """Apply the run's watermark window: ``(watermark_from, watermark_to]``.

    Half-open at the lower end and closed at the upper so that consecutive runs
    tile the timeline exactly once - no gap, no overlap - when each run's
    ``watermark_from`` is the previous run's ``watermark_to``.

    The watermark is on ``ingestion_ts`` rather than ``event_date`` deliberately.
    Event date moves backwards: a file delivered today can carry last month's
    events, and a high-water mark on event date would skip them forever.
    Ingestion time is stamped at write and only moves forward.
    """
    out = df.where(F.col(ts_col) <= F.lit(watermark_to))
    if watermark_from is not None:
        out = out.where(F.col(ts_col) > F.lit(watermark_from))
    return out


def restrict_to_slice(
    df: DataFrame, scope: SliceScope, column: str = SLICE_KEY
) -> DataFrame:
    """Restrict a frame to the run's slice. See :meth:`SliceScope.apply`."""
    return scope.apply(df, column)


def slice_frame(
    spark: SparkSession,
    cfg: Config,
    dataset: str,
    scope: SliceScope,
) -> DataFrame:
    """The complete Silver history of the slice's phone numbers.

    No watermark. This is the read every transform works from, and the reason
    the transforms have no lookback parameter and no boundary case at the start
    of a batch: the history is always complete, so a window over it never
    straddles the edge of the data.
    """
    return restrict_to_slice(read_silver(spark, cfg, dataset), scope)


def optional_slice_frame(
    spark: SparkSession,
    cfg: Config,
    dataset: str,
    scope: SliceScope,
) -> DataFrame:
    """:func:`slice_frame` over an enrichment feed that may not exist yet.

    The slice is still applied to the empty frame rather than skipped, so the
    disabled path and the populated path differ in their rows and in nothing
    else. A branch that returned an unsliced frame in one case would be a scope
    leak waiting for the day somebody enables the feed.
    """
    return restrict_to_slice(read_optional_silver(spark, cfg, dataset), scope)
