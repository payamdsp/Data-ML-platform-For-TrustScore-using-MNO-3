"""Writing Gold: a scoped merge, not an append.

Bronze -> Silver appends immutable events and can therefore drop a duplicate
batch outright. Gold holds *states*, and a single late event can move a boundary
written months ago, so the write has to be able to update and to delete as well
as insert. That is one ``MERGE`` statement per table, scoped to the run's slice.

The delete branch is the dangerous part and most of this module exists to make
it safe. ``WHEN NOT MATCHED BY SOURCE ... THEN DELETE`` without the slice
predicate deletes every row the batch did not happen to produce - which, for a
batch of forty thousand phone numbers against a table of hundreds of millions,
is the entire table. The predicate is therefore never written by hand at a call
site: it comes from :class:`~.reader.SliceScope`, which refuses to produce one
for an empty slice, and the merge additionally refuses to proceed when the
deletion it is about to perform is out of proportion to the slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import Config, ConfigError
from ..logging_utils import get_logger
from ..schemas import SLICE_KEY, GoldTableSpec, gold_table, schema_to_ddl
from ..spark import qualified_table_name, table_exists
from .reader import SliceScope

__all__ = [
    "GoldSink",
    "IcebergGoldSink",
    "ParquetGoldSink",
    "MergeStats",
    "DeleteRatioExceeded",
    "build_sink",
]

LOGGER = get_logger(__name__)


class DeleteRatioExceeded(RuntimeError):
    """A merge would have *lost* more of the slice than ``write.max_delete_ratio``.

    Lost, not deleted, and the distinction is the whole of the check. The thing
    worth stopping a run for is a recomputation that produced far less than it
    should have - an input that was empty when it should not have been - and
    what that looks like is rows leaving the slice and nothing arriving in their
    place.

    A key migration looks nothing like that and used to be indistinguishable
    from it. ``acct_id`` is a hash of its chain's root lifecycle, so an account
    that gains an earlier lifecycle re-keys, every row of it deletes under the
    old key and re-inserts under the new one, and the *gross* delete count is
    100% of the slice while the table has not lost a single row. Measured
    gross, the ceiling refused exactly the case the incremental design is built
    around; the comment above this class even named "a row whose key migrated"
    as a legitimate deletion while the arithmetic below counted it as a fault.

    So the ratio is on the net: deletions in excess of insertions. A migration
    nets zero and passes. An empty input inserts nothing, nets the full delete
    count, and is stopped.
    """


@dataclass(frozen=True)
class MergeStats:
    """What one merge did. Written to the DQ metrics table as ``info``."""

    table: str
    slice_size: int
    source_rows: int
    target_rows_before: int
    inserted: int
    updated: int
    deleted: int
    delete_ratio: float

    def format_line(self) -> str:
        return (
            f"{self.table}: +{self.inserted} ~{self.updated} -{self.deleted} "
            f"(slice={self.slice_size}, target_before={self.target_rows_before}, "
            f"delete_ratio={self.delete_ratio:.4f})"
        )


# ==========================================================================
# sinks
# ==========================================================================


class GoldSink:
    """Base class. One instance per Gold table per run."""

    def __init__(self, spec: GoldTableSpec, cfg: Config, spark: SparkSession) -> None:
        self.spec = spec
        self.cfg = cfg
        self.spark = spark

    # -- to implement ------------------------------------------------------
    def describe(self) -> str:
        raise NotImplementedError

    def exists(self) -> bool:
        raise NotImplementedError

    def ensure_exists(self) -> None:
        raise NotImplementedError

    def read(self) -> DataFrame:
        raise NotImplementedError

    def merge(self, source: DataFrame, scope: SliceScope, run_ts: datetime) -> MergeStats:
        raise NotImplementedError

    # -- shared ------------------------------------------------------------
    def _match_condition(self, target: str = "t", source: str = "s") -> str:
        """The ``ON`` clause: the merge key, plus the slice key when it is not
        already part of the merge key.

        Including the slice key in the join even when it is redundant is not
        redundant to the *planner* - it is what lets Iceberg prune target files
        by their min/max on the sort key rather than scanning them all.
        """
        keys = list(self.spec.merge_keys)
        if SLICE_KEY not in keys:
            keys.append(SLICE_KEY)
        return " AND ".join(f"{target}.{k} = {source}.{k}" for k in keys)

    def _changed_condition(self, target: str = "t", source: str = "s") -> str:
        """``IS DISTINCT FROM`` over the payload columns.

        Without this guard, every run rewrites every row in its slice,
        ``updated_ts`` degrades into "the last time this phone number appeared
        in any batch", and the table accumulates delete files for no reason.
        With it, re-running an already-applied batch writes nothing.
        """
        payload = self.spec.payload_columns
        if not payload:  # pragma: no cover - every table has payload columns
            return "false"
        return " OR ".join(
            f"({target}.{c} IS DISTINCT FROM {source}.{c})" for c in payload
        )

    def _changed_rows(self, source: DataFrame, target_in_slice: DataFrame) -> DataFrame:
        """The source rows a matched-row update would actually rewrite.

        The DataFrame counterpart of :meth:`_changed_condition`, and it exists so
        the two sinks agree on the one number the convergence claim rests on. A
        re-run of an already-applied window is supposed to report ``+0 ~0 -0``;
        counting every matched row as an update reports ``~n`` instead and makes
        a converged run indistinguishable from one that rewrote its whole slice.

        Returns the rows of ``source`` whose payload differs from the target row
        with the same merge key, with the source's own columns and nothing else.
        """
        keys = list(self.spec.merge_keys)
        payload = list(self.spec.payload_columns)
        if not payload:  # pragma: no cover - every table has payload columns
            return source.limit(0)
        prior = target_in_slice.select(
            *keys, *[F.col(c).alias(f"__t_{c}") for c in payload]
        )
        condition = None
        for col in payload:
            term = ~F.col(col).eqNullSafe(F.col(f"__t_{col}"))
            condition = term if condition is None else (condition | term)
        return source.join(prior, keys, "inner").where(condition).select(*source.columns)

    def _updated_column(self) -> str | None:
        """The ``updated_ts``-style column this table carries, if it has one."""
        cols = set(self.spec.columns)
        for candidate in ("updated_ts", "last_updated_ts"):
            if candidate in cols:
                return candidate
        return None

    def _check_delete_ratio(
        self, stats_deleted: int, slice_rows: int, stats_inserted: int = 0
    ) -> float:
        """Veto a merge that would shrink the slice past ``write.max_delete_ratio``.

        Returns the *gross* delete ratio, which is what ``MergeStats`` reports
        and what the DQ metrics record - the number a reader wants is still "how
        much of this slice was rewritten". The number the ceiling is applied to
        is the net, for the reason in :class:`DeleteRatioExceeded`.

        ``stats_inserted`` defaults to zero so that a caller which has not
        worked out its insert count gets the old, stricter behaviour rather than
        an accidentally disabled check. Both sinks pass it.
        """
        ratio = 0.0 if slice_rows == 0 else stats_deleted / float(slice_rows)
        net_lost = max(stats_deleted - stats_inserted, 0)
        net_ratio = 0.0 if slice_rows == 0 else net_lost / float(slice_rows)
        ceiling = float(self.cfg.get("write.max_delete_ratio", 0.25))
        if ceiling >= 0 and net_ratio > ceiling:
            raise DeleteRatioExceeded(
                f"merge into {self.spec.name} would leave the slice {net_lost} "
                f"rows smaller: {stats_deleted} deleted against "
                f"{stats_inserted} inserted, out of {slice_rows} rows in the "
                f"slice ({net_ratio:.1%} net), above "
                f"write.max_delete_ratio={ceiling:.1%}. This is almost always a "
                "sign that a recomputation input was empty - a key migration "
                "deletes and inserts in equal numbers and does not reach this "
                "message. Check that the Silver reads for this slice returned "
                "rows before doing anything else. Re-run with "
                "--set write.max_delete_ratio=1.0 only after confirming the "
                "deletions are correct."
            )
        return ratio


class IcebergGoldSink(GoldSink):
    """Iceberg ``MERGE INTO``, scoped to the slice."""

    def __init__(self, spec: GoldTableSpec, cfg: Config, spark: SparkSession) -> None:
        super().__init__(spec, cfg, spark)
        section = cfg.section(f"sinks.{spec.name}")
        self.table = qualified_table_name(cfg, str(section.require("table")))
        self.location = section.get("location", None)
        # Version 2, not 3. Nothing in this pipeline needs v3 - MERGE,
        # row-level deletes and partition evolution are all v2 - and the query
        # engines these tables are read with do not all support v3 yet. One of
        # them refuses a v3 table outright rather than degrading, and that is
        # not recoverable at read time: a table cannot be moved back down a
        # format version, so the only remedy once it has been written at 3 is to
        # drop it and create it again. The cheaper place to be careful is here.
        self.format_version = str(section.get("format_version", "2"))
        self.create_if_missing = bool(section.get("create_if_missing", True))
        self.table_properties = dict(section.get("table_properties", {}) or {})
        self.use_not_matched_by_source = bool(
            cfg.get("write.use_not_matched_by_source", True)
        )

    def describe(self) -> str:
        return f"iceberg table {self.table}"

    def exists(self) -> bool:
        return table_exists(self.spark, self.table)

    def ensure_exists(self) -> None:
        if self.exists():
            return
        if not self.create_if_missing:
            raise ConfigError(
                f"gold table {self.table} does not exist and "
                f"sinks.{self.spec.name}.create_if_missing is false"
            )
        partition = ""
        if self.spec.partition_by:
            col = self.spec.partition_by[0]
            field = {f.name: f for f in self.spec.schema.fields}[col]
            expr = col if field.dataType.simpleString() == "date" else f"days({col})"
            partition = f"\nPARTITIONED BY ({expr})"
        props = {
            "format-version": self.format_version,
            # Iceberg chooses how a write is distributed from the table itself,
            # and a table with a declared write order - which the statement
            # below gives this one - defaults to `range`. That samples the
            # incoming data and then globally sorts it across the shuffle on
            # every single write. On an incremental slice it is affordable; on a
            # full rebuild it moves the whole history through one shuffle and
            # spills it to the executors' local volumes, which is where a
            # rebuild of the base tables ran out of disk.
            #
            # `hash` distributes by the partition column instead: no sampling
            # pass, no global sort, and each task still sorts locally by the
            # write order, which is what the per-file min/max pruning below
            # actually depends on. The cost is that two tasks can write files
            # covering the same range of the sort key, so pruning is a little
            # less sharp - a much smaller price than the write not completing.
            "write.distribution-mode": "hash",
            # How many files one read task is allowed to open at once, said
            # indirectly.
            #
            # Iceberg's columnar reader opens a Parquet reader - and with it an
            # S3 connection - for every file in a task group, in the reader's
            # constructor, before a row comes back. Planning bin-packs files
            # into groups of `read.split.target-size` (128MB) and charges each
            # file at least `read.split.open-file-cost`, which defaults to 4MB.
            # A table of small files therefore hands one task up to thirty-two
            # files, and one executor up to thirty-two times its core count of
            # simultaneous connections. The client pools about fifty. Reading
            # this table back is how the lifecycle rebuild died, four times, on
            # `Timeout waiting for connection from pool`.
            #
            # Charging every file the full target size packs one file to a
            # task, so an executor holds one connection per core whatever the
            # files look like and the pool size stops being able to decide
            # whether the job runs. Large files are unaffected: they already
            # weighed more than the target and were already alone in their
            # group. Small files cost more tasks, which is scheduling overhead
            # measured against a stage that does not complete.
            #
            # `write.distribution-mode: hash` above is what makes this matter
            # here rather than in the abstract - it writes one file per
            # partition per task, so the tables this sink produces are exactly
            # the many-small-files shape the default is wrong for.
            "read.split.open-file-cost": "134217728",
            **self.table_properties,
        }
        prop_sql = ", ".join(f"'{k}' = '{v}'" for k, v in props.items())
        location = f"\nLOCATION '{self.location}'" if self.location else ""
        self.spark.sql(
            f"CREATE TABLE IF NOT EXISTS {self.table} (\n"
            f"    {schema_to_ddl(self.spec.schema)}\n"
            f") USING iceberg{partition}{location}\n"
            f"TBLPROPERTIES ({prop_sql})"
        )
        # Write order is a requirement, not a preference: phone_number_AC_hash
        # is not the partition column, so per-file min/max on the sort key is
        # the only pruning a slice-scoped merge can use.
        order = ", ".join(self.spec.sort_by)
        self.spark.sql(f"ALTER TABLE {self.table} WRITE ORDERED BY {order}")
        LOGGER.info("created %s ordered by (%s)", self.table, order)

    def read(self) -> DataFrame:
        if not self.exists():
            return self.spark.createDataFrame([], self.spec.schema)
        return self.spark.table(self.table)

    def merge(self, source: DataFrame, scope: SliceScope, run_ts: datetime) -> MergeStats:
        self.ensure_exists()
        view = f"src_{self.spec.name}"
        source.createOrReplaceTempView(view)

        target_slice = scope.apply(self.read())
        target_rows_before = target_slice.count()
        source_rows = source.count()

        # Work out what the merge is about to do *before* doing it, so the
        # delete ceiling can veto it. Two anti-joins against the merge key.
        keys = list(self.spec.merge_keys)
        to_delete = target_slice.join(source, keys, "left_anti").count()
        to_insert = source.join(target_slice, keys, "left_anti").count()
        # Only the matched rows whose payload actually differs are updates. The
        # join is skipped when nothing can match, which is the whole of a full
        # rebuild - the case where the extra pass would cost the most.
        matched = max(source_rows - to_insert, 0)
        to_update = (
            0
            if target_rows_before == 0 or matched == 0
            else self._changed_rows(source, target_slice).count()
        )
        self._check_delete_ratio(to_delete, target_rows_before, to_insert)

        run_ts_lit = f"TIMESTAMP '{run_ts.strftime('%Y-%m-%d %H:%M:%S')}'"
        payload = [c for c in self.spec.payload_columns]
        updated_col = self._updated_column()
        set_clause = ",\n         ".join(
            [f"t.{c} = s.{c}" for c in payload]
            + ([f"t.{updated_col} = {run_ts_lit}"] if updated_col else [])
        )
        insert_cols = ", ".join(self.spec.columns)
        insert_vals = ", ".join(
            run_ts_lit if c in ("created_ts", "updated_ts", "last_updated_ts") else f"s.{c}"
            for c in self.spec.columns
        )

        # The slice predicate is applied to the target on both the match and the
        # delete branch. It comes from SliceScope, which refuses to produce one
        # for an empty slice.
        slice_pred = scope.predicate_sql("t." + SLICE_KEY) if scope.inlineable else None
        on_clause = self._match_condition()
        if slice_pred:
            on_clause = f"{on_clause}\n  AND {slice_pred}"

        sql = (
            f"MERGE INTO {self.table} t\n"
            f"USING {view} s\n"
            f"   ON {on_clause}\n"
            f"WHEN MATCHED AND ({self._changed_condition()})\n"
            f"     THEN UPDATE SET {set_clause}\n"
            f"WHEN NOT MATCHED\n"
            f"     THEN INSERT ({insert_cols}) VALUES ({insert_vals})"
        )
        inline_delete = self._delete_rides_in_merge(to_delete, scope)
        if inline_delete:
            sql += f"\nWHEN NOT MATCHED BY SOURCE AND {slice_pred}\n     THEN DELETE"

        LOGGER.info(
            "merging %s | slice=%d | source=%d | target_in_slice=%d "
            "| +%d -%d (pre-count)",
            self.spec.name,
            scope.size,
            source_rows,
            target_rows_before,
            to_insert,
            to_delete,
        )
        self.spark.sql(sql)

        if to_delete and not inline_delete:
            # Same outcome as the in-merge branch, reached in two statements
            # instead of one. It is not the default because between the MERGE
            # and this DELETE the table is briefly in a state neither run
            # produced: the new rows are in, and the rows that should have gone
            # are still there. Readers of an Iceberg table see whole snapshots,
            # so nobody observes a half-written statement - but they can
            # observe the intermediate snapshot, which the single-statement
            # form never creates.
            #
            # The ordering is deliberate all the same. Merging first and
            # deleting second means the transient state has too *many* rows;
            # the other order would mean too few, and a downstream read that
            # landed in that window would silently miss data rather than
            # double-count it.
            self._delete_missing(source, scope)

        stats = MergeStats(
            table=self.spec.name,
            slice_size=scope.size,
            source_rows=source_rows,
            target_rows_before=target_rows_before,
            inserted=to_insert,
            updated=to_update,
            deleted=to_delete,
            delete_ratio=(
                0.0 if target_rows_before == 0 else to_delete / float(target_rows_before)
            ),
        )
        LOGGER.info(stats.format_line())
        return stats

    def _delete_rides_in_merge(self, to_delete: int, scope: SliceScope) -> bool:
        """Do the deletes belong inside the ``MERGE``, or in their own statement?

        Three things have to be true for the ``WHEN NOT MATCHED BY SOURCE``
        branch to be both legal and worth having.

        **It is switched on.** ``write.use_not_matched_by_source`` exists so a
        deployment that cannot use the branch - an engine that does not support
        it - can turn it off without a code change.

        **There is at least one row to delete.** ``merge`` counts this with an
        anti-join before it builds the statement, so asking is free. On a first
        run the target table is empty and the answer is always zero; on an
        ordinary incremental run it is usually zero too, because a slice
        normally re-derives the same rows it already had. Leaving out a clause
        that would match nothing costs nothing and removes the only part of the
        statement capable of deleting data.

        **The slice is small enough to inline as literals.** This is the one
        that bit us, and it is why this method exists rather than a one-line
        condition. Above ``slice.inline_limit`` - 20,000 phone numbers -
        :class:`SliceScope` stops handing out a literal ``IN`` list and
        :meth:`_staged_slice_predicate` falls back to
        ``t.<key> IN (SELECT <key> FROM <view>)``. Spark 3.5 refuses that::

            [UNSUPPORTED_MERGE_CONDITION.SUBQUERY] MERGE operation contains
            unsupported DELETE condition. Subqueries are not allowed:
            "(phone_number_AC_hash IN (listquery()))"

        No branch of a ``MERGE`` condition may contain a subquery. The first
        sandbox bootstrap run met this after thirty-two minutes of work, on the
        very last statement, with a slice of 1,391,563 phone numbers. Nothing
        local could have caught it: every fixture slice is orders of magnitude
        below the inline limit, so every test took the literal path.

        When the answer is ``False`` and there is still something to delete,
        :meth:`_delete_missing` does it in a separate ``DELETE FROM ... WHERE``,
        which *is* allowed to contain a subquery.
        """
        return bool(self.use_not_matched_by_source) and to_delete > 0 and scope.inlineable

    def _staged_slice_predicate(self, scope: SliceScope) -> str:
        """Slice predicate for a slice too large to inline as literals.

        Only reachable from :meth:`_delete_missing`. It must never reach a
        ``MERGE`` - see :meth:`_delete_rides_in_merge` for why.
        """
        view = f"slice_{self.spec.name}"
        scope.frame.createOrReplaceTempView(view)
        return f"t.{SLICE_KEY} IN (SELECT {SLICE_KEY} FROM {view})"

    def _delete_missing(self, source: DataFrame, scope: SliceScope) -> None:
        view = f"src_del_{self.spec.name}"
        source.createOrReplaceTempView(view)
        keys = " AND ".join(f"t.{k} = s.{k}" for k in self.spec.merge_keys)
        pred = (
            scope.predicate_sql("t." + SLICE_KEY)
            if scope.inlineable
            else self._staged_slice_predicate(scope)
        )
        self.spark.sql(
            f"DELETE FROM {self.table} t WHERE {pred} "
            f"AND NOT EXISTS (SELECT 1 FROM {view} s WHERE {keys})"
        )


class ParquetGoldSink(GoldSink):
    """Local / CI sink with merge semantics emulated in a read-modify-write.

    Not for production - it rewrites the whole dataset - but the *outcome* is
    identical to the Iceberg merge, which is the point: the same tests can
    assert on convergence, key migration and deletion without a catalog.
    """

    def __init__(self, spec: GoldTableSpec, cfg: Config, spark: SparkSession) -> None:
        super().__init__(spec, cfg, spark)
        section = cfg.section(f"sinks.{spec.name}")
        self.path = str(section.require("path"))

    def describe(self) -> str:
        return f"parquet {self.path}"

    def exists(self) -> bool:
        try:
            self.spark.read.parquet(self.path).schema
            return True
        except Exception:
            return False

    def ensure_exists(self) -> None:
        if not self.exists():
            self.spark.createDataFrame([], self.spec.schema).write.mode(
                "overwrite"
            ).parquet(self.path)

    def read(self) -> DataFrame:
        if not self.exists():
            return self.spark.createDataFrame([], self.spec.schema)
        return self.spark.read.parquet(self.path).select(
            *[F.col(f.name).cast(f.dataType) for f in self.spec.schema.fields]
        )

    def merge(self, source: DataFrame, scope: SliceScope, run_ts: datetime) -> MergeStats:
        self.ensure_exists()
        existing = self.read()
        in_slice = scope.apply(existing)
        target_rows_before = in_slice.count()
        source_rows = source.count()

        keys = list(self.spec.merge_keys)
        to_delete = in_slice.join(source, keys, "left_anti").count()
        to_insert = source.join(in_slice, keys, "left_anti").count()
        # The changed-row guard, which is `WHEN MATCHED AND (... IS DISTINCT
        # FROM ...)` on the Iceberg side. Without it this sink rewrote every
        # matched row with a fresh `updated_ts` and reported it as an update, so
        # the convergence check the README advertises - run it twice, get
        # `+0 ~0 -0` - could not be demonstrated on the sink that the local run
        # and the tests actually use.
        changed = (
            source.limit(0)
            if target_rows_before == 0
            else self._changed_rows(source, in_slice)
        ).persist()
        to_update = changed.count()
        self._check_delete_ratio(to_delete, target_rows_before, to_insert)

        merged = self._merged_slice(source, changed, in_slice, keys, run_ts)

        untouched = existing.join(
            scope.frame.withColumnRenamed(SLICE_KEY, "__k"),
            F.col(SLICE_KEY) == F.col("__k"),
            "left_anti",
        )
        out = untouched.unionByName(merged)
        tmp = self.path.rstrip("/") + "__tmp"
        out.write.mode("overwrite").parquet(tmp)
        self.spark.read.parquet(tmp).write.mode("overwrite").parquet(self.path)
        changed.unpersist()

        stats = MergeStats(
            table=self.spec.name,
            slice_size=scope.size,
            source_rows=source_rows,
            target_rows_before=target_rows_before,
            inserted=to_insert,
            updated=to_update,
            deleted=to_delete,
            delete_ratio=(
                0.0 if target_rows_before == 0 else to_delete / float(target_rows_before)
            ),
        )
        LOGGER.info(stats.format_line())
        return stats

    def _merged_slice(
        self,
        source: DataFrame,
        changed: DataFrame,
        in_slice: DataFrame,
        keys: list[str],
        run_ts: datetime,
    ) -> DataFrame:
        """The slice's rows after the merge: written, kept, or dropped.

        Three disjoint groups, matching the three branches of the Iceberg
        statement. Source rows that are new, or that matched and differ, are
        *written* - with ``created_ts`` carried over from the row that was
        already there, which is the whole difference between "first seen" and
        "last recomputed". Rows that matched and do not differ are *kept*
        exactly as they stand, ``updated_ts`` included; that is the changed-row
        guard, and it is what lets a re-run leave the table untouched rather
        than merely leave it holding the same values. Target rows the source no
        longer produces are in neither group and are therefore dropped, which is
        ``WHEN NOT MATCHED BY SOURCE ... THEN DELETE``.
        """
        cols = list(self.spec.columns)
        unchanged_keys = (
            in_slice.select(*keys)
            .join(source.select(*keys), keys, "left_semi")
            .join(changed.select(*keys), keys, "left_anti")
        )
        written = self._carry_created_ts(
            source.join(unchanged_keys, keys, "left_anti"), in_slice, run_ts
        )
        kept = in_slice.join(unchanged_keys, keys, "left_semi")
        return written.select(*cols).unionByName(kept.select(*cols))

    def _carry_created_ts(
        self, source: DataFrame, existing_in_slice: DataFrame, run_ts: datetime
    ) -> DataFrame:
        if "created_ts" not in self.spec.columns:
            return source
        keys = list(self.spec.merge_keys)
        prior = existing_in_slice.select(
            *keys, F.col("created_ts").alias("__prior_created_ts")
        )
        joined = source.join(prior, keys, "left")
        return joined.withColumn(
            "created_ts",
            F.coalesce(F.col("__prior_created_ts"), F.col("created_ts")),
        ).drop("__prior_created_ts")


def build_sink(table: str, cfg: Config, spark: SparkSession) -> GoldSink:
    """Construct the sink configured for ``table``."""
    spec = gold_table(table)
    mode = str(cfg.get(f"sinks.{table}.mode", "iceberg")).lower()
    if mode == "iceberg":
        return IcebergGoldSink(spec, cfg, spark)
    if mode == "parquet":
        return ParquetGoldSink(spec, cfg, spark)
    raise ConfigError(f"sinks.{table}.mode must be iceberg or parquet, got {mode!r}")
