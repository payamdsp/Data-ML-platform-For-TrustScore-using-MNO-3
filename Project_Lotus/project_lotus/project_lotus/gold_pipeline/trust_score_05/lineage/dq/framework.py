"""Data-quality framework: check definitions, runner, and the metrics sink.

Severity model
--------------
Checks are split into three tiers, which is what makes this usable in production
against real carrier feeds:

``abort``
    Structural / pipeline-integrity failures. A null or malformed ``record_id``,
    a non-positive ``source_event_id``, a null partition column - these mean the
    *pipeline* is broken, not that a carrier sent messy data. The job records
    the metrics and then refuses to append.

``warn``
    Business-contract failures. Null ``imei`` on an IMSI event, an ``mno`` /
    ``event_type`` pair outside the documented matrix, a hash that is not
    64-char hex. Per the product owner's note these are *KPIs to surface with carriers*,
    not reasons to block ingestion - the real 412M-row batch already violates
    several of them. They are counted, written to the metrics table, and the
    append proceeds.

``info``
    Observability only. Columns that are *known* to be sparse (``tac``,
    ``oldIMSI``, ``oldIMEI``, ``fromADCSnapshot``, ``subId``) would produce
    permanent, meaningless warnings if tiered as ``warn``. They are still
    measured and trended - a jump in the null rate is the signal - but they
    never read as a failure.

Efficiency
----------
Every row-level check is a boolean expression over one row, so the runner
evaluates all of them in a **single pass** with one aggregate of conditional
sums. Frame-level checks (uniqueness, freshness) that cannot be expressed that
way run separately and are declared explicitly.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import Config, ConfigError
from ..logging_utils import get_logger
from ..schemas import DQ_METRICS_SCHEMA

LOGGER = get_logger(__name__)

__all__ = [
    "ABORT",
    "WARN",
    "INFO",
    "SEVERITIES",
    "DIMENSIONS",
    "Check",
    "CheckResult",
    "DataQualityAbort",
    "run_checks",
    "run_checks_by_group",
    "apply_severity_overrides",
    "summarize",
    "log_results",
    "abort_failures",
    "raise_on_abort",
    "results_to_dataframe",
    "write_metrics",
    "not_null",
    "in_domain",
    "matches_when_present",
    "distinct_count_violation",
    "unique",
    "unique_by",
    "references",
    "counter",
]

ABORT = "abort"
WARN = "warn"
#: Informational: recorded as a metric, never a pass/fail signal. Used for
#: things that are expected to be non-zero, like the null rate on the phase-1
#: exempt columns or the share of synthetic correlation ids.
INFO = "info"

SEVERITIES = (ABORT, WARN, INFO)

#: Dimensions from section 12 of the Cross-Sector Bad Actor governance framework.
#:
#: ``Referential`` is added for Gold and does not appear in Silver. Silver holds
#: independent events, so there is nothing for one row to refer to. Gold holds a
#: hierarchy - lifecycles inside accounts inside customers, and events pointing
#: back at all three - and a dangling ``acct_id`` is a distinct kind of failure
#: from a malformed one. Trending them under the same heading would hide it.
DIMENSIONS = (
    "Accuracy",
    "Completeness",
    "Consistency",
    "Validity",
    "Uniqueness",
    "Timeliness",
    "Currency",
    "Lineage",
    "Referential",
)


class DataQualityAbort(RuntimeError):
    """One or more abort-severity checks failed; nothing was appended."""


# --------------------------------------------------------------------------
# check definition
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Check:
    """A single data-quality assertion.

    ``violation`` is a boolean Column that is **True for rows that break the
    contract**, so a passing check counts zero. Writing rules as violations
    (rather than as assertions) keeps the count directly reportable: "41,043,350
    rows are missing imei" is the number a carrier needs.
    """

    name: str
    dimension: str
    severity: str
    description: str
    checked_field: str | None = None
    violation: Column | None = None
    #: Frame-level checks compute their own violation count from the whole frame.
    frame_fn: Callable[[DataFrame], int] | None = None
    #: A counter is already measured by a transform/sink; it needs no new scan.
    constant_value: int | None = None

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ConfigError(
                f"check {self.name!r}: severity must be one of {SEVERITIES}, got {self.severity!r}"
            )
        if self.dimension not in DIMENSIONS:
            raise ConfigError(
                f"check {self.name!r}: dimension {self.dimension!r} not in {DIMENSIONS}"
            )
        if (self.violation is None) == (self.frame_fn is None):
            raise ConfigError(
                f"check {self.name!r}: set exactly one of `violation` (row-level) "
                "or `frame_fn` (frame-level)"
            )

    @property
    def is_row_level(self) -> bool:
        return self.violation is not None


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one check against one batch (optionally within one group)."""

    check: Check
    total_rows: int
    violation_rows: int
    #: Set when the check was evaluated inside a per-carrier breakdown.
    group_key: str | None = None
    group_value: str | None = None

    @property
    def passed(self) -> bool:
        return self.violation_rows == 0

    @property
    def status(self) -> str:
        if self.check.severity == INFO:
            return "INFO"
        return "PASS" if self.passed else "FAIL"

    @property
    def blocking(self) -> bool:
        """A failed abort-severity check that was evaluated on the whole batch.

        Group-scoped results never block on their own: they are a breakdown of
        a whole-batch check that has already been evaluated.
        """
        return (
            not self.passed
            and self.check.severity == ABORT
            and self.group_value is None
        )

    @property
    def pass_pct(self) -> float:
        """Percentage of rows satisfying the contract, to 4 decimals.

        An empty batch scores 100 - there is nothing wrong with zero rows.
        """
        if self.total_rows == 0:
            return 100.0
        return round(100.0 * (self.total_rows - self.violation_rows) / self.total_rows, 4)

    def format_line(self) -> str:
        if self.check.severity == INFO:
            marker = f"INFO ({self.violation_rows:,})"
        else:
            marker = "PASS" if self.passed else f"FAIL ({self.violation_rows:,})"
        scope = f" [{self.group_key}={self.group_value}]" if self.group_value else ""
        return (
            f"{marker:<20} [{self.check.severity:<5}] "
            f"{self.check.dimension:<12} {self.check.name}{scope}"
        )


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def run_checks(df: DataFrame, checks: Sequence[Check]) -> list[CheckResult]:
    """Evaluate every check against ``df``.

    Row-level checks are folded into one aggregate so a 400M-row batch is
    scanned once regardless of how many rules are declared.
    """
    checks = list(checks)
    if not checks:
        return []

    names = [c.name for c in checks]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ConfigError(f"duplicate DQ check name(s): {sorted(duplicates)}")

    row_level = [c for c in checks if c.is_row_level]
    frame_level = [c for c in checks if not c.is_row_level]

    aggregates = [F.count(F.lit(1)).alias("__total__")]
    aggregates += [
        F.sum(F.when(c.violation, F.lit(1)).otherwise(F.lit(0))).alias(f"chk__{i}")
        for i, c in enumerate(row_level)
    ]

    row = df.agg(*aggregates).collect()[0]
    total = int(row["__total__"])

    results: list[CheckResult] = [
        CheckResult(check=c, total_rows=total, violation_rows=int(row[f"chk__{i}"] or 0))
        for i, c in enumerate(row_level)
    ]

    for check in frame_level:
        violations = int(check.frame_fn(df))  # type: ignore[misc]
        results.append(CheckResult(check=check, total_rows=total, violation_rows=violations))

    # Preserve declaration order for readable logs.
    order = {c.name: i for i, c in enumerate(checks)}
    results.sort(key=lambda r: order[r.check.name])
    return results


def run_checks_by_group(
    df: DataFrame,
    checks: Sequence[Check],
    group_column: str = "mno",
) -> list[CheckResult]:
    """Evaluate the row-level checks separately for each value of ``group_column``.

    This is what turns the DQ table into something reportable per carrier:
    "BELL is missing imei on 12% of rows" rather than a single blended number.
    Frame-level and INFO checks are skipped - they are whole-batch by nature.

    Costs one extra pass over the batch, regardless of the number of checks.
    """
    row_level = [c for c in checks if c.is_row_level and c.severity != INFO]
    if not row_level or group_column not in df.columns:
        return []

    aggregates = [F.count(F.lit(1)).alias("__total__")]
    aggregates += [
        F.sum(F.when(c.violation, F.lit(1)).otherwise(F.lit(0))).alias(f"chk__{i}")
        for i, c in enumerate(row_level)
    ]

    results: list[CheckResult] = []
    for row in df.groupBy(group_column).agg(*aggregates).collect():
        group_value = row[group_column]
        total = int(row["__total__"])
        for i, check in enumerate(row_level):
            results.append(
                CheckResult(
                    check=check,
                    total_rows=total,
                    violation_rows=int(row[f"chk__{i}"] or 0),
                    group_key=group_column,
                    group_value=None if group_value is None else str(group_value),
                )
            )
    return results


def apply_severity_overrides(
    checks: Sequence[Check],
    overrides: dict | None,
) -> list[Check]:
    """Re-tier checks from config (``datasets.<name>.dq.severity_overrides``).

    Lets an operator promote a warning to a blocking failure - or relax one
    during a known upstream incident - without editing code.
    """
    if not overrides:
        return list(checks)

    known = {c.name for c in checks}
    unknown = set(overrides) - known
    if unknown:
        raise ConfigError(
            f"dq.severity_overrides references unknown check(s) {sorted(unknown)}; "
            f"known checks: {sorted(known)}"
        )

    out = []
    for check in checks:
        severity = str(overrides.get(check.name, check.severity)).lower()
        if severity == check.severity:
            out.append(check)
        else:
            LOGGER.info(
                "dq: check %s severity overridden %s -> %s",
                check.name, check.severity, severity,
            )
            out.append(
                Check(
                    name=check.name,
                    dimension=check.dimension,
                    severity=severity,
                    description=check.description,
                    checked_field=check.checked_field,
                    violation=check.violation,
                    frame_fn=check.frame_fn,
                    constant_value=check.constant_value,
                )
            )
    return out


def summarize(results: Sequence[CheckResult]) -> dict:
    """Counts by severity/outcome, for the run summary and for alerting."""
    scored = [r for r in results if r.check.severity != INFO and r.group_value is None]
    failures = [r for r in scored if not r.passed]
    return {
        "dq_checks_run": len(scored),
        "dq_checks_failed": len(failures),
        "dq_abort_failures": len([r for r in failures if r.check.severity == ABORT]),
        "dq_warn_failures": len([r for r in failures if r.check.severity == WARN]),
        "dq_violation_rows_total": sum(r.violation_rows for r in failures),
        "dq_metric_rows_written": len(results),
    }


def log_results(results: Sequence[CheckResult], title: str = "data quality") -> None:
    LOGGER.info("---- %s ----", title)
    whole_batch = [r for r in results if r.group_value is None]
    total = whole_batch[0].total_rows if whole_batch else 0
    LOGGER.info("rows evaluated: %s", f"{total:,}")
    for result in whole_batch:
        log = LOGGER.info if (result.passed or result.check.severity == INFO) else LOGGER.warning
        log("%s", result.format_line())
    LOGGER.info("summary: %s", summarize(results))


def abort_failures(results: Sequence[CheckResult]) -> list[CheckResult]:
    return [r for r in results if r.blocking]


def raise_on_abort(results: Sequence[CheckResult], dataset: str) -> None:
    """Raise :class:`DataQualityAbort` if any abort-severity check failed."""
    failures = abort_failures(results)
    if not failures:
        return
    detail = "\n".join(
        f"  - {r.check.name}: {r.violation_rows:,} / {r.total_rows:,} rows - {r.check.description}"
        for r in failures
    )
    # "nothing was merged" rather than "nothing was written": the checks run
    # against the frame the run built, before any sink is touched, so the Gold
    # table is exactly as the previous run left it. Saying which is the first
    # thing an operator reading this needs to know.
    raise DataQualityAbort(
        f"{dataset}: {len(failures)} structural data-quality check(s) failed; "
        f"nothing was merged into Gold.\n{detail}"
    )


# --------------------------------------------------------------------------
# metrics sink
# --------------------------------------------------------------------------

def results_to_dataframe(
    spark: SparkSession,
    results: Sequence[CheckResult],
    run_id: str,
    dataset: str,
    check_ts: _dt.datetime,
    layer: str = "gold",
    batch_id: str | None = None,
    event_date_min: _dt.date | None = None,
    event_date_max: _dt.date | None = None,
) -> DataFrame:
    """Materialize check results as a DataFrame matching ``DQ_METRICS_SCHEMA``.

    ``metric_value`` is stored as a string so the table can hold percentages
    today and richer values later without a schema migration - the same shape
    as the existing ``dq_metrics_*.csv`` reference files.
    """
    rows = [
        (
            run_id,
            dataset,
            layer,
            r.check.name,
            r.check.dimension,
            r.check.severity,
            r.check.checked_field,
            r.group_key,
            r.group_value,
            int(r.total_rows),
            int(r.violation_rows),
            f"{r.pass_pct}",
            r.status,
            r.check.description,
            batch_id,
            event_date_min,
            event_date_max,
            check_ts,
            check_ts.date(),
        )
        for r in results
    ]
    return spark.createDataFrame(rows, schema=DQ_METRICS_SCHEMA)


def write_metrics(
    spark: SparkSession,
    cfg: Config,
    metrics: DataFrame,
    *,
    idempotent: bool = False,
) -> str:
    """Write metric history, optionally inserting idempotently for staged retries.

    Returns the target description. Metrics are written **before** any abort is
    raised, so a blocked run still leaves evidence of why it was blocked.

    The default path appends. The staged path uses an insert-only key MERGE so
    a retry of the same logical run cannot duplicate its evidence. Every other
    table holds a *state* that a later batch can revise; the
    metrics table holds a *history* of what each run observed, and revising that
    would defeat the purpose of keeping it. The table has the same nineteen
    columns as the Silver one, deliberately, so both layers trend in one query.

    **There is no way to turn this off.** ``dq.output.enabled`` used to exist and
    has been removed. A DQ result that is computed, used to decide whether the
    run is blocked, and then discarded leaves the estate in the worst of both
    worlds: the run was gated on a number nobody can look up afterwards. And a
    switch like that is self-selecting for the moment it does most harm - it gets
    disabled during an incident, to make a run get through, and that is the run
    whose evidence is worth the most. A run that genuinely must not touch the
    table points the sink at a temporary directory instead, which is what
    ``conf/local.yaml`` and the test fixtures do.
    """
    # Imported here to avoid a circular import at module load.
    from ..spark import qualified_table_name, table_exists

    mode = str(cfg.get("dq.output.mode", "parquet")).lower()
    partition_by = list(cfg.get("dq.output.partition_by", ["check_date"]))
    aligned = metrics.select(
        *[F.col(f.name).cast(f.dataType).alias(f.name) for f in DQ_METRICS_SCHEMA.fields]
    )

    if mode == "iceberg":
        table = qualified_table_name(cfg, str(cfg.require("dq.output.table")))
        if not table_exists(spark, table):
            raise ConfigError(
                f"DQ metrics Iceberg table {table} is missing; provision it "
                "before submitting the Gold step"
            )
        if idempotent:
            # Staged retries keep the same logical run identity. A crash after
            # committing metrics must not append a second copy on the retry.
            # Metric keys include nullable group fields using null-safe equality.
            view = "__gold_staged_dq_metrics"
            aligned.createOrReplaceTempView(view)
            keys = ("run_id", "dataset", "layer", "metric_name", "group_key", "group_value")
            on = " AND ".join(f"t.{k} <=> s.{k}" for k in keys)
            cols = ", ".join(f.name for f in DQ_METRICS_SCHEMA.fields)
            vals = ", ".join(f"s.{f.name}" for f in DQ_METRICS_SCHEMA.fields)
            spark.sql(f"MERGE INTO {table} t USING {view} s ON {on} "
                      f"WHEN NOT MATCHED THEN INSERT ({cols}) VALUES ({vals})")
        else:
            aligned.writeTo(table).append()
        describe = f"iceberg table {table}"
    elif mode == "parquet":
        path = str(cfg.require("dq.output.path"))
        writer = aligned.write.mode("append")
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        writer.parquet(path)
        describe = f"parquet {path}"
    else:
        raise ConfigError(f"dq.output.mode must be 'iceberg' or 'parquet', got {mode!r}")

    LOGGER.info("dq metrics written to %s", describe)
    return describe


# --------------------------------------------------------------------------
# small builders used by the rule modules
# --------------------------------------------------------------------------

def not_null(
    column: str,
    dimension: str = "Completeness",
    severity: str = WARN,
    description: str | None = None,
) -> Check:
    """``column IS NOT NULL``."""
    return Check(
        name=f"completeness_{column.lower()}",
        dimension=dimension,
        severity=severity,
        checked_field=column,
        description=description or f"{column} is null",
        violation=F.col(column).isNull(),
    )


def in_domain(
    column: str,
    allowed: Iterable[str],
    name: str | None = None,
    severity: str = WARN,
    dimension: str = "Consistency",
    description: str | None = None,
) -> Check:
    """``column`` is null or outside the permitted value set."""
    allowed = sorted(set(allowed))
    return Check(
        name=name or f"consistency_{column.lower()}_domain",
        dimension=dimension,
        severity=severity,
        checked_field=column,
        description=description or f"{column} outside permitted values {allowed}",
        violation=~F.col(column).isin(allowed) | F.col(column).isNull(),
    )


def in_domain_when_present(
    column: str,
    allowed: Iterable[str],
    name: str | None = None,
    severity: str = WARN,
    dimension: str = "Consistency",
    description: str | None = None,
) -> Check:
    """Non-null ``column`` values must be in ``allowed``; nulls are exempt.

    :func:`in_domain` counts a null as a violation, which is right for a column
    that is not nullable and wrong for one that is. A nullable column checked
    with the stricter helper warns on every row where the value is legitimately
    absent, and a rule that fires on most of a table is a rule nobody reads.

    Use this one where the schema declares the column nullable and the null
    means something - ``boundary_used`` on a record with no lifecycle, say - and
    :func:`in_domain` everywhere else.
    """
    allowed = sorted(set(allowed))
    return Check(
        name=name or f"consistency_{column.lower()}_domain",
        dimension=dimension,
        severity=severity,
        checked_field=column,
        description=(
            description
            or f"{column} (when not null) outside permitted values {allowed}"
        ),
        violation=F.col(column).isNotNull() & ~F.col(column).isin(allowed),
    )


def matches_when_present(
    column: str,
    pattern: str,
    name: str | None = None,
    severity: str = WARN,
    dimension: str = "Validity",
    description: str | None = None,
) -> Check:
    """Non-null ``column`` values must match ``pattern``; nulls are exempt."""
    return Check(
        name=name or f"validity_{column.lower()}_format",
        dimension=dimension,
        severity=severity,
        checked_field=column,
        description=description or f"{column} (when not null) does not match {pattern}",
        violation=F.col(column).isNotNull() & ~F.col(column).rlike(pattern),
    )


def distinct_count_violation(column: str) -> Callable[[DataFrame], int]:
    """Frame-level helper: number of *extra* rows beyond one per distinct key."""

    def _fn(df: DataFrame) -> int:
        total = df.count()
        distinct = df.select(column).distinct().count()
        return max(total - distinct, 0)

    return _fn


def unique(
    column: str,
    severity: str = WARN,
    description: str | None = None,
) -> Check:
    """Key uniqueness within the batch, counted as surplus rows."""
    return Check(
        name=f"uniqueness_{column.lower()}",
        dimension="Uniqueness",
        severity=severity,
        checked_field=column,
        description=description or f"duplicate {column} values within the batch",
        frame_fn=distinct_count_violation(column),
    )


def unique_by(
    columns: Sequence[str],
    name: str,
    severity: str = WARN,
    description: str | None = None,
) -> Check:
    """Uniqueness over a composite key, counted as surplus rows.

    Gold needs this where Silver did not: a lifecycle is unique by
    ``lifecycle_uid`` alone, but a device segment is unique by
    ``(phone, imei, from_ts)`` and no single column identifies it.
    """
    cols = list(columns)

    def _fn(df: DataFrame) -> int:
        total = df.count()
        distinct = df.select(*cols).distinct().count()
        return max(total - distinct, 0)

    return Check(
        name=name,
        dimension="Uniqueness",
        severity=severity,
        checked_field=", ".join(cols),
        description=description or f"duplicate ({', '.join(cols)}) rows",
        frame_fn=_fn,
    )


def references(
    column: str,
    reference: DataFrame,
    reference_column: str,
    name: str,
    severity: str = ABORT,
    description: str | None = None,
) -> Check:
    """Every non-null ``column`` value must exist in ``reference``.

    The Gold-only dimension. A ``lifecycle_uid`` in the account mapping that no
    lifecycle row carries, or an ``acct_id`` on a normalized event that no
    account holds, means two stages of the same run disagreed about what exists -
    which is a pipeline failure, not a carrier one, and so defaults to ``abort``.
    """
    keys = reference.select(F.col(reference_column).alias("__ref")).distinct()

    def _fn(df: DataFrame) -> int:
        present = df.where(F.col(column).isNotNull())
        return present.join(keys, F.col(column) == F.col("__ref"), "left_anti").count()

    return Check(
        name=name,
        dimension="Referential",
        severity=severity,
        checked_field=column,
        description=description
        or f"{column} has no matching {reference_column} in the run's own output",
        frame_fn=_fn,
    )


def counter(
    name: str,
    value: int,
    dimension: str,
    severity: str = INFO,
    description: str = "",
    checked_field: str | None = None,
) -> Check:
    """Record a number the pipeline already computed as a DQ metric.

    Not everything worth trending is a scan over the output. How many
    number-change cycles had to be broken, how many rows a merge deleted, how
    many cross-carrier reappearances fell just outside the port window - these
    are known by the stage that did the work and would be expensive or
    impossible to recover afterwards. This lets that stage hand the number to the
    same table as everything else, so one query answers "what did this run do?"
    """
    frozen = int(value)
    return Check(
        name=name,
        dimension=dimension,
        severity=severity,
        checked_field=checked_field,
        description=description,
        frame_fn=lambda _df: frozen,
        constant_value=frozen,
    )


def counter_results(checks: Sequence[Check], total_rows: int) -> list[CheckResult]:
    """Preserve the exact counter metrics using an already measured row count."""
    if any(check.constant_value is None for check in checks):
        raise ValueError("counter_results accepts only precomputed counter checks")
    return [CheckResult(check, int(total_rows), int(check.constant_value)) for check in checks]


def pack_results(results: Sequence[CheckResult]) -> list[dict]:
    """Store measured DQ evidence between steps; never serialize Spark Columns."""
    return [{
        "check": {name: getattr(r.check, name) for name in (
            "name", "dimension", "severity", "description", "checked_field")},
        "total_rows": r.total_rows, "violation_rows": r.violation_rows,
        "group_key": r.group_key, "group_value": r.group_value,
    } for r in results]


def unpack_results(items: Sequence[dict]) -> list[CheckResult]:
    """Rehydrate evidence for reporting and gating; no check is re-evaluated."""
    return [CheckResult(
        counter(value=item["violation_rows"], **item["check"]),
        item["total_rows"], item["violation_rows"], item["group_key"], item["group_value"],
    ) for item in items]
