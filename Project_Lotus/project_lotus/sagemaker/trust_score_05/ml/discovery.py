"""What is actually in the data, and whether the pipeline may proceed on it.

This is the first stage of a run and the only one whose job is to say no. It
inventories the schema of every configured prefix, classifies every column with
:mod:`trust_score_05.ml.selection`, runs a fixed set of checks, and writes a
report. Whether the stage succeeded is a property of the checks, not of whether
the code reached the end.

That distinction is the whole reason the module exists. The notebook's discovery
step collected ``df.dtypes`` for each path, caught any read failure into a row
whose column name was ``__ERROR__``, wrote seven required data-quality reports
as *empty* CSV files, and then wrote ``_READY.json`` unconditionally. A run in
which every single path failed to read produced a green marker, an empty schema
summary, and eleven empty reports, and the training stages downstream read the
marker, believed discovery had passed, and failed one by one on their own.

The checks are tiered. An ``error`` means the pipeline must not continue: a
configured prefix that cannot be read, a population missing its join key, no
eligible features, or no numeric column shared between the fraud and non-fraud
populations — the last of which is the condition under which every model would
be fitted on one population and scored on a disjoint set of columns in the
other. A ``warning`` is something a human should see but that does not
invalidate the run: a column present in some months and not others, a column
whose dtype disagrees between prefixes, a feature that is almost entirely null.
An ``info`` is a count.

:attr:`DataQualityReport.ok` is false if any check is an error, and the job
entry point writes ``_READY.json`` only when it is true. A failed discovery
writes ``_FAILED.json`` with the failing checks in it, so the next stage's
resume logic sees no ready marker and stops instead of proceeding on a schema
nobody validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from trust_score_05.common.io.s3 import join_uri, write_json, write_pandas
from trust_score_05.ml.config import MLConfig
from trust_score_05.ml.datasets import (
    CUSTOMER_KEY_COLUMN,
    SOURCE_PATH_COLUMN,
    discover_fraud_paths,
    discover_testing_nonfraud_paths,
    discover_training_nonfraud_paths,
    read_parquet_path,
)
from trust_score_05.ml.selection import (
    classify_columns,
    is_numeric_dtype,
)

__all__ = [
    "DATASET_FRAUD",
    "DATASET_TESTING_NONFRAUD",
    "DATASET_TRAINING_NONFRAUD",
    "SCHEMA_COLUMNS",
    "SEVERITY_ERROR",
    "SEVERITY_INFO",
    "SEVERITY_WARNING",
    "Check",
    "DataQualityReport",
    "DiscoveryError",
    "collect_schema",
    "common_numeric_columns",
    "dtype_disagreements",
    "partial_columns",
    "run_discovery",
    "run_quality_checks",
    "write_discovery_reports",
]

LOGGER = logging.getLogger(__name__)

DATASET_TRAINING_NONFRAUD = "training_nonfraud"
DATASET_TESTING_NONFRAUD = "testing_nonfraud"
DATASET_FRAUD = "fraud"

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

SCHEMA_COLUMNS: Tuple[str, ...] = ("dataset", "window", "path", "column", "dtype")


class DiscoveryError(RuntimeError):
    """Discovery could not inventory the configured data at all."""


# --------------------------------------------------------------------------
# the inventory
# --------------------------------------------------------------------------

def collect_schema(
    spark: Any,
    prefixes: Sequence[Tuple[str, str, str]],
) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    """Inventory ``(dataset, window, path)`` triples into one long frame.

    Returns the schema rows and, separately, the paths that could not be read.

    Read failures come back as their own list rather than as rows in the schema
    frame. The notebook wrote them into the frame as a column literally named
    ``__ERROR__`` with the exception text in the dtype field, which meant a
    failure was invisible to every consumer that grouped by column name, and the
    string ``__ERROR__`` was itself a candidate column in the set arithmetic
    that followed.
    """
    rows: List[Dict[str, str]] = []
    failures: List[Dict[str, str]] = []
    for dataset, window, path in prefixes:
        try:
            frame = read_parquet_path(spark, path)
            dtypes = frame.dtypes
        except Exception as error:  # noqa: BLE001 - recorded and re-raised by the caller
            LOGGER.error("cannot read %s prefix %s: %s", dataset, path, error)
            failures.append(
                {
                    "dataset": dataset,
                    "window": window,
                    "path": path,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        for column, dtype in dtypes:
            rows.append(
                {
                    "dataset": dataset,
                    "window": window,
                    "path": path,
                    "column": column,
                    "dtype": dtype,
                }
            )
    schema = pd.DataFrame(rows, columns=list(SCHEMA_COLUMNS))
    return schema, failures


def _configured_prefixes(cfg: MLConfig, spark: Any) -> Tuple[List[Tuple[str, str, str]], Dict]:
    """Every prefix the run will read, tagged with its dataset and window."""
    del spark  # discovery of prefixes needs the store, not the session
    training = discover_training_nonfraud_paths(cfg)
    testing = discover_testing_nonfraud_paths(cfg)
    fraud = discover_fraud_paths(cfg)

    prefixes: List[Tuple[str, str, str]] = []
    for window, path in training.found.items():
        prefixes.append((DATASET_TRAINING_NONFRAUD, window, path))
    for window, path in testing.found.items():
        prefixes.append((DATASET_TESTING_NONFRAUD, window, path))
    for source, paths in fraud.items():
        for path in paths:
            prefixes.append((DATASET_FRAUD, source, path))

    summary = {
        "training_windows_found": list(training.found),
        "training_windows_missing": list(training.missing),
        "testing_windows_found": list(testing.found),
        "testing_windows_missing": list(testing.missing),
        "fraud_partitions_by_source": {source: len(paths) for source, paths in fraud.items()},
    }
    return prefixes, summary


def common_numeric_columns(schema: pd.DataFrame) -> List[str]:
    """Numeric columns present in both the fraud and non-fraud populations.

    A feature must exist on both sides to be usable: the model is fitted on the
    non-fraud population and scored on a union of the two, and a column present
    only in the fraud population is a perfect label proxy that
    :mod:`trust_score_05.ml.selection` also excludes by name where it can.

    "Numeric" is decided per column across every prefix that has it, and one
    numeric sighting is enough. A column read as ``string`` in a month where it
    was entirely null and as ``double`` elsewhere is numeric; requiring
    unanimity would drop it, and
    :func:`~trust_score_05.ml.datasets.add_missing_and_select` casts it anyway.
    The disagreement is reported by :func:`dtype_disagreements` instead.
    """
    if schema.empty:
        return []
    numeric = (
        schema.groupby("column")["dtype"]
        .apply(lambda values: any(is_numeric_dtype(v) for v in values))
        .pipe(lambda flags: set(flags.index[flags]))
    )
    fraud = set(schema.loc[schema["dataset"].eq(DATASET_FRAUD), "column"])
    nonfraud = set(
        schema.loc[
            schema["dataset"].isin([DATASET_TRAINING_NONFRAUD, DATASET_TESTING_NONFRAUD]),
            "column",
        ]
    )
    return sorted((fraud & nonfraud & numeric) - {CUSTOMER_KEY_COLUMN, SOURCE_PATH_COLUMN})


def dtype_disagreements(schema: pd.DataFrame) -> pd.DataFrame:
    """Columns read as more than one dtype across the configured prefixes.

    Reported rather than fatal. Every feature is cast to ``double`` before it
    reaches a model, so a disagreement between ``bigint`` and ``double`` is
    harmless; a disagreement between ``string`` and ``double`` usually means one
    month wrote an all-null column with no inferable type. Both are worth
    knowing about, and neither should stop a run — but a disagreement involving
    a type nothing can cast, such as a struct or an array, will surface as
    every value becoming null, which the null-rate warning then catches.
    """
    if schema.empty:
        return pd.DataFrame(columns=["column", "dtypes", "n_dtypes"])
    grouped = (
        schema.groupby("column")["dtype"]
        .apply(lambda values: sorted({str(value) for value in values}))
        .reset_index(name="dtypes")
    )
    grouped["n_dtypes"] = grouped["dtypes"].apply(len)
    out = grouped[grouped["n_dtypes"] > 1].copy()
    out["dtypes"] = out["dtypes"].apply(lambda values: ", ".join(values))
    return out.sort_values("column").reset_index(drop=True)


def partial_columns(schema: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Columns present in some of a dataset's prefixes but not all of them.

    This is the check that explains an unexpectedly high null rate later. A
    feature added upstream in the most recent month is absent from the earlier
    months, becomes null for them in the union, and is then dropped by the
    null-rate gate — which looks like the feature being useless rather than the
    feature being new.
    """
    subset = schema[schema["dataset"].eq(dataset)]
    if subset.empty:
        return pd.DataFrame(columns=["column", "n_paths_present", "n_paths_total", "coverage"])
    total = subset["path"].nunique()
    counts = subset.groupby("column")["path"].nunique().reset_index(name="n_paths_present")
    counts["n_paths_total"] = total
    counts["coverage"] = counts["n_paths_present"] / max(total, 1)
    partial = counts[counts["n_paths_present"] < total].copy()
    return partial.sort_values(["coverage", "column"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Check:
    """One data-quality finding: what was checked, how it went, and the numbers."""

    name: str
    severity: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityReport:
    """The outcome of discovery, and whether the run may continue.

    ``ok`` is the gate. It is false when any check of severity ``error`` failed,
    and it is what decides between a ``_READY.json`` and a ``_FAILED.json``. It
    is deliberately not "no exception was raised": the notebook's marker meant
    exactly that, which is why an entirely failed discovery reported ready.
    """

    checks: List[Check] = field(default_factory=list)
    schema: pd.DataFrame = field(default_factory=pd.DataFrame)
    classification: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(
            check.severity == SEVERITY_ERROR and not check.passed for check in self.checks
        )

    @property
    def failures(self) -> List[Check]:
        return [
            check
            for check in self.checks
            if check.severity == SEVERITY_ERROR and not check.passed
        ]

    @property
    def warnings(self) -> List[Check]:
        return [
            check
            for check in self.checks
            if check.severity == SEVERITY_WARNING and not check.passed
        ]

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "check": check.name,
                    "severity": check.severity,
                    "passed": check.passed,
                    "message": check.message,
                }
                for check in self.checks
            ],
            columns=["check", "severity", "passed", "message"],
        )

    def manifest(self) -> Dict[str, Any]:
        return {
            "status": "READY" if self.ok else "FAILED",
            "checks_total": len(self.checks),
            "checks_failed": len([c for c in self.checks if not c.passed]),
            "errors": [{"check": c.name, "message": c.message} for c in self.failures],
            "warnings": [{"check": c.name, "message": c.message} for c in self.warnings],
            "summary": self.summary,
        }


def run_quality_checks(
    cfg: MLConfig,
    schema: pd.DataFrame,
    failures: Sequence[Dict[str, str]],
    discovery_summary: Dict[str, Any],
    max_null_warn_rate: float = 0.99,
    null_rates: Optional[pd.DataFrame] = None,
) -> DataQualityReport:
    """Evaluate the fixed check set against an inventory.

    A pure function of frames and dicts: it performs no reads and writes
    nothing, so a check can be exercised against a hand-built schema frame
    rather than against a cluster holding the real data. The notebook's
    equivalents were expressions interleaved with the reads that produced them
    and could not be evaluated separately at all.
    """
    checks: List[Check] = []

    checks.append(
        Check(
            name="prefixes_readable",
            severity=SEVERITY_ERROR,
            passed=not failures,
            message=(
                "every configured prefix was readable"
                if not failures
                else f"{len(failures)} configured prefixes could not be read"
            ),
            details={"failures": list(failures)},
        )
    )

    for dataset, configured_key, missing_key in (
        (DATASET_TRAINING_NONFRAUD, "training_windows_found", "training_windows_missing"),
        (DATASET_TESTING_NONFRAUD, "testing_windows_found", "testing_windows_missing"),
    ):
        found = discovery_summary.get(configured_key, [])
        missing = discovery_summary.get(missing_key, [])
        checks.append(
            Check(
                name=f"{dataset}_windows_present",
                severity=SEVERITY_ERROR if not found else SEVERITY_WARNING,
                passed=bool(found) and not missing,
                message=(
                    f"{len(found)} {dataset} windows resolved, {len(missing)} missing"
                    if found
                    else f"no {dataset} window resolved to a readable prefix"
                ),
                details={"found": list(found), "missing": list(missing)},
            )
        )

    fraud_counts = discovery_summary.get("fraud_partitions_by_source", {})
    total_fraud_partitions = sum(fraud_counts.values())
    checks.append(
        Check(
            name="fraud_partitions_present",
            severity=SEVERITY_ERROR,
            passed=total_fraud_partitions > 0,
            message=(
                f"{total_fraud_partitions} fraud partitions across "
                f"{len(fraud_counts)} feeds"
                if total_fraud_partitions
                else "no fraud partitions were discovered for any configured feed"
            ),
            details={"by_source": dict(fraud_counts)},
        )
    )
    empty_feeds = [source for source, count in fraud_counts.items() if not count]
    if fraud_counts:
        checks.append(
            Check(
                name="every_fraud_feed_has_data",
                severity=SEVERITY_WARNING,
                passed=not empty_feeds,
                message=(
                    "every configured fraud feed has partitions"
                    if not empty_feeds
                    else f"fraud feeds with no partitions: {empty_feeds}"
                ),
                details={"empty_feeds": empty_feeds},
            )
        )

    for dataset in (DATASET_TRAINING_NONFRAUD, DATASET_TESTING_NONFRAUD, DATASET_FRAUD):
        subset = schema[schema["dataset"].eq(dataset)]
        if subset.empty:
            continue
        paths_with_key = set(
            subset.loc[subset["column"].eq(CUSTOMER_KEY_COLUMN), "path"]
        )
        paths_without = sorted(set(subset["path"]) - paths_with_key)
        checks.append(
            Check(
                name=f"{dataset}_has_join_key",
                severity=SEVERITY_ERROR,
                passed=not paths_without,
                message=(
                    f"every {dataset} prefix has {CUSTOMER_KEY_COLUMN}"
                    if not paths_without
                    else f"{len(paths_without)} {dataset} prefixes lack "
                    f"{CUSTOMER_KEY_COLUMN}"
                ),
                details={"paths": paths_without[:20]},
            )
        )

    shared = common_numeric_columns(schema)
    checks.append(
        Check(
            name="fraud_and_nonfraud_share_numeric_columns",
            severity=SEVERITY_ERROR,
            passed=len(shared) >= 2,
            message=(
                f"{len(shared)} numeric columns exist in both populations"
                if len(shared) >= 2
                else f"only {len(shared)} numeric columns exist in both the fraud and "
                "non-fraud populations; a model fitted on one would be scored on "
                "columns the other does not have"
            ),
            details={"n_shared": len(shared), "sample": shared[:20]},
        )
    )

    classification = classify_columns(
        [(name, dtype) for name, dtype in _first_dtype_per_column(schema).items()],
        join_key=CUSTOMER_KEY_COLUMN,
        provenance_columns=[SOURCE_PATH_COLUMN],
    )
    eligible = classification.loc[classification["eligible"], "column"].tolist()
    eligible_shared = sorted(set(eligible) & set(shared)) if shared else []
    checks.append(
        Check(
            name="eligible_features_exist",
            severity=SEVERITY_ERROR,
            passed=len(eligible_shared) >= 2,
            message=(
                f"{len(eligible_shared)} columns are both eligible and shared"
                if len(eligible_shared) >= 2
                else f"only {len(eligible_shared)} columns survive exclusion and exist "
                "in both populations"
            ),
            details={
                "n_eligible": len(eligible),
                "n_eligible_and_shared": len(eligible_shared),
                "exclusion_reasons": classification.loc[
                    ~classification["eligible"], "reason"
                ]
                .value_counts()
                .to_dict(),
            },
        )
    )

    disagreements = dtype_disagreements(schema)
    checks.append(
        Check(
            name="dtypes_agree_across_prefixes",
            severity=SEVERITY_WARNING,
            passed=disagreements.empty,
            message=(
                "every column has one dtype everywhere"
                if disagreements.empty
                else f"{len(disagreements)} columns are read as more than one dtype"
            ),
            details={"columns": disagreements["column"].tolist()[:20]},
        )
    )

    partial = partial_columns(schema, DATASET_TRAINING_NONFRAUD)
    checks.append(
        Check(
            name="training_columns_present_in_every_month",
            severity=SEVERITY_WARNING,
            passed=partial.empty,
            message=(
                "every training column is present in every month"
                if partial.empty
                else f"{len(partial)} training columns are missing from at least one "
                "month and will be null there"
            ),
            details={"columns": partial["column"].tolist()[:20]},
        )
    )

    if null_rates is not None and not null_rates.empty and "null_rate" in null_rates:
        nearly_empty = null_rates[null_rates["null_rate"] >= max_null_warn_rate]
        checks.append(
            Check(
                name="features_are_populated",
                severity=SEVERITY_WARNING,
                passed=nearly_empty.empty,
                message=(
                    "no feature is almost entirely null"
                    if nearly_empty.empty
                    else f"{len(nearly_empty)} features are at least "
                    f"{max_null_warn_rate:.0%} null"
                ),
                details={
                    "features": nearly_empty["feature"].tolist()[:20],
                    "threshold": max_null_warn_rate,
                },
            )
        )

    checks.append(
        Check(
            name="mode",
            severity=SEVERITY_INFO,
            passed=True,
            message=f"discovery ran in {cfg.mode} mode for version {cfg.models_version}",
            details={"mode": cfg.mode, "run_id": cfg.run_id},
        )
    )

    summary = dict(discovery_summary)
    summary.update(
        {
            "columns_seen": int(schema["column"].nunique()) if not schema.empty else 0,
            "prefixes_read": int(schema["path"].nunique()) if not schema.empty else 0,
            "prefixes_failed": len(failures),
            "common_numeric_columns": len(shared),
            "eligible_and_shared_columns": len(eligible_shared),
        }
    )
    return DataQualityReport(
        checks=checks, schema=schema, classification=classification, summary=summary
    )


def _first_dtype_per_column(schema: pd.DataFrame) -> Dict[str, str]:
    """One dtype per column, preferring a numeric sighting over a non-numeric one.

    A column that one month wrote as an untyped all-null ``string`` and another
    wrote as ``double`` must classify as numeric, or the schema of the emptier
    month decides whether it is a feature.
    """
    if schema.empty:
        return {}
    out: Dict[str, str] = {}
    for column, dtypes in schema.groupby("column")["dtype"]:
        values = [str(value) for value in dtypes]
        numeric = next((v for v in values if is_numeric_dtype(v)), None)
        out[str(column)] = numeric or values[0]
    return out


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------

def write_discovery_reports(
    report: DataQualityReport,
    prefix: str,
    failures: Sequence[Dict[str, str]] = (),
) -> Dict[str, str]:
    """Write every discovery artifact under ``prefix``, returning the URIs.

    The per-reason files — ``label_leakage_columns``, ``timestamp_columns``,
    ``id_hash_columns``, ``metadata_columns``, ``excluded_columns`` — are derived
    from the classification frame, so they are populated by construction. The
    notebook wrote these five names, plus ``null_rate_by_feature`` and
    ``numeric_feature_profile``, as ``pd.DataFrame()`` — zero rows, no columns —
    to satisfy a requirement that the files exist.
    """
    classification = report.classification
    written: Dict[str, str] = {}

    def write(name: str, frame: pd.DataFrame) -> None:
        uri = join_uri(prefix, f"{name}.csv")
        write_pandas(frame, uri)
        written[name] = uri

    write("schema_summary", report.schema)
    write("data_quality_checks", report.as_frame())
    write("column_classification", classification)
    write("excluded_columns", _by_reason(classification, exclude=("",)))
    write("label_leakage_columns", _by_reason(classification, include=("label_leakage",)))
    write(
        "timestamp_columns",
        _by_reason(classification, include=("timestamp_pattern",)),
    )
    write("id_hash_columns", _by_reason(classification, include=("id_pattern", "join_key")))
    write("metadata_columns", _by_reason(classification, include=("provenance", "not_numeric")))
    write("readmitted_columns", classification[classification["readmitted"]])
    write("dtype_disagreements", dtype_disagreements(report.schema))
    write(
        "training_partial_columns",
        partial_columns(report.schema, DATASET_TRAINING_NONFRAUD),
    )
    write("unreadable_prefixes", pd.DataFrame(list(failures)))

    manifest_uri = join_uri(prefix, "schema_manifest.json")
    write_json(manifest_uri, report.manifest())
    written["schema_manifest"] = manifest_uri
    return written


def _by_reason(
    classification: pd.DataFrame,
    include: Iterable[str] = (),
    exclude: Iterable[str] = (),
) -> pd.DataFrame:
    if classification.empty:
        return classification
    if include:
        return classification[classification["reason"].isin(list(include))].reset_index(drop=True)
    return classification[~classification["reason"].isin(list(exclude))].reset_index(drop=True)


def run_discovery(
    spark: Any,
    cfg: MLConfig,
    null_rates: Optional[pd.DataFrame] = None,
) -> Tuple[DataQualityReport, List[Dict[str, str]]]:
    """Inventory every configured prefix and evaluate the checks.

    Does not write anything and does not raise on a failed check — the caller
    writes the reports and chooses the marker, so that a failing discovery still
    produces its diagnostics. The only exception raised here is when there is
    nothing at all to inventory, which no report could usefully describe.
    """
    prefixes, discovery_summary = _configured_prefixes(cfg, spark)
    if not prefixes:
        raise DiscoveryError(
            "no configured prefix resolved to anything readable; nothing to "
            f"inventory for version {cfg.models_version} run {cfg.run_id}"
        )
    schema, failures = collect_schema(spark, prefixes)
    report = run_quality_checks(
        cfg=cfg,
        schema=schema,
        failures=failures,
        discovery_summary=discovery_summary,
        null_rates=null_rates,
    )
    return report, failures
