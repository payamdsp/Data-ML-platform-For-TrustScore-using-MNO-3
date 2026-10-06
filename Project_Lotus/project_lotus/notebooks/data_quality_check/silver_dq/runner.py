"""Run every configured DQ family against a silver table and write the canonical outputs.

Split by cost, not by output file:
  _scan            the only Spark step; one pass for the overall numbers, one more
                   for the per-MNO breakdown. See its docstring before changing it.
  build_summary    pure Python: the verdicts -> summary.json
  build_metrics    pure Python: the numbers  -> metrics.json

Both build_* take a plain dict, so they are testable (and fixable) without a
SparkSession, and they cannot disagree; they are shaped from the same _scan result.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from pyspark.sql import functions as F

from . import metrics as M
from .config import CheckConfig

SCHEMA_VERSION = "1.0.0"  # version of the summary/metrics output contract
FAMILIES = ("completeness", "consistency", "validity", "uniqueness", "timeliness", "victim_exclusion")


def _timeliness_on(cfg, columns):
    """Timeliness runs only if the yaml asked for it AND the columns it needs exist."""
    if not cfg.timeliness_enabled or cfg.event_timestamp_col not in columns:
        return False
    if cfg.timeliness_mode == "streaming":
        return cfg.ingestion_timestamp_col in columns
    return True


def _scan(df, cfg, columns, with_partition, run_date):
    """Read the table and return the raw per-family counts. THE ONLY SPARK STEP.

    Every family is concatenated into ONE .agg(), so a DQ run costs at most
    two passes over the table (one overall + one grouped by MNO) no matter how
    many rules the yaml configures.

    Do NOT split this per family or per output file. account_changes_batch is
    ~400M rows and audit_trail_services_3 ~436M; every extra split is another
    full pass. It would also let summary.json and metrics.json disagree: they are
    consistent today precisely because both are shaped from this one result.
    """
    exprs = [F.count(F.lit(1)).alias("row_count")]
    exprs += M.completeness_exprs(columns)
    exprs += M.consistency_exprs(cfg.allowed_values, cfg.allowed_values_by_mno, cfg.partition_by)
    exprs += M.validity_exprs(cfg.validity_formats, columns)
    exprs += M.uniqueness_exprs(cfg.primary_keys, columns)
    if cfg.victim_exclusion_col:
        exprs += M.victim_exclusion_exprs(cfg.victim_exclusion_col, cfg.victim_allowed_values, columns)
    if _timeliness_on(cfg, columns):
        exprs += M.timeliness_exprs(cfg, run_date, columns)
    if with_partition:
        return df.groupBy(cfg.partition_by).agg(*exprs).collect()
    return df.agg(*exprs).collect()


def _families_from_row(row, cfg, columns):
    """One _scan Row -> {row_count, <family>: {numbers, rules}}. Rates and verdicts happen here."""
    row_count = int(row["row_count"] or 0)
    out = {
        "row_count": row_count,
        "completeness": M.parse_completeness(row, columns, row_count, set(cfg.required_columns)),
        "consistency": M.parse_consistency(row, cfg.allowed_values, cfg.allowed_values_by_mno, row_count),
        "uniqueness": M.parse_uniqueness(row, cfg.primary_keys, row_count),
    }
    if cfg.validity_formats:
        out["validity"] = M.parse_validity(row, cfg.validity_formats, columns, row_count)
    if cfg.victim_exclusion_col:
        out["victim_exclusion"] = M.parse_victim_exclusion(
            row, cfg.victim_exclusion_col, cfg.victim_allowed_values, columns, row_count)
    if _timeliness_on(cfg, columns):
        out["timeliness"] = M.parse_timeliness(row, cfg, columns, row_count)
    return out


def _metrics_view(families):
    """Keep the numbers, drop the verdicts (verdicts are summary.json's job)."""
    out = {"row_count": families["row_count"]}
    for fam in FAMILIES:
        if fam in families:
            out[fam] = {k: v for k, v in families[fam].items() if k in ("per_column", "metrics")}
    return out


def build_summary(overall, cfg, run_date):
    """The verdicts -> summary.json. Pure Python: feed it a _families_from_row dict, no Spark."""
    rules = []
    for fam in FAMILIES:
        if fam in overall:
            rules.extend(overall[fam]["rules"])

    # An empty table satisfies every "violations == 0" rule vacuously: 0 nulls, 0 domain
    # violations, 0 duplicates, 0 future events -> passed=true for a table with no data.
    # That is the exact false green this module exists to prevent (see the ticket's
    # Motivation: lineage needs a dependable "is this silver run trustworthy" signal).
    # Verified 2026-07-16 on a real run: device_lookup_batch at row_count=0 reported
    # 16/16 rules passed and validated cleanly against the committed schema.
    if overall["row_count"] == 0:
        rules.append({"family": "completeness", "rule": "table is not empty",
                      "column": "*", "violations": 1, "passed": False})

    passed = sum(1 for r in rules if r["passed"])
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": cfg.dataset,
        "run_date": run_date,
        "row_count": overall["row_count"],
        "rule_count": len(rules),
        "passed_count": passed,
        "failed_count": len(rules) - passed,
        "passed": all(r["passed"] for r in rules),
        "rules": sorted(rules, key=lambda r: (r["family"], r["column"], r["rule"])),
    }


def build_metrics(overall, per_mno, cfg, run_date):
    """The numbers -> metrics.json. Pure Python: feed it _families_from_row dicts, no Spark."""
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": cfg.dataset,
        "run_date": run_date,
        "partition_by": cfg.partition_by,
        "overall": _metrics_view(overall),
        "per_mno": {k: _metrics_view(v) for k, v in per_mno.items()},
    }


def compute_dq(df, cfg, run_date):
    """Scan once, shape twice. Deterministic: reads only the DataFrame + cfg, never wall-clock."""
    columns = list(df.columns)
    overall = _families_from_row(_scan(df, cfg, columns, False, run_date)[0], cfg, columns)
    per_mno = {}
    if cfg.partition_by is not None:
        for r in _scan(df, cfg, columns, True, run_date):
            key = r[cfg.partition_by]
            per_mno[str(key) if key is not None else "__null__"] = _families_from_row(r, cfg, columns)
    return (build_summary(overall, cfg, run_date),
            build_metrics(overall, per_mno, cfg, run_date))


# #
# contract validation + canonical-path writer (local or s3://)
# #
def read_schema(schema_dir: str, name: str) -> dict:
    """One committed schema, from a local dir (repo copy) or an s3:// prefix (published copy)."""
    if schema_dir.startswith("s3://"):
        import boto3  # lazy: local tests don't need boto3/creds

        bucket, prefix = schema_dir[len("s3://"):].split("/", 1)
        key = f"{prefix.rstrip('/')}/{name}.schema.json"
        return json.loads(boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read())
    with open(os.path.join(schema_dir, f"{name}.schema.json")) as fh:
        return json.load(fh)


def validate(summary: dict, metrics: dict, schema_dir: str) -> None:
    """Raise jsonschema.ValidationError if outputs don't match the committed schema.

    schema_dir holds summary.schema.json + metrics.schema.json; pass the repo copy
    (a local path) or the published copy (an s3:// prefix). Same contract either way;
    if the two disagree, someone edited schemas/ without re-publishing.
    """
    import jsonschema

    jsonschema.validate(summary, read_schema(schema_dir, "summary"))
    jsonschema.validate(metrics, read_schema(schema_dir, "metrics"))


def result_paths(dq_root: str, cfg: CheckConfig, run_date: str) -> tuple[str, str]:
    base = (
        f"{dq_root.rstrip('/')}/results/"
        f"dataset={cfg.dataset}/run_date={run_date}"
    )
    return f"{base}/summary.json", f"{base}/metrics.json"


def _dumps(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False)


def write_results(
    summary: dict,
    metrics: dict,
    dq_root: str,
    cfg: CheckConfig,
    run_date: str,
    schema_dir: Optional[str] = None,
) -> tuple[str, str]:
    if schema_dir is not None:
        validate(summary, metrics, schema_dir)

    s_path, m_path = result_paths(dq_root, cfg, run_date)
    for path, obj in ((s_path, summary), (m_path, metrics)):
        payload = _dumps(obj)
        if path.startswith("s3://"):
            import boto3  # lazy

            bucket, key = path[len("s3://"):].split("/", 1)
            boto3.client("s3").put_object(
                Bucket=bucket, Key=key, Body=payload.encode("utf-8"),
                ContentType="application/json",
            )
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fh:
                fh.write(payload)
    return s_path, m_path
