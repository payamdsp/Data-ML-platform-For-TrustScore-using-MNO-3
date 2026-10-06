"""One EMR entry point for all three Bronze-to-Silver datasets.

The Bronze schema Lambda/query is an external prerequisite. This runner starts
with an already-approved immutable Bronze Parquet path, quarantines rows that
fail the selected YAML rules, and publishes only accepted rows to Iceberg.
"""
import argparse
import logging
import re
from datetime import date, datetime, timezone

from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from . import quarantine
from .artifacts import write_json
from .common import exact_deduplicate
from .config import DATASETS, load, load_api_types
from .dq import annotate, issue_rows, metrics
from .publish import publish, target_preflight
from .references import load_references
from .transforms import account, audit, device


LOG = logging.getLogger("silver_pipeline")


class QualityBlocked(RuntimeError):
    """Raised for a delivery-level problem, such as an empty Bronze input."""


def uri(root, *parts):
    """Join S3-style path components without changing the URI scheme."""
    return "/".join([root.rstrip("/"), *[part.strip("/") for part in parts]])


def parse_args(argv=None):
    """Parse and validate the small public command-line interface."""
    parser = argparse.ArgumentParser(
        description="Transform one approved Bronze Parquet dataset to Silver"
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--mode", choices=("validate", "publish"), default="validate")
    parser.add_argument("--run-date", required=True, help="Toronto business date: YYYY-MM-DD")
    parser.add_argument("--run-id", required=True, help="New safe ID for this attempt")
    parser.add_argument("--bronze-path", required=True, help="Approved immutable Parquet path")
    parser.add_argument("--artifact-root", required=True, help="S3 root for reports and issue rows")
    parser.add_argument("--table", help="Existing Iceberg table, required in publish mode")
    parser.add_argument(
        "--quarantine-table",
        help="Dataset-specific Iceberg table used to retain rejected-record evidence",
    )
    parser.add_argument(
        "--quarantine-table-location",
        help="Reviewed S3 location of the dataset-specific quarantine Iceberg table",
    )
    parser.add_argument(
        "--allow-quarantine-bootstrap",
        action="store_true",
        help="Validate-only smoke option to create a missing quarantine table without Terraform",
    )
    parser.add_argument(
        "--partner-reference",
        help="Audit only: partners.csv path",
    )
    parser.add_argument(
        "--provider-reference",
        help="Audit only: service_providers.csv path",
    )
    args = parser.parse_args(argv)

    try:
        date.fromisoformat(args.run_date)
    except ValueError as exc:
        parser.error(f"--run-date must be YYYY-MM-DD: {exc}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.run_id):
        parser.error("--run-id must contain only letters, numbers, dot, underscore or hyphen")
    if args.mode == "publish" and not args.table:
        parser.error("--table is required when --mode publish")
    # A table name and its reviewed S3 location are an inseparable pair.  A
    # table-only setting could silently use the catalog warehouse; a location
    # without a table leaves rejected records unreachable.
    if bool(args.quarantine_table) != bool(args.quarantine_table_location):
        parser.error(
            "--quarantine-table and --quarantine-table-location must be supplied together"
        )
    if args.mode == "publish" and not args.quarantine_table:
        parser.error(
            "publish mode requires --quarantine-table and --quarantine-table-location"
        )
    if args.allow_quarantine_bootstrap and not args.quarantine_table:
        parser.error(
            "--allow-quarantine-bootstrap requires the quarantine table argument pair"
        )
    if args.allow_quarantine_bootstrap and args.mode != "validate":
        parser.error(
            "--allow-quarantine-bootstrap is validate-only; publish requires Terraform provisioning"
        )
    if (
        args.quarantine_table_location
        and not args.quarantine_table_location.startswith("s3://")
    ):
        parser.error("--quarantine-table-location must use an s3:// URI")
    if args.quarantine_table and len(args.quarantine_table.split(".")) != 3:
        parser.error("--quarantine-table must use catalog.database.table")
    if (
        args.table
        and args.quarantine_table
        and args.table.lower() == args.quarantine_table.lower()
    ):
        parser.error("--table and --quarantine-table must refer to different tables")
    if args.quarantine_table_location and args.artifact_root.startswith("s3://"):
        artifact_root = args.artifact_root.rstrip("/") + "/"
        quarantine_root = args.quarantine_table_location.rstrip("/") + "/"
        if artifact_root.startswith(quarantine_root) or quarantine_root.startswith(
            artifact_root
        ):
            parser.error(
                "artifact-root and quarantine-table-location must use separate S3 prefixes"
            )
    if args.dataset == "audit_trail_services_3":
        if not args.partner_reference or not args.provider_reference:
            parser.error(
                "audit_trail_services_3 requires --partner-reference and "
                "--provider-reference"
            )
    elif args.partner_reference or args.provider_reference:
        parser.error("reference CSV arguments are used only by audit_trail_services_3")
    return args


def write_parquet_once(frame, path):
    """Write a run artifact once; reusing a run ID stops instead of overwriting."""
    frame.write.mode("errorifexists").parquet(path)


def execute(spark, args):
    """Run transform, DQ, exact deduplication, and optional Iceberg publication."""
    cfg = load(args.dataset)

    # These settings make timestamp parsing and bad-cast handling deterministic.
    spark.conf.set("spark.sql.session.timeZone", cfg["timezone"])
    spark.conf.set("spark.sql.timestampType", "TIMESTAMP_LTZ")
    # Glue commonly lowercases names even when the reviewed output uses client
    # casing such as subId. Make resolution deterministic across environments.
    spark.conf.set("spark.sql.caseSensitive", "false")
    spark.conf.set("spark.sql.ansi.enabled", "true")
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "CORRECTED")
    spark.conf.set("spark.sql.adaptive.enabled", "true")

    started = datetime.now(timezone.utc)
    ingestion_ts = started.isoformat()
    run_root = uri(
        args.artifact_root,
        f"v{cfg['schema_version']}",
        args.dataset,
        args.run_id,
    )
    summary_path = uri(run_root, "summary.json")
    report = {
        "dataset": args.dataset,
        "mode": args.mode,
        "run_id": args.run_id,
        "run_date": args.run_date,
        "started_at_utc": started.isoformat(),
        "bronze_path": args.bronze_path,
        "bronze_approval": "External Lambda/query prerequisite",
        "config_sha256": cfg["_hash"],
        "status": "running",
        "artifacts": {},
        # Production publishes must never lose evidence for a rejected record.
        # Persist rejected rows first, then continue with the accepted rows.
        "quarantine": {
            "enabled": bool(args.quarantine_table),
            "policy": "QUARANTINE_BAD_PUBLISH_GOOD",
            "table": args.quarantine_table,
            "location": args.quarantine_table_location,
            "attempted_rows": 0,
            "inserted_rows": 0,
            "stages": {},
        },
    }
    cached = []

    def quarantine_rows(validation_stage, frame):
        """Persist rejected source evidence before processing accepted rows.

        The updated pipeline starts from a direct approved Bronze path, so it
        has no gate-manifest fields to populate.  The stable quarantine schema
        retains those fields as SQL nulls for compatibility with the reviewed
        Terraform table contract.
        """

        if not args.quarantine_table:
            # Validate mode may be used on a laptop or an EMR notebook before
            # an Iceberg catalog is available. Publish mode rejects this path.
            return
        if validation_stage == "TARGET_CONFLICT":
            entries = quarantine.target_conflicts(
                frame,
                cfg,
                dataset=args.dataset,
                run_id=args.run_id,
                run_date=args.run_date,
                quarantined_at=started,
            )
        else:
            entries = quarantine.dq_failures(
                frame,
                cfg,
                dataset=args.dataset,
                run_id=args.run_id,
                run_date=args.run_date,
                validation_stage=validation_stage,
                quarantined_at=started,
            )
        outcome = quarantine.write_quarantine(
            spark,
            args.quarantine_table,
            args.quarantine_table_location,
            entries,
            args.dataset,
            create_if_missing=args.allow_quarantine_bootstrap,
        )
        report["quarantine"]["attempted_rows"] += outcome["attempted_rows"]
        report["quarantine"]["inserted_rows"] += outcome["inserted_rows"]
        report["quarantine"]["stages"][validation_stage] = outcome

    try:
        # Fail before reading Bronze if the quarantine destination is not the
        # Terraform-reviewed Iceberg table. This prevents a valid Silver write
        # after the job has discovered that it cannot retain future rejects.
        if args.quarantine_table:
            quarantine.ensure_quarantine_table(
                spark,
                args.quarantine_table,
                args.quarantine_table_location,
                args.dataset,
                create_if_missing=args.allow_quarantine_bootstrap,
            )
        bronze = spark.read.parquet(args.bronze_path)

        if args.dataset == "account_changes_batch":
            transformed = account.transform(bronze, cfg, ingestion_ts)
        elif args.dataset == "device_lookup_batch":
            transformed = device.transform(bronze, cfg, ingestion_ts)
        else:
            # Only audit needs these two business-reference CSVs.
            partners, providers = load_references(
                spark, args.partner_reference, args.provider_reference
            )
            report["references"] = {
                "partner_path": args.partner_reference,
                "provider_path": args.provider_reference,
                "partner_rows": partners.count(),
                "provider_rows": providers.count(),
            }
            api_types = load_api_types()
            transformed, excluded = audit.transform(
                bronze, cfg, partners, providers, api_types
            )
            excluded = excluded.persist(StorageLevel.MEMORY_AND_DISK)
            cached.append(excluded)
            excluded_count = excluded.count()
            report["excluded_test_identity_rows"] = excluded_count
            if excluded_count:
                excluded_path = uri(run_root, "excluded_test_identities")
                write_parquet_once(excluded, excluded_path)
                report["artifacts"]["excluded_test_identities"] = excluded_path

        transformed = transformed.persist(StorageLevel.MEMORY_AND_DISK)
        cached.append(transformed)

        # Evaluate conversion flags and YAML rules before deduplication. This
        # prevents two distinct invalid source rows from both collapsing to null.
        raw_checked = annotate(
            transformed, cfg, args.run_date, check_unique=False
        ).persist(StorageLevel.MEMORY_AND_DISK)
        cached.append(raw_checked)
        raw_metrics = metrics(raw_checked, cfg)
        report["before_deduplication"] = raw_metrics
        if raw_metrics["overall"]["rows"] == 0:
            raise QualityBlocked("Approved Bronze input contains no records")
        if raw_metrics["overall"]["issue_rows"]:
            raw_issues_path = uri(run_root, "issues_before_deduplication")
            write_parquet_once(issue_rows(raw_checked, cfg), raw_issues_path)
            report["artifacts"]["issues_before_deduplication"] = raw_issues_path
        # One quarantine entry holds all error reasons for each rejected record.
        # Warning-only rows remain candidates and are represented in issue files.
        if raw_metrics["overall"]["error_rows"]:
            quarantine_rows("RAW_DQ", raw_checked)

        # Never send a row that failed raw parsing or YAML validation to the
        # normal Silver table. Quarantine has already committed before this
        # filter is evaluated, so a failed quarantine write stops publication.
        accepted_raw = raw_checked.filter(~F.col("_has_errors")).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        cached.append(accepted_raw)
        report["accepted_rows_before_deduplication"] = (
            raw_metrics["overall"]["rows"] - raw_metrics["overall"]["error_rows"]
        )

        # Exact business duplicates are safe to remove. The later uniqueness
        # check quarantines all remaining rows for an ID with different payloads.
        candidate = exact_deduplicate(accepted_raw, cfg).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        cached.append(candidate)
        candidate_checked = annotate(
            candidate, cfg, args.run_date, check_unique=True
        ).persist(StorageLevel.MEMORY_AND_DISK)
        cached.append(candidate_checked)
        candidate_metrics = metrics(candidate_checked, cfg)
        report["after_exact_deduplication"] = candidate_metrics
        if candidate_metrics["overall"]["issue_rows"]:
            candidate_issues_path = uri(run_root, "issues_after_deduplication")
            write_parquet_once(
                issue_rows(candidate_checked, cfg), candidate_issues_path
            )
            report["artifacts"]["issues_after_deduplication"] = candidate_issues_path
        if candidate_metrics["overall"]["error_rows"]:
            quarantine_rows("CANDIDATE_DQ", candidate_checked)

        accepted_candidate = candidate_checked.filter(
            ~F.col("_has_errors")
        ).select(*candidate.columns).persist(StorageLevel.MEMORY_AND_DISK)
        cached.append(accepted_candidate)
        report["accepted_rows_after_deduplication"] = (
            candidate_metrics["overall"]["rows"]
            - candidate_metrics["overall"]["error_rows"]
        )

        if args.mode == "publish":
            conflicts = target_preflight(spark, args.table, accepted_candidate, cfg).persist(
                StorageLevel.MEMORY_AND_DISK
            )
            cached.append(conflicts)
            conflict_count = conflicts.count()
            report["existing_silver_conflicts"] = conflict_count
            if conflict_count:
                conflict_path = uri(run_root, "existing_silver_conflicts")
                write_parquet_once(conflicts, conflict_path)
                report["artifacts"]["existing_silver_conflicts"] = conflict_path
                quarantine_rows("TARGET_CONFLICT", conflicts)
                # A conflicting ID must not overwrite the existing Silver row.
                # Keep all unrelated accepted IDs eligible for publication.
                publishable = accepted_candidate.join(
                    conflicts.select("record_id").distinct(),
                    "record_id",
                    "left_anti",
                ).persist(StorageLevel.MEMORY_AND_DISK)
                cached.append(publishable)
            else:
                publishable = accepted_candidate

            publishable_count = publishable.count()
            report["eligible_rows_for_publish"] = publishable_count
            # An all-rejected batch is a completed quarantine operation, not a
            # reason to append an empty Iceberg snapshot or fail the EMR step.
            report["iceberg_snapshot_id"] = (
                publish(spark, args.table, publishable, cfg)
                if publishable_count else None
            )
            report["target_table"] = args.table

        rejected_count = (
            raw_metrics["overall"]["error_rows"]
            + candidate_metrics["overall"]["error_rows"]
            + report.get("existing_silver_conflicts", 0)
        )
        report["status"] = (
            "succeeded_with_quarantine"
            if rejected_count and args.quarantine_table
            else "succeeded_with_rejections"
            if rejected_count else "succeeded"
        )
        return 0
    except QualityBlocked as exc:
        LOG.error("Run blocked: %s", exc)
        report["status"] = "blocked"
        report["error"] = str(exc)
        return 2
    except Exception as exc:
        LOG.exception("Silver run failed")
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        return 1
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(summary_path, report)
        for frame in reversed(cached):
            frame.unpersist(blocking=False)


def main(argv=None):
    """Create the Spark session, run once, and return a process exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args(argv)
    spark = SparkSession.builder.appName(
        f"silver-{args.dataset}-{args.run_id}"
    ).getOrCreate()
    try:
        return execute(spark, args)
    finally:
        spark.stop()
