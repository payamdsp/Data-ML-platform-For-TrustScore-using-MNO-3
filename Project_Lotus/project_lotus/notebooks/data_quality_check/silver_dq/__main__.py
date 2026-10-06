"""Command line entry point; run DQ against one silver table, write the canonical outputs.

    python -m silver_dq --checks <uri> --table <fqn> --dq-root <uri>

This is the scheduled path: spark-submit / Glue job / Airflow calls this. The notebooks are
the interactive path and stay independent; they carry their own copy of the engine because
the remote %%pyspark session cannot import Studio files. Two paths, on purpose.

No SparkSession is configured here. Memory, cores and the S3FileIO connection pool are the
launcher's business (spark-submit --conf), not the module's; getOrCreate() picks up whatever
the cluster hands it.

Exit code is the signal Sprint 2 consumers gate on:
    0   every rule passed; the silver run is trustworthy
    1   a rule failed; results ARE still written; the red summary is the whole point
    2   could not run: unreadable checks.yaml, or --table and --checks disagree
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from pyspark.sql import SparkSession

from .config import load_checks
from .runner import compute_dq, validate, write_results

EXIT_PASS = 0
EXIT_FAILED_RULES = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m silver_dq",
        description="Compute the configured DQ families (completeness, consistency, validity, "
                    "uniqueness, timeliness, victim_exclusion) against one silver Iceberg "
                    "table and write summary.json + metrics.json under "
                    "<dq-root>/results/dataset=<n>/run_date=<d>/.",
        epilog="Exit codes: 0 = every rule passed, 1 = a rule failed (results are still "
               "written), 2 = could not run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checks", required=True, metavar="URI",
        help="this table's checks.yaml; local path or s3:// uri. Everything about the "
             "table (primary keys, required columns, allowed values, max delay) is read "
             "from here; the flags below only say which table and where to write.")
    parser.add_argument(
        "--table", required=True, metavar="FQN",
        help="source silver table, e.g. trust_score_v1_poc_silver.tu_portps. Its last "
             "name segment must equal the yaml's `dataset`.")
    parser.add_argument(
        "--dq-root", required=True, metavar="URI",
        help="results root, e.g. s3://enstream-lake-silver-dev/poc/dq")
    parser.add_argument(
        "--run-date", metavar="YYYY-MM-DD",
        help="partitions the output path. Default: today on the driver, which is UTC on a "
             "cluster; pass it explicitly if the calendar date matters.")
    parser.add_argument(
        "--filter", metavar="SQL", dest="filter_sql",
        help="row filter, e.g. \"event_date = DATE'2024-08-01'\". Default: the whole table.")
    parser.add_argument(
        "--schema-dir", metavar="URI",
        help="check both outputs against the committed JSON Schema before writing; local "
             "path (repo copy) or s3:// prefix (published copy). Skipped when not given.")
    parser.add_argument(
        "--split-size-mb", type=int, metavar="MB",
        help="Iceberg read split size. Raise it (e.g. 512) when the scan dies with "
             "'Timeout waiting for connection from pool'; bigger splits, fewer "
             "concurrent S3 readers.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="compute and report, write nothing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = load_checks(args.checks)
    except Exception as exc:  # unreadable uri, bad yaml, contradictory config
        print(f"could not load {args.checks}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    # The output path is built from the yaml's dataset, not from --table: a mismatched pair
    # would file this table's numbers under another table's name.
    table_name = args.table.split(".")[-1]
    if table_name != cfg.dataset:
        print(f"--checks says dataset={cfg.dataset!r} but --table is {args.table!r}. "
              f"Results are filed under the yaml's dataset, so this pair would write one "
              f"table's numbers under another's name.", file=sys.stderr)
        return EXIT_ERROR

    run_date = args.run_date or date.today().isoformat()
    spark = SparkSession.builder.appName(f"silver_dq {cfg.dataset}").getOrCreate()

    reader = spark.read
    if args.split_size_mb:
        # Iceberg ignores spark.sql.files.maxPartitionBytes; this overrides the table's
        # read.split.target-size for this read only.
        reader = reader.option("split-size", str(args.split_size_mb * 1024 * 1024))
    source_rows = reader.table(args.table)
    if args.filter_sql:
        source_rows = source_rows.where(args.filter_sql)

    # An empty table is not an error here the way it is in the notebook: build_summary turns
    # row_count == 0 into a failed rule, so the run still produces the red signal a scheduler
    # needs. Exiting early would leave consumers unable to tell "DQ says bad" from "DQ never ran".
    summary, metrics = compute_dq(source_rows, cfg, run_date)

    print(f"{cfg.dataset}  run_date={run_date}")
    print(f"rows={summary['row_count']:,}  rules={summary['rule_count']}  "
          f"passed={summary['passed_count']}  failed={summary['failed_count']}  "
          f"=> {'PASS' if summary['passed'] else 'FAIL'}")
    for rule in summary["rules"]:
        if not rule["passed"]:
            print(f"  FAILED  {rule['family']:<12} {rule['rule']}  "
                  f"(violations={rule['violations']:,})")

    if args.schema_dir:
        validate(summary, metrics, args.schema_dir)
        print("summary + metrics both validate against the committed schema")

    if args.dry_run:
        print("dry run; nothing written")
    else:
        summary_path, metrics_path = write_results(summary, metrics, args.dq_root, cfg, run_date)
        print(f"WROTE {summary_path}")
        print(f"WROTE {metrics_path}")

    return EXIT_PASS if summary["passed"] else EXIT_FAILED_RULES


if __name__ == "__main__":
    sys.exit(main())
