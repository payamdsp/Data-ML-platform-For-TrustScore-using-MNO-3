# Project Lotus — Trust Score Silver job

> **Current active-pipeline note:** the deployable implementation is
> `project_lotus/silver_pipeline` (with `run_trust_score_silver.py` as its EMR
> entry point when that source directory is packaged on `PYTHONPATH`). Its
> production DQ policy is **quarantine bad rows, then publish good rows**. See
> [the active quarantine guide](docs/docs/silver_pipeline_quarantine.md) and
> [the three-table Terraform definition](../accounts/data-sandbox/infra/silver_quarantine_iceberg.tf)
> for the reviewed current interface. Historical examples later in this README
> predate the replacement pipeline and must not be used as its CLI contract.

This repository now contains one deployable PySpark job for the three Trust
Score Bronze-to-Silver datasets:

- `account_changes_batch`
- `device_lookup_batch`
- `audit_trail_services_3`

The job is designed to run as a Spark step on Amazon EMR.  It keeps the
transformation code, data-quality decisions, and storage code separate so a
reader can see what each part does and why it exists.

## What was used as the specification

The implementation was reconciled against the supplied design PDF, reviewed
Bronze-to-Silver mapping, before/after examples, TS004 notebook and unit-test
materials, and the existing client cleaning notebooks in this repository.

The reviewed mapping is treated as the published P1 output contract where it
conflicts with legacy notebook output.  In particular:

- source identifiers are preserved as the default `record_id` / `source_event_id`;
- account and device P1 outputs do not publish legacy enrichment-only fields;
- audit P1 output does not publish raw MSISDN, source IP, response code, or
  operation;
- malformed or invalid data is quarantined instead of being silently written to
  Silver.

The supplied presentation provides useful downstream feature context but does
not define additional Silver transformations.  Fraud-label processing is also
not included because it is a separate source/contract from the three requested
tables.

## Layout

    run_trust_score_silver.py          EMR entry point
    project_lotus/silver/
      job.py                           argument parsing and orchestration
      transforms.py                    table-specific transformation rules
      dq.py                            record-level DQ rules and metrics
      io.py                            Bronze reads, Iceberg writes, quarantine writes
      common.py                        safe parsing, timestamps, deduplication helpers
      constants.py                     approved mappings and output contracts
    examples/
      trust_score_silver_emr_smoke.json  safe starting configuration
      emr-silver-step.json               EMR step template
    terraform/silver_iceberg/           Terraform Glue/Iceberg bootstrap
    tests/silver/                      focused transformation tests
    requirements-silver-dev.txt        optional local development dependencies

## Transformations covered

### Account changes

The job validates the source ID and change key, cleans sentinel values, parses
the `notes` JSON safely, standardizes the MNO, uppercases the change type, and
converts the source timestamp to the configured business time zone with DST
handling.  It produces the reviewed P1 fields and provenance columns.

The `notes.mno` value is uppercased when supplied, with
`BELL_LANDLINE` normalized to `BELL`. When `notes.mno` is absent but any known
notes field (`subId`, `msisdnChangeSide`, `correlationId`, or `otherMSISDN`) is
present, the job infers `BELL`. When no known notes field is present, it infers
`ROGERS`. Malformed `notes` JSON produces a null MNO and a DQ error; that row
is quarantined, while other valid rows can still be published.

### Device lookup

The job cleans `\\N`, blank, and `MISSING_DATA` sentinel values; preserves
meaningful IMEI/IMSI variants; standardizes MNO and status; parses relevant
`notes` attributes; and converts timestamps to the configured time zone.  DQ
rules understand that an event can be IMEI-related or IMSI-related, so they do
not require both identifiers on every row.

TAC is retained only when the explicit `legacy` output profile is requested.
It is absent from the reviewed P1 output profile.

### Audit trail services 3

The job trims and normalizes provider fields, uses the approved operation to
API-type mapping taken from the client implementation, preserves the supplied
SHA-256 MSISDN hash and encrypted MSISDN, converts timestamps, and enriches
partner and service-provider values from supplied reference data.  The
reference reader supports both the existing property/value configuration table
and simple direct reference files for testing.

Known test traffic is excluded only when an approved test-reference flag or
an explicitly enabled, versioned test-ID list is supplied.  This avoids hiding
production data based on an unreviewed hard-coded list.

## Data quality and quarantine

The active `silver_pipeline` applies record-level checks for required fields,
source-ID types, timestamp validity, valid event combinations, duplicate keys,
JSON parsing, and relevant reference-data enrichment. It writes failed rows
and their rule IDs to the dataset-specific Iceberg quarantine table, removes
those rows from the publish candidates, and continues with valid rows. After
exact-row deduplication of accepted raw rows, every remaining row sharing a
non-unique `record_id` is quarantined; an incoming row that conflicts with an
existing Silver ID is also
quarantined instead of replacing the existing record. Warning-only rows remain
candidates and appear in issue artifacts.

The `QUARANTINE_BAD_PUBLISH_GOOD` policy returns exit code 0 when all required
quarantine writes and the eligible Silver write succeed, even if some rows were
rejected. The summary status is `succeeded_with_quarantine` in that case. A
failed quarantine write still fails the job and prevents Silver publication;
an empty Bronze input still returns exit code 2. The active pipeline does not
make a CloudWatch metric API call. Every run's artifacts use its run ID, so
later runs do not overwrite earlier evidence. The detailed behavior, table
contract, recovery flow, and required command arguments are in
[the active quarantine guide](docs/docs/silver_pipeline_quarantine.md).

## Important runtime choices

The input source time zone, deduplication key, and ordering column are required
configuration rather than hidden assumptions.  Confirm these with the source
owner before production use.  For example, if Bronze timestamps are UTC and
the business output must be Toronto local time, use:

    --source-time-zone UTC --target-time-zone America/Toronto

The default `source_id` record-ID mode directly preserves the reviewed source
ID.  `sha256` mode exists only for account/device compatibility with the legacy
client output and requires the `legacy` output profile.

## Configuration

Start by copying `examples/trust_score_silver_emr_smoke.json`, replace every
`<...>` value and the example smoke database suffix, and keep the JSON outside the deployment ZIP.  The job accepts
configuration from a local file or an S3 URI:

    spark-submit run_trust_score_silver.py --config s3://<config-bucket>/trust-score/silver/run.json

The configuration may contain a `tables` list for all three datasets or one
dataset name for a focused smoke run.  It needs:

- input URI and format for each selected Bronze dataset;
- one shared `source_time_zone` and `target_time_zone`, plus a
  `dedup_key` and `dedup_order_column` mapping for every selected dataset;
- `output_root`, `warehouse_uri`, Glue catalog/database, schema version, and
  write/DQ policy;
- audit partner and service-provider reference URI/format when audit is
  selected.

Bronze column matching is case-insensitive, but a renamed field is deliberate
schema drift and stops the job with a clear error.  Update the mapping only
after the source contract has been reviewed.

The `property_value` audit-reference setting matches the existing client
`partner_configuration.csv`: a six-column, headerless CSV with multi-line
quoted values.  The job assigns its positional names (`seq`, `partner_id`,
`service_provider_id`, `section`, `property_name`, `property_value`) before
validating and joining it.  Use the `direct` layout only for an already
normalized reference with headers.

## Terraform Iceberg bootstrap

The [Terraform Silver Iceberg root](terraform/silver_iceberg/README.md)
creates the Glue database and the three initial Iceberg table contracts in
`lotus-sandbox-silver-conformed-data`.  Run `terraform init`, `plan`, and
`apply` from that directory before the first Silver Spark write.
The accompanying `terraform.tfvars.example` is a safe isolated smoke-run
starting point.

Terraform owns the initial schema, stable Iceberg field IDs, event-day
partition transforms, compression properties, and table protection.  Spark
owns data files, snapshots, and additive schema evolution.  This separation is
important: Iceberg schema changes must be made through Spark/Iceberg metadata,
not by replacing Glue tables from Terraform.  The module therefore uses
`prevent_destroy` and ignores post-bootstrap open-table-format drift.

For Phase 1, keep additive schema evolution enabled (`enable_schema_evolution:
true`) for reviewed nullable columns.  Existing rows remain readable because
Iceberg assigns stable field IDs and returns null for a newly added field.  Treat
renames, removals, type changes, and nullability tightening as explicit
migrations.  Phase 2 can extend the same table history after its schema has
been approved; it does not require rebuilding Phase 1 data.

## Local checks

Python 3.10 or 3.11 and Spark 3.5 are the intended development versions.
Install the optional development dependency, then run the unit tests:

    python -m pip install -r requirements-silver-dev.txt
    python -m pytest tests/silver -q

For a local no-write transformation/DQ check, use a one-table configuration
with accessible local input paths and set `write_mode` to `none`:

    python run_trust_score_silver.py --config examples/trust_score_silver_emr_smoke.json

No-write mode still calculates and reports DQ results, but it does not create
Silver, quarantine, or metrics outputs.

## Deploy and test on EMR

Use an EMR 7.13.0 (or compatible Spark 3.5 EMR) cluster with Iceberg enabled.
EMR 7.13 includes Spark 3.5.6 and Iceberg 1.10, so the job does not download a
second, incompatible Iceberg runtime.  See the [EMR 7.13 release notes](https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-7130-release.html)
and [EMR Iceberg guide](https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-iceberg.html).

1. Create an isolated release directory and ZIP the `project_lotus` package.

       Compress-Archive -Path project_lotus -DestinationPath project_lotus.zip -Force
       aws s3 cp run_trust_score_silver.py s3://<artifact-bucket>/trust-score/releases/<release>/run_trust_score_silver.py
       aws s3 cp project_lotus.zip s3://<artifact-bucket>/trust-score/releases/<release>/project_lotus.zip
       aws s3 cp examples/trust_score_silver_emr_smoke.json s3://<config-bucket>/trust-score/silver/smoke.json

2. Set the cluster classifications.  Replace all placeholders before creating
   the cluster.

       [
         {
           "Classification": "iceberg-defaults",
           "Properties": {"iceberg.enabled": "true"}
         },
         {
           "Classification": "spark-defaults",
           "Properties": {
             "spark.sql.extensions": "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
             "spark.sql.catalog.glue_catalog": "org.apache.iceberg.spark.SparkCatalog",
             "spark.sql.catalog.glue_catalog.type": "glue",
             "spark.sql.catalog.glue_catalog.warehouse": "s3://<data-bucket>/trust-score/iceberg-warehouse/",
             "spark.sql.defaultCatalog": "glue_catalog"
           }
         }
       ]

   This follows AWS's [Iceberg Spark configuration](https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-iceberg-use-spark-cluster.html)
   and [Glue catalog setup](https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-spark-glue.html).

3. Edit `examples/emr-silver-step.json` with the release and configuration S3
   URIs, then submit it:

       aws emr add-steps --cluster-id j-<cluster-id> --steps file://examples/emr-silver-step.json

   Monitor with:

       aws emr describe-step --cluster-id j-<cluster-id> --step-id s-<step-id>

   AWS documents [Spark submit steps](https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-spark-submit-step.html)
   and the [EMR CLI command](https://docs.aws.amazon.com/cli/latest/reference/emr/add-steps.html).

4. Start with the smoke configuration: one small, representative partition of
   each source, a new Glue database, a new `output_root`, and `dq_policy` set
   to `quarantine`.  Inspect the Iceberg tables, DQ summary JSON, rejected
   records, warnings, and test-exclusion outputs before pointing the job at a
   production Silver database.

The EMR runtime role needs least-privilege read access to Bronze/reference and
release artifacts; read/write/delete/multipart access to the selected Silver,
warehouse, quarantine, metrics, and log prefixes; Glue database/table access;
and KMS decrypt/encrypt/data-key permissions if those buckets use customer
managed keys.  AWS's [EMR instance-profile guidance](https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-iam-role-for-ec2.html)
and [encryption guidance](https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-encryption-enable.html)
cover the role and KMS side.  If Lake Formation governs the catalog, grant its
equivalent database/table and data-location permissions too.

## Production readiness checklist

- Approve the source time zone and deduplication keys/order for all three
  sources.
- Supply the effective-dated audit reference extracts and validate their
  uniqueness.
- Review unknown JSON keys from the initial run; use `notes_schema_mode=warn`
  only during controlled discovery, then return to `fail`.
- Validate the number of accepted/rejected rows against the Bronze input count.
- Review DQ evidence before enabling `dq_policy=fail` and before publishing to
  the production Glue database.
- Retain the run configuration, job logs, DQ summary, and Iceberg snapshot ID
  for auditability.
