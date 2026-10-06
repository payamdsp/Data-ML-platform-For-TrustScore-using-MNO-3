# Silver record quarantine in the active pipeline

This guide describes the active `project_lotus/silver_pipeline` implementation.
It uses **three dataset-specific Iceberg quarantine tables**, not one shared
table and not four quarantine tables.

## Runtime sequence

```text
approved Bronze Parquet
  -> transform and derive normalized fields, including MNO
  -> RAW_DQ validation
       -> write issue artifact -> quarantine failed rows -> keep valid rows
  -> exact-business-row deduplication of valid rows
  -> CANDIDATE_DQ validation
       -> write issue artifact -> quarantine duplicate-ID rows -> keep valid rows
  -> existing Silver conflict check (publish mode)
       -> write conflict artifact -> quarantine conflicting rows -> keep valid rows
  -> append unseen conforming rows to the normal Silver Iceberg table
```

The policy is **quarantine bad rows, then publish good rows**. A row-level DQ
error no longer fails the EMR step: if the required quarantine writes and any
eligible Silver write succeed, the job returns exit code 0 and reports
`succeeded_with_quarantine`. A batch with no eligible rows completes as a
quarantine-only run and does not append an empty Silver snapshot. If an Iceberg
quarantine write fails, publication stops and the job returns exit code 1; an
empty approved Bronze input returns exit code 2. Warning-only rows remain
candidates and are represented in issue artifacts; they are not quarantined.
A Bronze-file schema failure remains the validation Lambda's responsibility
because Spark never reads a rejected file. The job no longer sends quarantine
counts to CloudWatch; counts and snapshot IDs remain in `summary.json`.

## Table count and contracts

`accounts/data-sandbox/infra/iceberg.tf` creates the three normal Silver
business tables. `accounts/data-sandbox/infra/silver_quarantine_iceberg.tf`
creates exactly these three additional quarantine tables:

| Dataset | Quarantine table | S3 location |
| --- | --- | --- |
| `account_changes_batch` | `glue_catalog.lotus_sandbox_silver_conformed.silver_quarantine_account_changes_batch` | `s3://lotus-sandbox-silver-conformed-data/quarantine/iceberg/silver_quarantine_account_changes_batch/` |
| `device_lookup_batch` | `glue_catalog.lotus_sandbox_silver_conformed.silver_quarantine_device_lookup_batch` | `s3://lotus-sandbox-silver-conformed-data/quarantine/iceberg/silver_quarantine_device_lookup_batch/` |
| `audit_trail_services_3` | `glue_catalog.lotus_sandbox_silver_conformed.silver_quarantine_audit_trail_services_3` | `s3://lotus-sandbox-silver-conformed-data/quarantine/iceberg/silver_quarantine_audit_trail_services_3/` |

After Terraform applies both files, the catalog therefore contains **six**
Iceberg tables in total: three business tables plus three quarantine tables.
There is no fourth shared quarantine table in the active Terraform definition.

Each quarantine table has twelve common failure/audit fields (`quarantine_id`,
run/date/stage, error and warning JSON, raw payload, source file, quarantine
time/date, and recovery status) followed by the full typed Silver output schema
for its own dataset. It uses Iceberg v2, Parquet/Zstandard, no Iceberg
identifier fields, and an identity partition on `quarantined_date`.

## Required job arguments

Production `publish` requires the table/location pair for the selected
dataset. For example, an account run uses:

```text
--quarantine-table glue_catalog.lotus_sandbox_silver_conformed.silver_quarantine_account_changes_batch
--quarantine-table-location s3://lotus-sandbox-silver-conformed-data/quarantine/iceberg/silver_quarantine_account_changes_batch/
```

The runner validates the Iceberg format, exact schema, S3 location, and
`quarantined_date` partition before reading Bronze. Column-name case differences
introduced by Glue are tolerated, but missing, extra, ambiguous, or mistyped
columns still fail preflight. Passing a device or audit
quarantine table to an account run fails this schema check before any normal
Silver data can be published.

`validate` mode may omit quarantine for a local transformation check.
`--allow-quarantine-bootstrap` is for an isolated validate-mode EMR smoke test
only; production publish must use Terraform-created tables.

## Account MNO normalization and blocking rules

For `account_changes_batch`, MNO is derived before DQ and quarantine:

1. A malformed `notes` JSON value produces null MNO and the `flag_bad_notes`
   blocking rule fails.
2. Explicit `notes.mno = BELL_LANDLINE` becomes `BELL`.
3. Any other explicit MNO is uppercased.
4. With no explicit MNO, any of `subId`, `msisdnChangeSide`, `correlationId`,
   or `otherMSISDN` implies `BELL`.
5. With no explicit MNO and no known notes fields, MNO becomes `ROGERS`.

The account YAML has blocking rules for required/positive BIGINT `record_id`,
bad or overflowing IDs, required 64-hex AC hash, required event type, allowed
event-type/MNO combinations, required and parseable non-future timestamp,
required/allowed MNO, valid notes JSON, required lineage fields, and unique
`record_id` after exact-row deduplication.

## Anti-joins and operational behavior

The code has two distinct left anti-joins:

1. `publish.py` removes candidate `record_id` values already present in the
   normal Silver table. A separate preflight identifies an incoming ID whose
   business payload differs from existing Silver data; those rows are
   quarantined and removed from the publish candidates.
2. `quarantine.py` removes existing `quarantine_id` values before appending to
   the selected dataset's quarantine table. The ID includes the raw payload and
   source-row token so two identical invalid rows in one source file do not
   collapse before writing.

MNO inference always occurs first. The raw DQ rules then evaluate the inferred
value; only a failed rule produces a quarantine entry. One writer at a time is
required per dataset quarantine table because its anti-join and append are two
separate Iceberg operations. The three independent dataset tables may otherwise
be processed in parallel.

The run summary distinguishes `accepted_rows_before_deduplication`,
`accepted_rows_after_deduplication`, and `eligible_rows_for_publish`. The last
count excludes existing-Silver conflicts but can exceed newly inserted rows
because exact replays are removed inside `publish.py`. Rejected rows remain in
quarantine for later investigation; the job never marks them recovered.

## Terraform and recovery

Use the existing `silver_quarantine_iceberg.tf` file in the infrastructure root;
do not add a separate shared quarantine table. The EMR runtime role needs the
explicit Lake Formation permissions declared in
`emr_silver_lakeformation.tf`, plus its existing Glue, S3, Lake Formation, and
KMS access as applicable.

To recover a record, query the selected table for `PENDING_REMEDIATION`, fix
the upstream/reference data or an approved contract, rerun corrected immutable
input with a new run ID, and update recovery status through a reviewed process.
The Silver job never marks records recovered automatically.
