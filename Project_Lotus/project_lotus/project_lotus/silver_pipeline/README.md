# Silver Pipeline

Spark/EMR job that turns approved Bronze Parquet into the client-aligned Silver
tables for three datasets: `account_changes_batch`, `device_lookup_batch`, and
`audit_trail_services_3`. It runs as one EMR step per dataset (see
`project_lotus/docs/docs/pipeline_orchestration.md` for how the three steps fit
into the wider Bronze → Silver → Gold flow).

Importing this package does nothing by itself — it does not start Spark, read
data, or publish anything. All work happens through `runner.main()`.

## What this job assumes is already true

- The Bronze data at `--bronze-path` has already passed schema validation
  (an external Lambda/Athena gate, not part of this package) and is immutable
  for the duration of the run.
- For `publish` mode, the target Iceberg Silver table and the dataset's
  quarantine Iceberg table already exist, created by Terraform. This job never
  creates a Silver table, and only bootstraps a quarantine table for an
  explicit local smoke test (`--allow-quarantine-bootstrap`, validate mode
  only).

## Pipeline stages

```
approved Bronze Parquet
  -> transform (per-dataset module) + derive normalized fields (e.g. MNO)
  -> RAW_DQ validation
       -> write issue artifact -> quarantine failed rows -> keep valid rows
  -> exact business-row deduplication of valid rows
  -> CANDIDATE_DQ validation (adds a duplicate-record_id check)
       -> write issue artifact -> quarantine duplicate-ID rows -> keep valid rows
  -> existing-Silver conflict check (publish mode only)
       -> write conflict artifact -> quarantine conflicting rows -> keep valid rows
  -> append unseen conforming rows to the Silver Iceberg table
```

The policy is **quarantine bad rows, then publish good rows** — a row-level DQ
failure no longer fails the whole EMR step. See
`project_lotus/docs/docs/silver_pipeline_quarantine.md` for the full quarantine
contract (table schemas, exit codes, recovery process).

## Module map

| Module | Responsibility |
| --- | --- |
| `runner.py` | CLI entry point (`main`). Parses arguments, builds the Spark session, and drives one dataset through every stage above. Writes `summary.json` no matter how the run ends. |
| `config.py` | Loads and validates each dataset's YAML rule contract (`configs/*.yaml`). Rejects an unknown key, rule kind, or duplicate rule ID instead of silently ignoring it. |
| `common.py` | Shared Spark expressions used by every transform: input resolution (`prepare`), string/number/timestamp/date cleaning, the `finish` projection into the publishable schema, and exact-duplicate collapsing. |
| `dq.py` | Turns YAML rules into Spark predicates (`violation`), tags each row with the rules it broke (`annotate`), rolls those tags into per-dataset/per-MNO metrics (`metrics`), and flattens them into one row per violation for the issue artifacts (`issue_rows`). |
| `quarantine.py` | Owns the three dataset-specific quarantine Iceberg tables: validating/bootstrapping them, building one recoverable row per rejected record, and appending only rows not already recorded. |
| `publish.py` | Validates the existing Silver table's schema against the YAML contract, finds rows whose `record_id` already exists with different data, and appends genuinely new rows without ever running a MERGE. |
| `references.py` | Loads and validates the two Revenue-export CSVs (`partners.csv`, `service_providers.csv`) that only `audit_trail_services_3` needs for enrichment. |
| `artifacts.py` | Writes the small `summary.json` run report to S3 or a local path. |
| `transforms/account.py` | `account_changes_batch`: infers MNO from the notes JSON (no reliable source column exists) and normalizes event fields. |
| `transforms/device.py` | `device_lookup_batch`: normalizes device/IMEI/IMSI events; MNO here is a real source column, so it only needs uppercasing. |
| `transforms/audit.py` | `audit_trail_services_3`: excludes known test partner/provider identities, normalizes API call events, and left-enriches with the partner/provider reference data. |
| `configs/*.yaml` | The reviewed, per-dataset contract: accepted input aliases, notes fields, output schema/types, and the DQ rule list. Treated as a signed-off spec, not a tuning knob — `config.py` fails the run rather than guessing at an unexpected shape. |
| `yaml/` | Vendored PyYAML, bundled into the EMR dependency ZIP so the cluster does not need a `pip install` step at runtime. Not project code — do not edit it here. |

## Running it

```
python -m silver_pipeline.runner \
  --dataset account_changes_batch \
  --mode validate \
  --run-date 2026-09-29 \
  --run-id local-smoke-001 \
  --bronze-path s3://.../bronze/account_changes_batch/ingest_date=2026-09-29/ \
  --artifact-root s3://.../silver/artifacts/
```

- `--mode validate` runs the full transform/DQ pipeline without touching
  Iceberg; quarantine writes are optional in this mode.
- `--mode publish` additionally requires `--table`, `--quarantine-table`, and
  `--quarantine-table-location`, and appends accepted rows to the real Silver
  table.
- `audit_trail_services_3` additionally requires `--partner-reference` and
  `--provider-reference`; the other two datasets reject those flags.

Exit codes: `0` success (`summary.json` records whether anything was
quarantined or rejected), `1` unhandled failure, `2` the run was blocked
before any row could be evaluated (e.g. an empty approved Bronze input).

## Why the code looks the way it does

A few decisions repeat throughout the modules and are worth knowing before
changing them:

- **No Python UDFs, no implicit destructive casts.** Every transform is built
  from `pyspark.sql.functions` expressions so Catalyst can optimize and
  vectorize the whole plan; a bad cast becomes a null value plus a `_bad_*`
  flag rather than a silently wrong number.
- **DQ runs before deduplication, on every raw row.** Two different invalid
  source values (e.g. two malformed IDs) can both normalize to `null` and
  look identical after grouping. Evaluating rules first, then deduplicating,
  keeps that from hiding a real data problem.
- **Quarantine never overwrites, only appends new deterministic IDs.** Spark
  3.5 on EMR can fail to resolve target columns in a SQL `MERGE` against
  Iceberg, so both quarantine and publish use a left-anti-join-then-append
  pattern instead: remove IDs the target already has, then append the rest.
- **Column resolution is case-insensitive on purpose.** Glue/Iceberg commonly
  lowercase field names even when the reviewed YAML uses client casing (e.g.
  `subId`). `common.py`, `publish.py`, and `quarantine.py` all normalize by
  lowercase and reject any column set that becomes ambiguous once case is
  ignored.
- **Every YAML contract is hashed.** `config.load()` stores a SHA-256 of the
  parsed config on `cfg["_hash"]`, and `runner.py` puts it in `summary.json`,
  so a run report can always prove exactly which rule contract was applied.

## Related docs

- `project_lotus/docs/docs/pipeline_orchestration.md` — how these three EMR
  steps fit into the end-to-end Bronze → Silver → Gold orchestration.
- `project_lotus/docs/docs/silver_pipeline_quarantine.md` — the quarantine
  table contracts, required job arguments, and record recovery process.
