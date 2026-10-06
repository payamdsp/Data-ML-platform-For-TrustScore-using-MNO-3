# Lotus sandbox pipeline run handbook

## Before deploying

Apply the `accounts/data-sandbox/security` Terraform root before
`accounts/data-sandbox/infra`, reviewing each plan. `emr.tf` is untouched:
an ordinary infra apply can still create/retain its **separate static EMR
cluster and cost**. Terraform validation does not check actual AWS
permissions, S3 artifacts, or job execution.

Required artifacts in S3:

- `s3://lotus-sandbox-bronze-landing-data/artifacts/silver/run_trust_score_silver.py`
  from `project_lotus/run_trust_score_silver.py`.
- `s3://lotus-sandbox-bronze-landing-data/artifacts/silver/silver_pipeline.zip`
  with the importable `silver_pipeline` package at the ZIP root.
- `run_gold.py`, `gold_pipeline.zip`, and `lotus_sandbox.yaml` under
  `s3://lotus-sandbox-gold-curated-data/pipeline-code/gold/auto/`.

The Gold source artifacts are **not in this checkout**. Supply and verify
them before an end-to-end run. Confirm EMR runtime S3, Glue, and Lake
Formation grants as well.

## Automatic flow

The landing bucket is `s3://lotus-sandbox-bronze-landing-data/`. The S3
EventBridge rule matches `.parquet` objects directly under these three
independent dated folders:

```text
bronze/account_changes_batch/YYYY-MM-DD/*.parquet
bronze/device_lookup_batch/YYYY-MM-DD/*.parquet
```

The first object for a dataset/date starts its own 900-second wait; later
objects do not restart it. Afterward the Bronze validator scans the full
folder. PASS starts that dataset's Silver execution. The first passing
Silver execution for the date creates the transient EMR cluster; other
Silver executions reuse it. Silver permits one retry with a new run ID.
Gate 2 releases Gold only when both enabled Silver datasets succeed. Gold
runs `prepare`, `accounts`, `customers`, `validate`, `publish` sequentially,
with one automatic retry per failed stage. The cluster terminates after
successful publish or exhausted Gold retry. The default 24-hour idle
timeout is a fallback if Gold never starts; increase it if the three
datasets can arrive more than a day apart.

Tiny JSON objects under `control/bronze-arrival-claims/` provide the atomic
first-arrival claim. Objects under `control/pipeline-claims/cluster/` retain
the shared daily EMR cluster ID/status. There
is no orchestration Iceberg table and there are **no DynamoDB tables**. The
Bronze validation and Silver-to-Gold control tables remain Iceberg because
they hold actual gate results. A claimed arrival date is not retriggered by
later files.

## Config objects, parameters, and Spark submit

Terraform seeds these JSON objects once. Future applies deliberately ignore
content changes: inspect/edit the **live S3 object** for the next run, and
keep the reviewed value in source control.

| Job | Live JSON config | Repo seed |
| --- | --- | --- |
| Silver | `s3://lotus-sandbox-silver-conformed-data/pipeline-code/silver/job_config.json` | `accounts/data-sandbox/infra/silver_job_config.tf` |
| Gold | `s3://lotus-sandbox-gold-curated-data/pipeline-code/gold/auto/job_config.json` | `accounts/data-sandbox/infra/gold_job_config.json` |

The control Lambda reads the applicable JSON before each Silver attempt or
Gold stage, builds the complete `spark-submit` argument array, and the state
machine passes it to the EMR `command-runner.jar` step. The builder is
`project_lotus/lambdas/silver_gold_control/silver_gold_control.py`.

Silver's config has `application`, `python_package`, `artifact_root`,
`mode`, `spark_submit_flags`, `spark_confs`, and per-dataset `table`,
`quarantine_table`, `quarantine_table_location`. The workflow adds
`--dataset`, `--run-date`, `--run-id`, and `--bronze-path`. The application
is the existing `run_trust_score_silver.py`, importing
`silver_pipeline.runner`. Example for one account batch:

```text
spark-submit --master yarn --deploy-mode cluster \
  --py-files s3://lotus-sandbox-bronze-landing-data/artifacts/silver/silver_pipeline.zip \
  s3://lotus-sandbox-bronze-landing-data/artifacts/silver/run_trust_score_silver.py \
  --dataset account_changes_batch --mode publish --run-date 2026-08-16 \
  --run-id <unique-attempt-id> \
  --bronze-path s3://lotus-sandbox-bronze-landing-data/bronze/account_changes_batch/2026-08-16/ \
  --artifact-root s3://lotus-sandbox-silver-conformed-data/run-artifacts/ \
  --table glue_catalog.lotus_sandbox_silver_conformed.account_changes_batch \
  --quarantine-table glue_catalog.lotus_sandbox_silver_conformed.silver_quarantine_account_changes_batch \
  --quarantine-table-location s3://lotus-sandbox-silver-conformed-data/quarantine/iceberg/silver_quarantine_account_changes_batch/
```

Gold's JSON supplies `application`, `python_package`, `config` (the Gold
job YAML, distinct from this JSON), `job`, `staging_root`,
`spark_submit_flags`, `spark_confs`, `first_run_watermark_from`. The
workflow adds the common `--run-id` and each `--stage`. On a first run,
`prepare` alone gets `--first-run --watermark-from <configured timestamp>`.

```text
spark-submit --deploy-mode cluster [configured memory/cores and --conf flags] \
  --py-files s3://lotus-sandbox-gold-curated-data/pipeline-code/gold/auto/gold_pipeline.zip \
  s3://lotus-sandbox-gold-curated-data/pipeline-code/gold/auto/run_gold.py \
  --config s3://lotus-sandbox-gold-curated-data/pipeline-code/gold/auto/lotus_sandbox.yaml \
  --job lineage --run-id <gold-run-id> \
  --staging-root s3://lotus-sandbox-gold-curated-data/staging/lineage_auto/ \
  --stage prepare
```

## Running and checking

1. Verify job artifacts, live JSON configs, table grants, EMR network and
   roles. Land Parquet files in the three dated folders. Normal operation
   requires no manual state-machine start. A test object in a production
   date starts that date's 15-minute clock.
2. Monitor `lotus-sandbox-bronze-arrival` per folder, then
   `lotus-sandbox-trust-score-bronze-silver`, then Gold and EMR steps.
   CloudWatch `/aws/vendedlogs/states/...` and Lambda logs show failures.
3. Inspect the JSON claim objects for arrival and cluster status
   (`creating` → `ready` → `terminated`) and query
   `control.silver_gold_control` for the two Silver outcomes. Gold's release
   object is at `s3://lotus-sandbox-silver-conformed-data/gold-release/run_date=YYYY-MM-DD/release.json`.

For a controlled Bronze revalidation, invoke the validator with
`{"dataset_name":"account_changes_batch","run_date":"2026-08-16","folder_path":"s3://lotus-sandbox-bronze-landing-data/bronze/account_changes_batch/2026-08-16/"}`.
The Silver execution name is deterministic per dataset/date, so it will
not start a second execution with that same name. Do **not** casually
delete claim objects or edit Iceberg state to rerun a date: that can
duplicate publication or leak a cluster. Design a recovery run explicitly.
