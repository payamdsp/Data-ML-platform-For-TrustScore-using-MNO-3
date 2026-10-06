# Pipeline Orchestration — Bronze Gate → Silver → Gold

Living context document for the end-to-end orchestration design.
**Status:** all open questions resolved (§9). Full architecture in §12. Ready for implementation
planning; still a documentation-only artifact — no code has been written against this plan yet.

Last updated: 2026-09-28

---

## 1. Purpose

Defines how a daily snapshot moves from Bronze to Gold:

1. **Bronze schema validation** (Lambda, per dataset per dated folder)
2. **Gate query** (Athena, over the validation control table)
3. **Silver** (EMR, 3 datasets **in parallel**)
4. **Silver→Gold handoff control** (gate on all Silver datasets succeeding)
5. **Gold** (EMR, 5 stages **strictly sequential**)

Failure at any gate stops the flow and raises SNS.

---

## 2. Flow overview

```
                 ┌──────────────────────────────────────────┐
                 │ Bronze dated folder complete             │
                 │ s3://.../<dataset>/ingest_date=YYYY-MM-DD/│
                 └──────────────────┬───────────────────────┘
                                    │ one invocation per dataset
                                    ▼
                 ┌──────────────────────────────────────────┐
                 │ Lambda: bronze_schema_validator          │
                 │ writes per-file rows to control table    │
                 │ SNS on overall_status != PASS            │
                 └──────────────────┬───────────────────────┘
                                    ▼
                 ┌──────────────────────────────────────────┐
                 │ GATE 1 (Athena): all files in the dated  │
                 │ folder PASS/WARN, none PENDING/FAIL/ERROR│
                 │ and file count matches folder_file_count │
                 └──────────────────┬───────────────────────┘
                        all datasets pass │ else STOP + SNS
                                    ▼
         ┌──────────────────────────┴──────────────────────────┐
         │          SILVER — 3 independent EMR steps           │
         │              (run in PARALLEL)                      │
         │  account_changes_batch │ device_lookup_batch │ audit │
         └──────────────────────────┬──────────────────────────┘
                        all 3 succeed │ any fail → STOP + SNS
                                    ▼
                 ┌──────────────────────────────────────────┐
                 │ GATE 2: silver→gold trigger control table│
                 │ + manual config flag to release Gold     │
                 └──────────────────┬───────────────────────┘
                                    ▼
         ┌──────────────────────────┴──────────────────────────┐
         │      GOLD — 5 SEQUENTIAL stages, ONE run_id         │
         │ prepare → accounts → customers → validate → publish │
         │  any stage fails → STOP (no later stage) + SNS      │
         └─────────────────────────────────────────────────────┘
```

---

## 3. Stage 1 — Bronze schema validator Lambda

**Code:** `project_lotus/lambdas/bronze_schema_validator/bronze_schema_validator.py`
**Handler:** `lambda_handler`

### Input event

```json
{
  "dataset_name": "account_changes_batch",
  "folder_path": "s3://bucket/.../ingest_date=2026-09-03/",
  "schema_version": 1
}
```

`schema_version` is optional — omitted means the registry's `current.json` → `active_version` is used.
**One invocation validates one dataset for one dated folder.** Four invocations are needed to cover all datasets.

### Supported datasets

| Dataset | Has a Silver job? |
|---|---|
| `account_changes_batch` | Yes |
| `device_lookup_batch` | Yes |
| `audit_trail_services_3` | Yes |
| `partner_configuration` | **No** — see open question O-3 |

### Environment variables

| Variable | Required | Default |
|---|---|---|
| `CONTROL_DATABASE` | Yes | — |
| `CONTROL_TABLE` | Yes | — |
| `SCHEMA_BUCKET` | Yes | — |
| `SCHEMA_PREFIX` | No | `schema_registry` |
| `ATHENA_WORKGROUP` | No | `primary` |
| `ATHENA_OUTPUT` | No | `""` (uses workgroup config) |
| `SNS_TOPIC_ARN` | No | `""` (alerting disabled if unset) |
| `MAX_WORKERS` | No | `8` |
| `MERGE_CHUNK_SIZE` | No | `100` |
| `SEMANTIC_SAMPLE_SIZE` | No | `50` |
| `SEMANTIC_BATCH_SIZE` | No | `64` |
| `SEMANTIC_MAX_ROWS_SCANNED` | No | `1024` |

Schema registry layout: `s3://$SCHEMA_BUCKET/$SCHEMA_PREFIX/<dataset>/current.json` and `.../v<N>.json`.

### Execution order (matters for the gate)

1. Lists all `.parquet` files under the folder. **Zero files raises** — the folder must be complete before invoking.
2. **Registers every file as `PENDING`** via Athena `MERGE`.
3. Validates files in parallel (`MAX_WORKERS` threads), reading only the Parquet footer via S3 range GETs.
4. Merges the real per-file results.

Step 2 is deliberate: if the Lambda dies mid-run, the gate sees `PENDING` and blocks rather than seeing nothing and passing.

### Per-file status semantics

| Status | Meaning | Gate treatment |
|---|---|---|
| `PASS` | Exact canonical type match | Allowed |
| `WARN` | Non-breaking drift: extra columns, **or** physically-STRING column whose sampled values are coercible to the registry type | **Allowed** |
| `FAIL` | Breaking drift: removed columns, or a type mismatch not eligible for string coercion | Blocks |
| `ERROR` | Exception while reading the file | Blocks |
| `PENDING` | Registered but result never written (crashed run) | Blocks |

### Overall status returned by the Lambda

```
overall_status = PASS  iff  files_checked == folder_file_count
                       AND  (files_passed + files_warning) == folder_file_count
                 else BLOCK
```

SNS is published when `overall_status != PASS`.

### Control table columns

Written by `MERGE INTO "$CONTROL_DATABASE"."$CONTROL_TABLE"`:

`file_key`, `attempt_id`, `dataset_name`, `folder_path`, `file_name`, `file_path`, `file_etag`,
`file_size`, `folder_file_count`, `schema_version`, `status`, `added_columns_json`,
`removed_columns_json`, `datatype_changes_json`, `actual_schema_json`, `error_message`,
`attempt_count`, `validation_start_ts`, `validation_end_ts`

Key behaviours:

- `file_key = sha256(dataset_name | folder_path | file_path)` — **stable**, so re-validating the same folder **updates rows in place** rather than appending a new version. `attempt_count` increments.
- `folder_file_count` is stored on every row = number of files seen at validation time. The gate uses it to detect missing rows.
- `folder_path` is always normalised to a **trailing slash**.

---

## 4. Gate 1 — the Athena approval query

**Status: authoritative query supplied.** Adopted below, with one required fix flagged.

```sql
-- Gate 1: silver_gate_status = PASS only if the latest validation attempt for
-- EACH of the three Silver datasets, for THIS snapshot date, is clean.
WITH latest_attempts AS (
    SELECT
        dataset_name,
        attempt_id,
        ROW_NUMBER() OVER (
            PARTITION BY dataset_name
            ORDER BY validation_start_ts DESC
        ) AS rn
    FROM control.bronze_schema_validation
    WHERE dataset_name = :dataset_name          -- ONE dataset per gate check; see fix below
      AND folder_path  = :folder_path            -- that dataset's own dated folder
),

validation AS (
    SELECT
        v.dataset_name,
        COUNT(*)                                                  AS rows_found,
        MAX(v.folder_file_count)                                  AS files_expected,
        SUM(CASE WHEN v.status IN ('PASS', 'WARN') THEN 1 ELSE 0 END) AS files_allowed,
        SUM(CASE WHEN v.status NOT IN ('PASS', 'WARN') THEN 1 ELSE 0 END) AS blocking_files
    FROM control.bronze_schema_validation v
    JOIN latest_attempts l
      ON v.dataset_name = l.dataset_name
     AND v.attempt_id   = l.attempt_id
     AND l.rn = 1
    GROUP BY v.dataset_name
)

SELECT
    CASE
        WHEN COUNT(*) = 1
         AND SUM(
             CASE
                 WHEN rows_found = files_expected
                  AND files_allowed = files_expected
                  AND blocking_files = 0
                 THEN 1 ELSE 0
             END
         ) = 1
        THEN 'PASS'
        ELSE 'BLOCK'
    END AS dataset_gate_status
FROM validation;
```

### Required fix vs. the version as pasted

The version handed over filtered `latest_attempts` by `dataset_name IN (three datasets)` but by a
**single literal `folder_path`** built from `account_changes_batch`'s prefix. Applied to all three
datasets, that literal only matches `account_changes_batch` rows — `device_lookup_batch` and
`audit_trail_services_3` have different S3 prefixes and would join to **zero rows**, which the outer
`COUNT(*) = 3` check would then correctly report as `BLOCK`, but for the wrong reason (missing data,
not a real quality failure) and it would never distinguish "these two datasets haven't landed yet"
from "these two datasets failed validation."

**Fix adopted:** run the query **once per dataset**, each with its own `dataset_name` +
`folder_path` pair for the same snapshot date (e.g. `.../account_changes_batch/2026-09-11/`,
`.../device_lookup_batch/2026-09-11/`, `.../audit_trail_services_3/2026-09-11/`), and require all
three per-dataset results to come back `PASS` before releasing Silver. This also matches how the
orchestrator already invokes the Bronze Lambda — once per dataset — so the gate check re-uses the
same per-dataset loop.

Design points carried over from the original:

- Only the **latest** `attempt_id` per dataset counts (via `ROW_NUMBER() ... ORDER BY
  validation_start_ts DESC`), so a re-validated folder isn't double-counted against a stale attempt.
- `WARN` counts as allowed, matching the Lambda's own PASS rule.
- `rows_found = files_expected` catches a crashed Lambda that registered `PENDING` rows for only
  some files.

Orchestration rule: **proceed only if every required dataset's gate query returns `PASS`.** Any
`BLOCK`, including "no rows found yet" (folder not landed / Lambda not yet run), stops the flow —
retry the check rather than proceeding.

---

## 5. Stage 2 — Silver (parallel)

**Entry point:** `silver_pipeline.runner` (`main`)
**Datasets:** `account_changes_batch`, `device_lookup_batch`, `audit_trail_services_3`

The three datasets are independent — no cross-dataset reads — so all three run **in parallel**, on
their own clocks. Since Gate 1 is checked **per dataset** (§4, §9 O-8), a dataset's Silver step
starts as soon as *that dataset's own* Gate 1 passes — it does not wait for the other two Bronze
datasets to finish validating. Silver steps are therefore **staggered**, not launched in lockstep.

> **Revised (2026-09-29): this is not a Map/Parallel fan-out inside one execution.**
> An earlier version of this design put all three datasets in one Step Functions `Map` state inside
> a single execution. That is wrong for what "independent" actually means here:
> `account_changes_batch`, `device_lookup_batch` and `audit_trail_services_3` can each finish Bronze
> validation *hours apart* on the same calendar date — there is no single moment at which "start the
> Map" is a meaningful trigger, because there is no one event that means "all three are ready."
>
> The corrected design: **the Bronze schema validator Lambda starts one Step Functions execution per
> dataset**, the instant *that* dataset's own validation returns `PASS` — see §12.1's revised flow
> below. There is no Map, no fan-out, and no state machine that waits for more than one dataset.
> The join happens downstream instead, in Gate 2 (§6) — see the walkthrough in §12.2.

### Arguments

| Argument | Required | Notes |
|---|---|---|
| `--dataset` | Yes | One of the three above |
| `--mode` | No | `validate` (default) or `publish` |
| `--run-date` | Yes | Toronto business date, `YYYY-MM-DD` |
| `--run-id` | Yes | New unique ID per attempt; `[A-Za-z0-9][A-Za-z0-9_.-]{0,127}` |
| `--bronze-path` | Yes | The **gate-approved** immutable Parquet path |
| `--artifact-root` | Yes | S3 root for reports and issue rows |
| `--table` | publish only | Target Iceberg table |
| `--quarantine-table` | publish only | `catalog.database.table`, must differ from `--table` |
| `--quarantine-table-location` | publish only | `s3://` URI; paired with the above |
| `--partner-reference` | audit only | `partners.csv` |
| `--provider-reference` | audit only | `service_providers.csv` |

Constraints enforced by the runner:

- `publish` mode requires `--table`, `--quarantine-table` **and** `--quarantine-table-location`.
- `--quarantine-table` and `--quarantine-table-location` must be supplied **together**.
- `--artifact-root` and `--quarantine-table-location` must not be nested within each other.
- `audit_trail_services_3` **requires** both reference CSVs; the other two datasets **reject** them.
- Re-using a `--run-id` fails on artifact write (`mode("errorifexists")`) rather than overwriting.

### Exit codes and status

| Exit | `status` in `summary.json` | Meaning |
|---|---|---|
| 0 | `succeeded` / `succeeded_with_rejections` / `succeeded_with_quarantine` | Completed |
| 1 | `failed` | Unhandled failure |
| 2 | `blocked` | `QualityBlocked` — e.g. approved Bronze input had zero records |

`summary.json` is written to:
`{artifact_root}/v{schema_version}/{dataset}/{run_id}/summary.json`

Exit code 2 (`blocked`) is treated as a **failure** for SNS purposes — see §9 O-1.

### Retry policy — **decided: one automatic retry, then escalate (same as Gold, §7)**

On exit 1 or 2, retry the dataset's Silver step **once**, with a **freshly generated `--run-id`**
(not the same one — see §9 O-10 for why: `--run-id` reuse fails at the artifact-write step by
design, so this is a genuine second attempt from scratch, unlike Gold's in-place stage retry). If
the retry also fails or is blocked, stop that dataset's branch, write `status = failed|blocked` to
the trigger control table for it, and raise SNS on `silver-pipeline-alerts`. The other two Silver
branches are unaffected and continue independently.

### Failure handling

Any Silver step failing after its one retry → **SNS notification**, and that dataset's row in the
trigger control table is left at `failed`/`blocked`, which keeps Gate 2 from releasing Gold (§6).
Because the three run staggered/in parallel, the orchestrator must wait for **all three** to settle
(succeed after 0 or 1 retries, or fail after 1 retry) before Gate 2 can evaluate; it must not
release Gold on the first two successes while the third is still retrying.

---

## 6. Gate 2 — Silver → Gold trigger control

Two conditions must both hold before Gold starts:

1. **All three Silver datasets succeeded** for the snapshot date — recorded in a
   Silver→Gold trigger control table.
2. **Release config is written** — decided as **automatic by default, with a manual override**
   (§9 O-9): the normal daily run needs no person to act; an operator may also write/overwrite the
   same object to force a release, change config overlays, or trigger a repair run.

### Proposed trigger control table

> `NEEDS CONFIRMATION` — this table does not exist yet; shape proposed for review.

| Column | Type | Notes |
|---|---|---|
| `run_date` | `date` | Snapshot business date |
| `dataset_name` | `string` | Silver dataset |
| `silver_run_id` | `string` | The `--run-id` used |
| `status` | `string` | `succeeded` / `failed` / `blocked` |
| `rows_published` | `bigint` | From `summary.json` |
| `summary_uri` | `string` | Pointer to `summary.json` |
| `recorded_ts` | `timestamp` | Write time |

Gold releases only when all three datasets have `status = 'succeeded'` for `run_date`.
(Silver exit code `2`/`blocked` is recorded here as `blocked`, not `succeeded` — see §8, it also
raises SNS. A `blocked` row must gate Gold exactly like a `failed` one.)

### Release config — **decided: S3 JSON + EventBridge, written automatically by default**

Delivery mechanism: **an S3 object**, in a dedicated release-config prefix, picked up by
**EventBridge via S3 Event Notifications** (bucket → EventBridge → rule → target). This replaces
SSM/Step-Functions-input as the trigger surface while still feeding a Step Functions execution.

**Who writes it — decided (§9 O-9): automatic, with a manual override.** The default writer is an
automated step (e.g. a small Lambda subscribed to trigger-control-table writes, or a "wait for all
3 Silver branches" Step Functions state) that writes `release.json` itself the instant all enabled
Silver datasets show `succeeded` for `run_date` — the everyday run needs zero human action. An
operator can still write the same object by hand to force a release, override
`config_overlays`/`overrides`, or kick off a repair run; both paths converge on the identical
EventBridge → gate Lambda flow (§12), so nothing downstream needs to know which one happened.

**Object location and shape:**

```
s3://<bucket>/gold-release/run_date=2026-09-11/release.json
```

```json
{
  "run_date": "2026-09-11",
  "release_gold": true,
  "first_run": false,
  "dry_run": false,
  "config_overlays": ["conf/lineage/base.yaml", "conf/lineage/lotus_sandbox.yaml"],
  "overrides": []
}
```

Note: **`gold_run_id` is deliberately absent from this file** — it is generated downstream by the
gate Lambda at the moment Gold is released, not supplied manually. See §7.1 for why an explicit,
caller-supplied `--run-id` is still required on every one of the five Gold EMR steps despite being
machine-generated, and §12 for the full EventBridge wiring.

---

## 7. Stage 3 — Gold (5 sequential stages)

**Entry point:** `trust_score_05.lineage.jobs.staged_job` (`main_cli`)

### Terminology — important

Two different sets of five exist; do not conflate them.

- **Five execution STAGES** (what the orchestrator runs as 5 EMR steps):
  `prepare → accounts → customers → validate → publish`
- **Five output Gold TABLES** (what `publish` writes):
  `account_changes_canonical_events`, `msisdn_lifecycle`,
  `msisdn_lifecycle_account_mapping`, `msisdn_lifecycle_account_customer_mapping`,
  `normalized_canonical_events`

"Run the 5 jobs sequentially" = the five **stages** of **one logical run**, not five independent jobs.

### Hard orchestration constraints

These are enforced by the code and will fail the run if violated:

1. **Every stage must be passed the SAME `--run-id`.** Omitting `--run-id` raises
   `ConfigError: Every staged step requires the SAME explicit --run-id`.
2. **EMR step concurrency MUST be 1.** The code detects violation as
   `Duplicate running records; verify that EMR step concurrency is 1`.
3. **Stage N refuses to run until stage N−1's manifest exists** in the staging prefix
   (`Run <stage> successfully before this step`). The dependency is enforced by the job itself,
   not only by the orchestrator.
4. **`prepare` must be first.** Any other first stage raises
   `This run has no preparation manifest. Start with --stage prepare.`
5. **Config and code are frozen at `prepare`.** Resuming later stages with different config or a
   different package build raises `This run was staged with different code/config`.
6. **A new run's `run_id` must not already exist** in `gold_run_control`.
7. **A previous unfinished run blocks a new one** (`assert_no_running`).
8. Requires Iceberg for both `run_control` and `dq.output`.
9. All five Gold **target tables must already be provisioned** before the run starts.

### Per-stage arguments

Common to all five stages:

```
--stage {prepare|accounts|customers|validate|publish}
--run-id <SAME id for all five stages>
--config <path>            # repeatable; later overlays earlier
--set KEY=VALUE            # repeatable dotted-key override
```

Optional, **frozen at `prepare`** (must not differ on later stages):
`--watermark-from`, `--watermark-to`, `--phone-numbers`, `--first-run`, `--dry-run`,
`--run-timestamp`

Publish-only: `--resume-publish` (see recovery below).
Staging override: `--staging-root`.

### What each stage does

| # | Stage | Work | Writes Gold? |
|---|---|---|---|
| 1 | `prepare` | Reads Silver, resolves the slice (phone numbers + closure), builds canonical events / lifecycles / histories, runs **input** DQ | No |
| 2 | `accounts` | Account graph walk and assembly | No |
| 3 | `customers` | Customer graph walk from porting events | No |
| 4 | `validate` | Materialises all five Gold shapes, runs the **complete output DQ gate** | No |
| 5 | `publish` | MERGEs the five tables into Iceberg, closes the run | **Yes** |

Only `publish` mutates Gold, and only after all output checks have completed. A DQ abort at
`validate` stops the run before anything is published.

### Failure and retry semantics

- A failed stage **does not** close the run reservation — it stays `running` and is **retryable
  with the same `--run-id`**.
- A **completed** stage is skipped on retry (`Stage X already completed; reusing its immutable
  handoff`), so a retry resumes rather than restarting from scratch.
- A **failed `validate`** is a final reviewable result — it must not be overwritten or published.
  Fixing a business rule requires a **new logical run**, not a retry.
- `publish` keeps a per-table journal (`publish/<table>.json`, `writing` → `complete`). If a MERGE
  was interrupted with uncertain commit state, recovery is: confirm the old EMR application has
  stopped and no other writer touched Gold, then re-run publish with `--resume-publish`.
- Exit codes: `0` success, `2` configuration error, `1` any other failure.

### Orchestration rule

Run the five stages as five sequential EMR steps with concurrency 1. **If any stage exits
non-zero: do not run the following stages, route to the failure path, raise SNS.** The run
reservation is deliberately left open so the stage can be retried with the same `run_id`.

### Retry policy — **decided: one automatic retry, then escalate**

Each Gold stage gets **exactly one automatic retry** on failure. Because a failed stage leaves its
reservation open and resumes from the last completed stage (not from `prepare`), a retry is cheap —
it re-enters the same stage, not the whole run. If the retry also fails, **stop and escalate to a
human** (SNS) rather than retrying again automatically. In Step Functions terms: each Gold EMR-step
state gets `Retry: [{ MaxAttempts: 1, IntervalSeconds: 30 }]` with a `Catch` routing to the SNS
failure state on final failure. See §12 for the full state machine shape.

### 7.1 Who generates `gold_run_id`

**Decided: it is generated automatically** — but *where* matters, because the code has a hard
requirement that conflicts with a naive reading of "automatic":

> `staged_job.run()` raises `ConfigError("Every staged step requires the SAME explicit --run-id")`
> if `--run-id` is omitted on **any** of the five stages.

So "automatic" cannot mean "each EMR step generates its own ID" — that would violate constraint 1
in §7's hard-constraints list and constraint 6 (`run_id` must not already exist in
`gold_run_control`) would then apply five times to five different IDs instead of once. The correct
place to generate it is **once, in the orchestrator, before the first (`prepare`) stage is
launched** — specifically, in the EventBridge-triggered gate Lambda described in §6/§12, at the
moment Gate 2 passes and Gold is released. That single generated value is then passed as a fixed
input parameter to all five EMR steps in the Step Functions execution.

**Naming convention — decided: date + timestamp.**

```
gold-<run_date:YYYYMMDD>-<generation-timestamp:HHMMSS>
```

e.g. `gold-20260911-043217` (generated at 04:32:17 on the day Gold is released for `run_date`
2026-09-11). Generated **once**, by the gate Lambda, at the instant Gate 2 passes — not by any of
the five EMR steps themselves. That single string is then passed as a fixed input value
(`$.run_id`) to every one of the five Step Functions states, each of which supplies it as
`--run-id` on its `staged_job` invocation, satisfying the "same explicit `--run-id` on all five
stages" requirement above.

Two things worth keeping in mind with a date+timestamp ID rather than a random one:

- **Collision is possible in theory** (two Gate-2 passes for the same `run_date` within the same
  second — e.g. a manual re-release right after an automatic one), though unlikely given Gate 2
  fires at most a few times a day. If that ever matters, add a short random/hash suffix. Since
  `gold_run_control` already rejects a reused `run_id` (constraint 6 in §7), a collision **fails
  loudly** rather than silently double-running — it does not corrupt anything, it just needs a
  retry with a fresh timestamp.
- The timestamp in the ID is **generation time, not `run_date`'s business meaning** — don't parse
  it as anything other than "when Gold was kicked off for this business date." `run_date` is
  already carried separately in the Step Functions execution input for anything that needs the
  business date.

---

## 8. SNS notification points — **decided: separate topics per layer**

**Cost is not the deciding factor** — SNS charges per publish/delivery, not per topic, so N topics
at a given message volume cost the same as one topic carrying N times the traffic. Given that,
**separate topics per layer** is recommended: it lets different people/channels subscribe to each
layer (e.g. a data-quality channel for Bronze drift vs. an on-call pager for Gold publish failures)
without needing message-attribute filtering on a shared topic.

| Topic | Trigger | Source |
|---|---|---|
| `bronze-validation-alerts` | Bronze validation `overall_status != PASS` | Lambda (built in, needs `SNS_TOPIC_ARN`) |
| `bronze-validation-alerts` (or a `gate-alerts` topic) | Gate 1 returns any `BLOCK` dataset | Orchestrator |
| `silver-pipeline-alerts` | Any Silver dataset exits non-zero, **including exit code `2` (`blocked`)** — decided: exit 2 is alert-worthy, not a quiet stop | Orchestrator |
| `gold-pipeline-alerts` | Any Gold stage exits non-zero, including the final failure after the one retry in §7 | Orchestrator |

### Message payload contract (proposed, all topics)

```json
{
  "layer": "silver",
  "run_date": "2026-09-11",
  "dataset_or_stage": "device_lookup_batch",
  "run_id": "silver-device_lookup_batch-20260911-01",
  "status": "blocked",
  "exit_code": 2,
  "summary_uri": "s3://.../summary.json",
  "occurred_at_utc": "2026-09-11T04:32:10Z"
}
```

Keeping one shape across all three topics means a single downstream consumer (e.g. a Slack
Lambda subscriber) can handle all of them without three separate parsers.

---

## 9. Open questions / decisions needed

### Resolved

| ID | Question | Decision |
|---|---|---|
| O-1 | Is Silver exit code 2 (`blocked`) alert-worthy? | **Yes — treated as a failure, raises SNS on `silver-pipeline-alerts`.** Recorded as `blocked` (not `succeeded`) in the trigger control table, so it gates Gold exactly like `failed`. |
| O-2 | Delivery mechanism for manual Gold release config | **S3 JSON object + EventBridge S3 Event Notification**, triggering a gate Lambda. See §6, §12. |
| O-3 | Should Gate 1 require `partner_configuration` to be APPROVED before Silver runs? | **Ignored for now** — out of scope for this iteration. Revisit once `audit_trail_services_3` in Silver actually depends on it in production. |
| O-4 | Authoritative Gate 1 query | **Supplied and adopted**, with the per-dataset `folder_path` fix in §4. |
| O-5 | Who generates `gold_run_id`? | **Generated automatically, once, by the gate Lambda** at Gold-release time — not by each EMR step. Naming convention in §7.1. |
| O-6 | Gold stage retry policy | **One automatic retry per stage**, then escalate to SNS. See §7 retry policy. |
| O-7 | Are `tu_portps` / `mno_activation` in scope now? | **Not in scope now** (no data yet) — but confirmed in-scope for the project later. Design implication: keep both the Silver parallel fan-out (§5) and the Gold `prepare`-stage optional-source config **driven by a config list**, not hardcoded, so enabling them later is a config change. Pass each as a config flag/overlay at run time (e.g. `tu_portps.enabled`, `mno_activation.enabled`) rather than a code change — see §12 for where that config flows in. |
| O-8 | Does Gate 1 run once for all datasets, or per dataset? | **Per dataset — confirmed.** This matches the §4 fix directly: each dataset's own Gate 1 check runs against its own `folder_path` as soon as that dataset's Bronze Lambda finishes. Consequence: **Silver branches can start staggered** — a dataset whose Gate 1 passes early does not wait for the slowest Bronze dataset to also finish; its own Silver EMR step can launch immediately. The only place that still waits for *all three* is Gate 2 (§6), which needs all three Silver results before releasing Gold. |
| O-9 | Who writes `gold-release/run_date=.../release.json`? | **Automatic, with a manual override.** The default path is an automated step (e.g. a small Lambda subscribed to the trigger-control-table writes, or a Step Functions "wait for all 3 Silver branches" state) that writes `release.json` itself the moment all enabled Silver datasets show `succeeded` for `run_date` — no person needs to act for the normal daily run. The **manual override** is an operator writing (or re-writing) the same `release.json` object by hand — e.g. to force a release despite one dataset in a non-`succeeded` state, to redo Gold with different `config_overlays`/`--set` overrides, or to release a repair run. Both paths go through the identical EventBridge → gate Lambda flow in §12, so the Gold side of the system cannot tell the difference and does not need to. |
| O-10 | Does the one-retry policy apply to Silver too? | **Yes — Silver EMR steps get the same one-automatic-retry-then-escalate policy as Gold.** A Silver retry is cheap for the same underlying reason as Gold: `--run-id` uniqueness is enforced at the artifact-write level (`mode("errorifexists")`, §5), so a genuine retry needs a **new `--run-id`** for that attempt — unlike Gold, a Silver retry cannot resume the same run-id, it re-runs the dataset's transform from scratch with a fresh ID. If the retry also fails (or returns exit code 2 `blocked` again), stop and escalate to `silver-pipeline-alerts` rather than retrying further. |

---

## 10. Facts verified against code

Recorded so later edits do not drift from the implementation.

| Claim | Source |
|---|---|
| `WARN` is an approving status | `bronze_schema_validator.py` — `allowed = passed + warnings` |
| All files registered `PENDING` before validation | `bronze_schema_validator.py` — `merge_pending_rows` before the thread pool |
| `file_key` is stable → re-validation updates in place | `stable_file_key()` + `MERGE ... WHEN MATCHED THEN UPDATE` |
| Silver datasets are exactly three | `silver_pipeline/config.py` — `DATASETS` |
| Lambda supports four datasets | `bronze_schema_validator.py` — `SUPPORTED_DATASETS` |
| Silver exit codes 0/1/2 | `runner.py` — `execute()` return values |
| Gold stages are five, in this order | `lineage/execution.py` — `STAGES` |
| Gold output tables are five | `lineage/execution.py` — `TABLES` |
| Same `run_id` required across stages | `staged_job.py` — `run()` guard |
| Step concurrency must be 1 | `execution.py` `RunStore` docstring; `staged_job.py` `_reservation` |
| Stage N requires stage N−1 manifest | `staged_job.py` — `store.require(STAGES[index - 1])` |
| Failed stage stays retryable with same id | `staged_job.py` — `except` block comment, reservation not closed |
| Only `publish` writes Gold | `staged_job.py` module docstring |
| Config/code frozen at `prepare` | `staged_job.py` — `_initialize` sha comparison |

---

## 12. Full orchestration plan (v2 — with EventBridge)

> **Revised (deployment attempt, real `terraform apply` errors):** two real bugs surfaced on the
> first apply, both fixed in code (§12.1):
>
> 1. **Gate 1's Athena query can no longer be built inline as a Step Functions intrinsic-function
>    string.** `States.Format()` was used to assemble the Gate 1 SQL, which needs single quotes both
>    as SQL delimiters (`'PASS'`/`'WARN'`/`'BLOCK'`) and around the two runtime values. Amazon States
>    Language rejected it outright: `SCHEMA_VALIDATION_FAILED: QueryString.$ must be a valid
>    JSONPath or a valid intrinsic function call`. Fixed by moving the query into a new `gate_one`
>    action on the control Lambda (plain Python f-string SQL, the same pattern `record`/`evaluate`
>    already use) — the per-dataset Silver state machine now calls that Lambda instead of Athena
>    directly. See §12.2's diagram, updated accordingly.
> 2. **EventBridge can no longer target the control Lambda for Gate 2's release trigger.** Granting
>    that (`aws_lambda_permission`) calls `lambda:AddPermission`, which this account's CI/CD role
>    (`gha-lotus-data-sandbox-infra`) carries an **explicit deny** on — not something fixable from
>    this repo's Terraform. Fixed by having EventBridge target **the Gold state machine directly**
>    (a `states:StartExecution` grant, an ordinary identity-based permission, not a Lambda
>    resource policy). The logic that used to run in the Lambda *before* calling `StartExecution`
>    (read `release.json`, re-check the control table, mint `gold_run_id`) now runs as the Gold
>    state machine's own **first state**, `ResolveRelease` — see the revised diagram below.
>
> Both fixes are implementation detail, not decisions — nothing in §4–§9's resolved decisions
> changed. The diagram and §12.1 below reflect the corrected design.

End-to-end architecture, incorporating all decisions from §4/§6/§7.1/§8/§9.

```
 (1) BRONZE — per dataset, per dated folder
     ┌────────────────────────────────────────────────────────────┐
     │ S3 landing: s3://.../bronze/<dataset>/ingest_date=YYYY-MM-DD/│
     └───────────────────────────┬────────────────────────────────┘
                                  │ folder-complete signal (existing trigger,
                                  │ out of scope for this doc)
                                  ▼
     ┌────────────────────────────────────────────────────────────┐
     │ Lambda: bronze_schema_validator (one invocation per dataset)│
     │   → control.bronze_schema_validation (Athena/Glue table)   │
     │   → SNS "bronze-validation-alerts" if overall_status!=PASS │
     └───────────────────────────┬────────────────────────────────┘
                                  ▼
 (2) GATE 1 — checked PER DATASET, independently, as each Bronze Lambda finishes
     ┌────────────────────────────────────────────────────────────┐
     │ For EACH dataset independently (not a joint wait — §9 O-8): │
     │   as soon as that dataset's Bronze Lambda finishes,         │
     │   run the §4 per-dataset Athena query with that dataset's   │
     │   own folder_path for run_date                               │
     │   → dataset_gate_status = PASS | BLOCK                      │
     └───────────────────────────┬────────────────────────────────┘
      this dataset PASS │                 this dataset BLOCK
                        ▼                        ▼
 (3) SILVER — this dataset's EMR      Fail state (this dataset only)
     step starts immediately,          → SNS "bronze-validation-alerts"
     STAGGERED vs. the other two        (retry Gate 1 check for this
     (config-driven enabled list        dataset on the next attempt;
      — §5, §9 O-7)                     the other two datasets are
     ┌──────────────────────────────┐   unaffected)
     │ EMR step: silver_pipeline     │
     │  --dataset <name> --mode publish
     │  --run-id <fresh id, per attempt>
     │ Retry: MaxAttempts 1 (§9 O-10 —│
     │ decided: same policy as Gold,  │
     │ but with a FRESH --run-id, not │
     │ the same one — §5)             │
     │ On success → write trigger     │
     │  control row status=succeeded  │
     │ On failure/blocked after retry │
     │  (exit 1/2) → status=failed|   │
     │  blocked → SNS                 │
     │  "silver-pipeline-alerts"      │
     └──────────────┬────────────────┘
                     │ Gate 2 waits for ALL THREE branches to settle,
                     │ even though they started staggered
                     ▼
 (4) GATE 2 — automatic by default, manual override available (§9 O-9)
     ┌────────────────────────────────────────────────────────────┐
     │ Automated step (e.g. a Lambda subscribed to trigger-control-│
     │ table writes) checks: all enabled Silver datasets show      │
     │ 'succeeded' for run_date → writes release.json itself.      │
     │ No person needs to act for the normal daily run.            │
     │                                                              │
     │ Manual override: an operator may write/overwrite the same   │
     │ object to force a release, change config, or run a repair.  │
     │   s3://<bucket>/gold-release/run_date=.../release.json      │
     │                                                              │
     │ S3 bucket has "EventBridge notifications" enabled.           │
     │ EventBridge rule:                                            │
     │   source: aws.s3, detail-type: "Object Created"              │
     │   filter: bucket == <bucket>, key prefix "gold-release/"     │
     │   target: the Gold state machine itself (§12.1 REVISED —      │
     │           not a Lambda; see below)                            │
     └───────────────────────────┬────────────────────────────────┘
                                  ▼
 (5) GOLD — Step Functions state machine, 6 states,
     concurrency 1, SAME run_id threaded through every stage
     ┌────────────────────────────────────────────────────────────┐
     │ ResolveRelease → prepare → accounts → customers → validate  │
     │                          → publish                          │
     │ ResolveRelease = one lambda:invoke of the control Lambda's   │
     │   release_gate action: reads release.json, re-checks the     │
     │   control table, mints gold_run_id (gold-YYYYMMDD-HHMMSS),    │
     │   returns it - does NOT call StartExecution itself, because  │
     │   this execution already exists by the time it runs. Its     │
     │   ResultSelector merges run_id/run_date/etc. onto the         │
     │   execution's own root, so every stage below still reads      │
     │   $.run_id exactly as before.                                 │
     │ each state = one EMR step:                                  │
     │   staged_job --stage <name> --run-id $.run_id                │
     │               --config <from release.json config_overlays>  │
     │ each state: Retry MaxAttempts 1, IntervalSeconds 30 (§7)     │
     │ Catch → Fail state → SNS "gold-pipeline-alerts"               │
     │ (reservation stays open in gold_run_control; a human re-runs │
     │  the same execution's failed state after investigating)      │
     └────────────────────────────────────────────────────────────┘
```

### Why EventBridge here specifically (vs. S3 → Lambda direct trigger)

S3 can invoke a Lambda directly on `ObjectCreated` without EventBridge. The reasons to still route
through EventBridge, given you already said you want it:

- **Decouples the trigger from the handler.** The rule can fan out to more than one target later
  (e.g. also log the release event to a monitoring bus) without touching the S3 bucket's own
  notification config again.
- **Filtering is declarative** (event pattern on `detail.object.key` prefix/suffix) rather than
  living inside Lambda code, so "which prefix means a Gold release" is visible in the rule, not
  buried in a handler.
- **Retry/DLQ behavvior on the rule target** is configured the same way as any other EventBridge
  target, consistent with how the rest of this plan already uses Step Functions retries.

All of §9's questions are now resolved — see §9 "Resolved" for the decisions this diagram reflects.

---

## 12.1 What has been built

Terraform and Lambda code implementing the above. Both roots pass
`terraform validate`. Nothing has been applied.

| Component | File |
|---|---|
| Handoff control table (Iceberg), **in the Silver bucket's `control/` folder** | `accounts/data-sandbox/infra/silver_gold_control_iceberg.tf` |
| Its locals | `accounts/data-sandbox/infra/silver_gold_control_data.tf` |
| Lake Formation grants | `accounts/data-sandbox/infra/silver_gold_control_lakeformation.tf` |
| Control Lambda + EventBridge rule (Gate 2 release trigger, **targets the Gold state machine directly** — see the revision note above) | `accounts/data-sandbox/infra/silver_gold_control_lambda.tf` |
| Per-layer SNS topics | `accounts/data-sandbox/infra/sns_pipeline_alerts.tf` |
| **Per-dataset** Gate 1 → Silver → Gate 2-check state machine (**Gate 1 now calls the control Lambda's `gate_one` action**, not inline Athena) | `accounts/data-sandbox/infra/stepfunctions_bronze_silver.tf` |
| Gold state machine — **`ResolveRelease` + 5 stages** (`ResolveRelease` resolves the release object and mints `run_id`; see revision note) | `accounts/data-sandbox/infra/stepfunctions_gold.tf` |
| Step Functions role lookups, incl. **`eventbridge_gold_trigger`** (EventBridge's `states:StartExecution` role) | `accounts/data-sandbox/infra/stepfunctions_data.tf` |
| Control Lambda IAM role + policy (no longer needs `states:StartExecution` — it never starts Gold itself now) | `accounts/data-sandbox/security/silver_gold_control.tf` |
| Control Lambda LF data access | `accounts/data-sandbox/security/silver_gold_control_lakeformation.tf` |
| Step Functions IAM roles, incl. Bronze Lambda's `states:StartExecution`, `sfn_gold`'s `lambda:InvokeFunction` on the control Lambda (for `ResolveRelease`), and the new **`eventbridge_gold_trigger`** role/policy | `accounts/data-sandbox/security/stepfunctions_iam.tf`, `security/bronze_schema_validator.tf` |
| Control Lambda source — **`gate_one` action added; `release_gate` no longer calls `StartExecution`, returns resolved fields instead** | `project_lotus/lambdas/silver_gold_control/silver_gold_control.py` |
| Bronze validator Lambda — **now also starts Silver** on a PASS | `project_lotus/lambdas/bronze_schema_validator/bronze_schema_validator.py` |

Both the control table and the Gate 2 `gold-release/` release object live in the **Silver bucket**
(`var.silver_bucket_name`), under `control/`, not in the Bronze landing bucket — the earlier draft
had put them in the Bronze bucket by convenience-copying from the Bronze validator's own table;
that was wrong because this table describes the Silver→Gold boundary, not Bronze validation.

Apply order is unchanged from the existing convention: **`security/` first**, then
`infra/`, because every `aws_iam_role` lives in `security/` (so the permissions boundary is
applied in one place) and `infra/` looks the roles up by name.

### Blocking prerequisites — these are not yet satisfied

Both state machines reference build artifacts that **do not exist yet**. The Terraform is valid
and will apply, but the EMR steps will fail at submit time until these are produced:

1. **`lineage_staged_job.py` is not built by the Gold packaging script.**
   `project_lotus/project_lotus/gold/scripts/package.sh` generates entrypoints only for
   `lineage`, `imei_map`, `ml-discovery`, `ml-selection`, `ml-sweep`. The five-stage staged job
   (`trust_score_05.lineage.jobs.staged_job:main_cli`) has no entrypoint, so
   `${gold_artifact_root}/lineage_staged_job.py` is a dangling reference. Adding
   `[lineage_staged]=trust_score_05.lineage.jobs.staged_job` to that script's `JOB_MODULES` map
   is the fix.

2. **There is no Silver packaging script.** The Silver state machine expects
   `silver_pipeline.zip`, `pydeps.zip` and `silver_job.py` under
   `var.silver_gold_state_machine_artifact_root`. Gold has `scripts/package.sh`; Silver has no
   equivalent, so this artifact set has to be produced before the Silver steps can run.

3. **Athena workgroup `primary` must have a result location**, or `ATHENA_OUTPUT` must stay set.
   The control Lambda passes `ATHENA_OUTPUT` explicitly, so this is satisfied — noted because
   the Gate 1 states in the Silver state machine rely on the same configuration.

### Deferred / not built

- **The `record` calls assume the EMR step's exit code maps cleanly to success/failure.** Step
  Functions' `addStep.sync` fails the state on a non-zero step, so exit 1 and exit 2 both land in
  the `RecordFailed` path and are recorded as `failed`. Distinguishing exit 2 as `blocked` (the
  §9 O-1 decision) requires reading the Silver `summary.json` — the Lambda accepts a `blocked`
  status and the table stores it, but the state machine does not currently tell them apart.
- **No schedule triggers the Bronze validator Lambda itself.** Whatever currently invokes it per
  dataset (an existing schedule, an S3 folder-complete signal — out of scope for this doc) is
  unchanged. What *is* now wired is everything downstream of that: a `PASS` from the Bronze
  validator starts Silver directly (§12.2) — no separate schedule is needed for that hop anymore.

---

## 12.2 Corrected trigger walkthrough (per dataset, no fan-out)

**This supersedes the bundled-Map description that was here originally.** The fix: each dataset's
own Bronze `PASS` starts its own Silver run immediately, with no waiting on any other dataset — and
Gate 2 is checked fresh after every single successful Silver run, so whichever dataset happens to
finish last is the one that actually releases Gold. No dataset needs to know how many other
datasets exist.

```
For ONE dataset (e.g. device_lookup_batch), on its own clock:

  Bronze data lands for this dataset, this run_date
              │
              ▼
  bronze_schema_validator Lambda runs (existing, unchanged trigger)
    - validates every Parquet file's schema
    - writes per-file rows to bronze_schema_validation (Athena/Iceberg)
    - computes overall_status = PASS | BLOCK
              │
       ┌──────┴──────┐
    BLOCK           PASS
       │               │
       ▼               ▼
  SNS alert       NEW: calls stepfunctions:StartExecution
  (existing,      on the per-dataset state machine, name=
  unchanged)      "silver-{dataset}-{run_date}" (dedupes a
                  re-validation of the same folder), input =
                  {run_date, dataset_name, folder_path,
                   table, quarantine_table, quarantine_table_location}
                       │
                       ▼
         ┌─────────────────────────────────────────┐
         │ Step Functions execution (ONE dataset)   │
         │                                           │
         │ 1. Gate 1 query, re-read from the         │
         │    control table (defence in depth)       │
         │       BLOCK → record status=blocked, STOP │
         │       PASS  → continue                    │
         │                                           │
         │ 2. Silver EMR step for THIS dataset only  │
         │    (1 automatic retry, fresh --run-id)     │
         │       fails  → record status=failed, STOP │
         │       succeeds → continue                 │
         │                                           │
         │ 3. record status=succeeded to the         │
         │    Silver→Gold control table              │
         │                                           │
         │ 4. evaluate: re-read ALL enabled datasets' │
         │    rows for this run_date, right now       │
         │       any NOT succeeded → do nothing, STOP │
         │       ALL succeeded      → write           │
         │         gold-release/run_date=.../release.json
         └─────────────────────────────────────────┘
                       │
                       ▼ (only on the execution that completed the set)
         S3 write → EventBridge → gate Lambda → Gold starts
```

**Why this answers "what if account_changes_batch and device_lookup_batch run at different times
of day":** they do — that's normal, not a race condition to design around. Each dataset's execution
above is entirely self-contained; step 4 (`evaluate`) is the *only* place any dataset's run looks at
what any other dataset has done, and it does so by reading current, already-committed state from the
control table, not by waiting on a signal from another execution. Concretely:

- `account_changes_batch` finishes Silver at 03:00. Its execution calls `evaluate`. The control
  table shows `device_lookup_batch` and `audit_trail_services_3` not yet `succeeded` for today.
  Nothing is released. The execution ends normally (this is not a failure — see `NotReleased`/Succeed
  semantics carried over from the original design).
- `audit_trail_services_3` finishes at 07:00. Same check, same "not yet complete" result.
- `device_lookup_batch` finishes last, at 11:00. Its `evaluate` call is the one that finds all three
  rows now `succeeded`, and it is the one that writes `release.json`. Gold starts from *that*
  execution's action — not from `device_lookup_batch`'s execution "being the last one to run" in any
  special-cased sense, simply because its check happened to be the one where the condition first
  became true.

If the arrival order had been reversed — `device_lookup_batch` first, `account_changes_batch`
last — nothing above changes. Whichever execution's `evaluate` call observes a complete set is the
one that releases Gold. There is no ordering assumption anywhere in this design.

**Modularity — adding a fourth dataset:** touches exactly two config points, no new states, no new
Step Functions logic:

1. `local.silver_gold_enabled_datasets` (`silver_gold_control_data.tf`) — add the name. This both
   adds it to Gate 2's required set (`evaluate` now requires 4/4, not 3/3) and adds it to
   `DATASET_SPECS_JSON`, which the Bronze Lambda uses to decide it has a Silver job to start.
2. `SUPPORTED_DATASETS` in `bronze_schema_validator.py` — add the name, so the Bronze validator
   accepts it at all.

The per-dataset state machine's own definition does not change, because it already takes the
dataset's table/quarantine locations as **execution input** rather than hardcoding a per-dataset
branch — the same generic five states run for whichever dataset started that execution.

---

## 13. Change log

| Date | Change |
|---|---|
| 2026-09-28 | Initial draft from Lambda source, `silver_pipeline (4).zip`, `gold_pipeline (4).zip` |
| 2026-09-28 | Incorporated decisions: separate SNS topics, exit-2-alerts, S3+EventBridge Gold release, corrected Gate 1 query (per-dataset folder_path fix), gold_run_id generation point, one-retry policy, tu_portps/mno_activation deferred with config-driven fan-out note. Added §12 full EventBridge architecture. |
| 2026-09-28 | Built the control table, control Lambda, SNS topics, EventBridge release trigger, and both state machines (§12.1). Both Terraform roots validate; nothing applied. Recorded two blocking build-artifact prerequisites. |
| 2026-09-28 | Resolved all remaining open questions: `gold_run_id` = date+timestamp (`gold-YYYYMMDD-HHMMSS`), generated once by the gate Lambda; Gate 1 confirmed per-dataset (Silver starts staggered); Gate 2 release is automatic-by-default with manual override; one-retry policy extended to Silver (with a fresh `--run-id` per Silver retry, unlike Gold's in-place stage retry). §5, §6, §7.1, §9, §12 updated accordingly. |
| 2026-09-29 | **Corrected a real design flaw**: the Bronze→Silver state machine bundled all three datasets into one Map/Parallel execution, which silently assumed they land and validate together. They don't — each dataset's Bronze arrival and validation happens independently, sometimes hours apart. Replaced with a per-dataset state machine, started once per dataset by the Bronze validator Lambda the instant that dataset's own validation passes; Gate 2 (`evaluate`) now runs after every successful Silver run, so whichever dataset finishes last is the one that releases Gold, regardless of arrival order. Also moved the control table and release object from the Bronze bucket to the Silver bucket's `control/` folder (§12.1). See §12.2 for the full walkthrough. |
| 2026-09-29 | **Fixed two real `terraform apply` failures from the first deployment attempt.** (1) `SCHEMA_VALIDATION_FAILED: QueryString.$ must be a valid JSONPath or a valid intrinsic function call` — Gate 1's Athena SQL was built inline via a Step Functions `States.Format()` intrinsic, which cannot reliably handle the single quotes the SQL needed; moved the query into a new `gate_one` action on the control Lambda (plain Python, same pattern as `record`/`evaluate`). (2) `AccessDeniedException: ... not authorized to perform: lambda:AddPermission ... explicit deny` on the CI/CD role applying this Terraform — granting EventBridge permission to invoke the control Lambda directly is blocked at the account level and not fixable from this repo; reworked Gate 2's trigger so **EventBridge targets the Gold state machine directly** (`states:StartExecution`, an ordinary identity-based grant, not a Lambda resource policy) via a new `eventbridge_gold_trigger` role, and moved the release-resolution logic (read `release.json`, re-check the control table, mint `gold_run_id`) into the Gold state machine's own new first state, `ResolveRelease`. Both roots re-validated clean after the fix. See the revision note at the top of §12 and the updated §12.1 table. |
