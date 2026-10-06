"""Silver -> Gold handoff control.

One Lambda, six actions, because most of them read or write the same control
table and splitting them would mean separate roles and Lake Formation grants
for no isolation benefit. `build_gold_args` and `build_silver_args` are the
exception - they touch no control-table state at all - but they share this
Lambda's S3 read permissions and its existing IAM role rather than standing up
a second function.

    record          Called once per Silver dataset, after that dataset's EMR
                    step settles (success, failure, or blocked). Upserts one
                    row into the Silver->Gold control table.

    evaluate        Called after ALL Silver branches settle. If every enabled
                    dataset for the run date is 'succeeded', writes the Gold
                    release object. This is the automatic half of Gate 2.

    gate_one        Called once per dataset, from inside the per-dataset
                    Silver state machine, before that dataset's EMR step runs.
                    Re-derives Gate 1's PASS/BLOCK verdict from the control
                    table. Implemented here rather than as an inline Step
                    Functions intrinsic-function SQL string: Amazon States
                    Language's `States.Format()` has no reliable escape
                    sequence for a literal single quote, and this query needs
                    several - both as SQL string delimiters around
                    PASS/WARN/BLOCK and around the two runtime values.
                    Building it here reuses the same run_athena()/
                    athena_rows() plumbing every other action uses, with
                    Python's own (tested) SQL escaping.

    release_gate    Called by the FIRST STATE of the Gold state machine, which
                    EventBridge starts directly (as an `states:StartExecution`
                    target, not a Lambda target - see the Terraform for why:
                    routing through a Lambda here would need
                    `lambda:AddPermission`, which this account's CI role is
                    explicitly denied). Re-checks the control table and mints
                    the Gold run id; does NOT start anything itself; returns
                    the resolved fields for the state machine's own
                    ResultSelector to merge onto the root of its execution
                    state. Runs for both the automatic release and an
                    operator's manual override, which is why the re-check is
                    here and not in `evaluate`.

    build_gold_args   Called once per Gold stage, immediately before that
                      stage's EMR step. Reads the Gold job config object from
                      S3 and returns the full spark-submit argument list for
                      that one stage. This is what makes the config
                      console-editable: Amazon States Language has no
                      intrinsic that flattens an arbitrary-length list of
                      `--conf` entries into repeated flags, so building the
                      final argument list has to happen in code, not in the
                      state machine definition. Editing the S3 object and
                      re-running the state machine picks up the change with
                      no Terraform apply and no Lambda redeploy - see
                      accounts/data-sandbox/infra/gold_job_config.tf, which
                      seeds it once from Terraform-computed content and then
                      leaves it alone (see that resource's
                      `lifecycle.ignore_changes`).

    build_silver_args Called once per attempt, immediately before that
                      attempt's Silver EMR step - "attempt", not "stage",
                      because unlike Gold, a failed Silver run's retry needs a
                      genuinely fresh --run-id (Silver's artifact write uses
                      `mode("errorifexists")` and refuses to reuse one), so
                      the per-dataset state machine calls this action twice
                      on separate named states (BuildSilverArgs_1,
                      BuildSilverArgs_2) rather than relying on Amazon States
                      Language's own Retry, which would resubmit the exact
                      same already-resolved arguments - including the same
                      run-id - on every retry of a single Task state. Reads
                      the Silver job config object from S3 (see
                      accounts/data-sandbox/infra/silver_job_config.tf) the
                      same console-editable way build_gold_args does, and
                      mints the run-id itself - see its docstring for the
                      format and why it is safe to compute without
                      coordinating with any other execution.

Why the control table is re-read in `release_gate` rather than trusting the
release object: the object is writable by an operator, and an operator forcing
a release is a legitimate workflow (see the orchestration doc, Gate 2). The
re-check turns "somebody wrote a file" into a recorded, queryable decision -
it does not veto the override, it records that the override happened against a
known control state.
"""

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError


CONTROL_DATABASE = os.environ["CONTROL_DATABASE"]
CONTROL_TABLE = os.environ["CONTROL_TABLE"]
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
ATHENA_OUTPUT = os.environ.get("ATHENA_OUTPUT", "")

# The Bronze schema-validation control table Gate 1 reads. A different table in
# the same CONTROL_DATABASE, never written by this Lambda.
BRONZE_VALIDATION_TABLE = os.environ.get(
    "BRONZE_VALIDATION_TABLE", "bronze_schema_validation"
)

RELEASE_BUCKET = os.environ["RELEASE_BUCKET"]
RELEASE_PREFIX = os.environ.get("RELEASE_PREFIX", "gold-release").strip("/")

# s3://bucket/key of the Gold and Silver job config objects build_gold_args /
# build_silver_args read. Neither is read at import time - both are
# intentionally re-fetched on every call, so a console edit takes effect on
# the very next stage or dataset run this Lambda is asked to build arguments
# for, with no redeploy of this function.
GOLD_JOB_CONFIG_URI = os.environ.get("GOLD_JOB_CONFIG_URI", "")
SILVER_JOB_CONFIG_URI = os.environ.get("SILVER_JOB_CONFIG_URI", "")
CLAIM_BUCKET = os.environ.get("ORCHESTRATION_CLAIM_BUCKET", "")
CLAIM_PREFIX = os.environ.get("ORCHESTRATION_CLAIM_PREFIX", "control/pipeline-claims").strip("/")

SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")

# The datasets whose success is required before Gold may run. Config-driven so
# that adding tu_portps / mno_activation later is an environment change rather
# than a code change - see the orchestration doc, O-7.
ENABLED_DATASETS = [
    name.strip()
    for name in os.environ.get(
        "ENABLED_DATASETS",
        "account_changes_batch,device_lookup_batch",
    ).split(",")
    if name.strip()
]

TERMINAL_STATUSES = {"succeeded", "failed", "blocked"}

s3 = boto3.client("s3")
athena = boto3.client("athena")
sns = boto3.client("sns")


def sql_str(value):
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def sql_int(value):
    if value is None:
        return "NULL"
    return str(int(value))


def control_key(run_date: str, dataset_name: str) -> str:
    """Stable per (run_date, dataset). A retry updates its row, never adds one.

    This mirrors `file_key` in the Bronze validator for the same reason: the
    control table answers "what is the current state of this dataset for this
    date", and a retry that appended instead of updating would make that
    question return two answers.
    """
    raw = f"{run_date}|{dataset_name}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def release_key(run_date: str) -> str:
    return f"{RELEASE_PREFIX}/run_date={run_date}/release.json"


def run_athena(sql: str) -> str:
    request = {
        "QueryString": sql,
        "QueryExecutionContext": {"Database": CONTROL_DATABASE},
        "WorkGroup": ATHENA_WORKGROUP,
    }
    if ATHENA_OUTPUT:
        request["ResultConfiguration"] = {"OutputLocation": ATHENA_OUTPUT}

    query_id = athena.start_query_execution(**request)["QueryExecutionId"]

    while True:
        execution = athena.get_query_execution(QueryExecutionId=query_id)
        state = execution["QueryExecution"]["Status"]["State"]
        if state == "SUCCEEDED":
            return query_id
        if state in ("FAILED", "CANCELLED"):
            reason = execution["QueryExecution"]["Status"].get(
                "StateChangeReason", "Unknown Athena error"
            )
            raise RuntimeError(f"Athena query {query_id} {state}: {reason}")
        time.sleep(0.8)


def athena_rows(query_id: str):
    """Return result rows as dicts. Control-table reads are a handful of rows."""
    result = athena.get_query_results(QueryExecutionId=query_id)
    rows = result["ResultSet"]["Rows"]
    if not rows:
        return []
    header = [col.get("VarCharValue") for col in rows[0]["Data"]]
    out = []
    for row in rows[1:]:
        values = [cell.get("VarCharValue") for cell in row["Data"]]
        out.append(dict(zip(header, values)))
    return out


def publish_alert(subject: str, payload: dict):
    if not SNS_TOPIC_ARN:
        return
    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject=subject[:100],
        Message=json.dumps(payload, indent=2, default=str),
    )


def _orchestration_input(event):
    run_date = event["run_date"]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", run_date):
        raise ValueError("run_date must be YYYY-MM-DD")
    return run_date


def _cluster_key(run_date):
    return f"{CLAIM_PREFIX}/cluster/{run_date}.json"


def _read_cluster(run_date):
    try:
        response = s3.get_object(Bucket=CLAIM_BUCKET, Key=_cluster_key(run_date))
        return json.loads(response["Body"].read()), response.get("ETag", "").strip('"')
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return None, None
        raise


def _put_cluster(run_date, body, **conditions):
    s3.put_object(
        Bucket=CLAIM_BUCKET,
        Key=_cluster_key(run_date),
        Body=json.dumps(body).encode("utf-8"),
        ContentType="application/json",
        **conditions,
    )


def _claim_cluster(run_date, owner):
    if not CLAIM_BUCKET or not owner:
        raise ValueError("claim bucket and execution owner are required")
    body = {
        "run_date": run_date,
        "owner": owner,
        "status": "creating",
        "cluster_id": "",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        _put_cluster(run_date, body, IfNoneMatch="*")
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("PreconditionFailed", "ConditionalRequestConflict", "412", "409"):
            raise
        current, _ = _read_cluster(run_date)
        return bool(
            current
            and current.get("owner") == owner
            and current.get("status") == "creating"
        )


def action_claim_cluster(event):
    run_date = _orchestration_input(event)
    return {"claimed": _claim_cluster(run_date, event["owner"])}


def action_record_cluster(event):
    run_date = _orchestration_input(event)
    cluster_id = event["cluster_id"]
    owner = event["owner"]
    if not re.fullmatch(r"j-[A-Z0-9]+", cluster_id):
        raise ValueError("Invalid EMR cluster ID")
    current, _ = _read_cluster(run_date)
    if not current or current.get("owner") != owner:
        raise ValueError("Only the claiming execution can register this cluster")
    current.update(
        status="ready",
        cluster_id=cluster_id,
        updated_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    _put_cluster(run_date, current)
    return {"status": "ready", "cluster_id": cluster_id}


def action_get_cluster(event):
    run_date = _orchestration_input(event)
    current, _ = _read_cluster(run_date)
    if not current:
        return {"status": "missing", "cluster_id": ""}
    return {
        "status": current.get("status", "creating"),
        "cluster_id": current.get("cluster_id", ""),
    }


def action_release_cluster_claim(event):
    run_date = _orchestration_input(event)
    current, _ = _read_cluster(run_date)
    if not current:
        return {"released": False}
    if current.get("owner") != event["owner"]:
        raise ValueError("Only the claiming execution can release this cluster claim")
    s3.delete_object(Bucket=CLAIM_BUCKET, Key=_cluster_key(run_date))
    return {"released": True}


def action_close_cluster(event):
    run_date = _orchestration_input(event)
    current, _ = _read_cluster(run_date)
    if not current:
        return {"status": "missing"}
    current.update(
        status="terminated",
        updated_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    _put_cluster(run_date, current)
    return {"status": "terminated"}


# ---------------------------------------------------------------------------
# action: record
# ---------------------------------------------------------------------------


def action_record(event):
    """Upsert one dataset's Silver outcome for one run date.

    Event:
      {
        "action": "record",
        "run_date": "2026-09-11",
        "dataset_name": "device_lookup_batch",
        "silver_run_id": "silver-device_lookup_batch-20260911-01",
        "status": "succeeded" | "failed" | "blocked",
        "exit_code": 0,
        "attempt_number": 1,
        "rows_published": 12345,          optional
        "bronze_folder_path": "s3://...", optional
        "summary_uri": "s3://...",        optional
        "error_message": "..."            optional
      }
    """
    run_date = event["run_date"]
    dataset_name = event["dataset_name"]
    status = str(event["status"]).lower()

    if dataset_name not in ENABLED_DATASETS:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}; expected one of {ENABLED_DATASETS}"
        )
    if status not in TERMINAL_STATUSES:
        raise ValueError(
            f"status must be one of {sorted(TERMINAL_STATUSES)}, got {status!r}"
        )

    key = control_key(run_date, dataset_name)
    recorded_ts = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")

    sql = f'''\
MERGE INTO "{CONTROL_DATABASE}"."{CONTROL_TABLE}" t
USING (
    VALUES (
        {sql_str(key)},
        CAST({sql_str(run_date)} AS date),
        {sql_str(dataset_name)},
        {sql_str(event.get("silver_run_id"))},
        {sql_str(status)},
        {sql_int(event.get("exit_code"))},
        {sql_int(event.get("attempt_number", 1))},
        {sql_int(event.get("rows_published"))},
        {sql_str(event.get("bronze_folder_path"))},
        {sql_str(event.get("summary_uri"))},
        {sql_str((event.get("error_message") or "")[:4000] or None)},
        CAST({sql_str(recorded_ts)} AS timestamp)
    )
) AS s(
    control_key, run_date, dataset_name, silver_run_id, status, exit_code,
    attempt_number, rows_published, bronze_folder_path, summary_uri,
    error_message, recorded_ts
)
ON t.control_key = s.control_key
WHEN MATCHED THEN UPDATE SET
    run_date = s.run_date,
    dataset_name = s.dataset_name,
    silver_run_id = s.silver_run_id,
    status = s.status,
    exit_code = s.exit_code,
    attempt_number = s.attempt_number,
    rows_published = s.rows_published,
    bronze_folder_path = s.bronze_folder_path,
    summary_uri = s.summary_uri,
    error_message = s.error_message,
    recorded_ts = s.recorded_ts
WHEN NOT MATCHED THEN INSERT (
    control_key, run_date, dataset_name, silver_run_id, status, exit_code,
    attempt_number, rows_published, bronze_folder_path, summary_uri,
    error_message, recorded_ts
)
VALUES (
    s.control_key, s.run_date, s.dataset_name, s.silver_run_id, s.status,
    s.exit_code, s.attempt_number, s.rows_published, s.bronze_folder_path,
    s.summary_uri, s.error_message, s.recorded_ts
)
'''
    run_athena(sql)

    if status != "succeeded":
        publish_alert(
            f"Silver {status}: {dataset_name}",
            {
                "layer": "silver",
                "run_date": run_date,
                "dataset_or_stage": dataset_name,
                "run_id": event.get("silver_run_id"),
                "status": status,
                "exit_code": event.get("exit_code"),
                "summary_uri": event.get("summary_uri"),
                "occurred_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    return {"action": "record", "control_key": key, "run_date": run_date,
            "dataset_name": dataset_name, "status": status}


# ---------------------------------------------------------------------------
# action: gate_one
# ---------------------------------------------------------------------------


def action_gate_one(event):
    """PASS/BLOCK for one dataset's latest validation attempt at one folder.

    Event: {"action": "gate_one", "dataset_name": "...", "folder_path": "s3://.../"}

    Mirrors the query in the orchestration doc's Gate 1 section exactly, fixed
    to this dataset's own folder_path rather than one shared literal - a
    single folder path applied to every dataset only ever matches the first
    one, since each dataset lands under its own S3 prefix.
    """
    dataset_name = event["dataset_name"]
    folder_path = event["folder_path"]

    sql = f'''\
WITH latest_attempts AS (
    SELECT dataset_name, attempt_id,
           ROW_NUMBER() OVER (PARTITION BY dataset_name ORDER BY validation_start_ts DESC) AS rn
    FROM "{CONTROL_DATABASE}"."{BRONZE_VALIDATION_TABLE}"
    WHERE dataset_name = {sql_str(dataset_name)}
      AND folder_path = {sql_str(folder_path)}
),
validation AS (
    SELECT v.dataset_name,
           COUNT(*) AS rows_found,
           MAX(v.folder_file_count) AS files_expected,
           SUM(CASE WHEN v.status IN ('PASS','WARN') THEN 1 ELSE 0 END) AS files_allowed,
           SUM(CASE WHEN v.status NOT IN ('PASS','WARN') THEN 1 ELSE 0 END) AS blocking_files
    FROM "{CONTROL_DATABASE}"."{BRONZE_VALIDATION_TABLE}" v
    JOIN latest_attempts l
      ON v.dataset_name = l.dataset_name
     AND v.attempt_id = l.attempt_id
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
        THEN 'PASS' ELSE 'BLOCK'
    END AS dataset_gate_status
FROM validation
'''
    rows = athena_rows(run_athena(sql))
    status = rows[0]["dataset_gate_status"] if rows else "BLOCK"

    return {"action": "gate_one", "dataset_name": dataset_name,
            "folder_path": folder_path, "dataset_gate_status": status}


# ---------------------------------------------------------------------------
# shared: read the control state for one run date
# ---------------------------------------------------------------------------


def read_control_state(run_date: str):
    sql = f'''\
SELECT dataset_name, status, silver_run_id, attempt_number
FROM "{CONTROL_DATABASE}"."{CONTROL_TABLE}"
WHERE run_date = CAST({sql_str(run_date)} AS date)
  AND dataset_name IN ({",".join(sql_str(d) for d in ENABLED_DATASETS)})
'''
    rows = athena_rows(run_athena(sql))
    by_dataset = {row["dataset_name"]: row for row in rows}

    succeeded = [d for d in ENABLED_DATASETS
                 if by_dataset.get(d, {}).get("status") == "succeeded"]
    missing = [d for d in ENABLED_DATASETS if d not in by_dataset]
    not_succeeded = [
        d for d in ENABLED_DATASETS
        if d in by_dataset and by_dataset[d].get("status") != "succeeded"
    ]

    return {
        "run_date": run_date,
        "enabled_datasets": ENABLED_DATASETS,
        "succeeded": succeeded,
        "missing": missing,
        "not_succeeded": not_succeeded,
        "all_succeeded": len(succeeded) == len(ENABLED_DATASETS),
        "detail": by_dataset,
    }


# ---------------------------------------------------------------------------
# action: evaluate
# ---------------------------------------------------------------------------


def action_evaluate(event):
    """Write the Gold release object once every enabled dataset has succeeded.

    Event: {"action": "evaluate", "run_date": "2026-09-11", ...optional overrides}

    The write is conditional (IfNoneMatch). A release object that already exists
    is left exactly as it is: re-writing it would fire the EventBridge rule a
    second time and start a second Gold run for the same date, and the existing
    object may be an operator's override that this automatic path must not
    silently replace.
    """
    run_date = event["run_date"]
    state = read_control_state(run_date)

    if not state["all_succeeded"]:
        return {"action": "evaluate", "released": False,
                "reason": "not all enabled Silver datasets succeeded", **state}

    body = {
        "run_date": run_date,
        "release_gold": True,
        "first_run": bool(event.get("first_run", False)),
        "dry_run": bool(event.get("dry_run", False)),
        "config_overlays": event.get("config_overlays", []),
        "overrides": event.get("overrides", []),
        "released_by": "automatic",
        "silver_run_ids": {
            d: state["detail"].get(d, {}).get("silver_run_id")
            for d in ENABLED_DATASETS
        },
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    key = release_key(run_date)
    try:
        s3.put_object(
            Bucket=RELEASE_BUCKET,
            Key=key,
            Body=json.dumps(body, indent=2).encode("utf-8"),
            ContentType="application/json",
            IfNoneMatch="*",
        )
        created = True
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("PreconditionFailed", "412"):
            raise
        created = False

    return {"action": "evaluate", "released": created,
            "release_uri": f"s3://{RELEASE_BUCKET}/{key}",
            "reason": None if created else "release object already exists",
            **state}


# ---------------------------------------------------------------------------
# action: release_gate
# ---------------------------------------------------------------------------


def _release_object_from_event(event):
    """Accept either an EventBridge S3 notification or a direct invocation."""
    detail = event.get("detail") or {}
    bucket = (detail.get("bucket") or {}).get("name")
    key = (detail.get("object") or {}).get("key")
    if bucket and key:
        return bucket, key
    if event.get("release_uri"):
        without_scheme = event["release_uri"][len("s3://"):]
        bucket, _, key = without_scheme.partition("/")
        return bucket, key
    if event.get("run_date"):
        return RELEASE_BUCKET, release_key(event["run_date"])
    raise ValueError("release_gate needs an S3 event, release_uri, or run_date")


def action_release_gate(event):
    """Verify the release object and mint the Gold run id.

    Called from INSIDE the Gold state machine's own first state - EventBridge
    starts that execution directly (a `states:StartExecution` target), so by
    the time this runs, an execution already exists and is waiting on this
    call's return value to know its own run_id. This function therefore
    returns the resolved fields rather than calling `states:StartExecution`
    itself; the caller's ResultSelector merges them onto the execution's root
    state, which is what every later stage reads `$.run_id` from.
    """
    bucket, key = _release_object_from_event(event)
    body = json.loads(
        s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    )

    run_date = body["run_date"]
    if not body.get("release_gold", False):
        raise RuntimeError(
            f"release.json at s3://{bucket}/{key} has release_gold != true; "
            "this execution should not have been started"
        )

    state = read_control_state(run_date)
    released_by = body.get("released_by", "manual")

    # An override is allowed to proceed against an incomplete control state, but
    # it is recorded and alerted rather than passed through quietly. An
    # automatic release that somehow reaches here without a clean state is a
    # bug, not an override, so it is refused.
    if not state["all_succeeded"]:
        if released_by != "manual":
            raise RuntimeError(
                f"Automatic release for {run_date} but control state is not clean: "
                f"missing={state['missing']} not_succeeded={state['not_succeeded']}"
            )
        publish_alert(
            f"Gold released by manual override: {run_date}",
            {"layer": "gold", "run_date": run_date, "status": "manual_override",
             "missing": state["missing"], "not_succeeded": state["not_succeeded"],
             "release_uri": f"s3://{bucket}/{key}",
             "occurred_at_utc": datetime.now(timezone.utc).isoformat()},
        )

    # Generated exactly once, here. Every one of the five Gold EMR stages is
    # handed this same string as --run-id; the staged job refuses to run if they
    # ever differ, and gold_run_control refuses a reused id, so a collision
    # fails loudly at `prepare` rather than corrupting a run.
    now = datetime.now(timezone.utc)
    gold_run_id = f"gold-{run_date.replace('-', '')}-{now.strftime('%H%M%S')}"

    return {
        "action": "release_gate",
        "run_id": gold_run_id,
        "run_date": run_date,
        "first_run": bool(body.get("first_run", False)),
        "dry_run": bool(body.get("dry_run", False)),
        "config_overlays": body.get("config_overlays", []),
        "overrides": body.get("overrides", []),
        "released_by": released_by,
        "release_uri": f"s3://{bucket}/{key}",
        "control_state_clean": state["all_succeeded"],
    }


# ---------------------------------------------------------------------------
# action: build_gold_args
# ---------------------------------------------------------------------------


def _read_json_s3(uri: str) -> dict:
    without_scheme = uri[len("s3://"):]
    bucket, _, key = without_scheme.partition("/")
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(body.decode("utf-8"))


def action_build_gold_args(event):
    """The full spark-submit argument list for one Gold stage.

    Event: {"action": "build_gold_args", "stage": "prepare", "run_id": "...",
            "first_run": false}

    `--first-run` and `--watermark-from` are appended only for `prepare`, and
    only when `first_run` is true - the staged job reads them once, at
    `prepare`, and freezes them into its own run state; passing them again on
    a later stage would do nothing but would misstate what actually happened
    if anyone reads the EMR step's own argument list back later.
    """
    if not GOLD_JOB_CONFIG_URI:
        raise RuntimeError("GOLD_JOB_CONFIG_URI is not configured")

    stage = event["stage"]
    run_id = event["run_id"]
    first_run = bool(event.get("first_run", False))

    cfg = _read_json_s3(GOLD_JOB_CONFIG_URI)

    args = ["spark-submit"]
    args += cfg["spark_submit_flags"]
    for conf in cfg["spark_confs"]:
        args += ["--conf", conf]
    args += ["--py-files", cfg["python_package"]]
    args += [cfg["application"]]
    args += ["--config", cfg["config"]]
    args += ["--job", cfg["job"]]
    args += ["--run-id", run_id]
    args += ["--staging-root", cfg["staging_root"]]
    args += ["--stage", stage]

    if stage == "prepare" and first_run:
        args += ["--first-run", "--watermark-from", cfg["first_run_watermark_from"]]

    return {"action": "build_gold_args", "stage": stage, "run_id": run_id, "args": args}


# ---------------------------------------------------------------------------
# action: build_silver_args
# ---------------------------------------------------------------------------


def _silver_run_id(run_date: str, dataset_name: str, attempt: int) -> str:
    """``silver-<run_date, compact>-<dataset>-<2-digit attempt>``.

    Three deliberate choices behind this shape:

    - **Date first, compact (YYYYMMDD).** Sorting run ids lexicographically
      then sorts them chronologically - in the EMR console's step list, in S3
      prefixes, in CloudWatch Logs Insights - without parsing the string
      first. A bare run_date already ties it to a specific dataset+day.
    - **The dataset name is spelled out, not abbreviated.** An operator
      grepping logs for "which run was device_lookup_batch's" should not have
      to remember or guess an abbreviation scheme.
    - **Attempt is a small zero-padded integer, not a timestamp or a random
      suffix.** Uniqueness does not need a timestamp or randomness here: Step
      Functions itself already guarantees at most one execution exists for
      this (dataset, run_date) pair - the Bronze Lambda starts this execution
      with a deterministic name and a duplicate is rejected outright, see
      start_silver_execution() - so the only thing that can vary within one
      execution's lifetime is which attempt this is, and there are at most
      two (one automatic retry). A small integer says exactly that and
      nothing more; a timestamp or a UUID would imply a uniqueness concern
      that does not exist at this layer and would make two attempts of the
      same logical run harder to visually associate with each other.
    """
    return f"silver-{run_date.replace('-', '')}-{dataset_name}-{attempt:02d}"


def action_build_silver_args(event):
    """The full spark-submit argument list for one Silver attempt.

    Event: {"action": "build_silver_args", "dataset_name": "...",
            "run_date": "...", "folder_path": "s3://.../", "attempt": 1}

    Mints --run-id itself (see _silver_run_id) rather than receiving one,
    because the state machine calls this action fresh for every attempt - see
    this module's docstring for why Silver cannot reuse Amazon States
    Language's own per-state Retry the way Gold's build_gold_args can be
    called just once per stage. The returned run_id is also what the caller
    should record as `silver_run_id` on the matching `record` call, so the id
    that shows up in the control table is read from here rather than
    reconstructed a second time and risking drifting out of sync with it.
    """
    if not SILVER_JOB_CONFIG_URI:
        raise RuntimeError("SILVER_JOB_CONFIG_URI is not configured")

    dataset_name = event["dataset_name"]
    run_date = event["run_date"]
    folder_path = event["folder_path"]
    attempt = int(event.get("attempt", 1))

    cfg = _read_json_s3(SILVER_JOB_CONFIG_URI)
    dataset_cfg = cfg["datasets"].get(dataset_name)
    if dataset_cfg is None:
        raise ValueError(
            f"No 'datasets' entry for {dataset_name!r} in the Silver job "
            f"config at {SILVER_JOB_CONFIG_URI}"
        )

    run_id = _silver_run_id(run_date, dataset_name, attempt)

    args = ["spark-submit"]
    args += cfg["spark_submit_flags"]
    for conf in cfg.get("spark_confs", []):
        args += ["--conf", conf]
    args += ["--py-files", cfg["python_package"]]
    args += [cfg["application"]]
    args += ["--dataset", dataset_name]
    args += ["--mode", cfg.get("mode", "publish")]
    args += ["--run-date", run_date]
    args += ["--run-id", run_id]
    args += ["--bronze-path", folder_path]
    args += ["--artifact-root", cfg["artifact_root"]]
    args += ["--table", dataset_cfg["table"]]
    args += ["--quarantine-table", dataset_cfg["quarantine_table"]]
    args += ["--quarantine-table-location", dataset_cfg["quarantine_table_location"]]

    return {
        "action": "build_silver_args",
        "dataset_name": dataset_name,
        "run_id": run_id,
        "attempt": attempt,
        "args": args,
    }


# ---------------------------------------------------------------------------


ACTIONS = {
    "claim_cluster": action_claim_cluster,
    "record_cluster": action_record_cluster,
    "get_cluster": action_get_cluster,
    "release_cluster_claim": action_release_cluster_claim,
    "close_cluster": action_close_cluster,
    "record": action_record,
    "build_gold_args": action_build_gold_args,
    "build_silver_args": action_build_silver_args,
    "evaluate": action_evaluate,
    "gate_one": action_gate_one,
    "release_gate": action_release_gate,
}


def lambda_handler(event, context):
    action = event.get("action")
    if action not in ACTIONS:
        raise ValueError(
            f"Unknown action {action!r}; expected one of {sorted(ACTIONS)}"
        )
    return ACTIONS[action](event)
