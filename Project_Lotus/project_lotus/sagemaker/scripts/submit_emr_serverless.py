"""Submit one Trust Score Spark job to an EMR Serverless application.

    python scripts/submit_emr_serverless.py lineage       [--dry-run] [--wait]
    python scripts/submit_emr_serverless.py imei_map
    python scripts/submit_emr_serverless.py ml-discovery
    python scripts/submit_emr_serverless.py ml-selection

The EMR Serverless counterpart of ``scripts/submit_emr.sh``: same artifacts (the
ones ``scripts/package.sh`` uploads), same job names, same config families and
the same two rules that script enforces:

* **Configs travel with the job** as ``--files`` and are passed to the driver by
  basename, so the driver never needs boto3 to read its own config. Two configs
  with the same basename are refused (they would overwrite each other).
* **Iceberg only for the lineage family**, where ``MERGE INTO`` needs it; the
  EMR Serverless image ships the runtime jar at the path below.

What differs from EMR on EC2: there is no YARN and no cluster id. The run id
reaches the driver through ``spark.emr-serverless.driverEnv.*`` rather than
``spark.yarn.appMasterEnv.*``, and capacity is bounded per job with
``spark.dynamicAllocation.maxExecutors`` - that cap *is* the cost ceiling of a
run, since Serverless bills per vCPU-second actually used.

``ml-sweep`` is deliberately absent: the sweep runs on SageMaker (``launch.py``).

Environment (or flags):
    TS05_EMRS_APPLICATION_ID   the EMR Serverless application (Spark, emr-7.x)
    TS05_EMRS_ROLE_ARN         the job execution role
    TS05_ARTIFACT_ROOT         s3://.../ts05/  (what scripts/package.sh uploaded)
    TS05_EMRS_LOG_URI          optional, s3:// prefix for driver/executor logs
    TS05_RUN_ID                optional, forwarded to driver and executors
    AWS_REGION                 default ca-central-1

Default configs are the **sandbox** overlays: ``conf/<family>/base.yaml`` +
``conf/<family>/sandbox.yaml`` under the artifact root. Always check the two
``--config`` lines in a ``--dry-run`` before the first real submit.

To confirm on the first sandbox submit (cannot be tested without AWS): that the
application's image carries the pandas/pyarrow versions ``ml-discovery`` and
``ml-selection`` need on the driver. If it does not, build a venv archive and
pass ``--venv-archive`` (see sagemaker/README-sagemaker.md).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

JOBS = {
    # job name -> (config family, needs Iceberg)
    "lineage": ("lineage", True),
    "imei_map": ("lineage", True),
    "ml-discovery": ("ml", False),
    "ml-selection": ("ml", False),
}
ICEBERG_EXTENSIONS = "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions"
ICEBERG_JAR = "/usr/share/aws/iceberg/lib/iceberg-spark3-runtime.jar"
DEFAULT_REGION = "ca-central-1"
TERMINAL = ("SUCCESS", "FAILED", "CANCELLED")


def _required(value: Optional[str], flag: str, env: str, dry_run: bool) -> str:
    if value:
        return value
    if dry_run:
        return f"<{env}>"
    raise SystemExit(f"{flag} (or {env}) is required")


def config_files(args: argparse.Namespace, family: str, root: str) -> List[str]:
    if args.config_files:
        return [c.strip() for c in args.config_files.split(",") if c.strip()]
    return [f"{root}/conf/{family}/base.yaml", f"{root}/conf/{family}/{args.environment}.yaml"]


def build_request(args: argparse.Namespace) -> Dict[str, Any]:
    family, iceberg = JOBS[args.job]
    root = _required(args.artifact_root, "--artifact-root", "TS05_ARTIFACT_ROOT",
                     args.dry_run).rstrip("/")

    # Configs: s3 ones are localised with --files and named by basename.
    files: List[str] = []
    config_args: List[str] = []
    seen: set = set()
    for cfg in config_files(args, family, root):
        if cfg.startswith(("s3://", "s3a://")):
            name = cfg.rsplit("/", 1)[-1]
            if name in seen:
                raise SystemExit(f"two config files are both named {name}; they would "
                                 "overwrite each other when localised. Rename one.")
            seen.add(name)
            files.append(cfg)
            config_args += ["--config", name]
        else:
            config_args += ["--config", cfg]

    conf: Dict[str, str] = {
        # Same pin as submit_emr.sh and both base.yaml files: EST wall-clock time.
        "spark.sql.session.timeZone": "EST",
        "spark.driver.cores": str(args.driver_cores),
        "spark.driver.memory": args.driver_memory,
        "spark.executor.cores": str(args.executor_cores),
        "spark.executor.memory": args.executor_memory,
        "spark.dynamicAllocation.enabled": "true",
        "spark.dynamicAllocation.maxExecutors": str(args.max_executors),
        "spark.task.maxFailures": "8",
        "spark.stage.maxConsecutiveAttempts": "8",
    }
    if iceberg:
        conf["spark.sql.extensions"] = ICEBERG_EXTENSIONS
        conf["spark.jars"] = ICEBERG_JAR
    run_id = args.run_id or os.environ.get("TS05_RUN_ID")
    if run_id:
        conf["spark.emr-serverless.driverEnv.TS05_RUN_ID"] = run_id
        conf["spark.executorEnv.TS05_RUN_ID"] = run_id
    if args.venv_archive:
        python = "./environment/bin/python"
        conf["spark.archives"] = f"{args.venv_archive}#environment"
        conf["spark.emr-serverless.driverEnv.PYSPARK_DRIVER_PYTHON"] = python
        conf["spark.emr-serverless.driverEnv.PYSPARK_PYTHON"] = python
        conf["spark.executorEnv.PYSPARK_PYTHON"] = python
    for pair in args.spark_conf or []:
        key, _, value = pair.partition("=")
        conf[key.strip()] = value.strip()  # caller's settings win, as in submit_emr.sh

    params = [f"--py-files {root}/trust_score_05.zip,{root}/pydeps.zip"]
    if files:
        params.append("--files " + ",".join(files))
    params += [f"--conf {key}={value}" for key, value in conf.items()]

    extra = list(args.extra or [])
    request: Dict[str, Any] = {
        "applicationId": _required(args.application_id, "--application-id",
                                   "TS05_EMRS_APPLICATION_ID", args.dry_run),
        "executionRoleArn": _required(args.role_arn, "--role-arn", "TS05_EMRS_ROLE_ARN",
                                      args.dry_run),
        "name": f"trust_score_05.{args.job}.{datetime.now(timezone.utc):%Y%m%dT%H%M%S}",
        "jobDriver": {"sparkSubmit": {
            "entryPoint": f"{root}/{args.job}_job.py",
            "entryPointArguments": config_args + extra,
            "sparkSubmitParameters": " ".join(params),
        }},
        "executionTimeoutMinutes": int(args.timeout_minutes),
        "tags": {"project": "trust-score-05", "environment": args.environment,
                 "job": args.job},
    }
    if args.log_uri:
        request["configurationOverrides"] = {"monitoringConfiguration": {
            "s3MonitoringConfiguration": {"logUri": args.log_uri}}}
    return request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("job", choices=sorted(JOBS))
    parser.add_argument("--application-id", default=os.environ.get("TS05_EMRS_APPLICATION_ID"))
    parser.add_argument("--role-arn", default=os.environ.get("TS05_EMRS_ROLE_ARN"))
    parser.add_argument("--artifact-root", default=os.environ.get("TS05_ARTIFACT_ROOT"))
    parser.add_argument("--log-uri", default=os.environ.get("TS05_EMRS_LOG_URI"))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", DEFAULT_REGION))
    parser.add_argument("--environment", default="sandbox", choices=("sandbox", "dev"),
                        help="Which overlay the default configs name.")
    parser.add_argument("--config-files", default=None,
                        help="Comma-separated; overrides the <family>/base + <environment> default.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--venv-archive", default=None,
                        help="s3://.../venv.tar.gz to run the driver/executors in (see README).")
    parser.add_argument("--driver-cores", type=int, default=4)
    parser.add_argument("--driver-memory", default="16g",
                        help="Raise for the first lineage run (--first-run walks all history).")
    parser.add_argument("--executor-cores", type=int, default=4)
    parser.add_argument("--executor-memory", default="16g")
    parser.add_argument("--max-executors", type=int, default=20,
                        help="The cost ceiling of one run.")
    parser.add_argument("--timeout-minutes", type=int, default=360)
    parser.add_argument("--spark-conf", action="append", default=None, metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true", help="Print the request; send nothing.")
    parser.add_argument("--wait", action="store_true",
                        help="Poll until the run ends; exit 1 unless it SUCCEEDED.")
    parser.epilog = ("Arguments after a bare `--` are passed to the job itself, "
                     "e.g. `lineage --dry-run -- --first-run`.")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Our flags before ``--``, the job's own arguments after it.

    Split by hand rather than with ``nargs=REMAINDER``, which captures every
    token after the job name - so ``lineage --run-id X`` would hand ``--run-id``
    to the Spark job instead of reading it here.
    """
    raw = list(sys.argv[1:] if argv is None else argv)
    extra: List[str] = []
    if "--" in raw:
        cut = raw.index("--")
        raw, extra = raw[:cut], raw[cut + 1:]
    args = build_parser().parse_args(raw)
    args.extra = extra
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    request = build_request(args)
    if args.dry_run:
        print(json.dumps(request, indent=2))
        return 0

    import boto3

    client = boto3.client("emr-serverless", region_name=args.region)
    response = client.start_job_run(**request)
    run_id = response["jobRunId"]
    print(f"started {request['name']} as job run {run_id}")
    if not args.wait:
        return 0
    while True:
        run = client.get_job_run(applicationId=request["applicationId"], jobRunId=run_id)["jobRun"]
        state = run["state"]
        if state in TERMINAL:
            break
        time.sleep(30)
    print(f"{run_id}: {state} {run.get('stateDetails', '')}")
    return 0 if state == "SUCCESS" else 1


if __name__ == "__main__":
    sys.exit(main())
