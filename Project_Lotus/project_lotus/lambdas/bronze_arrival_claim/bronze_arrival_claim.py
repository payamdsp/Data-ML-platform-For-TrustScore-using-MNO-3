"""Elect exactly one 15-minute Bronze arrival window per dataset and date."""

import json
import os
import re
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError


CLAIM_BUCKET = os.environ["CLAIM_BUCKET"]
CLAIM_PREFIX = os.environ.get(
    "CLAIM_PREFIX", "control/bronze-arrival-claims"
).strip("/")
ENABLED_DATASETS = {
    value.strip()
    for value in os.environ.get("ENABLED_DATASETS", "").split(",")
    if value.strip()
}

s3 = boto3.client("s3")


def lambda_handler(event, context):
    dataset = event["dataset_name"]
    run_date = event["run_date"]
    folder_path = event["folder_path"]
    owner = event["owner"]

    if dataset not in ENABLED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", run_date):
        raise ValueError("run_date must be YYYY-MM-DD")

    expected_folder = f"s3://{CLAIM_BUCKET}/bronze/{dataset}/{run_date}/"
    if folder_path != expected_folder:
        raise ValueError(
            f"folder_path must be exactly {expected_folder!r}, got {folder_path!r}"
        )
    if not owner:
        raise ValueError("owner is required")

    key = f"{CLAIM_PREFIX}/{dataset}/{run_date}.json"
    body = {
        "dataset_name": dataset,
        "run_date": run_date,
        "folder_path": folder_path,
        "owner": owner,
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    try:
        s3.put_object(
            Bucket=CLAIM_BUCKET,
            Key=key,
            Body=json.dumps(body).encode("utf-8"),
            ContentType="application/json",
            IfNoneMatch="*",
        )
        return {"claimed": True, "claim_uri": f"s3://{CLAIM_BUCKET}/{key}"}
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code not in (
            "PreconditionFailed",
            "ConditionalRequestConflict",
            "412",
            "409",
        ):
            raise

    # If Lambda returned no response after creating the object, Step Functions
    # retries with the same execution ID. Treat that retry as the owner rather
    # than incorrectly routing it to AlreadyScheduled.
    current = json.loads(
        s3.get_object(Bucket=CLAIM_BUCKET, Key=key)["Body"].read()
    )
    return {
        "claimed": current.get("owner") == owner,
        "claim_uri": f"s3://{CLAIM_BUCKET}/{key}",
    }
