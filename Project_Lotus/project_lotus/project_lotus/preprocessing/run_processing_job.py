"""
Launches the SageMaker Processing Job for PySpark preprocessing.

Run this after:
  1. ecr.tf and iam_sagemaker_processing.tf have been applied via the
     normal Terraform PR pipeline (get the ECR repo and role ARN from
     their Terraform outputs).
  2. build_and_push.sh has pushed the container image to that ECR repo.

This script itself just calls the SageMaker API to start the job - it does
not build infrastructure.
"""

import os

import boto3
import sagemaker
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput

# --- Fill these in from Terraform outputs / your setup ---
AWS_ACCOUNT_ID = "162591926854"           # from earlier ARN in this conversation
AWS_REGION = "ca-central-1"
ECR_REPO_NAME = "lotus-sandbox-preprocessing"
IMAGE_TAG = "demo_latest"

EXECUTION_ROLE_ARN = "arn:aws:iam::{}:role/lotus-sandbox-sagemaker-execution".format(
    AWS_ACCOUNT_ID
)  # the shared role from security/iam_sagemaker.tf (local.name_prefix-sagemaker-execution)
# Pulled from that file's output: `sagemaker_execution_role_arn`

# --- Local file to upload, then use as the job's input ---
LOCAL_DATA_PATH = r"C:\Users\aakash.dv\Downloads\telecom_fraud_prediction_100_rows_test_data.csv"
INPUT_BUCKET = "lotus-sandbox-bronze-landing-data"
INPUT_KEY = "telecom_fraud_prediction_100_rows_test_data.csv"

OUTPUT_S3_URI = "s3://lotus-sandbox-gold-curated-data"  # confirm this matches sagemaker_write_bucket_arns in iam_sagemaker.tf

IMAGE_URI = f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_REPO_NAME}:{IMAGE_TAG}"

# --- Upload the local file to S3 first ---
# SageMaker Processing containers run in AWS and can only read from S3 -
# they have no access to your laptop's filesystem. This step bridges that gap.
if not os.path.exists(LOCAL_DATA_PATH):
    raise FileNotFoundError(f"Local file not found: {LOCAL_DATA_PATH}")

print(f"Uploading {LOCAL_DATA_PATH} to s3://{INPUT_BUCKET}/{INPUT_KEY} ...")
s3_client = boto3.client("s3", region_name=AWS_REGION)
s3_client.upload_file(LOCAL_DATA_PATH, INPUT_BUCKET, INPUT_KEY)
print("Upload complete.")

INPUT_S3_URI = f"s3://{INPUT_BUCKET}/{INPUT_KEY}"

session = sagemaker.Session()

processor = Processor(
    image_uri=IMAGE_URI,
    role=EXECUTION_ROLE_ARN,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    sagemaker_session=session,
)

processor.run(
    inputs=[
        ProcessingInput(
            source=INPUT_S3_URI,
            destination="/opt/ml/processing/input",
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=OUTPUT_S3_URI,
        )
    ],
)

print("Processing job submitted. Check status in the SageMaker console")
print("or via: aws sagemaker list-processing-jobs --region", AWS_REGION)
