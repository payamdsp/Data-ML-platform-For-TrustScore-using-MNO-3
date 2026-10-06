#!/bin/bash

set -euo pipefail

AWS_REGION="ca-central-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPO_NAME="lotus-sandbox-preprocessing"
IMAGE_TAG="${1:-demo_latest}"

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}"

echo "========================================"
echo "ECR Repository : $REPO_NAME"
echo "Image Tag      : $IMAGE_TAG"
echo "========================================"

echo "Checking whether image already exists..."

if aws ecr describe-images \
    --repository-name "$REPO_NAME" \
    --image-ids imageTag="$IMAGE_TAG" \
    --region "$AWS_REGION" >/dev/null 2>&1; then

    echo "Image already exists:"
    echo "$ECR_URI:$IMAGE_TAG"
    echo "Skipping build and push."

    exit 0
fi

echo "Image does not exist. Continuing..."

echo "Logging in to ECR..."

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login \
      --username AWS \
      --password-stdin \
      "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "Building image..."

docker build \
  -t "${REPO_NAME}:${IMAGE_TAG}" \
  .

echo "Tagging image..."

docker tag \
  "${REPO_NAME}:${IMAGE_TAG}" \
  "${ECR_URI}:${IMAGE_TAG}"

echo "Pushing image..."

docker push "${ECR_URI}:${IMAGE_TAG}"

echo "========================================"
echo "Successfully pushed:"
echo "${ECR_URI}:${IMAGE_TAG}"
echo "========================================"