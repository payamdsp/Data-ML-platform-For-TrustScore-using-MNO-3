# IAM roles and policies for Amazon ECR repository access.
# Pull role: used by execution compute (SageMaker, ECS tasks) to pull images.
# Push role: used by GitHub Actions CI via GHE OIDC — no static AWS keys required.

variable "ecr_repository_names" {
  description = "List of ECR repository names allowed for access under this IAM configuration."
  type        = list(string)
  default     = ["lotus-sandbox-preprocessing", "lotus-sandbox-model_autoencoder_training", "lotus-sandbox-model_autoencoder_inference"]
}

locals {
  ecr_repository_arns = [
    for repo_name in var.ecr_repository_names :
    "arn:${data.aws_partition.current.partition}:ecr:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:repository/${repo_name}"
  ]
}

# GHE OIDC provider already exists in this account (used by gha-lotus-data-sandbox-infra).
# Reference it — do NOT declare a new aws_iam_openid_connect_provider resource here.
data "aws_iam_openid_connect_provider" "ghe_actions" {
  url = "https://token.actions.enstream.ghe.com"
}

# ------------------------------------------------------------------------------
# 1. READ-ONLY ACCESS — PULL (execution compute: SageMaker, ECS tasks)
# ------------------------------------------------------------------------------

data "aws_iam_policy_document" "ecr_pull_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com", "sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ecr_pull" {
  name                 = "${local.name_prefix}-ecr-pull-role"
  assume_role_policy   = data.aws_iam_policy_document.ecr_pull_assume_role.json
  permissions_boundary = local.permissions_boundary
}

resource "aws_iam_role_policy" "ecr_pull_access" {
  name = "ecr-read-only-pull-access"
  role = aws_iam_role.ecr_pull.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ECRGetAuthToken"
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Sid    = "ECRPullImages"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:DescribeRepositories",
          "ecr:ListImages"
        ]
        Resource = local.ecr_repository_arns
      }
    ]
  })
}

# ------------------------------------------------------------------------------
# 2. READ-WRITE ACCESS — PUSH (GitHub Actions via GHE OIDC, no static keys)
#
# GitHub Actions exchanges its workflow JWT for short-lived AWS credentials by
# assuming this role through the GitHub OIDC provider. No IAM user or access
# keys are needed; credentials are scoped to the job's lifetime (~15 min).
# ------------------------------------------------------------------------------

data "aws_iam_policy_document" "ecr_push_assume_role" {
  statement {
    sid     = "GheOidcBuildAndPush"
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [data.aws_iam_openid_connect_provider.ghe_actions.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.enstream.ghe.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.enstream.ghe.com:ref"
      values   = ["refs/heads/main"]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.enstream.ghe.com:sub"
      values   = ["repo:data-and-ai/Project_Lotus:environment:data-sandbox-infra"]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.enstream.ghe.com:job_workflow_ref"
      values   = ["data-and-ai/Project_Lotus/.github/workflows/build-and-push.yml@refs/heads/main"]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.enstream.ghe.com:job_workflow_ref"
      values   = ["data-and-ai/Project_Lotus/.github/workflows/build-and-push-autoencoder.yml@refs/heads/main"]
    }
  }
}

resource "aws_iam_role" "ecr_push" {
  name                 = "${local.name_prefix}-ecr-push-role"
  assume_role_policy   = data.aws_iam_policy_document.ecr_push_assume_role.json
  permissions_boundary = local.permissions_boundary
  max_session_duration = 3600
}

resource "aws_iam_role_policy" "ecr_push_access" {
  name = "ecr-read-write-push-access"
  role = aws_iam_role.ecr_push.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ECRGetAuthToken"
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Sid    = "ECRPullAndPushImages"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:DescribeRepositories",
          "ecr:ListImages"
        ]
        Resource = local.ecr_repository_arns
      }
    ]
  })
}

# ------------------------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------------------------

output "ecr_pull_role_arn" {
  description = "ARN of the ECR read-only role used by SageMaker and ECS tasks to pull container images."
  value       = aws_iam_role.ecr_pull.arn
}

output "ecr_push_role_arn" {
  description = "ARN of the ECR read-write role assumed by GitHub Actions via GHE OIDC to build and push images."
  value       = aws_iam_role.ecr_push.arn
}