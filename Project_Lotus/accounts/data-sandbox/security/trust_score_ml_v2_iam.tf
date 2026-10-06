# =============================================================================
# VERSION 2 - Trust Score ML pipeline (SageMaker training + nightly scoring)
# =============================================================================
# Added in v2. No existing file in this root was modified; see
# ../CHANGES_V2.md for what this adds and why.
#
# This root owns every aws_iam_role so the permissions boundary is applied in
# one place. The ML pipeline's runtime pieces (state machines, Lambda, rules,
# SNS topic, SSM parameter) live in ../infra/trust_score_ml_v2_*.tf and look the
# roles below up BY NAME - the same split the Gold / Silver pipelines use.
#
# Apply this root before ../infra.
#
# Roles added:
#   <project>-<env>-trust-score-ml-sfn-role      Step Functions: runs the SageMaker jobs
#   <project>-<env>-trust-score-ml-events-role   EventBridge: starts the two state machines
#   <project>-<env>-trust-score-ml-helper-role   helper Lambda: logs + one SSM parameter
# Policy added to an EXISTING role (sagemaker_iam.tf, not edited):
#   <project>-<env>-sagemaker-execution  +  inline policy "trust-score-ml-v2-pipeline-access"
# =============================================================================

variable "ml_v2_output_prefix" {
  description = "Prefix in the SageMaker data bucket that the ML jobs write (models, selections, champion, scores). Must match ml_v2_output_prefix in ../infra and models_root in conf/ml/lotus_sandbox.yaml. Ends with '/'."
  type        = string
  default     = "demo_model_results/"

  validation {
    condition     = endswith(var.ml_v2_output_prefix, "/") && !startswith(var.ml_v2_output_prefix, "/")
    error_message = "ml_v2_output_prefix must end with '/' and must not start with '/'."
  }
}

variable "ml_v2_job_name_prefix" {
  description = "Prefix of every SageMaker job the pipeline starts. The Step Functions role may only create/describe/stop jobs whose names start with it. Must match ml_v2_job_name_prefix in ../infra."
  type        = string
  default     = "ts05"
}

variable "ml_v2_jobs_in_vpc" {
  description = "Set true when ../infra runs the jobs inside subnets (ml_v2_job_subnet_ids). SageMaker then needs the EC2 network-interface permissions on the execution role."
  type        = bool
  default     = false
}

locals {
  ml_v2_name = "${local.name_prefix}-trust-score-ml"

  ml_v2_arn_prefix = {
    sagemaker = "arn:${data.aws_partition.current.partition}:sagemaker:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}"
    states    = "arn:${data.aws_partition.current.partition}:states:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}"
    lambda    = "arn:${data.aws_partition.current.partition}:lambda:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}"
    events    = "arn:${data.aws_partition.current.partition}:events:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}"
    sns       = "arn:${data.aws_partition.current.partition}:sns:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}"
    ssm       = "arn:${data.aws_partition.current.partition}:ssm:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}"
    logs      = "arn:${data.aws_partition.current.partition}:logs:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}"
  }

  # Names of the runtime resources created in ../infra. Built here from the
  # same convention, never read from that root's state.
  ml_v2_training_state_machine_arn = "${local.ml_v2_arn_prefix.states}:stateMachine:${local.ml_v2_name}-training"
  ml_v2_scoring_state_machine_arn  = "${local.ml_v2_arn_prefix.states}:stateMachine:${local.ml_v2_name}-scoring"
  ml_v2_helper_function_name       = "${local.ml_v2_name}-helper"
  ml_v2_helper_function_arn        = "${local.ml_v2_arn_prefix.lambda}:function:${local.ml_v2_helper_function_name}"
  ml_v2_alert_topic_arn            = "${local.ml_v2_arn_prefix.sns}:${local.name_prefix}-ml-pipeline-alerts"
  # SSM parameter ARNs carry the name without its leading slash.
  ml_v2_last_training_parameter_arn = "${local.ml_v2_arn_prefix.ssm}:parameter/${var.project}/${var.environment}/trust-score-ml/last-training-start"

  ml_v2_output_bucket_arn = "arn:${data.aws_partition.current.partition}:s3:::${local.name_prefix}-sagemaker-data"
}

data "aws_iam_policy_document" "ml_v2_states_assume_role" {
  statement {
    sid     = "StatesAssumeRole"
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["states.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }
  }
}

data "aws_iam_policy_document" "ml_v2_events_assume_role" {
  statement {
    sid     = "EventsAssumeRole"
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }
  }
}

data "aws_iam_policy_document" "ml_v2_lambda_assume_role" {
  statement {
    sid     = "LambdaAssumeRole"
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

# -----------------------------------------------------------------------------
# 1. Step Functions role: runs the SageMaker jobs of both state machines
# -----------------------------------------------------------------------------

resource "aws_iam_role" "ml_v2_sfn" {
  name = "${local.ml_v2_name}-sfn-role"

  assume_role_policy   = data.aws_iam_policy_document.ml_v2_states_assume_role.json
  permissions_boundary = local.permissions_boundary
}

data "aws_iam_policy_document" "ml_v2_sfn" {
  # Scoped to the job-name prefix: the helper Lambda builds every job name from
  # it, so the pipeline cannot start or stop any other SageMaker job.
  statement {
    sid    = "RunPipelineSageMakerJobs"
    effect = "Allow"

    actions = [
      "sagemaker:CreateTrainingJob",
      "sagemaker:DescribeTrainingJob",
      "sagemaker:StopTrainingJob",
      "sagemaker:CreateProcessingJob",
      "sagemaker:DescribeProcessingJob",
      "sagemaker:StopProcessingJob",
      "sagemaker:AddTags",
      "sagemaker:ListTags"
    ]

    resources = [
      "${local.ml_v2_arn_prefix.sagemaker}:training-job/${var.ml_v2_job_name_prefix}-*",
      "${local.ml_v2_arn_prefix.sagemaker}:processing-job/${var.ml_v2_job_name_prefix}-*"
    ]
  }

  # The jobs run as the existing SageMaker execution role (sagemaker_iam.tf).
  statement {
    sid       = "PassExecutionRoleToSageMakerOnly"
    effect    = "Allow"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.sagemaker_execution.arn]

    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["sagemaker.amazonaws.com"]
    }
  }

  # The managed rules the .sync SageMaker integrations use to learn when a job
  # finishes. Without these the state machine fails at its first job.
  statement {
    sid    = "ManageSageMakerSyncRules"
    effect = "Allow"

    actions = [
      "events:PutTargets",
      "events:PutRule",
      "events:DescribeRule"
    ]

    resources = [
      "${local.ml_v2_arn_prefix.events}:rule/StepFunctionsGetEventsForSageMakerTrainingJobsRule",
      "${local.ml_v2_arn_prefix.events}:rule/StepFunctionsGetEventsForSageMakerProcessingJobsRule"
    ]
  }

  statement {
    sid       = "InvokeHelperLambda"
    effect    = "Allow"
    actions   = ["lambda:InvokeFunction"]
    resources = [local.ml_v2_helper_function_arn, "${local.ml_v2_helper_function_arn}:*"]
  }

  # The scoring machine reads the job's summary.json to decide on drift.
  statement {
    sid       = "ReadScoringSummaries"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = ["${local.ml_v2_output_bucket_arn}/${var.ml_v2_output_prefix}scoring-summaries/*"]
  }

  statement {
    sid       = "PublishMlPipelineAlerts"
    effect    = "Allow"
    actions   = ["sns:Publish"]
    resources = [local.ml_v2_alert_topic_arn]
  }

  # Training: the "is a run already in progress?" guard lists its own
  # executions. Scoring: a drift retrain starts the training machine.
  statement {
    sid       = "GuardAndStartTrainingMachine"
    effect    = "Allow"
    actions   = ["states:ListExecutions", "states:StartExecution"]
    resources = [local.ml_v2_training_state_machine_arn]
  }

  # Vended-logs delivery for logging_configuration. These actions do not
  # support resource-level permissions; same statement as sfn_common.
  statement {
    sid    = "WriteStateMachineLogs"
    effect = "Allow"

    actions = [
      "logs:CreateLogDelivery",
      "logs:GetLogDelivery",
      "logs:UpdateLogDelivery",
      "logs:DeleteLogDelivery",
      "logs:ListLogDeliveries",
      "logs:PutResourcePolicy",
      "logs:DescribeResourcePolicies",
      "logs:DescribeLogGroups"
    ]

    resources = ["*"]
  }
}

resource "aws_iam_policy" "ml_v2_sfn" {
  name   = "${local.ml_v2_name}-sfn-policy"
  policy = data.aws_iam_policy_document.ml_v2_sfn.json
}

resource "aws_iam_role_policy_attachment" "ml_v2_sfn" {
  role       = aws_iam_role.ml_v2_sfn.name
  policy_arn = aws_iam_policy.ml_v2_sfn.arn
}

# -----------------------------------------------------------------------------
# 2. EventBridge role: may start the two ML state machines, nothing else
# -----------------------------------------------------------------------------

resource "aws_iam_role" "ml_v2_events" {
  name = "${local.ml_v2_name}-events-role"

  assume_role_policy   = data.aws_iam_policy_document.ml_v2_events_assume_role.json
  permissions_boundary = local.permissions_boundary
}

data "aws_iam_policy_document" "ml_v2_events" {
  statement {
    sid       = "StartMlStateMachines"
    effect    = "Allow"
    actions   = ["states:StartExecution"]
    resources = [local.ml_v2_training_state_machine_arn, local.ml_v2_scoring_state_machine_arn]
  }
}

resource "aws_iam_policy" "ml_v2_events" {
  name   = "${local.ml_v2_name}-events-policy"
  policy = data.aws_iam_policy_document.ml_v2_events.json
}

resource "aws_iam_role_policy_attachment" "ml_v2_events" {
  role       = aws_iam_role.ml_v2_events.name
  policy_arn = aws_iam_policy.ml_v2_events.arn
}

# -----------------------------------------------------------------------------
# 3. Helper Lambda role: its own log group and one SSM parameter
# -----------------------------------------------------------------------------

resource "aws_iam_role" "ml_v2_helper" {
  name = "${local.ml_v2_name}-helper-role"

  assume_role_policy   = data.aws_iam_policy_document.ml_v2_lambda_assume_role.json
  permissions_boundary = local.permissions_boundary
}

data "aws_iam_policy_document" "ml_v2_helper" {
  statement {
    sid       = "WriteLambdaLogs"
    effect    = "Allow"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${local.ml_v2_arn_prefix.logs}:log-group:/aws/lambda/${local.ml_v2_helper_function_name}:*"]
  }

  # The retrain-cooldown clock: written when a training run starts, read by
  # the drift retrain gate.
  statement {
    sid       = "ReadWriteLastTrainingStart"
    effect    = "Allow"
    actions   = ["ssm:GetParameter", "ssm:PutParameter"]
    resources = [local.ml_v2_last_training_parameter_arn]
  }
}

resource "aws_iam_policy" "ml_v2_helper" {
  name   = "${local.ml_v2_name}-helper-policy"
  policy = data.aws_iam_policy_document.ml_v2_helper.json
}

resource "aws_iam_role_policy_attachment" "ml_v2_helper" {
  role       = aws_iam_role.ml_v2_helper.name
  policy_arn = aws_iam_policy.ml_v2_helper.arn
}

# -----------------------------------------------------------------------------
# 4. Addition to the EXISTING SageMaker execution role (sagemaker_iam.tf)
# -----------------------------------------------------------------------------
# sagemaker_iam.tf grants write access to the gold bucket only
# (var.sagemaker_write_bucket_arns). The ML jobs write every artifact - feature
# selections, models, the champion, nightly scores and summaries - to the
# SageMaker data bucket, so without this policy the first job fails with
# AccessDenied. A separate inline policy, so sagemaker_iam.tf stays untouched.
# Reading the gold bucket (input data) and pulling images are already granted.

data "aws_iam_policy_document" "ml_v2_execution_extra" {
  statement {
    sid       = "LocateSageMakerDataBucket"
    effect    = "Allow"
    actions   = ["s3:GetBucketLocation"]
    resources = [local.ml_v2_output_bucket_arn]
  }

  statement {
    sid       = "ListMlOutputPrefix"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [local.ml_v2_output_bucket_arn]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = [trimsuffix(var.ml_v2_output_prefix, "/"), "${var.ml_v2_output_prefix}*"]
    }
  }

  statement {
    sid    = "ReadWriteMlOutputs"
    effect = "Allow"

    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:AbortMultipartUpload",
      "s3:ListMultipartUploadParts"
    ]

    resources = ["${local.ml_v2_output_bucket_arn}/${var.ml_v2_output_prefix}*"]
  }

  # Only when the jobs run in a VPC: SageMaker creates ENIs as the execution
  # role. These EC2 actions do not support resource-level scoping.
  dynamic "statement" {
    for_each = var.ml_v2_jobs_in_vpc ? [1] : []

    content {
      sid    = "RunJobsInVpc"
      effect = "Allow"

      actions = [
        "ec2:CreateNetworkInterface",
        "ec2:CreateNetworkInterfacePermission",
        "ec2:DeleteNetworkInterface",
        "ec2:DeleteNetworkInterfacePermission",
        "ec2:DescribeNetworkInterfaces",
        "ec2:DescribeVpcs",
        "ec2:DescribeDhcpOptions",
        "ec2:DescribeSubnets",
        "ec2:DescribeSecurityGroups"
      ]

      resources = ["*"]
    }
  }
}

resource "aws_iam_role_policy" "ml_v2_execution_extra" {
  name   = "trust-score-ml-v2-pipeline-access"
  role   = aws_iam_role.sagemaker_execution.id
  policy = data.aws_iam_policy_document.ml_v2_execution_extra.json
}

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

output "ml_v2_sfn_role_arn" {
  description = "VERSION 2: role the Trust Score ML state machines run as."
  value       = aws_iam_role.ml_v2_sfn.arn
}

output "ml_v2_events_role_arn" {
  description = "VERSION 2: role EventBridge uses to start the ML state machines."
  value       = aws_iam_role.ml_v2_events.arn
}

output "ml_v2_helper_role_arn" {
  description = "VERSION 2: role of the ML pipeline helper Lambda."
  value       = aws_iam_role.ml_v2_helper.arn
}
