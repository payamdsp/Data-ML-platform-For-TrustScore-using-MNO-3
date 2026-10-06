locals {
  sfn_bronze_silver_name = "${var.project}-${var.environment}-trust-score-bronze-silver"
  sfn_gold_name          = "${var.project}-${var.environment}-trust-score-gold"

  sfn_control_lambda_name = "${var.project}-${var.environment}-silver-gold-control"

  sfn_alert_topic_arns = [
    "arn:${data.aws_partition.current.partition}:sns:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:${var.project}-${var.environment}-bronze-validation-alerts",
    "arn:${data.aws_partition.current.partition}:sns:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:${var.project}-${var.environment}-silver-pipeline-alerts",
    "arn:${data.aws_partition.current.partition}:sns:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:${var.project}-${var.environment}-gold-pipeline-alerts",
  ]
}

data "aws_iam_policy_document" "sfn_assume_role" {
  statement {
    sid     = "StatesAssumeRole"
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["states.amazonaws.com"]
    }
  }
}

# ---------------------------------------------------------------------------
# Shared statements
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "sfn_common" {
  statement {
    sid    = "AddAndWatchEmrSteps"
    effect = "Allow"

    actions = [
      "elasticmapreduce:AddJobFlowSteps",
      "elasticmapreduce:DescribeStep",
      "elasticmapreduce:CancelSteps",
      "elasticmapreduce:DescribeCluster",
      "elasticmapreduce:ListClusters"
    ]

    resources = ["*"]
  }

  # The .sync integration for EMR steps drives an EventBridge managed rule on
  # the caller's behalf; without this the state machine fails as soon as it
  # tries to wait for a step.
  statement {
    sid    = "ManageSyncExecutionRule"
    effect = "Allow"

    actions = [
      "events:PutTargets",
      "events:PutRule",
      "events:DescribeRule"
    ]

    resources = [
      "arn:${data.aws_partition.current.partition}:events:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:rule/StepFunctionsGetEventsForEMR*"
    ]
  }

  statement {
    sid    = "PublishPipelineAlerts"
    effect = "Allow"

    actions = ["sns:Publish"]

    resources = local.sfn_alert_topic_arns
  }

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

# ---------------------------------------------------------------------------
# Bronze -> Silver state machine
# ---------------------------------------------------------------------------

resource "aws_iam_role" "sfn_bronze_silver" {
  name = "${local.sfn_bronze_silver_name}-role"

  assume_role_policy   = data.aws_iam_policy_document.sfn_assume_role.json
  permissions_boundary = local.permissions_boundary
}

data "aws_iam_policy_document" "sfn_bronze_silver" {
  source_policy_documents = [data.aws_iam_policy_document.sfn_common.json]

  statement {
    sid       = "CreateTaggedDailyEmr"
    effect    = "Allow"
    actions   = ["elasticmapreduce:RunJobFlow"]
    resources = ["*"]

    condition {
      test     = "StringEquals"
      variable = "aws:RequestTag/lotus-workflow"
      values   = ["silver-gold-transient"]
    }
  }

  # EMR authorizes tags supplied during RunJobFlow with a separate AddTags
  # check. Scope it to cluster resources and require the workflow tag that the
  # state machine supplies in CreateCluster.
  statement {
    sid       = "TagTransientDailyEmr"
    effect    = "Allow"
    actions   = ["elasticmapreduce:AddTags"]
    resources = ["arn:${data.aws_partition.current.partition}:elasticmapreduce:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:cluster/*"]

    condition {
      test     = "StringEquals"
      variable = "aws:RequestTag/lotus-workflow"
      values   = ["silver-gold-transient"]
    }
  }

  statement {
    sid       = "CleanUpTaggedDailyEmr"
    effect    = "Allow"
    actions   = ["elasticmapreduce:TerminateJobFlows"]
    resources = ["arn:${data.aws_partition.current.partition}:elasticmapreduce:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:cluster/*"]

    condition {
      test     = "StringEquals"
      variable = "elasticmapreduce:ResourceTag/lotus-workflow"
      values   = ["silver-gold-transient"]
    }
  }

  statement {
    sid     = "PassEmrClusterRoles"
    effect  = "Allow"
    actions = ["iam:PassRole"]
    resources = [
      aws_iam_role.emr_service.arn,
      aws_iam_role.emr_ec2.arn
    ]
  }

  # Gate 1's Athena query runs inside the control Lambda's `gate_one` action,
  # not as an inline Athena Task on this state machine - see
  # stepfunctions_bronze_silver.tf for why. This role therefore needs no
  # Athena, Glue, or Lake Formation permissions of its own; invoking the
  # Lambda (below) is the only access it needs for Gate 1.
  statement {
    sid    = "InvokeControlLambda"
    effect = "Allow"

    actions = ["lambda:InvokeFunction"]

    resources = [
      "arn:${data.aws_partition.current.partition}:lambda:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:function:${local.sfn_control_lambda_name}"
    ]
  }

  # Silver EMR steps are submitted with an execution role; the submitter must be
  # allowed to pass it.
  statement {
    sid    = "PassSilverJobRuntimeRole"
    effect = "Allow"

    actions = ["iam:PassRole"]

    resources = [
      aws_iam_role.silver_job_runtime.arn
    ]
  }
}

resource "aws_iam_policy" "sfn_bronze_silver" {
  name   = "${local.sfn_bronze_silver_name}-policy"
  policy = data.aws_iam_policy_document.sfn_bronze_silver.json
}

resource "aws_iam_role_policy_attachment" "sfn_bronze_silver" {
  role       = aws_iam_role.sfn_bronze_silver.name
  policy_arn = aws_iam_policy.sfn_bronze_silver.arn
}

# EventBridge starts this workflow for every matching S3 object. Its
# conditional S3 claim allows only the first file per dataset/date to
# wait 15 minutes and invoke the existing Bronze validator.
resource "aws_iam_role" "sfn_bronze_arrival" {
  name                 = "${var.project}-${var.environment}-bronze-arrival-role"
  assume_role_policy   = data.aws_iam_policy_document.sfn_assume_role.json
  permissions_boundary = local.permissions_boundary
}

data "aws_iam_policy_document" "sfn_bronze_arrival" {
  statement {
    sid     = "ClaimAndValidateBronzeFolder"
    effect  = "Allow"
    actions = ["lambda:InvokeFunction"]
    resources = [
      "arn:${data.aws_partition.current.partition}:lambda:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:function:${var.project}-${var.environment}-bronze-schema-validator",
      "arn:${data.aws_partition.current.partition}:lambda:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:function:${var.project}-${var.environment}-bronze-arrival-claim"
    ]
  }

  statement {
    sid    = "WriteStateMachineLogs"
    effect = "Allow"
    actions = [
      "logs:CreateLogDelivery", "logs:GetLogDelivery", "logs:UpdateLogDelivery",
      "logs:DeleteLogDelivery", "logs:ListLogDeliveries", "logs:PutResourcePolicy",
      "logs:DescribeResourcePolicies", "logs:DescribeLogGroups"
    ]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "sfn_bronze_arrival" {
  name   = "${var.project}-${var.environment}-bronze-arrival-policy"
  policy = data.aws_iam_policy_document.sfn_bronze_arrival.json
}

resource "aws_iam_role_policy_attachment" "sfn_bronze_arrival" {
  role       = aws_iam_role.sfn_bronze_arrival.name
  policy_arn = aws_iam_policy.sfn_bronze_arrival.arn
}

resource "aws_iam_role" "eventbridge_bronze_arrival" {
  name                 = "${var.project}-${var.environment}-bronze-arrival-eventbridge-role"
  assume_role_policy   = data.aws_iam_policy_document.eventbridge_gold_trigger_assume_role.json
  permissions_boundary = local.permissions_boundary
}

data "aws_iam_policy_document" "eventbridge_bronze_arrival" {
  statement {
    sid       = "StartBronzeArrival"
    effect    = "Allow"
    actions   = ["states:StartExecution"]
    resources = ["arn:${data.aws_partition.current.partition}:states:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:stateMachine:${var.project}-${var.environment}-bronze-arrival"]
  }
}

resource "aws_iam_policy" "eventbridge_bronze_arrival" {
  name   = "${var.project}-${var.environment}-bronze-arrival-eventbridge-policy"
  policy = data.aws_iam_policy_document.eventbridge_bronze_arrival.json
}

resource "aws_iam_role_policy_attachment" "eventbridge_bronze_arrival" {
  role       = aws_iam_role.eventbridge_bronze_arrival.name
  policy_arn = aws_iam_policy.eventbridge_bronze_arrival.arn
}

# ---------------------------------------------------------------------------
# Gold state machine
# ---------------------------------------------------------------------------

resource "aws_iam_role" "sfn_gold" {
  name = "${local.sfn_gold_name}-role"

  assume_role_policy   = data.aws_iam_policy_document.sfn_assume_role.json
  permissions_boundary = local.permissions_boundary
}

data "aws_iam_policy_document" "sfn_gold" {
  source_policy_documents = [data.aws_iam_policy_document.sfn_common.json]

  statement {
    sid       = "TerminateTaggedDailyEmr"
    effect    = "Allow"
    actions   = ["elasticmapreduce:TerminateJobFlows"]
    resources = ["arn:${data.aws_partition.current.partition}:elasticmapreduce:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:cluster/*"]

    condition {
      test     = "StringEquals"
      variable = "elasticmapreduce:ResourceTag/lotus-workflow"
      values   = ["silver-gold-transient"]
    }
  }

  statement {
    sid    = "PassGoldJobRuntimeRole"
    effect = "Allow"

    actions = ["iam:PassRole"]

    resources = [
      aws_iam_role.gold_job_runtime.arn
    ]
  }

  # The execution's first state (ResolveRelease) calls the control Lambda's
  # `release_gate` action to verify the release object and mint the run id.
  # EventBridge starts this execution directly - see the comment on
  # aws_iam_role.eventbridge_gold_trigger below for why this Lambda is not
  # invoked by EventBridge itself.
  statement {
    sid    = "InvokeControlLambda"
    effect = "Allow"

    actions = ["lambda:InvokeFunction"]

    resources = [
      "arn:${data.aws_partition.current.partition}:lambda:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:function:${local.sfn_control_lambda_name}"
    ]
  }
}

resource "aws_iam_policy" "sfn_gold" {
  name   = "${local.sfn_gold_name}-policy"
  policy = data.aws_iam_policy_document.sfn_gold.json
}

resource "aws_iam_role_policy_attachment" "sfn_gold" {
  role       = aws_iam_role.sfn_gold.name
  policy_arn = aws_iam_policy.sfn_gold.arn
}

# ---------------------------------------------------------------------------
# EventBridge -> Gold state machine (Gate 2 release trigger)
#
# EventBridge targets the Gold state machine directly with `states:StartExecution`,
# rather than targeting the control Lambda the way the release-object rule
# used to. That original design hit a hard account-level guardrail: the CI/CD
# role that applies this Terraform (gha-lotus-data-sandbox-infra) carries an
# EXPLICIT DENY on lambda:AddPermission, which is exactly the API
# aws_lambda_permission calls to let EventBridge invoke a Lambda target. There
# is no Terraform-side workaround for an explicit deny - it can only be lifted
# by whoever owns that role's policy.
#
# Targeting Step Functions instead needs no Lambda resource-based policy at
# all: authorization is this ordinary identity-based role, assumed by
# EventBridge, granted nothing but states:StartExecution on one state machine.
# The logic the control Lambda used to run before calling StartExecution
# itself (read release.json, re-check the control table, mint the run id) now
# runs as the FIRST STATE inside the execution instead - see
# stepfunctions_gold.tf's ResolveRelease state - which only needs an ordinary
# lambda:InvokeFunction grant (above), not a resource-based one.
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "eventbridge_gold_trigger_assume_role" {
  statement {
    sid     = "EventBridgeAssumeRole"
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "eventbridge_gold_trigger" {
  name = "${local.sfn_gold_name}-eventbridge-trigger-role"

  assume_role_policy   = data.aws_iam_policy_document.eventbridge_gold_trigger_assume_role.json
  permissions_boundary = local.permissions_boundary
}

data "aws_iam_policy_document" "eventbridge_gold_trigger" {
  statement {
    sid    = "StartGoldStateMachine"
    effect = "Allow"

    actions = ["states:StartExecution"]

    resources = [
      "arn:${data.aws_partition.current.partition}:states:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:stateMachine:${local.sfn_gold_name}"
    ]
  }
}

resource "aws_iam_policy" "eventbridge_gold_trigger" {
  name   = "${local.sfn_gold_name}-eventbridge-trigger-policy"
  policy = data.aws_iam_policy_document.eventbridge_gold_trigger.json
}

resource "aws_iam_role_policy_attachment" "eventbridge_gold_trigger" {
  role       = aws_iam_role.eventbridge_gold_trigger.name
  policy_arn = aws_iam_policy.eventbridge_gold_trigger.arn
}
