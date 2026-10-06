locals {
  bronze_schema_validator_name        = "${var.project}-${var.environment}-bronze-schema-validator"
  bronze_schema_validator_bucket_name = "${var.project}-${var.environment}-bronze-landing-data"

  bronze_schema_validator_bucket_arn = "arn:${data.aws_partition.current.partition}:s3:::${local.bronze_schema_validator_bucket_name}"

  bronze_schema_validator_control_database_name = "control"
  bronze_schema_validator_control_table_name    = "bronze_schema_validation"

  bronze_schema_validator_athena_workgroup_name = "primary"
}

data "aws_iam_policy_document" "bronze_schema_validator_assume_role" {
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

resource "aws_iam_role" "bronze_schema_validator" {
  name = "${local.bronze_schema_validator_name}-role"

  assume_role_policy   = data.aws_iam_policy_document.bronze_schema_validator_assume_role.json
  permissions_boundary = local.permissions_boundary
}

data "aws_iam_policy_document" "bronze_schema_validator_runtime" {
  # Explicit maintenance action in the validator Lambda. Listing is account 
  # scoped, but tag reads and termination are limited to EMR clusters, and
  # termination additionally requires the workflow tag used at creation. 
  statement {
    sid    = "DiscoverTransientWorkflowClusters"
    effect = "Allow"
    actions = [
      "elasticmapreduce:ListClusters"
    ]
    resources = ["*"]
  }

  statement {
    sid    = "ReadTransientWorkflowClusterTags"
    effect = "Allow"
    actions = [
      "elasticmapreduce:DescribeCluster"
    ]
    resources = [
      "arn:${data.aws_partition.current.partition}:elasticmapreduce:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:cluster/*"
    ]
  }

  statement {
    sid    = "TerminateTaggedTransientWorkflowClusters"
    effect = "Allow"
    actions = [
      "elasticmapreduce:TerminateJobFlows"
    ]
    resources = [
      "arn:${data.aws_partition.current.partition}:elasticmapreduce:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:cluster/*"
    ]

    condition {
      test     = "StringEquals"
      variable = "elasticmapreduce:ResourceTag/lotus-workflow"
      values   = ["silver-gold-transient"]
    }
  }

  statement {
    sid    = "ListSchemaValidatorPrefixes"
    effect = "Allow"

    actions = [
      "s3:ListBucket"
    ]

    resources = [
      local.bronze_schema_validator_bucket_arn
    ]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"

      values = [
        "bronze",
        "bronze/*",
        "schema_registry",
        "schema_registry/*",
        "control",
        "control/*",
        "athena-results",
        "athena-results/*"
      ]
    }
  }

  statement {
    sid    = "GetSchemaValidatorBucketLocation"
    effect = "Allow"

    actions = [
      "s3:GetBucketLocation"
    ]

    resources = [
      local.bronze_schema_validator_bucket_arn
    ]
  }

  statement {
    sid    = "ReadBronzeParquetFiles"
    effect = "Allow"

    actions = [
      "s3:GetObject"
    ]

    resources = [
      "${local.bronze_schema_validator_bucket_arn}/bronze/*"
    ]
  }

  statement {
    sid    = "ReadSchemaRegistry"
    effect = "Allow"

    actions = [
      "s3:GetObject"
    ]

    resources = [
      "${local.bronze_schema_validator_bucket_arn}/schema_registry/*"
    ]
  }

  statement {
    sid    = "UseAthenaResultsLocation"
    effect = "Allow"

    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:AbortMultipartUpload",
      "s3:ListMultipartUploadParts"
    ]

    resources = [
      "${local.bronze_schema_validator_bucket_arn}/athena-results/*"
    ]
  }

  statement {
    sid    = "UseIcebergControlTableLocation"
    effect = "Allow"

    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:AbortMultipartUpload",
      "s3:ListMultipartUploadParts"
    ]

    resources = [
      "${local.bronze_schema_validator_bucket_arn}/control/bronze_schema_validation/*"
    ]
  }

  statement {
    sid    = "RunAthenaQueries"
    effect = "Allow"

    actions = [
      "athena:StartQueryExecution",
      "athena:GetQueryExecution"
    ]

    resources = [
      "arn:${data.aws_partition.current.partition}:athena:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:workgroup/${local.bronze_schema_validator_athena_workgroup_name}"
    ]
  }

  # The trigger into Silver: on a PASS this function starts one execution of
  # the per-dataset Gate1->Silver->Gate2-check state machine, scoped to
  # exactly the dataset it just validated. See silver_gold_control_lambda's
  # SILVER_STATE_MACHINE_ARN / DATASET_SPECS_JSON for the other half of this.
  statement {
    sid    = "StartSilverStateMachine"
    effect = "Allow"

    actions = [
      "states:StartExecution"
    ]

    resources = [
      "arn:${data.aws_partition.current.partition}:states:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:stateMachine:${local.sfn_bronze_silver_name}"
    ]
  }

  statement {
    sid    = "ReadGlueControlDatabase"
    effect = "Allow"

    actions = [
      "glue:GetDatabase"
    ]

    resources = [
      "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:catalog",
      "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:database/${local.bronze_schema_validator_control_database_name}"
    ]
  }

  statement {
    sid    = "AccessGlueIcebergControlTable"
    effect = "Allow"

    actions = [
      "glue:GetTable",
      "glue:GetTableVersion",
      "glue:GetTableVersions",
      "glue:GetPartitions",
      "glue:UpdateTable"
    ]

    resources = [
      "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:catalog",
      "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:database/${local.bronze_schema_validator_control_database_name}",
      "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:table/${local.bronze_schema_validator_control_database_name}/${local.bronze_schema_validator_control_table_name}"
    ]
  }

  statement {
    sid    = "WriteLambdaLogs"
    effect = "Allow"

    actions = [
      "logs:CreateLogStream",
      "logs:PutLogEvents"
    ]

    resources = [
      "arn:${data.aws_partition.current.partition}:logs:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/${local.bronze_schema_validator_name}:*"
    ]
  }
}

resource "aws_iam_policy" "bronze_schema_validator_runtime" {
  name   = "${local.bronze_schema_validator_name}-runtime-policy"
  policy = data.aws_iam_policy_document.bronze_schema_validator_runtime.json
}

resource "aws_iam_role_policy_attachment" "bronze_schema_validator_runtime" {
  role       = aws_iam_role.bronze_schema_validator.name
  policy_arn = aws_iam_policy.bronze_schema_validator_runtime.arn
}
