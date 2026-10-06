locals {
  silver_gold_control_name = "${var.project}-${var.environment}-silver-gold-control"

  # The Silver->Gold handoff control table and release object live in the
  # Silver bucket under control/, not the Bronze landing bucket - matching the
  # naming convention already used for the Silver bucket by the EMR runtime
  # roles (see emr_silver_runtime.tf / emr_gold_runtime.tf).
  silver_gold_control_bucket_name = "lotus-${var.environment}-silver-conformed-data"

  silver_gold_control_bucket_arn = "arn:${data.aws_partition.current.partition}:s3:::${local.silver_gold_control_bucket_name}"

  # The Bronze validation table's own bucket, still read (never written) here
  # for the gate_one action (called from the per-dataset Silver state machine).
  silver_gold_control_bronze_bucket_name = "${var.project}-${var.environment}-bronze-landing-data"
  silver_gold_control_bronze_bucket_arn  = "arn:${data.aws_partition.current.partition}:s3:::${local.silver_gold_control_bronze_bucket_name}"

  silver_gold_control_database_name = "control"
  silver_gold_control_table_name    = "silver_gold_control"

  # Read by the gate_one action; never written by this role.
  silver_gold_control_bronze_table_name = "bronze_schema_validation"
  pipeline_orchestration_claim_prefix   = "control/pipeline-claims"

  silver_gold_control_athena_workgroup_name = "primary"

  silver_gold_release_prefix = "gold-release"

  # Read by build_gold_args - the Gold job's own bucket, not Silver's or
  # Bronze's. This is a plain S3 object, not a Glue-cataloged table, so it
  # needs an ordinary s3:GetObject grant and no Lake Formation permission.
  #
  # This key must stay byte-for-byte in sync with
  # infra/gold_job_config.tf's local.gold_job_config_key - the two Terraform
  # roots do not share locals, so a rename on one side has to be made on both
  # or this Lambda loses read access to the object it just renamed.
  silver_gold_control_gold_bucket_name    = "lotus-${var.environment}-gold-curated-data"
  silver_gold_control_gold_bucket_arn     = "arn:${data.aws_partition.current.partition}:s3:::${local.silver_gold_control_gold_bucket_name}"
  silver_gold_control_gold_job_config_key = "pipeline-code/gold/auto/job_config.json"

  # Same pairing requirement against infra/silver_job_config.tf's
  # local.silver_job_config_key. This one lives in the Silver bucket
  # (silver_gold_control_bucket_arn, already defined above), not a separate
  # bucket - Silver's own job config sits next to the control table and the
  # release object, all three on the Silver->Gold boundary.
  silver_gold_control_silver_job_config_key = "pipeline-code/silver/job_config.json"
}

data "aws_iam_policy_document" "silver_gold_control_assume_role" {
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

resource "aws_iam_role" "silver_gold_control" {
  name = "${local.silver_gold_control_name}-role"

  assume_role_policy   = data.aws_iam_policy_document.silver_gold_control_assume_role.json
  permissions_boundary = local.permissions_boundary
}

data "aws_iam_policy_document" "silver_gold_control_runtime" {
  statement {
    sid    = "ListControlAndReleasePrefixes"
    effect = "Allow"

    actions = [
      "s3:ListBucket"
    ]

    resources = [
      local.silver_gold_control_bucket_arn
    ]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"

      values = [
        "control",
        "control/*",
        "athena-results",
        "athena-results/*",
        local.silver_gold_release_prefix,
        "${local.silver_gold_release_prefix}/*",
        "pipeline-code/silver",
        "pipeline-code/silver/*"
      ]
    }
  }

  statement {
    sid    = "GetControlBucketLocation"
    effect = "Allow"

    actions = [
      "s3:GetBucketLocation"
    ]

    resources = [
      local.silver_gold_control_bucket_arn
    ]
  }

  # The release object is both written (automatic release) and read (release
  # gate). Deletion is deliberately absent: withdrawing a release is an
  # operator action, not something this function should be able to do.
  statement {
    sid    = "ReadWriteGoldReleaseObjects"
    effect = "Allow"

    actions = [
      "s3:GetObject",
      "s3:PutObject"
    ]

    resources = [
      "${local.silver_gold_control_bucket_arn}/${local.silver_gold_release_prefix}/*"
    ]
  }

  # build_gold_args reads this object fresh on every call, deliberately - see
  # its docstring - so a console edit takes effect on the next Gold stage
  # without redeploying this function. Read-only: an operator edits it, this
  # role never writes it.
  statement {
    sid    = "ReadGoldJobConfig"
    effect = "Allow"

    actions = [
      "s3:GetObject"
    ]

    resources = [
      "${local.silver_gold_control_gold_bucket_arn}/${local.silver_gold_control_gold_job_config_key}"
    ]
  }

  # build_silver_args's counterpart to ReadGoldJobConfig above. Read-only,
  # same reasoning: an operator edits it, this role never writes it.
  statement {
    sid    = "ReadSilverJobConfig"
    effect = "Allow"

    actions = [
      "s3:GetObject"
    ]

    resources = [
      "${local.silver_gold_control_bucket_arn}/${local.silver_gold_control_silver_job_config_key}"
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
      "${local.silver_gold_control_bucket_arn}/athena-results/*"
    ]
  }

  statement {
    sid    = "UseIcebergHandoffControlTableLocation"
    effect = "Allow"

    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:AbortMultipartUpload",
      "s3:ListMultipartUploadParts"
    ]

    resources = [
      "${local.silver_gold_control_bucket_arn}/control/${local.silver_gold_control_table_name}/*"
    ]
  }

  # Read-only on the Bronze validation table's data files, for the Gate 1
  # gate_one action path. This table's data lives in the Bronze landing
  # bucket, not the Silver bucket the rest of this policy targets.
  statement {
    sid    = "ReadBronzeValidationTableLocation"
    effect = "Allow"

    actions = [
      "s3:GetObject"
    ]

    resources = [
      "${local.silver_gold_control_bronze_bucket_arn}/control/${local.silver_gold_control_bronze_table_name}/*"
    ]
  }

  statement {
    sid     = "CoordinateDailyClusterInS3"
    effect  = "Allow"
    actions = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = [
      "${local.silver_gold_control_bronze_bucket_arn}/${local.pipeline_orchestration_claim_prefix}/cluster/*"
    ]
  }

  statement {
    sid    = "ListBronzeValidationTablePrefix"
    effect = "Allow"

    actions = [
      "s3:ListBucket"
    ]

    resources = [
      local.silver_gold_control_bronze_bucket_arn
    ]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"

      values = [
        "control/${local.silver_gold_control_bronze_table_name}",
        "control/${local.silver_gold_control_bronze_table_name}/*"
      ]
    }
  }

  statement {
    sid    = "RunAthenaQueries"
    effect = "Allow"

    actions = [
      "athena:StartQueryExecution",
      "athena:GetQueryExecution",
      "athena:GetQueryResults"
    ]

    resources = [
      "arn:${data.aws_partition.current.partition}:athena:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:workgroup/${local.silver_gold_control_athena_workgroup_name}"
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
      "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:database/${local.silver_gold_control_database_name}"
    ]
  }

  statement {
    sid    = "AccessGlueIcebergHandoffControlTable"
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
      "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:database/${local.silver_gold_control_database_name}",
      "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:table/${local.silver_gold_control_database_name}/${local.silver_gold_control_table_name}"
    ]
  }

  # Bronze validation table: catalog read only, no UpdateTable.
  statement {
    sid    = "ReadGlueBronzeValidationTable"
    effect = "Allow"

    actions = [
      "glue:GetTable",
      "glue:GetTableVersion",
      "glue:GetTableVersions",
      "glue:GetPartitions"
    ]

    resources = [
      "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:catalog",
      "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:database/${local.silver_gold_control_database_name}",
      "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:table/${local.silver_gold_control_database_name}/${local.silver_gold_control_bronze_table_name}"
    ]
  }

  statement {
    sid    = "PublishPipelineAlerts"
    effect = "Allow"

    actions = [
      "sns:Publish"
    ]

    resources = [
      "arn:${data.aws_partition.current.partition}:sns:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:${var.project}-${var.environment}-silver-pipeline-alerts",
      "arn:${data.aws_partition.current.partition}:sns:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:${var.project}-${var.environment}-gold-pipeline-alerts"
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
      "arn:${data.aws_partition.current.partition}:logs:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/${local.silver_gold_control_name}:*"
    ]
  }
}

resource "aws_iam_policy" "silver_gold_control_runtime" {
  name   = "${local.silver_gold_control_name}-runtime-policy"
  policy = data.aws_iam_policy_document.silver_gold_control_runtime.json
}

resource "aws_iam_role_policy_attachment" "silver_gold_control_runtime" {
  role       = aws_iam_role.silver_gold_control.name
  policy_arn = aws_iam_policy.silver_gold_control_runtime.arn
}
