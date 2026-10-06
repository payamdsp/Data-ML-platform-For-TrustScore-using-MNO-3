# Persistent runtime role used by Gold Spark steps submitted to EMR.
#
# This role survives EMR cluster recreation. Each Gold EMR step should use:
#
#   --execution-role-arn <this role ARN>
#
# The existing EMR EC2 instance-profile role assumes this runtime role.
#
# The Gold pipeline reads Silver and merges into Gold, so this role is
# read-only on Silver and read-write on Gold - which is the whole reason it
# exists separately from the Silver runtime role rather than one shared role
# holding the union of both.

locals {
  gold_job_runtime_role_name = "${local.name_prefix}-gold-job-runtime"

  gold_runtime_silver_bucket_arn = "arn:${data.aws_partition.current.partition}:s3:::lotus-${var.environment}-silver-conformed-data"
  gold_runtime_gold_bucket_arn   = "arn:${data.aws_partition.current.partition}:s3:::lotus-${var.environment}-gold-curated-data"

  gold_runtime_silver_glue_database_name = replace(
    "${var.project}-${var.environment}-silver-conformed",
    "-",
    "_"
  )

  gold_runtime_gold_glue_database_name = replace(
    "${var.project}-${var.environment}-gold-curated",
    "-",
    "_"
  )

  gold_runtime_glue_catalog_arn = "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:catalog"

  gold_runtime_silver_glue_database_arn = "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:database/${local.gold_runtime_silver_glue_database_name}"
  gold_runtime_silver_glue_table_arn    = "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:table/${local.gold_runtime_silver_glue_database_name}/*"

  gold_runtime_gold_glue_database_arn = "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:database/${local.gold_runtime_gold_glue_database_name}"
  gold_runtime_gold_glue_table_arn    = "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:table/${local.gold_runtime_gold_glue_database_name}/*"
}


# ---------------------------------------------------------------------------
# Runtime-role trust policy
#
# EMR EC2 nodes assume the runtime role when an EMR step is submitted with
# --execution-role-arn.
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "gold_job_runtime_assume_role" {
  statement {
    sid    = "AllowEmrEc2RuntimeRoleAssumption"
    effect = "Allow"

    actions = [
      "sts:AssumeRole",
      "sts:TagSession"
    ]

    principals {
      type        = "AWS"
      identifiers = [aws_iam_role.emr_ec2.arn]
    }
  }
}


resource "aws_iam_role" "gold_job_runtime" {
  name                 = local.gold_job_runtime_role_name
  assume_role_policy   = data.aws_iam_policy_document.gold_job_runtime_assume_role.json
  permissions_boundary = local.permissions_boundary

  tags = {
    Name = local.gold_job_runtime_role_name
  }
}


# ---------------------------------------------------------------------------
# Gold job permissions
# ---------------------------------------------------------------------------

resource "aws_iam_role_policy" "gold_job_runtime_access" {
  name = "gold-job-data-and-catalog-access"
  role = aws_iam_role.gold_job_runtime.id

  policy = jsonencode({
    Version = "2012-10-17"

    Statement = [

      # ---------------------------------------------------------------------
      # Silver - read only
      #
      # The six Silver Iceberg tables are the pipeline's inputs, and the three
      # enrichment CSVs are read from the same bucket's seed/ prefix. The Gold
      # job never writes to Silver.
      # ---------------------------------------------------------------------

      {
        Sid    = "ListSilverBucket"
        Effect = "Allow"

        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]

        Resource = local.gold_runtime_silver_bucket_arn
      },

      {
        Sid    = "ReadSilverObjects"
        Effect = "Allow"

        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion"
        ]

        Resource = "${local.gold_runtime_silver_bucket_arn}/*"
      },

      # ---------------------------------------------------------------------
      # Gold - read/write
      #
      # Used for:
      # - Iceberg table locations and their metadata/manifests
      # - the run-control slice manifests written before a run mutates anything
      # - DQ metric output
      #
      # DeleteObject is required and is not optional: Gold is merged, never
      # appended, so a MERGE rewrites data files and expires the ones it
      # replaced.
      # ---------------------------------------------------------------------

      {
        Sid    = "ListGoldBucket"
        Effect = "Allow"

        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation",
          "s3:ListBucketMultipartUploads"
        ]

        Resource = local.gold_runtime_gold_bucket_arn
      },

      {
        Sid    = "ReadWriteGoldObjects"
        Effect = "Allow"

        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:AbortMultipartUpload",
          "s3:ListMultipartUploadParts"
        ]

        Resource = "${local.gold_runtime_gold_bucket_arn}/*"
      },

      # ---------------------------------------------------------------------
      # Glue Data Catalog
      #
      # Gold Iceberg tables already exist through Terraform, so CreateTable is
      # intentionally not required here - the same omission the Silver runtime
      # role makes. The pipeline's `create_if_missing` sinks must therefore be
      # set to false in conf/lineage, so that a mistyped table name fails
      # loudly rather than being denied halfway through a run.
      #
      # UpdateTable is required: an Iceberg commit rewrites the table's
      # metadata pointer in Glue on every merge.
      # ---------------------------------------------------------------------

      {
        Sid    = "GoldGlueCatalogAccess"
        Effect = "Allow"

        Action = [
          "glue:GetDatabase",
          "glue:GetDatabases",
          "glue:GetTable",
          "glue:GetTables",
          "glue:GetTableVersion",
          "glue:GetTableVersions",
          "glue:UpdateTable"
        ]

        Resource = [
          local.gold_runtime_glue_catalog_arn,
          local.gold_runtime_gold_glue_database_arn,
          local.gold_runtime_gold_glue_table_arn
        ]
      },

      {
        Sid    = "SilverGlueCatalogReadAccess"
        Effect = "Allow"

        Action = [
          "glue:GetDatabase",
          "glue:GetDatabases",
          "glue:GetTable",
          "glue:GetTables",
          "glue:GetTableVersion",
          "glue:GetTableVersions"
        ]

        Resource = [
          local.gold_runtime_glue_catalog_arn,
          local.gold_runtime_silver_glue_database_arn,
          local.gold_runtime_silver_glue_table_arn
        ]
      },

      # ---------------------------------------------------------------------
      # Lake Formation credential vending
      #
      # Actual database/table grants are intentionally created from INFRA
      # because the infra CI role is the Lake Formation Data Lake Admin.
      # ---------------------------------------------------------------------

      {
        Sid      = "LakeFormationDataAccess"
        Effect   = "Allow"
        Action   = ["lakeformation:GetDataAccess"]
        Resource = "*"
      }
    ]
  })
}


# ---------------------------------------------------------------------------
# Allow the existing EMR EC2 instance-profile role to assume this runtime role.
# Without this, --execution-role-arn fails at step submission.
# ---------------------------------------------------------------------------

resource "aws_iam_role_policy" "emr_ec2_assume_gold_runtime" {
  name = "assume-gold-job-runtime-role"
  role = aws_iam_role.emr_ec2.id

  policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Sid    = "AllowGoldRuntimeRoleUsage"
        Effect = "Allow"

        Action = [
          "sts:AssumeRole",
          "sts:TagSession"
        ]

        Resource = aws_iam_role.gold_job_runtime.arn
      }
    ]
  })
}


output "gold_job_runtime_role_arn" {
  description = "Runtime role ARN to supply using --execution-role-arn when submitting Gold EMR steps."
  value       = aws_iam_role.gold_job_runtime.arn
}
