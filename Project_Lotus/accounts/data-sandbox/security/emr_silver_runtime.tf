# Persistent runtime role used by Silver Spark steps submitted to EMR.
#
# This role survives EMR cluster recreation. Each Silver EMR step should use:
#
#   --execution-role-arn <this role ARN>
#
# The existing EMR EC2 instance-profile role assumes this runtime role.

locals {
  silver_job_runtime_role_name = "${local.name_prefix}-silver-job-runtime"

  silver_runtime_bronze_bucket_arn = "arn:${data.aws_partition.current.partition}:s3:::lotus-${var.environment}-bronze-landing-data"
  silver_runtime_silver_bucket_arn = "arn:${data.aws_partition.current.partition}:s3:::lotus-${var.environment}-silver-conformed-data"

  silver_runtime_glue_database_name = replace(
    "${var.project}-${var.environment}-silver-conformed",
    "-",
    "_"
  )

  silver_runtime_glue_catalog_arn = "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:catalog"

  silver_runtime_glue_database_arn = "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:database/${local.silver_runtime_glue_database_name}"

  silver_runtime_glue_table_arn = "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:table/${local.silver_runtime_glue_database_name}/*"
}


# ---------------------------------------------------------------------------
# Runtime-role trust policy
#
# EMR EC2 nodes assume the runtime role when an EMR step is submitted with
# --execution-role-arn.
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "silver_job_runtime_assume_role" {
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


resource "aws_iam_role" "silver_job_runtime" {
  name                 = local.silver_job_runtime_role_name
  assume_role_policy   = data.aws_iam_policy_document.silver_job_runtime_assume_role.json
  permissions_boundary = local.permissions_boundary

  tags = {
    Name = local.silver_job_runtime_role_name
  }
}


# ---------------------------------------------------------------------------
# Silver job permissions
# ---------------------------------------------------------------------------

resource "aws_iam_role_policy" "silver_job_runtime_access" {
  name = "silver-job-data-and-catalog-access"
  role = aws_iam_role.silver_job_runtime.id

  policy = jsonencode({
    Version = "2012-10-17"

    Statement = [

      # ---------------------------------------------------------------------
      # Bronze
      #
      # Source Parquet data + silver_pipeline.zip + run_silver.py are read
      # from the Bronze bucket.
      # ---------------------------------------------------------------------

      {
        Sid    = "ListBronzeBucket"
        Effect = "Allow"

        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]

        Resource = local.silver_runtime_bronze_bucket_arn
      },

      {
        Sid    = "ReadBronzeObjects"
        Effect = "Allow"

        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion"
        ]

        Resource = "${local.silver_runtime_bronze_bucket_arn}/*"
      },

      # ---------------------------------------------------------------------
      # Silver
      #
      # Used for:
      # - Iceberg table locations
      # - run artifacts
      # - temporary/output files required by the Silver pipeline
      # ---------------------------------------------------------------------

      {
        Sid    = "ListSilverBucket"
        Effect = "Allow"

        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation",
          "s3:ListBucketMultipartUploads"
        ]

        Resource = local.silver_runtime_silver_bucket_arn
      },

      {
        Sid    = "ReadWriteSilverObjects"
        Effect = "Allow"

        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:AbortMultipartUpload",
          "s3:ListMultipartUploadParts"
        ]

        Resource = "${local.silver_runtime_silver_bucket_arn}/*"
      },

      # ---------------------------------------------------------------------
      # Glue Data Catalog
      #
      # Silver Iceberg tables already exist through Terraform, so CreateTable
      # is intentionally not required here.
      # ---------------------------------------------------------------------

      {
        Sid    = "SilverGlueCatalogAccess"
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
          local.silver_runtime_glue_catalog_arn,
          local.silver_runtime_glue_database_arn,
          local.silver_runtime_glue_table_arn
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
# ---------------------------------------------------------------------------

resource "aws_iam_role_policy" "emr_ec2_assume_silver_runtime" {
  name = "assume-silver-job-runtime-role"
  role = aws_iam_role.emr_ec2.id

  policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Sid    = "AllowSilverRuntimeRoleUsage"
        Effect = "Allow"

        Action = [
          "sts:AssumeRole",
          "sts:TagSession"
        ]

        Resource = aws_iam_role.silver_job_runtime.arn
      }
    ]
  })
}


output "silver_job_runtime_role_arn" {
  description = "Runtime role ARN to supply using --execution-role-arn when submitting Silver EMR steps."
  value       = aws_iam_role.silver_job_runtime.arn
}