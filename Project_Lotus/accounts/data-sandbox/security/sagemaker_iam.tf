# IAM required by SageMaker Studio, Pipelines, Processing, Training, and Batch
# Transform jobs in ../infra/sagemaker.tf. One execution role is shared across
# all of these, which is the standard SageMaker pattern (the role a Studio
# notebook runs as is the same role it passes when it kicks off a job).

variable "sagemaker_read_bucket_arns" {
  description = "ARNs of S3 buckets SageMaker may list and read from."
  type        = list(string)

  default = [
    "arn:aws:s3:::lotus-sandbox-bronze-landing-data",
    "arn:aws:s3:::lotus-sandbox-gold-curated-data",
    "arn:aws:s3:::lotus-sandbox-sagemaker-data"
  ]
}

variable "sagemaker_write_bucket_arns" {
  description = "ARNs of S3 buckets SageMaker may list, read, and write (training/processing/batch-transform output, model artifacts)."
  type        = list(string)
  default = [
    "arn:aws:s3:::lotus-sandbox-gold-curated-data",
    "arn:aws:s3:::lotus-sandbox-sagemaker-data"
  ]
}

locals {
  sagemaker_glue_database_name = replace("${var.project}-${var.environment}-gold-curated", "-", "_")

  sagemaker_glue_database_arns = [
    "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:database/${local.sagemaker_glue_database_name}"
  ]
  sagemaker_glue_table_arns = [
    "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:table/${local.sagemaker_glue_database_name}/*"
  ]
  sagemaker_log_group_arn = "arn:${data.aws_partition.current.partition}:logs:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/*"
}

resource "aws_iam_service_linked_role" "sagemaker_notebooks" {
  aws_service_name = "sagemaker.amazonaws.com"
  description      = "Allows Amazon SageMaker to create and manage resources on your behalf for Studio domains and notebooks."
}

data "aws_iam_policy_document" "sagemaker_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker_execution" {
  name                 = "${local.name_prefix}-sagemaker-execution"
  assume_role_policy   = data.aws_iam_policy_document.sagemaker_assume_role.json
  permissions_boundary = local.permissions_boundary
}

resource "aws_iam_role_policy" "sagemaker_execution_access" {
  name = "gold-data-training-and-pipelines-access"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ListAndLocateDataBuckets"
        Effect = "Allow"

        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]

        Resource = concat(
          var.sagemaker_read_bucket_arns,
          var.sagemaker_write_bucket_arns
        )
      },
      {
        Sid      = "ReadGoldObjects"
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = [for bucket_arn in var.sagemaker_read_bucket_arns : "${bucket_arn}/*"]
      },
      {
        Sid    = "ReadWriteModelAndJobOutput"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [for bucket_arn in var.sagemaker_write_bucket_arns : "${bucket_arn}/*"]
      },
      {
        Sid      = "ReadGoldCatalogDatabases"
        Effect   = "Allow"
        Action   = ["glue:GetDatabase"]
        Resource = local.sagemaker_glue_database_arns
      },
      {
        Sid    = "ReadGoldCatalogTables"
        Effect = "Allow"
        Action = [
          "glue:GetTable",
          "glue:GetTables"
        ]
        Resource = local.sagemaker_glue_table_arns
      },
      # Lake Formation requires this exact wildcard resource for data access.
      # Database, table, and data-location grants remain owned by the data lake administrator.
      {
        Sid      = "AccessLakeFormationGovernedData"
        Effect   = "Allow"
        Action   = ["lakeformation:GetDataAccess"]
        Resource = "*"
      },
      {
        Sid    = "TrainingProcessingAndPipelineJobs"
        Effect = "Allow"
        Action = [
          "sagemaker:CreateAlgorithm",
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:StopTrainingJob",
          "sagemaker:CreateProcessingJob",
          "sagemaker:DescribeProcessingJob",
          "sagemaker:StopProcessingJob",
          "sagemaker:CreateTransformJob",
          "sagemaker:DescribeTransformJob",
          "sagemaker:StopTransformJob",
          "sagemaker:CreateModel",
          "sagemaker:CreateModelPackage",
          "sagemaker:DescribeModelPackage",
          "sagemaker:UpdateModelPackage",
          "sagemaker:CreatePipeline",
          "sagemaker:UpdatePipeline",
          "sagemaker:StartPipelineExecution",
          "sagemaker:DescribePipeline",
          "sagemaker:DescribePipelineExecution",
          "sagemaker:ListPipelineExecutions",
          "sagemaker:AddTags"
        ]
        # Job and pipeline names are dynamic at run time, so these cannot be
        # scoped to a specific resource ARN; the Action list is the least-
        # privilege boundary here.
        Resource = "*"
      },
      {
        Sid      = "PassExecutionRoleToOwnJobs"
        Effect   = "Allow"
        Action   = ["iam:PassRole"]
        Resource = aws_iam_role.sagemaker_execution.arn
        Condition = {
          StringEquals = {
            "iam:PassedToService" = "sagemaker.amazonaws.com"
          }
        }
      },
      {
        Sid    = "TrainingAndPipelineLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = local.sagemaker_log_group_arn
      },
      {
        Sid      = "TrainingMetrics"
        Effect   = "Allow"
        Action   = ["cloudwatch:PutMetricData"]
        Resource = "*"
      },
      {
        Sid      = "PullSageMakerAlgorithmImages"
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Sid    = "PullSageMakerAlgorithmImageLayers"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer"
        ]
        # AWS-owned SageMaker built-in algorithm/framework repositories,
        # one set per region; cannot be scoped to an account-owned ARN.
        Resource = "*"
      }
    ]
  })
}

output "sagemaker_execution_role_arn" {
  description = "ARN of the shared SageMaker execution role for Studio, Pipelines, Training, Processing, and Batch Transform."
  value       = aws_iam_role.sagemaker_execution.arn
}