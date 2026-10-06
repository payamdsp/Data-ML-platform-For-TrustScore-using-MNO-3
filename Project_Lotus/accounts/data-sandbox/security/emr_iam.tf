# IAM required by the persistent EMR cluster in ../infra/emr.tf.
# The service role is scoped through AWS's EMR v2 managed policy; application
# access from the EC2 nodes is granted only to the supplied data resources.

locals {
  medallion_zones = ["bronze-landing", "silver-conformed", "gold-curated"]
  medallion_bucket_arns = [
    for zone in local.medallion_zones :
    "arn:${data.aws_partition.current.partition}:s3:::${var.project}-${var.environment}-${zone}-data"
  ]

  medallion_glue_database_names = [
    for zone in local.medallion_zones :
    replace("${var.project}-${var.environment}-${zone}", "-", "_")
  ]

  emr_glue_database_arns = [
    for database_name in local.medallion_glue_database_names :
    "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:database/${database_name}"
  ]
  emr_glue_table_arns = [
    for database_name in local.medallion_glue_database_names :
    "arn:${data.aws_partition.current.partition}:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:table/${database_name}/*"
  ]
}

data "aws_iam_policy_document" "emr_service_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["elasticmapreduce.amazonaws.com"]
    }
  }
}

resource "aws_iam_policy" "lotus_emr_ec2" {
  name        = "${local.name_prefix}-emr-ec2-policy"
  description = "S3 access for EMR EC2 nodes to lotus sandbox medallion buckets"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "BronzeLandingRead"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          "arn:${data.aws_partition.current.partition}:s3:::lotus-${var.environment}-bronze-landing-data",
          "arn:${data.aws_partition.current.partition}:s3:::lotus-${var.environment}-bronze-landing-data/*"
        ]
      },
      {
        Sid    = "SilverConformedReadWrite"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          "arn:${data.aws_partition.current.partition}:s3:::lotus-${var.environment}-silver-conformed-data",
          "arn:${data.aws_partition.current.partition}:s3:::lotus-${var.environment}-silver-conformed-data/*"
        ]
      },
      {
        Sid    = "GoldCuratedReadWrite"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          "arn:${data.aws_partition.current.partition}:s3:::lotus-${var.environment}-gold-curated-data",
          "arn:${data.aws_partition.current.partition}:s3:::lotus-${var.environment}-gold-curated-data/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role" "emr_service" {
  name                 = "${local.name_prefix}-emr-service"
  assume_role_policy   = data.aws_iam_policy_document.emr_service_assume_role.json
  permissions_boundary = local.permissions_boundary

  tags = {
    "for-use-with-amazon-emr-managed-policies" = "true"
  }
}

resource "aws_iam_role_policy_attachment" "emr_service" {
  role       = aws_iam_role.emr_service.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AmazonEMRServicePolicy_v2"
}

# Required on the first Spot-backed EMR cluster in an account when the EC2 Spot
# service-linked role has not already been created.
resource "aws_iam_role_policy" "emr_service_spot_service_linked_role" {
  name = "create-ec2-spot-service-linked-role"
  role = aws_iam_role.emr_service.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid      = "CreateEc2SpotServiceLinkedRole"
      Effect   = "Allow"
      Action   = "iam:CreateServiceLinkedRole"
      Resource = "arn:${data.aws_partition.current.partition}:iam::*:role/aws-service-role/spot.amazonaws.com/AWSServiceRoleForEC2Spot"
      Condition = {
        StringEquals = {
          "iam:AWSServiceName" = "spot.amazonaws.com"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "emr_service_pass_ec2_role" {
  name = "pass-custom-named-ec2-role"
  role = aws_iam_role.emr_service.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid      = "PassRoleForEc2"
      Effect   = "Allow"
      Action   = "iam:PassRole"
      Resource = aws_iam_role.emr_ec2.arn
      Condition = {
        StringLike = {
          "iam:PassedToService" = "ec2.amazonaws.com*"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "emr_service_describe_prefix_lists" {
  name = "describe-prefix-lists-for-managed-security-group-egress"
  role = aws_iam_role.emr_service.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid      = "DescribePrefixListsForManagedSecurityGroupEgress"
      Effect   = "Allow"
      Action   = "ec2:DescribePrefixLists"
      Resource = "*"
    }]
  })
}

resource "aws_iam_role_policy" "emr_service_ec2_provisioning" {
  name = "ec2-provisioning-for-emr"
  role = aws_iam_role.emr_service.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid    = "Ec2ProvisioningForEmr"
      Effect = "Allow"
      Action = [
        "ec2:CancelSpotInstanceRequests",
        "ec2:RequestSpotInstances",
        "ec2:ModifyImageAttribute",
        "ec2:DetachNetworkInterface",
        "ec2:CreateVolume",
        "ec2:AttachVolume",
        "ec2:DetachVolume",
        "ec2:DeleteVolume",
        # Added for instance fleets: CreateFleet/ModifyFleet/DeleteFleets are a
        # distinct EC2 API surface from RequestSpotInstances (used by instance
        # groups), and are required now that master/core use fleet config.
        "ec2:CreateFleet",
        "ec2:ModifyFleet",
        "ec2:DeleteFleets",
        "ec2:DescribeFleets",
        "ec2:DescribeFleetInstances",
        "ec2:DescribeSpotPriceHistory"
      ]
      Resource = "*"
    }]
  })
}

data "aws_iam_policy_document" "emr_ec2_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "emr_ec2" {
  name                 = "${local.name_prefix}-emr-ec2"
  assume_role_policy   = data.aws_iam_policy_document.emr_ec2_assume_role.json
  permissions_boundary = local.permissions_boundary

  tags = {
    "for-use-with-amazon-emr-managed-policies" = "true"
  }
}

resource "aws_iam_role_policy_attachment" "emr_ec2_lotus" {
  role       = aws_iam_role.emr_ec2.name
  policy_arn = aws_iam_policy.lotus_emr_ec2.arn
}

resource "aws_iam_instance_profile" "emr_ec2" {
  name = "${local.name_prefix}-emr-ec2"
  role = aws_iam_role.emr_ec2.name

  tags = {
    "for-use-with-amazon-emr-managed-policies" = "true"
  }
}

resource "aws_iam_role_policy" "emr_ec2_data_access" {
  name = "iceberg-data-and-catalog-access"
  role = aws_iam_role.emr_ec2.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ListDataBuckets"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = local.medallion_bucket_arns
      },
      {
        Sid    = "ReadWriteDataObjects"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [for bucket_arn in local.medallion_bucket_arns : "${bucket_arn}/*"]
      },
      {
        Sid      = "ReadIcebergCatalogDatabases"
        Effect   = "Allow"
        Action   = ["glue:GetDatabase"]
        Resource = local.emr_glue_database_arns
      },
      {
        Sid    = "ReadWriteIcebergCatalogTables"
        Effect = "Allow"
        Action = [
          "glue:GetTable",
          "glue:GetTables",
          "glue:UpdateTable"
        ]
        Resource = local.emr_glue_table_arns
      },
      {
        Sid      = "CreateIcebergCatalogTables"
        Effect   = "Allow"
        Action   = ["glue:CreateTable"]
        Resource = local.emr_glue_database_arns
      },
      # Lake Formation requires this exact wildcard resource for data access.
      # Database, table, and data-location grants remain owned by the data lake administrator.
      {
        Sid      = "AccessLakeFormationGovernedData"
        Effect   = "Allow"
        Action   = ["lakeformation:GetDataAccess"]
        Resource = "*"
      },
      # Required for native EMR-to-CloudWatch Logs streaming (EMR 7.11.0+).

      {
        Sid    = "StreamLogsToCloudWatch"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:${data.aws_partition.current.partition}:logs:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/emr/*"
      }
    ]
  })
}

output "emr_service_role_arn" {
  description = "ARN of the EMR service role used to provision the cluster."
  value       = aws_iam_role.emr_service.arn
}

output "emr_ec2_instance_profile_arn" {
  description = "ARN of the EC2 instance profile used by EMR cluster nodes."
  value       = aws_iam_instance_profile.emr_ec2.arn
}
