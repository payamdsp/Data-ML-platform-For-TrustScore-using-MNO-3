data "aws_iam_policy_document" "bronze_arrival_claim_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "bronze_arrival_claim" {
  name                 = "${var.project}-${var.environment}-bronze-arrival-claim-role"
  assume_role_policy   = data.aws_iam_policy_document.bronze_arrival_claim_assume_role.json
  permissions_boundary = local.permissions_boundary
}

data "aws_iam_policy_document" "bronze_arrival_claim" {
  statement {
    sid     = "ClaimArrivalWindow"
    effect  = "Allow"
    actions = ["s3:GetObject", "s3:PutObject"]
    resources = [
      "arn:${data.aws_partition.current.partition}:s3:::${var.project}-${var.environment}-bronze-landing-data/control/bronze-arrival-claims/*"
    ]
  }

  statement {
    sid     = "WriteLambdaLogs"
    effect  = "Allow"
    actions = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = [
      "arn:${data.aws_partition.current.partition}:logs:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/${var.project}-${var.environment}-bronze-arrival-claim:*"
    ]
  }
}

resource "aws_iam_policy" "bronze_arrival_claim" {
  name   = "${var.project}-${var.environment}-bronze-arrival-claim-policy"
  policy = data.aws_iam_policy_document.bronze_arrival_claim.json
}

resource "aws_iam_role_policy_attachment" "bronze_arrival_claim" {
  role       = aws_iam_role.bronze_arrival_claim.name
  policy_arn = aws_iam_policy.bronze_arrival_claim.arn
}
