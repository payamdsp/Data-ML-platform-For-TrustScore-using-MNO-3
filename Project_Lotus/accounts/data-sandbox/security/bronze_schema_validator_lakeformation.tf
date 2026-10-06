data "aws_iam_role" "bronze_schema_validator_lf" {
  name = "${var.project}-${var.environment}-bronze-schema-validator-role"
}

data "aws_iam_policy_document" "bronze_schema_validator_lakeformation" {
  statement {
    sid    = "LakeFormationDataAccess"
    effect = "Allow"

    actions = [
      "lakeformation:GetDataAccess"
    ]

    resources = ["*"]
  }
}

resource "aws_iam_policy" "bronze_schema_validator_lakeformation" {
  name        = "${var.project}-${var.environment}-bronze-schema-validator-lakeformation-policy"
  description = "Allows the Bronze schema validator to access Lake Formation governed data."

  policy = data.aws_iam_policy_document.bronze_schema_validator_lakeformation.json
}

resource "aws_iam_role_policy_attachment" "bronze_schema_validator_lakeformation" {
  role       = data.aws_iam_role.bronze_schema_validator_lf.name
  policy_arn = aws_iam_policy.bronze_schema_validator_lakeformation.arn
}