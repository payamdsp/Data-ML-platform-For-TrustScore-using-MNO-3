# Lake Formation data access for the Silver -> Gold control Lambda.
#
# lakeformation:GetDataAccess is what turns an LF table grant into actual
# credentials for the underlying S3 objects. The table and database grants
# themselves live in ../infra/silver_gold_control_lakeformation.tf, next to the
# table they refer to; this file only gives the role the ability to exercise
# them.

data "aws_iam_policy_document" "silver_gold_control_lakeformation" {
  statement {
    sid    = "LakeFormationDataAccess"
    effect = "Allow"

    actions = [
      "lakeformation:GetDataAccess"
    ]

    resources = ["*"]
  }
}

resource "aws_iam_policy" "silver_gold_control_lakeformation" {
  name        = "${local.silver_gold_control_name}-lakeformation-policy"
  description = "Allows the Silver-to-Gold control Lambda to access Lake Formation governed data."

  policy = data.aws_iam_policy_document.silver_gold_control_lakeformation.json
}

resource "aws_iam_role_policy_attachment" "silver_gold_control_lakeformation" {
  role       = aws_iam_role.silver_gold_control.name
  policy_arn = aws_iam_policy.silver_gold_control_lakeformation.arn
}
