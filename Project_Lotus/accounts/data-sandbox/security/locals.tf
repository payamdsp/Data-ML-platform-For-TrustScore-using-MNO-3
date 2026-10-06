locals {
  name_prefix = "${var.project}-${var.environment}"

  # Every aws_iam_role in this root must set permissions_boundary = local.permissions_boundary.
  # The account boundary denies iam:CreateRole where the boundary in the request is
  # anything other than sandbox-boundary, and denies it outright where the request
  # sets none, so a role declared without this fails at apply rather than at plan.
  permissions_boundary = try(data.aws_ssm_parameter.permissions_boundary[0].value, null)
}
