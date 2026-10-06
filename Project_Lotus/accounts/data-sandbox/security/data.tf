# Section 4.4 boundary, resolved from the account contract rather than hardcoded,
# so this root carries no account id and stays copyable on promotion.
data "aws_ssm_parameter" "permissions_boundary" {
  count = var.permissions_boundary_ssm_path == null ? 0 : 1
  name  = var.permissions_boundary_ssm_path
}

data "aws_partition" "current" {}
data "aws_region" "current" {}
data "aws_caller_identity" "current" {}
