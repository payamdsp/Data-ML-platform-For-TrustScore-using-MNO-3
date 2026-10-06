# =============================================================================
# VERSION 2 (add-on) - who may call the Serverless inference endpoint
# =============================================================================
# Added in v2; no existing file in this root was modified. See ../CHANGES_V2.md.
#
# The endpoint itself needs nothing new here: it runs as the existing SageMaker
# execution role, which already pulls images and (with the v2 policy in
# trust_score_ml_v2_iam.tf) reads the champion from the SageMaker data bucket.
#
# What IS needed is permission for callers. This is a customer-managed policy
# with exactly one action on exactly one endpoint; attach it to the role of
# whatever calls the endpoint (an application, a Lambda, a person's role).
# A policy, not a role, so no permissions boundary applies.
# =============================================================================

data "aws_iam_policy_document" "ml_v2_invoke_endpoint" {
  statement {
    sid     = "InvokeTrustScoreEndpoint"
    effect  = "Allow"
    actions = ["sagemaker:InvokeEndpoint"]

    resources = [
      "arn:${data.aws_partition.current.partition}:sagemaker:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:endpoint/${local.ml_v2_name}-serverless"
    ]
  }
}

resource "aws_iam_policy" "ml_v2_invoke_endpoint" {
  name        = "${local.ml_v2_name}-invoke-endpoint"
  description = "VERSION 2: invoke the Trust Score Serverless inference endpoint, and nothing else."
  policy      = data.aws_iam_policy_document.ml_v2_invoke_endpoint.json
}

output "ml_v2_invoke_endpoint_policy_arn" {
  description = "VERSION 2: attach to any role that needs to call the inference endpoint."
  value       = aws_iam_policy.ml_v2_invoke_endpoint.arn
}
