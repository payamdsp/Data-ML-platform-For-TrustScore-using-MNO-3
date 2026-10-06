# =============================================================================
# VERSION 2 - Trust Score ML pipeline: "new data arrived" -> start a pipeline
# =============================================================================
# Added in v2; no existing file in this root was modified. See ../CHANGES_V2.md.
#
# The gold bucket sends object-created events to EventBridge; the two rules keep
# only COMPLETION MARKERS (keys ending in ml_v2_trigger_marker_suffix) under the
# training and scoring prefixes. One uploaded batch = one marker = one run.
# Same mechanism as the Gold release trigger in silver_gold_control_lambda.tf,
# except the target is a state machine (started with the events role from
# ../security) rather than a Lambda.
# =============================================================================

# Owns ALL notification settings of the gold bucket. No other file in this repo
# manages them (checked for v2); if any were set up by hand in the console,
# they would be replaced - see ml_v2_manage_gold_bucket_notification.
resource "aws_s3_bucket_notification" "ml_v2_sagemaker_eventbridge" {
  count = var.ml_v2_manage_sagemaker_bucket_notification ? 1 : 0

  bucket      = aws_s3_bucket.sagemaker.id
  eventbridge = true
}

resource "aws_cloudwatch_event_rule" "ml_v2_training_data_arrived" {
  name        = "${local.ml_v2_name}-training-data-arrived"
  description = "VERSION 2: a training batch is complete (${var.ml_v2_trigger_marker_suffix}) under s3://${local.ml_v2_input_bucket}/${var.ml_v2_training_data_prefix}"

  event_pattern = jsonencode({
    source        = ["aws.s3"]
    "detail-type" = ["Object Created"]
    detail = {
      bucket = { name = [local.ml_v2_input_bucket] }
      object = { key = [{ wildcard = "${var.ml_v2_training_data_prefix}*${var.ml_v2_trigger_marker_suffix}" }] }
    }
  })

  depends_on = [aws_s3_bucket_notification.ml_v2_sagemaker_eventbridge]
}

resource "aws_cloudwatch_event_rule" "ml_v2_scoring_data_arrived" {
  name        = "${local.ml_v2_name}-scoring-data-arrived"
  description = "VERSION 2: a scoring batch is complete (${var.ml_v2_trigger_marker_suffix}) under s3://${local.ml_v2_input_bucket}/${var.ml_v2_scoring_data_prefix}"

  event_pattern = jsonencode({
    source        = ["aws.s3"]
    "detail-type" = ["Object Created"]
    detail = {
      bucket = { name = [local.ml_v2_input_bucket] }
      object = { key = [{ wildcard = "${var.ml_v2_scoring_data_prefix}*${var.ml_v2_trigger_marker_suffix}" }] }
    }
  })

  depends_on = [aws_s3_bucket_notification.ml_v2_sagemaker_eventbridge]
}

resource "aws_cloudwatch_event_target" "ml_v2_start_training" {
  rule      = aws_cloudwatch_event_rule.ml_v2_training_data_arrived.name
  target_id = "trust-score-ml-training"
  arn       = aws_sfn_state_machine.ml_v2_training.arn
  role_arn  = data.aws_iam_role.ml_v2_events.arn

  retry_policy {
    maximum_event_age_in_seconds = 3600
    maximum_retry_attempts       = 10
  }
}

resource "aws_cloudwatch_event_target" "ml_v2_start_scoring" {
  rule      = aws_cloudwatch_event_rule.ml_v2_scoring_data_arrived.name
  target_id = "trust-score-ml-scoring"
  arn       = aws_sfn_state_machine.ml_v2_scoring.arn
  role_arn  = data.aws_iam_role.ml_v2_events.arn

  retry_policy {
    maximum_event_age_in_seconds = 3600
    maximum_retry_attempts       = 10
  }
}

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

output "ml_v2_training_state_machine_arn" {
  description = "VERSION 2: start manually with aws stepfunctions start-execution --state-machine-arn <this> --input '{}'"
  value       = aws_sfn_state_machine.ml_v2_training.arn
}

output "ml_v2_scoring_state_machine_arn" {
  description = "VERSION 2: nightly scoring state machine."
  value       = aws_sfn_state_machine.ml_v2_scoring.arn
}

output "ml_v2_alerts_topic_arn" {
  description = "VERSION 2: ML pipeline alerts (training finished/failed, scoring failed, drift)."
  value       = aws_sns_topic.ml_v2_alerts.arn
}

output "ml_v2_training_trigger" {
  description = "VERSION 2: upload this marker LAST, in a training batch folder, to start training."
  value       = "s3://${local.ml_v2_input_bucket}/${var.ml_v2_training_data_prefix}<batch folder>/${var.ml_v2_trigger_marker_suffix}"
}

output "ml_v2_scoring_trigger" {
  description = "VERSION 2: upload this marker LAST, in a scoring batch folder, to score it."
  value       = "s3://${local.ml_v2_input_bucket}/${var.ml_v2_scoring_data_prefix}batch_date=YYYY-MM-DD/${var.ml_v2_trigger_marker_suffix}"
}

output "ml_v2_images" {
  description = "VERSION 2: the exact images (by digest) the pipeline runs. Re-apply after pushing new images."
  value       = { training = local.ml_v2_training_image, inference = local.ml_v2_inference_image }
}
