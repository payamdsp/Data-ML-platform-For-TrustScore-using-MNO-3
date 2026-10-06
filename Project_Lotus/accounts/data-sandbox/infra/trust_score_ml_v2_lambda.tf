# =============================================================================
# VERSION 2 - Trust Score ML pipeline: helper Lambda, cooldown clock, alerts
# =============================================================================
# Added in v2; no existing file in this root was modified. See ../CHANGES_V2.md.
#
# The helper does what Step Functions cannot: builds the run id, the job names
# and each job's hyperparameters, reads the batch date out of an S3 key, and
# decides whether a drift retrain is inside the cooldown. It never starts a
# job itself. Source: lambda_v2/pipeline_helper.py (same file as
# sagemaker/terraform/lambda/, unit-tested by sagemaker/tests/test_pipeline_helper.py).
# Its role is in ../security/trust_score_ml_v2_iam.tf.
# =============================================================================

data "archive_file" "ml_v2_helper" {
  type        = "zip"
  source_file = "${path.module}/lambda_v2/pipeline_helper.py"
  output_path = "${path.module}/lambda_v2/build/pipeline_helper.zip"
}

# When the last training run started: written by the helper at the start of
# every run, read by the drift retrain gate. Terraform only creates it.
resource "aws_ssm_parameter" "ml_v2_last_training_start" {
  name        = "/${var.project}/${var.environment}/trust-score-ml/last-training-start"
  description = "UTC start time of the most recent Trust Score ML training run (written by the pipeline)."
  type        = "String"
  value       = "1970-01-01T00:00:00+00:00"

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_cloudwatch_log_group" "ml_v2_helper" {
  name              = "/aws/lambda/${local.ml_v2_name}-helper"
  retention_in_days = 14
}

resource "aws_lambda_function" "ml_v2_helper" {
  function_name = "${local.ml_v2_name}-helper"
  description   = "Trust Score ML pipeline: run ids, job names and arguments, retrain cooldown."

  role    = data.aws_iam_role.ml_v2_helper.arn
  runtime = "python3.12"
  handler = "pipeline_helper.handler"

  architectures = ["x86_64"]

  filename         = data.archive_file.ml_v2_helper.output_path
  source_code_hash = data.archive_file.ml_v2_helper.output_base64sha256

  # Pure string/date work plus one SSM call.
  memory_size = 128
  timeout     = 30

  environment {
    variables = {
      JOB_NAME_PREFIX = var.ml_v2_job_name_prefix
      CONFIG_FILES    = join(",", var.ml_v2_config_files)
      OVERRIDES = jsonencode([
        "data.training_root=s3://${local.ml_v2_input_bucket}/${var.ml_v2_training_data_prefix}",
        "data.testing_root=s3://${local.ml_v2_input_bucket}/${var.ml_v2_scoring_data_prefix}",
        "data.feature_selection_root=${local.ml_v2_output_root}",
        "data.models_root=${local.ml_v2_output_root}"
      ])
      MODELS                  = var.ml_v2_models
      ARMS_PER_MODEL          = tostring(var.ml_v2_arms_per_model)
      SLICE_COUNT             = tostring(var.ml_v2_sweep_slice_count)
      ALLOW_FAILED_CHECKS     = tostring(var.ml_v2_allow_failed_checks)
      TOP_K                   = tostring(var.ml_v2_review_queue_top_k)
      OUTPUT_BUCKET           = local.ml_v2_output_bucket
      OUTPUT_PREFIX           = var.ml_v2_output_prefix
      RETRAIN_COOLDOWN_DAYS   = tostring(var.ml_v2_retrain_cooldown_days)
      LAST_TRAINING_PARAMETER = aws_ssm_parameter.ml_v2_last_training_start.name
    }
  }

  depends_on = [aws_cloudwatch_log_group.ml_v2_helper]
}

# -----------------------------------------------------------------------------
# Alerts: its own topic, following sns_pipeline_alerts.tf (one topic per
# audience). Training finished / failed, scoring failed, drift detected.
# -----------------------------------------------------------------------------

resource "aws_sns_topic" "ml_v2_alerts" {
  name = "${var.project}-${var.environment}-ml-pipeline-alerts"
}

# Each address receives a confirmation email and gets nothing until it is
# confirmed; Terraform cannot confirm it.
resource "aws_sns_topic_subscription" "ml_v2_alert_email" {
  for_each = toset(var.ml_v2_alert_emails)

  topic_arn = aws_sns_topic.ml_v2_alerts.arn
  protocol  = "email"
  endpoint  = each.value
}
