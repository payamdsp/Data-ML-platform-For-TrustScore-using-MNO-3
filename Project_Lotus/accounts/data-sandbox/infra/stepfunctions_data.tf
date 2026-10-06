# Step Functions execution roles are created in ../security and looked up here,
# the same split the Lambda and EMR roles already use: security owns every
# aws_iam_role so the permissions boundary is applied in one root.
#
# Apply ../security before this root.

data "aws_iam_role" "sfn_bronze_silver" {
  name = "${var.project}-${var.environment}-trust-score-bronze-silver-role"
}

data "aws_iam_role" "sfn_gold" {
  name = "${var.project}-${var.environment}-trust-score-gold-role"
}

data "aws_iam_role" "sfn_bronze_arrival" {
  name = "${var.project}-${var.environment}-bronze-arrival-role"
}

data "aws_iam_role" "bronze_arrival_claim" {
  name = "${var.project}-${var.environment}-bronze-arrival-claim-role"
}

data "aws_iam_role" "eventbridge_bronze_arrival" {
  name = "${var.project}-${var.environment}-bronze-arrival-eventbridge-role"
}

# Assumed by EventBridge (not by the state machine itself) to start the Gold
# state machine directly as a StartExecution target - see
# silver_gold_control_lambda.tf's aws_cloudwatch_event_target.gold_release.
data "aws_iam_role" "eventbridge_gold_trigger" {
  name = "${var.project}-${var.environment}-trust-score-gold-eventbridge-trigger-role"
}
