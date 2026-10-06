# Separate alert topics per layer.
#
# SNS bills per publish and per delivery, not per topic, so three topics cost
# the same as one carrying the same traffic. The reason to split them is
# subscription routing: Bronze schema drift and a Gold publish failure have
# different audiences, and separate topics express that without every
# subscriber needing a message-attribute filter.

locals {
  pipeline_alert_topics = {
    bronze = "${var.project}-${var.environment}-bronze-validation-alerts"
    silver = "${var.project}-${var.environment}-silver-pipeline-alerts"
    gold   = "${var.project}-${var.environment}-gold-pipeline-alerts"
  }
}

resource "aws_sns_topic" "pipeline_alerts" {
  for_each = local.pipeline_alert_topics

  name = each.value
}

output "pipeline_alert_topic_arns" {
  description = "Per-layer SNS alert topic ARNs."
  value       = { for key, topic in aws_sns_topic.pipeline_alerts : key => topic.arn }
}
