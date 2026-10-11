resource "aws_sns_topic" "pipeline_alerts" {
  name = "${local.prefix}-pipeline-alerts"
}

resource "aws_sns_topic_subscription" "email_alerts" {
  topic_arn = aws_sns_topic.pipeline_alerts.arn
  protocol  = "email"
  endpoint  = var.failure_notification_email
}

output "sns_topic_arn" {
  description = "ARN of the pipeline alerts SNS topic"
  value       = aws_sns_topic.pipeline_alerts.arn
}
