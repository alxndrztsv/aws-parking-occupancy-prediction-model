# --- Lambda alarms ---
# Lambda throttle alarm
resource "aws_cloudwatch_metric_alarm" "ingest_lambda_throttle" {
  alarm_name          = "${local.prefix}-lambda-ingest-data-throttled"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "Throttles"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 1
  alarm_description   = "Ingest Lambda is being throttled"
  alarm_actions       = [aws_sns_topic.pipeline_alerts.arn]

  dimensions = {
    FunctionName = aws_lambda_function.fetch_data.function_name
  }
}

# ===== Logs =====
# For Lambda create explicit log groups
resource "aws_cloudwatch_log_group" "fetch_data" {
  name              = "/aws/lambda/${local.prefix}-fetch-data"
  retention_in_days = 365
}

resource "aws_cloudwatch_log_group" "select_model" {
  name              = "/aws/lambda/${local.prefix}-select-model"
  retention_in_days = 365
}

resource "aws_cloudwatch_log_group" "apply_decision" {
  name              = "/aws/lambda/${local.prefix}-apply-decision"
  retention_in_days = 365
}
