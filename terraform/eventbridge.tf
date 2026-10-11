resource "aws_iam_role" "eventbridge_role" {
  name = "${local.prefix}-eventbridge-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "events.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "eventbridge_policy" {
  name = "${local.prefix}-eventbridge-policy"
  role = aws_iam_role.eventbridge_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "states:StartExecution"
        Resource = aws_sfn_state_machine.pipeline_orchestrator.arn
      },
      {
        Effect   = "Allow"
        Action   = "sns:Publish"
        Resource = aws_sns_topic.pipeline_alerts.arn
      }
    ]
  })
}

# Step functions monthly schedule trigger
resource "aws_cloudwatch_event_rule" "monthly_pipeline" {
  name                = "${var.project_name}-monthly-pipeline"
  description         = "Triggers the pipeline monthly at 06:00 UTC on the third day"
  schedule_expression = "cron(0 6 3 * ? *)"
}

# EventBridge target (points to Step Functions machine state)
resource "aws_cloudwatch_event_target" "sfn_target" {
  rule      = aws_cloudwatch_event_rule.monthly_pipeline.name
  target_id = "StepFunctionTarget"
  arn       = aws_sfn_state_machine.pipeline_orchestrator.arn
  role_arn  = aws_iam_role.eventbridge_role.arn

  retry_policy {
    maximum_event_age_in_seconds = 3600
    maximum_retry_attempts       = 2
  }
}

# Step functions failure alerting
resource "aws_cloudwatch_event_rule" "pipeline_failed" {
  name        = "${local.prefix}-pipeline-failed"
  description = "Triggers when the Step Function fails, times out, or is aborted"

  event_pattern = jsonencode({
    source      = ["aws.states"]
    detail-type = ["Step Functions Execution Status Change"]
    detail = {
      status          = ["FAILED", "TIMED_OUT", "ABORTED"]
      stateMachineArn = [aws_sfn_state_machine.pipeline_orchestrator.arn]
    }
  })
}

resource "aws_cloudwatch_event_target" "failure_notification" {
  rule      = aws_cloudwatch_event_rule.pipeline_failed.name
  target_id = "SNSFailureTarget"
  arn       = aws_sns_topic.pipeline_alerts.arn

  input_transformer {
    input_paths = {
      execution_arn = "$.detail.executionArn"
      status        = "$.detail.status"
      start_date    = "$.detail.startDate"
    }

    input_template = "\"Pipeline execution failed! Status: <status> | Execution ARN: <execution_arn> | Started at: <start_date>. Check CloudWatch Logs for details.\""
  }
}

resource "aws_sns_topic_policy" "eventbridge_sns" {
  arn = aws_sns_topic.pipeline_alerts.arn

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "events.amazonaws.com"
        }
        Action   = "sns:Publish"
        Resource = aws_sns_topic.pipeline_alerts.arn
      }
    ]
  })
}
