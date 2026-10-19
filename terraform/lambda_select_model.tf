# Create IAM Role for Lambda
resource "aws_iam_role" "lambda_select_model_role" {
  name = "${local.prefix}-lambda-select-model-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "lambda_select_model_policy" {
  name = "${local.prefix}-lambda-select-model-policy"
  role = aws_iam_role.lambda_select_model_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      # Reads model_metrics.csv from metrics/ and writes pending.json into
      # historical_models/ plus an audit copy next to the metrics. It no longer
      # reads current_models/: promotion moved to lambda_apply_decision.
      {
        Effect = "Allow"
        Action = ["s3:GetObject"]
        Resource = [
          "${aws_s3_bucket.layers["artifacts"].arn}/${local.metrics_prefix}*",
          "${aws_s3_bucket.layers["artifacts"].arn}/${local.historical_models_prefix}*"
        ]
      },
      {
        Effect = "Allow"
        Action = ["s3:PutObject"]
        Resource = [
          "${aws_s3_bucket.layers["artifacts"].arn}/${local.metrics_prefix}*",
          "${aws_s3_bucket.layers["artifacts"].arn}/${local.historical_models_prefix}*"
        ]
      },
      {
        Effect   = "Allow"
        Action   = "sns:Publish"
        Resource = aws_sns_topic.model_decision.arn
      }
    ]
  })
}

data "archive_file" "lambda_select_model_zip" {
  type        = "zip"
  source_file = "${path.module}/../lambda_functions/lambda_select_model.py"
  output_path = "${path.module}/build/lambda_select_model.zip"
}

resource "aws_lambda_function" "select_model" {
  filename         = data.archive_file.lambda_select_model_zip.output_path
  function_name    = "${local.prefix}-select-model"
  role             = aws_iam_role.lambda_select_model_role.arn
  handler          = "lambda_select_model.lambda_handler"
  runtime          = "python3.11"
  source_code_hash = data.archive_file.lambda_select_model_zip.output_base64sha256
  timeout          = 60
  memory_size      = 256
  architectures    = ["x86_64"]

  environment {
    variables = {
      SELECTION_METRIC         = var.model_selection_metric
      HORIZONS_MINUTES         = var.horizons_minutes
      PROMOTION_THRESHOLD      = var.model_promotion_min_improvement
      LINK_TTL_DAYS            = var.model_approval_link_ttl_days
      MODEL_DECISION_TOPIC_ARN = aws_sns_topic.model_decision.arn
      APPROVAL_URL             = aws_lambda_function_url.apply_decision.function_url
    }
  }

  depends_on = [aws_cloudwatch_log_group.select_model]
}

# --- Outputs ---
output "lambda_select_model_name" {
  value = aws_lambda_function.select_model.function_name
}
