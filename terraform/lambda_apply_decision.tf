resource "aws_iam_role" "lambda_apply_decision_role" {
  name = "${local.prefix}-lambda-apply-decision-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "lambda_apply_decision_policy" {
  name = "${local.prefix}-lambda-apply-decision-policy"
  role = aws_iam_role.lambda_apply_decision_role.id
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
      # Reads pending.json and the registry, copies the approved bundle into
      # historical_models/, and writes the new registry plus its rollback copy.
      {
        Effect = "Allow"
        Action = ["s3:GetObject"]
        Resource = [
          "${aws_s3_bucket.layers["artifacts"].arn}/${local.historical_models_prefix}*",
          "${aws_s3_bucket.layers["artifacts"].arn}/${local.current_models_prefix}*"
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = "${aws_s3_bucket.layers["artifacts"].arn}/${local.historical_models_prefix}*"
      }
    ]
  })
}

data "archive_file" "lambda_apply_decision_zip" {
  type        = "zip"
  source_file = "${path.module}/../lambda_functions/lambda_apply_decision.py"
  output_path = "${path.module}/build/lambda_apply_decision.zip"
}

resource "aws_lambda_function" "apply_decision" {
  filename         = data.archive_file.lambda_apply_decision_zip.output_path
  function_name    = "${local.prefix}-apply-decision"
  role             = aws_iam_role.lambda_apply_decision_role.arn
  handler          = "lambda_apply_decision.lambda_handler"
  runtime          = "python3.11"
  source_code_hash = data.archive_file.lambda_apply_decision_zip.output_base64sha256
  timeout          = 60
  memory_size      = 256
  architectures    = ["x86_64"]

  environment {
    variables = {
      ARTIFACTS_BUCKET = aws_s3_bucket.layers["artifacts"].id
    }
  }

  depends_on = [aws_cloudwatch_log_group.apply_decision]
}

# Public HTTPS endpoint for the link in the model_decisions email, without an API
# Gateway in between. auth NONE is deliberate: the caller is a human clicking a link
# from a mailbox and cannot SigV4-sign a request. Access control is the single-use
# token in pending.json, plus the GET/POST split that stops mail scanners acting on it.
resource "aws_lambda_function_url" "apply_decision" {
  function_name      = aws_lambda_function.apply_decision.function_name
  authorization_type = "NONE"
  invoke_mode        = "BUFFERED"
}

# AuthType NONE is not enough on its own.
# This resource is needed for AuthType NONE to work properly.
resource "aws_lambda_permission" "apply_decision_url_invoke" {
  statement_id  = "FunctionURLInvokeAllowPublicAccess"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.apply_decision.function_name
  principal     = "*"
}

output "model_approval_url" {
  description = "Public base URL used to build the approve/reject link in the email"
  value       = aws_lambda_function_url.apply_decision.function_url
}

output "lambda_apply_decision_name" {
  value = aws_lambda_function.apply_decision.function_name
}
