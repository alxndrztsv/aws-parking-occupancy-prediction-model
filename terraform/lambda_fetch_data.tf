# Create IAM Role for Lambda
resource "aws_iam_role" "lambda_fetch_data_role" {
  name = "${local.prefix}-lambda-fetch-data-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "lambda_fetch_data_policy" {
  name = "${local.prefix}-lambda-fetch-data-policy"
  role = aws_iam_role.lambda_fetch_data_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # Cloudwatch logs
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      # Cloudwatch metrics
      {
        Effect   = "Allow"
        Action   = ["cloudwatch:PutMetricData"]
        Resource = "*"
      },
      # SSM parameter store
      {
        Effect = "Allow"
        Action = ["ssm:GetParameter"]
        Resource = [
          aws_ssm_parameter.api_password.arn,
          aws_ssm_parameter.api_login.arn
        ]
      },
      # S3 bucket
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = "${aws_s3_bucket.layers["bronze"].arn}/*"
      }
    ]
  })
}

# Package lambda_fetch_data.py into a .zip file
# The function is stdlib + boto3 only, so it needs no dependency layer
data "archive_file" "lambda_fetch_data_zip" {
  type        = "zip"
  source_file = "${path.module}/../lambda_functions/lambda_fetch_data.py"
  output_path = "${path.module}/build/lambda_fetch_data.zip"
}

resource "aws_lambda_function" "fetch_data" {
  filename         = data.archive_file.lambda_fetch_data_zip.output_path
  function_name    = "${local.prefix}-fetch-data"
  role             = aws_iam_role.lambda_fetch_data_role.arn
  handler          = "lambda_fetch_data.lambda_handler"
  runtime          = "python3.11"
  source_code_hash = data.archive_file.lambda_fetch_data_zip.output_base64sha256
  timeout          = 90
  memory_size      = 512
  architectures    = ["x86_64"]

  environment {
    variables = {
      BRONZE_BUCKET         = aws_s3_bucket.layers["bronze"].id
      API_LOGIN_SSM_PATH    = aws_ssm_parameter.api_login.name
      API_PASSWORD_SSM_PATH = aws_ssm_parameter.api_password.name
      API_URL               = var.api_url
      API_CITY_CODE         = var.api_city_code
    }
  }

  depends_on = [aws_cloudwatch_log_group.fetch_data]
}

# --- Outputs ---
output "lambda_fetch_data_name" {
  value = aws_lambda_function.fetch_data.function_name
}
