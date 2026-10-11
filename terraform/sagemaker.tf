# Sagemaker IAM execution role
resource "aws_iam_role" "sagemaker_role" {
  name = "${local.prefix}-sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "sagemaker_policy" {
  name = "${local.prefix}-sagemaker-policy"
  role = aws_iam_role.sagemaker_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject", # gold bucket (features)
          "s3:PutObject", # artifacts bucket (models)
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        # Separate bucket ARNs from object ARNs
        Resource = concat(
          [for k, b in aws_s3_bucket.layers : b.arn if contains(["gold", "artifacts"], k)],
          [for k, b in aws_s3_bucket.layers : "${b.arn}/*" if contains(["gold", "artifacts"], k)]
        )
      },
      # CloudWatch Logs permissions for training job output
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# Docker container.
# image_tag must resolve to a Python 3.10+ image: the training code uses PEP 604
# unions (float | None) in evaluated annotations and xgboost 3.x requires >=3.10,
# so the older 1.2-1 line (Python 3.8/3.9) cannot import it. The provider's own
# default tag is "1", which sagemaker-scikit-learn does not publish, so this is
# never left unset.
data "aws_sagemaker_prebuilt_ecr_image" "sklearn" {
  repository_name = "sagemaker-scikit-learn"
  image_tag       = var.sagemaker_sklearn_image_tag
}

# Outputs
output "sagemaker_role_arn" {
  value = aws_iam_role.sagemaker_role.arn
}

output "sagemaker_training_image" {
  description = "Resolved training image URI. Confirm this tag exists in the region before the first training run."
  value       = data.aws_sagemaker_prebuilt_ecr_image.sklearn.registry_path
}

output "sagemaker_training_code_uri" {
  description = "S3 key the CD workflow must upload train.tar.gz to (sagemaker_submit_directory)"
  value       = local.training_code_uri
}

output "artifacts_bucket" {
  value = aws_s3_bucket.layers["artifacts"].id
}
