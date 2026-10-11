locals {
  # Define the layers for the Medallion Architecture
  # Bronze: Raw API data
  # Silver: Cleaned/Normalized data
  # Gold:   Feature-engineered data
  # Artifacts: Deployment scripts and ML Models
  bucket_layers = ["bronze", "silver", "gold", "artifacts"]
}

resource "aws_s3_bucket" "layers" {
  for_each = toset(local.bucket_layers)

  bucket        = "${local.prefix}-${each.key}"
  force_destroy = true
  # Insert some metadata
  tags = {
    Layer = title(each.key) # capitalizes the layer name (Bronze, Silver, etc.)
  }
}

# Enforce ownership controls to disable ACLs
resource "aws_s3_bucket_ownership_controls" "layers" {
  for_each = aws_s3_bucket.layers

  bucket = each.value.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_policy" "layers" {
  for_each = aws_s3_bucket.layers

  bucket = each.value.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "DenyUnencryptedTransport"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource = [
          each.value.arn,
          "${each.value.arn}/*"
        ]
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      }
    ]
  })
}

resource "aws_s3_bucket_versioning" "layers" {
  for_each = aws_s3_bucket.layers

  bucket = each.value.id # point to previous bucket
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "layers" {
  for_each = aws_s3_bucket.layers

  bucket = each.value.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Ensure buckets cannot be accidentally made public
resource "aws_s3_bucket_public_access_block" "layers" {
  for_each = aws_s3_bucket.layers

  bucket = each.value.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- Bronze ---
# Bronze: Move raw data to cheaper storage after 1 year
resource "aws_s3_bucket_lifecycle_configuration" "bronze" {
  bucket = aws_s3_bucket.layers["bronze"].id

  rule {
    id     = "bronze-retention"
    status = "Enabled"
    filter {}
    transition {
      days          = 365
      storage_class = "STANDARD_IA"
    }
  }
}

# Enable EventBridge notifications on the Bronze bucket
resource "aws_s3_bucket_notification" "bronze_notification" {
  bucket      = aws_s3_bucket.layers["bronze"].id
  eventbridge = true # allows to trigger bronze_to_silver when lambda uploads CSV
}

# --- Artifacts ---
# Per-run outputs expire; the retained winners do not. Note that historical_models/
# is bounded on its own: the key is "<model>_<horizon>.joblib", so at most 16 bundles
# ever live there (4 models x 4 horizons) and a new winner overwrites its slot.
resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.layers["artifacts"].id

  # SageMaker's own output: one model.tar.gz per training job, keyed by a unique
  # job name. Keys are never reused, so noncurrent_version_expiration alone would
  # never fire - the current-version expiration is what actually reclaims space.
  rule {
    id     = "cleanup-old-models"
    status = "Enabled"
    filter {
      prefix = "models/"
    }
    expiration {
      days = 90
    }
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }

  # Every candidate bundle of every run. Only the promoted winners matter after the
  # SelectModel step has run, so 90 days is ample for debugging a bad selection.
  rule {
    id     = "cleanup-old-current-models"
    status = "Enabled"
    filter {
      prefix = "current_models/"
    }
    expiration {
      days = 90
    }
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }

  rule {
    id     = "cleanup-old-metrics"
    status = "Enabled"
    filter {
      prefix = "metrics/"
    }
    expiration {
      days = 365
    }
    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }

  # The retained winners and their registry. No current-version expiration: this is
  # the production model set. best_models.json is overwritten every run, so its
  # superseded versions are still worth reclaiming.
  rule {
    id     = "cleanup-historical-versions"
    status = "Enabled"
    filter {
      prefix = "historical_models/"
    }
    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }

  rule {
    id     = "cleanup-spot-checkpoints"
    status = "Enabled"
    filter {
      prefix = "checkpoints/"
    }
    expiration {
      days = 30
    }
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

resource "aws_s3_object" "glue_scripts" {
  for_each = fileset("${path.module}/../glue_scripts", "*.py")

  bucket = aws_s3_bucket.layers["artifacts"].id
  key    = "glue/${each.value}"
  source = "${path.module}/../glue_scripts/${each.value}"
  etag   = filemd5("${path.module}/../glue_scripts/${each.value}")
}
