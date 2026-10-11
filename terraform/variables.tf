# Define input variables used throughout Terraform configuration
variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
}

variable "project_name" {
  description = "Name of project, used for resource naming."
  type        = string

  validation {
    condition     = length(var.project_name) > 0
    error_message = "Must not be empty: it is part of every resource name."
  }
}

variable "environment" {
  description = "Environment (e.g., dev, staging, prod)."
  type        = string

  validation {
    condition     = length(var.environment) > 0
    error_message = "Must not be empty: it is part of every resource name."
  }
}

# --- Email notifications ---
variable "failure_notification_email" {
  description = "Email address for pipeline alerts"
  type        = string

  validation {
    condition     = can(regex("^[\\w-\\.]+@([\\w-]+\\.)+[\\w-]{2,4}$", var.failure_notification_email))
    error_message = "Must be a valid email address."
  }
}

# --- API configuration ---
variable "api_login" {
  description = "API login username"
  type        = string
  sensitive   = true
}

variable "api_password" {
  description = "API password"
  type        = string
  sensitive   = true
}

variable "api_url" {
  description = "API endpoint URL"
  type        = string
}

variable "api_city_code" {
  description = "API city code"
  type        = string
}

# --- Business logic ---
variable "bank_holidays" {
  description = "Comma-separated list of Irish bank holidays (YYYY-MM-DD). Closed days are forced to zero occupancy, so this must cover every year present in the bronze layer."
  type        = string
  default     = "2024-01-01,2024-02-05,2024-03-18,2024-04-01,2024-05-06,2024-06-03,2024-08-05,2024-10-28,2024-12-25,2024-12-26,2025-01-01,2025-02-03,2025-03-17,2025-04-21,2025-05-05,2025-06-02,2025-08-04,2025-10-27,2025-12-25,2025-12-26,2026-01-01,2026-02-02,2026-03-17,2026-04-06,2026-05-04,2026-06-01,2026-08-03,2026-10-26,2026-12-25,2026-12-26"
}

variable "horizons_minutes" {
  description = "Forecasting horizons to generate targets for (name:minutes). The names become the y_<name> columns and must match HORIZONS_MINUTES in src/config.py."
  type        = string
  default     = "h1h:60,h2h:120,h3h:180,h24h:1440"
}

variable "model_selection_metric" {
  description = "Metric column of model_metrics.csv used to pick the best model per horizon."
  type        = string
  default     = "valid_wape"

  validation {
    condition     = contains(["valid_wape", "valid_mae", "valid_rmse", "valid_mase"], var.model_selection_metric)
    error_message = "Must be one of the validation-split columns of model_metrics.csv: valid_wape, valid_mae, valid_rmse, valid_mase."
  }
}

# --- Sagemaker configuration ---
variable "sagemaker_instance_type" {
  description = "Sagemaker training instance type."
  type        = string
  default     = "ml.c5.4xlarge"
}

variable "sagemaker_volume_size_gb" {
  description = "EBS volume attached to the training instance, in GB. File mode downloads all three splits to it before training starts."
  type        = number
  default     = 50
}

variable "sagemaker_max_runtime_seconds" {
  description = "Sagemaker StoppingCondition for the training job, in seconds. Also used as MaxWaitTimeInSeconds, which is what a spot interruption is allowed to add."
  type        = number
  default     = 14400 # 4 hours
}

variable "sagemaker_sklearn_image_tag" {
  description = <<-EOT
    Tag of the sagemaker-scikit-learn prebuilt image, resolved through
    data.aws_sagemaker_prebuilt_ecr_image. Must be a Python 3.10+ image: the
    training code uses PEP 604 unions in evaluated annotations and xgboost 3.x
    requires >=3.10. Verify the tag exists in your region before the first run -
    "terraform output sagemaker_training_image" prints the resolved URI.
  EOT
  type        = string
  default     = "1.4-2-cpu-py312"

  validation {
    condition     = can(regex("^[0-9]", var.sagemaker_sklearn_image_tag))
    error_message = "Must be an explicit framework tag such as \"1.4-2-cpu-py3\"; the provider default \"1\" is not published for sagemaker-scikit-learn."
  }
}
