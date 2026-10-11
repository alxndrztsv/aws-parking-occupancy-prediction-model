locals {
  gold_splits_prefix = "s3://${aws_s3_bucket.layers["gold"].id}/splits"

  # Uploaded by the CD workflow
  training_code_uri = "s3://${aws_s3_bucket.layers["artifacts"].id}/code/train.tar.gz"
  checkpoints_uri   = "s3://${aws_s3_bucket.layers["artifacts"].id}/checkpoints/"

  # Winners promoted by the SelectModel step. The next training job re-scores these
  # on its own validation split, which is what makes the comparison valid.
  historical_models_uri = "s3://${aws_s3_bucket.layers["artifacts"].id}/historical_models/"

  # The bucket is known at plan time, only the run id is not. Both templates bake
  # the literal into the single-quoted format string: States.Format accepts only
  # JSONPath references or intrinsic calls as arguments, so a literal passed as a
  # second argument (quoted or not) is rejected with SCHEMA_VALIDATION_FAILED.
  metrics_uri_template = "States.Format('s3://${aws_s3_bucket.layers["artifacts"].id}/metrics/{}/', $.date_config.run_id)"
  training_job_name    = "States.Format('${local.prefix}-{}', $.date_config.run_id)"
}

# Create IAM Role for Step functions
resource "aws_iam_role" "sfn_role" {
  name = "${local.prefix}-sfn-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "states.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "sfn_policy" {
  name = "${local.prefix}-sfn-policy"
  role = aws_iam_role.sfn_role.id

  policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Effect = "Allow"
        Action = [
          "glue:GetJob",
          "glue:StartJobRun",
          "glue:GetJobRun",
          "glue:GetJobRuns",
          "glue:BatchStopJobRun",
        ]
        Resource = [
          aws_glue_job.bronze_to_silver.arn,
          aws_glue_job.silver_to_gold.arn
        ]
      },
      {
        Effect = "Allow"
        Action = "lambda:InvokeFunction"
        Resource = [
          aws_lambda_function.fetch_data.arn,
          aws_lambda_function.select_model.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:StopTrainingJob",
          "sagemaker:AddTags"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "events:PutRule",
          "events:PutTargets",
          "events:DescribeRule",
          "events:RemoveTargets",
          "events:DeleteRule"
        ]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = "iam:PassRole"
        Resource = aws_iam_role.sagemaker_role.arn
        Condition = {
          StringEquals = {
            "iam:PassedToService" = "sagemaker.amazonaws.com"
          }
        }
      },
      # Allow publishing to SNS for pipeline failures
      {
        Effect   = "Allow"
        Action   = "sns:Publish"
        Resource = aws_sns_topic.pipeline_alerts.arn
      }
    ]
  })
}

# State Functions state machine definition
resource "aws_sfn_state_machine" "pipeline_orchestrator" {
  name     = "${local.prefix}-pipeline-orchestrator"
  role_arn = aws_iam_role.sfn_role.arn

  definition = jsonencode({
    Comment        = "Full pipeline: Fetch -> Bronze -> Silver -> Gold -> Train -> SelectModel"
    TimeoutSeconds = 21600
    StartAt        = "FetchData"

    States = {
      # --- 1 step: Invoke lambda to fetch data ---
      FetchData = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"

        Parameters = {
          FunctionName = aws_lambda_function.fetch_data.arn
        }
        ResultSelector = {
          "year.$"   = "$.Payload.year"
          "month.$"  = "$.Payload.month"
          "run_id.$" = "$.Payload.run_id"
        }
        ResultPath = "$.date_config"
        Retry = [
          {
            ErrorEquals     = ["States.ALL"]
            IntervalSeconds = 30
            MaxAttempts     = 2
            BackoffRate     = 2.0
          }
        ]
        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            Next        = "PipelineFailure"
          }
        ]
        Next = "BronzeToSilver"
      }
      # --- 2 step: Bronze to Silver Glue job ---
      BronzeToSilver = {
        Type     = "Task"
        Resource = "arn:aws:states:::glue:startJobRun.sync"

        Parameters = {
          JobName = aws_glue_job.bronze_to_silver.name
          Arguments = {
            "--year.$"  = "$.date_config.year"
            "--month.$" = "$.date_config.month"
          }
        }

        ResultPath = null # discard Glue metadata
        Retry = [
          {
            ErrorEquals     = ["States.ALL"]
            IntervalSeconds = 60
            MaxAttempts     = 2
            BackoffRate     = 2.0
          }
        ]
        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            Next        = "PipelineFailure"
          }
        ]
        Next = "SilverToGold"
      }
      # --- 3 step: Silver to Gold Glue job ---
      SilverToGold = {
        Type     = "Task"
        Resource = "arn:aws:states:::glue:startJobRun.sync"

        Parameters = {
          JobName = aws_glue_job.silver_to_gold.name
        }

        ResultPath = null
        Retry = [
          {
            ErrorEquals     = ["States.ALL"]
            IntervalSeconds = 60
            MaxAttempts     = 2
            BackoffRate     = 2.0
          }
        ]
        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            Next        = "PipelineFailure"
          }
        ]
        Next = "TrainModel"
      }
      # --- 4 step: Sagemaker training job ---
      TrainModel = {
        Type     = "Task"
        Resource = "arn:aws:states:::sagemaker:createTrainingJob.sync"

        Parameters = {
          "TrainingJobName.$" = local.training_job_name # unique per execution

          AlgorithmSpecification = {
            TrainingImage     = data.aws_sagemaker_prebuilt_ecr_image.sklearn.registry_path
            TrainingInputMode = "File"
          }

          RoleArn = aws_iam_role.sagemaker_role.arn

          InputDataConfig = [
            for split in ["train", "valid", "test"] : {
              ChannelName = split
              DataSource = {
                S3DataSource = {
                  S3DataType             = "S3Prefix"
                  S3Uri                  = "${local.gold_splits_prefix}/${split}/"
                  S3DataDistributionType = "FullyReplicated"
                }
              }
            }
          ]

          OutputDataConfig = {
            S3OutputPath = "s3://${aws_s3_bucket.layers["artifacts"].id}/models/"
          }

          ResourceConfig = {
            InstanceType   = var.sagemaker_instance_type
            InstanceCount  = 1 # training is single-node
            VolumeSizeInGB = var.sagemaker_volume_size_gb
          }

          StoppingCondition = {
            MaxRuntimeInSeconds  = var.sagemaker_max_runtime_seconds
            MaxWaitTimeInSeconds = var.sagemaker_max_runtime_seconds
          }

          # Managed spot is up to ~70% cheaper
          EnableManagedSpotTraining = true
          CheckpointConfig = {
            LocalPath = "/opt/ml/checkpoints"
            S3Uri     = local.checkpoints_uri
          }

          HyperParameters = {
            "sagemaker_program"          = "train.py"
            "sagemaker_submit_directory" = local.training_code_uri
            "sagemaker_requirements"     = "requirements.txt"
            "run-id.$"                   = "$.date_config.run_id"
            "metrics-s3-uri.$"           = local.metrics_uri_template
            "historical-models-uri"      = local.historical_models_uri
            "artifacts-bucket"           = aws_s3_bucket.layers["artifacts"].id
          }
        }

        ResultPath = "$.training_result"
        Retry = [
          {
            ErrorEquals     = ["SageMaker.AmazonSageMakerException"]
            IntervalSeconds = 60
            MaxAttempts     = 2
            BackoffRate     = 2.0
          }
        ]
        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            Next        = "PipelineFailure"
          }
        ]
        Next = "SelectModel"
      }
      # --- 5 step: pick the winning model per horizon ---
      SelectModel = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"

        Parameters = {
          FunctionName = aws_lambda_function.select_model.arn
          Payload = {
            "run_id.$"                 = "$.date_config.run_id"
            "metrics_s3_uri.$"         = local.metrics_uri_template
            "model_artifacts_s3_uri.$" = "$.training_result.ModelArtifacts.S3ModelArtifacts"
          }
        }

        ResultPath = "$.selection"
        Retry = [
          {
            ErrorEquals     = ["States.ALL"]
            IntervalSeconds = 10
            MaxAttempts     = 2
            BackoffRate     = 2.0
          }
        ]
        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            Next        = "PipelineFailure"
          }
        ]
        Next = "PipelineSuccess"
      }

      # Handle successful completion
      PipelineSuccess = {
        Type = "Succeed"
      }

      # Handle pipeline failure
      PipelineFailure = {
        Type  = "Fail"
        Error = "PipelineExecutionFailed"
        Cause = "Check CloudWatch logs for details."
      }
    }
  })
}

output "sfn_arn" {
  description = "ARN of the Step Functions state machine"
  value       = aws_sfn_state_machine.pipeline_orchestrator.arn
}
