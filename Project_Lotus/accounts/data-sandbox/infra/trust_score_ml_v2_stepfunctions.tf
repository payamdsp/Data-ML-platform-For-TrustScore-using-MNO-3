# =============================================================================
# VERSION 2 - Trust Score ML pipeline: the two Step Functions state machines
# =============================================================================
# Added in v2; no existing file in this root was modified. See ../CHANGES_V2.md.
#
# TRAINING  (new training batch marker, or a drift retrain)
#   CheckAlreadyRunning -> BuildContext -> Prepare -> Sweep (Map, N slices in
#   parallel) -> CountFailedSlices -> Finalize -> NotifySuccess
#     Prepare  = train.py --mode prepare   data-quality gates + feature selection
#     Sweep    = train.py --mode sweep     one SageMaker job per slice, Spot
#     Finalize = train.py --mode finalize  merge the slices, publish the champion
#   Every job of one run shares TS05_RUN_ID, which is what lets finalize merge
#   the slices' outputs.
#
# SCORING   (new scoring batch marker)
#   BuildContext -> Score (Processing job) -> ReadSummary -> DriftDetected?
#   -> NotifyDrift -> [RetrainGate -> StartRetraining]   (ml_v2_auto_retrain_on_drift)
#
# Failure handling: transient SageMaker/Lambda errors are retried; a job that
# ran and FAILED is not retried automatically (re-running is a decision), and
# every failure path publishes to the ML alerts topic before failing.
# =============================================================================

# Optional pieces below are written `cond ? { ... } : null` inside merge().
# merge() skips null arguments, and null is the only "empty" value whose type
# unifies with an object that has attributes - `cond ? { a = 1 } : {}` is a type
# error ("true and false result expressions must have consistent types").

locals {
  ml_v2_hours = 3600

  ml_v2_models_root = "s3://${local.ml_v2_output_bucket}/${var.ml_v2_output_prefix}"
  ml_v2_job_environment = {
    "TS05_RUN_ID.$"    = "$.context.run_id"
    AWS_REGION         = data.aws_region.current.region
    AWS_DEFAULT_REGION = data.aws_region.current.region
  }

  ml_v2_sagemaker_retry = [{
    ErrorEquals     = ["SageMaker.AmazonSageMakerException", "SageMaker.ResourceLimitExceededException", "ThrottlingException"]
    IntervalSeconds = 60
    MaxAttempts     = 3
    BackoffRate     = 2
  }]

  ml_v2_lambda_retry = [{
    ErrorEquals     = ["Lambda.ServiceException", "Lambda.AWSLambdaException", "Lambda.SdkClientException", "Lambda.TooManyRequestsException"]
    IntervalSeconds = 2
    MaxAttempts     = 3
    BackoffRate     = 2
  }]

  ml_v2_catch_all = [{ ErrorEquals = ["States.ALL"], ResultPath = "$.error", Next = "NotifyFailure" }]

  ml_v2_job_result = {
    "job.$"    = "$.TrainingJobName"
    "status.$" = "$.TrainingJobStatus"
  }

  # One CreateTrainingJob request per job kind; only the name/hyperparameter
  # paths, the instance and the time limit differ.
  ml_v2_training_job = {
    for kind, spec in {
      prepare = {
        name = "$.context.prepare.job_name", params = "$.context.prepare.hyperparameters"
        type = var.ml_v2_prepare_instance_type, hours = var.ml_v2_max_runtime_hours.prepare
        env  = local.ml_v2_job_environment
      }
      sweep = {
        name = "$.slice.job_name", params = "$.slice.hyperparameters"
        type = var.ml_v2_sweep_instance_type, hours = var.ml_v2_max_runtime_hours.sweep
        env  = merge(local.ml_v2_job_environment, { "TS05_RUN_ID.$" = "$.run_id" })
      }
      finalize = {
        name = "$.context.finalize.job_name", params = "$.context.finalize.hyperparameters"
        type = var.ml_v2_finalize_instance_type, hours = var.ml_v2_max_runtime_hours.finalize
        env  = local.ml_v2_job_environment
      }
      } : kind => merge(
      {
        "TrainingJobName.$" = spec.name
        "HyperParameters.$" = spec.params
        AlgorithmSpecification = {
          TrainingImage     = local.ml_v2_training_image
          TrainingInputMode = "File"
        }
        RoleArn     = data.aws_iam_role.ml_v2_sagemaker_execution.arn
        Environment = spec.env
        OutputDataConfig = {
          S3OutputPath = "${local.ml_v2_output_root}sagemaker-output/training/"
        }
        ResourceConfig = {
          InstanceType   = spec.type
          InstanceCount  = 1
          VolumeSizeInGB = var.ml_v2_training_volume_size_gb
        }
        StoppingCondition = merge(
          { MaxRuntimeInSeconds = spec.hours * local.ml_v2_hours },
          var.ml_v2_use_spot_training ? { MaxWaitTimeInSeconds = max(spec.hours, var.ml_v2_spot_max_wait_hours) * local.ml_v2_hours } : null,
        )
        EnableManagedSpotTraining = var.ml_v2_use_spot_training
        Tags                      = local.ml_v2_job_tags
      },
      local.ml_v2_vpc_config != null ? { VpcConfig = local.ml_v2_vpc_config } : null,
    )
  }

  # ---------------------------------------------------------------------------
  # TRAINING
  # ---------------------------------------------------------------------------
  ml_v2_training_definition = {
    Comment = "Trust Score ML training (v2): data checks + feature selection -> sweep (N parallel Spot jobs) -> champion."
    StartAt = "CheckAlreadyRunning"
    States = {
      CheckAlreadyRunning = {
        Type     = "Task"
        Resource = "arn:aws:states:::aws-sdk:sfn:listExecutions"
        Parameters = {
          StateMachineArn = local.ml_v2_training_sm_arn
          StatusFilter    = "RUNNING"
          MaxResults      = 10
        }
        ResultSelector = { "running.$" = "States.ArrayLength($.Executions)" }
        ResultPath     = "$.guard"
        Next           = "IsAnotherRunRunning"
        Catch          = local.ml_v2_catch_all
      }
      # This execution is itself RUNNING, so "another run" means more than one.
      IsAnotherRunRunning = {
        Type = "Choice"
        Choices = [{
          Variable           = "$.guard.running"
          NumericGreaterThan = 1
          Next               = "SkippedAlreadyRunning"
        }]
        Default = "BuildContext"
      }
      SkippedAlreadyRunning = {
        Type    = "Succeed"
        Comment = "A training run is already in progress; it will use the new data. Nothing to do."
      }
      BuildContext = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.ml_v2_helper.arn
          Payload = {
            action             = "training_context"
            "execution_name.$" = "$$.Execution.Name"
            "start_time.$"     = "$$.Execution.StartTime"
          }
        }
        ResultSelector = {
          "run_id.$"     = "$.Payload.run_id"
          "total_arms.$" = "$.Payload.total_arms"
          "prepare.$"    = "$.Payload.prepare"
          "slices.$"     = "$.Payload.slices"
          "finalize.$"   = "$.Payload.finalize"
        }
        ResultPath = "$.context"
        Retry      = local.ml_v2_lambda_retry
        Catch      = local.ml_v2_catch_all
        Next       = "Prepare"
      }
      Prepare = {
        Type           = "Task"
        Resource       = "arn:aws:states:::sagemaker:createTrainingJob.sync"
        Parameters     = local.ml_v2_training_job.prepare
        ResultSelector = local.ml_v2_job_result
        ResultPath     = "$.results.prepare"
        Retry          = local.ml_v2_sagemaker_retry
        Catch          = local.ml_v2_catch_all
        Next           = "Sweep"
      }
      Sweep = {
        Type           = "Map"
        ItemsPath      = "$.context.slices"
        MaxConcurrency = var.ml_v2_sweep_slice_count
        ItemSelector = {
          "slice.$"  = "$$.Map.Item.Value"
          "run_id.$" = "$.context.run_id"
        }
        ItemProcessor = {
          ProcessorConfig = { Mode = "INLINE" }
          StartAt         = "SweepSlice"
          States = {
            SweepSlice = {
              Type           = "Task"
              Resource       = "arn:aws:states:::sagemaker:createTrainingJob.sync"
              Parameters     = local.ml_v2_training_job.sweep
              ResultSelector = local.ml_v2_job_result
              Retry          = local.ml_v2_sagemaker_retry
              # One failed slice must not abort the others: their work is
              # resumable and already paid for. Failures are counted below.
              Catch = [{ ErrorEquals = ["States.ALL"], ResultPath = "$.error", Next = "SliceFailed" }]
              End   = true
            }
            SliceFailed = {
              Type       = "Pass"
              Parameters = { "job.$" = "$.slice.job_name", status = "Failed", "error.$" = "$.error" }
              End        = true
            }
          }
        }
        ResultPath = "$.results.sweep"
        Catch      = local.ml_v2_catch_all
        Next       = "CountFailedSlices"
      }
      CountFailedSlices = {
        Type       = "Pass"
        Parameters = { "failed.$" = "$.results.sweep[?(@.status == 'Failed')]" }
        ResultPath = "$.slice_check"
        Next       = "CountFailedSlicesLength"
      }
      CountFailedSlicesLength = {
        Type       = "Pass"
        Parameters = { "count.$" = "States.ArrayLength($.slice_check.failed)", "failed.$" = "$.slice_check.failed" }
        ResultPath = "$.slice_check"
        Next       = "AllSlicesSucceeded"
      }
      AllSlicesSucceeded = {
        Type = "Choice"
        Choices = [{
          Variable           = "$.slice_check.count"
          NumericGreaterThan = 0
          Next               = "SlicesFailed"
        }]
        Default = "Finalize"
      }
      SlicesFailed = {
        Type = "Pass"
        Parameters = {
          Error     = "SweepSlicesFailed"
          "Cause.$" = "States.Format('{} sweep slice(s) failed; re-run the execution to resume them: {}', $.slice_check.count, States.JsonToString($.slice_check.failed))"
        }
        ResultPath = "$.error"
        Next       = "NotifyFailure"
      }
      Finalize = {
        Type           = "Task"
        Resource       = "arn:aws:states:::sagemaker:createTrainingJob.sync"
        Parameters     = local.ml_v2_training_job.finalize
        ResultSelector = local.ml_v2_job_result
        ResultPath     = "$.results.finalize"
        Retry          = local.ml_v2_sagemaker_retry
        Catch          = local.ml_v2_catch_all
        Next           = "NotifySuccess"
      }
      NotifySuccess = {
        Type     = "Task"
        Resource = "arn:aws:states:::sns:publish"
        Parameters = {
          TopicArn    = aws_sns_topic.ml_v2_alerts.arn
          Subject     = "Trust Score ML training finished"
          "Message.$" = "States.Format('Training run {} finished: {} arms swept, champion published by {}. Outputs: ${local.ml_v2_output_root}', $.context.run_id, $.context.total_arms, $.results.finalize.job)"
        }
        ResultPath = null
        End        = true
      }
      NotifyFailure = {
        Type     = "Task"
        Resource = "arn:aws:states:::sns:publish"
        Parameters = {
          TopicArn    = aws_sns_topic.ml_v2_alerts.arn
          Subject     = "Trust Score ML training FAILED"
          "Message.$" = "States.Format('Training execution {} failed: {}', $$.Execution.Name, States.JsonToString($.error))"
        }
        ResultPath = null
        Next       = "Failed"
      }
      Failed = { Type = "Fail", Error = "TrainingPipelineFailed", Cause = "See the SNS alert and the SageMaker job logs." }
    }
  }

  # ---------------------------------------------------------------------------
  # SCORING
  # ---------------------------------------------------------------------------
  ml_v2_scoring_job = merge(
    {
      "ProcessingJobName.$" = "$.context.job_name"
      ProcessingResources = {
        ClusterConfig = {
          InstanceCount  = 1
          InstanceType   = var.ml_v2_scoring_instance_type
          VolumeSizeInGB = var.ml_v2_scoring_volume_size_gb
        }
      }
      AppSpecification = {
        ImageUri               = local.ml_v2_inference_image
        ContainerEntrypoint    = ["python3", "/opt/program/inference.py"]
        "ContainerArguments.$" = "$.context.container_arguments"
      }
      Environment = {
        AWS_REGION         = data.aws_region.current.region
        AWS_DEFAULT_REGION = data.aws_region.current.region
      }
      ProcessingOutputConfig = {
        Outputs = [{
          OutputName = "summary"
          S3Output = {
            "S3Uri.$"    = "$.context.summary_s3_prefix"
            LocalPath    = "/opt/ml/processing/output"
            S3UploadMode = "EndOfJob"
          }
        }]
      }
      RoleArn           = data.aws_iam_role.ml_v2_sagemaker_execution.arn
      StoppingCondition = { MaxRuntimeInSeconds = var.ml_v2_max_runtime_hours.scoring * local.ml_v2_hours }
      Tags              = local.ml_v2_job_tags
    },
    local.ml_v2_vpc_config != null ? { NetworkConfig = { VpcConfig = local.ml_v2_vpc_config } } : null,
  )

  ml_v2_scoring_after_ok = var.ml_v2_notify_on_scoring_success ? "NotifyScored" : "Done"

  ml_v2_scoring_states = merge(
    {
      BuildContext = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.ml_v2_helper.arn
          Payload = {
            action             = "scoring_context"
            "execution_name.$" = "$$.Execution.Name"
            "start_time.$"     = "$$.Execution.StartTime"
            "detail.$"         = "$.detail"
          }
        }
        ResultSelector = {
          "batch_date.$"          = "$.Payload.batch_date"
          "input_uri.$"           = "$.Payload.input_uri"
          "job_name.$"            = "$.Payload.job_name"
          "container_arguments.$" = "$.Payload.container_arguments"
          "summary_s3_prefix.$"   = "$.Payload.summary_s3_prefix"
          "summary_bucket.$"      = "$.Payload.summary_bucket"
          "summary_key.$"         = "$.Payload.summary_key"
        }
        ResultPath = "$.context"
        Retry      = local.ml_v2_lambda_retry
        Catch      = local.ml_v2_catch_all
        Next       = "Score"
      }
      Score = {
        Type           = "Task"
        Resource       = "arn:aws:states:::sagemaker:createProcessingJob.sync"
        Parameters     = local.ml_v2_scoring_job
        ResultSelector = { "job.$" = "$.ProcessingJobName", "status.$" = "$.ProcessingJobStatus" }
        ResultPath     = "$.results.score"
        Retry          = local.ml_v2_sagemaker_retry
        Catch          = local.ml_v2_catch_all
        Next           = "ReadSummary"
      }
      ReadSummary = {
        Type     = "Task"
        Resource = "arn:aws:states:::aws-sdk:s3:getObject"
        Parameters = {
          "Bucket.$" = "$.context.summary_bucket"
          "Key.$"    = "$.context.summary_key"
        }
        ResultSelector = { "summary.$" = "States.StringToJson($.Body)" }
        ResultPath     = "$.scoring"
        # An EndOfJob upload can lag the job's Completed status by seconds.
        Retry = [{ ErrorEquals = ["S3.NoSuchKey", "S3.NoSuchKeyException", "S3.S3Exception"], IntervalSeconds = 10, MaxAttempts = 3, BackoffRate = 2 }]
        Catch = local.ml_v2_catch_all
        Next  = "DriftDetected"
      }
      DriftDetected = {
        Type = "Choice"
        Choices = [{
          Variable      = "$.scoring.summary.retrain_recommended"
          BooleanEquals = true
          Next          = "NotifyDrift"
        }]
        Default = local.ml_v2_scoring_after_ok
      }
      NotifyDrift = {
        Type     = "Task"
        Resource = "arn:aws:states:::sns:publish"
        Parameters = {
          TopicArn    = aws_sns_topic.ml_v2_alerts.arn
          Subject     = "Trust Score drift: retraining recommended"
          "Message.$" = "States.Format('Batch {} was scored, but the data has drifted from what the model was trained on: {}. Scores: {}. Automatic retraining is ${var.ml_v2_auto_retrain_on_drift ? "ON (subject to the ${var.ml_v2_retrain_cooldown_days}-day cooldown)" : "OFF - start the training state machine manually if appropriate"}.', $.context.batch_date, States.JsonToString($.scoring.summary.reasons), $.scoring.summary.output_prefix)"
        }
        ResultPath = null
        Next       = var.ml_v2_auto_retrain_on_drift ? "RetrainGate" : "Done"
      }
      NotifyFailure = {
        Type     = "Task"
        Resource = "arn:aws:states:::sns:publish"
        Parameters = {
          TopicArn    = aws_sns_topic.ml_v2_alerts.arn
          Subject     = "Trust Score batch scoring FAILED"
          "Message.$" = "States.Format('Scoring execution {} failed: {}', $$.Execution.Name, States.JsonToString($.error))"
        }
        ResultPath = null
        Next       = "Failed"
      }
      Done   = { Type = "Succeed" }
      Failed = { Type = "Fail", Error = "ScoringPipelineFailed", Cause = "See the SNS alert and the SageMaker processing job logs." }
    },
    var.ml_v2_notify_on_scoring_success ? {
      NotifyScored = {
        Type     = "Task"
        Resource = "arn:aws:states:::sns:publish"
        Parameters = {
          TopicArn    = aws_sns_topic.ml_v2_alerts.arn
          Subject     = "Trust Score batch scored"
          "Message.$" = "States.Format('Batch {} scored: {} rows, review queue of {} at {}.', $.context.batch_date, $.scoring.summary.rows_scored, $.scoring.summary.queue_length, $.scoring.summary.output_prefix)"
        }
        ResultPath = null
        Next       = "Done"
      }
    } : null,
    var.ml_v2_auto_retrain_on_drift ? {
      RetrainGate = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.ml_v2_helper.arn
          Payload = {
            action         = "retrain_gate"
            "start_time.$" = "$$.Execution.StartTime"
          }
        }
        ResultSelector = { "allowed.$" = "$.Payload.allowed", "next_allowed_after.$" = "$.Payload.next_allowed_after" }
        ResultPath     = "$.gate"
        Retry          = local.ml_v2_lambda_retry
        Catch          = local.ml_v2_catch_all
        Next           = "RetrainAllowed"
      }
      RetrainAllowed = {
        Type = "Choice"
        Choices = [{
          Variable      = "$.gate.allowed"
          BooleanEquals = true
          Next          = "StartRetraining"
        }]
        Default = "Done"
      }
      StartRetraining = {
        Type     = "Task"
        Resource = "arn:aws:states:::states:startExecution"
        Parameters = {
          StateMachineArn = local.ml_v2_training_sm_arn
          Input = {
            source                                         = "drift"
            "batch_date.$"                                 = "$.context.batch_date"
            "AWS_STEP_FUNCTIONS_STARTED_BY_EXECUTION_ID.$" = "$$.Execution.Id"
          }
        }
        ResultPath = null
        Catch      = local.ml_v2_catch_all
        Next       = "Done"
      }
    } : null,
  )

  ml_v2_scoring_definition = {
    Comment = "Trust Score ML nightly batch scoring (v2): score -> review queue -> drift check -> optional retrain."
    StartAt = "BuildContext"
    States  = local.ml_v2_scoring_states
  }
}

# Log groups and logging follow stepfunctions_gold.tf.
resource "aws_cloudwatch_log_group" "ml_v2_training_sm" {
  name              = "/aws/vendedlogs/states/${local.ml_v2_training_sm_name}"
  retention_in_days = 30
}

resource "aws_cloudwatch_log_group" "ml_v2_scoring_sm" {
  name              = "/aws/vendedlogs/states/${local.ml_v2_scoring_sm_name}"
  retention_in_days = 30
}

resource "aws_sfn_state_machine" "ml_v2_training" {
  name       = local.ml_v2_training_sm_name
  role_arn   = data.aws_iam_role.ml_v2_sfn.arn
  type       = "STANDARD"
  definition = jsonencode(local.ml_v2_training_definition)

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.ml_v2_training_sm.arn}:*"
    include_execution_data = true
    level                  = "ALL"
  }
}

resource "aws_sfn_state_machine" "ml_v2_scoring" {
  name       = local.ml_v2_scoring_sm_name
  role_arn   = data.aws_iam_role.ml_v2_sfn.arn
  type       = "STANDARD"
  definition = jsonencode(local.ml_v2_scoring_definition)

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.ml_v2_scoring_sm.arn}:*"
    include_execution_data = true
    level                  = "ALL"
  }
}
