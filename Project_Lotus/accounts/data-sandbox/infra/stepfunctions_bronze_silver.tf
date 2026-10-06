# One dataset's Gate 1 -> Silver -> record -> Gate 2 check, per execution.
#
# This state machine is started ONCE PER DATASET, independently, by the
# bronze_schema_validator Lambda the instant that dataset's own Bronze
# validation returns PASS - see the `start_silver_execution` call in
# bronze_schema_validator.py and the SILVER_STATE_MACHINE_ARN /
# DATASET_SPECS_JSON environment variables wired to it below.
#
# It is deliberately NOT a Map/Parallel state fanning out over all enabled
# datasets in one execution. account_changes_batch and device_lookup_batch
# land in Bronze and clear Gate 1 at different, unrelated
# times of day - sometimes hours apart - so there is no single moment at which
# "both are ready" to be handed to one Step Functions Map. Each dataset
# gets its own execution, on its own clock, and the join happens elsewhere:
# every execution that reaches a successful Silver run calls the control
# Lambda's `evaluate` action, which re-reads the FULL current state of every
# enabled dataset for that run_date and releases Gold only if all of them now
# show 'succeeded'. Whichever dataset's execution happens to finish last - in
# any order - is the one whose `evaluate` call finds the complete set and
# writes the release object. No dataset waits for any other, and no
# coordinator has to know how many datasets exist ahead of time.
#
# Modularity: adding a fourth dataset (tu_portps, mno_activation, ...) touches
# three places - one more than originally planned, because the real quarantine
# tables turned out not to follow a name formula this Terraform can derive:
#   1. local.silver_gold_enabled_datasets (silver_gold_control_data.tf) - adds
#      it to Gate 2's required set and to DATASET_SPECS_JSON below.
#   2. SUPPORTED_DATASETS in bronze_schema_validator.py - lets the Bronze
#      Lambda accept it at all.
#   3. A new aws_glue_catalog_table.silver_quarantine_<name> resource in
#      silver_quarantine_iceberg.tf, plus a matching entry in
#      local.silver_dataset_specs below. Each quarantine table is its own
#      distinct Iceberg table with a real-world-verified name and location
#      (see the mismatch note on silver_dataset_specs) - there is no formula
#      to derive a fourth one from the dataset name alone.

variable "silver_gold_state_machine_artifact_root" {
  description = "S3 prefix produced by the Silver package script, containing silver_pipeline.zip and the job entrypoint."
  type        = string
  default     = "s3://lotus-sandbox-bronze-landing-data/artifacts/silver/"
}

variable "silver_artifact_root" {
  description = "S3 root for Silver run reports and issue rows."
  type        = string
  default     = "s3://lotus-sandbox-silver-conformed-data/run-artifacts/"
}

variable "transient_emr_idle_timeout_seconds" {
  description = "Idle termination safety net for Step Functions and Terraform-managed EMR clusters; not a running-job timeout."
  type        = number
  default     = 3600

  validation {
    condition     = var.transient_emr_idle_timeout_seconds >= 60 && var.transient_emr_idle_timeout_seconds <= 604800 && floor(var.transient_emr_idle_timeout_seconds) == var.transient_emr_idle_timeout_seconds
    error_message = "EMR idle timeout must be a whole number of seconds between 60 and 604800."
  }
}

locals {
  sfn_bronze_silver_name = "${var.project}-${var.environment}-trust-score-bronze-silver"

  # Keyed by dataset name, one entry per dataset the Silver Step Functions
  # execution can run. `table` follows the standard `glue_catalog.<db>.<name>`
  # form, but `quarantine_table` / `quarantine_table_location` do NOT follow a
  # `<name>_quarantine` formula - they are the actual tables and S3 locations
  # silver_quarantine_iceberg.tf already provisions
  # (silver_quarantine_<name>, under quarantine/iceberg/, not
  # quarantine/<name>/). An earlier version of this local invented its own
  # naming formula instead of referencing those resources directly, which
  # silently diverged from what was actually deployed; referencing the real
  # resources here means a rename over there can no longer drift unnoticed.
  silver_dataset_specs = {
    account_changes_batch = {
      table                     = "glue_catalog.${var.glue_database_name}.account_changes_batch"
      quarantine_table          = "glue_catalog.${var.glue_database_name}.${aws_glue_catalog_table.silver_quarantine_account_changes_batch.name}"
      quarantine_table_location = local.quarantine_ac_location
    }
    device_lookup_batch = {
      table                     = "glue_catalog.${var.glue_database_name}.device_lookup_batch"
      quarantine_table          = "glue_catalog.${var.glue_database_name}.${aws_glue_catalog_table.silver_quarantine_device_lookup_batch.name}"
      quarantine_table_location = local.quarantine_dl_location
    }
    audit_trail_services_3 = {
      table                     = "glue_catalog.${var.glue_database_name}.audit_trail_services_3"
      quarantine_table          = "glue_catalog.${var.glue_database_name}.${aws_glue_catalog_table.silver_quarantine_audit_trail_services_3.name}"
      quarantine_table_location = local.quarantine_ats_location
    }
  }
}

resource "aws_cloudwatch_log_group" "sfn_bronze_silver" {
  name              = "/aws/vendedlogs/states/${local.sfn_bronze_silver_name}"
  retention_in_days = 30
}

resource "aws_sfn_state_machine" "bronze_silver" {
  name     = local.sfn_bronze_silver_name
  role_arn = data.aws_iam_role.sfn_bronze_silver.arn

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.sfn_bronze_silver.arn}:*"
    include_execution_data = true
    level                  = "ALL"
  }

  # Execution input, one dataset per execution - real values, not invented
  # ones, per local.silver_dataset_specs above:
  # {
  #   "run_date": "2026-08-12",
  #   "dataset_name": "account_changes_batch",
  #   "folder_path": "s3://lotus-sandbox-bronze-landing-data/bronze/account_changes_batch/2026-08-12/",
  #   "table": "glue_catalog.lotus_sandbox_silver_conformed.account_changes_batch",
  #   "quarantine_table": "glue_catalog.lotus_sandbox_silver_conformed.silver_quarantine_account_changes_batch",
  #   "quarantine_table_location": "s3://lotus-sandbox-silver-conformed-data/quarantine/iceberg/silver_quarantine_account_changes_batch/"
  # }
  definition = jsonencode({
    Comment = "One dataset: Gate 1 (re-verified) -> Silver -> record -> Gate 2 check."
    StartAt = "GateOneCheck"

    States = {

      # Re-verifies Gate 1 from the control table rather than trusting that
      # the caller already checked. The Bronze Lambda only starts this
      # execution when ITS OWN pass computed PASS, but re-reading the control
      # table here means a manually started execution, or one started against
      # a folder that was re-validated a second later, is still gated
      # correctly - the query is authoritative, the caller's decision is not.
      #
      # Runs the query inside the control Lambda's `gate_one` action rather
      # than as an inline Athena Task built from a `States.Format()` string.
      # Amazon States Language's intrinsic-function string literals have no
      # reliable escape sequence for a literal single quote, and that query
      # needs several - both as SQL delimiters around PASS/WARN/BLOCK and
      # around the two runtime values - which is what actually produced
      # "SCHEMA_VALIDATION_FAILED: QueryString.$ must be a valid JSONPath or a
      # valid intrinsic function call" the first time this was deployed.
      GateOneCheck = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.silver_gold_control.arn
          Payload = {
            action           = "gate_one"
            "dataset_name.$" = "$.dataset_name"
            "folder_path.$"  = "$.folder_path"
          }
        }
        ResultPath = "$.gate_one"
        Next       = "GateOneApproved"

        Retry = [{
          ErrorEquals     = ["States.ALL"]
          MaxAttempts     = 1
          IntervalSeconds = 15
          BackoffRate     = 1.0
        }]

        Catch = [{
          ErrorEquals = ["States.ALL"]
          ResultPath  = "$.error"
          Next        = "GateOneFailed"
        }]
      }

      GateOneApproved = {
        Type = "Choice"
        Choices = [{
          Variable     = "$.gate_one.Payload.dataset_gate_status"
          StringEquals = "PASS"
          Next         = "BuildSilverArgs_1"
        }]
        Default = "GateOneBlocked"
      }

      GateOneBlocked = {
        Type     = "Task"
        Resource = "arn:aws:states:::sns:publish"
        Parameters = {
          TopicArn    = aws_sns_topic.pipeline_alerts["bronze"].arn
          "Subject.$" = "States.Format('Gate 1 BLOCK: {}', $.dataset_name)"
          "Message.$" = "States.Format('Bronze schema validation gate blocked dataset {} for run_date {}. Silver was not started for this dataset. Folder: {}', $.dataset_name, $.run_date, $.folder_path)"
        }
        ResultPath = "$.alert"
        Next       = "RecordBlocked"
      }

      GateOneFailed = {
        Type     = "Task"
        Resource = "arn:aws:states:::sns:publish"
        Parameters = {
          TopicArn    = aws_sns_topic.pipeline_alerts["bronze"].arn
          "Subject.$" = "States.Format('Gate 1 ERROR: {}', $.dataset_name)"
          "Message.$" = "States.Format('Gate 1 query failed for dataset {} run_date {}.', $.dataset_name, $.run_date)"
        }
        ResultPath = "$.alert"
        Next       = "RecordBlocked"
      }

      # A dataset that never cleared Gate 1 is recorded as blocked so Gate 2
      # sees an explicit answer for it rather than a missing row. No Gate 2
      # check follows: this dataset did not succeed, so the full set cannot be
      # complete right now regardless of what any other dataset just did.
      RecordBlocked = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.silver_gold_control.arn
          Payload = {
            action           = "record"
            "run_date.$"     = "$.run_date"
            "dataset_name.$" = "$.dataset_name"
            status           = "blocked"
            exit_code        = 0
            attempt_number   = 1
            error_message    = "Bronze Gate 1 did not return PASS"
          }
        }
        ResultPath = "$.record"
        End        = true
      }

      # One transient cluster is shared by the independent Silver
      # executions for the same run_date. S3's conditional claim elects one
      # creator; Iceberg records the published cluster ID.
      ClaimCluster = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.silver_gold_control.arn
          Payload = {
            action       = "claim_cluster"
            "run_date.$" = "$.run_date"
            "owner.$"    = "$$.Execution.Id"
          }
        }
        ResultPath = "$.cluster_claim"
        Next       = "ClusterClaimed"
        Retry      = [{ ErrorEquals = ["States.ALL"], IntervalSeconds = 5, BackoffRate = 2, MaxAttempts = 2 }]
        Catch      = [{ ErrorEquals = ["States.ALL"], ResultPath = "$.error", Next = "ReleaseClusterClaim" }]
      }

      ClusterClaimed = {
        Type    = "Choice"
        Choices = [{ Variable = "$.cluster_claim.Payload.claimed", BooleanEquals = true, Next = "CreateCluster" }]
        Default = "StartClusterWait"
      }

      CreateCluster = {
        Type     = "Task"
        Resource = "arn:aws:states:::elasticmapreduce:createCluster.sync"
        Parameters = {
          "Name.$"              = "States.Format('${local.emr_cluster_name}-{}', $.run_date)"
          ReleaseLabel          = var.emr_release_label
          Applications          = [{ Name = "Hadoop" }, { Name = "Spark" }]
          ServiceRole           = data.aws_iam_role.emr_service.arn
          JobFlowRole           = data.aws_iam_instance_profile.emr_ec2.arn
          SecurityConfiguration = aws_emr_security_configuration.runtime_roles.name
          LogUri                = var.emr_log_uri
          VisibleToAllUsers     = true
          StepConcurrencyLevel  = local.emr_step_concurrency_level
          Tags = [
            {
              Key   = "lotus-workflow"
              Value = "silver-gold-transient"
            },
            {
              Key   = "for-use-with-amazon-emr-managed-policies"
              Value = "true"
            }
          ]
          Configurations = local.emr_configurations
          AutoTerminationPolicy = {
            IdleTimeout = var.transient_emr_idle_timeout_seconds
          }
          Instances = {
            KeepJobFlowAliveWhenNoSteps   = true
            TerminationProtected          = false
            Ec2SubnetIds                  = var.emr_subnet_ids
            EmrManagedMasterSecurityGroup = aws_security_group.emr_primary.id
            EmrManagedSlaveSecurityGroup  = aws_security_group.emr_core.id
            ServiceAccessSecurityGroup    = aws_security_group.emr_service_access.id
            InstanceFleets = concat([
              {
                Name                   = "master"
                InstanceFleetType      = "MASTER"
                TargetOnDemandCapacity = 1
                InstanceTypeConfigs = [{
                  InstanceType     = var.emr_primary_instance_type
                  WeightedCapacity = 1
                  EbsConfiguration = {
                    EbsBlockDeviceConfigs = [{
                      VolumeSpecification = { VolumeType = "gp3", SizeInGB = local.emr_primary_ebs_size_gib }
                      VolumesPerInstance  = 1
                    }]
                  }
                }]
              },
              {
                Name                   = "core"
                InstanceFleetType      = "CORE"
                TargetOnDemandCapacity = var.emr_core_target_on_demand_capacity
                InstanceTypeConfigs = [{
                  InstanceType     = var.emr_core_instance_type
                  WeightedCapacity = 1
                  EbsConfiguration = {
                    EbsBlockDeviceConfigs = [{
                      VolumeSpecification = { VolumeType = "gp3", SizeInGB = local.emr_worker_ebs_size_gib }
                      VolumesPerInstance  = 1
                    }]
                  }
                }]
              }
              ], var.emr_task_target_on_demand_capacity + var.emr_task_target_spot_capacity > 0 ? [
              merge({
                Name                   = "task"
                InstanceFleetType      = "TASK"
                TargetOnDemandCapacity = var.emr_task_target_on_demand_capacity
                TargetSpotCapacity     = var.emr_task_target_spot_capacity
                InstanceTypeConfigs = [{
                  InstanceType     = var.emr_task_instance_type
                  WeightedCapacity = 1
                  EbsConfiguration = {
                    EbsBlockDeviceConfigs = [{
                      VolumeSpecification = { VolumeType = "gp3", SizeInGB = local.emr_worker_ebs_size_gib }
                      VolumesPerInstance  = 1
                    }]
                  }
                }]
                }, var.emr_task_target_spot_capacity > 0 ? {
                LaunchSpecifications = {
                  SpotSpecification = {
                    AllocationStrategy     = "capacity-optimized"
                    TimeoutAction          = "SWITCH_TO_ON_DEMAND"
                    TimeoutDurationMinutes = 10
                  }
                }
              } : {})
            ] : [])
          }
        }
        ResultPath = "$.created_cluster"
        Next       = "RegisterCluster"
        Catch = [{
          ErrorEquals = ["States.ALL"]
          ResultPath  = "$.error"
          Next        = "ReleaseClusterClaim"
        }]
      }

      RegisterCluster = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.silver_gold_control.arn
          Payload = {
            action         = "record_cluster"
            "run_date.$"   = "$.run_date"
            "cluster_id.$" = "$.created_cluster.ClusterId"
            "owner.$"      = "$$.Execution.Id"
          }
        }
        ResultPath = "$.cluster_registration"
        Next       = "UseCreatedCluster"
        Retry = [{
          ErrorEquals     = ["States.ALL"]
          IntervalSeconds = 5
          BackoffRate     = 2
          MaxAttempts     = 3
        }]
        Catch = [{
          ErrorEquals = ["States.ALL"]
          ResultPath  = "$.error"
          Next        = "TerminateUnregisteredCluster"
        }]
      }

      TerminateUnregisteredCluster = {
        Type       = "Task"
        Resource   = "arn:aws:states:::elasticmapreduce:terminateCluster.sync"
        Parameters = { "ClusterId.$" = "$.created_cluster.ClusterId" }
        ResultPath = "$.cleanup"
        Next       = "ReleaseClusterClaim"
        Catch = [{
          ErrorEquals = ["States.ALL"]
          ResultPath  = "$.cleanup_error"
          Next        = "ClusterUnavailable"
        }]
      }

      ReleaseClusterClaim = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.silver_gold_control.arn
          Payload = {
            action       = "release_cluster_claim"
            "run_date.$" = "$.run_date"
            "owner.$"    = "$$.Execution.Id"
          }
        }
        ResultPath = "$.release_claim"
        Next       = "ClusterUnavailable"
      }

      StartClusterWait = {
        Type       = "Pass"
        Result     = { attempts = 0 }
        ResultPath = "$.cluster_wait"
        Next       = "ReadCluster"
      }

      ReadCluster = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.silver_gold_control.arn
          Payload      = { action = "get_cluster", "run_date.$" = "$.run_date" }
        }
        ResultPath = "$.cluster_record"
        Next       = "ClusterReady"
      }

      ClusterReady = {
        Type = "Choice"
        Choices = [
          {
            Variable     = "$.cluster_record.Payload.status"
            StringEquals = "missing"
            Next         = "ClaimCluster"
          },
          {
            Variable     = "$.cluster_record.Payload.status"
            StringEquals = "failed"
            Next         = "ClaimCluster"
          },
          {
            Variable     = "$.cluster_record.Payload.status"
            StringEquals = "ready"
            Next         = "UseExistingCluster"
          },
          {
            Variable     = "$.cluster_record.Payload.status"
            StringEquals = "terminated"
            Next         = "ClusterUnavailable"
          },
          {
            Variable                 = "$.cluster_wait.attempts"
            NumericGreaterThanEquals = 120
            Next                     = "ClusterUnavailable"
          }
        ]
        Default = "WaitForCluster"
      }

      WaitForCluster = {
        Type    = "Wait"
        Seconds = 30
        Next    = "IncrementClusterWait"
      }

      IncrementClusterWait = {
        Type       = "Pass"
        Parameters = { "attempts.$" = "States.MathAdd($.cluster_wait.attempts, 1)" }
        ResultPath = "$.cluster_wait"
        Next       = "ReadCluster"
      }

      UseCreatedCluster = {
        Type       = "Pass"
        Parameters = { "id.$" = "$.created_cluster.ClusterId" }
        ResultPath = "$.cluster"
        Next       = "SilverStep_1"
      }

      UseExistingCluster = {
        Type       = "Pass"
        Parameters = { "id.$" = "$.cluster_record.Payload.cluster_id" }
        ResultPath = "$.cluster"
        Next       = "SilverStep_1"
      }

      ClusterUnavailable = {
        Type     = "Task"
        Resource = "arn:aws:states:::sns:publish"
        Parameters = {
          TopicArn    = aws_sns_topic.pipeline_alerts["silver"].arn
          Subject     = "Silver cluster unavailable"
          "Message.$" = "States.Format('No usable daily EMR cluster for dataset {} and run date {}.', $.dataset_name, $.run_date)"
        }
        Next = "ClusterFailed"
      }

      ClusterFailed = {
        Type  = "Fail"
        Error = "SilverClusterUnavailable"
      }

      # Silver, as two explicit attempts rather than Amazon States Language's
      # own per-state Retry.
      #
      # ASL's Retry resubmits a Task's ALREADY-RESOLVED Parameters unchanged -
      # it does not re-run whatever built them. build_silver_args mints
      # --run-id itself (see its docstring), and Silver's artifact write uses
      # mode("errorifexists") and refuses to reuse one, so a retry that simply
      # resubmitted SilverStep_1's Parameters would resend the exact same
      # --run-id and fail immediately on the artifact write - not because the
      # transform failed again, but because Step Functions never asked for a
      # new run-id in the first place. Calling BuildSilverArgs a second time,
      # on its own named state, is what actually gets a fresh one.
      #
      # One retry total. A terminal failure terminates the shared daily cluster
      # and prevents Gate 2 from releasing Gold.
      BuildSilverArgs_1 = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.silver_gold_control.arn
          Payload = {
            action           = "build_silver_args"
            "dataset_name.$" = "$.dataset_name"
            "run_date.$"     = "$.run_date"
            "folder_path.$"  = "$.folder_path"
            attempt          = 1
          }
        }
        ResultPath = "$.silver_args"
        Next       = "ClaimCluster"

        Retry = [{
          ErrorEquals     = ["States.ALL"]
          MaxAttempts     = 1
          IntervalSeconds = 15
          BackoffRate     = 1.0
        }]

        # No run-id exists yet if build_silver_args itself failed (bad config
        # object, dataset missing from it, S3 read error) - RecordFailed
        # below needs one, so this goes to its own terminal path instead.
        Catch = [{
          ErrorEquals = ["States.ALL"]
          ResultPath  = "$.error"
          Next        = "BuildArgsFailed"
        }]
      }

      SilverStep_1 = {
        Type     = "Task"
        Resource = "arn:aws:states:::elasticmapreduce:addStep.sync"
        Parameters = {
          "ClusterId.$"    = "$.cluster.id"
          ExecutionRoleArn = data.aws_iam_role.silver_job_runtime.arn
          Step = {
            "Name.$"        = "$.silver_args.Payload.run_id"
            ActionOnFailure = "CONTINUE"
            HadoopJarStep = {
              Jar      = "command-runner.jar"
              "Args.$" = "$.silver_args.Payload.args"
            }
          }
        }
        ResultPath = "$.silver"
        Next       = "RecordSucceeded"

        Catch = [{
          ErrorEquals = ["States.ALL"]
          ResultPath  = "$.error"
          Next        = "BuildSilverArgs_2"
        }]
      }

      BuildSilverArgs_2 = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.silver_gold_control.arn
          Payload = {
            action           = "build_silver_args"
            "dataset_name.$" = "$.dataset_name"
            "run_date.$"     = "$.run_date"
            "folder_path.$"  = "$.folder_path"
            attempt          = 2
          }
        }
        ResultPath = "$.silver_args"
        Next       = "SilverStep_2"

        Retry = [{
          ErrorEquals     = ["States.ALL"]
          MaxAttempts     = 1
          IntervalSeconds = 15
          BackoffRate     = 1.0
        }]

        Catch = [{
          ErrorEquals = ["States.ALL"]
          ResultPath  = "$.error"
          Next        = "RecordRetryArgsFailed"
        }]
      }

      SilverStep_2 = {
        Type     = "Task"
        Resource = "arn:aws:states:::elasticmapreduce:addStep.sync"
        Parameters = {
          "ClusterId.$"    = "$.cluster.id"
          ExecutionRoleArn = data.aws_iam_role.silver_job_runtime.arn
          Step = {
            "Name.$"        = "$.silver_args.Payload.run_id"
            ActionOnFailure = "CONTINUE"
            HadoopJarStep = {
              Jar      = "command-runner.jar"
              "Args.$" = "$.silver_args.Payload.args"
            }
          }
        }
        ResultPath = "$.silver"
        Next       = "RecordSucceeded"

        Catch = [{
          ErrorEquals = ["States.ALL"]
          ResultPath  = "$.error"
          Next        = "RecordFailed"
        }]
      }

      # Reached only if build_silver_args itself failed on both attempts - a
      # configuration problem (bad JSON, dataset missing from it, S3 denied),
      # not a Silver transform failure. No silver_run_id was ever minted, so
      # this is recorded without one rather than guessing at a value.
      BuildArgsFailed = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.silver_gold_control.arn
          Payload = {
            action            = "record"
            "run_date.$"      = "$.run_date"
            "dataset_name.$"  = "$.dataset_name"
            status            = "failed"
            "error_message.$" = "States.Format('Could not build Silver spark-submit arguments: {}', $.error.Cause)"
          }
        }
        ResultPath = "$.record"
        End        = true
      }

      # Success is the only path that can possibly complete the "all enabled
      # datasets succeeded" set, so it is the only path that checks Gate 2.
      # Reachable from either SilverStep_1 or SilverStep_2 - both write to
      # $.silver_args the same way, so this state does not need to know which
      # attempt actually succeeded.
      RecordSucceeded = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.silver_gold_control.arn
          Payload = {
            action                 = "record"
            "run_date.$"           = "$.run_date"
            "dataset_name.$"       = "$.dataset_name"
            "silver_run_id.$"      = "$.silver_args.Payload.run_id"
            status                 = "succeeded"
            exit_code              = 0
            "attempt_number.$"     = "$.silver_args.Payload.attempt"
            "bronze_folder_path.$" = "$.folder_path"
          }
        }
        ResultPath = "$.record"
        Next       = "EvaluateGateTwo"
      }

      # Reachable only from SilverStep_2's Catch - both attempts are exhausted.
      RecordFailed = {
        Type           = "Task"
        Resource       = "arn:aws:states:::lambda:invoke"
        TimeoutSeconds = 60
        Parameters = {
          FunctionName = aws_lambda_function.silver_gold_control.arn
          Payload = {
            action             = "record"
            "run_date.$"       = "$.run_date"
            "dataset_name.$"   = "$.dataset_name"
            "silver_run_id.$"  = "$.silver_args.Payload.run_id"
            status             = "failed"
            "attempt_number.$" = "$.silver_args.Payload.attempt"
            "error_message.$"  = "States.Format('Silver step failed after retry: {}', States.JsonToString($.error))"
          }
        }
        ResultPath = "$.record"
        Next       = "TerminateClusterAfterSilverFailure"
        Catch = [{
          ErrorEquals = ["States.ALL"]
          ResultPath  = "$.record_error"
          Next        = "TerminateClusterAfterSilverFailure"
        }]
      }

      # BuildSilverArgs_2 runs only after SilverStep_1 failed and a cluster was
      # acquired. If the retry arguments cannot be produced, there is no EMR
      # retry to submit, so record the terminal failure and clean up that
      # cluster just as we do after SilverStep_2 fails.
      RecordRetryArgsFailed = {
        Type           = "Task"
        Resource       = "arn:aws:states:::lambda:invoke"
        TimeoutSeconds = 60
        Parameters = {
          FunctionName = aws_lambda_function.silver_gold_control.arn
          Payload = {
            action            = "record"
            "run_date.$"      = "$.run_date"
            "dataset_name.$"  = "$.dataset_name"
            status            = "failed"
            attempt_number    = 2
            "error_message.$" = "States.Format('Could not build Silver retry arguments: {}', States.JsonToString($.error))"
          }
        }
        ResultPath = "$.record"
        Next       = "TerminateClusterAfterSilverFailure"
        Catch = [{
          ErrorEquals = ["States.ALL"]
          ResultPath  = "$.record_error"
          Next        = "TerminateClusterAfterSilverFailure"
        }]
      }

      # The first Silver failure still receives one retry. Only a terminal
      # retry failure reaches this state. This cluster is shared by the day's
      # Silver/Gold work; terminating it here is intentional because Gate 2
      # cannot release Gold after any required Silver dataset has failed.
      TerminateClusterAfterSilverFailure = {
        Type       = "Task"
        Resource   = "arn:aws:states:::elasticmapreduce:terminateCluster.sync"
        Parameters = { "ClusterId.$" = "$.cluster.id" }
        ResultPath = "$.termination"
        Next       = "MarkClusterTerminatedAfterSilverFailure"
        Retry = [{
          ErrorEquals     = ["States.ALL"]
          IntervalSeconds = 30
          BackoffRate     = 2
          MaxAttempts     = 2
        }]
        Catch = [{
          ErrorEquals = ["States.ALL"]
          ResultPath  = "$.termination_error"
          Next        = "SilverFailureTerminationAlert"
        }]
      }

      MarkClusterTerminatedAfterSilverFailure = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.silver_gold_control.arn
          Payload      = { action = "close_cluster", "run_date.$" = "$.run_date" }
        }
        ResultPath = "$.cluster_closed"
        Next       = "SilverPipelineFailed"
        Retry = [{
          ErrorEquals     = ["Lambda.ServiceException", "Lambda.AWSLambdaException", "Lambda.SdkClientException", "Lambda.TooManyRequestsException"]
          IntervalSeconds = 5
          BackoffRate     = 2
          MaxAttempts     = 2
        }]
        Catch = [{
          ErrorEquals = ["States.ALL"]
          ResultPath  = "$.close_error"
          Next        = "SilverPipelineFailed"
        }]
      }

      SilverFailureTerminationAlert = {
        Type           = "Task"
        Resource       = "arn:aws:states:::sns:publish"
        TimeoutSeconds = 30
        Parameters = {
          TopicArn    = aws_sns_topic.pipeline_alerts["silver"].arn
          Subject     = "Silver EMR termination failed"
          "Message.$" = "States.Format('Silver failed for dataset {} on {} and cluster {} could not be terminated automatically. Error: {}', $.dataset_name, $.run_date, $.cluster.id, States.JsonToString($.termination_error))"
        }
        ResultPath = "$.termination_alert"
        Next       = "SilverPipelineFailed"
        Catch = [{
          ErrorEquals = ["States.ALL"]
          ResultPath  = "$.alert_error"
          Next        = "SilverPipelineFailed"
        }]
      }

      SilverPipelineFailed = {
        Type  = "Fail"
        Error = "SilverPipelineFailed"
        Cause = "Silver failed after its retry. The transient EMR termination path was executed; inspect execution history for cleanup status."
      }

      # Gate 2. Reads the control table fresh and, only if EVERY enabled
      # dataset now shows 'succeeded' for this run_date, writes the release
      # object - which is what fires EventBridge and starts Gold. This
      # execution never calls Gold directly; it only ever writes (or does not
      # write) that one S3 object. Whichever dataset's execution happens to
      # be the one that completes the set is the one whose check here
      # actually releases it - the other datasets' earlier checks simply
      # found the set incomplete and did nothing further.
      EvaluateGateTwo = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.silver_gold_control.arn
          Payload = {
            action       = "evaluate"
            "run_date.$" = "$.run_date"
          }
        }
        ResultPath = "$.gate_two"
        End        = true
      }
    }
  })

  depends_on = [
    aws_lambda_function.silver_gold_control,
    aws_sns_topic.pipeline_alerts
  ]
}

output "bronze_silver_state_machine_arn" {
  description = "ARN of the per-dataset Gate 1 -> Silver -> Gate 2 check state machine."
  value       = aws_sfn_state_machine.bronze_silver.arn
}
