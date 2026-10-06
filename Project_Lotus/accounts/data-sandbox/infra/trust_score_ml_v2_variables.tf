# =============================================================================
# VERSION 2 - Trust Score ML pipeline: settings, lookups and shared values
# =============================================================================
# Added in v2; no existing file in this root was modified. See ../CHANGES_V2.md.
#
# What the v2 files in this root build:
#
#   gold bucket  <training prefix>/<batch>/_SUCCESS ──► EventBridge ──► TRAINING state machine
#       CheckAlreadyRunning → BuildContext → Prepare (SageMaker Training)
#       → Sweep (N SageMaker Training jobs in parallel, Spot) → Finalize → SNS
#
#   gold bucket  <scoring prefix>/batch_date=YYYY-MM-DD/_SUCCESS ──► SCORING state machine
#       BuildContext → Score (SageMaker Processing) → ReadSummary → drift? → SNS
#       → (optional) start training after the cooldown
#
# The IAM roles come from ../security/trust_score_ml_v2_iam.tf (apply it first)
# and are looked up here by name, the same split the Gold pipeline uses.
# Every name below is ml_v2_* so nothing collides with the existing files.
# =============================================================================

# -----------------------------------------------------------------------------
# Data locations
# -----------------------------------------------------------------------------

variable "ml_v2_training_data_prefix" {
  description = "Prefix in the gold bucket whose completion marker starts a training run. Must end with '/'."
  type        = string
  default     = "mock-data/version1/train/"

  validation {
    condition     = endswith(var.ml_v2_training_data_prefix, "/")
    error_message = "ml_v2_training_data_prefix must end with '/'."
  }
}

variable "ml_v2_scoring_data_prefix" {
  description = "Prefix in the gold bucket whose completion markers start nightly scoring. Name each batch folder batch_date=YYYY-MM-DD. Must end with '/'."
  type        = string
  default     = "mock-data/version1/test/"

  validation {
    condition     = endswith(var.ml_v2_scoring_data_prefix, "/")
    error_message = "ml_v2_scoring_data_prefix must end with '/'."
  }
}

variable "ml_v2_trigger_marker_suffix" {
  description = "Object key suffix meaning 'this batch is complete'. Only these objects start a pipeline, so one uploaded batch starts one run. Producers write it LAST."
  type        = string
  default     = "_SUCCESS"
}

variable "ml_v2_output_prefix" {
  description = "Prefix in the SageMaker data bucket for everything the ML jobs write. Must match ml_v2_output_prefix in ../security (IAM) and models_root in conf/ml/lotus_sandbox.yaml. Ends with '/'."
  type        = string
  default     = "demo_model_results/"

  validation {
    condition     = endswith(var.ml_v2_output_prefix, "/") && !startswith(var.ml_v2_output_prefix, "/")
    error_message = "ml_v2_output_prefix must end with '/' and must not start with '/'."
  }
}

variable "ml_v2_manage_sagemaker_bucket_notification" {
  description = "Turn on S3 -> EventBridge for the gold bucket. WARNING: aws_s3_bucket_notification owns ALL notification settings of the bucket and replaces any configured outside Terraform (none are managed in this repo today). Set false if the gold bucket's notifications are managed elsewhere, and enable EventBridge there."
  type        = bool
  default     = true
}

# -----------------------------------------------------------------------------
# What the jobs run
# -----------------------------------------------------------------------------

variable "ml_v2_training_image_tag" {
  description = "Tag of the training image to run. null = the most recently pushed image. Either way the job is pinned by digest, so re-run terraform apply after pushing a new image."
  type        = string
  default     = null
}

variable "ml_v2_inference_image_tag" {
  description = "Tag of the inference image to run. null = the most recently pushed image."
  type        = string
  default     = null
}

variable "ml_v2_job_name_prefix" {
  description = "Prefix of every SageMaker job name. The Step Functions role is scoped to it, so it must match ml_v2_job_name_prefix in ../security. Short: job names are limited to 63 characters."
  type        = string
  default     = "ts05"

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{0,11}$", var.ml_v2_job_name_prefix))
    error_message = "ml_v2_job_name_prefix must be 1-12 characters of [a-z0-9-]."
  }
}

variable "ml_v2_config_files" {
  description = "ML config overlays inside the images (/opt/program), applied in order."
  type        = list(string)
  default     = ["conf/ml/base.yaml", "conf/ml/sandbox.yaml"]
}

variable "ml_v2_config_overrides" {
  description = "Optional dotted-key config overrides passed to every job, e.g. [\"run.mode=pilot\"]."
  type        = list(string)
  default     = []
}

variable "ml_v2_models" {
  description = "Comma-separated models to train. Fewer models = a cheaper, faster sweep (e.g. \"ecod,isolation_forest\" for a rehearsal)."
  type        = string
  default     = "kmeans,weighted_kmeans,hdbscan,isolation_forest,imf,copod,ecod,hbos,pca_reconstruction,autoencoder"
}

variable "ml_v2_arms_per_model" {
  description = "Sweep arms per model = len(top_n_feature_counts) x len(preprocessing_ids) in the ML config (7 x 9 = 63 in conf/ml/base.yaml). Change only if those lists change."
  type        = number
  default     = 63
}

variable "ml_v2_sweep_slice_count" {
  description = "Parallel SageMaker training jobs the sweep is split into. More = faster, about the same total cost; bounded by the account's Spot quota."
  type        = number
  default     = 6

  validation {
    condition     = var.ml_v2_sweep_slice_count >= 1 && var.ml_v2_sweep_slice_count <= 40
    error_message = "ml_v2_sweep_slice_count must be between 1 and 40."
  }
}

variable "ml_v2_allow_failed_checks" {
  description = "Let feature selection run even when a data-quality gate fails. Keep false."
  type        = bool
  default     = false
}

variable "ml_v2_review_queue_top_k" {
  description = "Length of the nightly review queue, in customers."
  type        = number
  default     = 500
}

# -----------------------------------------------------------------------------
# Instances, cost and limits
# -----------------------------------------------------------------------------

variable "ml_v2_prepare_instance_type" {
  description = "Data-quality checks + feature selection (local Spark in the training image)."
  type        = string
  default     = "ml.m6i.4xlarge"
}

variable "ml_v2_sweep_instance_type" {
  description = "Each sweep slice."
  type        = string
  default     = "ml.m6i.4xlarge"
}

variable "ml_v2_finalize_instance_type" {
  description = "Merging the slices and publishing the champion."
  type        = string
  default     = "ml.m6i.2xlarge"
}

variable "ml_v2_scoring_instance_type" {
  description = "The nightly batch-scoring processing job."
  type        = string
  default     = "ml.m6i.2xlarge"
}

variable "ml_v2_use_spot_training" {
  description = "Run the training jobs on Managed Spot capacity (much cheaper; an interrupted sweep resumes where it stopped)."
  type        = bool
  default     = true
}

variable "ml_v2_max_runtime_hours" {
  description = "Hard runtime limit per job kind, in hours."
  type        = object({ prepare = number, sweep = number, finalize = number, scoring = number })
  default     = { prepare = 6, sweep = 24, finalize = 4, scoring = 3 }
}

variable "ml_v2_spot_max_wait_hours" {
  description = "Spot only: how long a training job may wait for capacity, including its run time."
  type        = number
  default     = 36
}

variable "ml_v2_training_volume_size_gb" {
  type    = number
  default = 50
}

variable "ml_v2_scoring_volume_size_gb" {
  type    = number
  default = 30
}

variable "ml_v2_job_subnet_ids" {
  description = "Optional: run the jobs inside these subnets (e.g. var.emr_subnet_ids). Needs an S3 VPC endpoint or NAT, and ml_v2_jobs_in_vpc = true in ../security. Empty = SageMaker-managed network."
  type        = list(string)
  default     = []
}

variable "ml_v2_job_security_group_ids" {
  description = "Security groups for the jobs when ml_v2_job_subnet_ids is set."
  type        = list(string)
  default     = []
}

# -----------------------------------------------------------------------------
# Drift -> retraining, alerts, logs
# -----------------------------------------------------------------------------

variable "ml_v2_auto_retrain_on_drift" {
  description = "When nightly scoring recommends retraining, start the training pipeline automatically (after the cooldown). false = alert only."
  type        = bool
  default     = false
}

variable "ml_v2_retrain_cooldown_days" {
  description = "A drift-triggered retrain is skipped if any training run started within this many days."
  type        = number
  default     = 7
}

variable "ml_v2_alert_emails" {
  description = "Email addresses subscribed to the ML pipeline alerts. Each must confirm the subscription email."
  type        = list(string)
  default     = []
}

variable "ml_v2_notify_on_scoring_success" {
  description = "Also alert on every successful nightly scoring run, not only on failures and drift."
  type        = bool
  default     = false
}

# -----------------------------------------------------------------------------
# Lookups: roles from ../security, images from the existing ECR repositories
# -----------------------------------------------------------------------------

data "aws_iam_role" "ml_v2_sfn" {
  name = "${var.project}-${var.environment}-trust-score-ml-sfn-role"
}

data "aws_iam_role" "ml_v2_events" {
  name = "${var.project}-${var.environment}-trust-score-ml-events-role"
}

data "aws_iam_role" "ml_v2_helper" {
  name = "${var.project}-${var.environment}-trust-score-ml-helper-role"
}

data "aws_iam_role" "ml_v2_sagemaker_execution" {
  name = "${var.project}-${var.environment}-sagemaker-execution"
}

# The repositories are the existing ones in ecr.tf. Images are referenced BY
# DIGEST, so every job records exactly which build ran and re-pushing a tag
# cannot change a running pipeline.
data "aws_ecr_image" "ml_v2_training" {
  repository_name = aws_ecr_repository.images["autoencoder_training"].name
  image_tag       = var.ml_v2_training_image_tag
  most_recent     = var.ml_v2_training_image_tag == null ? true : null
}

data "aws_ecr_image" "ml_v2_inference" {
  repository_name = aws_ecr_repository.images["autoencoder_inference"].name
  image_tag       = var.ml_v2_inference_image_tag
  most_recent     = var.ml_v2_inference_image_tag == null ? true : null
}

locals {
  ml_v2_name = "${var.project}-${var.environment}-trust-score-ml"

  ml_v2_training_sm_name = "${local.ml_v2_name}-training"
  ml_v2_scoring_sm_name  = "${local.ml_v2_name}-scoring"
  # Built from the name so the training machine can list its own executions
  # without referring to itself.
  ml_v2_training_sm_arn = "arn:${data.aws_partition.current.partition}:states:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:stateMachine:${local.ml_v2_training_sm_name}"

  # Existing buckets from s3.tf.
  ml_v2_input_bucket  = aws_s3_bucket.sagemaker.bucket
  ml_v2_output_bucket = aws_s3_bucket.sagemaker.bucket
  ml_v2_output_root   = "s3://${local.ml_v2_output_bucket}/${var.ml_v2_output_prefix}"

  ml_v2_training_image  = "${aws_ecr_repository.images["autoencoder_training"].repository_url}@${data.aws_ecr_image.ml_v2_training.image_digest}"
  ml_v2_inference_image = "${aws_ecr_repository.images["autoencoder_inference"].repository_url}@${data.aws_ecr_image.ml_v2_inference.image_digest}"

  # Jobs are created by Step Functions at run time, so the provider's
  # default_tags never reach them; they carry the same tags explicitly.
  ml_v2_job_tags = [
    { Key = "Project", Value = var.project },
    { Key = "Stage", Value = var.environment },
    { Key = "ManagedBy", Value = "terraform" },
    { Key = "Pipeline", Value = "trust-score-ml" },
  ]

  ml_v2_vpc_config = length(var.ml_v2_job_subnet_ids) > 0 ? {
    Subnets          = var.ml_v2_job_subnet_ids
    SecurityGroupIds = var.ml_v2_job_security_group_ids
  } : null
}
