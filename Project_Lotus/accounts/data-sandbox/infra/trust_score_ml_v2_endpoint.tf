# =============================================================================
# VERSION 2 (add-on) - Trust Score ML: Serverless inference endpoint (Flask)
# =============================================================================
# Added in v2; no existing file in this root was modified. See ../CHANGES_V2.md.
#
# The inference image (sagemaker/Dockerfile.inference) serves the champion
# model through sagemaker/serve.py when SageMaker starts it with `serve`:
# GET /ping, POST /invocations on port 8080. Same image and same scoring code
# as the nightly batch job, so both give a row the same score.
#
# Serverless Inference: billed per request and per GB-second while a request
# runs, scales to zero when idle - no always-on instance. Limits to know:
#   * memory 1024-6144 MB (the model libraries need ~2-3 GB; 4096 is the default)
#   * cold start of several seconds after an idle period (provisioned
#     concurrency removes it, at an hourly cost)
#   * no VPC attachment
#
# The server loads champion/current.json on first use and re-checks it every
# ml_v2_endpoint_champion_refresh_seconds, so a newly promoted champion is
# picked up without a redeploy. Pushing a new inference IMAGE still needs
# `terraform apply` (the model is pinned to the image digest).
# =============================================================================

variable "ml_v2_enable_serverless_endpoint" {
  description = "Create the Serverless inference endpoint. It can be created before the first champion exists (/invocations answers 503 until one is published)."
  type        = bool
  default     = true
}

variable "ml_v2_endpoint_memory_mb" {
  description = "Serverless memory per concurrent invocation: 1024, 2048, 3072, 4096, 5120 or 6144."
  type        = number
  default     = 4096

  validation {
    condition     = contains([1024, 2048, 3072, 4096, 5120, 6144], var.ml_v2_endpoint_memory_mb)
    error_message = "ml_v2_endpoint_memory_mb must be one of 1024, 2048, 3072, 4096, 5120, 6144."
  }
}

variable "ml_v2_endpoint_max_concurrency" {
  description = "Maximum concurrent invocations (1-200). Caps both throughput and cost."
  type        = number
  default     = 5

  validation {
    condition     = var.ml_v2_endpoint_max_concurrency >= 1 && var.ml_v2_endpoint_max_concurrency <= 200
    error_message = "ml_v2_endpoint_max_concurrency must be between 1 and 200."
  }
}

variable "ml_v2_endpoint_provisioned_concurrency" {
  description = "Warm instances kept ready (removes cold starts, billed hourly). null = none, pure pay-per-request."
  type        = number
  default     = null
}

variable "ml_v2_endpoint_champion_refresh_seconds" {
  description = "How often the server re-checks champion/current.json for a newly promoted model."
  type        = number
  default     = 300
}

variable "ml_v2_endpoint_max_rows_per_request" {
  description = "Rows allowed in one /invocations request. Larger volumes belong to the nightly batch pipeline."
  type        = number
  default     = 10000
}

locals {
  ml_v2_endpoint_name = "${local.ml_v2_name}-serverless"

  ml_v2_endpoint_environment = {
    TS05_CONFIG_FILES             = join(",", concat(var.ml_v2_config_files)) # v2.1: + the rendered pipeline config
    TS05_CONFIG_OVERRIDES         = join(";", var.ml_v2_config_overrides)
    TS05_CHAMPION_REFRESH_SECONDS = tostring(var.ml_v2_endpoint_champion_refresh_seconds)
    TS05_MAX_ROWS_PER_REQUEST     = tostring(var.ml_v2_endpoint_max_rows_per_request)
    AWS_REGION                    = data.aws_region.current.region
    AWS_DEFAULT_REGION            = data.aws_region.current.region
  }

  # Models and endpoint configurations are immutable in SageMaker; a new image
  # or setting needs new ones. The suffix changes exactly when they must.
  ml_v2_endpoint_suffix = substr(sha1(jsonencode({
    image  = local.ml_v2_inference_image
    env    = local.ml_v2_endpoint_environment
    memory = var.ml_v2_endpoint_memory_mb
    conc   = var.ml_v2_endpoint_max_concurrency
    prov   = var.ml_v2_endpoint_provisioned_concurrency
  })), 0, 10)
}

resource "aws_sagemaker_model" "ml_v2_serve" {
  count = var.ml_v2_enable_serverless_endpoint ? 1 : 0

  name               = "${local.ml_v2_name}-serve-${local.ml_v2_endpoint_suffix}"
  execution_role_arn = data.aws_iam_role.ml_v2_sagemaker_execution.arn

  primary_container {
    image       = local.ml_v2_inference_image
    mode        = "SingleModel"
    environment = local.ml_v2_endpoint_environment
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_sagemaker_endpoint_configuration" "ml_v2_serve" {
  count = var.ml_v2_enable_serverless_endpoint ? 1 : 0

  name = "${local.ml_v2_name}-serve-${local.ml_v2_endpoint_suffix}"

  production_variants {
    variant_name = "champion"
    model_name   = aws_sagemaker_model.ml_v2_serve[0].name

    serverless_config {
      memory_size_in_mb       = var.ml_v2_endpoint_memory_mb
      max_concurrency         = var.ml_v2_endpoint_max_concurrency
      provisioned_concurrency = var.ml_v2_endpoint_provisioned_concurrency
    }
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_sagemaker_endpoint" "ml_v2_serve" {
  count = var.ml_v2_enable_serverless_endpoint ? 1 : 0

  name                 = local.ml_v2_endpoint_name
  endpoint_config_name = aws_sagemaker_endpoint_configuration.ml_v2_serve[0].name
}

output "ml_v2_endpoint_name" {
  description = "VERSION 2: Serverless inference endpoint. Invoke: aws sagemaker-runtime invoke-endpoint --endpoint-name <this> --content-type application/json --body fileb://rows.json out.json"
  value       = var.ml_v2_enable_serverless_endpoint ? aws_sagemaker_endpoint.ml_v2_serve[0].name : null
}
