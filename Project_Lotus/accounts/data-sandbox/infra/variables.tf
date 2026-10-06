variable "project" {
  description = "Project name. Prefixes every resource in this root, which is what keeps two projects sharing an account from colliding."
  type        = string
  default     = "lotus"
}

variable "environment" {
  description = "Deployment stage. Drives resource naming and every lookup. This is the single value that changes on promotion to another stage."
  type        = string
  default     = "sandbox"
}

variable "permissions_boundary" {
  description = "Permissions boundary applied to roles created in this account, published at /contract/_account/iam/sandbox-boundary-arn. Sandbox passes the ARN; every other stage passes null."
  type        = string
  default     = null
}

# variable "emr_vpc_id_ssm_path" {
#   description = "SSM parameter containing the VPC ID used for EMR and SageMaker resources."
#   type        = string
#   default     = "/contract/_account/network/vpc-id"
# }

# variable "emr_subnet_ids_ssm_path" {
#   description = "SSM parameter containing the subnet IDs available for EMR instance fleets."
#   type        = string
#   default     = "/contract/_account/network/subnet-ids"
# }

variable "emr_vpc_id" {
  description = "VPC ID in which to create the EMR security groups."
  type        = string
  default     = "vpc-0da21763de8d910b6"
}

variable "emr_subnet_ids" {
  description = "Subnet IDs instance fleets may launch into. Provide more than one (across AZs) to get the Spot-availability benefit of fleets."
  type        = list(string)
  default     = ["subnet-0c92f32a66b2ee308", "subnet-0b2754e35f0753ed3"]
}

variable "preprocessing_image_uri" {
  description = "ECR image URI to use for the SageMaker preprocessing job."
  type        = string
  default     = "162591926854.dkr.ecr.ca-central-1.amazonaws.com/lotus-sandbox-preprocessing:e14eec5c57c62f21bba5d95f101b90fff848e5e9"
}

#-------------------iceberg----------------------------------

variable "silver_bucket_name" {
  description = "Existing S3 bucket that stores the conformed Silver Iceberg data."
  type        = string
  default     = "lotus-sandbox-silver-conformed-data"
}

variable "table_root_prefix" {
  description = "S3 prefix under the Silver bucket containing Iceberg table directories."
  type        = string
  default     = "tables"

  validation {
    condition     = length(trim(var.table_root_prefix, "/")) > 0
    error_message = "table_root_prefix must not be empty."
  }
}

variable "iceberg_warehouse_uri" {
  description = "S3 warehouse prefix used by the Glue-backed Iceberg catalog."
  type        = string
  default     = ""

  validation {
    condition     = var.iceberg_warehouse_uri == "" || startswith(var.iceberg_warehouse_uri, "s3://")
    error_message = "iceberg_warehouse_uri must be empty or an s3:// URI."
  }
}

variable "glue_database_name" {
  description = "Lowercase Glue database name used by the Spark catalog."
  type        = string
  default     = "lotus_sandbox_silver_conformed"
}

variable "table_version" {
  description = "Published Silver schema version used in table locations."
  type        = number
  default     = 1

  validation {
    condition     = var.table_version >= 1 && floor(var.table_version) == var.table_version
    error_message = "table_version must be a positive whole number."
  }
}

variable "iceberg_format_version" {
  description = "Iceberg format version. v2 is the broadly compatible EMR default."
  type        = number
  default     = 2

  validation {
    condition     = contains([2, 3], var.iceberg_format_version)
    error_message = "iceberg_format_version must be 2 or 3."
  }
}

variable "enable_schema_evolution" {
  description = "Allow Spark to add new columns while retaining existing snapshots."
  type        = bool
  default     = true
}

variable "create_glue_database" {
  description = "Create the Silver Glue database. Keep true (default). If the database already exists in AWS but not in Terraform state, run: terraform import 'aws_glue_catalog_database.silver[0]' <account_id>:<database_name>"
  type        = bool
  default     = true
}