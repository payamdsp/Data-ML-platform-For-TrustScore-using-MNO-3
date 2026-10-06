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

variable "permissions_boundary_ssm_path" {
  description = "Contract parameter holding the permissions boundary ARN applied to every role this root creates. Section 4.4 requires a boundary in data-sandbox and publishes its ARN here. Set to null in accounts that use no boundary, which is every stage other than sandbox."
  type        = string
  default     = "/contract/_account/iam/sandbox-boundary-arn"
}
