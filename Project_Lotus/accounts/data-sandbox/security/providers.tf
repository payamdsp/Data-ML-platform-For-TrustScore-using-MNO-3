provider "aws" {
  region = "ca-central-1"

  default_tags {
    tags = {
      Project   = var.project
      Stage     = var.environment
      ManagedBy = "terraform"
    }
  }
}
