terraform {
  backend "s3" {
    bucket       = "enstream-tfstate-data-sandbox"
    key          = "security/lotus/terraform.tfstate"
    region       = "ca-central-1"
    encrypt      = true
    use_lockfile = true
  }
}
