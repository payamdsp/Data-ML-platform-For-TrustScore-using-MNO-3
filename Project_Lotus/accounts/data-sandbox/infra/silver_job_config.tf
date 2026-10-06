# The Silver job's spark-submit arguments, as a console-editable S3 object.
# Same pattern as gold_job_config.tf - see that file's header comment for why
# this has to be a Lambda-read S3 object rather than something the state
# machine assembles on its own.
#
# Unlike Gold's config, this one's `datasets` map is CONTENT Terraform
# computes from real resources (aws_glue_catalog_table.silver_quarantine_*,
# local.quarantine_*_location) rather than a hand-authored JSON file. An
# earlier version of local.silver_dataset_specs (stepfunctions_bronze_silver.tf)
# hand-derived a `<name>_quarantine` naming formula that silently diverged
# from what those resources actually provisioned - see that file's comment.
# Computing this object's seed content from the same resources those locals
# already reference means the first version ever written to S3 is guaranteed
# to match reality, by construction, not by someone copying values correctly.
#
# Terraform creates this object ONCE and then leaves it alone (see the
# lifecycle block) - editing it directly in the S3 console is the intended way
# to change an argument without a Terraform apply or a Lambda redeploy, and a
# later `terraform apply` for some unrelated resource must not silently revert
# that edit back to this computed seed. The tradeoff this accepts: the
# deployed object and what this file would compute today can drift apart,
# permanently, until someone deliberately re-seeds it (taint this resource).

locals {
  silver_job_config_bucket = var.silver_bucket_name
  silver_job_config_key    = "pipeline-code/silver/job_config.json"
  silver_job_config_uri    = "s3://${local.silver_job_config_bucket}/${local.silver_job_config_key}"

  silver_job_config_content = {
    application    = "${trimsuffix(var.silver_gold_state_machine_artifact_root, "/")}/run_trust_score_silver.py"
    python_package = "${trimsuffix(var.silver_gold_state_machine_artifact_root, "/")}/silver_pipeline.zip"
    artifact_root  = var.silver_artifact_root
    mode           = "publish"

    # No resource-sizing --driver-memory/--executor-memory/--conf list has
    # been confirmed for Silver yet, unlike Gold's - spark_confs starts empty
    # rather than guessed. Add entries here (or via the console, once this
    # object exists) when that's known; nothing else has to change.
    spark_submit_flags = ["--master", "yarn", "--deploy-mode", "cluster"]
    spark_confs        = []

    datasets = local.silver_dataset_specs
  }
}

resource "aws_s3_object" "silver_job_config" {
  bucket       = local.silver_job_config_bucket
  key          = local.silver_job_config_key
  content      = jsonencode(local.silver_job_config_content)
  content_type = "application/json"

  # Seed once from the computed content above; never re-push. See the header
  # comment for why - this is the same tradeoff gold_job_config.tf accepts.
  lifecycle {
    ignore_changes = [content]
  }
}

output "silver_job_config_uri" {
  description = "s3:// URI of the console-editable Silver job argument config. Edit this object directly to change spark-submit arguments without a Terraform apply."
  value       = local.silver_job_config_uri
}
