# Provisions three Silver record-level quarantine Iceberg tables — one per
# dataset — inside the existing silver Glue database.  Using the shared
# database means the existing LakeFormation wildcard grants already cover
# these tables; no additional LF or IAM resources are required.
#
# Schema per table:
#   Part 1 — "Why it failed" reason columns (12 cols, field IDs 1-12):
#             quarantine_id, run_id, run_date, validation_stage,
#             error_rule_ids_json, warning_rule_ids_json, error_details_json,
#             raw_payload, source_file, quarantined_at, quarantined_date,
#             recovery_status.
#             Field IDs 13-100 reserved for future reason column additions.
#   Part 2 — Full Silver output columns for this dataset (field IDs 101+):
#             Exact column names and types from the Silver YAML output schema.
#             record_id uses the correct type per dataset (long for AC/DL,
#             string for ATS). No cross-dataset columns or NULL padding.
#
# Tables:
#   silver_quarantine_account_changes_batch  (26 cols = 12 reason + 14 AC)
#   silver_quarantine_device_lookup_batch    (32 cols = 12 reason + 20 DL)
#   silver_quarantine_audit_trail_services_3 (36 cols = 12 reason + 24 ATS)

locals {
  # ------------------------------------------------------------------
  # S3 locations — one prefix, three table paths
  # ------------------------------------------------------------------
  _quarantine_base        = "s3://${var.silver_bucket_name}/quarantine/iceberg"
  quarantine_ac_location  = "${local._quarantine_base}/silver_quarantine_account_changes_batch/"
  quarantine_dl_location  = "${local._quarantine_base}/silver_quarantine_device_lookup_batch/"
  quarantine_ats_location = "${local._quarantine_base}/silver_quarantine_audit_trail_services_3/"

  # ------------------------------------------------------------------
  # Part 1: "Why it failed" reason columns — identical for all three tables.
  # Field IDs 1-12 must never be renumbered after first apply.
  # ------------------------------------------------------------------
  _quarantine_reason_fields = [
    { id = 1, name = "quarantine_id", type = "string", doc = "SHA-256 unique entry ID per (run, record_id, validation_stage, errors)." },
    { id = 2, name = "run_id", type = "string", doc = "Silver run that quarantined this record." },
    { id = 3, name = "run_date", type = "date", doc = "Requested Toronto business date." },
    { id = 4, name = "validation_stage", type = "string", doc = "RAW_DQ | CANDIDATE_DQ | TARGET_CONFLICT." },
    { id = 5, name = "error_rule_ids_json", type = "string", doc = "JSON array of blocking DQ rule IDs that fired." },
    { id = 6, name = "warning_rule_ids_json", type = "string", doc = "JSON array of co-occurring warning rule IDs." },
    { id = 7, name = "error_details_json", type = "string", doc = "JSON array of {rule_id, severity} objects." },
    { id = 8, name = "raw_payload", type = "string", doc = "Original Bronze row as JSON; may contain PII." },
    { id = 9, name = "source_file", type = "string", doc = "Bronze input file URI." },
    { id = 10, name = "quarantined_at", type = "timestamp", doc = "UTC time this entry was written." },
    { id = 11, name = "quarantined_date", type = "date", doc = "Iceberg identity partition column." },
    { id = 12, name = "recovery_status", type = "string", doc = "PENDING_REMEDIATION until operator resolves." },
    # IDs 13-100 reserved for future reason column additions.
  ]

  # ------------------------------------------------------------------
  # Part 2a: account_changes_batch Silver output columns (IDs 101-114)
  # ------------------------------------------------------------------
  _quarantine_ac_silver_fields = [
    { id = 101, name = "record_id", type = "long", doc = "Source record primary key." },
    { id = 102, name = "phone_number_AC_hash", type = "string", doc = "Account-change phone hash (change_key)." },
    { id = 103, name = "event_type", type = "string", doc = "Normalized event type." },
    { id = 104, name = "event_timestamp", type = "timestamp", doc = "Event time in Toronto timezone." },
    { id = 105, name = "mno", type = "string", doc = "Inferred or explicit mobile-network operator." },
    { id = 106, name = "subId", type = "string", doc = "Subscriber ID from notes (BELL)." },
    { id = 107, name = "msisdnChangeSide", type = "string", doc = "MSISDN change side from notes (BELL)." },
    { id = 108, name = "correlationId", type = "string", doc = "Correlation ID from notes (BELL)." },
    { id = 109, name = "otherMSISDN", type = "string", doc = "Other MSISDN from notes (BELL)." },
    { id = 110, name = "ingestion_ts", type = "timestamp", doc = "Silver ingestion timestamp." },
    { id = 111, name = "source_name", type = "string", doc = "Declared source lineage name." },
    { id = 112, name = "source_event_id", type = "long", doc = "Source record pointer (same as record_id for AC)." },
    { id = 113, name = "schema_version", type = "int", doc = "Silver YAML schema version." },
    { id = 114, name = "event_date", type = "date", doc = "Date derived from event_timestamp." },
  ]

  # ------------------------------------------------------------------
  # Part 2b: device_lookup_batch Silver output columns (IDs 101-120)
  # ------------------------------------------------------------------
  _quarantine_dl_silver_fields = [
    { id = 101, name = "record_id", type = "long", doc = "Source record primary key." },
    { id = 102, name = "phone_number_AC_hash", type = "string", doc = "Account-change phone hash (change_key)." },
    { id = 103, name = "phone_number_AT_hash", type = "string", doc = "Account-token phone hash (change_key_ats)." },
    { id = 104, name = "mno", type = "string", doc = "Mobile-network operator." },
    { id = 105, name = "imei", type = "string", doc = "Cleaned device IMEI." },
    { id = 106, name = "imsi", type = "string", doc = "Cleaned subscriber IMSI." },
    { id = 107, name = "date", type = "date", doc = "Source event date." },
    { id = 108, name = "event_timestamp", type = "timestamp", doc = "Event time in Toronto timezone." },
    { id = 109, name = "correlation_id", type = "string", doc = "Correlation identifier from notes." },
    { id = 110, name = "related_id", type = "string", doc = "Related source record ID." },
    { id = 111, name = "event_type", type = "string", doc = "Normalized event type." },
    { id = 112, name = "tac", type = "string", doc = "Type Allocation Code from notes." },
    { id = 113, name = "oldIMSI", type = "string", doc = "Previous IMSI from notes." },
    { id = 114, name = "oldIMEI", type = "string", doc = "Previous IMEI from notes." },
    { id = 115, name = "fromADCSnapshot", type = "string", doc = "ADC snapshot marker from notes." },
    { id = 116, name = "ingestion_ts", type = "timestamp", doc = "Silver ingestion timestamp." },
    { id = 117, name = "source_name", type = "string", doc = "Declared source lineage name." },
    { id = 118, name = "source_event_id", type = "long", doc = "Source record pointer (same as record_id for DL)." },
    { id = 119, name = "schema_version", type = "int", doc = "Silver YAML schema version." },
    { id = 120, name = "event_date", type = "date", doc = "Date derived from event_timestamp." },
  ]

  # ------------------------------------------------------------------
  # Part 2c: audit_trail_services_3 Silver output columns (IDs 101-124)
  # ------------------------------------------------------------------
  _quarantine_ats_silver_fields = [
    { id = 101, name = "record_id", type = "string", doc = "Source record primary key (request_id)." },
    { id = 102, name = "operation", type = "string", doc = "API operation name." },
    { id = 103, name = "mno", type = "string", doc = "Mobile-network operator (optional for ATS)." },
    { id = 104, name = "msisdn", type = "string", doc = "Raw MSISDN." },
    { id = 105, name = "phone_number_AT_hash", type = "string", doc = "Account-token phone hash." },
    { id = 106, name = "encrypted_msisdn", type = "string", doc = "Encrypted MSISDN." },
    { id = 107, name = "partner_id", type = "string", doc = "Partner identifier." },
    { id = 108, name = "partner_name", type = "string", doc = "Partner reference name (from partners CSV)." },
    { id = 109, name = "service_provider_id", type = "string", doc = "Service-provider identifier." },
    { id = 110, name = "service_provider_name", type = "string", doc = "Service-provider name (from providers CSV)." },
    { id = 111, name = "response_code", type = "string", doc = "API response code." },
    { id = 112, name = "source_ip", type = "string", doc = "Caller source IP." },
    { id = 113, name = "api_timestamp", type = "timestamp", doc = "API event time in Toronto timezone." },
    { id = 114, name = "date", type = "date", doc = "Source event date." },
    { id = 115, name = "timestamp_ms", type = "timestamp", doc = "Millisecond-precision event timestamp." },
    { id = 116, name = "brand", type = "string", doc = "Normalized source brand." },
    { id = 117, name = "industry", type = "string", doc = "Service-provider industry." },
    { id = 118, name = "api_type", type = "string", doc = "Versioned API operation classification." },
    { id = 119, name = "processing_time_ms", type = "long", doc = "Request processing duration in ms." },
    { id = 120, name = "correlation_id", type = "string", doc = "Correlation identifier." },
    { id = 121, name = "source_name", type = "string", doc = "Declared source lineage name." },
    { id = 122, name = "source_event_id", type = "string", doc = "Source record pointer (request_id)." },
    { id = 123, name = "schema_version", type = "int", doc = "Silver YAML schema version." },
    { id = 124, name = "event_date", type = "date", doc = "Date derived from api_timestamp." },
  ]

  # Combined field lists — reason cols first, then Silver cols.
  quarantine_ac_fields  = concat(local._quarantine_reason_fields, local._quarantine_ac_silver_fields)
  quarantine_dl_fields  = concat(local._quarantine_reason_fields, local._quarantine_dl_silver_fields)
  quarantine_ats_fields = concat(local._quarantine_reason_fields, local._quarantine_ats_silver_fields)

  _quarantine_tblproperties = {
    "format-version"                  = tostring(var.iceberg_format_version)
    "write.format.default"            = "parquet"
    "write.parquet.compression-codec" = "zstd"
    "write.spark.accept-any-schema"   = "false"
  }
}


# ---------------------------------------------------------------------------
# Table 1: account_changes_batch quarantine (26 cols)
# ---------------------------------------------------------------------------

resource "aws_glue_catalog_table" "silver_quarantine_account_changes_batch" {
  depends_on    = [aws_glue_catalog_database.silver]
  catalog_id    = data.aws_caller_identity.current.account_id
  database_name = var.glue_database_name
  name          = "silver_quarantine_account_changes_batch"
  description   = "Rejected account_changes_batch rows: failure reason + full typed Silver output columns."

  open_table_format_input {
    iceberg_input {
      metadata_operation = "CREATE"
      version            = var.iceberg_format_version

      iceberg_table_input {
        location   = local.quarantine_ac_location
        properties = local._quarantine_tblproperties

        schema {
          schema_id            = 0
          type                 = "struct"
          identifier_field_ids = []

          dynamic "fields" {
            for_each = local.quarantine_ac_fields
            content {
              id       = fields.value.id
              name     = fields.value.name
              required = false
              type     = jsonencode(fields.value.type)
              doc      = fields.value.doc
            }
          }
        }

        partition_spec {
          spec_id = 0
          # Partition by quarantined_date (field id=11) to prune investigation
          # queries by date without a full table scan.
          fields {
            field_id  = 1000
            name      = "quarantined_date"
            source_id = 11
            transform = "identity"
          }
        }
      }
    }
  }

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [open_table_format_input]
  }
}


# ---------------------------------------------------------------------------
# Table 2: device_lookup_batch quarantine (32 cols)
# ---------------------------------------------------------------------------

resource "aws_glue_catalog_table" "silver_quarantine_device_lookup_batch" {
  depends_on    = [aws_glue_catalog_database.silver]
  catalog_id    = data.aws_caller_identity.current.account_id
  database_name = var.glue_database_name
  name          = "silver_quarantine_device_lookup_batch"
  description   = "Rejected device_lookup_batch rows: failure reason + full typed Silver output columns."

  open_table_format_input {
    iceberg_input {
      metadata_operation = "CREATE"
      version            = var.iceberg_format_version

      iceberg_table_input {
        location   = local.quarantine_dl_location
        properties = local._quarantine_tblproperties

        schema {
          schema_id            = 0
          type                 = "struct"
          identifier_field_ids = []

          dynamic "fields" {
            for_each = local.quarantine_dl_fields
            content {
              id       = fields.value.id
              name     = fields.value.name
              required = false
              type     = jsonencode(fields.value.type)
              doc      = fields.value.doc
            }
          }
        }

        partition_spec {
          spec_id = 0
          fields {
            field_id  = 1000
            name      = "quarantined_date"
            source_id = 11
            transform = "identity"
          }
        }
      }
    }
  }

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [open_table_format_input]
  }
}


# ---------------------------------------------------------------------------
# Table 3: audit_trail_services_3 quarantine (36 cols)
# ---------------------------------------------------------------------------

resource "aws_glue_catalog_table" "silver_quarantine_audit_trail_services_3" {
  depends_on    = [aws_glue_catalog_database.silver]
  catalog_id    = data.aws_caller_identity.current.account_id
  database_name = var.glue_database_name
  name          = "silver_quarantine_audit_trail_services_3"
  description   = "Rejected audit_trail_services_3 rows: failure reason + full typed Silver output columns."

  open_table_format_input {
    iceberg_input {
      metadata_operation = "CREATE"
      version            = var.iceberg_format_version

      iceberg_table_input {
        location   = local.quarantine_ats_location
        properties = local._quarantine_tblproperties

        schema {
          schema_id            = 0
          type                 = "struct"
          identifier_field_ids = []

          dynamic "fields" {
            for_each = local.quarantine_ats_fields
            content {
              id       = fields.value.id
              name     = fields.value.name
              required = false
              type     = jsonencode(fields.value.type)
              doc      = fields.value.doc
            }
          }
        }

        partition_spec {
          spec_id = 0
          fields {
            field_id  = 1000
            name      = "quarantined_date"
            source_id = 11
            transform = "identity"
          }
        }
      }
    }
  }

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [open_table_format_input]
  }
}


# ---------------------------------------------------------------------------
# Outputs — pass these to EMR spark-submit arguments per dataset
# ---------------------------------------------------------------------------

output "quarantine_ac_iceberg_table" {
  description = "AC quarantine table; pass as --quarantine-table for account_changes_batch runs."
  value       = "glue_catalog.${var.glue_database_name}.${aws_glue_catalog_table.silver_quarantine_account_changes_batch.name}"
}

output "quarantine_ac_iceberg_location" {
  description = "AC quarantine S3 location; pass as --quarantine-table-location."
  value       = local.quarantine_ac_location
}

output "quarantine_dl_iceberg_table" {
  description = "DL quarantine table; pass as --quarantine-table for device_lookup_batch runs."
  value       = "glue_catalog.${var.glue_database_name}.${aws_glue_catalog_table.silver_quarantine_device_lookup_batch.name}"
}

output "quarantine_dl_iceberg_location" {
  description = "DL quarantine S3 location; pass as --quarantine-table-location."
  value       = local.quarantine_dl_location
}

output "quarantine_ats_iceberg_table" {
  description = "ATS quarantine table; pass as --quarantine-table for audit_trail_services_3 runs."
  value       = "glue_catalog.${var.glue_database_name}.${aws_glue_catalog_table.silver_quarantine_audit_trail_services_3.name}"
}

output "quarantine_ats_iceberg_location" {
  description = "ATS quarantine S3 location; pass as --quarantine-table-location."
  value       = local.quarantine_ats_location
}
