"""Durable, record-level Iceberg quarantine for Silver validation failures.

The normal Silver tables contain only conforming records.  This module keeps a
single recoverable, queryable entry for each record that failed an error rule.
It deliberately complements—not replaces—the Bronze schema-validation Lambda:
files rejected before Spark reads them remain a Bronze control-plane concern.
"""

import re

from pyspark.sql import functions as F

from .common import quoted
from .publish import identifier


# "Why it failed" columns — identical across all three quarantine tables.
# Kept intentionally small: everything an operator needs to understand and
# remediate the failure without navigating forensic envelope fields.
_QUARANTINE_REASON = {
    "quarantine_id": "string",        # SHA-256 per run, stage, and source-row evidence
    "run_id": "string",               # silver run that quarantined this record
    "run_date": "date",               # toronto business date
    "validation_stage": "string",     # RAW_DQ | CANDIDATE_DQ | TARGET_CONFLICT
    "error_rule_ids_json": "string",  # ["rule_id1", ...]  — blocking rules that fired
    "warning_rule_ids_json": "string",# ["rule_id1", ...]  — co-occurring warnings
    "error_details_json": "string",   # [{rule_id, severity}, ...]
    "raw_payload": "string",          # original bronze row as JSON
    "source_file": "string",          # bronze input file URI
    "quarantined_at": "timestamp",    # utc time this entry was written
    "quarantined_date": "date",       # iceberg partition column
    "recovery_status": "string",      # PENDING_REMEDIATION until operator resolves
}

# Full Silver output columns per dataset — exact names and types from the YAML.
# record_id uses the correct type per dataset (bigint for AC/DL, string for ATS).

_AC_SILVER_COLUMNS = {
    "record_id": "bigint",
    "phone_number_AC_hash": "string",
    "event_type": "string",
    "event_timestamp": "timestamp",
    "mno": "string",
    "subId": "string",
    "msisdnChangeSide": "string",
    "correlationId": "string",
    "otherMSISDN": "string",
    "ingestion_ts": "timestamp",
    "source_name": "string",
    "source_event_id": "bigint",
    "schema_version": "int",
    "event_date": "date",
}

_DL_SILVER_COLUMNS = {
    "record_id": "bigint",
    "phone_number_AC_hash": "string",
    "phone_number_AT_hash": "string",
    "mno": "string",
    "imei": "string",
    "imsi": "string",
    "date": "date",
    "event_timestamp": "timestamp",
    "correlation_id": "string",
    "related_id": "string",
    "event_type": "string",
    "tac": "string",
    "oldIMSI": "string",
    "oldIMEI": "string",
    "fromADCSnapshot": "string",
    "ingestion_ts": "timestamp",
    "source_name": "string",
    "source_event_id": "bigint",
    "schema_version": "int",
    "event_date": "date",
}

_ATS_SILVER_COLUMNS = {
    "record_id": "string",
    "operation": "string",
    "mno": "string",
    "msisdn": "string",
    "phone_number_AT_hash": "string",
    "encrypted_msisdn": "string",
    "partner_id": "string",
    "partner_name": "string",
    "service_provider_id": "string",
    "service_provider_name": "string",
    "response_code": "string",
    "source_ip": "string",
    "api_timestamp": "timestamp",
    "date": "date",
    "timestamp_ms": "timestamp",
    "brand": "string",
    "industry": "string",
    "api_type": "string",
    "processing_time_ms": "bigint",
    "correlation_id": "string",
    "source_name": "string",
    "source_event_id": "string",
    "schema_version": "int",
    "event_date": "date",
}

# Combined schema = reason cols + full Silver output cols.
QUARANTINE_ACCOUNT_CHANGES_BATCH = {**_QUARANTINE_REASON, **_AC_SILVER_COLUMNS}
QUARANTINE_DEVICE_LOOKUP_BATCH = {**_QUARANTINE_REASON, **_DL_SILVER_COLUMNS}
QUARANTINE_AUDIT_TRAIL_SERVICES_3 = {**_QUARANTINE_REASON, **_ATS_SILVER_COLUMNS}

# Keyed by the --dataset argument so ensure_quarantine_table() and
# write_quarantine() can resolve the expected schema without extra parameters.
DATASET_QUARANTINE_SCHEMA = {
    "account_changes_batch": QUARANTINE_ACCOUNT_CHANGES_BATCH,
    "device_lookup_batch": QUARANTINE_DEVICE_LOOKUP_BATCH,
    "audit_trail_services_3": QUARANTINE_AUDIT_TRAIL_SERVICES_3,
}


def _s3_location(location):
    """Validate a location before including it in the optional bootstrap DDL."""

    if not location.startswith("s3://") or "'" in location or "\n" in location:
        raise ValueError(
            "quarantine table location must be an s3:// URI without SQL quote/newline characters"
        )
    return location.rstrip("/") + "/"


def _comparable_type(data_type):
    """Treat Spark's two timestamp renderings as the same approved contract."""

    # Spark 3.5 can expose an Iceberg `timestamp` field as `timestamp_ntz`.
    # The Terraform contract still correctly declares it as a timestamp.
    return "timestamp" if data_type in {"timestamp", "timestamp_ntz"} else data_type


def _case_insensitive_field_types(fields):
    """Compare Glue/Iceberg fields by name without depending on letter case.

    Glue may return ``subId`` as ``subid`` even though Terraform and the YAML
    use the client-facing spelling. Spark resolves these columns without case
    sensitivity, and the write path aliases each value to the target's actual
    field name. Reject case-colliding names so this comparison cannot silently
    accept an ambiguous schema.
    """

    comparable = {}
    for name, data_type in fields.items():
        key = name.casefold()
        if key in comparable:
            raise ValueError(f"Quarantine table schema has ambiguous field names: {name}")
        comparable[key] = _comparable_type(data_type)
    return comparable


def _table_details(spark, name):
    """Return the key/value metadata emitted by DESCRIBE TABLE EXTENDED."""

    return {
        row["col_name"].strip(): row["data_type"].strip()
        for row in spark.sql(f"DESCRIBE TABLE EXTENDED {name}").collect()
        if row["col_name"] and row["data_type"]
    }


def ensure_quarantine_table(spark, table, location, dataset, create_if_missing=False):
    """Create an isolated smoke-test table or validate the Terraform table.

    Publish mode must use a Terraform-created table.  The runner exposes
    ``create_if_missing`` only for an explicitly requested validate-mode smoke
    test.  Validation is strict because a same-named table at another location
    could have a different retention or Lake Formation access policy.
    """

    name = identifier(table)
    location = _s3_location(location)
    exists = spark.catalog.tableExists(table)

    if not exists and not create_if_missing:
        raise ValueError(
            "Quarantine table does not exist; provision it through Terraform before publishing"
        )

    schema = DATASET_QUARANTINE_SCHEMA[dataset]
    if not exists:
        columns = ", ".join(
            f"{quoted(column)} {data_type.upper()}"
            for column, data_type in schema.items()
        )
        # Use a date derived from the job time, rather than event_date.  A bad
        # event timestamp is a common reason for quarantine and may be null.
        spark.sql(
            f"CREATE TABLE {name} ({columns}) USING iceberg "
            f"PARTITIONED BY ({quoted('quarantined_date')}) "
            f"LOCATION '{location}' "
            "TBLPROPERTIES ("
            "'format-version'='2', "
            "'write.format.default'='parquet', "
            "'write.parquet.compression-codec'='zstd'"
            ")"
        )

    properties = {
        row["key"]: row["value"]
        for row in spark.sql(f"SHOW TBLPROPERTIES {name}").collect()
    }
    if properties.get("format-version") != "2":
        raise ValueError("Quarantine target must be an Iceberg format-version 2 table")

    actual_fields = {
        field.name: _comparable_type(field.dataType.simpleString())
        for field in spark.table(table).schema.fields
    }
    # The Glue catalog can expose mixed-case Terraform field names in lower
    # case. This is harmless with Spark's case-insensitive resolution, but a
    # missing/extra field or changed type must still fail the preflight.
    if _case_insensitive_field_types(actual_fields) != _case_insensitive_field_types(schema):
        raise ValueError(
            "Quarantine table schema differs from the reviewed contract: "
            f"actual={actual_fields}; expected={schema}"
        )

    actual_location = _table_details(spark, name).get("Location")
    if not actual_location or actual_location.rstrip("/") + "/" != location:
        raise ValueError(
            "Quarantine table location differs from the reviewed location: "
            f"{actual_location}"
        )

    # A correct schema alone is not enough.  Verify the identity partition so
    # investigation queries can prune by quarantined_date.
    create_row = spark.sql(f"SHOW CREATE TABLE {name}").first()
    create_statement = create_row[0] if create_row else ""
    normalized_ddl = re.sub(r"\s+", " ", create_statement).lower()
    if not re.search(
        r"partitioned by \(\s*`?quarantined_date`?\s*\)", normalized_ddl
    ):
        raise ValueError("Quarantine table must be partitioned by quarantined_date")
    return name


def _column_or_null(frame, column, data_type="string"):
    """Use a source column when available, otherwise create a typed SQL null."""

    if column in frame.columns:
        return F.col(quoted(column)).cast(data_type)
    return F.lit(None).cast(data_type)


def _record_projection(
    frame,
    cfg,
    *,
    dataset,
    run_id,
    run_date,
    validation_stage,
    quarantined_at,
    error_rule_ids_json,
    warning_rule_ids_json,
    error_details_json,
):
    """Build one per-dataset quarantine record for every rejected source row.

    Each row contains 12 'why it failed' reason columns followed by the full
    Silver output columns for this dataset with their exact types.
    record_id uses the Silver type (bigint for AC/DL, string for ATS) so that
    investigators can join directly against the Silver table.
    """

    raw_payload = _column_or_null(frame, "_raw_payload")
    source_file = _column_or_null(frame, "_source_file")
    json_options = {"ignoreNullFields": "false"}

    # Keep the raw payload and distributed source-row token in the identity.
    # Invalid IDs often normalize to null, and two otherwise identical invalid
    # source rows may arrive in the same Bronze file.  Without this token,
    # dropDuplicates(["quarantine_id"]) could lose a failed record before it
    # reaches Iceberg.  The token is not part of the table contract, but it is
    # safely represented by the resulting deterministic audit identifier.
    source_row_id = _column_or_null(frame, "_source_row_id")
    quarantine_identity = F.to_json(
        F.struct(
            F.lit(dataset).alias("dataset_name"),
            F.lit(run_id).alias("run_id"),
            F.lit(validation_stage).alias("validation_stage"),
            _column_or_null(frame, "record_id").cast("string").alias("record_id"),
            source_file.alias("source_file"),
            source_row_id.alias("source_row_id"),
            raw_payload.alias("raw_payload"),
            error_rule_ids_json.alias("error_rule_ids_json"),
        ),
        json_options,
    )
    quarantined_timestamp = F.lit(quarantined_at).cast("timestamp")

    # Full Silver output columns for this dataset with exact types.
    dataset_schema = DATASET_QUARANTINE_SCHEMA[dataset]
    silver_columns = {k: v for k, v in dataset_schema.items() if k not in _QUARANTINE_REASON}
    silver_evidence = [
        _column_or_null(frame, col_name, data_type).alias(col_name)
        for col_name, data_type in silver_columns.items()
    ]

    return frame.select(
        F.sha2(quarantine_identity, 256).alias("quarantine_id"),
        F.lit(run_id).alias("run_id"),
        F.lit(run_date).cast("date").alias("run_date"),
        F.lit(validation_stage).alias("validation_stage"),
        error_rule_ids_json.alias("error_rule_ids_json"),
        warning_rule_ids_json.alias("warning_rule_ids_json"),
        error_details_json.alias("error_details_json"),
        raw_payload.alias("raw_payload"),
        source_file.alias("source_file"),
        quarantined_timestamp.alias("quarantined_at"),
        F.to_date(quarantined_timestamp).alias("quarantined_date"),
        F.lit("PENDING_REMEDIATION").alias("recovery_status"),
        *silver_evidence,
    ).select(*dataset_schema)


def dq_failures(
    frame,
    cfg,
    *,
    dataset,
    run_id,
    run_date,
    validation_stage,
    quarantined_at,
):
    """Return one quarantine row per record with at least one error rule.

    DQ issue artifacts deliberately have one row per rule.  Quarantine must be
    record-oriented for recovery, so all errors and warnings are retained as
    JSON arrays on one row instead of exploding one rejected record many times.
    """

    if "_issues" not in frame.columns or "_has_errors" not in frame.columns:
        raise ValueError("DQ quarantine requires annotated _issues and _has_errors columns")
    errors = F.filter(F.col("_issues"), lambda issue: issue["severity"] == "error")
    warnings = F.filter(
        F.col("_issues"), lambda issue: issue["severity"] == "warning"
    )
    return _record_projection(
        frame.filter(F.coalesce(F.col("_has_errors"), F.lit(False))),
        cfg,
        dataset=dataset,
        run_id=run_id,
        run_date=run_date,
        validation_stage=validation_stage,
        quarantined_at=quarantined_at,
        error_rule_ids_json=F.to_json(
            F.transform(errors, lambda issue: issue["rule_id"])
        ),
        warning_rule_ids_json=F.to_json(
            F.transform(warnings, lambda issue: issue["rule_id"])
        ),
        error_details_json=F.to_json(errors),
    )


def target_conflicts(
    frame,
    cfg,
    *,
    dataset,
    run_id,
    run_date,
    quarantined_at,
):
    """Create entries for incoming IDs with a different existing Silver row."""

    return _record_projection(
        frame,
        cfg,
        dataset=dataset,
        run_id=run_id,
        run_date=run_date,
        validation_stage="TARGET_CONFLICT",
        quarantined_at=quarantined_at,
        error_rule_ids_json=F.lit('["existing_record_id_conflict"]'),
        warning_rule_ids_json=F.lit("[]"),
        error_details_json=F.lit(
            '[{"rule_id":"existing_record_id_conflict","severity":"error"}]'
        ),
    )


def _align_to_target(rows, target):
    """Cast rows to the existing Iceberg schema without changing its field names."""

    incoming_by_lower = {column.lower(): column for column in rows.columns}
    expressions = []
    for field in target.schema.fields:
        source_name = incoming_by_lower.get(field.name.lower())
        if source_name is None:
            raise ValueError(f"Quarantine row is missing target field: {field.name}")
        expressions.append(
            F.col(quoted(source_name)).cast(field.dataType).alias(field.name)
        )
    return rows.select(*expressions)


def write_quarantine(spark, table, location, rows, dataset, create_if_missing=False):
    """Append unseen quarantine entries without overwriting recovery fields.

    The current EMR publish path avoids SQL MERGE because Spark 3.5/EMR can
    fail to resolve Iceberg target attributes in a MERGE.  Use the same proven
    append pattern here: remove existing deterministic IDs first, then append
    new rows.  Each dataset has its own quarantine table, but two writers for
    the *same* dataset table must still be serialized to prevent a race between
    the anti-join and append.
    """

    schema = DATASET_QUARANTINE_SCHEMA[dataset]
    if set(rows.columns) != set(schema):
        raise ValueError(
            f"Quarantine rows for {dataset!r} do not match the dataset contract"
        )

    ensure_quarantine_table(spark, table, location, dataset, create_if_missing)
    source = rows.select(*schema).dropDuplicates(["quarantine_id"])
    attempted_rows = source.count()
    if not attempted_rows:
        return {"attempted_rows": 0, "inserted_rows": 0, "snapshot_id": None}

    target = spark.table(table)
    existing_ids = target.select(
        F.col(quoted("quarantine_id")).alias("_existing_quarantine_id")
    )
    new_rows = source.alias("incoming").join(
        existing_ids.alias("existing"),
        F.col("incoming." + quoted("quarantine_id"))
        == F.col("existing." + quoted("_existing_quarantine_id")),
        "left_anti",
    )
    inserted_rows = new_rows.count()
    if inserted_rows:
        _align_to_target(new_rows, target).writeTo(table).option(
            "check-nullability", "false"
        ).append()

    # Do not claim a snapshot from another writer when this call found that all
    # deterministic IDs already existed. A snapshot ID in the run report must
    # identify this write, not merely the table's latest historical commit.
    snapshot = None
    if inserted_rows:
        name = identifier(table)
        snapshot = spark.sql(
            f"SELECT snapshot_id FROM {name}.snapshots "
            "ORDER BY committed_at DESC LIMIT 1"
        ).first()

    return {
        "attempted_rows": attempted_rows,
        "inserted_rows": inserted_rows,
        "snapshot_id": snapshot["snapshot_id"] if snapshot else None,
    }
