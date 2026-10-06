"""Validate and write an existing Iceberg Silver table.

This implementation never creates a table. Catalog/database/table provisioning
is an external prerequisite owned by the deployment team.
"""
import re
from functools import reduce

from pyspark.sql import functions as F

from .common import business_columns, quoted


def identifier(name):
    """Validate a one-to-three-part Spark identifier and quote every part."""
    parts = name.split(".")
    if not 1 <= len(parts) <= 3 or any(
        not re.fullmatch("[A-Za-z_][A-Za-z0-9_]*", part) for part in parts
    ):
        raise ValueError("Invalid catalog/table identifier: " + name)
    return ".".join(quoted(part) for part in parts)


def comparable_schema(columns):
    """Normalize schema details that Spark and Glue represent differently.

    Spark SQL is deliberately configured as case-insensitive for this pipeline,
    while Glue/Iceberg commonly return lowercase field names. Iceberg
    `timestamp` (without zone) is exposed to Spark 3.5 as `timestamp_ntz`.
    Source timestamps are parsed in the Toronto session timezone before Spark
    writes that local wall-clock value into the Iceberg timestamp column.
    """
    normalized = {}
    for column, data_type in columns.items():
        key = column.lower()
        if key in normalized:
            raise ValueError(
                f"Schema contains case-insensitive duplicate column: {column}"
            )
        normalized[key] = (
            "timestamp" if data_type in {"timestamp", "timestamp_ntz"} else data_type
        )
    return normalized


def existing_conflicts(incoming, existing, cfg):
    """Return same-ID rows whose business payload differs from existing Silver."""
    columns = business_columns(cfg)
    pairs = incoming.alias("s").join(
        existing.alias("t"),
        F.col("s.record_id").eqNullSafe(F.col("t.record_id")),
        "inner",
    )
    equal = reduce(
        lambda left, right: left & right,
        [
            F.col("s." + quoted(column)).eqNullSafe(
                F.col("t." + quoted(column))
            )
            for column in columns
        ],
    )
    return pairs.filter(~equal).select("s.*").dropDuplicates()


def target_preflight(spark, table, incoming, cfg):
    """Require the existing target to be compatible and return replay conflicts."""
    name = identifier(table)
    if not spark.catalog.tableExists(table):
        raise ValueError(f"Required existing Silver table was not found: {table}")

    properties = {
        row["key"]: row["value"]
        for row in spark.sql(f"SHOW TBLPROPERTIES {name}").collect()
    }
    if "format-version" not in properties:
        raise ValueError(f"Target is not recognized as an Iceberg table: {table}")

    target = spark.table(name)
    actual_raw = {
        field.name: field.dataType.simpleString() for field in target.schema
    }
    actual = comparable_schema(actual_raw)
    expected = comparable_schema(cfg["output"])
    if actual != expected:
        raise ValueError(
            "Existing target schema differs from the reviewed YAML contract: "
            f"actual={actual_raw}; expected={cfg['output']}"
        )

    relevant = target.join(
        incoming.select("record_id").distinct(), "record_id", "left_semi"
    )
    if relevant.groupBy("record_id").count().filter("count > 1").limit(1).count():
        raise ValueError(
            "Existing target contains duplicate IDs for this incoming batch"
        )
    return existing_conflicts(incoming, relevant, cfg)


def aligned_target_rows(candidate, target):
    """Return candidate rows aligned to the target's real names and types.

    Glue commonly stores field names in lowercase even when the reviewed YAML
    uses client casing such as ``subId``. Cast and rename every value to the
    actual Iceberg field. This also handles Iceberg ``timestamp_ntz``
    explicitly instead of relying on an implicit assignment cast.
    """
    incoming_by_lower = {}
    for column in candidate.columns:
        key = column.lower()
        if key in incoming_by_lower:
            raise ValueError(
                f"Candidate contains case-insensitive duplicate column: {column}"
            )
        incoming_by_lower[key] = column

    expressions = []
    for field in target.schema.fields:
        incoming_name = incoming_by_lower.get(field.name.lower())
        if incoming_name is None:
            raise ValueError(
                f"Candidate is missing target column required for publish: {field.name}"
            )
        expressions.append(
            F.col(quoted(incoming_name)).cast(field.dataType).alias(field.name)
        )

    return candidate.select(*expressions)


def publish(spark, table, candidate, cfg):
    """Insert new record IDs into the existing table and return snapshot ID."""
    name = identifier(table)
    target = spark.table(name)
    target_record_id = next(
        (
            field.name
            for field in target.schema.fields
            if field.name.lower() == "record_id"
        ),
        None,
    )
    if target_record_id is None:
        raise ValueError("Existing target has no record_id column")

    # target_preflight has already identified conflicting IDs, and the runner
    # quarantined/removed them. Remove IDs already present so exact replays
    # remain no-ops.
    # The explicit aliases keep this ordinary DataFrame join unambiguous.
    existing_ids = target.select(
        F.col(quoted(target_record_id)).alias("_existing_record_id")
    )
    new_rows = candidate.alias("incoming").join(
        existing_ids.alias("existing"),
        F.col("incoming." + quoted("record_id"))
        == F.col("existing." + quoted("_existing_record_id")),
        "left_anti",
    )
    aligned = aligned_target_rows(new_rows, target)

    # Spark conservatively keeps nullable=True on columns that came through a
    # cast, even after the DQ stage proved they contain no nulls. Iceberg checks
    # that schema flag by default and rejects an optional Spark field destined
    # for a required Iceberg field. Recheck the actual values immediately before
    # the write, then disable only the metadata-level nullability comparison.
    required_fields = [
        field.name for field in target.schema.fields if not field.nullable
    ]
    if required_fields:
        has_required_null = reduce(
            lambda left, right: left | right,
            [F.col(quoted(column)).isNull() for column in required_fields],
        )
        if aligned.filter(has_required_null).limit(1).count():
            raise ValueError(
                "Candidate contains null in required Iceberg field(s): "
                + ", ".join(required_fields)
            )

    # Spark 3.5 on EMR can fail to resolve target attributes in SQL MERGE even
    # when its error message suggests the exact same field. DataFrameWriterV2
    # append avoids that analyzer path. It preserves the intended insert-only
    # behavior because the conflict check and left-anti replay filter ran first.
    if aligned.limit(1).count():
        aligned.writeTo(table).option("check-nullability", "false").append()

    snapshot = spark.sql(
        f"SELECT snapshot_id FROM {name}.snapshots "
        "ORDER BY committed_at DESC LIMIT 1"
    ).first()
    return snapshot["snapshot_id"] if snapshot else None
