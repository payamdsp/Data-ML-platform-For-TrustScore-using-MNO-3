"""Audit reference data used by the client POC's Revenue-export approach.

Only audit_trail_services_3 imports these references. Account changes and device
lookup have no partner/provider enrichment and never call this module.
"""
from pyspark.sql import functions as F

from .common import clean, pad_identifier


def unique_reference(df, key, values):
    """Return one normalized mapping row per key or stop on ambiguity."""
    selected = df.select(
        pad_identifier(key).alias(key),
        *[clean(column).alias(column) for column in values],
    )
    selected = selected.filter(F.col(key).isNotNull()).dropDuplicates()

    if selected.groupBy(key).count().filter("count > 1").limit(1).count():
        raise ValueError(
            f"Conflicting reference rows for {key}; resolve the CSV before Silver"
        )
    if selected.filter(F.col(key).rlike("^[0-9]{11,}$")).limit(1).count():
        raise ValueError(f"Reference {key} exceeds ten numeric digits")
    if not selected.limit(1).count():
        raise ValueError(f"Reference CSV contains no usable {key} rows")
    return selected


def prepare_references(partners, providers):
    """Validate both mappings and add markers used after the audit left joins."""
    partner_mapping = unique_reference(
        partners, "partner_id", ["partner_name"]
    ).withColumn("_partner_match", F.lit(True))
    provider_mapping = unique_reference(
        providers,
        "service_provider_id",
        ["service_provider_name", "industry"],
    ).withColumn("_provider_match", F.lit(True))
    return partner_mapping, provider_mapping


def load_references(spark, partner_path, provider_path):
    """Read the two Revenue-export CSVs required only by the audit table.

    Expected partner columns:
      partner_id, partner_name_tableau

    Expected provider columns:
      service_provider_id, service_provider_name, service_provider_industry

    `multiLine=true` lets Spark keep a quoted field containing a line break in
    one CSV record. It is also safe for ordinary one-line CSV rows. FAILFAST
    stops instead of silently accepting structurally malformed CSV input.
    """
    def read_csv(path):
        """Read one reference CSV with the header/multiline/failfast options above."""
        return (
            spark.read.option("header", True)
            .option("multiLine", True)
            .option("mode", "FAILFAST")
            .csv(path)
        )

    partner_source = read_csv(partner_path)
    provider_source = read_csv(provider_path)

    required_partner = {"partner_id", "partner_name_tableau"}
    required_provider = {
        "service_provider_id",
        "service_provider_name",
        "service_provider_industry",
    }
    missing_partner = required_partner - set(partner_source.columns)
    missing_provider = required_provider - set(provider_source.columns)
    if missing_partner:
        raise ValueError(f"partners.csv is missing columns: {sorted(missing_partner)}")
    if missing_provider:
        raise ValueError(
            f"service_providers.csv is missing columns: {sorted(missing_provider)}"
        )

    partners = partner_source.select(
        "partner_id", F.col("partner_name_tableau").alias("partner_name")
    )
    providers = provider_source.select(
        "service_provider_id",
        "service_provider_name",
        F.col("service_provider_industry").alias("industry"),
    )
    return prepare_references(partners, providers)
