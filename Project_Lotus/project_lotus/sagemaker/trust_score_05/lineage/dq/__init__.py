"""Data quality for the Gold layer: the framework, and the rules per table.

The framework is the same one Silver uses - the same three tiers, the same
nineteen-column metrics table - so both layers trend in one query and an
operator reads one kind of report. What differs is what the checks are *about*,
and the registry below is where that difference is declared.

:func:`checks_for` is the single entry point a job calls. It takes the Gold
table's short name and returns the checks for it, which keeps the job driver
free of any knowledge about which table needs which rule. Two of the six
builders take a reference frame for their referential checks; :func:`checks_for`
passes whatever it is given through, and a builder that receives nothing simply
omits those checks rather than failing. That matters for a partial run - the
IMEI-map job has no lifecycle output of its own to check against when
``close_on_lifecycle_end`` is off - and it means a missing reference degrades
the report rather than the run.
"""

from __future__ import annotations

from typing import Sequence

from pyspark.sql import DataFrame

from .framework import (
    ABORT,
    DIMENSIONS,
    INFO,
    SEVERITIES,
    WARN,
    Check,
    CheckResult,
    DataQualityAbort,
    abort_failures,
    apply_severity_overrides,
    counter,
    distinct_count_violation,
    in_domain,
    in_domain_when_present,
    log_results,
    matches_when_present,
    not_null,
    raise_on_abort,
    references,
    results_to_dataframe,
    run_checks,
    run_checks_by_group,
    summarize,
    unique,
    unique_by,
    write_metrics,
)
from .rules_enrichment import (
    MNO_ACTIVATION_INPUT,
    TU_PORTPS_INPUT,
    first_seen_check,
    hash_agreement_checks,
    hour_drift,
    mno_activation_checks,
    modal_hour,
    modal_hour_checks,
    never_seen_check,
    tu_portps_checks,
)
from .rules_gold import (
    account_mapping_checks,
    canonical_events_checks,
    customer_mapping_checks,
    imei_map_checks,
    lifecycle_checks,
    merge_stat_checks,
    normalized_events_checks,
)

__all__ = [
    # framework
    "ABORT",
    "WARN",
    "INFO",
    "SEVERITIES",
    "DIMENSIONS",
    "Check",
    "CheckResult",
    "DataQualityAbort",
    "run_checks",
    "run_checks_by_group",
    "apply_severity_overrides",
    "summarize",
    "log_results",
    "abort_failures",
    "raise_on_abort",
    "results_to_dataframe",
    "write_metrics",
    "not_null",
    "in_domain",
    "in_domain_when_present",
    "matches_when_present",
    "distinct_count_violation",
    "unique",
    "unique_by",
    "references",
    "counter",
    # rules
    "canonical_events_checks",
    "lifecycle_checks",
    "account_mapping_checks",
    "customer_mapping_checks",
    "normalized_events_checks",
    "imei_map_checks",
    "merge_stat_checks",
    # enrichment inputs
    "TU_PORTPS_INPUT",
    "MNO_ACTIVATION_INPUT",
    "tu_portps_checks",
    "mno_activation_checks",
    "hash_agreement_checks",
    "modal_hour_checks",
    "first_seen_check",
    "never_seen_check",
    "modal_hour",
    "hour_drift",
    # registry
    "CHECK_BUILDERS",
    "REFERENCE_TABLE",
    "checks_for",
]


#: Gold table short name -> the function that builds its checks. Keyed by the
#: same names as :data:`..schemas.GOLD_TABLES`, so a job that iterates its output
#: tables gets the right rules without a second mapping to keep in step.
CHECK_BUILDERS = {
    "account_changes_canonical_events": canonical_events_checks,
    "msisdn_lifecycle": lifecycle_checks,
    "msisdn_lifecycle_account_mapping": account_mapping_checks,
    "msisdn_lifecycle_account_customer_mapping": customer_mapping_checks,
    "normalized_canonical_events": normalized_events_checks,
    "msisdn_imei_map": imei_map_checks,
}

#: Which table's output each builder wants as its referential reference. Three
#: of the six point one level up their own lineage, which is exactly the shape
#: of the hierarchy: an account mapping row names a lifecycle, a customer
#: mapping row names an account, a normalized event names an account.
REFERENCE_TABLE = {
    "msisdn_lifecycle_account_mapping": "msisdn_lifecycle",
    "msisdn_lifecycle_account_customer_mapping": "msisdn_lifecycle_account_mapping",
    "normalized_canonical_events": "msisdn_lifecycle_account_mapping",
}


def checks_for(table: str, reference: DataFrame | None = None) -> list[Check]:
    """The checks for one Gold table.

    ``reference`` is the frame named by :data:`REFERENCE_TABLE` for this table,
    when the caller has it. Passing ``None`` is legitimate and drops only the
    referential checks; passing a frame for a table that has no reference is
    ignored rather than being an error, so a caller can hand the same argument
    to every table in a loop.
    """
    try:
        builder = CHECK_BUILDERS[table]
    except KeyError:
        known = ", ".join(sorted(CHECK_BUILDERS))
        raise KeyError(
            f"no DQ rules registered for gold table {table!r}; known: {known}"
        ) from None

    if table in REFERENCE_TABLE:
        return list(builder(reference))
    return list(builder())


def checks_for_all(
    frames: dict[str, DataFrame]
) -> dict[str, Sequence[Check]]:
    """Build the checks for every table a run produced, wiring the references.

    ``frames`` maps Gold table short name to the frame this run built for it.
    The reference for each table is looked up in the same dict, so a run that
    produced only part of the lineage gets referential checks exactly where it
    has both sides and no checks - rather than an error - where it does not.
    """
    return {
        table: checks_for(table, frames.get(REFERENCE_TABLE.get(table, ""), None))
        for table in frames
        if table in CHECK_BUILDERS
    }
