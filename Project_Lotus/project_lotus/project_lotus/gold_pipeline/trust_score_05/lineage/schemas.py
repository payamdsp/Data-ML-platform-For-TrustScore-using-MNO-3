"""Gold schema definitions - the single source of truth for this pipeline.

Six output tables in two lineages, plus two operational tables:

``account_changes_canonical_events``            lineage 1, stage 1
``msisdn_lifecycle``                            lineage 1, stage 2
``msisdn_lifecycle_account_mapping``            lineage 1, stage 3
``msisdn_lifecycle_account_customer_mapping``   lineage 1, stage 4
``normalized_canonical_events``                 lineage 1, stage 5
``msisdn_imei_map``                             lineage 2

``gold_run_control``                            watermark / run bookkeeping
``dq_metrics``                                  tiered check results

The Gold tables already exist in production and this pipeline **merges** into
them, so column order, names and types are frozen. Column order matches the
live tables exactly; the markdown under ``data_schema/gold/`` documents the same
shape in prose and the two must not drift.

A note on case: the live column is ``phone_number_AC_hash`` with ``AC``
upper-case. Athena folds identifiers, so CSV extracts render it lower-case.
The names here are the live ones.
"""

from __future__ import annotations

import datetime as _dt

from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

__all__ = [
    # shared
    "MNO_DOMAIN",
    "HEX64_REGEX",
    "SCHEMA_VERSION",
    "SLICE_KEY",
    "schema_to_ddl",
    # domains
    "API_OUTCOME_DOMAIN",
    "TU_EVENT_TYPE_DOMAIN",
    "IDV_SUCCESS_RESPONSE_CODE",
    "TU_SUCCESS_RESPONSE_CODE",
    "AC_EVENT_TYPE_DOMAIN",
    "EXPLICIT_START_EVENTS",
    "FORCE_NEW_EVENTS",
    "END_EVENTS",
    "NEUTRAL_EVENTS",
    "SEGMENT_TYPE_DOMAIN",
    "ACCOUNT_TYPE_DOMAIN",
    "CONFIDENCE_DOMAIN",
    "LINK_ROUTE_DOMAIN",
    "UPDATED_SOURCE_MAPPING_DOMAIN",
    "UPDATED_SOURCE_NORMALIZED_DOMAIN",
    "LIFECYCLE_START_TOKENS",
    "LIFECYCLE_END_TOKENS",
    "LIFECYCLE_START_FLOOR",
    "LIFECYCLE_START_FLOOR_LITERAL",
    "LIFECYCLE_FLOOR_START_TOKEN",
    "IMEI_EVENT_TYPES",
    "EVENT_FAMILY_BY_TYPE",
    # table schemas
    "SILVER_ACCOUNT_CHANGES_SCHEMA",
    "SILVER_DEVICE_LOOKUP_SCHEMA",
    "SILVER_TU_PORTPS_SCHEMA",
    "SILVER_MNO_ACTIVATION_SCHEMA",
    "SILVER_IMEI_SCHEMA",
    "SILVER_SCHEMAS",
    "silver_schema_for",
    "IDV_DATASETS",
    "IDV_PRIMARY_KEY",
    "CANONICAL_EVENTS_SCHEMA",
    "LIFECYCLE_SCHEMA",
    "ACCOUNT_MAPPING_SCHEMA",
    "CUSTOMER_MAPPING_SCHEMA",
    "NORMALIZED_EVENTS_SCHEMA",
    "IMEI_MAP_SCHEMA",
    "RUN_CONTROL_SCHEMA",
    "DQ_METRICS_SCHEMA",
    # table registry
    "GOLD_TABLES",
    "GoldTableSpec",
    "gold_table",
    "gold_schema_for",
    "gold_ddl_for",
]

# ==========================================================================
# shared constants
# ==========================================================================

#: The carriers we hold a feed for. Membership in this set is what
#: ``mno_is_supported`` records, and it is *not* the domain of the ``mno`` column
#: any more: a TU-derived lifecycle sits at a carrier we have no feed from, and
#: its ``mno`` holds that carrier's resolved name. See ``mno_is_supported`` on
#: ``LIFECYCLE_SCHEMA`` and enrichment_design.md section 4.
MNO_DOMAIN = ("BELL", "ROGERS", "TELUS")

#: sha256 hex, case-insensitive.
HEX64_REGEX = "(?i)^[0-9a-f]{64}$"

#: Value written to ``schema_version`` on every row this pipeline produces.
#: It is also a seed component of every surrogate key, so bumping it
#: deliberately re-keys the whole table - never bump it casually.
SCHEMA_VERSION = 1

#: The one column that scopes a batch. Every Gold table carries it, which is
#: what lets a single predicate serve every read, merge and delete branch.
SLICE_KEY = "phone_number_AC_hash"


def schema_to_ddl(schema: StructType) -> str:
    """Render a StructType as the column list for a ``CREATE TABLE`` statement."""
    return ",\n    ".join(
        f"{f.name} {f.dataType.simpleString().upper()}" for f in schema.fields
    )


# ==========================================================================
# canonical event vocabulary
# ==========================================================================

#: The eleven canonical event types. Five of them are derived rather than sent
#: by a carrier; see incremental_design.md section 6.1.
AC_EVENT_TYPE_DOMAIN = (
    "account_activation",
    "account_reactivation",
    "account_suspension",
    "account_cancellation",
    "phone_number_change_from",
    "phone_number_change_to",
    "device_change",
    "sim_change",
    "account_activation_mno_port",
    "account_activation_number_recycle",
    "plan_change",
)

#: Events that open a lifecycle when none is open.
EXPLICIT_START_EVENTS = (
    "account_activation",
    "account_reactivation",
    "account_activation_mno_port",
    "account_activation_number_recycle",
    "phone_number_change_to",
)

#: Events that open a *new* lifecycle even when one is already open, closing the
#: open one at the same timestamp. A port in, a recycle and an incoming number
#: change are all statements that the previous stretch of life ended, whether or
#: not the carrier bothered to say so.
FORCE_NEW_EVENTS = (
    "account_activation_mno_port",
    "account_activation_number_recycle",
    "phone_number_change_to",
)

#: Events that close the open lifecycle.
END_EVENTS = (
    "account_cancellation",
    "phone_number_change_from",
)

#: Events that neither open nor close - they only prove the number was alive.
#: Seeing one with nothing open opens an inferred lifecycle.
NEUTRAL_EVENTS = (
    "account_suspension",
    "device_change",
    "sim_change",
    "plan_change",
)

#: Position of a lifecycle inside its account (and of an account inside its
#: customer, at the level above).
SEGMENT_TYPE_DOMAIN = (
    "single",
    "origin",
    "origin_unmatched",
    "intermediate",
    "final",
    "final_unmatched",
)

ACCOUNT_TYPE_DOMAIN = ("chain", "single")

#: Every confidence column on every table. Two values, deliberately - there is no
#: ``MEDIUM``. Confidence exists to route a decision (enrich this one, trust that
#: one) and a routing decision is binary, so a third value would only be a place
#: for a judgement nobody has to make to hide. "We do not know" is ``LOW``.
CONFIDENCE_DOMAIN = ("HIGH", "LOW")

#: ``link_route`` on the account mapping: the evidence that joined a lifecycle to
#: its predecessor. ``none`` is the first lifecycle of a chain, which was joined
#: to nothing - not a missing route but the absence of a link, and therefore the
#: one route that is always ``HIGH``.
LINK_ROUTE_DOMAIN = ("correlation_id", "counterparty", "imsi", "none")

#: ``updated_source`` on the two mapping tables. TransUnion and Port PS are out
#: of scope for this version; the column exists so they can be told apart later.
UPDATED_SOURCE_MAPPING_DOMAIN = ("enstream",)

#: ``updated_source`` on ``normalized_canonical_events``.
UPDATED_SOURCE_NORMALIZED_DOMAIN = (
    "enstream_event",
    "enstream_event_derived",
    "enstream_api",
)

#: ``lifecycle_infer_type`` is a start token plus an end token. The vocabulary
#: is open - DQ validates against it as a warning, not as a hard enum - but
#: these are the values the current logic can produce.
#:
#: The last three start tokens and the last end token are written by the
#: enrichment stages rather than by the lifecycle walk. They are declared here
#: with the rest because a reader parsing the column should not have to know
#: which stage wrote the row, and because a vocabulary split across two modules
#: is a vocabulary that will disagree with itself.
LIFECYCLE_START_TOKENS = (
    "explicit_activation",
    "explicit_mno_port",
    "explicit_phone_number_change",
    # 90 because the test that produces it is ``>= 90.0`` elapsed days - see
    # ``lifecycle.START_TOKENS``. It read 91 until August 2026, which mislabelled
    # every recycle whose gap fell in ``[90, 91)``.
    "number_recycle_after_gap_greater_or_equal_90_days",
    "inferred_start_from_first_observed_activity",
    "inferred_start_from_plan_change_only",
    "observed_start_from_idv_activation_date",
    "explicit_tu_port_in",
    "inferred_start_from_previous_lifecycle_end",
    "inferred_start_from_floor",
)

#: The start given to a lifecycle we know ended and never saw begin, when the
#: phone number has no earlier lifecycle to carry the start back to.
#:
#: A number whose first event in its whole history is a cancellation certainly
#: had a lifecycle - somebody cancelled something - and the pipeline has no
#: evidence at all about when it opened. That used to be written as a lifecycle
#: starting and ending at the cancellation, which said the opposite of what was
#: meant: a half-open ``[T, T)`` is live at no moment, so a lifecycle recording
#: the one thing we do know about the number was also a lifecycle that never
#: existed.
#:
#: A floor is the other way of writing "unknown": the interval is real, it
#: covers every moment the lifecycle could have been live, and the row says so
#: in three places at once - ``is_start_inferred = 1``,
#: ``start_confidence = LOW``, and the ``inferred_start_from_floor`` token.
#: Nothing may read this as a date. Anything measuring how long lifecycles run,
#: or asking whether a number was live on a given day, has to exclude inferred
#: starts, and did before this constant existed.
#:
#: 1970-01-01 rather than something older because it is the floor the estate
#: already uses - ``jobs.base.WATERMARK_FLOOR`` is the same instant for the same
#: kind of reason - and because Spark will not write a timestamp from
#: ``datetime.min`` at all. It is far enough below any Canadian mobile history
#: in these feeds to be unmistakable.
#:
#: **Naive, and it has to stay naive.** The lifecycle walk assigns this object
#: straight onto ``lifecycle_start_ts`` and then sorts and compares it against
#: the timestamps Spark hands the Python walk, which are naive
#: ``datetime.datetime``. An aware constant would raise ``TypeError`` on the
#: first comparison in the executor rather than being safer.
#:
#: The cost is that ``F.lit(LIFECYCLE_START_FLOOR)`` is *not* the same thing as
#: this constant. PySpark converts a naive datetime literal in the driver's local
#: zone, while ``spark.py:132`` pins the session zone to ``EST``, so on any host
#: not already running at -05:00 the two differ by the offset and a Spark-side
#: ``== LIFECYCLE_START_FLOOR`` predicate silently matches nothing. Nothing does
#: that today - every consumer keys off :data:`LIFECYCLE_FLOOR_START_TOKEN`
#: instead, which is the right design and the reason this has never bitten - and
#: :data:`LIFECYCLE_START_FLOOR_LITERAL` is here so that the one that eventually
#: wants a value has a correct one to reach for.
LIFECYCLE_START_FLOOR = _dt.datetime(1970, 1, 1)

#: :data:`LIFECYCLE_START_FLOOR` as a string, for Spark rather than for Python.
#:
#: ``F.lit(LIFECYCLE_START_FLOOR_LITERAL).cast("timestamp")`` resolves in the
#: **session** zone, which is the zone every other timestamp in these tables is
#: read and written in, so it agrees with the floor the walk wrote whatever
#: timezone the driver happens to be configured with. Any Spark expression that
#: needs the floor as a value has to use this and not the Python object.
LIFECYCLE_START_FLOOR_LITERAL = LIFECYCLE_START_FLOOR.strftime("%Y-%m-%d %H:%M:%S")

#: The token :data:`LIFECYCLE_START_FLOOR` travels with, named so that the two
#: stages which treat a floored start differently from every other inferred one
#: - ``transforms.boundaries`` and ``transforms.tu_lifecycle`` - can recognise it
#: without either of them owning the string or importing the other.
LIFECYCLE_FLOOR_START_TOKEN = "inferred_start_from_floor"

LIFECYCLE_END_TOKENS = (
    "_to_explicit_cancellation",
    "_to_explicit_phone_number_change",
    "_to_inferred_end_from_next_start",
    "_open_no_cancellation",
    "_to_explicit_tu_port_out",
)

#: Silver ``device_lookup_batch`` event types that drive the IMEI map. The
#: ``IMSI_*`` types are evidence for other things (ROGERS activations) and do
#: not move a device segment.
IMEI_EVENT_TYPES = (
    "IMEI_START",
    "IMEI_START_A",
    "IMEI_START_NC",
    "IMEI_END",
)

#: The outcome of one enrichment API call, derived from its ``response_code``.
#:
#: Three values and not two. ``ERROR`` is retryable - the call failed and we
#: learned nothing, so asking again may work. ``NOT_FOUND`` is an *answer*: the
#: provider has no record of this number and asking again produces the same
#: nothing. Collapsing them means either retrying forever or accepting a gap as a
#: fact, and both are worse than one more enum value.
API_OUTCOME_DOMAIN = ("OK", "NOT_FOUND", "ERROR")

#: ``mno_activation`` / ``imei`` success code. Held as a
#: string because it is a code and not a quantity - nothing sums it, and a
#: leading zero has to survive the round trip.
IDV_SUCCESS_RESPONSE_CODE = "0"

#: ``tu_portps`` success code. A different vocabulary from IDV's entirely, which
#: is why the two are separate constants rather than one shared "success".
TU_SUCCESS_RESPONSE_CODE = "3000"

#: ``tu_portps.event_type`` after normalization. Silver keeps only these two;
#: NPAC assignment / return / SPID update events are dropped upstream.
TU_EVENT_TYPE_DOMAIN = ("INTER_SPID_PORT", "INTRA_SPID_PORT")

#: Coarse grouping used by ``normalized_canonical_events.event_family``.
EVENT_FAMILY_BY_TYPE = {
    "account_activation": "account_status",
    "account_reactivation": "account_status",
    "account_suspension": "account_status",
    "account_cancellation": "account_status",
    "account_activation_mno_port": "port",
    "account_activation_number_recycle": "number_recycle",
    "plan_change": "account_status",
    "phone_number_change_from": "phone_number_change",
    "phone_number_change_to": "phone_number_change",
    "device_change": "device",
    "sim_change": "sim",
}


# ==========================================================================
# silver inputs
# ==========================================================================
# Declared here as an explicit contract rather than imported from the
# Bronze -> Silver package: this pipeline reads those tables, it does not own
# them, and a column disappearing upstream should fail here with a clear
# message instead of somewhere in the middle of a transform.

#: ``account_changes_batch`` silver - 14 columns.
SILVER_ACCOUNT_CHANGES_SCHEMA = StructType(
    [
        StructField("record_id", StringType(), False),
        StructField("phone_number_AC_hash", StringType(), False),
        StructField("event_type", StringType(), False),
        StructField("event_timestamp", TimestampType(), False),
        StructField("mno", StringType(), False),
        StructField("subId", StringType(), True),
        StructField("msisdnChangeSide", StringType(), True),
        StructField("correlationId", StringType(), False),
        StructField("otherMSISDN", StringType(), True),
        StructField("ingestion_ts", TimestampType(), False),
        StructField("source_name", StringType(), False),
        StructField("source_event_id", LongType(), False),
        StructField("schema_version", IntegerType(), False),
        StructField("event_date", DateType(), False),
    ]
)

#: ``device_lookup_batch`` silver - 20 columns.
SILVER_DEVICE_LOOKUP_SCHEMA = StructType(
    [
        StructField("record_id", StringType(), False),
        StructField("phone_number_AC_hash", StringType(), False),
        StructField("phone_number_AT_hash", StringType(), True),
        StructField("mno", StringType(), False),
        StructField("imei", StringType(), True),
        StructField("imsi", StringType(), True),
        StructField("date", DateType(), False),
        StructField("event_timestamp", TimestampType(), False),
        StructField("correlation_id", StringType(), False),
        StructField("related_id", StringType(), True),
        StructField("event_type", StringType(), False),
        StructField("tac", StringType(), True),
        StructField("oldIMSI", StringType(), True),
        StructField("oldIMEI", StringType(), True),
        StructField("fromADCSnapshot", StringType(), True),
        StructField("ingestion_ts", TimestampType(), False),
        StructField("source_name", StringType(), False),
        StructField("source_event_id", LongType(), False),
        StructField("schema_version", IntegerType(), False),
        StructField("event_date", DateType(), False),
    ]
)

# --------------------------------------------------------------------------
# enrichment inputs
# --------------------------------------------------------------------------
# Four tables this pipeline reads and does not own, exactly like the two above.
# They are *inputs* rather than outputs for the reason in enrichment_design.md
# section 1: Gold is rebuilt from scratch on every run, so a confidence value
# written onto a Gold row by anything other than the rebuild is silently reverted
# the next time its phone number appears in a batch. Enrichment changes what the
# rebuild reads.
#
# None of the four carries a ``msisdn`` this pipeline reads. The column is in the
# schema because the feed has it and because a column silently absent from a
# contract is worse than one present and unused - but every join here is on
# ``phone_number_AC_hash`` / ``phone_number_AT_hash`` plus a timestamp where the
# grain needs one, and the plaintext is field-level encrypted at rest.

#: ``tu_portps`` silver - 28 columns. One row per port event.
#: See data_schema/silver/tu_portps.md, which is authoritative over the samples.
SILVER_TU_PORTPS_SCHEMA = StructType(
    [
        StructField("request_id", StringType(), False),
        StructField("record_id", StringType(), False),
        StructField("msisdn", StringType(), False),
        StructField("phone_number_AC_hash", StringType(), False),
        StructField("phone_number_AT_hash", StringType(), False),
        StructField("ownership_status", StringType(), True),
        StructField("ownership_last_alt_spid", StringType(), True),
        StructField("region", StringType(), True),
        StructField("state", StringType(), True),
        StructField("event_timestamp", TimestampType(), False),
        StructField("event_date", DateType(), False),
        StructField("event_type", StringType(), False),
        StructField("event_seq", IntegerType(), True),
        StructField("lnp_type", IntegerType(), True),
        StructField("is_inter_carrier_port", BooleanType(), False),
        StructField("from_spid", StringType(), True),
        StructField("from_mno_raw", StringType(), True),
        StructField("from_mno_std", StringType(), True),
        StructField("to_spid", StringType(), False),
        StructField("to_mno_raw", StringType(), True),
        StructField("to_mno_std", StringType(), True),
        StructField("sv_type", StringType(), True),
        StructField("lrn", StringType(), True),
        StructField("correlation_id", StringType(), True),
        StructField("source_name", StringType(), False),
        StructField("schema_version", IntegerType(), False),
        StructField("ingestion_ts", TimestampType(), False),
        StructField("last_updated_ts", TimestampType(), False),
    ]
)

#: ``mno_activation`` silver - 9 columns. One row per ``activationDate`` call.
#:
#: ``activation_date`` is a ``date`` and not a timestamp because the API returns
#: ``MM/DD/YYYY`` with no time of day. Every comparison against it therefore
#: treats it as midnight, which is stated here because the resolver's
#: ``activation_date > fraud_event_ts`` gate turns on it and a fraud at 09:00 on
#: the activation date itself falls on the *later* side of midnight.
SILVER_MNO_ACTIVATION_SCHEMA = StructType(
    [
        StructField("request_id", StringType(), False),
        StructField("msisdn", StringType(), False),
        StructField("phone_number_AC_hash", StringType(), False),
        StructField("phone_number_AT_hash", StringType(), False),
        StructField("api_ts", TimestampType(), False),
        StructField("mno", StringType(), True),
        StructField("activation_date", DateType(), True),
        StructField("response_code", StringType(), False),
        StructField("created_ts", TimestampType(), False),
    ]
)

#: ``imei`` silver - 11 columns. One row per ``identityData`` call.
#:
#: ``imei_hashed`` and ``imsi_hashed`` arrive already hashed and are stored as
#: returned. They are **not** re-hashed by us and are not comparable to hashes we
#: compute ourselves unless the algorithm is confirmed to match, which it has not
#: been. Nothing in this pipeline joins on them.
SILVER_IMEI_SCHEMA = StructType(
    [
        StructField("request_id", StringType(), False),
        StructField("msisdn", StringType(), False),
        StructField("phone_number_AC_hash", StringType(), False),
        StructField("phone_number_AT_hash", StringType(), False),
        StructField("api_ts", TimestampType(), False),
        StructField("imei_hashed", StringType(), True),
        StructField("imsi_hashed", StringType(), True),
        StructField("sim_swap_ts", TimestampType(), True),
        StructField("response_code", StringType(), False),
        StructField("service_provider_id", StringType(), False),
        StructField("created_ts", TimestampType(), False),
    ]
)

SILVER_SCHEMAS = {
    "account_changes_batch": SILVER_ACCOUNT_CHANGES_SCHEMA,
    "device_lookup_batch": SILVER_DEVICE_LOOKUP_SCHEMA,
    "tu_portps": SILVER_TU_PORTPS_SCHEMA,
    "mno_activation": SILVER_MNO_ACTIVATION_SCHEMA,
    "imei": SILVER_IMEI_SCHEMA,
}

#: The IDV tables share a key, a timestamp column and a response-code vocabulary,
#: so the checks that apply to both are written once against this tuple rather
#: than twice.
#:
#: A third, ``number_recycle``, was specified and removed before it was built: the
#: ``checkRecycled`` endpoint ran the same derivation the resolver already does
#: from the activation date, the port history and the 90-day quarantine window.
#: Note that ``number_recycle_threshold_days`` and the canonical event
#: ``account_activation_number_recycle`` are unrelated to it and are live.
IDV_DATASETS = ("mno_activation", "imei")

#: Primary key of every IDV table.
#:
#: ``request_id`` alone would very likely do - the caller mints it per call - but
#: it comes from outside our control and a replay would silently overwrite an
#: observation. ``(phone_number_AC_hash, api_ts)`` is *not* the key even though it
#: is the natural read path: two endpoints can be called for one number in the
#: same second, and the samples show exactly that.
IDV_PRIMARY_KEY = ("request_id", "api_ts")


def silver_schema_for(dataset: str) -> StructType:
    try:
        return SILVER_SCHEMAS[dataset]
    except KeyError:
        known = ", ".join(sorted(SILVER_SCHEMAS))
        raise KeyError(f"unknown silver dataset {dataset!r}; known: {known}") from None


# ==========================================================================
# lineage 1 - account / customer map
# ==========================================================================

#: ``account_changes_canonical_events`` - 9 columns.
#: Carrier vocabulary translated into ours, plus the five derived types.
CANONICAL_EVENTS_SCHEMA = StructType(
    [
        StructField("ac_event_id", StringType(), False),
        StructField("phone_number_AC_hash", StringType(), False),
        StructField("mno", StringType(), False),
        StructField("ac_event_type", StringType(), False),
        StructField("ac_event_ts", TimestampType(), False),
        StructField("correlation_id", StringType(), True),
        StructField("updated_ts", TimestampType(), False),
        StructField("created_ts", TimestampType(), False),
        StructField("schema_version", IntegerType(), False),
    ]
)

#: ``msisdn_lifecycle`` - 28 columns. One row per lifecycle.
#:
#: The three confidence columns are appended after ``schema_version`` rather than
#: placed next to the ``is_*_inferred`` flags they describe. That reads oddly and
#: it is the right way round: the live table gains them through
#: ``ALTER TABLE ... ADD COLUMN``, which appends, and writing them anywhere else
#: here would mean the frame and the table disagree about column order on the
#: first merge after the alter. ``mno_is_supported`` and ``tu_enriched`` are
#: appended after them for the same reason and in that order, which is the order
#: the alter statements run in.
#:
#: ``mno_is_supported`` is what lets ``mno`` hold a carrier outside
#: ``MNO_DOMAIN``. A TU-derived lifecycle sits at a carrier we have no feed from,
#: and every rule that asserted domain membership relaxes to *non-empty, and in
#: the domain if supported*. Every row that exists today gets ``1``, because
#: every row that exists today came from a supported feed - which is what makes
#: this a merge-compatible addition rather than a change to an existing column.
#:
#: ``tu_enriched`` records that TU was asked about this phone number, whatever it
#: said. Gold is enriched unevenly - TU is fetched per fraud record while these
#: tables are shared by every phone number - so without this flag a reader cannot
#: distinguish "TU had nothing" from "TU was never asked", and those two justify
#: opposite actions.
#:
#: The last four columns are the lifecycle's **boundary events**, appended last
#: for the same alter-order reason as everything above them. They were
#: work-frame-only until 20 August 2026, and keeping them out of the table was a
#: false economy: the account stage matches an edge by boundary event id and
#: type, so every notebook and job that needed accounts had to re-run the whole
#: lifecycle walk to get four columns the walk had already computed. Publishing
#: them turns a graph walk over the full event history into a table read.
#:
#: All four are nullable here and two of them are not nullable in the work frame.
#: That is deliberate. A published column that a later boundary rule cannot
#: always fill would otherwise have to be widened by a second migration, and a
#: nullable column that is never null costs a reader nothing.
LIFECYCLE_SCHEMA = StructType(
    [
        StructField("lifecycle_uid", StringType(), False),
        StructField("phone_number_AC_hash", StringType(), False),
        StructField("mno", StringType(), False),
        StructField("mno_segment_id", IntegerType(), False),
        StructField("lifecycle_id", IntegerType(), False),
        StructField("lifecycle_start_ts", TimestampType(), False),
        StructField("lifecycle_end_ts", TimestampType(), True),
        StructField("lifecycle_is_open", IntegerType(), False),
        StructField("lifecycle_infer_type", StringType(), False),
        StructField("is_start_inferred", IntegerType(), False),
        StructField("is_end_inferred", IntegerType(), False),
        StructField("recycle_ind", IntegerType(), False),
        StructField("mno_port_ind", IntegerType(), False),
        StructField("temporary_pn_ind", IntegerType(), False),
        StructField("plan_change_ind", IntegerType(), False),
        StructField("plan_change_cnt", IntegerType(), False),
        StructField("updated_ts", TimestampType(), False),
        StructField("created_ts", TimestampType(), False),
        StructField("schema_version", IntegerType(), False),
        StructField("start_confidence", StringType(), False),
        StructField("end_confidence", StringType(), False),
        StructField("confidence", StringType(), False),
        StructField("mno_is_supported", IntegerType(), False),
        StructField("tu_enriched", IntegerType(), False),
        StructField("start_event_id", StringType(), True),
        StructField("start_event_type", StringType(), True),
        StructField("end_event_id", StringType(), True),
        StructField("end_event_type", StringType(), True),
    ]
)

#: ``msisdn_lifecycle_account_mapping`` - 16 columns.
#: One row per lifecycle within an account.
#:
#: ``link_confidence`` describes the *incoming* link - the evidence that joined
#: this lifecycle to the one before it in the account. ``acct_link_confidence_min``
#: is the whole chain's worst link, repeated on every row of the chain, so a
#: consumer holding one row knows how sound the account it names is without
#: having to fetch the rest of it.
ACCOUNT_MAPPING_SCHEMA = StructType(
    [
        StructField("acct_id", StringType(), False),
        StructField("phone_number_AC_hash", StringType(), False),
        StructField("mno", StringType(), False),
        StructField("lifecycle_uid", StringType(), False),
        StructField("source_event_id", StringType(), True),
        StructField("from_ts", TimestampType(), False),
        StructField("to_ts", TimestampType(), True),
        StructField("seq_order", IntegerType(), False),
        StructField("segment_type", StringType(), False),
        StructField("account_type", StringType(), False),
        StructField("updated_ts", TimestampType(), False),
        StructField("created_ts", TimestampType(), False),
        StructField("schema_version", IntegerType(), False),
        StructField("link_route", StringType(), False),
        StructField("link_confidence", StringType(), False),
        StructField("acct_link_confidence_min", StringType(), False),
    ]
)

#: ``msisdn_lifecycle_account_customer_mapping`` - 17 columns.
#: The same grain as the account mapping, carried up one level so a customer
#: can be read without a join.
#:
#: There is no ``link_route`` here. A customer link is always a port - that is
#: the whole rule - so a route column would carry one value forever.
CUSTOMER_MAPPING_SCHEMA = StructType(
    [
        StructField("customer_id", StringType(), False),
        StructField("acct_id", StringType(), False),
        StructField("mno", StringType(), False),
        StructField("phone_number_AC_hash", StringType(), False),
        StructField("lifecycle_uid", StringType(), False),
        StructField("seq_order", IntegerType(), False),
        StructField("source_event_id", StringType(), True),
        StructField("from_ts", TimestampType(), False),
        StructField("to_ts", TimestampType(), True),
        StructField("segment_type", StringType(), False),
        StructField("account_type", StringType(), False),
        StructField("updated_ts", TimestampType(), False),
        StructField("updated_source", StringType(), False),
        StructField("created_ts", TimestampType(), False),
        StructField("schema_version", IntegerType(), False),
        StructField("link_confidence", StringType(), False),
        StructField("customer_link_confidence_min", StringType(), False),
    ]
)

#: ``normalized_canonical_events`` - 18 columns. The serving shape: every
#: canonical event flattened onto the account and customer it falls inside.
NORMALIZED_EVENTS_SCHEMA = StructType(
    [
        StructField("customer_id", StringType(), True),
        StructField("acct_id", StringType(), True),
        StructField("phone_number_AC_hash", StringType(), True),
        StructField("phone_number_AT_hash", StringType(), True),
        StructField("mno", StringType(), True),
        StructField("event_id", StringType(), False),
        StructField("event_ts", TimestampType(), False),
        StructField("event_family", StringType(), False),
        StructField("event_type", StringType(), True),
        StructField("event_source", StringType(), False),
        StructField("event_weight", DoubleType(), False),
        StructField("industry", StringType(), True),
        StructField("service_provider_name", StringType(), True),
        StructField("use_case", StringType(), True),
        StructField("updated_ts", TimestampType(), False),
        StructField("updated_source", StringType(), False),
        StructField("created_ts", TimestampType(), False),
        StructField("schema_version", IntegerType(), False),
    ]
)


# ==========================================================================
# lineage 2 - msisdn <-> imei map
# ==========================================================================

#: ``msisdn_imei_map`` - 10 columns. One row per *contiguous* association
#: between a phone number and a device, valid ``[from_ts, to_ts)``.
IMEI_MAP_SCHEMA = StructType(
    [
        StructField("phone_number_AC_hash", StringType(), False),
        StructField("phone_number_AT_hash", StringType(), True),
        StructField("imei", StringType(), False),
        StructField("mno", StringType(), False),
        StructField("tac", StringType(), True),
        StructField("oldIMEI", StringType(), True),
        StructField("from_ts", TimestampType(), False),
        StructField("to_ts", TimestampType(), True),
        StructField("last_updated_ts", TimestampType(), False),
        StructField("event_date", DateType(), False),
    ]
)


# ==========================================================================
# operational tables
# ==========================================================================

#: ``gold_run_control`` - one row per run per job. This is what makes a batch
#: reproducible: the watermark a run consumed is recorded, not recomputed.
#: ``status`` is one of ``running`` / ``succeeded`` / ``failed``.
RUN_CONTROL_SCHEMA = StructType(
    [
        StructField("job_name", StringType(), False),
        StructField("run_id", StringType(), False),
        StructField("watermark_from", TimestampType(), True),
        StructField("watermark_to", TimestampType(), False),
        StructField("slice_size", LongType(), False),
        StructField("status", StringType(), False),
        StructField("started_ts", TimestampType(), False),
        StructField("finished_ts", TimestampType(), True),
    ]
)

#: ``dq_metrics`` - the same shape the Bronze -> Silver pipeline writes, so the
#: two layers can be trended in one query. ``layer`` carries ``gold`` and
#: ``dataset`` carries the Gold table the check ran against, which is how one
#: Gold run reports checks for six different tables into one table.
DQ_METRICS_SCHEMA = StructType(
    [
        StructField("run_id", StringType(), False),
        StructField("dataset", StringType(), False),
        StructField("layer", StringType(), False),
        StructField("metric_name", StringType(), False),
        StructField("dimension", StringType(), False),
        StructField("severity", StringType(), False),
        StructField("checked_field", StringType(), True),
        StructField("group_key", StringType(), True),
        StructField("group_value", StringType(), True),
        StructField("total_rows", LongType(), False),
        StructField("violation_rows", LongType(), False),
        StructField("metric_value", StringType(), False),
        StructField("status", StringType(), False),
        StructField("description", StringType(), True),
        StructField("batch_id", StringType(), True),
        StructField("event_date_min", DateType(), True),
        StructField("event_date_max", DateType(), True),
        StructField("check_ts", TimestampType(), False),
        StructField("check_date", DateType(), False),
    ]
)


# ==========================================================================
# table registry
# ==========================================================================


class GoldTableSpec:
    """Everything the merge sink needs to know about one Gold table.

    ``merge_keys`` is the row identity used by the ``MERGE ... ON`` clause. It
    is *not* always the surrogate key: the account mapping is keyed on
    ``(acct_id, lifecycle_uid)`` because one account holds several lifecycles,
    and the IMEI map is keyed on ``(phone, imei, from_ts)`` because one phone
    number can return to the same device later.

    ``sort_by`` leads with ``phone_number_AC_hash`` on every table. The phone
    hash is not the partition column, so Iceberg's per-file min/max on the sort
    key is the only pruning a slice merge can use. It is a requirement, not a
    tuning preference.
    """

    __slots__ = ("name", "schema", "merge_keys", "partition_by", "sort_by", "payload_exclude")

    def __init__(
        self,
        name: str,
        schema: StructType,
        merge_keys: tuple[str, ...],
        partition_by: tuple[str, ...],
        sort_by: tuple[str, ...],
        payload_exclude: tuple[str, ...] = ("created_ts", "updated_ts", "last_updated_ts"),
    ) -> None:
        self.name = name
        self.schema = schema
        self.merge_keys = merge_keys
        self.partition_by = partition_by
        self.sort_by = sort_by
        #: Columns compared to decide whether a matched row actually changed.
        #: Bookkeeping timestamps are excluded, otherwise every row in the slice
        #: would look dirty on every run and the merge would rewrite the world.
        self.payload_exclude = payload_exclude

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.schema.fields)

    @property
    def payload_columns(self) -> tuple[str, ...]:
        """Columns whose change makes a matched row worth updating."""
        skip = set(self.merge_keys) | set(self.payload_exclude)
        return tuple(c for c in self.columns if c not in skip)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"GoldTableSpec({self.name!r})"


GOLD_TABLES: dict[str, GoldTableSpec] = {
    "account_changes_canonical_events": GoldTableSpec(
        name="account_changes_canonical_events",
        schema=CANONICAL_EVENTS_SCHEMA,
        merge_keys=("ac_event_id",),
        partition_by=("ac_event_ts",),
        sort_by=("phone_number_AC_hash", "ac_event_ts"),
    ),
    "msisdn_lifecycle": GoldTableSpec(
        name="msisdn_lifecycle",
        schema=LIFECYCLE_SCHEMA,
        merge_keys=("lifecycle_uid",),
        partition_by=("lifecycle_start_ts",),
        sort_by=("phone_number_AC_hash", "lifecycle_start_ts"),
    ),
    "msisdn_lifecycle_account_mapping": GoldTableSpec(
        name="msisdn_lifecycle_account_mapping",
        schema=ACCOUNT_MAPPING_SCHEMA,
        merge_keys=("acct_id", "lifecycle_uid"),
        partition_by=("from_ts",),
        sort_by=("phone_number_AC_hash", "from_ts"),
    ),
    "msisdn_lifecycle_account_customer_mapping": GoldTableSpec(
        name="msisdn_lifecycle_account_customer_mapping",
        schema=CUSTOMER_MAPPING_SCHEMA,
        merge_keys=("customer_id", "acct_id", "lifecycle_uid"),
        partition_by=("from_ts",),
        sort_by=("phone_number_AC_hash", "from_ts"),
    ),
    "normalized_canonical_events": GoldTableSpec(
        name="normalized_canonical_events",
        schema=NORMALIZED_EVENTS_SCHEMA,
        merge_keys=("event_id",),
        partition_by=("event_ts",),
        sort_by=("phone_number_AC_hash", "event_ts"),
    ),
    "msisdn_imei_map": GoldTableSpec(
        name="msisdn_imei_map",
        schema=IMEI_MAP_SCHEMA,
        merge_keys=("phone_number_AC_hash", "imei", "from_ts"),
        partition_by=("event_date",),
        sort_by=("phone_number_AC_hash", "from_ts"),
    ),
}


def gold_table(name: str) -> GoldTableSpec:
    """Look up a table spec by short name, with a helpful error."""
    try:
        return GOLD_TABLES[name]
    except KeyError:
        known = ", ".join(sorted(GOLD_TABLES))
        raise KeyError(f"unknown gold table {name!r}; known tables: {known}") from None


def gold_schema_for(name: str) -> StructType:
    return gold_table(name).schema


def gold_ddl_for(name: str) -> str:
    return schema_to_ddl(gold_table(name).schema)
