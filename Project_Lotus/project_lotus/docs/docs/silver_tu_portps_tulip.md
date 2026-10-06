# Silver data model design: tu_portps_tulip

## 1. Background and data shape

One row per port event (Inter-/Intra-SPID), read from bronze `tu_portps_tulip`. Non-port events (NPAC assignment / return) stay in bronze but are dropped from silver (used to compute `from_spid` first, see §7).

---

## 2. Schema definition

### 2.1 Naming / type / normalization principles

- All identifiers are `string`: SPIDs contain letters (`328F`/`709G`), LRN is a routing number not a numeric, casting to long would break it.
- The three null states are normalized to a true `null`: `"NULL"` / empty string / the lake's `\N` all become null; SPIDs split out of arrays are always trimmed.
- Timestamps: the source is UTC, shifted to EST.
- Unknown `event_type` / `sv_type` values sent to DQ to review.

### 2.2 Column list with types, nullability, descriptions, mapped from TU's API field names

Column legend: **Column** = silver field name; **Type** = data type; **Nullable** = whether null is allowed; **Description** = what the field is for; **Example** = a value transformed from source to silver (full walkthrough in §8); **New?** = `N` comes directly from a TU source column, `Y` is derived / extracted / injected.

| Column | Type | Nullable | Description | Example (source → silver) | New? |
|---|---|:--:|---|---|---|
| `request_id` | string | No | API-call id; first column, non-null. Repeats across a call's event rows — provenance, not a unique row key. | `9f3c2a...-e41b` | Y (bronze `request_id`; POC mints a deterministic UUID as a stand-in) |
| `record_id` | string | No | Content hash of the natural key; unique per event. **Primary key.** | `9f3c2a...e41b` | Y (sha256 of natural key, 64 hex) |
| `phone_number_AC_hash` | string | No | Hashed number; join key into `account_changes_batch`. | `c4d9...8a17` | Y (upstream `acHash`) |
| `phone_number_AT_hash` | string | No | Hashed number; join key into `audit_trail_services_3`. | `7b2e...0f9c` | Y (upstream `atHash`) |
| `event_timestamp` | timestamp | No | When the port happened (EST). | `... UTC -> ... EST` | N (XML `<Date>`) |
| `event_date` | date | No | Date of the event (partition column). | `2024-06-04` | Y (`to_date(event_timestamp)`) |
| `event_type` | string | No | Port type; silver keeps only `INTER_SPID_PORT` / `INTRA_SPID_PORT`. | `INTER_SPID_PORT` | N (XML `<Event>` normalized) |
| `event_seq` | int | Yes | Order of the event within the number (1 = earliest). | `10` | Y (window sequence) |
| `lnp_type` | int | Yes | Port type code: 0 = inter-carrier (`from_spid` ≠ `to_spid`), 1 = intra-carrier (same SPID). | `0` | Y (derived from `event_type`) |
| `is_inter_carrier_port` | boolean | No | True when `from_spid` ≠ `to_spid` (inter-SPID port). | `true` | Y (derived from `event_type`) |
| `from_spid` | string | Yes | Carrier the number ported out from. | `6574` | Y (`lag` over `to_spid`) |
| `from_mno_raw` | string | Yes | Brand name of the port-out carrier. | `BELL` | Y (`from_spid` dict lookup) |
| `from_mno_std` | string | Yes | Port-out carrier normalized to big-3 (Bell/Rogers/Telus), else null. | `BELL` | Y (parent dict) |
| `to_spid` | string | No | Carrier the number ported in to (this event). | `8303` | N (XML `<Spid>`) |
| `to_mno_raw` | string | Yes | Brand name of the port-in carrier. | `TELUS` | Y (`to_spid` dict lookup) |
| `to_mno_std` | string | Yes | Port-in carrier normalized to big-3, else null. | `TELUS` | Y (parent dict) |
| `from_to_same_carrier` | int | Yes | Whether port-out and port-in are the same brand (1 / 0 / null if unknown). | `0` | Y (`from_mno_raw == to_mno_raw`) |
| `sv_type` | string | Yes | Service type of the event (Wireless/Wireline/VoIP/…). | `Wireless` | N (XML `<SvType>`) |
| `lrn` | string | Yes | Routing number of the switch the number landed on. | `4186180002` | N (XML `<LRN>` strip dashes) |
| `correlation_id` | string | Yes | Correlation id to match TU against the MNO feed; null (TU carries none), kept for schema alignment. | `(null)` | Y (placeholder, all null) |
| `source_name` | string | No | Data source id. | `tu_portps` | Y (constant) |
| `schema_version` | int | No | Schema version. | `1` | Y (constant) |
| `ingestion_ts` | timestamp | No | When the platform received the API response. | `2026-06-10 14:00` | Y (bronze `ingestion_ts`; POC uses run time) |
| `last_updated_ts` | timestamp | No | Equal to `ingestion_ts`. Tulip only. | `2026-06-10 14:00` | Y (= `ingestion_ts`) |

### 2.3 Primary key / uniqueness constraint

- Primary key: `record_id` — sha256 of the natural key `(phone_number_AC_hash, event_timestamp, to_spid, event_type)`, unique per event and non-null.
- `request_id` — API-call id, first column, non-null. Repeats across a call's events → not unique, provenance only.

### 2.4 Partitioning strategy, date-partitioned at minimum

- Table: `spark_catalog.trust_score_v1_poc_silver.tu_portps_tulip`, Iceberg V3 (Glue).
- Partition: `event_date`.
- Write order: `(phone_number_AC_hash, event_timestamp)` for per-number lookups.
- Path: `s3://enstream-lake-silver-dev/poc/tables/dataset=tu_portps_tulip/version=1/`.

---

## 3. Cross-carrier reconciliation

### 3.1 How a port from carrier A to carrier B is represented: single row, two rows, or correlation ID? Decision with rationale

Single row per port-in, grain `(phone_number_AC_hash, event_timestamp)`: `to_spid` = this event, `from_spid` = previous event (`lag`).

- No two rows: TU history is already linear.
- `correlation_id` is a null placeholder (kept for schema alignment); the TU-vs-MNO matching is gold's job.

### 3.2 How to handle ports involving unsupported carriers

TU only sees the port itself, not what happens on the unsupported side afterward.

- Port to/from an unsupported carrier: `*_mno_raw` keeps the real brand, `*_mno_std = null`.
- `*_mno_raw` set + `*_mno_std = null` = unsupported upstream; `*_mno_raw = null` = native (no upstream).

### 3.3 How the schema makes "recent port-out within last 30 days" efficient to query

```sql
WHERE event_date >= date_sub(current_date, 30)   -- partition pruning
  AND is_inter_carrier_port                       -- inter-SPID only, no join
  AND from_mno_std IS NOT NULL                     -- ported out from Rogers/Bell/Telus
```

`event_date` prunes partitions; `is_inter_carrier_port` and the `from_mno_std` null-check need no dictionary join.

---

## 4. Derivations and dropped fields

### 4.1 SPID → carrier dictionary

External reference data, two levels:

- **L1** SPID → brand → `*_mno_raw`.
- **L2** brand → big-3 parent → `*_mno_std` (non-big-3 → null).

Unknown SPID → `*_mno_raw` null, logged to `dq/issues/`. Limits: current ownership (not point-in-time), resellers left independent. 

### 4.2 Dropped TU fields (Phase 1)

Dropped ≠ lost: bronze keeps the raw payload + `Raw_XML_Response`. 
| TU field | Reason for dropping |
|---|---|
| `Routing_*DpcSsn` (current snapshot) | SS7 signaling address (DPC/SSN), routing detail fraud doesn't use. |
| `History_*DpcSsn` (per-event) | Same, per-event signaling address. |
| `Additional_BillingId` / `EndUserLocation` / `EndUserLocationType` | Optional NPAC fields, essentially always null in the source. |
| `IpFields_VoiceUri` / `MmsUri` / `SmsUri` (incl. per-event) | IP service routing URIs, always null. |
| `Ownership_AltSpid` / `AltCompany` / `History_AltSpid` | Secondary service provider (reseller / facility), essentially always null. |
| `Routing_Lrn` (current snapshot) | Replaced by per-event `lrn` (current LRN = latest event's LRN). |
| `Ownership_SvType` (current snapshot) | Replaced by per-event `sv_type` (current SvType = latest event's SvType). |
| `Ownership_Spid` (current snapshot) | Was `current_owner_spid` / `current_owner_carrier_name` / `is_current_owner`; dropped — the snapshot can't be reliably tied to a port event, so "current owner" can't be confirmed. |
| `CodeInfo_CnaCodeOwner` / `CodeInfo_Ocn` | Was `code_owner_ocn` / `code_owner_name`; dropped — number-block allocation metadata (OCN / company), not the porting lineage. |
| `Ownership_Company` / `History_*Company` | Company-name format is dirty; derive `*_mno_*` via the SPID dict instead. |
| `API_ResponseMessage` | Only `ResponseCode` is used to check 3000. |
| `API_ResponseCode` | Used as a filter (keep only 3000), not stored as a column. |
| `Raw_XML <PhoneNumber>` (`raw_phone_number`) | PII, plaintext number; not kept in silver. Only `phone_number_AC_hash` is exposed. |
| `Raw_XML_Response` (`raw_xml_response`) | PII, contains plaintext `<PhoneNumber>`; not kept in silver. Bronze keeps the original immutable payload.|

---

## 5. Lineage join keys

### 5.1 Which columns join into msisdn_lifecycle

`phone_number_AC_hash` plus `event_timestamp` (plus `is_inter_carrier_port` / `from_mno_std` / `to_mno_std` / `from_to_same_carrier` to tell gold which boundary to insert, and skip same-brand SPID shuffles).

### 5.2 Which columns drive updated_source = transunion toggle

`source_name` (= `tu_portps`) tells gold to set `updated_source = transunion`; plus `is_inter_carrier_port` / `to_mno_std` / `from_mno_std` / `from_to_same_carrier` / `event_timestamp`.

### 5.3 Which columns feed the unresolved-lifecycle confidence check

`phone_number_AC_hash`, `is_inter_carrier_port`, `event_timestamp`, `from_mno_std`, `to_mno_std`, `from_to_same_carrier`.

---

## 6. Edge cases

| # | Scenario | How this table handles it |
|---|---|---|
| 1 | Cancelled port (cancelled before completion) | TU only returns completed ports; cancelled / in-flight ones never appear in the source. |
| 2 | Rejected port (losing carrier rejects) | Same, not visible in this source. |
| 3 | Port across a number-recycle | When a number is recycled (`RETURN_TO_NPAC_CODE_ASSIGNEE`), the next port's `from_spid` is set null so the new user isn't linked to the old owner. |
| 4 | Late-arriving event | `ingestion_ts` is the watermark; lands in an old `event_date` partition, `event_seq` re-orders. |
| 5 | Port to / from unsupported (TU sees only one side) | `*_mno_std` set null; the unsupported side's later activity isn't visible. |
| 6 | Empty 3000 (valid 3000, no porting data) | No porting history → 0 rows, counted in `invalid_input_breakdown`. |

---

## 7. Zero data loss and DQ

No data is dropped. Unusable input is bucketed; the raw payload always stays in bronze.

Buckets:

- `bad_xml`: `<TNResponse>` fails to parse or has an unexpected namespace.
- `non_3000`: `API_ResponseCode != 3000` (auth / permission / limit)
- `no_events` : valid 3000 with no `<PortingHistory>` → 0 rows. Classified in `invalid_input_breakdown` (US numbers, invalid NPA, etc.)
- `has_events`: valid 3000 with ≥1 event. Split under the port-only policy:
  - `has_inter_intra_spid`: has ≥1 Inter-/Intra-SPID event → silver rows.
  - `no_inter_intra_spid`: events but all non-port → 0 silver rows (still in bronze).

`response_breakdown` asserts closure (`total == non_3000 + ok_3000`, `ok_3000 == bad_xml + empty_3000 + has_events`, `has_events == has_inter_intra_spid + no_inter_intra_spid`). Silver rows come only from `has_inter_intra_spid`.

DQ dimensions:
| Dimension | Check |
|---|---|
| completeness | Non-null ratio for all 24 columns; the 14 `MUST_NONNULL` columns must be 1.0. |
| uniqueness | `record_id` unique (error-level PK); its natural-key preimage `(phone_number_AC_hash, event_timestamp, to_spid, event_type)` unique. `request_id` is non-null but not unique (repeats across a call's events). |
| consistency | (1) `is_inter_carrier_port` ⇔ `lnp_type == 0`; (2) `event_date == date(event_timestamp)`; (3) `event_seq` contiguous `1..n` per number. |
| validity | Values in range: `event_type` ∈ {INTER/INTRA}; `lnp_type` ∈ {0,1}; `sv_type` in vocab; `lrn` = 10 digits; SPIDs = 4 chars. |
| timeliness | Streaming mode: no event later than `ingestion_ts` (negative-gap check). The delay gate (`ingestion_ts - event_timestamp`) is left off — TU returns full history, so `event_timestamp` can be years old and the gap measures port age, not pipeline lag. |
| currency | Data freshness: `ingestion_ts` no older than a configurable max age before the run date. |
| referential_integrity | Orphan rate of `phone_number_AC_hash` vs `account_changes_batch` (left-anti). Expected non-zero; ≥ 0.99 = suspected hash mismatch. |

`API_ResponseCode` values:

| code | meaning |
|---|---|
| `3000` | Processed, passed validation, returns number data (the only one entering silver, but may be empty, see edge 6) |
| `2000` | Wrong LoginId / Password, Auth Failed |
| `2001` | No permission to access the API |
| `2009` | Number validation failed |
| `2010` | System Error |
| `2020` | Request needs at least one number |
| `2030` | A number is invalid |
| `2040` | Number count exceeds the user's limit |

`event_type` values (`<Event>` normalized):
| raw `<Event>` | normalized | `lnp_type` | `is_inter_carrier_port` |
|---|---|:--:|:--:|
| `Inter-SPID Port` | `INTER_SPID_PORT` | 0 | true |
| `Intra-SPID Port` | `INTRA_SPID_PORT` | 1 | false |

Other `<Event>` types (NPAC code assignment / return / SPID update / migration) and unmapped values are dropped.

---

## 8. Two example raw TU API responses walked through end-to-end
### Example A, `4186791160`

| bronze column | bronze value | silver column | silver value |
|---|---|---|---|
| bronze `request_id` (this API call) | `9f3c2a...-e41b` | `request_id` | `9f3c2a...-e41b` (illustrative) |
| synthesized | (none) | `record_id` | `9f3c2a...e41b` (illustrative) |
| Raw_XML `<PhoneNumber>` | `4186791160` | `phone_number_AC_hash` | `c4d9...8a17` (illustrative) |
| `<PortingInfo><Date>` | `2024-06-04 11:42:07.64805` | `event_timestamp` | `2024-06-04 06:42:07.648` (EST) |
| `<PortingInfo><Date>` | `2024-06-04 11:42:07.64805` | `event_date` | `2024-06-04` |
| `<PortingInfo><Event>` | `Inter-SPID Port` | `event_type` | `INTER_SPID_PORT` |
| ranked by time | (none) | `event_seq` | `10` |
| derived from `event_type` | `INTER_SPID_PORT` | `lnp_type` | `0` |
| derived from `event_type` | `INTER_SPID_PORT` | `is_inter_carrier_port` | `true` |
| `<PortingInfo><Spid>` (prev event lag) | `6574` | `from_spid` | `6574` |
| `from_spid` dict lookup | `6574` | `from_mno_raw` | `BELL` |
| big-3 lookup | `BELL` | `from_mno_std` | `BELL` |
| `<PortingInfo><Spid>` (this event) | `8303` | `to_spid` | `8303` |
| `to_spid` dict lookup | `8303` | `to_mno_raw` | `TELUS` |
| big-3 lookup | `TELUS` | `to_mno_std` | `TELUS` |
| `from_mno_raw` vs `to_mno_raw` | `BELL` vs `TELUS` | `from_to_same_carrier` | `0` |
| `<PortingInfo><SvType>` | `Wireless` | `sv_type` | `Wireless` |
| `<PortingInfo><LRN>` | `418-618-0002` | `lrn` | `4186180002` |
| (none) | (none) | `correlation_id` | `(null)` |
| constant | (none) | `source_name` | `tu_portps` |
| constant | (none) | `schema_version` | `1` |
| bronze `ingestion_ts` | `2026-06-10 14:00:00` | `ingestion_ts` | `2026-06-10 14:00:00` (illustrative) |
| runtime | (none) | `last_updated_ts` | `2026-06-10 14:00:00` (illustrative; == `ingestion_ts` with no upsert) |

### Example B, `4372216057`

| bronze column | bronze value | silver column | silver value |
|---|---|---|---|
| bronze `request_id` (this API call) | `1a2b3c...-9f8e` | `request_id` | illustrative |
| synthesized | (none) | `record_id` | illustrative |
| Raw_XML `<PhoneNumber>` | `4372216057` | `phone_number_AC_hash` | illustrative |
| `<PortingInfo><Date>` | `2019-05-23` (time-of-day per source) | `event_timestamp` | `2019-05-23 ...` (EST) |
| `<PortingInfo><Date>` | `2019-05-23` | `event_date` | `2019-05-23` |
| `<PortingInfo><Event>` | `Intra-SPID Port` | `event_type` | `INTRA_SPID_PORT` |
| ranked by time | (none) | `event_seq` | `2` |
| derived from `event_type` | `INTRA_SPID_PORT` | `lnp_type` | `1` |
| derived from `event_type` | `INTRA_SPID_PORT` | `is_inter_carrier_port` | `false` |
| `<PortingInfo><Spid>` (prev event lag) | `8821` | `from_spid` | `8821` |
| dict lookup | `8821` | `from_mno_raw` | `ROGERS` |
| big-3 lookup | `ROGERS` | `from_mno_std` | `ROGERS` |
| `<PortingInfo><Spid>` (this event) | `8821` | `to_spid` | `8821` |
| dict lookup | `8821` | `to_mno_raw` | `ROGERS` |
| big-3 lookup | `ROGERS` | `to_mno_std` | `ROGERS` |
| `from_mno_raw` vs `to_mno_raw` | `ROGERS` vs `ROGERS` | `from_to_same_carrier` | `1` |
| `<PortingInfo><SvType>` | `Wireless` | `sv_type` | `Wireless` |
| `<PortingInfo><LRN>` | `437-999-1111` | `lrn` | `4379991111` |
| (none) | (none) | `correlation_id` | `(null)` |
| constant | (none) | `source_name` | `tu_portps` |
| constant | (none) | `schema_version` | `1` |
| bronze `ingestion_ts` | illustrative | `ingestion_ts` | illustrative |
| runtime | (none) | `last_updated_ts` | illustrative (== `ingestion_ts` with no upsert) |
