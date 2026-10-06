# Bronze schema: tu_portps (batch)

Raw TransUnion Port/PS API responses, batch-exported to CSV and landed in
`s3://enstream-lake-bronze-dev/landing/`. One row per API response, immutable / as-received.
Two files, joined on `ID`. All columns land as strings (raw CSV).

---

## `BNC_processed.csv`

The batch pull's main file: one row per TU Port/PS response — TU's flattened fields plus the verbatim
`<TNResponse>` payload in `Raw_XML_Response`.

| Column | Description | Sample |
|---|---|---|
| `ID` | Row id, unique per response | `1` |
| `API_ResponseCode` | TU envelope code (`3000` = ok) | `3000` |
| `API_ResponseMessage` | TU envelope message | `Success` |
| `Raw_XML_Response` | Verbatim `<TNResponse>` document (see below) | `<TNResponse>…</TNResponse>` |
| `Ownership_PhoneNumber` | Current owner MSISDN (plaintext, PII) | `4186791160` |
| `Ownership_Spid` | Current owner SPID | `8303` |
| `Ownership_Company` | Current owner company name | `TELUS MOBILITY` |
| `Ownership_AltSpid` | Secondary service-provider SPID | `\N` |
| `Ownership_AltCompany` | Secondary service-provider company | `\N` |
| `Ownership_SvType` | Current service type | `Wireless` |
| `CodeInfo_CnaCodeOwner` | Number-block code owner | `BELL` |
| `CodeInfo_Ocn` | Operating company number | `6574` |
| `Routing_Lrn` | Current routing number | `418-618-0002` |
| `Routing_DpcSsn` | SS7 signaling address (DPC/SSN) | `254-100-003` |
| `IpFields_VoiceUri` | IP voice routing URI | `\N` |
| `IpFields_MmsUri` | IP MMS routing URI | `\N` |
| `IpFields_SmsUri` | IP SMS routing URI | `\N` |
| `Additional_BillingId` | NPAC billing id | `\N` |
| `EndUserLocation` | NPAC end-user location | `\N` |
| `EndUserLocationType` | NPAC end-user location type | `\N` |
| `History_Date` | Port event timestamp (UTC) | `2024-06-04 11:42:07.64805` |
| `History_Event` | Port event type (e.g. `Inter-SPID Port`) | `Inter-SPID Port` |
| `History_Spid` | Port-in SPID for the event | `8303` |
| `History_LRN` | Routing number for the event | `418-618-0002` |
| `History_SvType` | Service type for the event | `Wireless` |
| `History_DpcSsn` | Per-event SS7 signaling address | `254-100-003` |
| `History_AltSpid` | Per-event secondary SPID | `\N` |
| `History_Company` | Per-event company name | `TELUS MOBILITY` |

Porting history is multi-valued; the authoritative event stream is `<PortingHistory>` inside
`Raw_XML_Response`. The `History_*` columns are TU's denormalized copy.

---

## `BNC_hashed.csv`

Per-response phone-number hashes, kept out of the payload file. Joined to `BNC_processed.csv` on `ID`.

| Column | Description | Sample |
|---|---|---|
| `ID` | Join key to `BNC_processed.csv` | `1` |
| `acHash` | SHA-256 of the phone number (account-changes hashing) | `c4d9...8a17` (64 hex) |
| `atHash` | SHA-256 of the phone number (audit-trail hashing) | `7b2e...0f9c` (64 hex) |

---

## `Raw_XML_Response` — `<TNResponse>` shape

```xml
<TNResponse>
  <Ownership>
    <PhoneNumber>4165551234</PhoneNumber>
    <Spid>6574</Spid>
    <Company>BELL MOBILITY</Company>
  </Ownership>
  <CodeInfo>
    <CnaCodeOwner>BELL</CnaCodeOwner>
    <Ocn>6574</Ocn>
  </CodeInfo>
  <PortingHistory>
    <PortingInfo>
      <Date>2020-01-15 00:00:00</Date>
      <Event>Inter-SPID Port</Event>
      <Spid>6574</Spid>
      <LRN>416-555-0000</LRN>
      <SvType>wireline</SvType>
    </PortingInfo>
    <!-- ... older events, newest first ... -->
  </PortingHistory>
</TNResponse>
```
