"""Local pytest for the TU PortPS tulip silver cleaning / feature logic.

Extracted verbatim from the unit-test section of ``TU_PortPS_tulip_silver_tests.ipynb`` so the
invariant tests run in CI on a local Spark session, with no Glue connection. Keep in sync with that
notebook and ``TU_PortPS_tulip_silver.ipynb``. The DQ-validation half of the notebook is not
covered here: it reads the written Iceberg table and needs S3 / Glue.

Run: pytest project_lotus/notebooks/data_preparation/tests/test_tu_portps_tulip.py
"""
import contextlib
import re
import xml.etree.ElementTree as ET

from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import lit
from pyspark.sql.window import Window
from pyspark.sql.types import (
    ArrayType, BooleanType, MapType,
    StringType, StructField, StructType,
)

# One local Spark session for the module; the notebook uses a global `spark`, so the extracted
# helpers (`_df`, `_ts`, `_toronto_session`) and test functions close over this name unchanged.
spark = (
    SparkSession.builder.master("local[1]")
    .appName("tu_portps_tulip_unit")
    .config("spark.sql.shuffle.partitions", "1")
    .config("spark.ui.enabled", "false")
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")



# --- XML parse (mirrors the cleaning notebook) -------------------------------


# XML parse (mirrors the cleaning notebook).
parse_struct = StructType([
    StructField("parse_ok",     BooleanType(),                                  True),
    StructField("phone_number", StringType(),                                   True),
    StructField("snapshot",     MapType(StringType(), StringType()),            True),
    StructField("events",       ArrayType(MapType(StringType(), StringType())), True),
])

def _local(tag):
    return tag.split("}", 1)[1] if "}" in tag else tag

def parse_tnresponse(xml_str):
    if not xml_str or not xml_str.strip():
        return (False, None, {}, [])
    try:
        root = ET.fromstring(xml_str)
        if "}" in root.tag:
            return (False, None, {}, [])
        snapshot = {}
        def walk(el, prefix):
            for ch in el:
                name = _local(ch.tag)
                if name == "PortingHistory":
                    continue
                path = (prefix + "/" + name) if prefix else name
                if len(ch):
                    walk(ch, path)
                else:
                    text = ch.text.strip() if (ch.text and ch.text.strip()) else None
                    if text is not None:
                        snapshot[path] = text
        walk(root, "")
        events = []
        ph = root.find("PortingHistory")
        if ph is not None:
            for pi in ph.findall("PortingInfo"):
                m = {}
                for ch in pi:
                    text = ch.text.strip() if (ch.text and ch.text.strip()) else None
                    if text is not None:
                        m[_local(ch.tag)] = text
                events.append(m)
        return (True, snapshot.get("Ownership/PhoneNumber"), snapshot, events)
    except Exception:
        return (False, None, {}, [])


# --- norm + vocabularies -----------------------------------------------------


# norm + vocabularies (mirror the cleaning notebook).
def norm(c):
    c = F.trim(c)
    return F.when(c.isin("", "NULL", "\\N"), None).otherwise(c)

EVENT_TYPE_MAP = {
    "Inter-SPID Port":              "INTER_SPID_PORT",
    "Intra-SPID Port":              "INTRA_SPID_PORT",
    "NPAC Code Assignment":         "NPAC_CODE_ASSIGNMENT",
    "Return to NPAC Code Assignee": "RETURN_TO_NPAC_CODE_ASSIGNEE",
    "NPAC SPID Update":             "NPAC_SPID_UPDATE",
    "NPAC SPID Migration":          "NPAC_SPID_MIGRATION",
}
LNP_TYPE_MAP = {"INTER_SPID_PORT": 0, "INTRA_SPID_PORT": 1}
# sv_type allowed values (TU API guide, Appendix B "SV Type"). Unknown values go to dq/issues for review.
SV_TYPE_VOCAB = ["Wireline", "Wireless", "VoWIFI", "VoIP", "Pre-Paid Wireless",
                 "Class 2 Interconnected VoIP"]

# SPID dict, trimmed to what the tests touch (full maps in the cleaning notebook).
SPID_TO_CARRIER = {
    "6574": "BELL_MOBILITY", "5643": "FIDO", "8820": "FIDO", "497E": "SHAW",
    "8084": "TELUS", "8303": "TELUS_MOBILITY", "328F": "VIDEOTRON", "8306": "VIDEOTRON",
    "383F": "FREEDOM", "9868": "SASKTEL", "8088": "MTS", "329A": "ALIANT",
}
CARRIER_PARENT = {
    "BELL": "BELL", "BELL_MOBILITY": "BELL", "ALIANT": "BELL", "MTS": "BELL",
    "ROGERS": "ROGERS", "FIDO": "ROGERS", "SHAW": "ROGERS",
    "TELUS": "TELUS", "TELUS_MOBILITY": "TELUS",
}
BIG3 = ["BELL", "ROGERS", "TELUS"]


# --- feature logic as functions (incl. add_request_id) -----------------------


# Feature logic as functions for unit tests. Mirror the cleaning notebook.

def add_event_type(df):
    et_map = F.create_map([x for k, v in EVENT_TYPE_MAP.items() for x in (lit(k), lit(v))])
    df = df.withColumn("event_type", et_map[norm(F.col("raw_event"))])
    df = df.withColumn("_event_type_unknown",
                       norm(F.col("raw_event")).isNotNull() & F.col("event_type").isNull())
    return df

def add_lnp_type(df):
    lnp_map = F.create_map([x for k, v in LNP_TYPE_MAP.items() for x in (lit(k), lit(v))])
    df = df.withColumn("lnp_type", lnp_map[F.col("event_type")])
    df = df.withColumn("is_inter_carrier_port", F.coalesce(F.col("lnp_type") == lit(0), lit(False)))
    return df

def add_record_id(df):
    # record_id = 64-hex sha256 of the natural key.
    return df.withColumn("record_id", F.sha2(F.concat_ws("|",
        F.col("phone_number_AC_hash"),
        F.col("event_timestamp").cast("string"),
        F.coalesce(F.col("to_spid"), lit("")),
        F.coalesce(F.col("event_type"), lit("")),
    ), 256))

def add_event_seq_and_from_spid(df):
    w = Window.partitionBy("_phone").orderBy(F.col("event_timestamp").asc())
    df = df.withColumn("event_seq", F.row_number().over(w))
    df = df.withColumn("_prev_to_spid",    F.lag("to_spid").over(w))
    df = df.withColumn("_prev_event_type", F.lag("event_type").over(w))
    df = df.withColumn("from_spid",
        F.when(F.col("_prev_to_spid").isNull(), None)                                  # first event
         .when(F.col("event_type") == lit("NPAC_CODE_ASSIGNMENT"), None)
         .when(F.col("_prev_event_type") == lit("RETURN_TO_NPAC_CODE_ASSIGNEE"), None)  # recycle boundary
         .otherwise(F.col("_prev_to_spid")))
    return df

def add_mno(df, spid_col, raw_out, mno_out):
    brand_map  = F.create_map([x for k, v in SPID_TO_CARRIER.items() for x in (lit(k), lit(v))])
    parent_map = F.create_map([x for k, v in CARRIER_PARENT.items()  for x in (lit(k), lit(v))])
    big3_arr   = F.array(*[lit(c) for c in BIG3])
    df = df.withColumn(raw_out, brand_map[F.col(spid_col)])
    df = df.withColumn("_parent_tmp", parent_map[F.col(raw_out)])
    df = df.withColumn(mno_out,
                       F.when(F.array_contains(big3_arr, F.col("_parent_tmp")), F.col("_parent_tmp")).otherwise(None))
    return df.drop("_parent_tmp")

def add_from_to_same_carrier(df):
    # 1 = same raw brand, 0 = different, null when either side is null.
    return df.withColumn("from_to_same_carrier",
        F.when(F.col("from_mno_raw").isNull() | F.col("to_mno_raw").isNull(), None)
         .otherwise((F.col("from_mno_raw") == F.col("to_mno_raw")).cast("int")))


def add_request_id(df):
    # request_id = deterministic UUID from bronze id + natural key (unique, non-null). Mirrors
    # the tulip cleaning notebook; production reads it from the bronze request_id column.
    seed = F.concat_ws("|",
        F.col("bronze_row_id").cast("string"),
        F.col("phone_number_AC_hash"),
        F.col("event_timestamp").cast("string"),
        F.coalesce(F.col("to_spid"), lit("")),
        F.coalesce(F.col("event_type"), lit("")),
    )
    h = F.sha2(seed, 256)
    return df.withColumn("request_id", F.concat_ws("-",
        F.substring(h, 1, 8),  F.substring(h, 9, 4),  F.substring(h, 13, 4),
        F.substring(h, 17, 4), F.substring(h, 21, 12)))


# --- synthetic <TNResponse> fixtures -----------------------------------------


# Synthetic <TNResponse> docs for the parser variants.
OK_XML = (
    "<TNResponse>"
    "  <Ownership><PhoneNumber>4165551234</PhoneNumber><Spid>6574</Spid>"
    "    <Company>BELL MOBILITY</Company></Ownership>"
    "  <CodeInfo><CnaCodeOwner>BELL</CnaCodeOwner><Ocn>6574</Ocn></CodeInfo>"
    "  <PortingHistory>"
    "    <PortingInfo><Date>2020-01-15 00:00:00</Date><Event>Inter-SPID Port</Event>"
    "      <Spid>6574</Spid><LRN>416-555-0000</LRN><SvType>wireline</SvType></PortingInfo>"
    "  </PortingHistory>"
    "</TNResponse>"
)

# Valid 3000, no porting history (e.g. a US number).
EMPTY_XML = (
    "<TNResponse>"
    "  <Ownership><PhoneNumber>2025551234</PhoneNumber>"
    "    <Company>No customer agreement for region: Northeast</Company></Ownership>"
    "</TNResponse>"
)

# Two events in one response (history is new->old in the source).
MULTI_XML = (
    "<TNResponse>"
    "  <Ownership><PhoneNumber>5145550000</PhoneNumber><Spid>328F</Spid></Ownership>"
    "  <PortingHistory>"
    "    <PortingInfo><Date>2021-06-01 00:00:00</Date><Event>Inter-SPID Port</Event><Spid>328F</Spid></PortingInfo>"
    "    <PortingInfo><Date>2018-01-01 00:00:00</Date><Event>NPAC Code Assignment</Event><Spid>6574</Spid></PortingInfo>"
    "  </PortingHistory>"
    "</TNResponse>"
)

# Namespaced root -> parser marks it bad.
NS_XML = '<TNResponse xmlns="http://example.com/tn"><Ownership><Spid>6574</Spid></Ownership></TNResponse>'


# --- test helpers ------------------------------------------------------------


def _df(rows):
    """DataFrame from a list of Rows."""
    return spark.createDataFrame(rows)

def _ts(df):
    """Cast event_timestamp string to timestamp."""
    return df.withColumn("event_timestamp", F.col("event_timestamp").cast("timestamp"))

@contextlib.contextmanager
def _toronto_session():
    """Pin session tz to America/Toronto for a test."""
    prev = spark.conf.get("spark.sql.session.timeZone")
    spark.conf.set("spark.sql.session.timeZone", "America/Toronto")
    try:
        yield
    finally:
        spark.conf.set("spark.sql.session.timeZone", prev)


# --- tests: XML parse --------------------------------------------------------


# XML parse (pure Python).
def test_parse_ok_single_event():
    ok, phone, snap, events = parse_tnresponse(OK_XML)
    assert ok is True
    assert phone == "4165551234"
    assert snap["Ownership/Spid"] == "6574"
    assert snap["CodeInfo/CnaCodeOwner"] == "BELL"
    assert len(events) == 1
    assert events[0]["Event"] == "Inter-SPID Port" and events[0]["Spid"] == "6574"

def test_parse_no_events_response():
    ok, phone, snap, events = parse_tnresponse(EMPTY_XML)
    assert ok is True
    assert events == []
    assert snap["Ownership/Company"].startswith("No customer agreement")

def test_parse_multiple_events_in_document_order():
    ok, phone, snap, events = parse_tnresponse(MULTI_XML)
    assert ok is True and len(events) == 2
    assert events[0]["Spid"] == "328F" and events[1]["Spid"] == "6574"

def test_parse_snapshot_excludes_history_leaves():
    _, _, snap, _ = parse_tnresponse(OK_XML)
    assert all("PortingHistory" not in k for k in snap.keys())

def test_parse_namespace_marked_bad():
    ok, _, _, events = parse_tnresponse(NS_XML)
    assert ok is False and events == []

def test_parse_bad_and_empty_inputs():
    assert parse_tnresponse("not xml <<<")[0] is False
    assert parse_tnresponse("")[0] is False
    assert parse_tnresponse(None)[0] is False
    assert parse_tnresponse("   ")[0] is False


# --- tests: norm + vocabularies ----------------------------------------------


# norm + vocabularies.
def test_norm_nulls_empty_and_trim():
    rows = [Row(v=""), Row(v="NULL"), Row(v="\\N"), Row(v="  x  "), Row(v="y"), Row(v="   ")]
    out = [r[0] for r in _df(rows).select(norm(F.col("v"))).collect()]
    assert out == [None, None, None, "x", "y", None]

def test_event_type_known_and_unknown():
    rows = [Row(raw_event="Inter-SPID Port"), Row(raw_event="NPAC SPID Update"),
            Row(raw_event="Totally Unknown"), Row(raw_event=None)]
    out = add_event_type(_df(rows)).select("event_type", "_event_type_unknown").collect()
    assert out[0]["event_type"] == "INTER_SPID_PORT" and out[0]["_event_type_unknown"] is False
    assert out[1]["event_type"] == "NPAC_SPID_UPDATE"
    assert out[2]["event_type"] is None and out[2]["_event_type_unknown"] is True   # unknown is flagged
    assert out[3]["event_type"] is None and out[3]["_event_type_unknown"] is False  # null in -> not flagged

def test_lnp_type_and_inter_flag():
    rows = [Row(event_type="INTER_SPID_PORT"), Row(event_type="INTRA_SPID_PORT"),
            Row(event_type="NPAC_CODE_ASSIGNMENT")]
    out = add_lnp_type(_df(rows)).select("lnp_type", "is_inter_carrier_port").collect()
    assert out[0]["lnp_type"] == 0 and out[0]["is_inter_carrier_port"] is True
    assert out[1]["lnp_type"] == 1 and out[1]["is_inter_carrier_port"] is False
    assert out[2]["lnp_type"] is None and out[2]["is_inter_carrier_port"] is False


# --- tests: SPID -> carrier --------------------------------------------------


# SPID -> carrier.
def test_mno_brand_and_parent_rollup():
    rows = [Row(to_spid="5643"), Row(to_spid="6574"), Row(to_spid="328F"),
            Row(to_spid="9999"), Row(to_spid=None)]
    out = add_mno(_df(rows), "to_spid", "to_mno_raw", "to_mno_std").select("to_spid", "to_mno_raw", "to_mno_std").collect()
    by = {r["to_spid"]: (r["to_mno_raw"], r["to_mno_std"]) for r in out}
    assert by["5643"] == ("FIDO", "ROGERS")          # flanker -> big-3 parent
    assert by["6574"] == ("BELL_MOBILITY", "BELL")
    assert by["328F"] == ("VIDEOTRON", None)          # known brand, not big-3 -> std null
    assert by["9999"] == (None, None)                 # unknown spid
    assert by[None]  == (None, None)


# --- tests: record_id + from_to_same_carrier ---------------------------------


# record_id.
def test_record_id_deterministic_and_uses_to_spid():
    base = dict(phone_number_AC_hash="h1", event_timestamp="2020-01-01 00:00:00", event_type="INTER_SPID_PORT")
    rows = [Row(**base, to_spid="A"), Row(**base, to_spid="A"), Row(**base, to_spid="B")]
    df = add_record_id(_ts(_df(rows)))
    ids = [r[0] for r in df.select("record_id").collect()]
    assert ids[0] == ids[1]                # same inputs -> same id (deterministic / replay-safe)
    assert ids[0] != ids[2]                # to_spid is part of the 4-column key
    assert all(len(x) == 64 for x in ids)  # full sha256 hex

def test_from_to_same_carrier():
    rows = [Row(from_mno_raw="FIDO", to_mno_raw="FIDO"),   # same brand -> 1
            Row(from_mno_raw="BELL", to_mno_raw="TELUS"),  # different  -> 0
            Row(from_mno_raw=None,   to_mno_raw="TELUS")]  # one side null -> null
    vals = [r[0] for r in add_from_to_same_carrier(_df(rows)).select("from_to_same_carrier").collect()]
    assert vals == [1, 0, None]


# --- tests: event_seq + from_spid --------------------------------------------


# event_seq + from_spid.
def test_event_seq_orders_by_time_not_input_order():
    rows = [
        Row(_phone="p", event_timestamp="2020-03-01 00:00:00", to_spid="C", event_type="INTER_SPID_PORT"),
        Row(_phone="p", event_timestamp="2020-01-01 00:00:00", to_spid="A", event_type="INTER_SPID_PORT"),
        Row(_phone="p", event_timestamp="2020-02-01 00:00:00", to_spid="B", event_type="INTER_SPID_PORT"),
    ]
    out = add_event_seq_and_from_spid(_ts(_df(rows))).select("to_spid", "event_seq").collect()
    seq = {r["to_spid"]: r["event_seq"] for r in out}
    assert seq["A"] == 1 and seq["B"] == 2 and seq["C"] == 3   # by time, not input order

def test_from_spid_three_null_rules():
    rows = [
        Row(_phone="p", event_timestamp="2020-01-01 00:00:00", to_spid="A", event_type="INTER_SPID_PORT"),
        Row(_phone="p", event_timestamp="2020-02-01 00:00:00", to_spid="B", event_type="INTER_SPID_PORT"),
        Row(_phone="p", event_timestamp="2020-03-01 00:00:00", to_spid="C", event_type="NPAC_CODE_ASSIGNMENT"),
        Row(_phone="p", event_timestamp="2020-04-01 00:00:00", to_spid="D", event_type="RETURN_TO_NPAC_CODE_ASSIGNEE"),
        Row(_phone="p", event_timestamp="2020-05-01 00:00:00", to_spid="E", event_type="INTER_SPID_PORT"),
    ]
    out = add_event_seq_and_from_spid(_ts(_df(rows))).select("to_spid", "from_spid").collect()
    frm = {r["to_spid"]: r["from_spid"] for r in out}
    assert frm["A"] is None      # first event
    assert frm["B"] == "A"       # normal lag
    assert frm["C"] is None      # this row is NPAC_CODE_ASSIGNMENT
    assert frm["D"] == "C"       # RETURN row itself keeps its lagged from
    assert frm["E"] is None      # previous row was RETURN -> recycle boundary


# --- tests: request_id (tulip) -------------------------------------------


# request_id (tulip): deterministic UUID, unique + non-null per event.
import re
_UUID_RE = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"

def test_request_id_unique_nonnull_uuid_shape():
    rows = [
        Row(bronze_row_id=1, phone_number_AC_hash="h1", event_timestamp="2020-01-01 00:00:00", to_spid="A", event_type="INTER_SPID_PORT"),
        Row(bronze_row_id=1, phone_number_AC_hash="h1", event_timestamp="2020-01-01 00:00:00", to_spid="B", event_type="INTER_SPID_PORT"),
        Row(bronze_row_id=2, phone_number_AC_hash="h1", event_timestamp="2020-01-01 00:00:00", to_spid="A", event_type="INTER_SPID_PORT"),
    ]
    ids = [r[0] for r in add_request_id(_ts(_df(rows))).select("request_id").collect()]
    assert all(x is not None for x in ids)                       # non-null
    assert len(set(ids)) == len(ids)                             # unique per row (to_spid / bronze_row_id vary)
    assert all(re.match(_UUID_RE, x) for x in ids)               # UUID shape

def test_request_id_deterministic():
    row = [Row(bronze_row_id=7, phone_number_AC_hash="h9", event_timestamp="2021-05-05 00:00:00", to_spid="Z", event_type="INTRA_SPID_PORT")]
    a = add_request_id(_ts(_df(row))).select("request_id").collect()[0][0]
    b = add_request_id(_ts(_df(row))).select("request_id").collect()[0][0]
    assert a == b                                                # replay-stable
