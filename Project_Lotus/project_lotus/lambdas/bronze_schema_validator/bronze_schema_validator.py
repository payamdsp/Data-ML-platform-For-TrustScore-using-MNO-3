import os

import io
import json
import time
import uuid
import hashlib
import math
import re
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import pyarrow as pa
import pyarrow.parquet as pq


CONTROL_DATABASE = os.environ["CONTROL_DATABASE"]
CONTROL_TABLE = os.environ["CONTROL_TABLE"]
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
ATHENA_OUTPUT = os.environ.get("ATHENA_OUTPUT", "")
SCHEMA_BUCKET = os.environ["SCHEMA_BUCKET"]
SCHEMA_PREFIX = os.environ.get("SCHEMA_PREFIX", "schema_registry").strip("/")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "8"))
MERGE_CHUNK_SIZE = int(os.environ.get("MERGE_CHUNK_SIZE", "100"))
SEMANTIC_SAMPLE_SIZE = int(os.environ.get("SEMANTIC_SAMPLE_SIZE", "50"))
SEMANTIC_BATCH_SIZE = int(os.environ.get("SEMANTIC_BATCH_SIZE", "64"))
SEMANTIC_MAX_ROWS_SCANNED = int(os.environ.get("SEMANTIC_MAX_ROWS_SCANNED", "1024"))

# The trigger into Silver. Left blank, this Lambda validates and stops - which
# is what a standalone/local invocation wants. In the deployed environment
# both are set, and a PASS starts exactly one dataset's Silver run.
SILVER_STATE_MACHINE_ARN = os.environ.get("SILVER_STATE_MACHINE_ARN", "")
DATASET_SPECS = json.loads(os.environ.get("DATASET_SPECS_JSON", "{}"))

SUPPORTED_DATASETS = {
    "account_changes_batch",
    "device_lookup_batch",
    "audit_trail_services_3",
    "partner_configuration",
}

s3 = boto3.client("s3")
athena = boto3.client("athena")
sns = boto3.client("sns")
stepfunctions = boto3.client("stepfunctions")
emr = boto3.client("emr")


EMR_WORKFLOW_TAG_KEY = "lotus-workflow"
EMR_WORKFLOW_TAG_VALUE = "silver-gold-transient"
ACTIVE_EMR_STATES = ["STARTING", "BOOTSTRAPPING", "RUNNING", "WAITING"]



def parse_s3_uri(uri: str):
    p = urlparse(uri)
    if p.scheme != "s3" or not p.netloc:
        raise ValueError(f"Invalid S3 URI: {uri}")
    bucket = p.netloc
    prefix = p.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix


def normalize_folder_uri(uri: str):
    return uri.rstrip("/") + "/"


def terminate_workflow_clusters(event):
    """Terminate active EMR clusters created by this Silver/Gold workflow.

    This is an explicit maintenance action, never part of normal Bronze schema 
    validation. Requiring ``confirm`` and checking the cluster resource tag
    prevents an accidental validation event from terminating EMR workloads and
    keeps unrelated clusters outside this Lambda's scope.
    """
    if event.get("confirm") is not True:
        raise ValueError(
            "terminate_workflow_clusters requires confirm=true"
        )

    matched = []
    marker = None
    while True:
        request = {"ClusterStates": ACTIVE_EMR_STATES}
        if marker:
            request["Marker"] = marker

        response = emr.list_clusters(**request)
        for cluster in response.get("Clusters", []):
            cluster_id = cluster["Id"]
            cluster_detail = emr.describe_cluster(ClusterId=cluster_id)["Cluster"]
            tags = {
                tag["Key"]: tag.get("Value", "")
                for tag in cluster_detail.get("Tags", [])
            }
            if tags.get(EMR_WORKFLOW_TAG_KEY) == EMR_WORKFLOW_TAG_VALUE:
                matched.append({
                    "cluster_id": cluster_id,
                    "name": cluster.get("Name", ""),
                    "state": cluster.get("Status", {}).get("State", ""),
                })

        marker = response.get("Marker")
        if not marker:
            break

    terminated_cluster_ids = []
    for cluster in matched:
        emr.terminate_job_flows(JobFlowIds=[cluster["cluster_id"]])
        terminated_cluster_ids.append(cluster["cluster_id"])

    return {
        "action": "terminate_workflow_clusters",
        "tag_filter": {
            "key": EMR_WORKFLOW_TAG_KEY,
            "value": EMR_WORKFLOW_TAG_VALUE,
        },
        "active_states_checked": ACTIVE_EMR_STATES,
        "matched_clusters": matched,
        "termination_requested": terminated_cluster_ids,
        "termination_requested_count": len(terminated_cluster_ids),
    }


def sql_str(value):
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def json_text(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def stable_file_key(dataset_name: str, folder_path: str, file_path: str):
    raw = f"{dataset_name}|{folder_path}|{file_path}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_json_s3(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def canonical_contract_type(type_name: str):
    t = type_name.strip().lower()

    aliases = {
        # Integer types
        "long": "bigint",
        "int64": "bigint",
        "bigint": "bigint",

        "integer": "int",
        "int32": "int",
        "int16": "int",
        "int8": "int",
        "smallint": "int",
        "tinyint": "int",
        "int": "int",

        # String types
        "varchar": "string",
        "char": "string",
        "utf8": "string",
        "large_utf8": "string",
        "string": "string",

        # Boolean
        "bool": "boolean",
        "boolean": "boolean",

        # Float
        "float32": "float",
        "float": "float",

        "float64": "double",
        "double": "double",

        # Date
        "date32": "date",
        "date64": "date",
        "date": "date",

        # Binary
        "binary": "binary",
        "large_binary": "binary",

        # DateTime / Timestamp
        "datetime": "datetime",
        "timestamp": "datetime",
        "timestamp[ms]": "datetime",
        "timestamp[us]": "datetime",
        "timestamp[ns]": "datetime",
        "timestamp[s]": "datetime",
    }

    if t in aliases:
        return aliases[t]

    # Keep decimal precision and scale
    if t.startswith("decimal"):
        return t.replace(" ", "")

    return t


def load_expected_schema(dataset_name: str, requested_version=None):
    base = f"{SCHEMA_PREFIX}/{dataset_name}"
    if requested_version is None:
        current = read_json_s3(SCHEMA_BUCKET, f"{base}/current.json")
        version = int(current["active_version"])
    else:
        version = int(requested_version)

    schema_doc = read_json_s3(SCHEMA_BUCKET, f"{base}/v{version}.json")
    if schema_doc.get("dataset_name") != dataset_name:
        raise ValueError("Schema registry dataset mismatch")
    if int(schema_doc.get("schema_version")) != version:
        raise ValueError("Schema registry version mismatch")

    expected = {
        col["name"]: canonical_contract_type(col["type"])
        for col in schema_doc["columns"]
    }
    return version, expected


def canonical_arrow_type(dtype: pa.DataType):

    if pa.types.is_int64(dtype):
        return "bigint"

    if (
        pa.types.is_int8(dtype)
        or pa.types.is_int16(dtype)
        or pa.types.is_int32(dtype)
        or pa.types.is_uint8(dtype)
        or pa.types.is_uint16(dtype)
        or pa.types.is_uint32(dtype)
    ):
        return "int"

    if pa.types.is_uint64(dtype):
        return "uint64"

    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return "string"

    if pa.types.is_boolean(dtype):
        return "boolean"

    if pa.types.is_float32(dtype):
        return "float"

    if pa.types.is_float64(dtype):
        return "double"

    if pa.types.is_date32(dtype) or pa.types.is_date64(dtype):
        return "date"

    if pa.types.is_timestamp(dtype):
        return "datetime"

    if pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype):
        return "binary"

    if pa.types.is_decimal(dtype):
        return f"decimal({dtype.precision},{dtype.scale})"

    return str(dtype).lower().replace(" ", "")

def list_parquet_files(folder_uri: str):
    bucket, prefix = parse_s3_uri(folder_uri)
    paginator = s3.get_paginator("list_objects_v2")
    files = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.lower().endswith(".parquet"):
                continue
            files.append({
                "bucket": bucket,
                "key": key,
                "file_name": key.rsplit("/", 1)[-1],
                "file_path": f"s3://{bucket}/{key}",
                "etag": obj.get("ETag", "").strip('"'),
                "size": int(obj.get("Size", 0)),
            })

    files.sort(key=lambda x: x["key"])
    return files
class S3RangeReader(io.RawIOBase):
    """
    File-like object backed by S3 Range GET requests.

    PyArrow can seek/read this like a normal file.
    Internally, we fetch only the requested byte ranges from S3.
    """

    def __init__(self, s3_client, bucket, key, size):
        self.s3_client = s3_client
        self.bucket = bucket
        self.key = key
        self.size = size
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            new_pos = offset
        elif whence == io.SEEK_CUR:
            new_pos = self.pos + offset
        elif whence == io.SEEK_END:
            new_pos = self.size + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")

        if new_pos < 0:
            raise ValueError("Negative seek position")

        self.pos = new_pos
        return self.pos

    def read(self, n=-1):
        if self.pos >= self.size:
            return b""

        if n is None or n < 0:
            n = self.size - self.pos

        if n == 0:
            return b""

        end = min(self.pos + n, self.size) - 1

        response = self.s3_client.get_object(
            Bucket=self.bucket,
            Key=self.key,
            Range=f"bytes={self.pos}-{end}"
        )

        data = response["Body"].read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)

def read_parquet_schema(file_info):
    """
    Production-friendly Lambda version:
    Read Parquet schema using S3 Range GETs.

    This avoids:
    - downloading the whole file
    - scanning all rows
    - pyarrow.fs.S3FileSystem dependency
    """

    reader = S3RangeReader(
        s3_client=s3,
        bucket=file_info["bucket"],
        key=file_info["key"],
        size=file_info["size"]
    )

    arrow_schema = pq.read_schema(reader)

    actual = {}
    actual_verbose = {}

    for field in arrow_schema:
        actual[field.name] = canonical_arrow_type(field.type)
        actual_verbose[field.name] = {
            "arrow_type": str(field.type),
            "canonical_type": canonical_arrow_type(field.type),
            "nullable": bool(field.nullable),
        }

    return actual, actual_verbose


_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_DECIMAL_TYPE_RE = re.compile(r"^decimal\((\d+),(\d+)\)$")


def _string_value_matches_expected_type(value, expected_type: str) -> bool:
    """
    Strict semantic validation for a STRING value when the schema registry
    expects a non-string type.

    The registry remains authoritative. We do not infer types for columns that
    are expected to be strings.
    """
    if not isinstance(value, str):
        return False

    value = value.strip()

    # Empty string is not treated as NULL and is not coercible.
    if value == "":
        return False

    if expected_type == "bigint":
        if not _INTEGER_RE.fullmatch(value):
            return False
        number = int(value)
        return -(2 ** 63) <= number <= (2 ** 63 - 1)

    if expected_type == "int":
        if not _INTEGER_RE.fullmatch(value):
            return False
        number = int(value)
        return -(2 ** 31) <= number <= (2 ** 31 - 1)

    if expected_type == "uint64":
        if not _INTEGER_RE.fullmatch(value):
            return False
        number = int(value)
        return 0 <= number <= (2 ** 64 - 1)

    if expected_type == "date":
        try:
            # Strict ISO calendar date: YYYY-MM-DD.
            date.fromisoformat(value)
            return True
        except ValueError:
            return False

    if expected_type == "datetime":
        # Avoid treating a date-only string as a datetime.
        if "T" not in value and " " not in value:
            return False
        try:
            normalized = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
            datetime.fromisoformat(normalized)
            return True
        except ValueError:
            return False

    if expected_type in ("float", "double"):
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return False

    if expected_type == "boolean":
        # Keep this deliberately strict. Add other source encodings only if
        # the data contract explicitly allows them.
        return value.lower() in {"true", "false"}

    decimal_match = _DECIMAL_TYPE_RE.fullmatch(expected_type)
    if decimal_match:
        precision = int(decimal_match.group(1))
        scale = int(decimal_match.group(2))

        try:
            number = Decimal(value)
        except InvalidOperation:
            return False

        if not number.is_finite():
            return False

        sign, digits, exponent = number.as_tuple()
        digits_count = len(digits)

        if exponent >= 0:
            fractional_digits = 0
            integer_digits = digits_count + exponent
        else:
            fractional_digits = -exponent
            integer_digits = max(digits_count - fractional_digits, 0)

        return (
            fractional_digits <= scale
            and integer_digits <= (precision - scale)
            and integer_digits + fractional_digits <= precision
        )

    # No approved string -> expected-type coercion rule exists.
    return False


def read_string_column_sample(file_info, column_name: str):
    """
    Read a bounded sample from one Parquet column only.

    This is called only when:
      - the registry expects a non-string type, AND
      - the Parquet footer says the physical/canonical type is string.

    We deliberately do not store sampled values in the control table.
    """
    reader = S3RangeReader(
        s3_client=s3,
        bucket=file_info["bucket"],
        key=file_info["key"],
        size=file_info["size"],
    )

    parquet_file = pq.ParquetFile(reader)

    samples = []
    rows_scanned = 0

    for batch in parquet_file.iter_batches(
        batch_size=SEMANTIC_BATCH_SIZE,
        columns=[column_name],
        use_threads=False,
    ):
        values = batch.column(0).to_pylist()

        for value in values:
            rows_scanned += 1

            if value is not None:
                samples.append(value)
                if len(samples) >= SEMANTIC_SAMPLE_SIZE:
                    return samples, rows_scanned

            if rows_scanned >= SEMANTIC_MAX_ROWS_SCANNED:
                return samples, rows_scanned

    return samples, rows_scanned


def semantic_check_string_column(file_info, column_name: str, expected_type: str):
    """
    Secondary semantic check for string-wrapped non-string values.

    A successful check does NOT erase the physical schema drift. It converts
    the result from FAIL to WARN because the Parquet column is still physically
    STRING even though sampled values are compatible with the registry type.
    """
    samples, rows_scanned = read_string_column_sample(file_info, column_name)

    if not samples:
        return {
            "semantic_check": "FAIL",
            "coercible": False,
            "sampled_non_null_values": 0,
            "rows_scanned": rows_scanned,
            "semantic_reason": "no_non_null_values_found_in_bounded_sample",
        }

    for value in samples:
        if not _string_value_matches_expected_type(value, expected_type):
            return {
                "semantic_check": "FAIL",
                "coercible": False,
                "sampled_non_null_values": len(samples),
                "rows_scanned": rows_scanned,
                "semantic_reason": f"sample_contains_value_not_coercible_to_{expected_type}",
            }

    return {
        "semantic_check": "PASS",
        "coercible": True,
        "sampled_non_null_values": len(samples),
        "rows_scanned": rows_scanned,
        "semantic_reason": f"sample_values_coercible_to_{expected_type}",
    }


def compare_schema(expected, actual, file_info):
    expected_cols = set(expected)
    actual_cols = set(actual)

    added = sorted(actual_cols - expected_cols)
    removed = sorted(expected_cols - actual_cols)

    datatype_changes = []
    has_breaking_datatype_change = False
    has_coercible_datatype_change = False

    for col in sorted(expected_cols & actual_cols):
        expected_type = expected[col]
        actual_type = actual[col]

        if expected_type == actual_type:
            continue

        change = {
            "column": col,
            "expected": expected_type,
            "actual": actual_type,
        }

        # Targeted second layer:
        # only inspect row values when registry expects NON-string but the
        # Parquet footer reports STRING.
        if expected_type != "string" and actual_type == "string":
            semantic_result = semantic_check_string_column(
                file_info=file_info,
                column_name=col,
                expected_type=expected_type,
            )
            change.update(semantic_result)

            if semantic_result["coercible"]:
                has_coercible_datatype_change = True
            else:
                has_breaking_datatype_change = True
        else:
            change.update({
                "semantic_check": "NOT_APPLICABLE",
                "coercible": False,
                "semantic_reason": "physical_type_mismatch_not_eligible_for_string_coercion_check",
            })
            has_breaking_datatype_change = True

        datatype_changes.append(change)

    # Breaking drift
    if removed or has_breaking_datatype_change:
        status = "FAIL"

    # Non-breaking drift:
    # - extra columns
    # - physical STRING -> expected non-string where sampled values are coercible
    elif added or has_coercible_datatype_change:
        status = "WARN"

    # Exact physical/canonical match
    else:
        status = "PASS"

    return status, added, removed, datatype_changes


def validate_one_file(file_info, expected_schema, dataset_name, folder_path,
                      schema_version, attempt_id, folder_file_count):
    base = {
        "file_key": stable_file_key(dataset_name, folder_path, file_info["file_path"]),
        "attempt_id": attempt_id,
        "dataset_name": dataset_name,
        "folder_path": folder_path,
        "file_name": file_info["file_name"],
        "file_path": file_info["file_path"],
        "file_etag": file_info["etag"],
        "file_size": file_info["size"],
        "folder_file_count": folder_file_count,
        "schema_version": schema_version,
    }

    try:
        actual, verbose = read_parquet_schema(file_info)
        status, added, removed, changes = compare_schema(expected_schema, actual, file_info)
        return {
            **base,
            "status": status,
            "added_columns_json": json_text(added),
            "removed_columns_json": json_text(removed),
            "datatype_changes_json": json_text(changes),
            "actual_schema_json": json_text(verbose),
            "error_message": None,
        }
    except Exception as exc:
        return {
            **base,
            "status": "ERROR",
            "added_columns_json": "[]",
            "removed_columns_json": "[]",
            "datatype_changes_json": "[]",
            "actual_schema_json": "{}",
            "error_message": f"{type(exc).__name__}: {exc}"[:4000],
        }


def run_athena(sql: str):
    request = {
        "QueryString": sql,
        "QueryExecutionContext": {"Database": CONTROL_DATABASE},
        "WorkGroup": ATHENA_WORKGROUP,
    }

    # Only use customer-managed S3 query results if ATHENA_OUTPUT is provided.
    # If empty, Athena uses the workgroup configuration, e.g. Athena managed storage.
    if ATHENA_OUTPUT:
        request["ResultConfiguration"] = {
            "OutputLocation": ATHENA_OUTPUT
        }

    response = athena.start_query_execution(**request)
    query_id = response["QueryExecutionId"]

    while True:
        q = athena.get_query_execution(QueryExecutionId=query_id)
        state = q["QueryExecution"]["Status"]["State"]

        if state == "SUCCEEDED":
            return query_id

        if state in ("FAILED", "CANCELLED"):
            reason = q["QueryExecution"]["Status"].get(
                "StateChangeReason",
                "Unknown Athena error"
            )
            raise RuntimeError(f"Athena query {query_id} {state}: {reason}")

        time.sleep(0.8)


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def merge_pending_rows(rows):
    if not rows:
        return

    for batch in chunks(rows, MERGE_CHUNK_SIZE):
        values = []
        for r in batch:
            values.append(
                "(" + ",".join([
                    sql_str(r["file_key"]),
                    sql_str(r["attempt_id"]),
                    sql_str(r["dataset_name"]),
                    sql_str(r["folder_path"]),
                    sql_str(r["file_name"]),
                    sql_str(r["file_path"]),
                    sql_str(r["file_etag"]),
                    str(int(r["file_size"])),
                    str(int(r["folder_file_count"])),
                    str(int(r["schema_version"])),
                ]) + ")"
            )

        sql = f'''\
MERGE INTO "{CONTROL_DATABASE}"."{CONTROL_TABLE}" t
USING (
    VALUES {",".join(values)}
) AS s(file_key,attempt_id,dataset_name,folder_path,file_name,file_path,file_etag,file_size,folder_file_count,schema_version)
ON t.file_key = s.file_key
WHEN MATCHED THEN UPDATE SET
    attempt_id = s.attempt_id,
    dataset_name = s.dataset_name,
    folder_path = s.folder_path,
    file_name = s.file_name,
    file_path = s.file_path,
    file_etag = s.file_etag,
    file_size = s.file_size,
    folder_file_count = s.folder_file_count,
    schema_version = s.schema_version,
    status = 'PENDING',
    added_columns_json = '[]',
    removed_columns_json = '[]',
    datatype_changes_json = '[]',
    actual_schema_json = '{{}}',
    error_message = NULL,
    attempt_count = COALESCE(t.attempt_count, 0) + 1,
    validation_start_ts = current_timestamp,
    validation_end_ts = NULL
WHEN NOT MATCHED THEN INSERT (
    file_key,attempt_id,dataset_name,folder_path,file_name,file_path,file_etag,file_size,
    folder_file_count,schema_version,status,added_columns_json,removed_columns_json,
    datatype_changes_json,actual_schema_json,error_message,attempt_count,
    validation_start_ts,validation_end_ts
)
VALUES (
    s.file_key,s.attempt_id,s.dataset_name,s.folder_path,s.file_name,s.file_path,s.file_etag,s.file_size,
    s.folder_file_count,s.schema_version,'PENDING','[]','[]','[]','{{}}',NULL,1,current_timestamp,NULL
)
'''
        run_athena(sql)


def merge_validation_results(rows):
    if not rows:
        return

    for batch in chunks(rows, MERGE_CHUNK_SIZE):
        values = []
        for r in batch:
            values.append(
                "(" + ",".join([
                    sql_str(r["file_key"]),
                    sql_str(r["attempt_id"]),
                    sql_str(r["status"]),
                    sql_str(r["added_columns_json"]),
                    sql_str(r["removed_columns_json"]),
                    sql_str(r["datatype_changes_json"]),
                    sql_str(r["actual_schema_json"]),
                    sql_str(r["error_message"]),
                ]) + ")"
            )

        sql = f'''\
MERGE INTO "{CONTROL_DATABASE}"."{CONTROL_TABLE}" t
USING (
    VALUES {",".join(values)}
) AS s(file_key,attempt_id,status,added_columns_json,removed_columns_json,datatype_changes_json,actual_schema_json,error_message)
ON t.file_key = s.file_key
WHEN MATCHED THEN UPDATE SET
    attempt_id = s.attempt_id,
    status = s.status,
    added_columns_json = s.added_columns_json,
    removed_columns_json = s.removed_columns_json,
    datatype_changes_json = s.datatype_changes_json,
    actual_schema_json = s.actual_schema_json,
    error_message = s.error_message,
    validation_end_ts = current_timestamp
'''
        run_athena(sql)


def publish_alert(dataset_name, folder_path, attempt_id, counts):
    if not SNS_TOPIC_ARN:
        return
    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject=f"Bronze schema validation failed: {dataset_name}",
        Message=json.dumps({
            "dataset_name": dataset_name,
            "folder_path": folder_path,
            "attempt_id": attempt_id,
            **counts,
        }, indent=2),
    )


_RUN_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def resolve_run_date(event, folder_path):
    """The Silver dated-folder date, not "today" - a backfill names an older one.

    Prefers an explicit event["run_date"] so a caller with its own naming
    convention never has to match a regex. Falls back to the last YYYY-MM-DD
    substring in folder_path, which matches both an "ingest_date=YYYY-MM-DD/"
    partition and a bare "YYYY-MM-DD/" dated folder without this Lambda having
    to know which convention the caller uses.
    """
    if event.get("run_date"):
        return event["run_date"]
    match = _RUN_DATE_RE.search(folder_path)
    if not match:
        raise ValueError(
            f"Could not derive run_date from folder_path {folder_path!r}; "
            "pass 'run_date' explicitly in the event"
        )
    return match.group(1)


def start_silver_execution(dataset_name, run_date, folder_path, attempt_id):
    """Fire this dataset's own Silver run. No join with any other dataset.

    Each of account_changes_batch / device_lookup_batch / audit_trail_services_3
    lands and validates on its own schedule, sometimes hours apart. There is
    deliberately no wait here for "the other datasets are ready too" - that
    join happens downstream, in the Silver->Gold control Lambda's `evaluate`
    action, which re-checks the full set fresh every time any one dataset's
    Silver run succeeds. See accounts/data-sandbox/infra/stepfunctions_bronze_silver.tf.

    Returns the execution ARN, "ALREADY_STARTED" for an idempotent retry of
    the same validation attempt, or None if no Silver job is wired up for
    this dataset (SILVER_STATE_MACHINE_ARN unset, or the dataset - e.g.
    partner_configuration - has no entry in DATASET_SPECS).
    """
    if not SILVER_STATE_MACHINE_ARN:
        return None
    spec = DATASET_SPECS.get(dataset_name)
    if spec is None:
        return None

    # Include the validation attempt so a corrected/re-landed batch can start
    # a new Silver execution after an earlier attempt failed. Repeated starts
    # from the same Lambda attempt remain idempotent.
    execution_name = f"silver-{dataset_name}-{run_date}-{attempt_id}"[:80]
    try:
        response = stepfunctions.start_execution(
            stateMachineArn=SILVER_STATE_MACHINE_ARN,
            name=execution_name,
            input=json.dumps({
                "run_date": run_date,
                "dataset_name": dataset_name,
                "folder_path": folder_path,
                "table": spec["table"],
                "quarantine_table": spec["quarantine_table"],
                "quarantine_table_location": spec["quarantine_table_location"],
            }),
        )
        return response["executionArn"]
    except stepfunctions.exceptions.ExecutionAlreadyExists:
        return "ALREADY_STARTED"


def lambda_handler(event, context):
    """
    Validation event example:
    {
      "dataset_name": "account_changes_batch",
      "folder_path": "s3://bucket/.../ingest_date=2026-09-03/",
      "schema_version": 1
    }

    Explicit tagged-cluster cleanup event:
    {
      "action": "terminate_workflow_clusters",
      "confirm": true
    }

    schema_version is optional. If omitted, current.json is used.
    Trigger only after the Bronze folder is complete.
    """
    if event.get("action") == "terminate_workflow_clusters":
        return terminate_workflow_clusters(event)

    dataset_name = event["dataset_name"]
    folder_path = normalize_folder_uri(event["folder_path"])
    requested_version = event.get("schema_version")

    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Expected one of {sorted(SUPPORTED_DATASETS)}"
        )

    attempt_id = str(uuid.uuid4())
    schema_version, expected_schema = load_expected_schema(dataset_name, requested_version)

    parquet_files = list_parquet_files(folder_path)
    folder_file_count = len(parquet_files)
    if folder_file_count == 0:
        raise RuntimeError(f"No .parquet files found under {folder_path}")

    pending_rows = []
    for f in parquet_files:
        pending_rows.append({
            "file_key": stable_file_key(dataset_name, folder_path, f["file_path"]),
            "attempt_id": attempt_id,
            "dataset_name": dataset_name,
            "folder_path": folder_path,
            "file_name": f["file_name"],
            "file_path": f["file_path"],
            "file_etag": f["etag"],
            "file_size": f["size"],
            "folder_file_count": folder_file_count,
            "schema_version": schema_version,
        })

    # Register ALL files as PENDING first. 
    # If this invocation crashes later, Silver sees PENDING and blocks.
    merge_pending_rows(pending_rows)

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                validate_one_file,
                f,
                expected_schema,
                dataset_name,
                folder_path,
                schema_version,
                attempt_id,
                folder_file_count,
            ): f
            for f in parquet_files
        }

        for future in as_completed(futures):
            results.append(future.result())

    merge_validation_results(results)

    passed = sum(r["status"] == "PASS" for r in results)
    warnings = sum(r["status"] == "WARN" for r in results)
    failed = sum(r["status"] == "FAIL" for r in results)
    errors = sum(r["status"] == "ERROR" for r in results)

    allowed = passed + warnings

    overall_status = (
        "PASS"
        if len(results) == folder_file_count and allowed == folder_file_count
        else "BLOCK"
    )

    counts = {
        "files_found": folder_file_count,
        "files_checked": len(results),
        "files_passed": passed,
        "files_warning": warnings,
        "files_failed_schema": failed,
        "files_error": errors,
        "overall_status": overall_status,
        "schema_version": schema_version,
    }

    if overall_status != "PASS":
        publish_alert(dataset_name, folder_path, attempt_id, counts)
        silver_execution_arn = None
    else:
        run_date = resolve_run_date(event, folder_path)
        silver_execution_arn = start_silver_execution(
            dataset_name, run_date, folder_path, attempt_id
        )

    return {
        "dataset_name": dataset_name,
        "folder_path": folder_path,
        "attempt_id": attempt_id,
        "silver_execution_arn": silver_execution_arn,
        **counts,
    }
