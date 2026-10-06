"""SageMaker Processing (Spark cluster) entry point: bi-weekly feature engineering.

Gold/Silver tables in, feature tables in S3 out - the input every later stage
(prepare, sweep, finalize, nightly scoring) reads. Runs as a SageMaker Processing
job on the managed Spark container (``Dockerfile.features``), across as many
instances as the Terraform asks for.

Where the logic comes from
--------------------------
* The **feature library** is ``trust_score_05.features`` - updated in this copy
  to the v5 notebook (``Helena/updated_biweekly_feature_extraction_v5_resume.ipynb``)
  verbatim: every function it shares with the notebook is code-identical.
* The **driver** below (window generation, cohort sampling, the per-window loop,
  the feature-group writer, the manifest and resume logic, ``materialize`` and
  ``build_and_materialize_changes``) is copied **verbatim** from the same
  notebook, cells 8, 10, 14, 26, 27 and 28. The notebook set its settings as
  constants at the top of each cell; here the same names are set from the job's
  parameters *before* the functions are defined, because several functions bind
  them as default arguments.
* Only three pieces are new, and each is marked ``SAGEMAKER ADAPTATION``:
  reading the inputs by configured name (the notebook hard-coded table names),
  assembling the written feature groups into the layout ``trust_score_05.ml``
  reads (with the package's own ``join_feature_frames``), and attaching fraud
  labels for the evaluation population.

Parameters
----------
Every setting comes from the pipeline config JSON rendered by Terraform
(``trust_score_ml_v2_config.tf``), section ``"features"``. SageMaker downloads it
to ``/opt/ml/processing/input/config/``; ``--config`` names the file.

    feature_engineering.py --config /opt/ml/processing/input/config/pipeline_config.json \
        --mode training
    feature_engineering.py --config ... --mode scoring --batch-date 2026-10-01

``training``: every window between ``window_start_ts`` and ``window_end_exclusive_ts``,
then the ML layout (non-fraud per window + fraud per window).
``scoring``: one window whose reference time is the batch date, then one wide
table at ``<features_root>scoring/batch_date=<date>/data/`` for ``inference.py``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from functools import reduce
import json
import math
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# The package ships to every executor as a zip (--py-files); on the driver the
# image's /opt/program is also on the path.
_PROGRAM_DIR = os.environ.get("PROGRAM_DIR", os.path.dirname(os.path.abspath(__file__)))
if _PROGRAM_DIR not in sys.path:
    sys.path.insert(0, _PROGRAM_DIR)

from pyspark.sql import DataFrame, SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql.types import LongType, StringType, StructField, StructType  # noqa: E402
from pyspark.storagelevel import StorageLevel  # noqa: E402

from trust_score_05.features.assemble import (  # noqa: E402
    ENSTREAM_FAMILY,
    assemble_scope_feature_groups,
    join_feature_frames,
)
from trust_score_05.features.config import (  # noqa: E402
    DEFAULT_STATE_TRANSITIONS,
    GENERIC_EVENT_FAMILIES,
    SCOPES,
    FeatureConfig,
)
from trust_score_05.features.events import (  # noqa: E402
    build_changes_table,
    normalize_enstream_events,
    preprocess_enstream_events,
)
from trust_score_05.features.expressions import (  # noqa: E402
    closed_lineage_intervals,
    dedupe_columns,
)
from trust_score_05.features.snapshots import (  # noqa: E402
    attach_events_to_scope,
    build_reference_df,
    build_scope_snapshots,
)


# ==========================================================================
# SAGEMAKER ADAPTATION: parameters, loaded before the notebook's functions are
# defined (several of them bind these names as default arguments).
# ==========================================================================

def _cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="feature_engineering.py")
    parser.add_argument("--config", default=os.environ.get("TS05_PIPELINE_CONFIG"),
                        help="The pipeline config JSON rendered by Terraform.")
    parser.add_argument("--mode", choices=("training", "scoring"), default="training")
    parser.add_argument("--batch-date", default=None, help="scoring mode: YYYY-MM-DD")
    parser.add_argument("--run-id", default=os.environ.get("TS05_RUN_ID"))
    args, _ = parser.parse_known_args(argv)
    return args


def _load_params(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        raise SystemExit("feature_engineering.py: --config (or TS05_PIPELINE_CONFIG) is required")
    with open(path, encoding="utf-8") as handle:
        doc = json.load(handle)
    if "features" not in doc:
        raise SystemExit(f"{path} has no 'features' section")
    return doc["features"]


ARGS = _cli()
PARAMS = _load_params(ARGS.config)
FEATURES_ROOT = PARAMS["features_root"].rstrip("/")

# cell 04 (runtime switches) and cell 08 (windows)
ENABLE_ROW_COUNTS = bool(PARAMS.get("enable_row_counts", False))
DEBUG_SHOW = False
if ARGS.mode == "scoring":
    if not ARGS.batch_date:
        raise SystemExit("--mode scoring needs --batch-date YYYY-MM-DD")
    # One window whose reference time is the batch date (the notebook's windows
    # take reference_ts = window start), so the features describe each customer
    # as of the night being scored.
    _day = datetime.strptime(ARGS.batch_date, "%Y-%m-%d")
    WINDOW_START_TS = _day.strftime("%Y-%m-%d %H:%M:%S")
    WINDOW_END_EXCLUSIVE_TS = (_day + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    WINDOW_SIZE_DAYS = 1
    USE_WINDOW_END_AS_REFERENCE_TS = False
else:
    WINDOW_START_TS = PARAMS["window_start_ts"]
    WINDOW_END_EXCLUSIVE_TS = PARAMS["window_end_exclusive_ts"]
    WINDOW_SIZE_DAYS = int(PARAMS.get("window_size_days", 14))
    USE_WINDOW_END_AS_REFERENCE_TS = bool(PARAMS.get("use_window_end_as_reference_ts", False))

# cell 10
SCRATCH_BASE_PATH = f"{FEATURES_ROOT}/_scratch"

# cell 26
OUTPUT_VERSION = PARAMS.get("output_version", "v5.0.bi-weekly")
GOLD_FEATURE_BASE_PATH = f"{FEATURES_ROOT}/groups/{ARGS.mode}"
OUTPUT_WRITE_MODE = "overwrite"
FORCE_OVERWRITE = bool(PARAMS.get("force_overwrite", False))
RUN_ENSTREAM_ONLY = False
ENSTREAM_ONLY_GROUPS = ("enstream_api_call", "enstream_partner", "state_transitions")
REUSE_SCOPED_EVENTS = bool(PARAMS.get("reuse_scoped_events", True))
FAIL_FAST = bool(PARAMS.get("fail_fast", False))
OUTPUT_PARTITIONS = int(PARAMS.get("output_partitions", 512))
MIN_OUTPUT_PARTITIONS = int(PARAMS.get("min_output_partitions", 128))
TARGET_OUTPUT_FILE_BYTES = 128 * 1024 * 1024
ASSUMED_BYTES_PER_VALUE = 8
ASSUMED_BYTES_PER_STRING = 32
ASSUMED_PARQUET_COMPRESSION = 0.35
OUTPUT_WRITE_FORMAT = "parquet"

# cell 27
SAMPLE_ENABLED = bool(PARAMS.get("sample_enabled", False))
SAMPLE_FRACTION = float(PARAMS.get("sample_fraction", 1.0))
SAMPLE_BUCKETS = int(PARAMS.get("sample_buckets", 100))
SAMPLE_SALT = str(PARAMS.get("sample_salt", "biweekly_v4"))
SAMPLE_PER_WINDOW = bool(PARAMS.get("sample_per_window", False))
SAMPLE_VERIFY = bool(PARAMS.get("sample_verify", True))
SAMPLE_TAG = (
    f"frac{int(round(SAMPLE_FRACTION * 1000)):04d}_{SAMPLE_SALT}"
    + ("_perwindow" if SAMPLE_PER_WINDOW else "")
    if SAMPLE_ENABLED
    else "full"
)
_sample_verified = False

# cell 28
RESUME_RUN_ID = PARAMS.get("resume_run_id") or None
MAX_WINDOWS_PER_RUN: Optional[int] = PARAMS.get("max_windows_per_run")
RUN_ID = RESUME_RUN_ID or ARGS.run_id or f"biweekly_feature_extraction_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
ATTEMPT_ID = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
MANIFEST_BASE_PATH = (
    f"{GOLD_FEATURE_BASE_PATH}/_manifests/biweekly_feature_extraction"
    f"/run_id={RUN_ID}/attempt={ATTEMPT_ID}"
)
COMPLETED_GROUPS: Set[Tuple[str, str]] = set()

# cell 04: the notebook runs with FeatureConfig's (v5) defaults.
fcfg = FeatureConfig()

# Set by main(); the notebook's functions read the global `spark`.
spark: Optional[SparkSession] = None


# ==========================================================================
# Verbatim from the notebook below this line, unless marked.
# ==========================================================================


# ---------------- notebook cell 08 (verbatim) ----------------

def log(message: str) -> None:
    print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] {message}")

def parse_ts(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")

def fmt_ts(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")

def build_biweekly_windows(start_ts: str, end_exclusive_ts: str, window_days: int) -> List[Dict[str, str]]:
    start_dt = parse_ts(start_ts)
    end_dt = parse_ts(end_exclusive_ts)
    if start_dt >= end_dt:
        raise ValueError("WINDOW_START_TS must be before WINDOW_END_EXCLUSIVE_TS.")
    if window_days <= 0:
        raise ValueError("WINDOW_SIZE_DAYS must be positive.")

    windows: List[Dict[str, str]] = []
    current = start_dt
    idx = 1
    while current < end_dt:
        nxt = min(current + timedelta(days=window_days), end_dt)
        reference_dt = nxt if USE_WINDOW_END_AS_REFERENCE_TS else current
        windows.append(
            {
                "window_id": f"bw_{idx:02d}",
                "window_start_ts": fmt_ts(current),
                "window_end_ts": fmt_ts(nxt),
                "reference_ts": fmt_ts(reference_dt),
                "partition_ts": fmt_ts(current),
                "window_days": str((nxt - current).days),
            }
        )
        current = nxt
        idx += 1
    return windows

def validate_windows(windows: Sequence[Dict[str, str]]) -> None:
    if not windows:
        raise ValueError("No bi-weekly windows were generated.")

    previous_end = None
    seen_ids = set()
    for w in windows:
        window_id = w["window_id"]
        if window_id in seen_ids:
            raise ValueError(f"Duplicate window_id found: {window_id}")
        seen_ids.add(window_id)

        start_dt = parse_ts(w["window_start_ts"])
        end_dt = parse_ts(w["window_end_ts"])
        ref_dt = parse_ts(w["reference_ts"])

        if start_dt >= end_dt:
            raise ValueError(f"Invalid window {window_id}: start >= end.")
        if previous_end is not None and start_dt != previous_end:
            raise ValueError(
                f"Invalid windows: expected {fmt_ts(previous_end)} as next start, got {w['window_start_ts']}"
            )
        if not (start_dt <= ref_dt <= end_dt):
            raise ValueError(f"Invalid reference_ts for {window_id}: {w['reference_ts']} is outside the window.")

        previous_end = end_dt

    expected_final = parse_ts(WINDOW_END_EXCLUSIVE_TS)
    if previous_end != expected_final:
        raise ValueError(f"Last window ends at {fmt_ts(previous_end)}, expected {fmt_ts(expected_final)}")

    log("Window validation passed. Generated windows:")
    for w in windows:
        log(
            f"{w['window_id']}: start={w['window_start_ts']} | end={w['window_end_ts']} "
            f"| reference_ts={w['reference_ts']} | partition_dt={w['partition_ts']}"
        )

# ---------------- notebook cell 10 (verbatim) ----------------

def materialize(
    df: DataFrame,
    name: str,
    run_id: str,
    base_path: str = SCRATCH_BASE_PATH,
    partition_by: Optional[Sequence[str]] = None,
) -> DataFrame:
    """Write ``df`` to S3 Parquet and return a fresh reader over it.
 
    This is a DAG cut, not a cache. Everything upstream -- the window
    functions, the unions, the range joins -- runs exactly once. Every later
    reader gets a columnar scan with predicate and projection pushdown instead
    of a replay.
 
    Preferred over ``persist(DISK_ONLY)`` here because local disk is the
    constrained resource: DISK_ONLY competes with shuffle spill for the same
    volume, which is what produced the "No space left on device" failures.
    """
    path = f"{base_path}/run_id={run_id}/{name}"
    writer = df.write.mode("overwrite").format("parquet")
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.save(path)
    return spark.read.parquet(path)  # noqa: F821  (spark is notebook-global)

# ---------------- notebook cell 14 (verbatim) ----------------

def build_and_materialize_changes(
    lineage_df, account_changes_df, enstream_df, hash_mapping_df, run_id, cfg,
):
    lineage = materialize(
        closed_lineage_intervals(lineage_df, cfg), "closed_lineage", run_id,
    )

    preprocessed = preprocess_enstream_events(
        enstream_df,
        hash_mapping_df,
        hash_mapping_ac_col="phone_number_AC_hash",
        hash_mapping_at_col="phone_number_AT_hash",
        enstream_msisdn_col="phone_number_AT_hash",
        cfg=cfg,
    )
    normalized_enstream = materialize(
        normalize_enstream_events(preprocessed, lineage, cfg),
        "normalized_enstream",
        run_id,
    )

    # RAW account_changes_df. build_changes_table normalizes it internally,
    # and builds the number-change and port frames internally too — so no
    # pre-normalized frame and no extra_event_frames.
    changes = build_changes_table(
        account_changes_df=account_changes_df,
        lineage_df=lineage,
        normalized_enstream_events_df=normalized_enstream,
        cfg=cfg,
    )
    changes_df = materialize(
        changes, "changes", run_id, partition_by=[cfg.event_family_col],
    )
    return lineage, changes_df, normalized_enstream

# ---------------- notebook cell 26 (verbatim) ----------------

def safe_count(df: DataFrame, label: str) -> Optional[int]:
    if not ENABLE_ROW_COUNTS:
        log(f"Row count skipped for {label} because ENABLE_ROW_COUNTS=False")
        return None
    try:
        cnt = df.count()
        log(f"Row count [{label}]: {cnt:,}")
        return int(cnt)
    except Exception as exc:
        log(f"Count failed for {label}: {exc}")
        if FAIL_FAST:
            raise
        return None

def output_partitions_for(
    output_df: DataFrame,
    entity_cols: Sequence[str],
    row_count: Optional[int],
) -> int:
    """How many files one feature group's output should be written as.

    Estimates a row's uncompressed width from the frame's own schema --
    ``dtypes`` is metadata, so this costs no Spark job -- multiplies by the row
    count, applies the assumed compression, and divides by the target file size.

    ``row_count`` is the caller's ``snapshot_row_count``, which is the right
    number: the output is a LEFT join from the scope snapshot onto one feature
    frame, and every feature frame is grouped by ``entity_cols`` so it carries
    at most one row per key. Output rows therefore equal snapshot rows. Were a
    calculator ever to emit duplicate keys the join would fan out and this
    would under-partition -- which is what the duplicate-key check that used to
    run here was guarding.

    Falls back to the ``OUTPUT_PARTITIONS`` ceiling when the row count is
    unavailable (``ENABLE_ROW_COUNTS=False``) or zero, since guessing is worse
    than the old behaviour. The result is clamped to
    ``[1, OUTPUT_PARTITIONS]``, so this can only lower the file count, never
    raise it past the cap.
    """
    if not row_count or row_count <= 0:
        return OUTPUT_PARTITIONS

    key_cols = set(entity_cols)
    string_cols = sum(
        1 for name, dtype in output_df.dtypes if dtype == "string" or name in key_cols
    )
    other_cols = len(output_df.dtypes) - string_cols

    bytes_per_row = (
        other_cols * ASSUMED_BYTES_PER_VALUE + string_cols * ASSUMED_BYTES_PER_STRING
    )
    estimated_bytes = row_count * bytes_per_row * ASSUMED_PARQUET_COMPRESSION
    wanted = math.ceil(estimated_bytes / TARGET_OUTPUT_FILE_BYTES)  # noqa: F821
    return max(MIN_OUTPUT_PARTITIONS, min(OUTPUT_PARTITIONS, wanted))

def split_feature_set_name(scope: str, feature_group: str) -> str:
    return f"time_aware_event_based_behavioural_{scope}_{feature_group}"

def fmt_partition_ts(value) -> str:
    if isinstance(value, str):
        value = parse_ts(value)          # window["partition_ts"] is "%Y-%m-%d %H:%M:%S"
    return value.strftime("%Y-%m-%dT%H_%M_%S")

def _path_exists(path: str) -> bool:
    """Cheap existence check (a Hadoop FS metadata call, not a data scan).

    Used to checkpoint both the scratch tables (closed_lineage/changes/
    normalized_enstream/scoped_events) and the final per-window feature
    group outputs, so a re-run of the same run_id can skip anything already
    written instead of recomputing it.
    """
    hadoop_conf = spark._jsc.hadoopConfiguration()  # noqa: F821
    jpath = spark._jvm.org.apache.hadoop.fs.Path(path)  # noqa: F821
    fs = jpath.getFileSystem(hadoop_conf)
    return fs.exists(jpath)

def _scratch_table_path(name: str, run_id: Optional[str] = None) -> str:
    return f"{SCRATCH_BASE_PATH}/run_id={run_id or RUN_ID}/{name}"  # noqa: F821

def expected_feature_groups() -> Tuple[str, ...]:
    """Every feature group a window is supposed to produce, by output name.

    Derived from the same constants the write loop uses, so it cannot drift
    from what actually gets written, and normalised the way the manifest
    records it -- `enstream_api_call` is written as `enstream_api_call_generic`.
    """
    if RUN_ENSTREAM_ONLY:  # noqa: F821
        groups = ENSTREAM_ONLY_GROUPS  # noqa: F821
    else:
        groups = tuple(GENERIC_EVENT_FAMILIES) + (  # noqa: F821
            "enstream_partner",
            "stability",
            "state_transitions",
        )
    return tuple(
        "enstream_api_call_generic" if g == "enstream_api_call" else g for g in groups
    )

def load_completed_groups(
    run_id: str,
    gold_base_path: str = GOLD_FEATURE_BASE_PATH,  # noqa: F821
    scope: Optional[str] = None,
) -> Set[Tuple[str, str]]:
    """The (window_id, feature_group) pairs that succeeded under ``run_id``.

    Read from the manifest rather than by probing the output paths, and that
    distinction is the whole point of this function. `_path_exists` was what
    decided "already done" before, and it cannot tell a finished write from a
    half-written directory: `.save` in overwrite mode clears the path and then
    writes, so a task that dies partway leaves a directory that exists and is
    incomplete. Every failure in the run being resumed died exactly there, at
    `.save`, so path probing would have declared a good number of ruined
    outputs complete and skipped them for good. The manifest only records
    `success` after the write returned.

    Both the original manifest and every later `attempt=` manifest under the
    same run id are read, so a resume of a resume accumulates rather than
    forgetting what the attempt before it managed.
    """
    base = (
        f"{gold_base_path}/_manifests/biweekly_feature_extraction/run_id={run_id}"
    )
    candidates = [
        f"{base}/window_feature_group_manifest",
        f"{base}/attempt=*/window_feature_group_manifest",
    ]

    frames = []
    for path in candidates:
        try:
            frames.append(spark.read.option("header", True).csv(path))  # noqa: F821
        except Exception as exc:  # no manifest at this location yet
            log(f"  resume: no manifest at {path} ({type(exc).__name__})")  # noqa: F821

    if not frames:
        log(  # noqa: F821
            f"resume: no manifest found under {base}; nothing will be skipped."
        )
        return set()

    prior = reduce(  # noqa: F821
        lambda left, right: left.unionByName(right, allowMissingColumns=True), frames
    )
    if scope is not None:
        prior = prior.filter(F.col("scope") == scope)

    rows = (
        prior.filter(F.col("status") == "success")
        .select("window_id", "feature_group")
        .distinct()
        .collect()
    )
    completed = {(r["window_id"], r["feature_group"]) for r in rows}
    log(  # noqa: F821
        f"resume: {len(completed)} (window, feature_group) pairs already succeeded "
        f"under run_id={run_id}."
    )
    return completed

def log_resume_plan(windows: Sequence[Dict[str, str]], scope: str) -> None:
    """Print, before any work starts, exactly what this attempt will redo."""
    expected = expected_feature_groups()
    total_todo = 0
    log("=" * 100)  # noqa: F821
    log(f"RESUME PLAN  ({len(expected)} feature groups per window)")  # noqa: F821
    for w in windows:
        wid = w["window_id"]
        todo = [g for g in expected if (wid, g) not in COMPLETED_GROUPS]
        total_todo += len(todo)
        scoped_path = _scratch_table_path(
            f"scoped_events/{SAMPLE_TAG}/"  # noqa: F821
            + ("tenure" if not fcfg.prune_event_window else f"pruned{fcfg.scan_days}d")  # noqa: F821
            + f"/{wid}"
        )
        if not todo:
            log(f"  {wid}: complete, skipping")  # noqa: F821
            continue
        joined = "reuse scoped_events" if _path_exists(scoped_path) else "REBUILD range join"
        log(  # noqa: F821
            f"  {wid}: {len(todo):2d}/{len(expected)} groups to write | {joined} | "
            + ", ".join(todo)
        )
    log(f"  TOTAL: {total_todo} feature-group writes")  # noqa: F821
    log("=" * 100)  # noqa: F821

def split_feature_output_path(
    scope: str,
    feature_group: str,
    partition_ts: str,
    gold_base_path: str = GOLD_FEATURE_BASE_PATH,
    version: str = OUTPUT_VERSION,
) -> str:
    return (
        f"{gold_base_path}/feature_set={split_feature_set_name(scope, feature_group)}"
        f"/version={version}/dt={fmt_partition_ts(partition_ts)}"
    )

def create_manifest_df(records: List[Dict[str, Any]]) -> DataFrame:
    schema = StructType([
        StructField("run_id", StringType(), True),
        StructField("window_id", StringType(), True),
        StructField("window_start_ts", StringType(), True),
        StructField("window_end_ts", StringType(), True),
        StructField("reference_ts", StringType(), True),
        StructField("partition_ts", StringType(), True),
        StructField("scope", StringType(), True),
        StructField("feature_group", StringType(), True),
        StructField("feature_set", StringType(), True),
        StructField("output_path", StringType(), True),
        StructField("status", StringType(), True),
        StructField("error_message", StringType(), True),
        StructField("reference_row_count", LongType(), True),
        StructField("snapshot_row_count", LongType(), True),
        StructField("changes_row_count", LongType(), True),
        StructField("enstream_row_count", LongType(), True),
        StructField("feature_row_count", LongType(), True),
        StructField("final_row_count", LongType(), True),
        StructField("duplicate_key_groups", LongType(), True),
        StructField("created_at_utc", StringType(), True),
    ])
    if not records:
        return spark.createDataFrame([], schema)
    normalized_records = []
    field_names = [f.name for f in schema.fields]
    for r in records:
        normalized_records.append({k: r.get(k) for k in field_names})
    return spark.createDataFrame(normalized_records, schema=schema)

def save_manifest(records: List[Dict[str, Any]]) -> None:
    manifest_df = create_manifest_df(records)
    manifest_path = f"{MANIFEST_BASE_PATH}/window_feature_group_manifest"
    log(f"Saving manifest to {manifest_path}")
    manifest_df.coalesce(1).write.mode("overwrite").option("header", True).csv(manifest_path)

    windows_df = spark.createDataFrame(
        [
            {
                "run_id": RUN_ID,
                "window_id": w["window_id"],
                "window_start_ts": w["window_start_ts"],
                "window_end_ts": w["window_end_ts"],
                "reference_ts": w["reference_ts"],
                "partition_ts": w["partition_ts"],
                "window_days": int(w["window_days"]),
            }
            for w in GENERATED_WINDOWS
        ]
    )
    windows_path = f"{MANIFEST_BASE_PATH}/generated_windows"
    log(f"Saving generated window definition to {windows_path}")
    windows_df.coalesce(1).write.mode("overwrite").option("header", True).csv(windows_path)

def write_feature_group_output_logged(
    feature_df: DataFrame,
    scope_snapshot_df: DataFrame,
    scope: str,
    feature_group: str,
    window: Dict[str, str],
    reference_row_count: Optional[int],
    snapshot_row_count: Optional[int],
    changes_row_count: Optional[int],
    enstream_row_count: Optional[int],
    manifest_records: List[Dict[str, Any]],
    cfg: "FeatureConfig" = None,  # noqa: RUF013
) -> None:
    """As before, but each output frame is computed once.
 
    The original called `safe_count(feature_df)`, then built `output_df`, then
    `safe_count(output_df)`, then `duplicate_key_count(output_df)`, then
    `printSchema`, then wrote -- five actions over the same unmaterialised
    plan, so the feature computation ran five times per group. Here the write
    happens first and the counts come off the written output, which is a cheap
    Parquet scan.
    """
    cfg = FeatureConfig() if cfg is None else cfg  # noqa: F821
    entity_cols = list(cfg.entity_cols_for(scope))  # noqa: F821
    feature_set = split_feature_set_name(scope, feature_group)  # noqa: F821
    output_path = split_feature_output_path(scope, feature_group, window["partition_ts"])  # noqa: F821
 
    base_record = {
        "run_id": RUN_ID,  # noqa: F821
        "window_id": window["window_id"],
        "window_start_ts": window["window_start_ts"],
        "window_end_ts": window["window_end_ts"],
        "reference_ts": window["reference_ts"],
        "partition_ts": window["partition_ts"],
        "scope": scope,
        "feature_group": feature_group,
        "feature_set": feature_set,
        "output_path": output_path,
        "reference_row_count": reference_row_count,
        "snapshot_row_count": snapshot_row_count,
        "changes_row_count": changes_row_count,
        "enstream_row_count": enstream_row_count,
        "created_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }
 
    try:
        log(f"Writing feature_set={feature_set} to {output_path}")  # noqa: F821
        output_df = dedupe_columns(  # noqa: F821
            join_feature_frames(scope_snapshot_df, [feature_df], entity_cols)  # noqa: F821
        )
 
        output_partition_cols = [c for c in entity_cols if c in output_df.columns]
        partitions = output_partitions_for(  # noqa: F821
            output_df, entity_cols, snapshot_row_count
        )
        log(  # noqa: F821
            f"{feature_set}: {len(output_df.columns)} columns x "
            f"{snapshot_row_count if snapshot_row_count is not None else 'unknown'} rows "
            f"-> {partitions} partitions (cap {OUTPUT_PARTITIONS})"  # noqa: F821
            + (" [AT CAP - consider raising it]" if partitions == OUTPUT_PARTITIONS else "")  # noqa: F821
        )
        writer_input = (
            output_df.repartition(partitions, *output_partition_cols)
            if output_partition_cols
            else output_df.repartition(partitions)
        )
        # Parquet rather than CSV: a wide feature frame written as CSV loses
        # types, cannot be column-pruned by readers, and is several times
        # larger. Switch OUTPUT_WRITE_FORMAT back to "csv" if a downstream
        # consumer requires it.
        writer_input.write.mode(OUTPUT_WRITE_MODE).format(OUTPUT_WRITE_FORMAT).save(output_path)  # noqa: F821
 
        # Counts from the written output, not from the plan.
        written = spark.read.format(OUTPUT_WRITE_FORMAT).load(output_path)  # noqa: F821
        final_count = safe_count(written, f"{window['window_id']} {scope}/{feature_group} output")  # noqa: F821
        key_cols = [c for c in entity_cols if c in written.columns]
        # dup_groups = (
        #     duplicate_key_count(  # noqa: F821
        #         written, key_cols, f"{window['window_id']} {scope}/{feature_group}"
        #     )
        #     if ENABLE_DUPLICATE_CHECKS  # noqa: F821
        #     else None
        # )
 
        status = "empty_written" if final_count == 0 else "success"
        if final_count == 0:
            log(f"WARNING: {feature_set} wrote zero rows. path={output_path}")  # noqa: F821
 
        manifest_records.append(
            {
                **base_record,
                "status": status,
                "error_message": None,
                "feature_row_count": final_count,
                "final_row_count": final_count,
            }
        )
 
    except Exception as exc:
        err = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        log(f"ERROR writing {feature_set}: {err}")  # noqa: F821
        manifest_records.append(
            {
                **base_record,
                "status": "failed",
                "error_message": err[:4000],
                "feature_row_count": None,
                "final_row_count": None,
            }
        )
        if FAIL_FAST:  # noqa: F821
            raise

def process_one_window(
    window: Dict[str, str],
    lineage_df: DataFrame,
    normalized_enstream_events_df: DataFrame,
    changes_df: DataFrame,
    manifest_records: List[Dict[str, Any]],
    state_transitions: Sequence[Tuple[str, str]] = None,  # noqa: RUF013
    cfg: "FeatureConfig" = None,  # noqa: RUF013
) -> None:
    state_transitions = (
        DEFAULT_STATE_TRANSITIONS if state_transitions is None else state_transitions  # noqa: F821
    )
    cfg = FeatureConfig() if cfg is None else cfg  # noqa: F821
 
    log("=" * 100)  # noqa: F821
    log(  # noqa: F821
        f"Processing {window['window_id']} | start={window['window_start_ts']} "
        f"| end={window['window_end_ts']} | reference_ts={window['reference_ts']} "
        f"| partition_dt={window['partition_ts']}"
    )
 
    reference_ts_dt = parse_ts(window["reference_ts"])  # noqa: F821
    SCOPE = "customer"

    # Window-level checkpoint: if every target feature group for this window
    # was already written under the current RUN_ID, skip the window entirely
    # -- reference_df/snapshot/scoped_events never get built. This is what
    # lets a re-run of the same run_id resume after a crash instead of
    # redoing every window from the top.
    # Derived from the same list the write loop filters on, so a group added
    # there cannot be left out of the resume check and let a window count as
    # done without it. Empty when every group runs, because the full set is not
    # enumerable here -- and an empty tuple must NOT read as "all present",
    # hence the explicit guard on the `all(...)` below.
    checkpoint_groups = expected_feature_groups()  # noqa: F821
    checkpoint_paths = {
        group: split_feature_output_path(SCOPE, group, window["partition_ts"])
        for group in checkpoint_groups
    }
    if (
        not FORCE_OVERWRITE
        and checkpoint_groups
        and all(
            (window["window_id"], group) in COMPLETED_GROUPS  # noqa: F821
            for group in checkpoint_groups
        )
    ):
        log(  # noqa: F821
            f"{window['window_id']}: all {len(checkpoint_paths)} target feature groups already "
            f"written under run_id={RUN_ID}; skipping (checkpoint)."  # noqa: F821
        )
        for group, path in checkpoint_paths.items():
            manifest_records.append(
                {
                    "run_id": RUN_ID,  # noqa: F821
                    "window_id": window["window_id"],
                    "window_start_ts": window["window_start_ts"],
                    "window_end_ts": window["window_end_ts"],
                    "reference_ts": window["reference_ts"],
                    "partition_ts": window["partition_ts"],
                    "scope": SCOPE,
                    "feature_group": group,
                    "feature_set": split_feature_set_name(SCOPE, group),  # noqa: F821
                    "output_path": path,
                    "status": "skipped_checkpoint",
                    "error_message": None,
                    "reference_row_count": None,
                    "snapshot_row_count": None,
                    "changes_row_count": None,
                    "enstream_row_count": None,
                    "feature_row_count": None,
                    "final_row_count": None,
                    "duplicate_key_groups": None,
                    "created_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),  # noqa: F821
                }
            )
        return

    reference_df = None
    scoped_events = None
    snapshot_scope: Dict[str, DataFrame] = {}
 
    try:
        # Keyword args: build_reference_df's second positional is
        # msisdn_reference_df, not reference_ts.
        # One row per customer holding a lineage interval that covers this
        # window's reference_ts. Customers whose lineage had closed before it,
        # or opens after it, are not carried into the window at all.
        log(f"Building reference dataframe (customers active at {window['reference_ts']}).")  # noqa: F821
        reference_df = build_reference_df(  # noqa: F821
            lineage_df,
            reference_ts=reference_ts_dt,
            cfg=cfg,
        ).persist(StorageLevel.MEMORY_AND_DISK)
        reference_row_count = safe_count(  # noqa: F821
            reference_df, f"{window['window_id']} active customers"
        )
 
        log("Building scope snapshots.")  # noqa: F821
        snapshot_scope = build_scope_snapshots(lineage_df, reference_df, cfg)  # noqa: F821
 
        # `base` feeds the stability and enstream-partner calculators; the
        # active scope feeds the range join and every output write. Both are
        # read many times per window, so both are persisted. The other two
        # scopes are never touched at SCOPE="customer" and are left lazy.
        snapshot_scope["base"] = snapshot_scope["base"].persist(StorageLevel.MEMORY_AND_DISK)
        snapshot_scope[SCOPE] = snapshot_scope[SCOPE].persist(StorageLevel.MEMORY_AND_DISK)
        snapshot_row_count = safe_count(  # noqa: F821
            snapshot_scope[SCOPE], f"{window['window_id']} {SCOPE} snapshot"
        )
 
        if DEBUG_SHOW:
            for key in ["base", *SCOPES]:  # noqa: F821
                frame = snapshot_scope[key]
                print(f"{key:10s} {frame.count():4d} rows   {frame.columns}")
            snapshot_scope["base"].show(truncate=False)
            snapshot_scope[SCOPE].show(truncate=False)
 
        # The range join, once per window. Every calculator reads this frame,
        # so it is persisted rather than recomputed per group. This is the
        # single join that `assemble_scope_features` used to redo on all ten
        # of its (identical) invocations.
        # The scratch name is keyed on the cohort tag as well as the window,
        # because scoped_events is a function of WHICH customers are in
        # changes_df. Without the tag, re-running the same run_id after
        # changing SAMPLE_FRACTION / SAMPLE_SALT / SAMPLE_PER_WINDOW would
        # read back the previous cohort's frame and every feature below it
        # would describe the wrong population, with nothing raised.
        # FORCE_OVERWRITE now bypasses this checkpoint too -- previously it
        # bypassed only the two output checkpoints, so a forced re-run still
        # reused stale scoped_events.
        # The scratch name encodes BOTH the cohort and the scan mode, because
        # scoped_events is a function of which customers are in changes_df and
        # of how far back the range join reached. SAMPLE_TAG alone covered only
        # the first, so flipping cfg.prune_event_window and re-running the same
        # run_id would have read back a frame built under the other scan mode:
        # a full-tenure feature set silently computed from a 97-day frame, or
        # the reverse. Nothing would have raised.
        scan_tag = (
            f"pruned{cfg.scan_days}d" if cfg.prune_event_window else "tenure"
        )
        scoped_events_name = (
            f"scoped_events/{SAMPLE_TAG}/{scan_tag}/{window['window_id']}"  # noqa: F821
        )
        scoped_events_path = _scratch_table_path(scoped_events_name)
        if (REUSE_SCOPED_EVENTS or not FORCE_OVERWRITE) and _path_exists(scoped_events_path):  # noqa: F821
            log(f"Reusing materialized scoped_events for {window['window_id']} from {scoped_events_path}.")  # noqa: F821
            scoped_events = spark.read.parquet(scoped_events_path)  # noqa: F821
        else:
            scoped_events = materialize(
                attach_events_to_scope(snapshot_scope, changes_df, SCOPE, cfg),
                scoped_events_name,
                RUN_ID,
                partition_by=[cfg.event_family_col],
            )
 
        changes_row_count = safe_count(scoped_events, f"{window['window_id']} scoped_events")  # noqa: F821
        enstream_row_count = None  # counted once outside the loop, not per window
 
        if DEBUG_SHOW:
            print("entities :", scoped_events.select(*cfg.entity_cols_for(SCOPE)).distinct().count())
            scoped_events.select(
                *cfg.entity_cols_for(SCOPE), cfg.event_ts_col, cfg.event_family_col
            ).orderBy(*cfg.entity_cols_for(SCOPE), cfg.event_ts_col).show(30, truncate=False)
 
        # One computation of every group, instead of ten computations of all
        # groups. GENERIC_EVENT_FAMILIES rather than None -- None was a
        # TypeError in the family loop.
        # --- TEMP: only recompute EnStream-related feature groups while
        # validating the api_timestamp fix (see preprocess_enstream_events).
        # state_transitions is passed through in full (all pairs, not just
        # the ones naming ENSTREAM_API_CALL): calculate_state_transition_features
        # shares one ordered window across every type named by any pair
        # (next_event_window in the events/expressions module), so a
        # narrower pair list changes only which output columns get built,
        # not the underlying per-partition ordering -- and this run does not
        # rely on that distinction. event_families stays restricted to just
        # EnStream since the other 8 families are genuinely unaffected by
        # this bug. To go back to a full run: pass
        # event_families=GENERIC_EVENT_FAMILIES below and delete the
        # feature_group_frames filter line.
        feature_group_frames = assemble_scope_feature_groups(
            snapshot_scope=snapshot_scope,
            scoped_events=scoped_events,
            normalized_enstream_events_df=normalized_enstream_events_df,
            scope=SCOPE,
            state_transitions=state_transitions,
            event_families=([ENSTREAM_FAMILY] if RUN_ENSTREAM_ONLY else GENERIC_EVENT_FAMILIES),  # noqa: F821
            cfg=cfg,
            changes_df=changes_df,  # feeds calculate_stability_features's occurred_in_tenure flags
        )
        if RUN_ENSTREAM_ONLY:  # noqa: F821
            feature_group_frames = {
                k: v for k, v in feature_group_frames.items()
                if k in ENSTREAM_ONLY_GROUPS  # noqa: F821
            }
 
        for feature_group, features_df in feature_group_frames.items():
            normalized_feature_group = (
                "enstream_api_call_generic"
                if feature_group == "enstream_api_call"
                else feature_group
            )
            group_output_path = split_feature_output_path(SCOPE, normalized_feature_group, window["partition_ts"])
            # Manifest, not path existence -- see load_completed_groups for
            # why a directory that exists is not evidence of a finished write.
            if (
                not FORCE_OVERWRITE  # noqa: F821
                and (window["window_id"], normalized_feature_group) in COMPLETED_GROUPS  # noqa: F821
            ):
                log(  # noqa: F821
                    f"{window['window_id']} {SCOPE}/{normalized_feature_group}: output already exists "
                    f"at {group_output_path}; skipping (checkpoint)."
                )
                manifest_records.append(
                    {
                        "run_id": RUN_ID,  # noqa: F821
                        "window_id": window["window_id"],
                        "window_start_ts": window["window_start_ts"],
                        "window_end_ts": window["window_end_ts"],
                        "reference_ts": window["reference_ts"],
                        "partition_ts": window["partition_ts"],
                        "scope": SCOPE,
                        "feature_group": normalized_feature_group,
                        "feature_set": split_feature_set_name(SCOPE, normalized_feature_group),  # noqa: F821
                        "output_path": group_output_path,
                        "status": "skipped_checkpoint",
                        "error_message": None,
                        "reference_row_count": reference_row_count,
                        "snapshot_row_count": snapshot_row_count,
                        "changes_row_count": changes_row_count,
                        "enstream_row_count": enstream_row_count,
                        "feature_row_count": None,
                        "final_row_count": None,
                        "duplicate_key_groups": None,
                        "created_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),  # noqa: F821
                    }
                )
                continue
            try:
                write_feature_group_output_logged(  # noqa: F821
                    feature_df=features_df,
                    scope_snapshot_df=snapshot_scope[SCOPE],
                    scope=SCOPE,
                    feature_group=normalized_feature_group,
                    window=window,
                    reference_row_count=reference_row_count,
                    snapshot_row_count=snapshot_row_count,
                    changes_row_count=changes_row_count,
                    enstream_row_count=enstream_row_count,
                    manifest_records=manifest_records,
                    cfg=cfg,
                )
            except Exception as exc:
                err = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                log(f"ERROR calculating {SCOPE}/{normalized_feature_group}: {err}")  # noqa: F821
                manifest_records.append(
                    {
                        "run_id": RUN_ID,  # noqa: F821
                        "window_id": window["window_id"],
                        "window_start_ts": window["window_start_ts"],
                        "window_end_ts": window["window_end_ts"],
                        "reference_ts": window["reference_ts"],
                        "partition_ts": window["partition_ts"],
                        "scope": SCOPE,
                        "feature_group": normalized_feature_group,
                        "feature_set": split_feature_set_name(SCOPE, normalized_feature_group),  # noqa: F821
                        "output_path": split_feature_output_path(  # noqa: F821
                            SCOPE, normalized_feature_group, window["partition_ts"]
                        ),
                        "status": "failed",
                        "error_message": err[:4000],
                        "reference_row_count": reference_row_count,
                        "snapshot_row_count": snapshot_row_count,
                        "changes_row_count": changes_row_count,
                        "enstream_row_count": enstream_row_count,
                        "feature_row_count": None,
                        "final_row_count": None,
                        "duplicate_key_groups": None,
                        "created_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
                if FAIL_FAST:  # noqa: F821
                    raise
 
    except Exception as exc:
        err = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        log(f"WINDOW FAILED: {window['window_id']}: {err}")  # noqa: F821
        manifest_records.append(
            {
                "run_id": RUN_ID,  # noqa: F821
                "window_id": window["window_id"],
                "window_start_ts": window["window_start_ts"],
                "window_end_ts": window["window_end_ts"],
                "reference_ts": window["reference_ts"],
                "partition_ts": window["partition_ts"],
                "scope": None,
                "feature_group": None,
                "feature_set": None,
                "output_path": None,
                "status": "window_failed",
                "error_message": err[:4000],
                "reference_row_count": None,
                "snapshot_row_count": None,
                "changes_row_count": None,
                "enstream_row_count": None,
                "feature_row_count": None,
                "final_row_count": None,
                "duplicate_key_groups": None,
                "created_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        if FAIL_FAST:  # noqa: F821
            raise
    finally:
        # Unpersist in reverse dependency order so nothing is dropped while a
        # frame that reads it is still materialised.
        for frame in (scoped_events, reference_df):
            if frame is not None:
                frame.unpersist()
        for key in ("base", SCOPE):
            frame = snapshot_scope.get(key)
            if frame is not None:
                try:
                    frame.unpersist()
                except Exception:
                    pass

# ---------------- notebook cell 27 (verbatim) ----------------

def _cohort_predicate(cfg, keep_buckets: int, salt: str):
    bucket = F.pmod(
        F.xxhash64(
            F.concat_ws("::", F.lit(salt), F.col(cfg.customer_col).cast("string"))
        ),
        F.lit(SAMPLE_BUCKETS),
    )
    return bucket < F.lit(keep_buckets)

def sample_customer_cohort(
    lineage_df: DataFrame,
    changes_df: DataFrame,
    normalized_enstream_events_df: Optional[DataFrame],
    cfg: "FeatureConfig",
    fraction: float = None,  # noqa: RUF013
    window_id: Optional[str] = None,
):
    """Filter lineage and events to a stable pseudo-random customer cohort.

    Returns new frames and never mutates its inputs. The caller must keep the
    unsampled frames under their own names -- rebinding the loop's source
    frames to the sampled result is how a per-window draw turns into a
    compounding sample (0.5, then 0.25, then 0.125 ...).
    """
    global _sample_verified

    fraction = SAMPLE_FRACTION if fraction is None else float(fraction)
    if not 0 < fraction <= 1:
        raise ValueError(f"SAMPLE_FRACTION must be in (0, 1], got {fraction}.")

    if fraction == 1.0:
        log("SAMPLE_FRACTION=1.0; keeping every customer.")
        return lineage_df, changes_df, normalized_enstream_events_df

    keep_buckets = max(1, round(SAMPLE_BUCKETS * fraction))
    salt = SAMPLE_SALT if window_id is None else f"{SAMPLE_SALT}::{window_id}"
    cond = _cohort_predicate(cfg, keep_buckets, salt)

    log(
        f"Cohort sample: keeping {keep_buckets}/{SAMPLE_BUCKETS} buckets "
        f"(~{fraction:.1%} of distinct {cfg.customer_col}) with salt={salt!r}."
    )

    lineage_sampled = lineage_df.filter(cond)
    changes_sampled = changes_df.filter(cond)

    enstream_sampled = normalized_enstream_events_df
    if (
        normalized_enstream_events_df is not None
        and cfg.customer_col in normalized_enstream_events_df.columns
    ):
        enstream_sampled = normalized_enstream_events_df.filter(cond)
    elif normalized_enstream_events_df is not None:
        log(
            f"WARNING: normalized_enstream_events_df has no {cfg.customer_col} "
            "column; it is NOT being sampled, so its events will not line up "
            "with the sampled lineage."
        )

    # One verification, not one per window: two distinct counts over lineage.
    if SAMPLE_VERIFY and not _sample_verified:
        _sample_verified = True
        total = lineage_df.select(cfg.customer_col).distinct().count()
        kept = lineage_sampled.select(cfg.customer_col).distinct().count()
        share = (kept / total) if total else 0.0
        log(
            f"Cohort check: {kept:,} of {total:,} distinct {cfg.customer_col} "
            f"kept ({share:.3%}; target {fraction:.1%})."
        )

    return lineage_sampled, changes_sampled, enstream_sampled

# cell 08 (module level): the window list every loop below walks.
GENERATED_WINDOWS = build_biweekly_windows(
    start_ts=WINDOW_START_TS,
    end_exclusive_ts=WINDOW_END_EXCLUSIVE_TS,
    window_days=WINDOW_SIZE_DAYS,
)


# ==========================================================================
# SAGEMAKER ADAPTATION: the Spark session and the inputs (cells 01 and 06)
# ==========================================================================

# cell 01's settings that do not depend on the cluster's size. Executor count,
# cores and memory are left to SageMaker, which derives them from the instance
# type and count of the processing job; hard-coding the notebook's 18 x 18g
# would be wrong on any other cluster. Terraform's `spark_conf` is applied last
# and wins (the Iceberg/Glue catalog lives there).
NOTEBOOK_SPARK_CONF = {
    "spark.sql.adaptive.advisoryPartitionSizeInBytes": "128MB",
    "spark.sql.shuffle.partitions": "2000",
    "spark.shuffle.file.buffer": "1m",
    "spark.reducer.maxSizeInFlight": "96m",
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled": "true",
    "spark.sql.adaptive.skewJoin.skewedPartitionFactor": "3",
    "spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes": "64MB",
    "spark.memory.fraction": "0.8",
    "spark.memory.storageFraction": "0.2",
    "spark.shuffle.compress": "true",
    "spark.shuffle.spill.compress": "true",
    "spark.io.compression.codec": "snappy",
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
    "spark.sql.broadcastTimeout": "600",
    "spark.driver.maxResultSize": "4g",
    # The package's convention (conf/lineage/base.yaml, conf/ml/base.yaml, the
    # features test suite): Gold timestamps are wall-clock EST.
    "spark.sql.session.timeZone": "EST",
}


def build_spark() -> SparkSession:
    builder = SparkSession.builder.appName(f"ts05_feature_engineering_{ARGS.mode}")
    for key, value in {**NOTEBOOK_SPARK_CONF, **(PARAMS.get("spark_conf") or {})}.items():
        builder = builder.config(key, str(value))
    return builder.getOrCreate()


def read_source(name: str) -> DataFrame:
    """A table by catalog name, or a parquet dataset by ``s3://`` path."""
    if str(name).startswith(("s3://", "s3a://")):
        return spark.read.parquet(name)
    return spark.table(name)


def read_inputs():
    """cell 06, with the table names from the parameters instead of literals."""
    inputs = PARAMS["inputs"]
    lineage_df = read_source(inputs["lineage_table"]).withColumnRenamed(
        "phone_number_AC_hash", "msisdn"
    )
    account_changes_df = read_source(inputs["account_changes_table"])
    enstream_df = read_source(inputs["enstream_table"]).filter(
        F.col("api_type") != F.lit(inputs.get("enstream_excluded_api_type", "internal"))
    )
    # The notebook's own derivation of hash_mapping (cell 06, the block that
    # built trust_score_v1_poc_silver.hash_mapping from device_lookup_batch).
    hash_mapping_df = (
        read_source(inputs["hash_mapping_table"])
        .select("phone_number_AC_hash", "phone_number_AT_hash")
        .drop_duplicates()
    )
    return lineage_df, account_changes_df, enstream_df, hash_mapping_df


# ==========================================================================
# cell 28: the run - verbatim logic, wrapped in a function
# ==========================================================================

def run_extraction(lineage_df, account_changes_df, enstream_df, hash_mapping_df) -> List[Dict[str, Any]]:
    global COMPLETED_GROUPS
    manifest_records: List[Dict[str, Any]] = []
    log(f"Attempt {ATTEMPT_ID}; manifest will be written to {MANIFEST_BASE_PATH}")

    COMPLETED_GROUPS = load_completed_groups(RUN_ID, scope="customer") if RESUME_RUN_ID else set()

    _resume_table_names = ("closed_lineage", "changes", "normalized_enstream")
    if RESUME_RUN_ID and all(_path_exists(_scratch_table_path(name)) for name in _resume_table_names):
        log(f"Reusing materialized tables from run_id={RESUME_RUN_ID}; skipping build_and_materialize_changes.")
        lineage_closed = spark.read.parquet(_scratch_table_path("closed_lineage"))
        changes_df = spark.read.parquet(_scratch_table_path("changes"))
        normalized_enstream_events_df = spark.read.parquet(_scratch_table_path("normalized_enstream"))
    else:
        lineage_closed, changes_df, normalized_enstream_events_df = build_and_materialize_changes(
            lineage_df=lineage_df,
            account_changes_df=account_changes_df,
            enstream_df=enstream_df,
            hash_mapping_df=hash_mapping_df,
            run_id=RUN_ID,
            cfg=fcfg,
        )

    log(f"canonical event rows: {changes_df.count():,}")

    lineage_full = lineage_closed
    changes_full = changes_df
    enstream_full = normalized_enstream_events_df

    if SAMPLE_ENABLED and not SAMPLE_PER_WINDOW:
        lineage_cohort, changes_cohort, enstream_cohort = sample_customer_cohort(
            lineage_df=lineage_full,
            changes_df=changes_full,
            normalized_enstream_events_df=enstream_full,
            cfg=fcfg,
        )
    else:
        lineage_cohort, changes_cohort, enstream_cohort = lineage_full, changes_full, enstream_full

    log_resume_plan(GENERATED_WINDOWS, "customer")

    _windows_processed = 0
    try:
        for window in GENERATED_WINDOWS:
            if not FORCE_OVERWRITE and all(
                (window["window_id"], group) in COMPLETED_GROUPS
                for group in expected_feature_groups()
            ):
                log(f"{window['window_id']}: every feature group already succeeded; skipping.")
                continue

            if MAX_WINDOWS_PER_RUN is not None and _windows_processed >= MAX_WINDOWS_PER_RUN:
                log(
                    f"MAX_WINDOWS_PER_RUN={MAX_WINDOWS_PER_RUN} reached with windows still "
                    "outstanding; stopping cleanly. Re-run this cell to continue."
                )
                break
            _windows_processed += 1

            if SAMPLE_ENABLED and SAMPLE_PER_WINDOW:
                w_lineage, w_changes, w_enstream = sample_customer_cohort(
                    lineage_df=lineage_full,
                    changes_df=changes_full,
                    normalized_enstream_events_df=enstream_full,
                    cfg=fcfg,
                    window_id=window["window_id"],
                )
            else:
                w_lineage, w_changes, w_enstream = lineage_cohort, changes_cohort, enstream_cohort

            process_one_window(
                window=window,
                lineage_df=w_lineage,
                normalized_enstream_events_df=w_enstream,
                changes_df=w_changes,
                manifest_records=manifest_records,
                state_transitions=DEFAULT_STATE_TRANSITIONS,
                cfg=fcfg,
            )
    finally:
        save_manifest(manifest_records)

    log(f"Attempt {ATTEMPT_ID} finished: {_windows_processed} window(s) processed. "
        f"Manifest: {MANIFEST_BASE_PATH}")
    return manifest_records


# ==========================================================================
# SAGEMAKER ADAPTATION: the layout trust_score_05.ml reads
# ==========================================================================
# The notebook stops at one table per (window, feature group). The ML stages
# read one wide table per window:
#     <features_root>/ml/non-fraud/time_window=<YYYY-MM-DD_HH-MM-SS>/data/
#     <features_root>/ml/fraud/<fraud_source>/dt=<YYYY-MM-DD>/data/
# The wide table is the groups joined with the package's `join_feature_frames`
# on the scope's entity columns - the same left join assemble_scope_features
# uses to build "the full feature row for one scope" - so the ML stages see
# exactly the feature row the package defines, split and rejoined.

ML_ROOT = f"{FEATURES_ROOT}/ml"
_GROUP_OK = {"success", "empty_written", "skipped_checkpoint"}


def time_window_label(partition_ts: str) -> str:
    """``2024-08-01 00:00:00`` -> ``2024-08-01_00-00-00`` (the ML package's spelling)."""
    return str(partition_ts).replace(" ", "_").replace(":", "-")


def window_groups_complete(window: Dict[str, str], manifest_records) -> List[str]:
    """The expected groups of ``window`` that did NOT finish, per the manifest."""
    done = {
        r["feature_group"] for r in manifest_records
        if r.get("window_id") == window["window_id"] and r.get("status") in _GROUP_OK
    }
    done |= {g for (w, g) in COMPLETED_GROUPS if w == window["window_id"]}
    return [g for g in expected_feature_groups() if g not in done]


def assemble_wide(window: Dict[str, str], scope: str = "customer") -> DataFrame:
    entity_cols = list(fcfg.entity_cols_for(scope))
    frames = [
        spark.read.parquet(split_feature_output_path(scope, group, window["partition_ts"]))
        for group in expected_feature_groups()
    ]
    return dedupe_columns(join_feature_frames(frames[0], frames[1:], entity_cols))


def write_wide(df: DataFrame, path: str, scope: str = "customer") -> int:
    entity_cols = list(fcfg.entity_cols_for(scope))
    rows = df.count()
    partitions = output_partitions_for(df, entity_cols, rows)
    df.repartition(partitions).write.mode("overwrite").parquet(path)
    log(f"wrote {rows:,} rows x {len(df.columns)} columns -> {path}")
    return rows


def read_fraud_labels() -> Optional[DataFrame]:
    """The labelled fraud events, in the columns the ML package's fraud reader expects.

    ``customer_id``, ``fraud_type``, ``fraud_timestamp`` (a ``yyyy-MM-dd HH:mm:ss``
    string, which ``trust_score_05.ml.datasets.load_fraud_frame`` parses), and
    ``fraud_source``, the feed name ``trust_score_05.ml.taxonomy`` maps types
    with. Column names in the source table come from the parameters.
    """
    spec = PARAMS.get("fraud_labels") or {}
    if not spec.get("table"):
        log("No fraud_labels.table configured; the fraud population is not written.")
        return None
    labels = read_source(spec["table"])
    return labels.select(
        F.col(spec.get("customer_col", "customer_id")).cast("string").alias(fcfg.customer_col),
        F.col(spec.get("fraud_type_col", "fraud_type")).cast("string").alias("fraud_type"),
        F.date_format(
            F.col(spec.get("fraud_timestamp_col", "fraud_timestamp")).cast("timestamp"),
            "yyyy-MM-dd HH:mm:ss",
        ).alias("fraud_timestamp"),
    ).withColumn("fraud_source", F.lit(spec.get("fraud_source", "Phase1-v1.0")))


def write_fraud_rows(window: Dict[str, str], wide: DataFrame, labels: DataFrame) -> int:
    """Labelled customers whose fraud event falls inside ``window``, with that
    window's feature row.

    The row was computed as of the window's reference time (its start), which
    precedes every event matched here - the features never see the fraud they
    are labelled with.
    """
    start, end = window["window_start_ts"], window["window_end_ts"]
    in_window = labels.filter(
        (F.col("fraud_timestamp") >= F.lit(start)) & (F.col("fraud_timestamp") < F.lit(end))
    )
    rows = wide.join(in_window, on=fcfg.customer_col, how="inner")
    source = (PARAMS.get("fraud_labels") or {}).get("fraud_source", "Phase1-v1.0")
    path = f"{ML_ROOT}/fraud/{source}/dt={start[:10]}/data"
    return write_wide(rows, path)


def publish(manifest_records) -> Tuple[List[Dict[str, Any]], List[str]]:
    written, failed = [], []
    labels = read_fraud_labels() if ARGS.mode == "training" else None
    for window in GENERATED_WINDOWS:
        missing = window_groups_complete(window, manifest_records)
        if missing:
            failed.append(window["window_id"])
            log(f"{window['window_id']}: NOT published - feature groups not finished: {missing}")
            continue
        wide = assemble_wide(window)
        if ARGS.mode == "scoring":
            path = f"{FEATURES_ROOT}/scoring/batch_date={ARGS.batch_date}/data"
            written.append({"window_id": window["window_id"], "path": path,
                            "rows": write_wide(wide, path)})
            continue
        path = f"{ML_ROOT}/non-fraud/time_window={time_window_label(window['partition_ts'])}/data"
        record = {"window_id": window["window_id"], "time_window": time_window_label(window["partition_ts"]),
                  "path": path, "rows": write_wide(wide, path)}
        if labels is not None:
            record["fraud_rows"] = write_fraud_rows(window, spark.read.parquet(path), labels)
        written.append(record)
    return written, failed


def main() -> int:
    global spark
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    validate_windows(GENERATED_WINDOWS)
    log(f"mode={ARGS.mode} run_id={RUN_ID} windows={len(GENERATED_WINDOWS)} features_root={FEATURES_ROOT}")

    manifest_records = run_extraction(*read_inputs())
    written, failed = publish(manifest_records)

    summary = {
        "mode": ARGS.mode, "run_id": RUN_ID, "batch_date": ARGS.batch_date,
        "windows": len(GENERATED_WINDOWS), "published": written, "not_published": failed,
        "ml_root": f"{ML_ROOT}/", "manifest": MANIFEST_BASE_PATH,
    }
    out_dir = os.environ.get("TS05_PROCESSING_OUTPUT", "/opt/ml/processing/output")
    if os.path.isdir(out_dir):
        with open(os.path.join(out_dir, "feature_engineering_summary.json"), "w", encoding="utf-8") as h:
            json.dump(summary, h, indent=2, default=str)
    log("summary: " + json.dumps(summary, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
