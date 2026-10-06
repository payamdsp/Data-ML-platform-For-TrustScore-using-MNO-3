"""End-to-end smoke test of the SageMaker entry points, on this machine, no AWS.

    local\\.venv\\Scripts\\python.exe sagemaker\\tests\\smoke_local.py

What it runs, each stage in its own process as it would run in its own job:

    ml-discovery        package job, real Spark, against a local-disk copy
    (ml-selection)      its output artifact is written directly - see below
    train.py sweep      slice 1 of 2 (arms 0-3)      SageMaker Training, spot
    train.py sweep      slice 2 of 2 (arms 4-7)      SageMaker Training, spot
    train.py finalize   merge, champion, drift reference, promote
    inference.py        a normal night    -> scores, queue, no retrain
    inference.py        a drifted night   -> retrain recommended
    inference.py        a broken batch    -> exit 2

**Storage is a mock S3** (``moto`` server on localhost, reached through
``AWS_ENDPOINT_URL``), with the same ``s3://`` URIs the sandbox uses, so the
package's real S3 code paths (listing, downloads, uploads, markers) are the ones
exercised. A local directory would also work in principle, but the model
layout nests deeper than Windows' 260-character path limit.

**Spark's reads are swapped for pandas** (``pandas_spark_standin.py``) for the
training stages, because open-source Spark cannot read files on Windows without
the Hadoop native binaries. Everything after the read is unmodified package
code. On Linux (the SageMaker image) the real Spark readers run.

**Selection** ranks features with Spark aggregations and is an EMR Serverless
stage; its output contract (``selected_features.json``) is written directly.

Needs: the local/.venv environment (pyspark, pyod, moto) and Java 17 for the
discovery stage. The data is synthetic; what is tested is the plumbing between
stages, the fan-out merge, champion selection, scoring and drift detection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
SAGEMAKER_DIR = HERE.parent
PYTHON = sys.executable
sys.path.insert(0, str(SAGEMAKER_DIR))

BUCKET = "ts05-smoke-sandbox"
REGION = "ca-central-1"
FEATURES = {
    # name: (kind, fraud multiplier)
    "cust_sim_change_cnt_30d": ("count", 4.0),
    "cust_device_change_cnt_30d": ("count", 3.0),
    "cust_enstream_api_call_cnt_7d": ("count", 6.0),
    "cust_mno_port_cnt_90d": ("count", 2.0),
    "cust_account_activation_cnt_365d": ("count", 1.0),
    "cust_status_change_c_cnt_7d": ("count", 1.0),
    "cust_phone_number_change_cnt_30d": ("count", 2.5),
    "cust_use_case_distinct_cnt": ("count", 2.0),
    "cust_sim_change_days_since_last": ("recency", 0.1),
    "cust_device_change_days_since_last": ("recency", 0.1),
    "cust_days_since_first_seen": ("recency", 0.15),
    "cust_account_cancellation_interval_mean": ("interval", 1.0),
}
TRAINING_MONTHS = ["2025-01-01_00-00-00", "2025-02-01_00-00-00"]
TESTING_WINDOW = "2025-03-01_00-00-00"
FRAUD_SOURCE = "Phase1-v1.0"
MODELS = "ecod,isolation_forest"
RUN_ID = "20260929T010000Z"
SELECTION_RUN_ID = "20260929T000000Z"


# --------------------------------------------------------------------------
# synthetic data
# --------------------------------------------------------------------------

def make_population(n: int, seed: int, prefix: str, fraud: bool = False,
                    shift: float = 1.0) -> pd.DataFrame:
    """``n`` customers; ``fraud`` makes events commoner and more recent, and
    ``shift`` moves the whole population to fake drift since training."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"customer_id": [f"{prefix}{i:06d}" for i in range(n)]})
    for name, (kind, multiplier) in FEATURES.items():
        factor = multiplier if fraud else 1.0
        if kind == "count":
            values = rng.poisson(lam=0.6 * factor * shift, size=n).astype(float)
        elif kind == "recency":
            values = rng.gamma(2.0, 400.0, size=n) * factor / shift
            values[rng.random(n) < 0.2] = np.nan  # the event never happened
        else:
            values = rng.normal(180.0, 60.0, size=n) * factor
            values[rng.random(n) < 0.5] = np.nan
        frame[name] = values
    return frame


def write_parquet(frame: pd.DataFrame, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / "part-00000.parquet", index=False)


def build_local_layout(root: Path) -> None:
    """The sandbox feature layout, on disk first (discovery reads it here)."""
    training = root / "ml" / "features" / "training"
    testing = root / "ml" / "features" / "testing"
    for i, month in enumerate(TRAINING_MONTHS):
        write_parquet(make_population(4000, 100 + i, f"t{i}_"),
                      training / f"time_window={month}" / "data")
    write_parquet(make_population(3000, 200, "e_"),
                  testing / "non-fraud" / f"time_window={TESTING_WINDOW}" / "data")
    fraud = make_population(80, 300, "f_", fraud=True)
    fraud["fraud_type"] = "FA"
    fraud["fraud_timestamp"] = "2025-03-15 00:00:00"
    fraud["fraud_source"] = FRAUD_SOURCE
    write_parquet(fraud, testing / "fraud" / FRAUD_SOURCE / "dt=2025-03-15" / "data")

    batches = root / "batches"
    normal = pd.concat([make_population(5000, 400, "n_"),
                        make_population(20, 401, "nf_", fraud=True)], ignore_index=True)
    write_parquet(normal, batches / "batch_date=2026-09-28")
    write_parquet(make_population(5000, 500, "d_", shift=3.0), batches / "batch_date=2026-09-29")
    broken = normal.drop(columns=["cust_sim_change_cnt_30d"])
    write_parquet(broken, batches / "batch_date=2026-09-30")


def upload_tree(root: Path) -> None:
    import boto3

    client = boto3.client("s3")
    for path in root.rglob("*.parquet"):
        client.upload_file(str(path), BUCKET, path.relative_to(root).as_posix())


def overlay(root_uri: str, name: str, workdir: Path) -> Path:
    doc = {
        "run": {"mode": "pilot", "models_version": "smoke", "run_id": None},
        "data": {
            "training_root": f"{root_uri}/ml/features/training/",
            "testing_root": f"{root_uri}/ml/features/testing/",
            "feature_selection_root": f"{root_uri}/ml/feature_selection/",
            "models_root": f"{root_uri}/ml/models/",
            "training_months": TRAINING_MONTHS,
            "testing_windows": [TESTING_WINDOW],
            "fraud_sources": [FRAUD_SOURCE],
        },
        "sampling": {"train_rows_per_month_pilot": 3000, "profile_rows_pilot": 6000,
                     "max_plot_rows": 5000},
        "sweep": {
            "top_n_feature_counts": [6, 10],
            "preprocessing_ids": ["median_impute_standard_scaler", "domain_impute_robust_scaler"],
            "max_hyperparam_configs": 2,
        },
        "best_config": {"level": "customer", "metric": "recall_at_k", "k_percent": 5.0},
        "behaviour": {"write_scores_csv": False},
    }
    path = workdir / name
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def run(label: str, args: list, env: dict, expect: int = 0) -> None:
    print(f"\n=== {label}", flush=True)
    started = time.time()
    proc = subprocess.run(args, env=env, cwd=str(SAGEMAKER_DIR), capture_output=True, text=True)
    lines = [ln for ln in (proc.stdout + proc.stderr).splitlines()
             if ln.strip() and "could not be terminated" not in ln
             and "operation attempted is not supported" not in ln
             and "no running instance" not in ln]
    if proc.returncode != expect:
        print("\n".join(lines[-60:]))
        raise SystemExit(f"FAIL: {label} exited {proc.returncode}, expected {expect}")
    for line in lines[-3:]:
        print("    | " + line[:220])
    print(f"    exit {proc.returncode} as expected ({time.time() - started:.0f}s)", flush=True)


def package_job(module: str, argv: list) -> list:
    code = ("import sys; sys.path.insert(0, r'%s'); "
            "from trust_score_05.ml.jobs import %s as job; "
            "sys.exit(job.main_cli(%r))" % (SAGEMAKER_DIR, module, argv))
    return [PYTHON, "-c", code]


def train_job(argv: list) -> list:
    """``train.py`` with Spark's reads swapped for pandas (see pandas_spark_standin)."""
    code = ("import sys; sys.path[:0] = [r'%s', r'%s']; "
            "import pandas_spark_standin; pandas_spark_standin.install(); "
            "import train; sys.exit(train.main(%r))" % (SAGEMAKER_DIR, HERE, argv))
    return [PYTHON, "-c", code]


def write_selection(config: Path, sample: pd.DataFrame) -> None:
    """The artifact ``selection_job`` writes, in the format ``sweep_job`` reads."""
    from trust_score_05.common.io.s3 import join_uri, write_json
    from trust_score_05.ml.config import load_ml_config
    from trust_score_05.ml.jobs.base import SELECTED_FEATURES_FILE
    from trust_score_05.ml.paths import feature_selection_prefix
    from trust_score_05.ml.selection import METHOD_NAME

    cfg = load_ml_config(config_paths=[str(SAGEMAKER_DIR / "conf/ml/base.yaml"), str(config)],
                         overrides=[f"run.run_id={SELECTION_RUN_ID}"])
    ranked = sample[list(FEATURES)].var().sort_values(ascending=False).index.tolist()
    for model in MODELS.split(","):
        prefix = feature_selection_prefix(cfg, model, METHOD_NAME)
        write_json(join_uri(prefix, SELECTED_FEATURES_FILE), {
            "model": model, "method": METHOD_NAME, "run_id": cfg.run_id,
            "models_version": cfg.models_version,
            "selections": {str(n): ranked[:n] for n in cfg.top_n_feature_counts},
        })
    print(f"\n=== selection artifact written for {MODELS}, counts {list(cfg.top_n_feature_counts)}")


def check(condition: bool, message: str) -> None:
    print(("  PASS  " if condition else "  FAIL  ") + message, flush=True)
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true", help="Keep the local working directory.")
    opts = parser.parse_args()

    import logging

    from moto.server import ThreadedMotoServer

    logging.getLogger("werkzeug").setLevel(logging.ERROR)  # one line per mock S3 request
    port = free_port()
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=port)
    server.start()
    os.environ.update({
        "AWS_ENDPOINT_URL": f"http://127.0.0.1:{port}",
        "AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_DEFAULT_REGION": REGION, "AWS_REGION": REGION,
    })
    os.environ.pop("AWS_PROFILE", None)
    os.environ.pop("TS05_RUN_ID", None)
    workdir = Path(tempfile.mkdtemp(prefix="ts05_"))
    try:
        import boto3

        boto3.client("s3").create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": REGION})
        print(f"mock S3 at {os.environ['AWS_ENDPOINT_URL']}, bucket s3://{BUCKET}")

        env = dict(os.environ)
        env.update({"PYSPARK_PYTHON": PYTHON, "PYSPARK_DRIVER_PYTHON": PYTHON,
                    "SPARK_LOCAL_IP": "127.0.0.1", "MPLBACKEND": "Agg",
                    "TS05_PROCESSING_OUTPUT": str(workdir / "processing_output")})
        (workdir / "processing_output").mkdir()

        data = workdir / "data"
        build_local_layout(data)
        upload_tree(data)
        s3_cfg = overlay(f"s3://{BUCKET}", "smoke_s3.yaml", workdir)
        disk_cfg = overlay(data.as_posix(), "smoke_disk.yaml", workdir)
        base = str(SAGEMAKER_DIR / "conf/ml/base.yaml")
        s3_configs = ["--config", base, "--config", str(s3_cfg)]

        # -- EMR Serverless stages ------------------------------------------
        run("ml-discovery (package job, real local Spark, disk copy)",
            package_job("discovery_job", ["--config", base, "--config", str(disk_cfg),
                                          "--allow-failed-checks"]), env)
        write_selection(s3_cfg, pd.read_parquet(
            data / "ml/features/training" / f"time_window={TRAINING_MONTHS[0]}" / "data"))

        # -- SageMaker Training: 2 slices of one run id, then finalize ------
        train = ["--run-id", RUN_ID, "--models", MODELS] + s3_configs
        run("train.py --mode sweep, slice 1/2 (arms 0-3)",
            train_job(train + ["--mode", "sweep", "--arm-start", "0", "--arm-count", "4"]), env)
        run("train.py --mode sweep, slice 2/2 (arms 4-7)",
            train_job(train + ["--mode", "sweep", "--arm-start", "4", "--arm-count", "4"]), env)
        run("train.py --mode finalize", train_job(train + ["--mode", "finalize"]), env)

        from sm_common import read_csv_any, read_parquet_any
        from trust_score_05.common.io.s3 import read_json, s3_exists

        version = f"s3://{BUCKET}/ml/models/version_id=smoke"
        summary = f"{version}/sweep-summary/run_id={RUN_ID}"
        package_arms = read_csv_any(f"{summary}/arms.csv")
        merged_arms = read_csv_any(f"{summary}/merged/arms.csv")
        board = read_csv_any(f"{summary}/merged/config_leaderboard.csv")
        champion = read_json(f"{version}/champion/current.json")

        print("\nchecks after training")
        check(len(package_arms) == 4 and len(merged_arms) == 8,
              f"the package's own summary holds only the last slice ({len(package_arms)} arms); "
              f"finalize merged all {len(merged_arms)}")
        check(set(merged_arms["status"]) == {"READY"}, "all 8 arms of both slices are READY")
        check(board["model"].nunique() == 2,
              f"merged leaderboard ranks {len(board)} configurations across both models")
        check(s3_exists(champion["bundle_uri"]),
              f"champion = {champion['model']} / top {champion['top_n_features']} / "
              f"{champion['preprocessing']} ({champion['criteria']['metric']}="
              f"{champion['value']:.3f})")
        check(champion["n_reference_scores"] == 6000,
              f"drift reference: {champion['n_reference_scores']} training scores")
        check(s3_exists(f"{version}/champion/history/run_id={RUN_ID}/_READY.json"),
              "champion history written with READY marker")

        # -- SageMaker Processing: nightly scoring --------------------------
        infer = [PYTHON, str(SAGEMAKER_DIR / "inference.py")] + s3_configs + ["--top-k", "50"]
        run("inference.py, normal night 2026-09-28",
            infer + ["--input", f"s3://{BUCKET}/batches/batch_date=2026-09-28/",
                     "--batch-date", "2026-09-28"], env)
        run("inference.py, drifted night 2026-09-29",
            infer + ["--input", f"s3://{BUCKET}/batches/batch_date=2026-09-29/",
                     "--batch-date", "2026-09-29"], env)
        run("inference.py, batch missing a feature (must exit 2)",
            infer + ["--input", f"s3://{BUCKET}/batches/batch_date=2026-09-30/",
                     "--batch-date", "2026-09-30"], env, expect=2)

        normal = f"{version}/scoring/batch_date=2026-09-28"
        drifted = f"{version}/scoring/batch_date=2026-09-29"
        normal_report = read_json(f"{normal}/drift_report.json")
        drifted_report = read_json(f"{drifted}/drift_report.json")
        queue = read_csv_any(f"{normal}/review_queue.csv")
        scores = read_parquet_any(f"{normal}/scores.parquet")

        print("\nchecks after scoring")
        check(len(scores) == 5020, f"every batch row scored ({len(scores)})")
        check(len(queue) == 50 and queue["customer_id"].is_unique,
              "review queue: 50 unique customers")
        planted = int(queue["customer_id"].str.startswith("nf_").sum())
        check(planted >= 10, f"{planted} of the 20 planted fraud-like customers are in the top 50")
        check(not normal_report["retrain_recommended"],
              f"normal night: no retrain (score PSI {normal_report['score']['psi']:.3f}, "
              f"{normal_report['features']['n_drifted']} features drifted)")
        check(drifted_report["retrain_recommended"],
              "drifted night: retrain recommended - " + "; ".join(drifted_report["reasons"]))
        check(s3_exists(f"{drifted}/_RETRAIN_RECOMMENDED.json")
              and not s3_exists(f"{normal}/_RETRAIN_RECOMMENDED.json"),
              "retrain marker on the drifted night only")
        check(s3_exists(f"{version}/champion/retrain_requests/batch_date=2026-09-29/request.json"),
              "retrain request published under champion/retrain_requests/")
        check(not s3_exists(f"{version}/scoring/batch_date=2026-09-30/_READY.json"),
              "broken batch left no READY marker")

        print("\nSMOKE TEST PASSED")
        return 0
    finally:
        server.stop()
        if not opts.keep:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"(local working directory kept: {workdir})")


if __name__ == "__main__":
    raise SystemExit(main())
