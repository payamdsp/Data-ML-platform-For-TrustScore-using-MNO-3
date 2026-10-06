"""Test-only: stand in for the package's Spark *readers* with pandas equivalents.

Why this exists: open-source Spark cannot list or read local files on Windows
without the Hadoop native binaries (winutils.exe / hadoop.dll), which are not
installed on the development machine. SageMaker and EMR Serverless run Linux, so
this is a laptop limitation only.

What is replaced - only the functions that *read data through Spark*:

* ``trust_score_05.ml.datasets.load_training_matrix``
* ``trust_score_05.ml.datasets.load_fraud_frame``
* ``trust_score_05.ml.datasets.load_scenario_frame``
* ``trust_score_05.ml.datasets.bounded_to_pandas`` (pandas input passes through)
* the SparkSession builders (a no-op session object is returned)

Each stand-in reproduces what the real function returns - the same columns, the
same fraud-customer removal before sampling, the same per-month sampling with
the run's seed, the same ``label`` convention and the same training manifest -
with pandas instead of Spark. Everything downstream of the read (arm slicing,
preprocessing, fitting, evaluation, markers, the sweep summary, finalize,
drift, scoring) is the real, unmodified code. Never imported by the entry
points themselves.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


class _NoSpark:
    """Stands in for a SparkSession: the jobs only log from it and stop it."""

    version = "standin"

    class sparkContext:  # noqa: N801 - mirrors the attribute name
        appName = "pandas-standin"

        @staticmethod
        def setLogLevel(level: str) -> None:  # noqa: N802
            return None

    class conf:  # noqa: N801
        @staticmethod
        def set(key: str, value: str) -> None:
            return None

    def stop(self) -> None:
        return None


def _read(path: str) -> pd.DataFrame:
    from sm_common import read_parquet_any

    return read_parquet_any(path)


def _project(frame: pd.DataFrame, features: Sequence[str], meta: Sequence[str]) -> pd.DataFrame:
    """``add_missing_and_select``: meta then features, absent ones as all-null."""
    out = pd.DataFrame(index=frame.index)
    for name in list(meta) + list(features):
        out[name] = frame[name] if name in frame.columns else np.nan
    return out.reset_index(drop=True)


def install() -> None:
    import sm_common
    from trust_score_05.common.io.s3 import join_uri, write_pandas
    from trust_score_05.common.splitting import CUSTOMER_KEY_COLUMN, LABEL_COLUMN
    from trust_score_05.ml import datasets
    from trust_score_05.ml.jobs import base as jobs_base
    from trust_score_05.ml.jobs import sweep_job

    def build_spark_session(cfg: Any, job_name: str) -> _NoSpark:
        return _NoSpark()

    def build_local_spark(app_name: str, driver_memory: Any = None) -> _NoSpark:
        return _NoSpark()

    def bounded_to_pandas(frame: Any, max_rows: int = 0, seed: int = 42,
                          context: str = "data") -> pd.DataFrame:
        if max_rows and len(frame) > max_rows:
            return frame.sample(n=int(max_rows), random_state=int(seed)).reset_index(drop=True)
        return frame.reset_index(drop=True)

    def _fraud_rows(cfg: Any) -> pd.DataFrame:
        paths = [p for prefixes in datasets.discover_fraud_paths(cfg).values() for p in prefixes]
        frame = pd.concat([_read(p) for p in paths], ignore_index=True)
        stamp = pd.to_datetime(frame[datasets.FRAUD_TIMESTAMP_COLUMN], errors="coerce")
        return frame.loc[stamp >= pd.Timestamp(cfg.fraud_min_timestamp)].reset_index(drop=True)

    def load_training_matrix(spark: Any, cfg: Any, features: Sequence[str],
                             run_prefix: str) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
        discovered = datasets.discover_training_nonfraud_paths(cfg).require("training non-fraud")
        fraud_ids = set(_fraud_rows(cfg)[CUSTOMER_KEY_COLUMN].astype(str))
        frames, manifest_rows = [], []
        for window, path in discovered.found.items():
            frame = _read(path)
            frame[CUSTOMER_KEY_COLUMN] = frame[CUSTOMER_KEY_COLUMN].astype(str)
            before = len(frame)
            frame = frame.loc[~frame[CUSTOMER_KEY_COLUMN].isin(fraud_ids)]
            month = bounded_to_pandas(_project(frame, features, [CUSTOMER_KEY_COLUMN]),
                                      max_rows=cfg.train_rows_per_month, seed=cfg.seed)
            month["time_window"] = window
            frames.append(month)
            manifest_rows.append({"time_window": window, "path": path,
                                  "rows_before_fraud_removal": before,
                                  "rows_after_fraud_removal": len(frame),
                                  "fraud_customers_removed": before - len(frame),
                                  "rows_sampled": len(month),
                                  "rows_requested": cfg.train_rows_per_month,
                                  "missing_feature_count": 0})
        manifest = pd.DataFrame(manifest_rows)
        write_pandas(manifest, join_uri(run_prefix, "reports", "training_manifest.csv"))
        training = pd.concat(frames, ignore_index=True)
        present = [n for n in features if n in training.columns and training[n].notna().any()]
        return training, present, manifest

    def load_fraud_frame(spark: Any, cfg: Any, features: Sequence[str],
                         run_prefix: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        frame = _fraud_rows(cfg)
        meta = [c for c in datasets.FRAUD_META_COLUMNS if c in frame.columns]
        out = _project(frame, features, meta)
        out[LABEL_COLUMN] = 1
        manifest = {"rows_in": len(frame), "rows_kept": len(out), "rows_null_timestamp": 0,
                    "rows_unparseable_timestamp": 0, "rows_before_min_timestamp": 0,
                    "fraud_min_timestamp": cfg.fraud_min_timestamp}
        return out, manifest

    def load_scenario_frame(spark: Any, cfg: Any, scenario: str, nonfraud_path: str,
                            fraud_frame: Any, features: Sequence[str]) -> pd.DataFrame:
        negatives = _project(_read(nonfraud_path), features, [CUSTOMER_KEY_COLUMN])
        negatives[LABEL_COLUMN] = 0
        negatives = bounded_to_pandas(negatives, max_rows=cfg.max_eval_rows_per_scenario,
                                      seed=cfg.seed)
        frame = pd.concat([negatives, fraud_frame.copy()], ignore_index=True)
        frame["scenario"] = scenario
        return frame

    datasets.load_training_matrix = load_training_matrix
    datasets.load_fraud_frame = load_fraud_frame
    datasets.load_scenario_frame = load_scenario_frame
    datasets.bounded_to_pandas = bounded_to_pandas
    # sweep_job bound these names at import time, so they are replaced there too.
    sweep_job.load_training_matrix = load_training_matrix
    sweep_job.load_fraud_frame = load_fraud_frame
    sweep_job.load_scenario_frame = load_scenario_frame
    sweep_job.bounded_to_pandas = bounded_to_pandas
    jobs_base.build_spark_session = build_spark_session
    sm_common.build_local_spark = build_local_spark

    import train  # the entry point binds build_local_spark by name

    train.build_local_spark = build_local_spark
