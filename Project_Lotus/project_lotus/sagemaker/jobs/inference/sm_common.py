"""Plumbing shared by the two SageMaker entry points, ``train.py`` and ``inference.py``.

Nothing in here makes a modelling decision. Every one of those lives in the
``trust_score_05`` package and is called unchanged; this module only adapts that
package to the SageMaker container contract:

* **Parameters.** A SageMaker *training* job hands its hyperparameters to the
  container as ``/opt/ml/input/config/hyperparameters.json`` and runs the image
  with the single argument ``train``. A *processing* job passes ordinary
  command-line arguments. :func:`container_argv` turns both into one argv so the
  entry points need a single ``argparse`` parser.
* **Spark inside one container.** The ML stages read their inputs through Spark
  (``trust_score_05.ml.datasets``). In SageMaker there is no cluster, so the
  session is ``local[*]`` sized to the instance, with ``s3://`` mapped onto the
  S3A connector baked into the image. The package's own
  ``build_spark_session`` calls ``getOrCreate`` and therefore attaches to the
  session built here instead of creating a bare one.
* **The champion pointer.** Where ``train.py`` publishes the chosen model and
  where ``inference.py`` finds it. One place, so the two cannot disagree.
"""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence

#: The directory holding this file, ``trust_score_05/`` and ``conf/``. In the
#: image that is ``/opt/program``; locally it is the ``sagemaker/`` folder.
PROGRAM_DIR = Path(__file__).resolve().parent

# The package is imported from PROGRAM_DIR, never from an installed copy, so the
# code that runs is exactly the code that was copied into the image.
if str(PROGRAM_DIR) not in sys.path:
    sys.path.insert(0, str(PROGRAM_DIR))

#: The environment this deployment targets. Every default below is sandbox; dev
#: is reached only by naming ``conf/ml/dev.yaml`` explicitly.
DEFAULT_CONFIGS = (
    str(PROGRAM_DIR / "conf" / "ml" / "base.yaml"),
    str(PROGRAM_DIR / "conf" / "ml" / "sandbox.yaml"),
)

SM_HYPERPARAMETERS = Path("/opt/ml/input/config/hyperparameters.json")
SM_MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
SM_FAILURE_FILE = Path("/opt/ml/output/failure")

#: The champion layout, under ``<models_root>/version_id=<version>/champion/``.
CHAMPION_DIRNAME = "champion"
CURRENT_CHAMPION_FILE = "current.json"
CHAMPION_FILE = "champion.json"
REFERENCE_SCORES_FILE = "reference_scores.parquet"
REFERENCE_FEATURES_FILE = "reference_features.parquet"

#: Exit codes, the same contract as the package's jobs: 0 success, 2 a
#: configuration problem an operator must fix, 1 anything else.
EXIT_OK, EXIT_FAILED, EXIT_CONFIG = 0, 1, 2

LOGGER = logging.getLogger("ts05.sagemaker")


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------

def _decode(value: Any) -> str:
    """One hyperparameter value as the string a command line would carry.

    CreateTrainingJob only accepts strings, but the SageMaker Python SDK
    JSON-encodes them first, so ``"abc"`` can arrive as ``"\\"abc\\""``. Both
    forms are accepted so a job launched from either tool behaves the same.
    """
    text = str(value)
    if len(text) >= 2 and text[0] == text[-1] == '"':
        try:
            return str(json.loads(text))
        except ValueError:
            return text
    return text


def hyperparameters_to_argv(params: Dict[str, Any]) -> List[str]:
    """``{"arm-start": "10", "force": "true", "config": "a.yaml,b.yaml"}`` -> argv.

    ``true``/``false`` become a bare flag or nothing, and ``config`` and ``set``
    - the two repeatable options - are split on commas and semicolons
    respectively, because a hyperparameter can only carry one string.
    """
    argv: List[str] = []
    for key in sorted(params):
        flag = "--" + str(key).replace("_", "-")
        value = _decode(params[key])
        lowered = value.strip().lower()
        if lowered == "true":
            argv.append(flag)
        elif lowered == "false" or value == "":
            continue
        elif key == "config":
            for item in value.split(","):
                if item.strip():
                    argv += [flag, item.strip()]
        elif key == "set":
            for item in value.split(";"):
                if item.strip():
                    argv += [flag, item.strip()]
        else:
            argv += [flag, value]
    return argv


def container_argv(argv: Optional[Sequence[str]] = None) -> List[str]:
    """The argv an entry point should parse, from whichever launcher ran it.

    SageMaker training runs the image as ``<entrypoint> train``; that token is
    dropped. Hyperparameters, when the file exists, come first so that anything
    on the real command line can override them.
    """
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] in ("train", "serve"):
        raw = raw[1:]
    from_file: List[str] = []
    if argv is None and SM_HYPERPARAMETERS.is_file():
        from_file = hyperparameters_to_argv(json.loads(SM_HYPERPARAMETERS.read_text()))
    return from_file + raw


def resolve_configs(configs: Optional[Sequence[str]]) -> List[str]:
    """Config paths, relative ones resolved against :data:`PROGRAM_DIR`.

    Resolved here because the working directory inside a SageMaker container is
    not the program directory, and ``conf/ml/sandbox.yaml`` passed as a
    hyperparameter has to mean the copy baked into the image.
    """
    out: List[str] = []
    for item in configs or DEFAULT_CONFIGS:
        text = str(item)
        if "://" in text or os.path.isabs(text):
            out.append(text)
        else:
            out.append(str(PROGRAM_DIR / text))
    return out


def utc_run_id() -> str:
    """A run id in the package's own default format (see ``MLConfig.run_id``)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def record_failure(message: str) -> None:
    """Write SageMaker's failure-reason file, when running inside SageMaker.

    SageMaker shows the first 1 KB of this file as the job's FailureReason, which
    is the difference between an operator reading "configuration error: no
    feature selection for model 'ecod'" in the console and reading "AlgorithmError".
    """
    try:
        if SM_FAILURE_FILE.parent.is_dir():
            SM_FAILURE_FILE.write_text(message[:1024])
    except OSError:  # pragma: no cover - best effort by definition
        pass


# --------------------------------------------------------------------------
# Spark, local to the container
# --------------------------------------------------------------------------

def _total_memory_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (AttributeError, ValueError, OSError):
        try:
            import psutil

            return psutil.virtual_memory().total / 1024**3
        except Exception:  # noqa: BLE001
            return 8.0


def build_local_spark(app_name: str, driver_memory: Optional[str] = None) -> Any:
    """A ``local[*]`` SparkSession sized to this machine, able to read ``s3://``.

    ``spark.driver.memory`` defaults to 70% of physical memory: in local mode the
    driver *is* the executor, and the training sample is collected to pandas in
    the same process, so the rest is left for pandas, sklearn and the models.
    ``maxResultSize`` is unbounded for the same reason - the collection is
    already bounded by ``train_rows_per_month`` in config.

    ``s3://`` is mapped onto S3A because the package's paths are ``s3://`` URIs
    and open-source Spark has no filesystem for that scheme. The connector jars
    are baked into the image (see the Dockerfile); on a laptop the mapping is
    inert as long as the paths are local.
    """
    from pyspark.sql import SparkSession

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

    cpus = os.cpu_count() or 2
    memory = driver_memory or f"{max(2, int(_total_memory_gb() * 0.7))}g"
    builder = (
        SparkSession.builder.appName(app_name)
        .master(f"local[{cpus}]")
        .config("spark.driver.memory", memory)
        .config("spark.driver.maxResultSize", "0")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", str(max(8, cpus * 2)))
        .config("spark.hadoop.fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
        )
        .config("spark.hadoop.fs.s3a.fast.upload", "true")
    )
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if region:
        builder = builder.config("spark.hadoop.fs.s3a.endpoint.region", region)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    LOGGER.info("local Spark ready | cores=%d | driver memory=%s", cpus, memory)
    return spark


# --------------------------------------------------------------------------
# object store helpers the package does not provide
# --------------------------------------------------------------------------

def read_csv_any(uri: str):
    """A CSV at ``uri`` (local or ``s3://``) as a pandas frame."""
    import pandas as pd

    from trust_score_05.common.io.s3 import read_text

    return pd.read_csv(io.StringIO(read_text(uri)))


def read_parquet_any(uri: str):
    """Every parquet file at or under ``uri`` as one pandas frame.

    Accepts a single file or a prefix, local or ``s3://``. S3 objects are
    downloaded to scratch space and read with pyarrow rather than streamed, so
    no extra filesystem dependency (s3fs) is needed in the image.
    """
    import pandas as pd

    from trust_score_05.common.io.s3 import is_s3_uri, list_objects, s3_client, split_s3_uri

    text = str(uri)
    if not is_s3_uri(text):
        if not os.path.exists(text):
            raise FileNotFoundError(f"no parquet at {text}")
        return pd.read_parquet(text)

    keys = [text] if text.endswith((".parquet", ".pq")) else list_objects(text)
    keys = [key for key in keys if key.endswith((".parquet", ".pq")) or "/part-" in key]
    if not keys:
        raise FileNotFoundError(f"no parquet objects under {text}")
    scratch = tempfile.mkdtemp(prefix="ts05_read_")
    try:
        frames = []
        client = s3_client()
        for index, key_uri in enumerate(keys):
            bucket, key = split_s3_uri(key_uri)
            local = os.path.join(scratch, f"{index:06d}.parquet")
            client.download_file(bucket, key, local)
            frames.append(pd.read_parquet(local))
        return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def copy_object(source: str, target: str) -> None:
    """Copy one object between any two of local / ``s3://``."""
    from trust_score_05.common.io.s3 import is_s3_uri, s3_client, split_s3_uri, upload_file

    if not is_s3_uri(source):
        upload_file(source, target)
        return
    scratch = tempfile.mkdtemp(prefix="ts05_copy_")
    try:
        local = os.path.join(scratch, os.path.basename(source) or "object")
        bucket, key = split_s3_uri(source)
        s3_client().download_file(bucket, key, local)
        upload_file(local, target)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# --------------------------------------------------------------------------
# the champion layout
# --------------------------------------------------------------------------

def champion_root(cfg: Any) -> str:
    from trust_score_05.common.io.s3 import join_uri

    return join_uri(cfg.versioned_models_root, CHAMPION_DIRNAME)


def current_champion_uri(cfg: Any) -> str:
    from trust_score_05.common.io.s3 import join_uri

    return join_uri(champion_root(cfg), CURRENT_CHAMPION_FILE)


def champion_history_prefix(cfg: Any, run_id: str) -> str:
    from trust_score_05.common.io.s3 import join_uri

    return join_uri(champion_root(cfg), "history", f"run_id={run_id}")


def configure(log_level: str = "INFO") -> None:
    """The package's logging setup, so every line has the same shape."""
    from trust_score_05.lineage.logging_utils import configure_logging

    configure_logging(log_level)
