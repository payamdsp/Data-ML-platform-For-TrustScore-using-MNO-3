"""Where every artifact goes, and how to find one again.

The output layout is a nested set of partition-style segments:

.. code-block:: text

    <models_root>/version_id=<version>/
        model=<model>/
            feature_selection_method=<method>/
                top_n_features=<n>/
                    preprocessing=<preprocessing_id>/
                        run_id=<run_id>/            <- the "run prefix"
                            _STARTED.json  _READY.json  _FAILED.json
                            config/run_config.json
                            reports/…
                            metrics/…
                            plots/  plot_data/
                            configs/<hyperparam_id>/ <- the "config prefix"
                                model_artifacts/model.joblib
                                metrics/scenario=<s>/{metrics,k_metrics}.csv
                                scores/scenario=<s>/scored_records_parquet/

Every segment is ``key=value``, which is not decoration: it means the layout is
readable as a Hive-partitioned table, so the whole sweep's metrics can be queried
with one ``spark.read.csv`` over the version root and the six dimensions come
back as columns. The notebook wrote exactly this layout and then never used the
property — its cross-model comparison walked every key under the version root,
kept the ones ending in a metrics filename, read each one individually, and
concatenated them *without parsing the path*, so the resulting table had no model
column and its leaderboard could not say which model a row belonged to.
:func:`parse_run_prefix` is here so that does not happen again.

The functions build paths; they never create or check anything. Existence
questions go to :mod:`trust_score_05.common.io.s3`.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

from trust_score_05.common.io.s3 import join_uri
from trust_score_05.ml.config import MLConfig

__all__ = [
    "RUN_PREFIX_SEGMENTS",
    "config_prefix",
    "feature_selection_prefix",
    "fraud_source_root",
    "hyperparam_id",
    "parse_run_prefix",
    "run_prefix",
    "scenario_metrics_prefix",
    "scenario_scores_prefix",
    "schema_prefix",
    "sweep_summary_prefix",
    "testing_nonfraud_candidates",
    "training_nonfraud_candidates",
]

#: The partition keys of a run prefix, outermost first. The order is the
#: directory order and is also the order a reader should think in: which model,
#: selected how, how many features, preprocessed how, which execution.
RUN_PREFIX_SEGMENTS = (
    "version_id",
    "model",
    "feature_selection_method",
    "top_n_features",
    "preprocessing",
    "run_id",
)

_SEGMENT_PATTERN = re.compile(r"(?P<key>[a-z_]+)=(?P<value>[^/]+)")

# A path segment value must not contain a slash (it would create a directory) or
# an equals sign (it would break the key=value parse). Everything else is left
# alone; hyphens and dots are legitimate in scenario names and versions.
_UNSAFE_SEGMENT_CHARS = re.compile(r"[/=\s]+")


def _segment(key: str, value: object) -> str:
    """``key=value``, with the value made safe as a single path component."""
    text = _UNSAFE_SEGMENT_CHARS.sub("_", str(value)).strip("_")
    if not text:
        raise ValueError(f"path segment {key}= has an empty value (got {value!r})")
    return f"{key}={text}"


# --------------------------------------------------------------------------
# input discovery
# --------------------------------------------------------------------------

def training_nonfraud_candidates(cfg: MLConfig, time_window: str) -> List[str]:
    """Candidate ``/data/`` prefixes for one training month, most likely first.

    Two layouts exist in the bucket and both are real. The training non-fraud
    population sits directly under the version root
    (``version=…/time_window=…/data/``); the testing non-fraud population sits
    under a ``non-fraud/`` family segment. The caller probes these in order and
    takes the first that exists.

    The direct layout is tried first *for training* deliberately: there is a
    stale ``non-fraud/`` tree under the training root from an earlier extract,
    and preferring the family layout would silently read it.
    """
    root = cfg.training_root
    if not root:
        raise ValueError("training_root is not configured")
    return [
        join_uri(root, _segment("time_window", time_window), "data") + "/",
        join_uri(root, "non-fraud", _segment("time_window", time_window), "data") + "/",
    ]


def testing_nonfraud_candidates(cfg: MLConfig, time_window: str) -> List[str]:
    """Candidate ``/data/`` prefixes for one evaluation window, most likely first.

    The mirror of :func:`training_nonfraud_candidates`, with the family layout
    preferred because that is where the testing extract actually writes.
    """
    root = cfg.testing_root
    if not root:
        raise ValueError("testing_root is not configured")
    return [
        join_uri(root, "non-fraud", _segment("time_window", time_window), "data") + "/",
        join_uri(root, _segment("time_window", time_window), "data") + "/",
    ]


def fraud_source_root(cfg: MLConfig, fraud_source: str) -> str:
    """The prefix holding one fraud feed's date partitions.

    The partition values under it are not enumerable from config — the feed adds
    them — so the caller lists them with
    :func:`trust_score_05.common.io.s3.list_common_prefixes`.
    """
    if not cfg.testing_root:
        raise ValueError("testing_root is not configured")
    return join_uri(cfg.testing_root, "fraud", str(fraud_source)) + "/"


# --------------------------------------------------------------------------
# output layout
# --------------------------------------------------------------------------

def schema_prefix(cfg: MLConfig) -> str:
    """Where the discovery stage writes its schema inventory and DQ report."""
    return join_uri(
        cfg.versioned_models_root,
        "schema-discovery",
        _segment("run_id", cfg.run_id),
    )


def sweep_summary_prefix(cfg: MLConfig) -> str:
    """Where a sweep's cross-arm summary and leaderboard go.

    Deliberately a sibling of the ``model=`` trees rather than inside one. What
    is written here is the one artifact of a sweep that belongs to no single arm:
    the ranking *across* models, feature counts and preprocessing variants. The
    notebook had nowhere to put it and so put it under whichever model ran last,
    which meant the cross-model comparison was findable only by knowing the
    submission order.

    Under ``run_id=`` because a fanned-out sweep's arms share one run id (see
    ``conf/ml/sandbox.yaml``): every submission's summary lands in the same
    place, and the last one to finish sees the whole sweep.
    """
    return join_uri(
        cfg.versioned_models_root,
        "sweep-summary",
        _segment("run_id", cfg.run_id),
    )


def feature_selection_prefix(
    cfg: MLConfig,
    model: str,
    method: str,
    run_id: Optional[str] = None,
) -> str:
    """Where one model's selected feature lists are written.

    Kept under ``feature_selection_root`` rather than under the models root,
    because a selection is reused across many training runs and versioning it
    with the models would force a re-selection whenever the model version was
    bumped.
    """
    if not cfg.feature_selection_root:
        raise ValueError("feature_selection_root is not configured")
    return join_uri(
        cfg.feature_selection_root,
        _segment("model", model),
        _segment("method", method),
        _segment("run_id", run_id if run_id is not None else cfg.run_id),
    )


def run_prefix(
    cfg: MLConfig,
    model: str,
    method: str,
    top_n_features: int,
    preprocessing_id: str,
    run_id: Optional[str] = None,
) -> str:
    """The prefix for one ``(model, method, n, preprocessing, run)`` combination.

    Everything a single training job writes lives under this, so it is the unit
    the ``_READY.json`` marker applies to and the unit a resumable sweep skips.
    """
    return join_uri(
        cfg.versioned_models_root,
        _segment("model", model),
        _segment("feature_selection_method", method),
        _segment("top_n_features", int(top_n_features)),
        _segment("preprocessing", preprocessing_id),
        _segment("run_id", run_id if run_id is not None else cfg.run_id),
    )


def config_prefix(run: str, hyperparam: str) -> str:
    """The prefix for one hyperparameter configuration inside a run."""
    return join_uri(run, "configs", _UNSAFE_SEGMENT_CHARS.sub("_", str(hyperparam)))


def scenario_scores_prefix(config: str, scenario: str) -> str:
    """Where a configuration's scored records for one scenario are written."""
    return join_uri(config, "scores", _segment("scenario", scenario), "scored_records_parquet")


def scenario_metrics_prefix(config: str, scenario: str) -> str:
    """Where a configuration's metrics for one scenario are written."""
    return join_uri(config, "metrics", _segment("scenario", scenario))


def hyperparam_id(model: str, params: Dict[str, object], max_length: int = 190) -> str:
    """A short, stable, filesystem-safe identifier for a parameter dict.

    Built from the sorted key/value pairs so that the same parameters always
    produce the same identifier regardless of dict insertion order — which
    matters because it is how a resumable sweep recognises a configuration it has
    already finished.

    Long identifiers are truncated and given a hash suffix. The hash is over the
    *full* parameter set, so two configurations that agree on their first 180
    characters and differ later still get different identifiers. Truncating
    without the hash — which is what happens if the suffix is forgotten — makes
    them collide, and the second one then finds the first one's ``_READY.json``
    and is skipped.
    """
    import hashlib
    import json

    parts = [str(model)]
    for key in sorted(params):
        value = params[key]
        text = "none" if value is None else str(value)
        text = _UNSAFE_SEGMENT_CHARS.sub("", text).replace(".", "p").replace(",", "-")
        parts.append(f"{key}-{text}")
    name = "_".join(parts)

    if len(name) > max_length:
        digest = hashlib.md5(
            json.dumps({"model": model, "params": params}, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]
        name = name[: max_length - 9].rstrip("_") + "_" + digest
    return name


# --------------------------------------------------------------------------
# reading the layout back
# --------------------------------------------------------------------------

def parse_run_prefix(uri: str, keys: Sequence[str] = RUN_PREFIX_SEGMENTS) -> Dict[str, str]:
    """Recover the ``key=value`` segments from a path under the version root.

    Returns only the keys in ``keys`` that were present. A key appearing twice —
    which cannot happen in a well-formed path but can in a hand-assembled one —
    resolves to the *last* occurrence, since the deeper segment is the more
    specific.

    This is what makes a cross-model leaderboard possible: a metrics CSV found by
    a recursive listing carries no model identity in its contents, only in its
    path. The notebook read those CSVs and concatenated them without this step,
    which is why its "leaderboard" was a list of anonymous rows.
    """
    wanted = set(keys)
    found: Dict[str, str] = {}
    for match in _SEGMENT_PATTERN.finditer(str(uri)):
        key = match.group("key")
        if key in wanted:
            found[key] = match.group("value")
    return found
