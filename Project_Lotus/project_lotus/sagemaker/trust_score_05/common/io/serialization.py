"""Saving and loading a fitted model as one self-describing bundle.

A scored record is only reproducible if everything that stood between the raw
feature row and the score is preserved together: the feature list *in order*,
the imputation values that were learned on the training set, the fitted scaler,
the fitted model, and the hyperparameters. Persisting the estimator alone —
which is what a bare ``joblib.dump(model)`` does — loses the other four, and the
loss is silent: the bundle loads, scoring runs, and the numbers are wrong
because the columns arrived in a different order or were imputed from a
different population.

So the unit of persistence here is a :class:`ModelBundle`, and it is validated on
the way out and on the way in. The notebook wrote a bare dict with five to nine
keys depending on which of two code paths ran — one wrote ``model.joblib`` with
five keys, the other ``model_artifact.joblib`` with nine — and nothing checked
either shape, so a consumer had to guess.

``joblib`` rather than ``pickle`` because scikit-learn estimators carry large
numpy arrays and joblib stores them without the round trip through pickle's
byte-string protocol. Neither format is portable across library versions, hence
:attr:`ModelBundle.library_versions`: a bundle that will not load, or loads and
behaves differently, should be able to say why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
import shutil
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence

from trust_score_05.common.io.s3 import (
    ObjectStoreError,
    is_s3_uri,
    join_uri,
    s3_client,
    split_s3_uri,
    upload_file,
    write_json,
)

__all__ = [
    "BUNDLE_FILENAME",
    "BUNDLE_FORMAT_VERSION",
    "BundleError",
    "ModelBundle",
    "bundle_as_dict",
    "bundle_from_parts",
    "load_bundle",
    "save_bundle",
]

#: The name every bundle is written under inside its ``model_artifacts/`` prefix.
#: Fixed rather than parameterised so that a consumer can find the artifact from
#: the run prefix alone.
BUNDLE_FILENAME = "model.joblib"

#: Bumped when the bundle's field set changes incompatibly. :func:`load_bundle`
#: refuses a bundle from a future version rather than reading the fields it
#: understands and quietly dropping the rest.
BUNDLE_FORMAT_VERSION = 1

_REQUIRED_FIELDS = ("model_name", "features", "preprocessing_id", "preprocessor", "model")

_VERSIONED_LIBRARIES = ("numpy", "pandas", "sklearn", "joblib", "pyod", "hdbscan", "torch")


class BundleError(RuntimeError):
    """Raised when a bundle is malformed, incomplete, or of an unknown version."""


def _library_versions() -> Dict[str, str]:
    """Versions of the libraries whose objects end up inside a bundle.

    Collected best-effort: an absent library is omitted, because the bundle for a
    COPOD run legitimately has no torch in it and recording ``"torch": "not
    installed"`` on every non-autoencoder run is noise.
    """
    versions: Dict[str, str] = {}
    for module_name in _VERSIONED_LIBRARIES:
        try:
            module = __import__(module_name)
        except Exception:  # noqa: BLE001 - a broken optional install is not fatal here
            continue
        version = getattr(module, "__version__", None)
        if version:
            versions[module_name] = str(version)
    return versions


@dataclass
class ModelBundle:
    """Everything needed to reproduce a score from a raw feature row.

    Attributes:
        model_name: The adapter that produced ``model``, e.g. ``kmeans``. Kept
            because several adapters produce the same object type — weighted and
            unweighted k-means both hold a ``MiniBatchKMeans`` — and the scoring
            behaviour differs. Dispatching on the stored object's class was one
            of the notebook's live bugs: the weighted fit returned a bundle
            marked as plain k-means, so scoring ran in the unweighted space
            against weighted centroids.
        features: The feature columns, in the order the preprocessor was fitted
            on. Order is part of the contract, not metadata.
        preprocessing_id: The preprocessing variant identifier, e.g.
            ``domain_impute_robust_scaler``.
        impute_rules: Per-feature fill values *learned on the training set*. The
            notebook recomputed these at scoring time from whatever rows were
            being scored, which imputed the fraud population from the fraud
            population and made the imputed value itself label-correlated.
        preprocessor: The fitted transformer.
        model: The fitted estimator, or whatever object the adapter scores with.
        params: The hyperparameters, as the adapter received them.
        extra: Adapter-specific state that does not fit above — feature weights
            for weighted k-means, per-cluster score statistics for the k-means
            score variants.
    """

    model_name: str
    features: List[str]
    preprocessing_id: str
    impute_rules: Dict[str, float]
    preprocessor: Any
    model: Any
    params: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- provenance, filled in by save_bundle ------------------------------
    format_version: int = BUNDLE_FORMAT_VERSION
    saved_at_utc: Optional[str] = None
    library_versions: Dict[str, str] = field(default_factory=dict)

    def validate(self) -> "ModelBundle":
        """Check the invariants a scorer relies on. Returns ``self``.

        ``features`` must be non-empty and free of duplicates, because a
        duplicate column means the preprocessor was fitted on a wider matrix than
        the feature list describes and every column after the duplicate is offset
        by one. ``impute_rules`` must not name a feature outside ``features``,
        which catches a rule set built from a different selection. And ``model``
        must be present, since a bundle without one is a preprocessing pipeline
        that has been mislabelled.
        """
        if not self.model_name:
            raise BundleError("bundle has no model_name")
        if not self.features:
            raise BundleError("bundle has no features")
        seen: Dict[str, int] = {}
        for name in self.features:
            seen[name] = seen.get(name, 0) + 1
        duplicates = sorted(name for name, count in seen.items() if count > 1)
        if duplicates:
            raise BundleError(f"bundle feature list has duplicates: {duplicates}")
        if self.model is None:
            raise BundleError("bundle has no fitted model")
        unknown = sorted(set(self.impute_rules) - set(self.features))
        if unknown:
            raise BundleError(
                "bundle impute_rules name features that are not in the feature "
                f"list: {unknown[:10]}"
            )
        return self

    def metadata(self) -> Dict[str, Any]:
        """The bundle without the two heavy objects, for writing alongside it.

        Written as ``model_artifacts/model_metadata.json`` so that a run's
        configuration is readable without loading the joblib file — which needs
        the library versions the bundle was written with, and is therefore
        exactly what one cannot do while investigating why a bundle will not
        load.
        """
        payload = {
            name: getattr(self, name)
            for name in type(self).__dataclass_fields__
            if name not in ("preprocessor", "model")
        }
        payload["extra_keys"] = sorted(self.extra)
        payload["extra"] = {
            key: value
            for key, value in self.extra.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        }
        return payload


def bundle_as_dict(bundle: ModelBundle) -> Dict[str, Any]:
    """The bundle as a plain dict, without recursing into its values.

    ``dataclasses.asdict`` deep-copies, and a deep copy of a fitted estimator is
    both slow and — for the torch-backed autoencoder, whose module holds
    non-copyable handles — capable of failing outright. The persisted form only
    needs the field mapping; joblib serialises the values itself.
    """
    return {name: getattr(bundle, name) for name in type(bundle).__dataclass_fields__}


def save_bundle(bundle: ModelBundle, prefix: str) -> str:
    """Write ``bundle`` and its metadata sidecar under ``prefix``. Returns the URI.

    ``prefix`` is the run's ``model_artifacts`` prefix. Validation happens before
    anything is written, so a malformed bundle fails without leaving a partial
    artifact for a later run to find and trust.
    """
    import joblib

    bundle.validate()
    bundle.saved_at_utc = datetime.now(timezone.utc).isoformat()
    bundle.library_versions = _library_versions()

    target = join_uri(prefix, BUNDLE_FILENAME)
    scratch = tempfile.mkdtemp(prefix="ts05_bundle_")
    try:
        local = os.path.join(scratch, BUNDLE_FILENAME)
        joblib.dump(bundle_as_dict(bundle), local)
        upload_file(local, target)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    write_json(join_uri(prefix, "model_metadata.json"), bundle.metadata())
    return target


def _download_to_temp(uri: str, scratch: str) -> str:
    """Fetch ``uri`` into ``scratch`` and return the local path.

    Both branches raise :class:`ObjectStoreError`, so a caller has one exception
    type to catch for "the artifact could not be fetched" regardless of which
    backend it came from. Without the local branch's own check, a missing file
    surfaced as a bare ``FileNotFoundError`` from ``shutil.copy2`` naming the
    *scratch* destination, which reads as a bug in this module rather than as a
    missing bundle.
    """
    local = os.path.join(scratch, os.path.basename(str(uri)) or BUNDLE_FILENAME)
    if not is_s3_uri(uri):
        if not os.path.isfile(str(uri)):
            raise ObjectStoreError(f"could not download {uri}: no such file")
        shutil.copy2(str(uri), local)
        return local
    bucket, key = split_s3_uri(uri)
    try:
        s3_client().download_file(bucket, key, local)
    except Exception as exc:  # noqa: BLE001
        raise ObjectStoreError(f"could not download {uri}: {exc}") from exc
    return local


def load_bundle(uri: str) -> ModelBundle:
    """Read a bundle from ``uri``, which may name the file or its prefix.

    Both forms are accepted because callers have both: the training sweep knows
    the ``model_artifacts`` prefix it wrote to, while the leaderboard has a full
    object key from a recursive listing.
    """
    import joblib

    target = str(uri) if str(uri).endswith(".joblib") else join_uri(uri, BUNDLE_FILENAME)
    scratch = tempfile.mkdtemp(prefix="ts05_bundle_")
    try:
        payload = joblib.load(_download_to_temp(target, scratch))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    if not isinstance(payload, Mapping):
        raise BundleError(
            f"{target} does not contain a model bundle; it holds a "
            f"{type(payload).__name__}. A bare pickled estimator is not enough — "
            "see this module's docstring."
        )

    version = int(payload.get("format_version", 0))
    if version > BUNDLE_FORMAT_VERSION:
        raise BundleError(
            f"{target} was written in bundle format v{version}; this code "
            f"understands up to v{BUNDLE_FORMAT_VERSION}."
        )

    missing = sorted(name for name in _REQUIRED_FIELDS if name not in payload)
    if missing:
        raise BundleError(f"{target} is missing required bundle fields: {missing}")

    known = set(ModelBundle.__dataclass_fields__)
    return ModelBundle(**{k: v for k, v in payload.items() if k in known}).validate()


def bundle_from_parts(
    model_name: str,
    features: Sequence[str],
    preprocessing_id: str,
    impute_rules: Mapping[str, float],
    preprocessor: Any,
    model: Any,
    params: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> ModelBundle:
    """Build a validated bundle from loose parts.

    A convenience for the training path, which holds all seven pieces as separate
    locals and would otherwise construct the dataclass positionally.
    """
    return ModelBundle(
        model_name=str(model_name),
        features=list(features),
        preprocessing_id=str(preprocessing_id),
        impute_rules={str(k): float(v) for k, v in dict(impute_rules).items()},
        preprocessor=preprocessor,
        model=model,
        params=dict(params or {}),
        extra=dict(extra or {}),
    ).validate()
