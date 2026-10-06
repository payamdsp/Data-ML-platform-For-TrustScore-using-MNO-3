"""Configuration loading for the Silver -> Gold jobs.

Everything that differs between "run it on my laptop against the sample CSV" and
"run it on EMR against S3" lives in a config file, never in code.

Supported sources, applied in this order (later wins):

1. ``conf/lineage/base.yaml``            - defaults shared by every environment
2. one or more overlay files     - ``--config conf/lineage/local.yaml`` (YAML **or** JSON)
3. ``--set key.path=value``      - single-value overrides from the command line
4. ``${ENV_VAR}`` interpolation  - resolved last, anywhere in the merged tree

Both YAML and JSON are accepted; the format is chosen by file extension
(``.yaml`` / ``.yml`` -> YAML, ``.json`` -> JSON).

A config path may be a local file or an ``s3://`` URI. The second form is what a
cluster run uses: ``spark-submit`` ships the package as a zip and the config
files stay in the artifact prefix, so ``--config s3://.../conf/sandbox.yaml`` is
the only way the driver can name them.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

__all__ = [
    "Config",
    "ConfigError",
    "load_config",
    "deep_merge",
    "interpolate_env",
]


class ConfigError(RuntimeError):
    """Raised when configuration is missing, malformed, or self-inconsistent."""


# --------------------------------------------------------------------------
# file loading
# --------------------------------------------------------------------------

# `s3://bucket/key`, and the `s3a://` and `s3n://` spellings Hadoop uses for the
# same object. Matched here rather than handed to `Path`, because `Path` folds
# the double slash - `Path("s3://b/k")` is `s3:/b/k` - and the resulting "config
# file not found: s3:/b/k" sends the reader looking for a typo that is not there.
_S3_URI = re.compile(r"^s3[an]?://(?P<bucket>[^/]+)/(?P<key>.+)$")


def _read_s3_text(bucket: str, key: str, where: str) -> str:
    """Fetch a config object from S3 and return it as text.

    boto3 rather than the Spark session, because config is read before there is
    a session to read with - the session builder takes its settings from this
    file. On EMR boto3 is present on every node; off EMR this is only reached by
    someone who deliberately passed an s3:// path.
    """
    try:
        import boto3  # imported lazily: a local run never needs it
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ConfigError(
            f"{where} is an S3 path but boto3 is not installed. "
            "Install it (`pip install boto3`) or pass a local config file."
        ) from exc

    try:
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as exc:  # noqa: BLE001 - any S3 failure is a config failure
        raise ConfigError(f"could not read config from {where}: {exc}") from exc

    return body.decode("utf-8")


def _config_present(path: str | os.PathLike) -> bool:
    """Whether a config path is worth trying. S3 paths are always tried.

    Only the local branch can answer cheaply, and only the local branch has a
    caller that needs the answer (see ``load_config`` on the default base file).
    An S3 path reports ``True`` so that a real S3 failure still raises with the
    reason attached, rather than being silently skipped.
    """
    return True if _S3_URI.match(str(path)) else Path(path).exists()


def _read_structured_file(path: str | os.PathLike) -> dict:
    """Load a single YAML or JSON config file into a plain dict.

    ``path`` is a local filesystem path or an ``s3://`` URI.
    """
    where = str(path)
    match = _S3_URI.match(where)

    if match:
        text = _read_s3_text(match.group("bucket"), match.group("key"), where)
        # PurePosixPath, not Path: an S3 key is always slash-separated, whatever
        # the operating system running the driver happens to separate with.
        suffix = PurePosixPath(match.group("key")).suffix.lower()
    else:
        local = Path(path)
        if not local.exists():
            raise ConfigError(f"config file not found: {local}")
        text = local.read_text(encoding="utf-8")
        suffix = local.suffix.lower()

    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # imported lazily so JSON-only users need no PyYAML
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ConfigError(
                f"{where} is YAML but PyYAML is not installed. "
                "Install it (`pip install pyyaml`) or use a .json config."
            ) from exc
        loaded = yaml.safe_load(text)
    elif suffix == ".json":
        loaded = json.loads(text)
    else:
        raise ConfigError(
            f"unsupported config extension {suffix!r} for {where}; "
            "expected .yaml, .yml, or .json"
        )

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{where} must contain a mapping at the top level, got {type(loaded).__name__}")
    return loaded


# --------------------------------------------------------------------------
# merging / interpolation / coercion
# --------------------------------------------------------------------------

def deep_merge(base: Mapping, overlay: Mapping) -> dict:
    """Recursively merge ``overlay`` into ``base``; ``overlay`` wins on conflict.

    Nested dicts are merged key by key. Every other type (including lists) is
    replaced wholesale, so an overlay can shorten a list rather than only ever
    growing it.
    """
    out: dict = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def interpolate_env(value: Any, env: Mapping[str, str] | None = None) -> Any:
    """Replace ``${VAR}`` / ``${VAR:-default}`` in every string in the tree.

    An unset variable with no default raises, so a typo in a config file fails
    at submit time rather than three stages into a Spark job.
    """
    environ = os.environ if env is None else env

    if isinstance(value, str):
        def _sub(match: re.Match) -> str:
            name, default = match.group(1), match.group(2)
            if name in environ:
                return environ[name]
            if default is not None:
                return default
            raise ConfigError(
                f"environment variable ${{{name}}} referenced in config is not set "
                f"and has no ${{{name}:-default}} fallback"
            )
        return _ENV_PATTERN.sub(_sub, value)

    if isinstance(value, Mapping):
        return {k: interpolate_env(v, environ) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [interpolate_env(v, environ) for v in value]
    return value


def _coerce_scalar(raw: str) -> Any:
    """Turn a ``--set`` right-hand side into a bool / int / float / None / str."""
    lowered = raw.strip().lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "none", "~"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if "," in raw:
        return [part.strip() for part in raw.split(",")]
    return raw


def _assign_dotted(tree: MutableMapping, dotted_key: str, value: Any) -> None:
    """Set ``tree['a']['b']['c'] = value`` from the key ``"a.b.c"``."""
    parts = dotted_key.split(".")
    node: MutableMapping = tree
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, MutableMapping):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


# --------------------------------------------------------------------------
# Config object
# --------------------------------------------------------------------------

_MISSING = object()


@dataclass(frozen=True)
class Config:
    """Read-only view over the merged configuration tree."""

    data: Mapping[str, Any] = field(default_factory=dict)
    sources: Sequence[str] = field(default_factory=tuple)

    # -- access ------------------------------------------------------------
    def get(self, dotted_key: str, default: Any = _MISSING) -> Any:
        """Fetch ``a.b.c``; return ``default`` (or raise) when absent."""
        node: Any = self.data
        for part in dotted_key.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                if default is _MISSING:
                    raise ConfigError(
                        f"missing required config key {dotted_key!r} "
                        f"(loaded from: {', '.join(self.sources) or '<none>'})"
                    )
                return default
        return node

    def require(self, dotted_key: str) -> Any:
        """Fetch ``a.b.c``, raising if it is absent or None/empty-string."""
        value = self.get(dotted_key)
        if value is None or value == "":
            raise ConfigError(f"config key {dotted_key!r} must be set to a non-empty value")
        return value

    def section(self, dotted_key: str) -> "Config":
        """Return a sub-``Config`` rooted at ``dotted_key``."""
        node = self.get(dotted_key)
        if not isinstance(node, Mapping):
            raise ConfigError(f"config key {dotted_key!r} is not a mapping")
        return Config(data=node, sources=self.sources)

    def dataset(self, dataset_key: str) -> "Config":
        """Return the ``datasets.<dataset_key>`` sub-config."""
        try:
            return self.section(f"datasets.{dataset_key}")
        except ConfigError as exc:
            known = ", ".join(sorted(self.get("datasets", {}).keys())) or "<none>"
            raise ConfigError(
                f"no configuration for dataset {dataset_key!r}; configured datasets: {known}"
            ) from exc

    def as_dict(self) -> dict:
        return copy.deepcopy(dict(self.data))

    def __contains__(self, dotted_key: str) -> bool:  # pragma: no cover - trivial
        return self.get(dotted_key, None) is not None


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------

def load_config(
    config_paths: Iterable[str | os.PathLike] | None = None,
    overrides: Iterable[str] | None = None,
    base_path: str | os.PathLike | None = None,
    env: Mapping[str, str] | None = None,
) -> Config:
    """Build the effective :class:`Config`.

    Parameters
    ----------
    config_paths
        Overlay files applied in order after the base file. YAML or JSON.
    overrides
        ``key.path=value`` strings, applied after all files.
    base_path
        Defaults to ``<repo>/conf/lineage/base.yaml``.
    env
        Environment mapping used for ``${VAR}`` interpolation (defaults to ``os.environ``).
    """
    sources: list[str] = []
    overlays = list(config_paths or ())

    if base_path is None:
        base_path = Path(__file__).resolve().parents[2] / "conf" / "lineage" / "base.yaml"

    # The default base is derived from `__file__`, which is a real directory in a
    # checkout and a path *inside a zip* on a cluster: `spark-submit --py-files`
    # ships `trust_score_05.zip`, so `parents[2]` resolves to the YARN
    # container's working directory and there is no `conf/` under it. Treating
    # that as fatal made every cluster submit exit 2 before reading a line of
    # the config it was given on the command line.
    #
    # So the base is required only when it is the sole source. A run that names
    # overlays has said where its configuration comes from - and on the cluster
    # the first of those overlays *is* base.yaml, out of the artifact prefix. A
    # run that names nothing still fails loudly, because an empty config would
    # otherwise surface much later as a missing key with no hint of the cause.
    if _config_present(base_path):
        merged = _read_structured_file(base_path)
        sources.append(str(base_path))
    elif overlays:
        merged = {}
        sources.append(f"{base_path} (absent; overlays supplied)")
    else:
        raise ConfigError(
            f"config file not found: {base_path}, and no --config overlay was given. "
            "Pass the base explicitly (--config .../conf/lineage/base.yaml) when running "
            "from a packaged zip rather than from a checkout."
        )

    for path in overlays:
        merged = deep_merge(merged, _read_structured_file(path))
        sources.append(str(path))

    for override in overrides or ():
        if "=" not in override:
            raise ConfigError(f"--set expects key.path=value, got {override!r}")
        key, _, raw = override.partition("=")
        _assign_dotted(merged, key.strip(), _coerce_scalar(raw))
        sources.append(f"--set {key.strip()}")

    merged = interpolate_env(merged, env)
    return Config(data=merged, sources=tuple(sources))
