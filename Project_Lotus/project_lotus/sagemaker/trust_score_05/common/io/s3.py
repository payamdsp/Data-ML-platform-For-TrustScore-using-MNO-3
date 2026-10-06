"""Object-store access for the ML pipeline: text, JSON, markers, tables.

Every function here takes a *URI* rather than a path, and accepts either an
``s3://`` URI or a local filesystem path. That dual form is deliberate: the
pipeline runs on SageMaker against S3 and in tests against a ``tmp_path``, and
the alternative — a filesystem abstraction injected through every call site —
buys nothing when the only two backends are "S3" and "a directory".

Three things in here are corrections rather than ports.

``s3_exists`` distinguishes a *failed* listing from an *empty* one. The notebook
version caught every exception from ``list_objects_v2`` and returned ``False``,
which meant an expired credential, a throttle, or a typo'd bucket all reported
"this prefix does not exist" — and the callers of ``ready_exists`` read that as
"the upstream stage has not run yet" and went on to recompute it. A missing
object is a fact about the data; a failed request is a fact about the client, and
the two must not share a return value.

``join_uri`` exists because the notebook code wrote ``prefix.rstrip() + "/name"``
in roughly twenty places. ``str.rstrip()`` with no argument strips *whitespace*,
not the trailing slash that was meant, so any prefix that ended in ``/`` — which
is most of them, since they are built by concatenation — produced
``.../run_id=x//_READY.json``. S3 keeps that double slash as a literal character
in the key, so the marker was written to one key and looked for at another.

``write_pandas`` refuses to guess. The notebook chose parquet when the URI ended
in ``.parquet`` and CSV otherwise, which silently wrote a CSV to a URI ending in
``.pq`` and to one ending in no extension at all.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
import shutil
import tempfile
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = [
    "MARKER_FAILED",
    "MARKER_READY",
    "MARKER_STARTED",
    "ObjectStoreError",
    "is_s3_uri",
    "join_uri",
    "list_common_prefixes",
    "list_objects",
    "read_json",
    "read_text",
    "ready_exists",
    "s3_client",
    "s3_exists",
    "split_s3_uri",
    "upload_file",
    "write_json",
    "write_marker",
    "write_pandas",
    "write_text",
]

LOGGER = logging.getLogger(__name__)

_S3_SCHEME = "s3://"

#: Marker names the pipeline writes. ``STARTED`` is written when a stage claims
#: a prefix, ``READY`` when it finishes, ``FAILED`` when it raises. A prefix with
#: ``STARTED`` and neither of the others is a crashed run, which is a different
#: thing from a run that has not begun — and the distinction is the only way a
#: resumable sweep can tell "skip this, it is done" from "redo this, it died".
MARKER_STARTED = "STARTED"
MARKER_READY = "READY"
MARKER_FAILED = "FAILED"


class ObjectStoreError(RuntimeError):
    """Raised when the object store itself fails, as opposed to a key being absent.

    The whole reason this exception exists is so that :func:`s3_exists` can
    return ``False`` for "no such key" and raise for "the request did not
    complete".
    """


# --------------------------------------------------------------------------
# URIs
# --------------------------------------------------------------------------

def is_s3_uri(uri: str) -> bool:
    """Whether ``uri`` names an S3 object rather than a local path."""
    return str(uri).startswith(_S3_SCHEME)


def split_s3_uri(uri: str) -> Tuple[str, str]:
    """``s3://bucket/a/b`` -> ``("bucket", "a/b")``.

    The key may be empty, for a URI naming a bucket root. It is never prefixed
    with a slash, because the S3 API treats a leading slash as part of the key
    and creates an object whose name begins with one.
    """
    text = str(uri)
    if not is_s3_uri(text):
        raise ValueError(f"not an S3 URI: {text!r}")
    bucket, _, key = text[len(_S3_SCHEME) :].partition("/")
    if not bucket:
        raise ValueError(f"S3 URI has no bucket: {text!r}")
    return bucket, key


def join_uri(prefix: str, *parts: str) -> str:
    """Join URI components with exactly one slash between each.

    ``join_uri("s3://b/p/", "/metrics/", "k.csv")`` is ``"s3://b/p/metrics/k.csv"``.
    Empty parts are dropped, so a caller can pass a conditional segment without
    branching. The scheme's own ``//`` survives because only the *separators the
    function inserts* are normalised.
    """
    head = str(prefix).rstrip("/")
    tail = [str(part).strip("/") for part in parts]
    return "/".join([head, *[part for part in tail if part]])


# --------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------

def s3_client():  # pragma: no cover - trivial, and needs credentials to test
    """A boto3 S3 client.

    Imported lazily and not cached. Lazily because a purely local run — the test
    suite, a laptop pointed at a temp directory — should not require boto3 to be
    installed. Uncached because a cached client outlives a credential refresh in
    a long-running notebook kernel, and the failure that produces is an
    ``ExpiredToken`` five hours into a sweep.
    """
    try:
        import boto3
    except ImportError as exc:
        raise ObjectStoreError(
            "boto3 is required to reach s3:// URIs; install it or point the "
            "pipeline at a local directory."
        ) from exc
    return boto3.client("s3")


# --------------------------------------------------------------------------
# existence
# --------------------------------------------------------------------------

def _is_not_found(exc: BaseException) -> bool:
    """Whether a botocore exception means "no such key" rather than "no answer".

    Matched on the HTTP status in the response metadata rather than on the
    exception class, because ``head_object`` raises a generic ``ClientError``
    with a 404 inside it rather than the ``NoSuchKey`` that the equivalent
    ``get_object`` raises. 403 counts as not-found because a bucket policy that
    denies ``HeadObject`` on a missing key returns it in place of a 404.
    """
    response = getattr(exc, "response", None)
    if not isinstance(response, Mapping):
        return False
    metadata = response.get("ResponseMetadata")
    status = metadata.get("HTTPStatusCode") if isinstance(metadata, Mapping) else None
    return status in (403, 404)


def s3_exists(uri: str) -> bool:
    """Whether ``uri`` names an existing object or a non-empty prefix.

    Both forms are checked, in that order, because callers use this for two
    different kinds of thing: marker *objects* such as ``.../_READY.json``, and
    data *prefixes* such as ``.../time_window=2024-03-01/data/``. A URI ending in
    ``/`` is only ever a prefix, so the object probe is skipped for those.

    Raises :class:`ObjectStoreError` when the store cannot answer. This is the
    correction described in the module docstring: the notebook returned ``False``
    on any exception, so a throttle or an expired token was indistinguishable
    from an absent key, and the callers that treat absence as "not computed yet"
    recomputed finished work — or, worse, treated a fully populated prefix as
    empty and wrote a fresh ``_READY.json`` over it.
    """
    text = str(uri)
    if not is_s3_uri(text):
        return os.path.exists(text)

    bucket, key = split_s3_uri(text.rstrip("/"))
    if not key:
        return True
    client = s3_client()

    if not text.endswith("/"):
        try:
            client.head_object(Bucket=bucket, Key=key)
            return True
        except Exception as exc:  # noqa: BLE001 - the type depends on botocore
            if not _is_not_found(exc):
                raise ObjectStoreError(f"could not probe {text}: {exc}") from exc

    try:
        response = client.list_objects_v2(
            Bucket=bucket, Prefix=key.rstrip("/") + "/", MaxKeys=1
        )
    except Exception as exc:  # noqa: BLE001
        raise ObjectStoreError(f"could not list {text}: {exc}") from exc
    return int(response.get("KeyCount", 0)) > 0


def ready_exists(prefix: str) -> bool:
    """Whether ``prefix`` carries a ``_READY.json`` marker."""
    return s3_exists(join_uri(prefix, f"_{MARKER_READY}.json"))


# --------------------------------------------------------------------------
# text and JSON
# --------------------------------------------------------------------------

def write_text(uri: str, text: str) -> None:
    """Write ``text`` to ``uri`` as UTF-8, creating local parents as needed."""
    if is_s3_uri(uri):
        bucket, key = split_s3_uri(uri)
        try:
            s3_client().put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise ObjectStoreError(f"could not write {uri}: {exc}") from exc
        return

    parent = os.path.dirname(uri)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(uri, "w", encoding="utf-8") as handle:
        handle.write(text)


def read_text(uri: str) -> str:
    """Read ``uri`` as UTF-8 text."""
    if is_s3_uri(uri):
        bucket, key = split_s3_uri(uri)
        try:
            body = s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
        except Exception as exc:  # noqa: BLE001
            raise ObjectStoreError(f"could not read {uri}: {exc}") from exc
        return body.decode("utf-8")
    with open(uri, "r", encoding="utf-8") as handle:
        return handle.read()


def write_json(uri: str, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` as indented JSON.

    ``default=str`` so that a payload containing a numpy scalar, a timestamp or a
    ``Path`` serialises instead of raising. The pipeline's JSON is diagnostic,
    not an interchange format, and a run that dies at the end because its
    manifest contained an ``np.int64`` has lost real work for no reason.
    """
    write_text(uri, json.dumps(dict(payload), indent=2, default=str, sort_keys=True))


def read_json(uri: str) -> Any:
    """Read and parse a JSON object."""
    return json.loads(read_text(uri))


def write_marker(
    prefix: str,
    name: str,
    payload: Optional[Mapping[str, Any]] = None,
) -> str:
    """Write ``_<name>.json`` under ``prefix`` and return the URI written.

    ``marker`` and ``written_at_utc`` are added to the payload. The timestamp is
    the only record of *when* a stage finished, since S3 object times are not
    preserved across a copy.
    """
    body: Dict[str, Any] = dict(payload or {})
    body["marker"] = name
    body["written_at_utc"] = datetime.now(timezone.utc).isoformat()
    uri = join_uri(prefix, f"_{name}.json")
    write_json(uri, body)
    return uri


# --------------------------------------------------------------------------
# files and tables
# --------------------------------------------------------------------------

def upload_file(local_path: str, uri: str) -> None:
    """Copy a local file to ``uri``."""
    if is_s3_uri(uri):
        bucket, key = split_s3_uri(uri)
        try:
            s3_client().upload_file(str(local_path), bucket, key)
        except Exception as exc:  # noqa: BLE001
            raise ObjectStoreError(f"could not upload to {uri}: {exc}") from exc
        return

    parent = os.path.dirname(uri)
    if parent:
        os.makedirs(parent, exist_ok=True)
    shutil.copy2(str(local_path), uri)


def write_pandas(frame: "pd.DataFrame", uri: str, index: bool = False) -> None:
    """Write a pandas frame to ``uri`` as CSV or parquet, chosen by extension.

    The extension must be one of ``.csv``, ``.parquet`` or ``.pq``. The notebook
    version treated "not ``.parquet``" as "CSV", which wrote comma-separated text
    to a URI ending in ``.pq`` and to several ending in no extension at all;
    whatever read those back got one column named after the whole header row.
    """
    suffix = os.path.splitext(str(uri))[1].lower()
    if suffix in (".parquet", ".pq"):
        filename = "data.parquet"

        def writer(path: str) -> None:
            frame.to_parquet(path, index=index)

    elif suffix == ".csv":
        filename = "data.csv"

        def writer(path: str) -> None:
            frame.to_csv(path, index=index)

    else:
        raise ValueError(
            f"cannot infer a format for {uri!r}; the extension must be "
            ".csv, .parquet or .pq"
        )

    scratch = tempfile.mkdtemp(prefix="ts05_write_")
    try:
        local = os.path.join(scratch, filename)
        writer(local)
        upload_file(local, uri)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# --------------------------------------------------------------------------
# listing
# --------------------------------------------------------------------------

def list_common_prefixes(prefix_uri: str, delimiter: str = "/") -> List[str]:
    """The immediate "subdirectories" of ``prefix_uri``, as full URIs.

    Used to discover partition values without knowing them — which fraud ``dt=``
    partitions exist, which ``run_id=`` directories a model has — so it must
    paginate. A model that has been swept over a hundred configurations has more
    than the thousand keys a single ``list_objects_v2`` page returns.
    """
    if not is_s3_uri(prefix_uri):
        root = str(prefix_uri).rstrip("/")
        if not os.path.isdir(root):
            return []
        return sorted(
            os.path.join(root, name) + "/"
            for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        )

    bucket, key = split_s3_uri(join_uri(prefix_uri) + "/")
    client = s3_client()
    found: List[str] = []
    token: Optional[str] = None
    while True:
        kwargs: Dict[str, Any] = {
            "Bucket": bucket,
            "Prefix": key,
            "Delimiter": delimiter,
        }
        if token:
            kwargs["ContinuationToken"] = token
        try:
            response = client.list_objects_v2(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ObjectStoreError(f"could not list {prefix_uri}: {exc}") from exc
        for entry in response.get("CommonPrefixes", []):
            found.append(f"{_S3_SCHEME}{bucket}/{entry['Prefix']}")
        if not response.get("IsTruncated"):
            break
        token = response.get("NextContinuationToken")
    return found


def list_objects(prefix_uri: str, suffix: Optional[str] = None) -> List[str]:
    """Every object under ``prefix_uri``, optionally filtered by key suffix.

    A full recursive walk, which is what the cross-model comparison needs: it has
    no index of the runs that exist and finds them by looking for every key ending
    in a particular metrics filename. Expensive, and the caller should pass the
    narrowest prefix it can.
    """
    if not is_s3_uri(prefix_uri):
        root = str(prefix_uri).rstrip("/")
        out: List[str] = []
        for directory, _subdirs, files in os.walk(root):
            for name in files:
                path = os.path.join(directory, name)
                if suffix is None or path.endswith(suffix):
                    out.append(path)
        return sorted(out)

    bucket, key = split_s3_uri(join_uri(prefix_uri) + "/")
    client = s3_client()
    keys: List[str] = []
    token: Optional[str] = None
    while True:
        kwargs: Dict[str, Any] = {"Bucket": bucket, "Prefix": key}
        if token:
            kwargs["ContinuationToken"] = token
        try:
            response = client.list_objects_v2(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ObjectStoreError(f"could not list {prefix_uri}: {exc}") from exc
        for entry in response.get("Contents", []):
            entry_key = entry["Key"]
            if suffix is None or entry_key.endswith(suffix):
                keys.append(f"{_S3_SCHEME}{bucket}/{entry_key}")
        if not response.get("IsTruncated"):
            break
        token = response.get("NextContinuationToken")
    return keys
