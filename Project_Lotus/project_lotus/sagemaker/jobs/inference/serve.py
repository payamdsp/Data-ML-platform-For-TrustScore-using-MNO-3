"""SageMaker inference server (Flask): score customers on request with the champion.

SageMaker's hosting contract, which a Serverless or real-time endpoint calls:

    GET  /ping          200 when the server is up (health check)
    POST /invocations   score the rows in the request body

It initialises the same code the nightly batch job uses - ``inference.py``'s
``load_champion`` and ``check_batch`` and the package's ``score_population`` -
so a score from the endpoint is identical to the score the same row gets in the
nightly run (``sagemaker/tests/smoke_local.py`` checks exactly that).

Request formats (``Content-Type``):

* ``application/json``: a list of records ``[{"customer_id": ..., "<feature>": ...}, ...]``,
  or ``{"instances": [records]}``, or pandas-split ``{"columns": [...], "data": [[...]]}``.
* ``text/csv``: a header row, then one row per record.

Every record needs ``customer_id`` and every feature the champion uses (extra
columns are ignored; a missing feature is a 400, never silently imputed).
Response (``application/json``, or ``text/csv`` when ``Accept: text/csv``)::

    {"champion_run_id": ..., "model": ..., "n": 2,
     "predictions": [{"customer_id": "c1", "anomaly_score": 12.3,
                      "training_percentile": 0.997, "above_training_p99": true}, ...]}

``training_percentile`` places the score within the champion's *training* score
distribution (the drift reference train.py stores), so it is comparable across
requests - unlike a percentile within one small request.

The champion is read from ``champion/current.json`` on first use and re-checked
every ``TS05_CHAMPION_REFRESH_SECONDS``: when training promotes a new champion,
the endpoint switches to it without a redeploy. ``/ping`` answers 200 even before
the first champion exists, so the endpoint can be created before the first
training run; ``/invocations`` then returns 503 until a champion is published.

Environment (all optional):
    TS05_CONFIG_FILES               comma-separated, default conf/ml/base.yaml,conf/ml/lotus_sandbox.yaml
    TS05_CONFIG_OVERRIDES           ';'-separated KEY=VALUE overrides
    TS05_CHAMPION_URI               score with this champion.json instead of current.json
    TS05_CHAMPION_REFRESH_SECONDS   default 300
    TS05_MAX_ROWS_PER_REQUEST       default 10000

Run locally for a demo:  python serve.py --port 8080   (Flask's own server)
In the container:         gunicorn, started by container_entry.py when SageMaker passes `serve`.
"""

from __future__ import annotations

import argparse
import io
import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sm_common  # noqa: F401 - puts PROGRAM_DIR on sys.path before the package import
from sm_common import configure, read_parquet_any, resolve_configs

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from flask import Flask, Response, jsonify, request  # noqa: E402

from inference import ScoringInputError, check_batch, load_champion  # noqa: E402
from trust_score_05.common.io.serialization import load_bundle  # noqa: E402
from trust_score_05.common.splitting import CUSTOMER_KEY_COLUMN  # noqa: E402
from trust_score_05.lineage.config import ConfigError  # noqa: E402
from trust_score_05.lineage.logging_utils import get_logger  # noqa: E402
from trust_score_05.ml.config import MLConfig, load_ml_config  # noqa: E402
from trust_score_05.ml.evaluation import SCORE_COLUMN, score_population  # noqa: E402

LOGGER = get_logger("ts05.sagemaker.serve")

DEFAULT_SERVING_CONFIGS = "conf/ml/base.yaml,conf/ml/sandbox.yaml"


class NoChampionError(RuntimeError):
    """No champion has been published yet. Answered with 503."""


# --------------------------------------------------------------------------
# the champion, loaded once and refreshed when training promotes a new one
# --------------------------------------------------------------------------

class ChampionCache:
    """Holds the loaded champion and swaps it when ``current.json`` changes.

    The check is one small JSON read per ``refresh_seconds``; the bundle and
    the reference scores are only re-read when the champion's run id or bundle
    changed. The swap happens under a lock, so a request is always scored by
    one complete champion, never half of an old one and half of a new one.
    """

    def __init__(self, cfg: MLConfig, champion_uri: Optional[str], refresh_seconds: float):
        self.cfg = cfg
        self.champion_uri = champion_uri
        self.refresh_seconds = float(refresh_seconds)
        self._lock = threading.Lock()
        self._checked_at = 0.0
        self.champion: Optional[Dict[str, Any]] = None
        self.bundle: Any = None
        self.reference_sorted: Optional[np.ndarray] = None
        self.train_p99: Optional[float] = None

    def _identity(self, champion: Dict[str, Any]) -> Tuple[str, str]:
        return str(champion.get("run_id")), str(champion.get("bundle_uri"))

    def get(self):
        now = time.time()
        if self.champion is None or now - self._checked_at >= self.refresh_seconds:
            with self._lock:
                if self.champion is None or now - self._checked_at >= self.refresh_seconds:
                    self._refresh()
                    self._checked_at = time.time()
        if self.champion is None:
            raise NoChampionError("no champion has been published yet; run the training pipeline")
        return self.champion, self.bundle, self.reference_sorted, self.train_p99

    def _refresh(self) -> None:
        try:
            champion = load_champion(self.cfg, self.champion_uri)
        except ConfigError as exc:
            if self.champion is None:
                LOGGER.warning("no champion available yet: %s", exc)
                return
            # Keep serving the champion already loaded; a transient read failure
            # must not take a working endpoint down.
            LOGGER.warning("champion re-check failed, keeping %s: %s", self.champion["run_id"], exc)
            return
        if self.champion is not None and self._identity(champion) == self._identity(self.champion):
            return

        bundle = load_bundle(champion["bundle_uri"])
        if list(bundle.features) != list(champion["features"]):
            raise ConfigError("the champion document and its bundle disagree on the feature list")
        reference = read_parquet_any(champion["reference_scores_uri"])[SCORE_COLUMN].to_numpy(float)
        reference = np.sort(reference[np.isfinite(reference)])

        self.champion, self.bundle = champion, bundle
        self.reference_sorted = reference
        self.train_p99 = float(np.quantile(reference, 0.99)) if reference.size else None
        LOGGER.info("serving champion run_id=%s model=%s (%d features)",
                    champion["run_id"], champion.get("model"), len(bundle.features))


# --------------------------------------------------------------------------
# request parsing and response building
# --------------------------------------------------------------------------

def parse_request(body: bytes, content_type: str) -> pd.DataFrame:
    kind = (content_type or "application/json").split(";")[0].strip().lower()
    if kind in ("text/csv", "application/csv"):
        return pd.read_csv(io.BytesIO(body), dtype={CUSTOMER_KEY_COLUMN: str})
    if kind == "application/json":
        import json

        payload = json.loads(body.decode("utf-8") or "null")
        if isinstance(payload, dict) and "instances" in payload:
            payload = payload["instances"]
        if isinstance(payload, dict) and {"columns", "data"} <= set(payload):
            return pd.DataFrame(payload["data"], columns=payload["columns"])
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            raise ScoringInputError("JSON body must be a record, a list of records, "
                                    "{'instances': [...]} or {'columns': [...], 'data': [...]}")
        frame = pd.DataFrame(payload)
        if CUSTOMER_KEY_COLUMN in frame.columns:
            frame[CUSTOMER_KEY_COLUMN] = frame[CUSTOMER_KEY_COLUMN].astype(str)
        return frame
    raise ScoringInputError(f"unsupported Content-Type {content_type!r}; use application/json or text/csv")


def score_rows(frame: pd.DataFrame, bundle: Any, reference_sorted: np.ndarray,
               train_p99: Optional[float]) -> pd.DataFrame:
    """Score with the exact code path the nightly batch uses."""
    # Feature columns arrive from JSON as objects; the preprocessor expects numbers.
    features = list(bundle.features)
    frame = frame.copy()
    frame[features] = frame[features].apply(pd.to_numeric, errors="coerce")
    scored = score_population(frame, bundle.preprocessor, bundle.model)
    scores = scored[SCORE_COLUMN].to_numpy(dtype=float)
    out = pd.DataFrame({
        CUSTOMER_KEY_COLUMN: frame[CUSTOMER_KEY_COLUMN].astype(str).to_numpy(),
        SCORE_COLUMN: scores,
    })
    if reference_sorted is not None and reference_sorted.size:
        out["training_percentile"] = (
            np.searchsorted(reference_sorted, scores, side="right") / reference_sorted.size
        )
    if train_p99 is not None:
        out["above_training_p99"] = scores > train_p99
    return out


def _error(status: int, message: str) -> Response:
    response = jsonify({"error": message})
    response.status_code = status
    return response


# --------------------------------------------------------------------------
# the app
# --------------------------------------------------------------------------

def create_app(config_files: Optional[Sequence[str]] = None,
               overrides: Optional[Sequence[str]] = None,
               champion_uri: Optional[str] = None,
               refresh_seconds: Optional[float] = None,
               max_rows: Optional[int] = None) -> Flask:
    """Build the Flask app. Arguments default to the TS05_* environment variables."""
    configure(os.environ.get("TS05_LOG_LEVEL", "INFO"))
    files = list(config_files or os.environ.get("TS05_CONFIG_FILES", DEFAULT_SERVING_CONFIGS).split(","))
    sets = list(overrides if overrides is not None else
                [o for o in os.environ.get("TS05_CONFIG_OVERRIDES", "").split(";") if o.strip()])
    cfg = load_ml_config(config_paths=resolve_configs([f.strip() for f in files if f.strip()]),
                         overrides=sets)
    cache = ChampionCache(
        cfg,
        champion_uri or os.environ.get("TS05_CHAMPION_URI") or None,
        refresh_seconds if refresh_seconds is not None
        else float(os.environ.get("TS05_CHAMPION_REFRESH_SECONDS", "300")),
    )
    limit = int(max_rows if max_rows is not None else os.environ.get("TS05_MAX_ROWS_PER_REQUEST", "10000"))

    app = Flask("ts05-inference")
    app.config["TS05_CACHE"] = cache

    @app.get("/ping")
    def ping() -> Response:
        # Healthy as soon as the server is up: the endpoint can be created before
        # the first champion exists. Whether one is loaded is reported, not gated.
        return jsonify({"status": "ok", "champion_loaded": cache.champion is not None,
                        "champion_run_id": (cache.champion or {}).get("run_id")})

    @app.post("/invocations")
    def invocations() -> Response:
        started = time.time()
        try:
            champion, bundle, reference_sorted, train_p99 = cache.get()
        except NoChampionError as exc:
            LOGGER.warning("champion unavailable: %s", exc, exc_info=True)
            return _error(503, "champion is not available yet; please retry shortly")
        except Exception as exc:  # noqa: BLE001 - loading the model failed
            LOGGER.error("could not load the champion: %s", exc, exc_info=True)
            return _error(500, "internal server error")

        try:
            frame = parse_request(request.get_data(), request.content_type or "")
            if len(frame) > limit:
                return _error(413, f"{len(frame)} rows in one request; the limit is {limit}. "
                                   "Use the nightly batch pipeline for large volumes.")
            check_batch(frame, bundle.features)
        except (ScoringInputError, ValueError) as exc:
            LOGGER.info("invalid invocation request: %s", exc, exc_info=True)
            return _error(400, "invalid request payload")

        scored = score_rows(frame, bundle, reference_sorted, train_p99)
        LOGGER.info("scored %d row(s) in %.0f ms", len(scored), (time.time() - started) * 1000)

        if "text/csv" in (request.headers.get("Accept") or ""):
            return Response(scored.to_csv(index=False), mimetype="text/csv")
        return jsonify({
            "champion_run_id": champion["run_id"],
            "model": champion.get("model"),
            "hyperparam_id": champion.get("hyperparam_id"),
            "n": int(len(scored)),
            "predictions": scored.to_dict(orient="records"),
        })

    return app


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the inference server locally (Flask's server).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--config", action="append", default=None)
    args = parser.parse_args(argv)
    create_app(config_files=args.config).run(host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
