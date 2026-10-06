"""Logging helpers.

EMR ships the driver's stdout/stderr to the ``enstream-lake-logs-<env>`` bucket,
so a plain stream handler is exactly what we want. The banner helpers keep the
step logs readable when you are scrolling through a Spot-interrupted run.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Mapping

_CONFIGURED = False
_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Attach a single stdout handler to the root logger (idempotent)."""
    global _CONFIGURED
    resolved = getattr(logging, str(level).upper(), logging.INFO)
    if _CONFIGURED:
        logging.getLogger().setLevel(resolved)
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))

    root = logging.getLogger()
    root.setLevel(resolved)
    root.addHandler(handler)

    # py4j is extremely chatty at INFO and drowns out everything useful.
    logging.getLogger("py4j").setLevel(logging.WARNING)
    logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_banner(logger: logging.Logger, title: str, width: int = 78) -> None:
    """Print a section header so job phases are easy to find in the logs."""
    logger.info("=" * width)
    logger.info(title)
    logger.info("=" * width)


def log_mapping(logger: logging.Logger, title: str, mapping: Mapping[str, Any]) -> None:
    """Print a small key/value block (job settings, run stats)."""
    logger.info("%s:", title)
    if not mapping:
        logger.info("    <empty>")
        return
    pad = max(len(str(k)) for k in mapping)
    for key, value in mapping.items():
        logger.info("    %-*s = %s", pad, key, value)
