"""
utils/logging.py
================
Structured logging setup used by every module in the repository.

* JSON format  → machine-parseable, compatible with log aggregation.
* Text format  → human-friendly for interactive debugging.

Usage
-----
    from utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("server started", extra={"port": 8000})
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

try:
    from pythonjsonlogger import jsonlogger
    _HAS_JSON_LOGGER = True
except ImportError:
    _HAS_JSON_LOGGER = False


# ── Internal formatter ──────────────────────────────────────

class _JsonFormatterFallback(logging.Formatter):
    """Minimal JSON formatter used when python-json-logger is absent."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge any extra keys the caller passed
        for key in ("job_id", "model", "config", "git_commit", "seed",
                     "latency", "throughput", "port", "pid"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val
        return json.dumps(payload, default=str)


# ── Public API ──────────────────────────────────────────────

_CONFIGURED = False


def setup_logging(
    level: str = "INFO",
    fmt: str = "json",
    log_dir: Optional[str] = None,
) -> None:
    """Configure the root logger once per process.

    Parameters
    ----------
    level : str
        Standard Python log level name (DEBUG, INFO, WARNING, …).
    fmt : str
        ``"json"`` for structured JSON lines, ``"text"`` for plain text.
    log_dir : str | None
        If given, a ``FileHandler`` is also attached writing to
        ``<log_dir>/<date>.log``.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    # Formatter
    if fmt == "json":
        if _HAS_JSON_LOGGER:
            formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s",
                rename_fields={"asctime": "ts", "levelname": "level", "name": "logger"},
            )
        else:
            formatter = _JsonFormatterFallback()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    # Stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    root.addHandler(sh)

    # Optional file handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        today = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
        fh = logging.FileHandler(os.path.join(log_dir, f"{today}.log"))
        fh.setFormatter(formatter)
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """Return a named child logger.

    Ensures ``setup_logging`` has been called at least once with defaults,
    so callers never see zero handlers.
    """
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
