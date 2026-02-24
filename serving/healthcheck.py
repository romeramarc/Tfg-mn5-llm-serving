"""
serving/healthcheck.py
======================
Lightweight health-check probe for the vLLM OpenAI-compatible server.

Intended usage inside SLURM jobs and CI pipelines:

    python -m serving.healthcheck --url http://localhost:8000

Exit code 0 means the server is ready; non-zero means it is not.
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx

from utils.logging import get_logger

logger = get_logger(__name__)

_MAX_RETRIES = 60          # total attempts
_RETRY_INTERVAL_S = 5.0    # seconds between probes


def probe(base_url: str, retries: int = _MAX_RETRIES,
          interval: float = _RETRY_INTERVAL_S) -> bool:
    """Block until the vLLM server at *base_url* answers ``/health``.

    Returns ``True`` when the server is healthy, ``False`` after all
    retries are exhausted.
    """
    health_url = f"{base_url.rstrip('/')}/health"
    for attempt in range(1, retries + 1):
        try:
            resp = httpx.get(health_url, timeout=10.0)
            if resp.status_code == 200:
                logger.info("Server healthy", extra={
                    "url": health_url, "attempt": attempt,
                })
                return True
        except httpx.RequestError:
            pass
        logger.info("Waiting for server …",
                     extra={"attempt": attempt, "url": health_url})
        time.sleep(interval)

    logger.error("Server did not become healthy",
                 extra={"url": health_url, "retries": retries})
    return False


# ── CLI entry-point ─────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM health-check probe")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Base URL of the vLLM server")
    parser.add_argument("--retries", type=int, default=_MAX_RETRIES)
    parser.add_argument("--interval", type=float, default=_RETRY_INTERVAL_S)
    args = parser.parse_args()

    ok = probe(args.url, retries=args.retries, interval=args.interval)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
