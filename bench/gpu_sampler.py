"""
bench/gpu_sampler.py
====================
Phase A sidecar — periodic GPU telemetry sampler.

Polls ``nvidia-smi`` at a fixed interval and appends one JSON record per
sample to a JSONL file. Designed to be launched as an out-of-process
sidecar by ``bench/run_load_capture.py`` so that capture-client wall-clock
timing is not perturbed by sampling overhead.

Each record has the following shape::

    {
      "ts": "2026-05-04T19:42:13.123456+00:00",
      "ts_monotonic": 12345.678,
      "gpus": [
        {
          "index": 0,
          "name": "NVIDIA H100 80GB HBM3",
          "utilization_gpu_pct": 92.0,
          "utilization_memory_pct": 31.0,
          "memory_used_mib": 24536.0,
          "memory_total_mib": 81920.0,
          "power_draw_w": 410.5
        }
      ]
    }

Usage
-----
    python -m bench.gpu_sampler --output PATH [--interval 0.2]

The process loops until it is signalled (SIGINT/SIGTERM) or until
``--max-seconds`` elapses, whichever happens first.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


# ── Query definition ────────────────────────────────────────

# Order matters — must match column order requested from nvidia-smi.
_NVSMI_FIELDS = [
    "index",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
]

_NVSMI_QUERY = ",".join(_NVSMI_FIELDS)


@dataclass
class _GpuRow:
    """Parsed single-GPU telemetry row."""

    index: int
    name: str
    utilization_gpu_pct: Optional[float]
    utilization_memory_pct: Optional[float]
    memory_used_mib: Optional[float]
    memory_total_mib: Optional[float]
    power_draw_w: Optional[float]

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "utilization_gpu_pct": self.utilization_gpu_pct,
            "utilization_memory_pct": self.utilization_memory_pct,
            "memory_used_mib": self.memory_used_mib,
            "memory_total_mib": self.memory_total_mib,
            "power_draw_w": self.power_draw_w,
        }


def _parse_float(token: str) -> Optional[float]:
    """Parse one nvidia-smi cell, tolerating ``[Not Supported]`` etc."""
    text = token.strip()
    if not text:
        return None
    if text.lower().startswith("[not"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _query_once() -> List[_GpuRow]:
    """Run a single ``nvidia-smi`` query and return parsed rows."""
    cmd = [
        "nvidia-smi",
        f"--query-gpu={_NVSMI_QUERY}",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(
            cmd,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []

    rows: List[_GpuRow] = []
    for line in out.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) < len(_NVSMI_FIELDS):
            continue
        try:
            idx = int(cells[0])
        except ValueError:
            continue
        rows.append(
            _GpuRow(
                index=idx,
                name=cells[1],
                utilization_gpu_pct=_parse_float(cells[2]),
                utilization_memory_pct=_parse_float(cells[3]),
                memory_used_mib=_parse_float(cells[4]),
                memory_total_mib=_parse_float(cells[5]),
                power_draw_w=_parse_float(cells[6]),
            )
        )
    return rows


# ── Sampling loop ───────────────────────────────────────────

def sample_loop(
    output: Path,
    interval_s: float,
    max_seconds: Optional[float],
) -> int:
    """Run the sampling loop and return the number of records written.

    The loop terminates on SIGINT/SIGTERM or when ``max_seconds`` elapse.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    stop = {"flag": False}

    def _handle(_signum, _frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    n_written = 0
    t0 = time.monotonic()

    with output.open("w", encoding="utf-8") as fh:
        while not stop["flag"]:
            t_mono = time.monotonic()
            if max_seconds is not None and (t_mono - t0) > max_seconds:
                break

            rows = _query_once()
            record = {
                "ts": datetime.now(tz=timezone.utc).isoformat(),
                "ts_monotonic": t_mono,
                "gpus": [r.to_dict() for r in rows],
            }
            fh.write(json.dumps(record, default=str) + "\n")
            fh.flush()
            n_written += 1

            # Sleep with frequent stop checks for snappy shutdown.
            slept = 0.0
            step = min(0.05, interval_s)
            while slept < interval_s and not stop["flag"]:
                time.sleep(step)
                slept += step

    return n_written


# ── CLI ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="GPU telemetry sampler (Phase A sidecar)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--interval", type=float, default=0.2,
                        help="Sampling interval in seconds (default: 0.2)")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Optional time cap; the process exits cleanly when reached")
    args = parser.parse_args()

    setup_logging()
    logger.info(
        "GPU sampler starting",
        extra={"output": args.output, "interval_s": args.interval},
    )
    n = sample_loop(
        output=Path(args.output),
        interval_s=float(args.interval),
        max_seconds=float(args.max_seconds) if args.max_seconds is not None else None,
    )
    logger.info("GPU sampler finished", extra={"records": n})
    sys.exit(0)


if __name__ == "__main__":
    main()
