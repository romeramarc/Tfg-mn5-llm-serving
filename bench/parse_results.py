"""
bench/parse_results.py
======================
Parse raw output from ``vllm bench serve`` (or the legacy
``benchmark_serving.py``) and normalise it into a structured dict.

The parser handles both the JSON output format (``--result-filename``)
and the older text-based stdout format.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


def parse_json_result(path: str | Path) -> Dict[str, Any]:
    """Load a JSON results file emitted by ``vllm bench serve``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Result file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return _normalise(data)


def parse_stdout(text: str) -> Dict[str, Any]:
    """Extract key–value pairs from the text stdout of vllm bench."""
    results: Dict[str, Any] = {}

    patterns = {
        "completed_requests": r"Successful requests:\s+(\d+)",
        "total_time_s": r"Benchmark duration \(s\):\s+([\d.]+)",
        "request_throughput_rps": r"Request throughput \(req/s\):\s+([\d.]+)",
        "input_throughput_tps": r"Input token throughput \(tok/s\):\s+([\d.]+)",
        "output_throughput_tps": r"Output token throughput \(tok/s\):\s+([\d.]+)",
        "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([\d.]+)",
        "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([\d.]+)",
        "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([\d.]+)",
        "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([\d.]+)",
        "mean_itl_ms": r"Mean ITL \(ms\):\s+([\d.]+)",
        "p99_itl_ms": r"P99 ITL \(ms\):\s+([\d.]+)",
    }

    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            val = m.group(1)
            results[key] = int(val) if val.isdigit() else float(val)

    return results


# ── Internal helpers ────────────────────────────────────────

def _normalise(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Map known vllm-bench JSON keys to a consistent schema."""
    out: Dict[str, Any] = {}
    key_map = {
        "completed": "completed_requests",
        "duration": "total_time_s",
        "request_throughput": "request_throughput_rps",
        "input_throughput": "input_throughput_tps",
        "output_throughput": "output_throughput_tps",
        "mean_ttft_ms": "mean_ttft_ms",
        "p99_ttft_ms": "p99_ttft_ms",
        "mean_tpot_ms": "mean_tpot_ms",
        "p99_tpot_ms": "p99_tpot_ms",
        "mean_itl_ms": "mean_itl_ms",
        "p99_itl_ms": "p99_itl_ms",
    }
    for src, dst in key_map.items():
        if src in raw:
            out[dst] = raw[src]
    # Pass through anything not mapped
    for k, v in raw.items():
        if k not in key_map:
            out[k] = v
    return out

def main() -> None:
    parser = argparse.ArgumentParser(description="Offline throughput benchmark")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
