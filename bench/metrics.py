"""
bench/metrics.py
================
Metric computation and persistence utilities shared by the throughput
and online-load benchmarking scripts.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ── Percentile helper ───────────────────────────────────────

def percentile(data: Sequence[float], p: float) -> float:
    """Compute the *p*-th percentile (0–100) of *data* using linear
    interpolation between adjacent ranks."""
    if not data:
        return float("nan")
    xs = sorted(data)
    k = (len(xs) - 1) * p / 100.0
    f = int(math.floor(k))
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] * (c - k) + xs[c] * (k - f)


# ── Latency summary ────────────────────────────────────────

def summarise_latencies(latencies: Sequence[float]) -> Dict[str, float]:
    """Return a dict with count, mean, and p50 / p95 / p99."""
    if not latencies:
        return {}
    n = len(latencies)
    return {
        "count": n,
        "mean_ms": sum(latencies) / n,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
    }


# ── Persistence ─────────────────────────────────────────────

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: str | Path) -> Path:
    """Write *obj* as pretty-printed JSON."""
    p = Path(path)
    _ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return p


def save_csv(
    rows: List[Dict[str, Any]],
    path: str | Path,
    fieldnames: Optional[List[str]] = None,
) -> Path:
    """Write a list of dicts as a CSV file."""
    p = Path(path)
    _ensure_dir(p.parent)
    if not rows:
        p.touch()
        return p
    fields = fieldnames or list(rows[0].keys())
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return p
