#!/usr/bin/env python3
"""
scripts/collect_all_results.py
==============================
Scan the results/ directory tree and aggregate quality + efficiency
metrics into a single CSV for plotting and analysis.

Output: results/summary_all_models.csv

Usage:
    python scripts/collect_all_results.py
    python scripts/collect_all_results.py --results-dir results/ --output results/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_role_from_dir(dirname: str) -> str:
    """Extract the role tag from a run directory name.

    E.g. 'quality-teacher-20260307T110225Z' -> 'teacher'
         'quality-distilled_student_mid-20260309T083140Z' -> 'distilled_student_mid'
    """
    parts = dirname.split("-")
    if len(parts) >= 3:
        ts = parts[-1]
        if len(ts) >= 15 and ts[0] == "2":
            role = "-".join(parts[1:-1])
            return role
    return dirname


def collect_quality(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Collect quality metrics grouped by role."""
    quality_dir = results_dir / "quality"
    if not quality_dir.exists():
        return {}

    role_metrics: Dict[str, Dict[str, Any]] = {}

    for run_dir in sorted(quality_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        summary_file = run_dir / "quality_summary.json"
        if not summary_file.exists():
            continue

        role = _extract_role_from_dir(run_dir.name)
        data = _load_json(summary_file)

        entry: Dict[str, Any] = {"role": role, "quality_run": run_dir.name}
        if isinstance(data, list):
            for bench in data:
                bname = bench.get("benchmark", "unknown")
                entry[f"{bname}_acc"] = bench.get("accuracy_pct", bench.get("accuracy", 0) * 100)
                entry[f"{bname}_correct"] = bench.get("correct", 0)
                total_examples = bench.get(
                    "total_examples",
                    bench.get("scorable_examples", 0) + bench.get("unscorable_examples", 0),
                )
                scorable_examples = bench.get("scorable_examples", total_examples)
                unscorable_examples = bench.get(
                    "unscorable_examples",
                    max(0, total_examples - scorable_examples),
                )
                entry[f"{bname}_total"] = total_examples
                entry[f"{bname}_scorable"] = scorable_examples
                entry[f"{bname}_unscorable"] = unscorable_examples
                entry["model"] = bench.get("model", "")

        role_metrics[role] = entry

    return role_metrics


def collect_throughput(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Collect throughput metrics grouped by role (last run per role)."""
    tp_dir = results_dir / "throughput"
    if not tp_dir.exists():
        return {}

    role_metrics: Dict[str, Dict[str, Any]] = {}

    for run_dir in sorted(tp_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        result_file = run_dir / "throughput_results.json"
        if not result_file.exists():
            continue

        role = _extract_role_from_dir(run_dir.name)
        data = _load_json(result_file)

        entry: Dict[str, Any] = {"role": role, "throughput_run": run_dir.name}
        for key in ["request_throughput_rps", "output_throughput_tps",
                     "mean_ttft_ms", "p99_ttft_ms",
                     "mean_tpot_ms", "p99_tpot_ms",
                     "mean_itl_ms", "p99_itl_ms",
                     "completed_requests", "total_time_s"]:
            if key in data:
                entry[key] = data[key]

        role_metrics[role] = entry

    return role_metrics


def collect_online(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Collect online load metrics grouped by role (last run per role)."""
    online_dir = results_dir / "online"
    if not online_dir.exists():
        return {}

    role_metrics: Dict[str, Dict[str, Any]] = {}

    for run_dir in sorted(online_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        result_file = run_dir / "online_results.json"
        if not result_file.exists():
            continue

        role = _extract_role_from_dir(run_dir.name)
        data = _load_json(result_file)

        if isinstance(data, list) and data:
            entry: Dict[str, Any] = {"role": role, "online_run": run_dir.name}
            first = data[0]
            for key in ["request_rate", "successful_requests",
                         "effective_throughput_tps",
                         "latency_mean_ms", "latency_p50_ms",
                         "latency_p95_ms", "latency_p99_ms",
                         "ttfb_mean_ms", "ttfb_p50_ms",
                         "ttfb_p95_ms", "ttfb_p99_ms"]:
                if key in first:
                    entry[f"online_{key}"] = first[key]
            role_metrics[role] = entry

    return role_metrics


def merge_results(
    quality: Dict[str, Dict],
    throughput: Dict[str, Dict],
    online: Dict[str, Dict],
) -> List[Dict[str, Any]]:
    """Merge quality, throughput, and online metrics by role."""
    all_roles = set(quality.keys()) | set(throughput.keys()) | set(online.keys())
    rows = []

    for role in sorted(all_roles):
        row: Dict[str, Any] = {"role": role}
        if role in quality:
            row.update(quality[role])
        if role in throughput:
            row.update(throughput[role])
        if role in online:
            row.update(online[role])
        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect and aggregate all experiment results into CSV",
    )
    parser.add_argument("--results-dir", default="results",
                        help="Root results directory (default: results/)")
    parser.add_argument("--output", default="results/summary_all_models.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: {results_dir} does not exist")
        return

    print(f"Scanning {results_dir}/ ...")
    quality = collect_quality(results_dir)
    throughput = collect_throughput(results_dir)
    online = collect_online(results_dir)

    print(f"  Quality runs:    {len(quality)}")
    print(f"  Throughput runs: {len(throughput)}")
    print(f"  Online runs:     {len(online)}")

    rows = merge_results(quality, throughput, online)
    if not rows:
        print("No results found.")
        return

    all_fields = []
    for r in rows:
        for k in r:
            if k not in all_fields:
                all_fields.append(k)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSummary written to {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
