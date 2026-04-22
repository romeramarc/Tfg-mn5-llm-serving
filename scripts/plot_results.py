#!/usr/bin/env python3
"""
scripts/plot_results.py
=======================
Generate publication-quality plots for the TFG thesis.

Produces four figures:
  1. kd_impact_bars.pdf      — Pre vs Post KD accuracy (GSM8K & MATH-500)
  2. quality_vs_throughput.pdf — Scatter: throughput vs accuracy
  3. quality_vs_latency.pdf   — Scatter: latency (TPOT) vs accuracy
  4. all_models_overview.pdf  — Bar chart of all model configurations

Usage (LOCAL — after ``collect_all_results.py`` and pulling from BSC):
    python scripts/collect_all_results.py
    python scripts/plot_results.py --from-summary results/summary_all_models.csv --output plots/
    python scripts/plot_results.py
    python scripts/plot_results.py --input custom.csv --output plots/

If no ``--from-summary`` / ``--input``, uses the embedded MODELS list below.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ═══════════════════════════════════════════════════════════════
# DATA — Update after each experiment run
# ═══════════════════════════════════════════════════════════════
# "v1" = definitive pipeline (Mar 9 2026, BEFORE collator fix)
# "v2" = improved pipeline   (after collator fix + MATH data)
#
# After running the v2 pipeline on BSC, add new entries with
# version="v2" and the script will automatically show both.
# ═══════════════════════════════════════════════════════════════

MODELS: List[Dict[str, Any]] = [
    # ── Baselines (unchanged across versions) ─────────────
    {
        "name": "Teacher 14B", "short": "14B", "params": 14,
        "state": "baseline", "version": "v1",
        "gsm8k": 92.41, "math": 72.11,
        "throughput_rps": 12.29, "throughput_tps": 3146,
        "tpot_ms": 71.01, "p99_tpot_ms": 104.52,
        "ttft_ms": None,
    },
    {
        "name": "Student 7B", "short": "7B", "params": 7,
        "state": "pre-KD", "version": "v1",
        "gsm8k": 85.78, "math": 71.49,
        "throughput_rps": 25.08, "throughput_tps": 6421,
        "tpot_ms": 37.76, "p99_tpot_ms": 38.57,
        "ttft_ms": None,
    },
    {
        "name": "Student 1.5B", "short": "1.5B", "params": 1.5,
        "state": "pre-KD", "version": "v1",
        "gsm8k": 64.05, "math": 36.11,
        "throughput_rps": 50.26, "throughput_tps": 12866,
        "tpot_ms": 18.59, "p99_tpot_ms": 19.34,
        "ttft_ms": None,
    },
    # ── v1 distilled (collator bug — kept for comparison) ─
    {
        "name": "7B KD-v1", "short": "7B-KDv1", "params": 7,
        "state": "post-KD", "version": "v1",
        "gsm8k": 79.83, "math": 68.35,
        "throughput_rps": 25.58, "throughput_tps": 6549,
        "tpot_ms": 37.82, "p99_tpot_ms": 38.45,
        "ttft_ms": None,
    },
    {
        "name": "1.5B KD-v1", "short": "1.5B-KDv1", "params": 1.5,
        "state": "post-KD", "version": "v1",
        "gsm8k": 64.11, "math": 43.87,
        "throughput_rps": 49.86, "throughput_tps": 12765,
        "tpot_ms": 18.81, "p99_tpot_ms": 20.09,
        "ttft_ms": None,
    },
    # ── v2 distilled — PLACEHOLDER (update after BSC run) ─
    # {
    #     "name": "7B KD-v2", "short": "7B-KDv2", "params": 7,
    #     "state": "post-KD", "version": "v2",
    #     "gsm8k": ???, "math": ???,
    #     "throughput_rps": ???, "throughput_tps": ???,
    #     "tpot_ms": ???, "p99_tpot_ms": ???,
    #     "ttft_ms": ???,
    # },
    # {
    #     "name": "1.5B KD-v2", "short": "1.5B-KDv2", "params": 1.5,
    #     "state": "post-KD", "version": "v2",
    #     "gsm8k": ???, "math": ???,
    #     "throughput_rps": ???, "throughput_tps": ???,
    #     "tpot_ms": ???, "p99_tpot_ms": ???,
    #     "ttft_ms": ???,
    # },
]


# ═══════════════════════════════════════════════════════════════
# STYLE
# ═══════════════════════════════════════════════════════════════

COLOR = {
    "teacher": "#2563EB",
    "7B_pre": "#16A34A",
    "7B_post": "#86EFAC",
    "1.5B_pre": "#DC2626",
    "1.5B_post": "#FCA5A5",
    "gsm8k": "#2563EB",
    "math": "#F97316",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ═══════════════════════════════════════════════════════════════
# CSV LOADER
# ═══════════════════════════════════════════════════════════════

def load_csv(path: str) -> List[Dict[str, Any]]:
    """Load model data from a CSV file."""
    models = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            m: Dict[str, Any] = {}
            for k, v in row.items():
                try:
                    m[k] = float(v)
                except (ValueError, TypeError):
                    m[k] = v
            models.append(m)
    return models


# Roles produced by scripts/collect_all_results.py → plot MODELS schema
_SUMMARY_ROLE_MAP: Dict[str, Dict[str, Any]] = {
    "teacher": {
        "name": "Teacher 14B", "short": "14B", "state": "baseline",
        "params": 14, "version": "",
    },
    "student_mid": {
        "name": "Student 7B", "short": "7B", "state": "pre-KD",
        "params": 7, "version": "",
    },
    "student_small": {
        "name": "Student 1.5B", "short": "1.5B", "state": "pre-KD",
        "params": 1.5, "version": "",
    },
    "distilled_student_mid": {
        "name": "7B post-KD", "short": "7B KD", "state": "post-KD",
        "params": 7, "version": "v2",
    },
    "distilled_student_small": {
        "name": "1.5B post-KD", "short": "1.5B KD", "state": "post-KD",
        "params": 1.5, "version": "v2",
    },
    "distilled_student_small_v6": {
        "name": "1.5B post-KD (Exp. 6)", "short": "1.5B KD v6", "state": "post-KD",
        "params": 1.5, "version": "v6",
    },
}


def load_summary_all_models(path: Path) -> List[Dict[str, Any]]:
    """Map ``results/summary_all_models.csv`` to the MODELS list schema."""
    models: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            role = (row.get("role") or "").strip()
            if role not in _SUMMARY_ROLE_MAP:
                continue
            gsm_raw, math_raw = row.get("gsm8k_acc"), row.get("math_acc")
            if gsm_raw in (None, "") or math_raw in (None, ""):
                continue
            try:
                gsm8k = float(gsm_raw)
                math_v = float(math_raw)
            except (TypeError, ValueError):
                continue
            meta = dict(_SUMMARY_ROLE_MAP[role])
            m: Dict[str, Any] = {
                **meta,
                "gsm8k": gsm8k,
                "math": math_v,
            }
            for col, key in [
                ("request_throughput_rps", "throughput_rps"),
                ("output_throughput_tps", "throughput_tps"),
                ("mean_tpot_ms", "tpot_ms"),
            ]:
                v = row.get(col)
                if v not in (None, ""):
                    try:
                        m[key] = float(v)
                    except (TypeError, ValueError):
                        pass
            models.append(m)
    return models


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: KD Impact Bar Chart
# ═══════════════════════════════════════════════════════════════

def plot_kd_impact(models: List[Dict], out: Path) -> None:
    """Grouped bar chart: pre-KD vs post-KD for 7B and 1.5B."""
    pre_7b = next((m for m in models if "7B" in m.get("short", "") and m["state"] == "pre-KD"), None)
    post_7b = [m for m in models if "7B" in m.get("short", "") and m["state"] == "post-KD"]
    pre_15 = next((m for m in models if "1.5B" in m.get("short", "") and m["state"] == "pre-KD"), None)
    post_15 = [m for m in models if "1.5B" in m.get("short", "") and m["state"] == "post-KD"]
    teacher = next((m for m in models if m["state"] == "baseline"), None)

    if not (pre_7b and post_7b and pre_15 and post_15):
        print("  [SKIP] kd_impact — insufficient data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    benchmarks = ["gsm8k", "math"]
    bench_labels = ["GSM8K", "MATH-500"]

    for ax, bench, blabel in zip(axes, benchmarks, bench_labels):
        labels = []
        pre_vals = []
        post_vals_list = []

        for pre, posts, tag in [(pre_7b, post_7b, "7B"), (pre_15, post_15, "1.5B")]:
            for p in posts:
                ver = p.get("version", "")
                labels.append(f"{tag} {ver}" if ver else tag)
                pre_vals.append(pre[bench])
                post_vals_list.append(p[bench])

        x = np.arange(len(labels))
        w = 0.35

        bars_pre = ax.bar(x - w / 2, pre_vals, w, label="Pre-KD",
                          color=COLOR["gsm8k"], alpha=0.7, edgecolor="white")
        bars_post = ax.bar(x + w / 2, post_vals_list, w, label="Post-KD",
                           color=COLOR["math"], alpha=0.7, edgecolor="white")

        for i, (pre_v, post_v) in enumerate(zip(pre_vals, post_vals_list)):
            delta = post_v - pre_v
            sign = "+" if delta >= 0 else ""
            color = "#16A34A" if delta >= 0 else "#DC2626"
            ax.annotate(f"{sign}{delta:.1f}pp",
                        xy=(x[i] + w / 2, post_v),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=9, fontweight="bold", color=color)

        if teacher:
            ax.axhline(y=teacher[bench], color=COLOR["teacher"],
                       linestyle="--", linewidth=1.5, alpha=0.6,
                       label=f"Teacher 14B ({teacher[bench]:.1f}%)")

        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy (%)" if ax == axes[0] else "")
        ax.set_title(blabel, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0, 105)
        ax.legend(loc="upper right", fontsize=9)

        for bar_group in [bars_pre, bars_post]:
            for bar in bar_group:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Knowledge Distillation Impact on Quality", fontweight="bold", fontsize=14)
    plt.tight_layout()
    path = out / "kd_impact_bars.pdf"
    fig.savefig(path)
    fig.savefig(out / "kd_impact_bars.png")
    plt.close(fig)
    print(f"  -> {path}")


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Quality vs Throughput
# ═══════════════════════════════════════════════════════════════

def plot_quality_vs_throughput(models: List[Dict], out: Path) -> None:
    """Scatter: throughput (req/s) vs average accuracy."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for m in models:
        if m.get("throughput_rps") is None:
            continue
        avg_acc = (m["gsm8k"] + m["math"]) / 2
        tput = m["throughput_rps"]
        params = m.get("params", 7)
        size = params * 25 + 40

        if m["state"] == "baseline":
            c, marker = COLOR["teacher"], "D"
        elif "7B" in m.get("short", ""):
            c = COLOR["7B_pre"] if m["state"] == "pre-KD" else COLOR["7B_post"]
            marker = "o" if m["state"] == "pre-KD" else "s"
        else:
            c = COLOR["1.5B_pre"] if m["state"] == "pre-KD" else COLOR["1.5B_post"]
            marker = "o" if m["state"] == "pre-KD" else "s"

        ax.scatter(tput, avg_acc, s=size, c=c, marker=marker,
                   edgecolors="black", linewidths=0.8, zorder=5)
        ax.annotate(m.get("short", m["name"]),
                    (tput, avg_acc), textcoords="offset points",
                    xytext=(8, 5), fontsize=9)

    # Arrows from pre-KD to post-KD
    for pre_key, post_models in [("7B", []), ("1.5B", [])]:
        pre = next((m for m in models if pre_key in m.get("short", "")
                     and m["state"] == "pre-KD"), None)
        posts = [m for m in models if pre_key in m.get("short", "")
                 and m["state"] == "post-KD"]
        if pre:
            for p in posts:
                if p.get("throughput_rps") is None:
                    continue
                pre_acc = (pre["gsm8k"] + pre["math"]) / 2
                post_acc = (p["gsm8k"] + p["math"]) / 2
                ax.annotate("",
                            xy=(p["throughput_rps"], post_acc),
                            xytext=(pre["throughput_rps"], pre_acc),
                            arrowprops=dict(arrowstyle="->", color="gray",
                                            lw=1.5, connectionstyle="arc3,rad=0.1"))

    ax.set_xlabel("Throughput (req/s)", fontweight="bold")
    ax.set_ylabel("Average Accuracy (GSM8K + MATH) %", fontweight="bold")
    ax.set_title("Quality vs Throughput — All Model Configurations", fontweight="bold")

    legend_elements = [
        mpatches.Patch(color=COLOR["teacher"], label="Teacher 14B"),
        mpatches.Patch(color=COLOR["7B_pre"], label="7B Pre-KD"),
        mpatches.Patch(color=COLOR["7B_post"], label="7B Post-KD"),
        mpatches.Patch(color=COLOR["1.5B_pre"], label="1.5B Pre-KD"),
        mpatches.Patch(color=COLOR["1.5B_post"], label="1.5B Post-KD"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    path = out / "quality_vs_throughput.pdf"
    fig.savefig(path)
    fig.savefig(out / "quality_vs_throughput.png")
    plt.close(fig)
    print(f"  -> {path}")


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Quality vs Latency
# ═══════════════════════════════════════════════════════════════

def plot_quality_vs_latency(models: List[Dict], out: Path) -> None:
    """Scatter: TPOT latency (ms) vs average accuracy."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for m in models:
        if m.get("tpot_ms") is None:
            continue
        avg_acc = (m["gsm8k"] + m["math"]) / 2
        lat = m["tpot_ms"]
        params = m.get("params", 7)
        size = params * 25 + 40

        if m["state"] == "baseline":
            c, marker = COLOR["teacher"], "D"
        elif "7B" in m.get("short", ""):
            c = COLOR["7B_pre"] if m["state"] == "pre-KD" else COLOR["7B_post"]
            marker = "o" if m["state"] == "pre-KD" else "s"
        else:
            c = COLOR["1.5B_pre"] if m["state"] == "pre-KD" else COLOR["1.5B_post"]
            marker = "o" if m["state"] == "pre-KD" else "s"

        ax.scatter(lat, avg_acc, s=size, c=c, marker=marker,
                   edgecolors="black", linewidths=0.8, zorder=5)
        ax.annotate(m.get("short", m["name"]),
                    (lat, avg_acc), textcoords="offset points",
                    xytext=(8, 5), fontsize=9)

    ax.set_xlabel("TPOT Latency (ms) — lower is better", fontweight="bold")
    ax.set_ylabel("Average Accuracy (GSM8K + MATH) %", fontweight="bold")
    ax.set_title("Quality vs Latency — All Model Configurations", fontweight="bold")
    ax.invert_xaxis()

    legend_elements = [
        mpatches.Patch(color=COLOR["teacher"], label="Teacher 14B"),
        mpatches.Patch(color=COLOR["7B_pre"], label="7B Pre-KD"),
        mpatches.Patch(color=COLOR["7B_post"], label="7B Post-KD"),
        mpatches.Patch(color=COLOR["1.5B_pre"], label="1.5B Pre-KD"),
        mpatches.Patch(color=COLOR["1.5B_post"], label="1.5B Post-KD"),
    ]
    ax.legend(handles=legend_elements, loc="lower left")

    plt.tight_layout()
    path = out / "quality_vs_latency.pdf"
    fig.savefig(path)
    fig.savefig(out / "quality_vs_latency.png")
    plt.close(fig)
    print(f"  -> {path}")


# ═══════════════════════════════════════════════════════════════
# FIGURE 4: All Models Overview
# ═══════════════════════════════════════════════════════════════

def plot_all_models(models: List[Dict], out: Path) -> None:
    """Bar chart comparing GSM8K and MATH-500 for every configuration."""
    fig, ax = plt.subplots(figsize=(14, 7))

    names = [m.get("short", m["name"]) for m in models]
    gsm = [m["gsm8k"] for m in models]
    math_vals = [m["math"] for m in models]

    x = np.arange(len(names))
    w = 0.35

    bars1 = ax.bar(x - w / 2, gsm, w, label="GSM8K", color=COLOR["gsm8k"], alpha=0.8)
    bars2 = ax.bar(x + w / 2, math_vals, w, label="MATH-500", color=COLOR["math"], alpha=0.8)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Model Configuration")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Quality Comparison — All Model Configurations", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0, 108)
    ax.legend(loc="upper right")

    plt.tight_layout()
    path = out / "all_models_overview.pdf"
    fig.savefig(path)
    fig.savefig(out / "all_models_overview.png")
    plt.close(fig)
    print(f"  -> {path}")


# ═══════════════════════════════════════════════════════════════
# FIGURE 5: System Metrics Comparison
# ═══════════════════════════════════════════════════════════════

def plot_system_metrics(models: List[Dict], out: Path) -> None:
    """Horizontal bar charts: req/s, tok/s (output), TPOT — same model order."""
    valid = [m for m in models if m.get("throughput_rps") is not None]
    if not valid:
        print("  [SKIP] system_metrics — no throughput data")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    names = [m.get("short", m["name"]) for m in valid]
    tput_rps = [m["throughput_rps"] for m in valid]
    tput_tps = [m.get("throughput_tps") for m in valid]
    tpot = [m["tpot_ms"] for m in valid]
    y = np.arange(len(names))

    colors = []
    for m in valid:
        if m["state"] == "baseline":
            colors.append(COLOR["teacher"])
        elif "7B" in m.get("short", ""):
            colors.append(COLOR["7B_pre"] if m["state"] == "pre-KD" else COLOR["7B_post"])
        else:
            colors.append(COLOR["1.5B_pre"] if m["state"] == "pre-KD" else COLOR["1.5B_post"])

    ax1.barh(y, tput_rps, color=colors, edgecolor="white", height=0.6)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names)
    ax1.set_xlabel("Throughput (req/s)")
    ax1.set_title("Requests / s", fontweight="bold")
    for i, v in enumerate(tput_rps):
        ax1.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=9)

    if all(x is not None for x in tput_tps):
        ax2.barh(y, tput_tps, color=colors, edgecolor="white", height=0.6)
        ax2.set_yticks(y)
        ax2.set_yticklabels(names)
        ax2.set_xlabel("Output throughput (tok/s)")
        ax2.set_title("Tokens / s", fontweight="bold")
        for i, v in enumerate(tput_tps):
            ax2.text(v + 50, i, f"{v:.0f}", va="center", fontsize=9)
    else:
        ax2.text(0.5, 0.5, "No tok/s in data", ha="center", transform=ax2.transAxes)
        ax2.axis("off")

    ax3.barh(y, tpot, color=colors, edgecolor="white", height=0.6)
    ax3.set_yticks(y)
    ax3.set_yticklabels(names)
    ax3.set_xlabel("TPOT (ms) — lower is better")
    ax3.set_title("Latency (TPOT)", fontweight="bold")
    for i, v in enumerate(tpot):
        ax3.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=9)

    fig.suptitle("System Performance Metrics", fontweight="bold", fontsize=14)
    plt.tight_layout()
    path = out / "system_metrics.pdf"
    fig.savefig(path)
    fig.savefig(out / "system_metrics.png")
    plt.close(fig)
    print(f"  -> {path}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate thesis plots from experiment results",
    )
    parser.add_argument("--input", default=None,
                        help="CSV file with model data (columns: name,short,params,"
                             "state,version,gsm8k,math,throughput_rps,...)")
    parser.add_argument("--from-summary", dest="from_summary", default=None,
                        metavar="CSV",
                        help="Use collect_all_results.py output (e.g. results/summary_all_models.csv)")
    parser.add_argument("--output", default="plots",
                        help="Output directory for plots (default: plots/)")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.from_summary:
        sp = Path(args.from_summary)
        models = load_summary_all_models(sp)
        print(f"Loaded {len(models)} models from summary CSV {sp}")
    elif args.input:
        models = load_csv(args.input)
        print(f"Loaded {len(models)} models from {args.input}")
    else:
        models = MODELS
        print(f"Using embedded data ({len(models)} models)")

    print(f"Output directory: {out}/\n")

    plot_kd_impact(models, out)
    plot_quality_vs_throughput(models, out)
    plot_quality_vs_latency(models, out)
    plot_all_models(models, out)
    plot_system_metrics(models, out)

    print(f"\nAll plots saved to {out}/")


if __name__ == "__main__":
    main()
