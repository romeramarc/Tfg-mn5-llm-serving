"""
eval/run_quality.py
===================
Unified entry-point for all quality evaluation benchmarks.

Reads ``configs/eval.yaml``, then runs each enabled benchmark
(GSM8K, MATH, RouterBench) sequentially against the teacher endpoint.
Results are stored under a single timestamped run directory with
per-benchmark subdirectories.

Usage
-----
    python -m eval.run_quality [--config configs/eval.yaml]

Requires a running vLLM teacher server.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from bench.metrics import save_json
from utils.config_loader import load_yaml
from utils.logging import setup_logging, get_logger
from utils.reproducibility import (
    collect_metadata,
    make_run_dir,
    save_metadata,
    set_seed,
    snapshot_configs,
)

logger = get_logger(__name__)


def run(
    config_path: str = "configs/eval.yaml",
    model_override: str | None = None,
    role: str | None = None,
) -> Path:
    """Execute all enabled quality benchmarks and return the run directory."""
    cfg = load_yaml(config_path)
    if model_override:
        cfg.setdefault("common", {})["model"] = model_override
    common = cfg.get("common", {})
    seed = common.get("seed", 42)
    set_seed(seed)
    setup_logging()

    # ── Run directory ───────────────────────────────────────
    # Tag format: quality-<role>  so results/quality/quality-teacher-<ts>/
    tag = f"quality-{role}" if role else "quality"
    run_dir = make_run_dir(
        common.get("results_base_dir", "results/quality"),
        tag=tag,
    )
    snapshot_configs(
        [config_path, "configs/serving.yaml", "configs/models.yaml"],
        run_dir,
    )
    meta = collect_metadata(seed, cfg)
    save_metadata(meta, run_dir)

    benchmarks_cfg = cfg.get("benchmarks", {})
    all_metrics: List[Dict[str, Any]] = []

    # ── GSM8K ───────────────────────────────────────────────
    if benchmarks_cfg.get("gsm8k", {}).get("enabled", False):
        logger.info("Starting GSM8K evaluation")
        try:
            from eval.gsm8k import run_gsm8k
            metrics = run_gsm8k(cfg, run_dir)
            all_metrics.append(metrics)
        except FileNotFoundError as exc:
            logger.error("GSM8K dataset not available", extra={"error": str(exc)})
        except Exception:
            logger.exception("GSM8K evaluation failed")
    else:
        logger.info("GSM8K evaluation disabled — skipping")

    # ── MATH ────────────────────────────────────────────────
    if benchmarks_cfg.get("math", {}).get("enabled", False):
        logger.info("Starting MATH evaluation")
        try:
            from eval.math_eval import run_math
            metrics = run_math(cfg, run_dir)
            all_metrics.append(metrics)
        except FileNotFoundError as exc:
            logger.error("MATH dataset not available", extra={"error": str(exc)})
        except Exception:
            logger.exception("MATH evaluation failed")
    else:
        logger.info("MATH evaluation disabled — skipping")

    # ── ARC-Challenge ───────────────────────────────────────
    if benchmarks_cfg.get("arc_challenge", {}).get("enabled", False):
        logger.info("Starting ARC-Challenge evaluation")
        try:
            from eval.arc_eval import run_arc
            metrics = run_arc(cfg, run_dir)
            all_metrics.append(metrics)
        except Exception:
            logger.exception("ARC-Challenge evaluation failed")
    else:
        logger.info("ARC-Challenge evaluation disabled — skipping")

    # ── RouterBench (disabled — uses .pkl files, not load_dataset-compatible)
    if benchmarks_cfg.get("routerbench", {}).get("enabled", False):
        logger.info("Starting RouterBench evaluation")
        try:
            from eval.routerbench import run_routerbench
            metrics = run_routerbench(cfg, run_dir)
            all_metrics.append(metrics)
        except Exception:
            logger.exception("RouterBench evaluation failed")
    else:
        logger.info("RouterBench evaluation disabled — skipping")

    # ── Aggregate summary ───────────────────────────────────
    save_json(all_metrics, run_dir / "quality_summary.json")

    if all_metrics:
        with (run_dir / "quality_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            fields = list(all_metrics[0].keys())
            # Ensure all keys are covered
            for m in all_metrics[1:]:
                for k in m:
                    if k not in fields:
                        fields.append(k)
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_metrics)

    logger.info(
        "Quality evaluation complete",
        extra={
            "run_dir": str(run_dir),
            "benchmarks_run": len(all_metrics),
        },
    )
    return run_dir


# ── CLI entry-point ─────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run quality evaluation benchmarks (GSM8K, MATH, RouterBench)",
    )
    parser.add_argument("--config", default="configs/eval.yaml",
                        help="Path to the evaluation YAML configuration file.")
    parser.add_argument("--model", default=None,
                        help="Override common.model in the config (e.g. Qwen/Qwen2.5-7B-Instruct).")
    parser.add_argument("--role", default=None,
                        help="Label for this run: teacher | student_mid | student_small.")
    args = parser.parse_args()
    run(args.config, model_override=args.model, role=args.role)


if __name__ == "__main__":
    main()
