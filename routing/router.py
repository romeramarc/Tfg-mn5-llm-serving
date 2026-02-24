"""
routing/router.py
=================
Top-level routing experiment runner.

Reads ``configs/routing.yaml``, selects the active policy, dispatches
every prompt through the chosen policy, and writes structured logs
including model selection, latency, routing reason, and confidence.

Usage
-----
    python -m routing.router [--config configs/routing.yaml]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import httpx

from distill.dataset_utils import load_prompts
from routing.policies import POLICIES, RoutingDecision
from bench.metrics import save_csv, save_json, summarise_latencies
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


# ── Async dispatcher ───────────────────────────────────────

async def _dispatch(
    prompts: List[str],
    policy_fn,
    ctx: Dict[str, Any],
    concurrency: int = 16,
) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    records: List[Dict[str, Any]] = []

    async def _handle(idx: int, prompt: str) -> None:
        async with sem:
            async with httpx.AsyncClient() as client:
                decision: RoutingDecision = await policy_fn(client, prompt, ctx)
            rec = asdict(decision)
            rec["prompt_index"] = idx
            rec["prompt_preview"] = prompt[:120]
            records.append(rec)

    tasks = [asyncio.create_task(_handle(i, p)) for i, p in enumerate(prompts)]
    await asyncio.gather(*tasks)
    return records


# ── Public entry-point ──────────────────────────────────────

def run(config_path: str = "configs/routing.yaml") -> Path:
    cfg = load_yaml(config_path)
    exp = cfg.get("experiment", {})
    seed = exp.get("seed", 42)
    set_seed(seed)
    setup_logging()

    run_dir = make_run_dir(
        exp.get("results_dir", "results/routing"),
        tag=cfg.get("active_policy", "routing"),
    )
    snapshot_configs([config_path], run_dir)
    save_metadata(collect_metadata(seed, cfg), run_dir)

    # ── Select policy ───────────────────────────────────────
    policy_name = cfg.get("active_policy", "always_teacher")
    if policy_name not in POLICIES:
        raise ValueError(
            f"Unknown policy '{policy_name}'. "
            f"Available: {list(POLICIES.keys())}"
        )
    policy_fn = POLICIES[policy_name]

    # ── Build context ───────────────────────────────────────
    ctx: Dict[str, Any] = {
        "endpoints": cfg.get("endpoints", {}),
        "max_tokens": exp.get("max_tokens", 256),
        "temperature": exp.get("temperature", 0.0),
    }
    # Merge policy-specific params
    policy_params = cfg.get("policies", {}).get(policy_name, {})
    ctx.update(policy_params)

    # ── Load prompts ────────────────────────────────────────
    prompts = load_prompts(exp.get("prompts_file", "configs/prompts.jsonl"))
    num_requests = min(exp.get("num_requests", len(prompts)), len(prompts))
    prompts = prompts[:num_requests]
    logger.info("Routing experiment",
                 extra={"policy": policy_name, "prompts": len(prompts)})

    # ── Run ─────────────────────────────────────────────────
    records = asyncio.run(_dispatch(prompts, policy_fn, ctx))

    # ── Summaries ───────────────────────────────────────────
    latencies = [r["latency_ms"] for r in records]
    teacher_count = sum(1 for r in records
                        if r["selected_model"] == cfg["endpoints"]["teacher"]["model"])
    student_count = len(records) - teacher_count

    summary: Dict[str, Any] = {
        "policy": policy_name,
        "total_requests": len(records),
        "teacher_routed": teacher_count,
        "student_routed": student_count,
        **{f"latency_{k}": v for k, v in summarise_latencies(latencies).items()},
    }

    save_json(records, run_dir / "routing_decisions.json")
    save_json(summary, run_dir / "routing_summary.json")
    save_csv(records, run_dir / "routing_decisions.csv")
    logger.info("Routing experiment complete",
                 extra={"run_dir": str(run_dir), "summary": summary})

    return run_dir


# ── CLI ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run routing experiment")
    parser.add_argument("--config", default="configs/routing.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
