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
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import httpx

from distill.dataset_utils import read_jsonl
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
    prompt_records: List[Dict[str, Any]],
    policy_fn,
    ctx: Dict[str, Any],
    concurrency: int = 16,
) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    records: List[Dict[str, Any] | None] = [None] * len(prompt_records)

    async with httpx.AsyncClient() as client:

        async def _handle(idx: int, prompt_record: Dict[str, Any]) -> None:
            prompt = prompt_record["prompt"]
            request_ctx = dict(ctx)
            request_ctx["request_id"] = prompt_record["request_id"]
            request_ctx["prompt_metadata"] = prompt_record.get("metadata", {})

            async with sem:
                decision: RoutingDecision = await policy_fn(client, prompt, request_ctx)

            rec = asdict(decision)
            rec["prompt_index"] = idx
            rec["prompt_preview"] = prompt[:120]
            rec["prompt_metadata"] = prompt_record.get("metadata", {})
            rec["policy"] = ctx.get("policy_name")
            records[idx] = rec

        tasks = [
            asyncio.create_task(_handle(i, p))
            for i, p in enumerate(prompt_records)
        ]
        await asyncio.gather(*tasks)

    return [r for r in records if r is not None]


def _load_prompt_records(prompts_file: str, num_requests: int) -> List[Dict[str, Any]]:
    """Load prompts with stable request IDs and pass-through metadata."""
    rows = read_jsonl(prompts_file)
    records: List[Dict[str, Any]] = []

    for i, row in enumerate(rows):
        if "prompt" not in row:
            raise KeyError(f"Missing 'prompt' key in row {i} of {prompts_file}")

        request_id = str(row.get("request_id") or f"req-{i:06d}")
        metadata = {
            k: v
            for k, v in row.items()
            if k not in {"prompt", "request_id"}
        }
        records.append({
            "request_id": request_id,
            "prompt": row["prompt"],
            "metadata": metadata,
        })

    if num_requests > 0:
        return records[:num_requests]
    return records


def _flatten_attempts(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand nested attempt traces into one row per attempt."""
    rows: List[Dict[str, Any]] = []
    for rec in records:
        attempts = rec.get("attempts") or []
        for attempt_index, attempt in enumerate(attempts):
            row: Dict[str, Any] = {
                "request_id": rec.get("request_id"),
                "prompt_index": rec.get("prompt_index"),
                "attempt_index": attempt_index,
                "policy": rec.get("policy"),
                "final_selected_model": rec.get("selected_model"),
                "final_reason": rec.get("reason"),
                "final_latency_ms": rec.get("latency_ms"),
            }
            row.update(attempt)
            rows.append(row)
    return rows


def _build_summary(records: List[Dict[str, Any]], policy_name: str) -> Dict[str, Any]:
    """Compute aggregate routing/service metrics for the run."""
    latencies = [r.get("latency_ms", 0.0) for r in records]
    model_counts = Counter(r.get("selected_model", "unknown") for r in records)
    reason_counts = Counter(r.get("reason", "unknown") for r in records)
    hop_counts = Counter(len(r.get("attempts") or []) for r in records)

    all_attempts = _flatten_attempts(records)
    attempt_stage_counts = Counter(a.get("stage", "unknown") for a in all_attempts)
    attempt_error_count = sum(1 for a in all_attempts if a.get("error"))

    return {
        "policy": policy_name,
        "total_requests": len(records),
        "cascaded_requests": sum(1 for r in records if len(r.get("attempts") or []) > 1),
        "model_selection_counts": dict(model_counts),
        "reason_counts": dict(reason_counts),
        "hop_counts": {str(k): v for k, v in hop_counts.items()},
        "attempt_stage_counts": dict(attempt_stage_counts),
        "attempt_error_count": attempt_error_count,
        **{f"latency_{k}": v for k, v in summarise_latencies(latencies).items()},
    }


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
        "policy_name": policy_name,
    }
    # Merge policy-specific params
    policy_params = cfg.get("policies", {}).get(policy_name, {})
    ctx.update(policy_params)

    # ── Load prompts ────────────────────────────────────────
    prompts_file = exp.get("prompts_file", "configs/prompts.jsonl")
    requested = int(exp.get("num_requests", 0) or 0)
    prompt_records = _load_prompt_records(prompts_file, requested)
    logger.info("Routing experiment",
                extra={"policy": policy_name, "prompts": len(prompt_records)})

    # ── Run ─────────────────────────────────────────────────
    records = asyncio.run(
        _dispatch(
            prompt_records,
            policy_fn,
            ctx,
            concurrency=int(exp.get("concurrency", 16)),
        )
    )

    # ── Summaries ───────────────────────────────────────────
    summary = _build_summary(records, policy_name)
    attempt_rows = _flatten_attempts(records)

    save_json(records, run_dir / "routing_decisions.json")
    save_json(summary, run_dir / "routing_summary.json")
    save_csv(records, run_dir / "routing_decisions.csv")
    save_json(attempt_rows, run_dir / "routing_attempts.json")
    save_csv(attempt_rows, run_dir / "routing_attempts.csv")
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
