"""
distill/generate_teacher_outputs.py
====================================
Query the teacher model via its OpenAI-compatible ``/v1/completions``
endpoint and persist (prompt, response, metadata) triples as JSONL.

The output file becomes the training dataset for the student SFT step.

Usage
-----
    python -m distill.generate_teacher_outputs [--config configs/distill.yaml]

Requires a running vLLM teacher server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx

from distill.dataset_utils import load_prompts, write_jsonl
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


# ── Async query helper ──────────────────────────────────────

async def _query_teacher(
    client: httpx.AsyncClient,
    url: str,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
) -> Dict[str, Any]:
    """Send a single completion request and return structured output."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, json=payload, timeout=timeout)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        resp.raise_for_status()
        body = resp.json()

        # Extract generated text from OpenAI-compatible response
        text = ""
        if "choices" in body and body["choices"]:
            text = body["choices"][0].get("text", "")

        return {
            "prompt": prompt,
            "response": text,
            "model": model,
            "latency_ms": latency_ms,
            "usage": body.get("usage"),
            "finish_reason": body["choices"][0].get("finish_reason")
                             if body.get("choices") else None,
            "error": None,
        }
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.error("Teacher query failed",
                      extra={"prompt_preview": prompt[:80], "error": str(exc)})
        return {
            "prompt": prompt,
            "response": None,
            "model": model,
            "latency_ms": latency_ms,
            "usage": None,
            "finish_reason": None,
            "error": str(exc),
        }


async def _generate_all(
    prompts: List[str],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Batch-query the teacher asynchronously."""
    gen = cfg.get("generation", {})
    base_url = gen.get("teacher_base_url", "http://localhost:8000")
    url = f"{base_url.rstrip('/')}/v1/completions"
    model = gen.get("teacher_model", "")
    max_tokens = gen.get("max_tokens", 512)
    temperature = gen.get("temperature", 0.0)
    top_p = gen.get("top_p", 1.0)
    batch_size = gen.get("batch_size", 32)
    timeout = gen.get("timeout_seconds", 120)

    sem = asyncio.Semaphore(batch_size)
    results: List[Dict[str, Any]] = []

    async def _bounded(prompt: str) -> None:
        async with sem:
            res = await _query_teacher(
                client, url, prompt, model,
                max_tokens, temperature, top_p, timeout,
            )
            results.append(res)

    async with httpx.AsyncClient() as client:
        tasks = [asyncio.create_task(_bounded(p)) for p in prompts]
        await asyncio.gather(*tasks)

    return results


# ── Public entry-point ──────────────────────────────────────

def run(config_path: str = "configs/distill.yaml") -> Path:
    cfg = load_yaml(config_path)
    gen = cfg.get("generation", {})
    seed = gen.get("seed", 42)
    set_seed(seed)
    setup_logging()

    run_dir = make_run_dir("results/distill", tag="teacher-gen")
    snapshot_configs([config_path], run_dir)
    save_metadata(collect_metadata(seed, cfg), run_dir)

    prompts = load_prompts(gen.get("prompts_file", "configs/prompts.jsonl"))
    logger.info("Generating teacher outputs",
                 extra={"num_prompts": len(prompts)})

    results = asyncio.run(_generate_all(prompts, cfg))

    # Filter out failures
    ok = [r for r in results if r["error"] is None]
    logger.info("Generation complete",
                 extra={"total": len(results), "success": len(ok),
                        "failed": len(results) - len(ok)})

    # Save — always write all results (including errors) for auditing
    out_path = Path(gen.get("output_file", "results/distill/teacher_outputs.jsonl"))
    write_jsonl(results, out_path)
    # Also save into the run directory
    write_jsonl(results, run_dir / "teacher_outputs.jsonl")

    return run_dir


# ── CLI ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate teacher outputs")
    parser.add_argument("--config", default="configs/distill.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
