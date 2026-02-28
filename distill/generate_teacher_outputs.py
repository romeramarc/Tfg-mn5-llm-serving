"""
distill/generate_teacher_outputs.py
====================================
Query the teacher model via its OpenAI-compatible ``/v1/completions``
endpoint and persist (prompt, response, metadata) triples as JSONL.

The output file becomes the training dataset for the student SFT step.

Prompts are sourced from the **same evaluation benchmarks** used in Phase 1
(GSM8K, MATH-500, ARC-Challenge) to ensure that the distillation dataset
exercises the exact capabilities we will evaluate.  A configurable
``extra_prompts_file`` can add supplementary prompts beyond the benchmarks.

JSONL output schema (one record per line)::

    {
        "id":                     "<benchmark>-<index>",
        "benchmark":              "gsm8k" | "math" | "arc_challenge" | "extra",
        "prompt":                 "<full prompt text>",
        "teacher_completion":     "<greedy teacher answer>",
        "teacher_model":          "Qwen/Qwen2.5-14B-Instruct",
        "generation_parameters":  {"temperature": 0.0, ...},
        "latency_ms":             123.4,
        "finish_reason":          "stop" | "length",
        "error":                  null | "<error message>"
    }

Usage
-----
    python -m distill.generate_teacher_outputs \\
        --config configs/distill.yaml

Requires a running vLLM teacher server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from distill.dataset_utils import write_jsonl
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


# ── Prompt collection from benchmarks ───────────────────────

def _collect_gsm8k_prompts(cfg: dict) -> List[Dict[str, Any]]:
    """Collect prompts from GSM8K test set."""
    from datasets import load_dataset

    bench = cfg.get("benchmarks", {}).get("gsm8k", {})
    if not bench.get("enabled", True):
        return []

    ds = load_dataset(
        bench.get("dataset_name", "openai/gsm8k"),
        "main",
        split=bench.get("dataset_split", "test"),
    )
    template = bench.get("prompt_template",
        "Solve the following math problem step by step.\n"
        "Put your final numeric answer after ####.\n\n"
        "Question: {question}\n\nAnswer:")
    prompts = []
    for i, row in enumerate(ds):
        prompts.append({
            "id": f"gsm8k-{i}",
            "benchmark": "gsm8k",
            "prompt": template.format(question=row["question"]),
        })
    subset = bench.get("subset_size")
    if subset and subset > 0:
        prompts = prompts[:subset]
    logger.info("Collected GSM8K prompts", extra={"n": len(prompts)})
    return prompts


def _collect_math_prompts(cfg: dict) -> List[Dict[str, Any]]:
    """Collect prompts from MATH-500 test set."""
    from datasets import load_dataset

    bench = cfg.get("benchmarks", {}).get("math", {})
    if not bench.get("enabled", True):
        return []

    ds = load_dataset(
        bench.get("dataset_name", "HuggingFaceH4/MATH-500"),
        split=bench.get("dataset_split", "test"),
    )
    template = bench.get("prompt_template",
        "Solve the following mathematics problem.\n"
        "Put your final answer inside \\boxed{{}}.\n\n"
        "Problem: {problem}\n\nSolution:")
    prompts = []
    for i, row in enumerate(ds):
        prompts.append({
            "id": f"math-{i}",
            "benchmark": "math",
            "prompt": template.format(problem=row["problem"]),
        })
    subset = bench.get("subset_size")
    if subset and subset > 0:
        prompts = prompts[:subset]
    logger.info("Collected MATH prompts", extra={"n": len(prompts)})
    return prompts


def _collect_arc_prompts(cfg: dict) -> List[Dict[str, Any]]:
    """Collect prompts from ARC-Challenge test set."""
    from datasets import load_dataset

    bench = cfg.get("benchmarks", {}).get("arc_challenge", {})
    if not bench.get("enabled", True):
        return []

    ds = load_dataset(
        bench.get("dataset_name", "allenai/ai2_arc"),
        bench.get("dataset_config", "ARC-Challenge"),
        split=bench.get("dataset_split", "test"),
    )
    label_map = {"1": "A", "2": "B", "3": "C", "4": "D",
                 "A": "A", "B": "B", "C": "C", "D": "D", "E": "E"}
    prompts = []
    for i, row in enumerate(ds):
        choices = row["choices"]
        answers_block = "\n".join(
            f"{label_map.get(lbl, lbl)}) {txt}"
            for lbl, txt in zip(choices["label"], choices["text"])
        )
        prompt = (
            "Answer the following multiple-choice science question.\n"
            "Write only the letter of the correct answer (A, B, C, or D).\n\n"
            f"Question: {row['question']}\n"
            f"{answers_block}\n\n"
            "Answer:"
        )
        prompts.append({
            "id": f"arc-{i}",
            "benchmark": "arc_challenge",
            "prompt": prompt,
        })
    subset = bench.get("subset_size")
    if subset and subset > 0:
        prompts = prompts[:subset]
    logger.info("Collected ARC prompts", extra={"n": len(prompts)})
    return prompts


def _collect_extra_prompts(path: Optional[str]) -> List[Dict[str, Any]]:
    """Load additional prompts from a JSONL file (optional)."""
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        logger.warning("Extra prompts file not found, skipping",
                        extra={"path": str(p)})
        return []
    prompts = []
    with p.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append({
                "id": f"extra-{i}",
                "benchmark": "extra",
                "prompt": obj.get("prompt", ""),
            })
    logger.info("Collected extra prompts", extra={"n": len(prompts)})
    return prompts


def collect_all_prompts(cfg: dict) -> List[Dict[str, Any]]:
    """Gather prompts from all configured benchmark sources + extras."""
    all_prompts: List[Dict[str, Any]] = []
    all_prompts.extend(_collect_gsm8k_prompts(cfg))
    all_prompts.extend(_collect_math_prompts(cfg))
    all_prompts.extend(_collect_arc_prompts(cfg))

    extra_path = cfg.get("generation", {}).get("extra_prompts_file")
    all_prompts.extend(_collect_extra_prompts(extra_path))

    logger.info("Total prompts collected", extra={"n": len(all_prompts)})
    return all_prompts


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

        text = ""
        if "choices" in body and body["choices"]:
            text = body["choices"][0].get("text", "")

        return {
            "teacher_completion": text,
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
            "teacher_completion": None,
            "latency_ms": latency_ms,
            "usage": None,
            "finish_reason": None,
            "error": str(exc),
        }


async def _generate_all(
    prompts: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Batch-query the teacher asynchronously."""
    gen = cfg.get("generation", {})
    base_url = gen.get("teacher_base_url", "http://localhost:8000")
    url = f"{base_url.rstrip('/')}/v1/completions"
    model = gen.get("teacher_model", "")
    max_tokens = gen.get("max_tokens", 1024)
    temperature = gen.get("temperature", 0.0)
    top_p = gen.get("top_p", 1.0)
    batch_size = gen.get("batch_size", 32)
    timeout = gen.get("timeout_seconds", 180)

    gen_params = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    sem = asyncio.Semaphore(batch_size)
    results: List[Dict[str, Any]] = []

    async def _bounded(item: Dict[str, Any]) -> None:
        async with sem:
            res = await _query_teacher(
                client, url, item["prompt"], model,
                max_tokens, temperature, top_p, timeout,
            )
        record = {
            "id": item["id"],
            "benchmark": item["benchmark"],
            "prompt": item["prompt"],
            "teacher_completion": res["teacher_completion"],
            "teacher_model": model,
            "generation_parameters": gen_params,
            "latency_ms": res["latency_ms"],
            "finish_reason": res["finish_reason"],
            "error": res["error"],
        }
        results.append(record)

    async with httpx.AsyncClient() as client:
        tasks = [asyncio.create_task(_bounded(item)) for item in prompts]
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
    snapshot_configs([config_path, "configs/eval.yaml"], run_dir)
    save_metadata(collect_metadata(seed, cfg), run_dir)

    # Load eval config for benchmark prompt templates
    eval_cfg = load_yaml(gen.get("eval_config", "configs/eval.yaml"))

    prompts = collect_all_prompts(eval_cfg)
    logger.info("Generating teacher outputs",
                 extra={"num_prompts": len(prompts)})

    results = asyncio.run(_generate_all(prompts, cfg))

    # Stats
    ok = [r for r in results if r["error"] is None]
    failed = len(results) - len(ok)
    logger.info("Generation complete",
                 extra={"total": len(results), "success": len(ok),
                        "failed": failed})
    if failed > 0:
        logger.warning(f"{failed} prompts failed — stored with error field")

    # Save — always write all results (including errors) for auditing
    out_path = Path(gen.get("output_file",
                            "results/distill/teacher_outputs.jsonl"))
    write_jsonl(results, out_path)
    write_jsonl(results, run_dir / "teacher_outputs.jsonl")

    # Summary stats
    summary = {
        "total_prompts": len(results),
        "successful": len(ok),
        "failed": failed,
        "by_benchmark": {},
    }
    for r in results:
        b = r["benchmark"]
        if b not in summary["by_benchmark"]:
            summary["by_benchmark"][b] = {"total": 0, "ok": 0}
        summary["by_benchmark"][b]["total"] += 1
        if r["error"] is None:
            summary["by_benchmark"][b]["ok"] += 1

    with (run_dir / "generation_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    return run_dir


# ── CLI ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate teacher outputs")
    parser.add_argument("--config", default="configs/distill.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
