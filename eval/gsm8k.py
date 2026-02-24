"""
eval/gsm8k.py
=============
GSM8K quality evaluation runner.

Loads the GSM8K dataset from HuggingFace, queries the teacher model
for each problem, extracts the final numeric answer, and scores via
exact-match against the reference answer.

Usage (standalone)
------------------
    python -m eval.gsm8k --config configs/eval.yaml

Normally invoked through ``eval/run_quality.py``.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from eval.scoring import (
    compute_accuracy,
    extract_numeric_answer,
    numeric_match,
)
from utils.logging import get_logger

logger = get_logger(__name__)


# ── Dataset loading ────────────────────────────────────────

def load_gsm8k(
    dataset_name: str = "openai/gsm8k",
    split: str = "test",
    subset_size: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Load GSM8K from HuggingFace datasets and return a list of
    ``{"question": ..., "reference_answer": ...}`` dicts.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, "main", split=split)
    examples: list[dict] = []
    for row in ds:
        # GSM8K answer field has format: "... #### <number>"
        raw_answer = row.get("answer", "")
        # Extract the number after ####
        match = re.search(r"####\s*([\-\d,\.]+)", raw_answer)
        ref = match.group(1).strip() if match else raw_answer.strip()
        examples.append({
            "question": row["question"],
            "reference_answer": ref,
            "full_solution": raw_answer,
        })
    if subset_size is not None and subset_size > 0:
        examples = examples[:subset_size]
    logger.info("Loaded GSM8K", extra={"split": split, "examples": len(examples)})
    return examples


# ── Async evaluation ──────────────────────────────────────

async def _query_model(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> Dict[str, Any]:
    """Send a single completion request to the vLLM endpoint."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, json=payload, timeout=timeout)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        resp.raise_for_status()
        body = resp.json()
        text = ""
        if body.get("choices"):
            text = body["choices"][0].get("text", "")
        return {"text": text, "latency_ms": latency_ms, "error": None}
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {"text": "", "latency_ms": latency_ms, "error": str(exc)}


async def _evaluate_batch(
    examples: List[Dict[str, str]],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Query the model for every GSM8K example and score responses."""
    common = cfg.get("common", {})
    bench_cfg = cfg.get("benchmarks", {}).get("gsm8k", {})
    req_cfg = cfg.get("request", {})

    base_url = common.get("base_url", "http://localhost:8000")
    url = f"{base_url.rstrip('/')}/v1/completions"
    model = common.get("model", "")
    max_tokens = common.get("max_tokens", 1024)
    temperature = common.get("temperature", 0.0)
    timeout = req_cfg.get("timeout_seconds", 180)
    batch_size = req_cfg.get("batch_size", 16)
    answer_pattern = bench_cfg.get(
        "answer_extraction_pattern", r"####\s*([\-\d,\.]+)"
    )
    prompt_template = bench_cfg.get("prompt_template", "Question: {question}\n\nAnswer:")

    sem = asyncio.Semaphore(batch_size)
    results: list[dict] = []

    async def _handle(idx: int, ex: dict) -> None:
        prompt = prompt_template.replace("{question}", ex["question"])
        async with sem:
            resp = await _query_model(
                client, url, model, prompt, max_tokens, temperature, timeout
            )

        predicted_raw = extract_numeric_answer(resp["text"], answer_pattern)
        ref = ex["reference_answer"]
        correct = False
        scorable = True

        if resp["error"]:
            scorable = False
        elif predicted_raw is None:
            scorable = False
        else:
            correct = numeric_match(predicted_raw, ref)

        results.append({
            "index": idx,
            "question": ex["question"],
            "reference_answer": ref,
            "model_response": resp["text"],
            "predicted_answer": predicted_raw,
            "correct": correct,
            "scorable": scorable,
            "latency_ms": resp["latency_ms"],
            "error": resp["error"],
        })

    async with httpx.AsyncClient() as client:
        tasks = [asyncio.create_task(_handle(i, ex))
                 for i, ex in enumerate(examples)]
        await asyncio.gather(*tasks)

    # Sort by original index
    results.sort(key=lambda r: r["index"])
    return results


# ── Public interface ────────────────────────────────────────

def run_gsm8k(cfg: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """Execute the GSM8K evaluation and write results to *run_dir*.

    Returns the metrics dict.
    """
    bench_cfg = cfg.get("benchmarks", {}).get("gsm8k", {})

    examples = load_gsm8k(
        dataset_name=bench_cfg.get("dataset_name", "openai/gsm8k"),
        split=bench_cfg.get("dataset_split", "test"),
        subset_size=bench_cfg.get("subset_size"),
    )

    results = asyncio.run(_evaluate_batch(examples, cfg))
    metrics = compute_accuracy(results)
    metrics["benchmark"] = "gsm8k"
    metrics["model"] = cfg.get("common", {}).get("model", "")

    # Persist
    out_dir = run_dir / "gsm8k"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-example JSONL
    with (out_dir / "gsm8k_results.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    # Metrics JSON + CSV
    with (out_dir / "gsm8k_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    import csv
    with (out_dir / "gsm8k_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    # Log unscorable examples separately
    unscorable = [r for r in results if not r.get("scorable", True)]
    if unscorable:
        with (out_dir / "gsm8k_unscorable.jsonl").open("w", encoding="utf-8") as fh:
            for r in unscorable:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    logger.info("GSM8K evaluation complete", extra=metrics)
    return metrics
