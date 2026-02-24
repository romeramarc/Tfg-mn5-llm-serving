"""
eval/routerbench.py
===================
RouterBench quality evaluation runner.

RouterBench is loaded from a local JSONL file placed under
``data/routerbench/``.  If labels are available in the dataset,
accuracy is computed; otherwise, raw model outputs are stored
and a placeholder scorer logs a TODO.

Dataset acquisition
-------------------
The RouterBench dataset is NOT bundled with this repository.
See ``data/routerbench/README.md`` for download instructions.

Usage (standalone)
------------------
    python -m eval.routerbench --config configs/eval.yaml

Normally invoked through ``eval/run_quality.py``.
"""

from __future__ import annotations

import asyncio
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from eval.scoring import compute_accuracy
from utils.logging import get_logger

logger = get_logger(__name__)


# ── Dataset loading ─────────────────────────────────────────

def load_routerbench(
    dataset_path: str = "data/routerbench/routerbench.jsonl",
    subset_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load RouterBench from a local JSONL file.

    Each line must be a JSON object with at least a ``"prompt"`` key.
    Optional keys: ``"label"``, ``"expected_answer"``, ``"metadata"``.

    Raises ``FileNotFoundError`` with a helpful message if the file
    is missing.
    """
    p = Path(dataset_path)
    if not p.exists():
        raise FileNotFoundError(
            f"RouterBench dataset not found at {p.resolve()}.\n"
            f"Please follow the instructions in data/routerbench/README.md "
            f"to download and place the dataset."
        )

    examples: list[dict] = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line",
                               extra={"lineno": lineno, "error": str(exc)})
                continue
            if "prompt" not in obj:
                logger.warning("Skipping line without 'prompt' key",
                               extra={"lineno": lineno})
                continue
            examples.append(obj)

    if subset_size is not None and subset_size > 0:
        examples = examples[:subset_size]

    logger.info("Loaded RouterBench",
                 extra={"path": str(p), "examples": len(examples)})
    return examples


# ── Async evaluation ───────────────────────────────────────

async def _query_model(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> Dict[str, Any]:
    """Send a single completion request."""
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
    examples: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Query the model for every RouterBench example."""
    common = cfg.get("common", {})
    bench_cfg = cfg.get("benchmarks", {}).get("routerbench", {})
    req_cfg = cfg.get("request", {})

    base_url = common.get("base_url", "http://localhost:8000")
    url = f"{base_url.rstrip('/')}/v1/completions"
    model = common.get("model", "")
    max_tokens = common.get("max_tokens", 1024)
    temperature = common.get("temperature", 0.0)
    timeout = req_cfg.get("timeout_seconds", 180)
    batch_size = req_cfg.get("batch_size", 16)
    has_labels = bench_cfg.get("has_labels", False)
    prompt_template = bench_cfg.get("prompt_template")

    sem = asyncio.Semaphore(batch_size)
    results: list[dict] = []

    async def _handle(idx: int, ex: dict) -> None:
        raw_prompt = ex["prompt"]
        if prompt_template:
            prompt = prompt_template.replace("{prompt}", raw_prompt)
        else:
            prompt = raw_prompt

        async with sem:
            resp = await _query_model(
                client, url, model, prompt, max_tokens, temperature, timeout
            )

        record: dict = {
            "index": idx,
            "prompt": raw_prompt[:300],
            "model_response": resp["text"],
            "latency_ms": resp["latency_ms"],
            "error": resp["error"],
        }

        # Scoring if labels are available
        if has_labels:
            label = ex.get("label") or ex.get("expected_answer", "")
            record["label"] = label
            if resp["error"]:
                record["correct"] = False
                record["scorable"] = False
            else:
                # Simple exact-match on stripped text
                pred = resp["text"].strip()
                record["predicted_answer"] = pred
                record["correct"] = pred == str(label).strip()
                record["scorable"] = True
        else:
            record["correct"] = None
            record["scorable"] = False

        # Preserve any extra metadata from the dataset
        if "metadata" in ex:
            record["example_metadata"] = ex["metadata"]

        results.append(record)

    async with httpx.AsyncClient() as client:
        tasks = [asyncio.create_task(_handle(i, ex))
                 for i, ex in enumerate(examples)]
        await asyncio.gather(*tasks)

    results.sort(key=lambda r: r["index"])
    return results


# ── Public interface ────────────────────────────────────────

def run_routerbench(cfg: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """Execute the RouterBench evaluation and write results to *run_dir*.

    Returns the metrics dict.
    """
    bench_cfg = cfg.get("benchmarks", {}).get("routerbench", {})
    has_labels = bench_cfg.get("has_labels", False)

    examples = load_routerbench(
        dataset_path=bench_cfg.get("dataset_path",
                                   "data/routerbench/routerbench.jsonl"),
        subset_size=bench_cfg.get("subset_size"),
    )

    results = asyncio.run(_evaluate_batch(examples, cfg))

    # Build metrics
    if has_labels:
        metrics = compute_accuracy(results)
    else:
        metrics = {
            "total_examples": len(results),
            "scorable_examples": 0,
            "unscorable_examples": len(results),
            "correct": 0,
            "accuracy": None,
            "accuracy_pct": None,
            "note": (
                "RouterBench labels not available. Raw outputs stored. "
                "TODO: implement domain-specific scoring once label format "
                "is confirmed."
            ),
        }

    metrics["benchmark"] = "routerbench"
    metrics["model"] = cfg.get("common", {}).get("model", "")

    # Persist
    out_dir = run_dir / "routerbench"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "routerbench_results.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    with (out_dir / "routerbench_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    with (out_dir / "routerbench_metrics.csv").open("w",
          newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    logger.info("RouterBench evaluation complete", extra=metrics)
    return metrics
