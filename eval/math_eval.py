"""
eval/math_eval.py
=================
MATH benchmark quality evaluation runner.

Loads the MATH dataset from HuggingFace, queries the teacher model,
extracts the ``\\boxed{...}`` answer, and scores via normalised
exact-match against the reference.

Ambiguous or unscorable cases (formatting issues, missing boxed answer)
are logged separately.

Usage (standalone)
------------------
    python -m eval.math_eval --config configs/eval.yaml

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

from eval.scoring import (
    compute_accuracy,
    extract_boxed_answer,
    math_answer_match,
    normalise_math_answer,
)
from utils.logging import get_logger

logger = get_logger(__name__)


# ── Dataset loading ─────────────────────────────────────────

# All subject configs in hendrycks/competition_math
_MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def load_math(
    dataset_name: str = "hendrycks/competition_math",
    split: str = "test",
    subset_size: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Load MATH from HuggingFace (all subjects) and return structured examples.

    ``hendrycks/competition_math`` is split by subject rather than having a
    single ``all`` config.  This function loads each subject and concatenates.
    """
    from datasets import load_dataset

    all_rows: list[Any] = []
    for subject in _MATH_SUBJECTS:
        subject_ds = load_dataset(
            dataset_name, name=subject, split=split, trust_remote_code=True
        )
        all_rows.extend(subject_ds)
        logger.info(
            "Loaded MATH subject",
            extra={"subject": subject, "rows": len(subject_ds)},
        )

    examples: list[dict] = []
    for row in all_rows:
        # The reference answer is in the 'solution' field;
        # extract the boxed answer for scoring.
        solution = row.get("solution", "")
        ref = extract_boxed_answer(solution)
        examples.append({
            "problem": row.get("problem", ""),
            "level": row.get("level", ""),
            "type": row.get("type", ""),
            "full_solution": solution,
            "reference_answer": ref if ref else solution.strip(),
            "reference_is_boxed": ref is not None,
        })
    if subset_size is not None and subset_size > 0:
        examples = examples[:subset_size]
    logger.info("Loaded MATH", extra={"split": split, "examples": len(examples)})
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
    examples: List[Dict[str, str]],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Query the model for every MATH example and score responses."""
    common = cfg.get("common", {})
    bench_cfg = cfg.get("benchmarks", {}).get("math", {})
    req_cfg = cfg.get("request", {})

    base_url = common.get("base_url", "http://localhost:8000")
    url = f"{base_url.rstrip('/')}/v1/completions"
    model = common.get("model", "")
    max_tokens = common.get("max_tokens", 1024)
    temperature = common.get("temperature", 0.0)
    timeout = req_cfg.get("timeout_seconds", 180)
    batch_size = req_cfg.get("batch_size", 16)
    prompt_template = bench_cfg.get(
        "prompt_template",
        "Solve the following mathematics problem.\n"
        "Put your final answer inside \\boxed{}.\n\n"
        "Problem: {problem}\n\nSolution:"
    )

    sem = asyncio.Semaphore(batch_size)
    results: list[dict] = []

    async def _handle(idx: int, ex: dict) -> None:
        prompt = prompt_template.replace("{problem}", ex["problem"])
        async with sem:
            resp = await _query_model(
                client, url, model, prompt, max_tokens, temperature, timeout
            )

        predicted_boxed = extract_boxed_answer(resp["text"])
        ref = ex["reference_answer"]
        correct = False
        scorable = True
        ambiguity_reason: Optional[str] = None

        if resp["error"]:
            scorable = False
            ambiguity_reason = f"request_error: {resp['error']}"
        elif predicted_boxed is None:
            scorable = False
            ambiguity_reason = "no_boxed_answer_in_response"
        elif not ex.get("reference_is_boxed", True):
            # Reference itself was not boxed — scoring may be unreliable
            scorable = False
            ambiguity_reason = "reference_not_boxed"
        else:
            correct = math_answer_match(predicted_boxed, ref)

        results.append({
            "index": idx,
            "problem": ex["problem"][:200],  # truncate for log readability
            "level": ex.get("level", ""),
            "type": ex.get("type", ""),
            "reference_answer": ref,
            "model_response": resp["text"],
            "predicted_answer": predicted_boxed,
            "correct": correct,
            "scorable": scorable,
            "ambiguity_reason": ambiguity_reason,
            "latency_ms": resp["latency_ms"],
            "error": resp["error"],
        })

    async with httpx.AsyncClient() as client:
        tasks = [asyncio.create_task(_handle(i, ex))
                 for i, ex in enumerate(examples)]
        await asyncio.gather(*tasks)

    results.sort(key=lambda r: r["index"])
    return results


# ── Public interface ────────────────────────────────────────

def run_math(cfg: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """Execute the MATH evaluation and write results to *run_dir*.

    Returns the metrics dict.
    """
    bench_cfg = cfg.get("benchmarks", {}).get("math", {})

    examples = load_math(
        dataset_name=bench_cfg.get("dataset_name", "lighteval/MATH"),
        split=bench_cfg.get("dataset_split", "test"),
        subset_size=bench_cfg.get("subset_size"),
    )

    results = asyncio.run(_evaluate_batch(examples, cfg))
    metrics = compute_accuracy(results)
    metrics["benchmark"] = "math"
    metrics["model"] = cfg.get("common", {}).get("model", "")

    # Persist
    out_dir = run_dir / "math"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-example JSONL
    with (out_dir / "math_results.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    # Metrics JSON + CSV
    with (out_dir / "math_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    with (out_dir / "math_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    # Log ambiguity/unscorable cases separately
    unscorable = [r for r in results if not r.get("scorable", True)]
    if unscorable:
        with (out_dir / "math_unscorable.jsonl").open("w", encoding="utf-8") as fh:
            for r in unscorable:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        logger.info("MATH unscorable cases logged",
                     extra={"count": len(unscorable)})

    logger.info("MATH evaluation complete", extra=metrics)
    return metrics
