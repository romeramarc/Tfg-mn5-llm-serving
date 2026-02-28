"""
eval/routerbench.py
===================
RouterBench quality evaluation runner.

RouterBench (rbench/routerbench on HuggingFace) is a diverse QA benchmark
that aggregates queries from MMLU, GSM8K, BBH, and other sources.  Each
entry has a ground-truth answer, enabling accuracy measurement across a
wide range of task types.

The scorer handles two answer formats automatically:
  * Multiple-choice (A/B/C/D) â€” extracts the option letter from the model
    response and compares to the expected letter.
  * Free-text / numeric â€” uses the same normalised numeric + sympy symbolic
    comparison pipeline as the MATH-500 scorer (see eval/scoring.py).

Results include per-source-dataset accuracy breakdown in addition to the
overall accuracy, enabling fine-grained analysis in the thesis.

Usage (standalone)
------------------
    python -m eval.routerbench --config configs/eval.yaml

Normally invoked through ``eval/run_quality.py``.
"""

from __future__ import annotations

import asyncio
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from eval.scoring import (
    compute_accuracy,
    math_answer_match,
    normalise_numeric,
    numeric_match,
)
from utils.logging import get_logger

logger = get_logger(__name__)


# â”€â”€ Dataset loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_ANSWER_KEYS = ("correct_answer", "answer", "label", "expected_answer", "gold")
_SOURCE_KEYS  = ("source_dataset", "source", "dataset", "category")
_PROMPT_KEYS  = ("query", "prompt", "question", "input")


def _get_field(row: dict, candidates: tuple, default: Any = None) -> Any:
    """Return the first matching key from *candidates* found in *row*."""
    for k in candidates:
        if k in row:
            return row[k]
    return default


def load_routerbench(
    dataset_name: str = "rbench/routerbench",
    split: str = "test",
    subset_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load RouterBench from HuggingFace datasets.

    Returns a list of normalised dicts with keys:
      ``prompt``, ``reference_answer``, ``answer_type``, ``source_dataset``.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    logger.info("Loaded RouterBench raw",
                extra={"split": split, "rows": len(ds),
                       "columns": ds.column_names})

    examples: list[dict] = []
    for row in ds:
        prompt    = _get_field(row, _PROMPT_KEYS, "")
        reference = _get_field(row, _ANSWER_KEYS, "")
        source    = _get_field(row, _SOURCE_KEYS, "unknown")

        if not prompt:
            continue

        reference = str(reference).strip()

        # Detect multiple-choice: reference is a single letter A-E
        is_mc = bool(re.fullmatch(r"[A-Ea-e]", reference))

        examples.append({
            "prompt":            prompt,
            "reference_answer":  reference.upper() if is_mc else reference,
            "answer_type":       "multiple_choice" if is_mc else "free_text",
            "source_dataset":    str(source),
        })

    if subset_size is not None and subset_size > 0:
        examples = examples[:subset_size]

    mc_count = sum(1 for e in examples if e["answer_type"] == "multiple_choice")
    ft_count = len(examples) - mc_count
    logger.info("RouterBench processed",
                extra={"total": len(examples),
                       "multiple_choice": mc_count,
                       "free_text": ft_count})
    return examples


# â”€â”€ Answer extraction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_MC_PATTERN = re.compile(
    r"""
    (?:
        \b(?:answer|option|choice)\s*(?:is\s*)?   # "answer is A"
        |^                                          # start of string
        |[.\n]\s*                                  # after period/newline
    )
    ([A-Ea-e])                                     # the letter
    (?:\b|[^a-zA-Z])                               # word boundary
    """,
    re.VERBOSE | re.IGNORECASE | re.MULTILINE,
)


def extract_mc_answer(text: str) -> Optional[str]:
    """Extract the most likely multiple-choice answer letter from *text*.

    Strategy:
    1. Look for explicit "answer is X" / "The answer is (X)" patterns.
    2. Take the **last** standalone capital letter A-E in the response
       (models typically restate their final answer at the end).
    """
    # Explicit patterns first
    explicit = re.findall(
        r'(?:answer|option|choice)\s*(?:is)?\s*[:(]?\s*([A-Ea-e])',
        text, re.IGNORECASE
    )
    if explicit:
        return explicit[-1].upper()
    # Last standalone letter
    standalone = re.findall(r'\b([A-Ea-e])\b', text)
    if standalone:
        return standalone[-1].upper()
    return None


def score_answer(
    predicted_text: str,
    reference: str,
    answer_type: str,
) -> tuple[bool, bool, str | None]:
    """Score a model response against the reference.

    Returns (correct, scorable, predicted_answer).
    """
    if not predicted_text:
        return False, False, None

    if answer_type == "multiple_choice":
        pred = extract_mc_answer(predicted_text)
        if pred is None:
            return False, False, None  # could not parse â†’ unscorable
        return pred == reference.upper(), True, pred

    else:  # free_text
        pred = predicted_text.strip()
        # Numeric fast path
        pred_num = normalise_numeric(pred)
        ref_num  = normalise_numeric(reference)
        if pred_num is not None and ref_num is not None:
            return numeric_match(pred, reference), True, pred
        # Symbolic / normalised string (handles LaTeX math, fractions, etc.)
        matched = math_answer_match(pred, reference)
        return matched, True, pred


# â”€â”€ Async evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def _query_model(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> Dict[str, Any]:
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
    common  = cfg.get("common", {})
    req_cfg = cfg.get("request", {})

    base_url    = common.get("base_url", "http://localhost:8000")
    url         = f"{base_url.rstrip('/')}/v1/completions"
    model       = common.get("model", "")
    max_tokens  = common.get("max_tokens", 512)
    temperature = common.get("temperature", 0.0)
    timeout     = req_cfg.get("timeout_seconds", 60)
    batch_size  = req_cfg.get("batch_size", 32)

    sem     = asyncio.Semaphore(batch_size)
    results: list[dict | None] = [None] * len(examples)

    async def _handle(idx: int, ex: dict) -> None:
        async with sem:
            resp = await _query_model(
                client, url, model, ex["prompt"],
                max_tokens, temperature, timeout,
            )

        correct, scorable, pred = score_answer(
            resp["text"], ex["reference_answer"], ex["answer_type"]
        )

        results[idx] = {
            "index":            idx,
            "source_dataset":   ex["source_dataset"],
            "answer_type":      ex["answer_type"],
            "reference_answer": ex["reference_answer"],
            "predicted_answer": pred,
            "model_response":   resp["text"][:500],
            "correct":          correct,
            "scorable":         scorable,
            "latency_ms":       resp["latency_ms"],
            "error":            resp["error"],
        }

    async with httpx.AsyncClient() as client:
        tasks = [asyncio.create_task(_handle(i, ex))
                 for i, ex in enumerate(examples)]
        await asyncio.gather(*tasks)

    return [r for r in results if r is not None]


# â”€â”€ Public interface â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_routerbench(cfg: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """Execute the RouterBench evaluation and write results to *run_dir*.

    Returns the metrics dict (overall + per-source breakdown).
    """
    bench_cfg = cfg.get("benchmarks", {}).get("routerbench", {})

    examples = load_routerbench(
        dataset_name=bench_cfg.get("dataset_name", "rbench/routerbench"),
        split=bench_cfg.get("dataset_split", "test"),
        subset_size=bench_cfg.get("subset_size"),
    )

    results = asyncio.run(_evaluate_batch(examples, cfg))

    # â”€â”€ Overall metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    metrics = compute_accuracy(results)
    metrics["benchmark"] = "routerbench"
    metrics["model"]     = cfg.get("common", {}).get("model", "")

    # â”€â”€ Per-source breakdown â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    by_source: dict[str, list] = defaultdict(list)
    for r in results:
        by_source[r["source_dataset"]].append(r)

    breakdown: dict[str, dict] = {}
    for source, rows in sorted(by_source.items()):
        bm = compute_accuracy(rows)
        breakdown[source] = {
            "accuracy_pct": bm["accuracy_pct"],
            "correct":      bm["correct"],
            "scorable":     bm["scorable_examples"],
            "total":        bm["total_examples"],
        }
    metrics["per_source_breakdown"] = breakdown

    # â”€â”€ Persist â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    out_dir = run_dir / "routerbench"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "routerbench_results.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    with (out_dir / "routerbench_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    source_rows = [
        {"source": src, **vals}
        for src, vals in breakdown.items()
    ]
    if source_rows:
        with (out_dir / "routerbench_breakdown.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            writer = csv.DictWriter(fh, fieldnames=list(source_rows[0].keys()))
            writer.writeheader()
            writer.writerows(source_rows)

    logger.info("RouterBench evaluation complete", extra={
        "accuracy_pct": metrics["accuracy_pct"],
        "total": metrics["total_examples"],
        "sources": len(breakdown),
    })
    return metrics
