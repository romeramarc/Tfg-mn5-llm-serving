"""
eval/arc_eval.py
================
ARC-Challenge quality evaluation runner.

ARC-Challenge (allenai/ai2_arc, config=ARC-Challenge) is a set of 1172
science questions in 4-choice multiple-choice format.  It is widely used
as a commonsense-reasoning benchmark complementary to mathematical benchmarks
such as GSM8K and MATH-500.

Scoring
-------
The predicted answer letter (A/B/C/D) is extracted from the model response
using the same two-stage extractor used in the RouterBench scorer:
  1. Explicit pattern ``The answer is [A-D]`` / ``Answer: [A-D]``
  2. Last standalone capital letter in the response

Results include per-category accuracy (Life Science, Physical Science, etc.)
when that metadata is available in the split.

Usage (standalone)
------------------
    python -m eval.arc_eval --config configs/eval.yaml

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

from eval.scoring import compute_accuracy
from utils.logging import get_logger

logger = get_logger(__name__)


# ── Answer extraction ──────────────────────────────────────

_EXPLICIT_PAT = re.compile(
    r"""(?:
        (?:the\s+)?answer\s+is\s*[:\-]?\s*\(?([A-D])\)?  |
        \b([A-D])\)\s                                       |
        ^([A-D])$
    )""",
    re.IGNORECASE | re.VERBOSE | re.MULTILINE,
)


def extract_mc_answer(text: str) -> Optional[str]:
    """Return the predicted option letter (A-D) from *text*, or None."""
    for m in _EXPLICIT_PAT.finditer(text):
        letter = next(g for g in m.groups() if g is not None)
        return letter.upper()
    # Fallback: last standalone capital letter A-D
    tokens = re.findall(r'\b([A-D])\b', text, re.IGNORECASE)
    return tokens[-1].upper() if tokens else None


# ── Dataset loading ────────────────────────────────────────

def load_arc(
    dataset_name: str = "allenai/ai2_arc",
    dataset_config: str = "ARC-Challenge",
    split: str = "test",
    subset_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load ARC-Challenge from HuggingFace and normalise to a flat list.

    Each returned dict has keys:
        ``id``, ``prompt``, ``reference_answer``, ``category``
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, dataset_config, split=split)
    logger.info("Loaded ARC-Challenge raw",
                extra={"split": split, "rows": len(ds)})

    examples: List[Dict[str, Any]] = []
    for row in ds:
        choices = row["choices"]
        labels: List[str] = choices["label"]   # ['A', 'B', 'C', 'D'] or ['1','2','3','4']
        texts: List[str]  = choices["text"]

        # Normalise numeric labels to letters
        label_map = {"1": "A", "2": "B", "3": "C", "4": "D",
                     "A": "A", "B": "B", "C": "C", "D": "D", "E": "E"}
        answers_block = "\n".join(
            f"{label_map.get(lbl, lbl)}) {txt}"
            for lbl, txt in zip(labels, texts)
        )

        prompt = (
            "Answer the following multiple-choice science question.\n"
            "Write only the letter of the correct answer (A, B, C, or D).\n\n"
            f"Question: {row['question']}\n"
            f"{answers_block}\n\n"
            "Answer:"
        )

        answer_key = row["answerKey"]
        ref = label_map.get(answer_key, answer_key)

        examples.append({
            "id": row.get("id", ""),
            "prompt": prompt,
            "reference_answer": ref,
            "answer_type": "mc",
            "category": row.get("category", "arc-challenge"),
        })

    if subset_size and subset_size > 0:
        examples = examples[:subset_size]

    logger.info("ARC-Challenge examples prepared", extra={"n": len(examples)})
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
        text = body.get("choices", [{}])[0].get("text", "")
        return {"text": text, "latency_ms": latency_ms, "error": None}
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {"text": "", "latency_ms": latency_ms, "error": str(exc)}


async def _evaluate_batch(
    examples: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    common    = cfg.get("common", {})
    req_cfg   = cfg.get("request", {})

    base_url    = common.get("base_url", "http://localhost:8000")
    url         = f"{base_url.rstrip('/')}/v1/completions"
    model       = common.get("model", "")
    max_tokens  = common.get("max_tokens", 16)   # short: only need one letter
    temperature = common.get("temperature", 0.0)
    timeout     = req_cfg.get("timeout_seconds", 60)
    batch_size  = req_cfg.get("batch_size", 32)

    sem     = asyncio.Semaphore(batch_size)
    results : list[dict] = []

    async def _handle(idx: int, ex: dict) -> None:
        async with sem:
            resp = await _query_model(
                client, url, model, ex["prompt"],
                max_tokens, temperature, timeout
            )
        pred_letter = extract_mc_answer(resp["text"]) if not resp["error"] else None
        ref_letter  = ex["reference_answer"]
        correct     = (pred_letter == ref_letter) if pred_letter is not None else False
        results.append({
            "index":              idx,
            "id":                 ex["id"],
            "reference_answer":   ref_letter,
            "model_response":     resp["text"],
            "predicted_answer":   pred_letter,
            "correct":            correct,
            "scorable":           pred_letter is not None,
            "latency_ms":         resp["latency_ms"],
            "error":              resp["error"],
            "category":           ex.get("category", ""),
        })

    async with httpx.AsyncClient() as client:
        tasks = [asyncio.create_task(_handle(i, ex))
                 for i, ex in enumerate(examples)]
        await asyncio.gather(*tasks)

    results.sort(key=lambda r: r["index"])
    return results


# ── Public interface ───────────────────────────────────────

def run_arc(cfg: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """Execute ARC-Challenge evaluation and write results to *run_dir*.

    Returns the metrics dict.
    """
    bench_cfg = cfg.get("benchmarks", {}).get("arc_challenge", {})

    examples = load_arc(
        dataset_name   = bench_cfg.get("dataset_name", "allenai/ai2_arc"),
        dataset_config = bench_cfg.get("dataset_config", "ARC-Challenge"),
        split          = bench_cfg.get("dataset_split", "test"),
        subset_size    = bench_cfg.get("subset_size"),
    )

    results = asyncio.run(_evaluate_batch(examples, cfg))

    # ── Per-category breakdown ─────────────────────────────
    from collections import defaultdict as _dd  # already imported at top-level
    breakdown: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        cat = r.get("category") or "arc-challenge"
        breakdown[cat]["total"]   += 1
        breakdown[cat]["correct"] += int(bool(r.get("correct")))

    # ── Global metrics ─────────────────────────────────────
    metrics = compute_accuracy(results)
    metrics["benchmark"] = "arc_challenge"
    metrics["model"]     = cfg.get("common", {}).get("model", "")

    # ── Persist ────────────────────────────────────────────
    out_dir = run_dir / "arc_challenge"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "arc_results.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    with (out_dir / "arc_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    with (out_dir / "arc_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    # Per-category CSV
    cat_rows = [
        {
            "category": cat,
            "correct":  v["correct"],
            "total":    v["total"],
            "accuracy_pct": round(100.0 * v["correct"] / v["total"], 2)
            if v["total"] > 0 else None,
        }
        for cat, v in sorted(breakdown.items())
    ]
    if cat_rows:
        with (out_dir / "arc_breakdown.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            writer2 = csv.DictWriter(fh, fieldnames=list(cat_rows[0].keys()))
            writer2.writeheader()
            writer2.writerows(cat_rows)

    logger.info("ARC-Challenge evaluation complete", extra={
        "accuracy_pct": metrics["accuracy_pct"],
        "total":        metrics["total_examples"],
    })
    return metrics
