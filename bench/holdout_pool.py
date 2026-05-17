"""
bench/holdout_pool.py
=====================
Build the shared holdout prompt pool for routing policy evaluation.

GSM8K test ids >= min_example_id (unseen in Phase B capture).
Hendrycks MATH train (disjoint from MATH-500 used in Phase B).

Optional stratified sampling by prompt length (short / medium / long)
so the workload spans heterogeneous input sizes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from bench.run_quality_capture import _load_gsm8k, _load_math
from distill.dataset_utils import write_jsonl
from utils.config_loader import load_yaml
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

_HENDRICKS_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def _load_hendrycks_math_train(
    *,
    dataset_name: str,
    subjects: Sequence[str],
    prompt_template: str,
) -> List[Dict[str, Any]]:
    from datasets import concatenate_datasets, load_dataset

    parts = [load_dataset(dataset_name, subj, split="train") for subj in subjects]
    ds = concatenate_datasets(parts)
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        from eval.scoring import extract_boxed_answer

        problem = row.get("problem", "")
        solution = row.get("solution", "")
        ref_boxed = extract_boxed_answer(str(solution))
        ref = ref_boxed if ref_boxed else str(solution).strip()
        out.append({
            "benchmark": "math",
            "example_id": str(idx),
            "question": problem,
            "reference_answer": ref,
            "reference_is_boxed": ref_boxed is not None,
            "prompt_text": prompt_template.format(problem=problem),
        })
    return out


def _length_bucket(char_len: int) -> str:
    if char_len < 280:
        return "short"
    if char_len < 520:
        return "medium"
    return "long"


def _stratified_sample(
    examples: List[Dict[str, Any]],
    subset_size: int,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    if subset_size <= 0 or subset_size >= len(examples):
        return list(examples)

    buckets: Dict[str, List[Dict[str, Any]]] = {"short": [], "medium": [], "long": []}
    for ex in examples:
        buckets[_length_bucket(len(ex["prompt_text"]))].append(ex)

    per_bucket = subset_size // 3
    remainder = subset_size % 3
    order = ["short", "medium", "long"]
    picked: List[Dict[str, Any]] = []

    for i, key in enumerate(order):
        pool = buckets[key]
        if not pool:
            continue
        k = per_bucket + (1 if i < remainder else 0)
        k = min(k, len(pool))
        if k <= 0:
            continue
        idx = rng.choice(len(pool), size=k, replace=False)
        picked.extend(pool[int(j)] for j in idx)

    if len(picked) < subset_size:
        remaining = [ex for ex in examples if ex not in picked]
        extra = subset_size - len(picked)
        if remaining and extra > 0:
            idx = rng.choice(len(remaining), size=min(extra, len(remaining)), replace=False)
            picked.extend(remaining[int(j)] for j in idx)

    rng.shuffle(picked)
    return picked


def build_holdout_pool(cfg: Dict[str, Any], *, seed: int = 42) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    benchmarks_cfg = cfg.get("benchmarks") or {}
    pool: List[Dict[str, Any]] = []

    gsm_cfg = benchmarks_cfg.get("gsm8k_holdout") or {}
    if gsm_cfg.get("enabled", True):
        gsm = _load_gsm8k(
            dataset_name=gsm_cfg.get("dataset_name", "openai/gsm8k"),
            split=gsm_cfg.get("dataset_split", "test"),
            subset_size=None,
            prompt_template=gsm_cfg.get("prompt_template", "Question: {question}\n\nAnswer:"),
        )
        min_id = int(gsm_cfg.get("min_example_id", 0))
        gsm = [ex for ex in gsm if int(ex["example_id"]) >= min_id]
        subset = gsm_cfg.get("subset_size")
        if subset is not None and int(subset) > 0:
            gsm = _stratified_sample(gsm, int(subset), rng)
        for ex in gsm:
            ex["benchmark"] = gsm_cfg.get("benchmark_label", "gsm8k")
            ex["length_bucket"] = _length_bucket(len(ex["prompt_text"]))
        pool.extend(gsm)
        logger.info("GSM8K holdout", extra={"count": len(gsm), "min_example_id": min_id})

    math_cfg = benchmarks_cfg.get("hendrycks_math_holdout") or {}
    if math_cfg.get("enabled", True):
        subjects = math_cfg.get("dataset_subjects") or _HENDRICKS_SUBJECTS
        math_rows = _load_hendrycks_math_train(
            dataset_name=math_cfg.get("dataset_name", "EleutherAI/hendrycks_math"),
            subjects=subjects,
            prompt_template=math_cfg.get(
                "prompt_template",
                "Problem: {problem}\n\nSolution:",
            ),
        )
        subset = math_cfg.get("subset_size")
        if subset is not None and int(subset) > 0:
            math_rows = _stratified_sample(math_rows, int(subset), rng)
        for ex in math_rows:
            ex["benchmark"] = math_cfg.get("benchmark_label", "math")
            ex["length_bucket"] = _length_bucket(len(ex["prompt_text"]))
        pool.extend(math_rows)
        logger.info("Hendrycks MATH holdout", extra={"count": len(math_rows)})

    if not pool:
        raise ValueError("Holdout pool is empty; check benchmarks config.")

    for i, ex in enumerate(pool):
        ex["pool_index"] = i
        ex["request_id"] = f"holdout:{ex['benchmark']}:{ex['example_id']}"

    return pool


def export_prompt_pool(
    cfg: Dict[str, Any],
    out_path: Path,
    *,
    seed: int = 42,
    shuffle: bool = True,
) -> Dict[str, Any]:
    pool = build_holdout_pool(cfg, seed=seed)
    rng = np.random.default_rng(seed)
    if shuffle:
        order = np.arange(len(pool))
        rng.shuffle(order)
        pool = [pool[int(i)] for i in order]

    rows = []
    for i, ex in enumerate(pool):
        rows.append({
            "request_id": ex["request_id"],
            "pool_index": i,
            "benchmark": ex["benchmark"],
            "example_id": ex["example_id"],
            "length_bucket": ex.get("length_bucket"),
            "prompt_char_len": len(ex["prompt_text"]),
            "prompt": ex["prompt_text"],
            "prompt_text": ex["prompt_text"],
            "reference_answer": ex["reference_answer"],
            "reference_is_boxed": ex.get("reference_is_boxed", False),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, out_path)

    buckets = {}
    for ex in rows:
        buckets[ex["length_bucket"]] = buckets.get(ex["length_bucket"], 0) + 1
    by_bench = {}
    for ex in rows:
        by_bench[ex["benchmark"]] = by_bench.get(ex["benchmark"], 0) + 1

    manifest = {
        "total": len(rows),
        "by_benchmark": by_bench,
        "by_length_bucket": buckets,
        "prompt_char_len_p50": float(np.median([r["prompt_char_len"] for r in rows])),
        "prompt_char_len_p95": float(np.percentile([r["prompt_char_len"] for r in rows], 95)),
        "seed": seed,
        "path": str(out_path),
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Exported holdout pool", extra=manifest)
    return manifest


def load_prompt_pool(path: Path) -> List[Dict[str, Any]]:
    from distill.dataset_utils import read_jsonl

    rows = read_jsonl(str(path))
    for row in rows:
        if "prompt_text" not in row and "prompt" in row:
            row["prompt_text"] = row["prompt"]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/export holdout prompt pool")
    parser.add_argument("--config", default="configs/routing_eval_holdout.yaml")
    parser.add_argument(
        "--output",
        default="results/routing_eval_holdout/prompt_pool.jsonl",
    )
    args = parser.parse_args()
    setup_logging()
    cfg = load_yaml(args.config)
    seed = int((cfg.get("common") or {}).get("seed", 42))
    export_prompt_pool(cfg, Path(args.output), seed=seed)


if __name__ == "__main__":
    main()
