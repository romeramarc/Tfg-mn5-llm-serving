"""
distill/build_residual_dataset.py
=================================
Build a stage-2 residual KD dataset for GSM8K.

Residual examples are defined as:
- teacher is reliable/correct on GSM8K (consensus-aware when available), and
- current student still fails the same prompt.

The output dataset is a configurable mixture of:
- residual GSM8K rows (hard failures), and
- stable rows (teacher-reliable GSM8K solved by student + non-GSM8K rows).

Usage
-----
    python -m distill.build_residual_dataset --config configs/distill_1p5b_focus.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from distill.dataset_utils import read_jsonl, write_jsonl
from eval.scoring import extract_numeric_answer, numeric_match
from utils.config_loader import load_yaml
from utils.logging import get_logger, setup_logging
from utils.reproducibility import (
    collect_metadata,
    make_run_dir,
    save_metadata,
    set_seed,
    snapshot_configs,
)

logger = get_logger(__name__)


def _gsm8k_answer_pattern(cfg: Dict[str, Any]) -> str:
    bench = cfg.get("benchmarks", {}).get("gsm8k", {})
    return str(bench.get("answer_extraction_pattern", r"####\s*([\-\d,\.]+)"))


def _is_trainable_row(row: Dict[str, Any]) -> bool:
    return row.get("error") is None and bool(row.get("teacher_completion"))


def _teacher_reliable_gsm8k(row: Dict[str, Any]) -> bool:
    if row.get("teacher_consensus_correct") is not None:
        return bool(row.get("teacher_consensus_correct"))
    return bool(row.get("teacher_correct"))


async def _query_student(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
) -> Dict[str, Any]:
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
        if body.get("choices"):
            text = body["choices"][0].get("text", "")
        return {
            "text": text,
            "latency_ms": latency_ms,
            "finish_reason": body["choices"][0].get("finish_reason") if body.get("choices") else None,
            "error": None,
        }
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "text": "",
            "latency_ms": latency_ms,
            "finish_reason": None,
            "error": str(exc),
        }


async def _score_student_on_gsm8k(
    rows: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run student inference on GSM8K rows and annotate residual status."""
    residual_cfg = cfg.get("residual_distillation", {}) or {}
    req_cfg = residual_cfg.get("request", {}) or {}

    base_url = str(residual_cfg.get("student_base_url", "http://localhost:8001")).rstrip("/")
    url = f"{base_url}/v1/completions"
    model = str(residual_cfg.get("student_model", cfg.get("training", {}).get("student_model", "")))
    if not model:
        raise ValueError("residual_distillation.student_model is required")

    max_tokens = int(req_cfg.get("max_tokens", cfg.get("generation", {}).get("max_tokens", 896)))
    temperature = float(req_cfg.get("temperature", 0.0))
    top_p = float(req_cfg.get("top_p", 1.0))
    timeout = float(req_cfg.get("timeout_seconds", 180))
    batch_size = int(req_cfg.get("batch_size", 16))
    answer_pattern = _gsm8k_answer_pattern(cfg)

    sem = asyncio.Semaphore(batch_size)
    scored: List[Dict[str, Any]] = []

    async def _score_one(row: Dict[str, Any]) -> None:
        prompt = str(row.get("prompt") or "")
        reference = row.get("reference_answer")

        async with sem:
            response = await _query_student(
                client,
                url,
                model,
                prompt,
                max_tokens,
                temperature,
                top_p,
                timeout,
            )

        student_pred = None
        student_correct = False
        if response["error"] is None:
            student_pred = extract_numeric_answer(str(response.get("text") or ""), answer_pattern)
            if student_pred is not None and reference is not None:
                student_correct = numeric_match(student_pred, str(reference))

        out = dict(row)
        out["residual_student_model"] = model
        out["residual_student_completion"] = response.get("text")
        out["residual_student_predicted_answer"] = student_pred
        out["residual_student_correct"] = bool(student_correct)
        out["residual_student_latency_ms"] = response.get("latency_ms")
        out["residual_student_finish_reason"] = response.get("finish_reason")
        out["residual_student_error"] = response.get("error")
        out["residual_candidate"] = bool(response.get("error") is None and not student_correct)
        scored.append(out)

    async with httpx.AsyncClient() as client:
        tasks = [asyncio.create_task(_score_one(row)) for row in rows]
        await asyncio.gather(*tasks)

    return scored


def _sample_rows(
    residual_pool: List[Dict[str, Any]],
    stable_pool: List[Dict[str, Any]],
    target_size: int,
    residual_fraction: float,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)

    target_size = max(1, target_size)
    desired_residual = int(round(target_size * max(0.0, min(1.0, residual_fraction))))

    residual_n = min(len(residual_pool), desired_residual)
    residual_selected = rng.sample(residual_pool, residual_n) if residual_n > 0 else []

    stable_target = max(0, target_size - len(residual_selected))
    stable_n = min(len(stable_pool), stable_target)
    stable_selected = rng.sample(stable_pool, stable_n) if stable_n > 0 else []

    selected = residual_selected + stable_selected

    # Backfill if one pool is too small.
    if len(selected) < target_size:
        residual_ids = {id(r) for r in residual_selected}
        stable_ids = {id(r) for r in stable_selected}
        residual_left = [r for r in residual_pool if id(r) not in residual_ids]
        stable_left = [r for r in stable_pool if id(r) not in stable_ids]
        backfill = residual_left + stable_left
        rng.shuffle(backfill)
        need = target_size - len(selected)
        selected.extend(backfill[:need])

    rng.shuffle(selected)

    return {
        "rows": selected,
        "residual_selected": len(residual_selected),
        "stable_selected": len(stable_selected),
    }


def run(config_path: str = "configs/distill_1p5b_focus.yaml") -> Path:
    cfg = load_yaml(config_path)
    residual_cfg = cfg.get("residual_distillation", {}) or {}
    if not bool(residual_cfg.get("enabled", False)):
        raise ValueError("residual_distillation.enabled=false; enable it before running stage2 dataset build")

    mix_cfg = residual_cfg.get("mixing", {}) or {}
    seed = int(mix_cfg.get("seed", cfg.get("generation", {}).get("seed", 42)))
    set_seed(seed)
    setup_logging()

    run_dir = make_run_dir("results/distill", tag="residual-build")
    snapshot_configs([config_path], run_dir)
    save_metadata(collect_metadata(seed, cfg), run_dir)

    source_path = Path(
        residual_cfg.get(
            "source_dataset_path",
            cfg.get("training", {}).get("dataset_path", "results/distill/teacher_outputs.jsonl"),
        )
    )
    rows = read_jsonl(source_path)
    valid_rows = [row for row in rows if _is_trainable_row(row)]

    # Stable pool always includes non-GSM8K rows.
    stable_pool: List[Dict[str, Any]] = []
    residual_input_rows: List[Dict[str, Any]] = []
    rejected_teacher_rows = 0

    for row in valid_rows:
        benchmark = str(row.get("benchmark") or "")
        if benchmark != "gsm8k":
            out = dict(row)
            out["residual_stage2_example"] = False
            out["residual_selection_reason"] = "non_gsm8k_stable"
            stable_pool.append(out)
            continue

        if row.get("teacher_accepted_for_kd") is False:
            rejected_teacher_rows += 1
            continue
        if row.get("reference_answer") is None:
            rejected_teacher_rows += 1
            continue
        if not _teacher_reliable_gsm8k(row):
            rejected_teacher_rows += 1
            continue

        residual_input_rows.append(row)

    logger.info(
        "Scoring stage1 student on GSM8K teacher-reliable rows",
        extra={"rows": len(residual_input_rows)},
    )
    scored_rows = asyncio.run(_score_student_on_gsm8k(residual_input_rows, cfg))

    residual_pool: List[Dict[str, Any]] = []
    student_query_errors = 0
    for row in scored_rows:
        if row.get("residual_student_error") is not None:
            student_query_errors += 1
            # Keep problematic samples in stable pool to avoid dropping data unexpectedly.
            out = dict(row)
            out["residual_stage2_example"] = False
            out["residual_selection_reason"] = "student_query_error"
            stable_pool.append(out)
            continue

        if bool(row.get("residual_candidate")):
            out = dict(row)
            out["residual_stage2_example"] = True
            out["residual_selection_reason"] = "teacher_reliable_student_failed"
            residual_pool.append(out)
        else:
            out = dict(row)
            out["residual_stage2_example"] = False
            out["residual_selection_reason"] = "teacher_reliable_student_correct"
            stable_pool.append(out)

    default_target_size = len(valid_rows)
    target_size_cfg = mix_cfg.get("target_size")
    target_size = int(target_size_cfg) if target_size_cfg not in (None, "null") else default_target_size
    residual_fraction = float(mix_cfg.get("residual_fraction", 0.70))

    sampled = _sample_rows(
        residual_pool=residual_pool,
        stable_pool=stable_pool,
        target_size=target_size,
        residual_fraction=residual_fraction,
        seed=seed,
    )
    output_rows = sampled["rows"]

    for row in output_rows:
        row["distillation_stage"] = "stage2_residual"
        row["residual_stage2_selected"] = True

    output_path = Path(
        residual_cfg.get(
            "output_file",
            "results/distill/teacher_outputs_residual_stage2.jsonl",
        )
    )
    write_jsonl(output_rows, output_path)
    write_jsonl(output_rows, run_dir / "residual_dataset.jsonl")

    preview_n = int(residual_cfg.get("preview_examples", 25))
    write_jsonl(residual_pool[:preview_n], run_dir / "residual_pool_preview.jsonl")
    write_jsonl(stable_pool[:preview_n], run_dir / "stable_pool_preview.jsonl")

    consensus_hist: Dict[str, int] = {}
    for row in residual_input_rows:
        size = int(row.get("teacher_consensus_size") or 0)
        consensus_hist[str(size)] = consensus_hist.get(str(size), 0) + 1

    summary = {
        "source_path": str(source_path),
        "output_path": str(output_path),
        "total_source_rows": len(rows),
        "valid_source_rows": len(valid_rows),
        "teacher_rejected_gsm8k_rows": rejected_teacher_rows,
        "gsm8k_teacher_reliable_rows": len(residual_input_rows),
        "residual_pool_rows": len(residual_pool),
        "stable_pool_rows": len(stable_pool),
        "student_query_errors": student_query_errors,
        "target_size": target_size,
        "actual_output_rows": len(output_rows),
        "residual_fraction_target": residual_fraction,
        "residual_selected": sampled["residual_selected"],
        "stable_selected": sampled["stable_selected"],
        "residual_fraction_actual": (
            sampled["residual_selected"] / len(output_rows) if output_rows else 0.0
        ),
        "consensus_size_histogram_teacher_reliable": consensus_hist,
    }
    with (run_dir / "residual_build_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("Residual dataset build complete", extra=summary)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stage-2 residual KD dataset")
    parser.add_argument("--config", default="configs/distill_1p5b_focus.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
