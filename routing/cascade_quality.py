"""
routing/cascade_quality.py
==========================
Quality evaluation for routing/cascade policies.

This module evaluates a routing policy (including the fixed 3-tier cascade)
on the same benchmark protocol used by eval/run_quality.py, so results remain
comparable to teacher-only and single-model baselines.

Usage
-----
    python -m routing.cascade_quality \
        --eval-config configs/eval.yaml \
        --routing-config configs/routing_phase2.yaml \
        --role cascade_phase2
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from bench.metrics import save_json
from eval.arc_eval import extract_mc_answer, load_arc
from eval.gsm8k import load_gsm8k
from eval.math_eval import load_math
from eval.scoring import (
    compute_accuracy,
    extract_boxed_answer,
    extract_numeric_answer,
    math_answer_match,
    numeric_match,
)
from routing.policies import POLICIES, RoutingDecision
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


def _write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_metrics_csv(metrics: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def _write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _is_final_error(record: Dict[str, Any]) -> bool:
    attempts = record.get("attempts") or []
    if attempts:
        return bool(attempts[-1].get("error"))
    return "error" in str(record.get("reason", ""))


def _flatten_attempts(result_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for row in result_rows:
        attempts = row.get("attempts") or []
        for attempt_index, attempt in enumerate(attempts):
            rec = {
                "request_id": row.get("request_id"),
                "benchmark": row.get("benchmark"),
                "index": row.get("index"),
                "attempt_index": attempt_index,
                "final_selected_model": row.get("selected_model"),
                "final_reason": row.get("route_reason"),
                "final_latency_ms": row.get("latency_ms"),
            }
            rec.update(attempt)
            flat.append(rec)
    return flat


async def _route_prompts(
    prompts: List[str],
    request_prefix: str,
    policy_fn,
    base_ctx: Dict[str, Any],
    concurrency: int,
) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    records: List[Dict[str, Any] | None] = [None] * len(prompts)

    async with httpx.AsyncClient() as client:

        async def _handle(idx: int, prompt: str) -> None:
            req_ctx = dict(base_ctx)
            req_ctx["request_id"] = f"{request_prefix}-{idx:06d}"
            req_ctx["prompt_metadata"] = {"benchmark": request_prefix, "index": idx}

            async with sem:
                decision: RoutingDecision = await policy_fn(client, prompt, req_ctx)

            records[idx] = asdict(decision)

        tasks = [
            asyncio.create_task(_handle(i, prompt))
            for i, prompt in enumerate(prompts)
        ]
        await asyncio.gather(*tasks)

    return [r for r in records if r is not None]


async def _evaluate_gsm8k(
    eval_cfg: Dict[str, Any],
    policy_fn,
    base_ctx: Dict[str, Any],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    bench_cfg = eval_cfg.get("benchmarks", {}).get("gsm8k", {})
    examples = load_gsm8k(
        dataset_name=bench_cfg.get("dataset_name", "openai/gsm8k"),
        split=bench_cfg.get("dataset_split", "test"),
        subset_size=bench_cfg.get("subset_size"),
    )
    prompt_template = bench_cfg.get("prompt_template", "Question: {question}\n\nAnswer:")
    answer_pattern = bench_cfg.get("answer_extraction_pattern", r"####\s*([\-\d,\.]+)")

    prompts = [prompt_template.replace("{question}", ex["question"]) for ex in examples]
    records = await _route_prompts(
        prompts,
        request_prefix="gsm8k",
        policy_fn=policy_fn,
        base_ctx=base_ctx,
        concurrency=eval_cfg.get("request", {}).get("batch_size", 16),
    )

    result_rows: List[Dict[str, Any]] = []
    for idx, (ex, rec) in enumerate(zip(examples, records)):
        predicted_raw = extract_numeric_answer(rec.get("response_text", ""), answer_pattern)
        final_error = _is_final_error(rec)
        scorable = not final_error and predicted_raw is not None
        correct = bool(scorable and numeric_match(predicted_raw, ex["reference_answer"]))

        result_rows.append({
            "benchmark": "gsm8k",
            "index": idx,
            "request_id": rec.get("request_id"),
            "question": ex["question"],
            "reference_answer": ex["reference_answer"],
            "model_response": rec.get("response_text", ""),
            "predicted_answer": predicted_raw,
            "correct": correct,
            "scorable": scorable,
            "latency_ms": rec.get("latency_ms", 0.0),
            "selected_model": rec.get("selected_model"),
            "route_reason": rec.get("reason"),
            "route_confidence": rec.get("confidence"),
            "attempt_count": len(rec.get("attempts") or []),
            "attempts": rec.get("attempts") or [],
            "error": rec.get("attempts", [{}])[-1].get("error") if rec.get("attempts") else None,
        })

    metrics = compute_accuracy(result_rows)
    metrics["benchmark"] = "gsm8k"
    metrics["model"] = "cascade_policy"
    return metrics, result_rows


async def _evaluate_math(
    eval_cfg: Dict[str, Any],
    policy_fn,
    base_ctx: Dict[str, Any],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    bench_cfg = eval_cfg.get("benchmarks", {}).get("math", {})
    examples = load_math(
        dataset_name=bench_cfg.get("dataset_name", "HuggingFaceH4/MATH-500"),
        split=bench_cfg.get("dataset_split", "test"),
        subset_size=bench_cfg.get("subset_size"),
    )
    prompt_template = bench_cfg.get(
        "prompt_template",
        "Solve the following mathematics problem. Put your final answer inside \\boxed{{}}.\n\nProblem: {problem}\n\nSolution:",
    )

    prompts = [prompt_template.replace("{problem}", ex["problem"]) for ex in examples]
    records = await _route_prompts(
        prompts,
        request_prefix="math",
        policy_fn=policy_fn,
        base_ctx=base_ctx,
        concurrency=eval_cfg.get("request", {}).get("batch_size", 16),
    )

    result_rows: List[Dict[str, Any]] = []
    for idx, (ex, rec) in enumerate(zip(examples, records)):
        final_error = _is_final_error(rec)
        predicted_boxed = extract_boxed_answer(rec.get("response_text", ""))
        ambiguity_reason: Optional[str] = None

        if final_error:
            scorable = False
            correct = False
            ambiguity_reason = "request_error"
        elif predicted_boxed is None:
            scorable = False
            correct = False
            ambiguity_reason = "no_boxed_answer_in_response"
        elif not ex.get("reference_is_boxed", True):
            scorable = False
            correct = False
            ambiguity_reason = "reference_not_boxed"
        else:
            scorable = True
            correct = math_answer_match(predicted_boxed, ex["reference_answer"])

        result_rows.append({
            "benchmark": "math",
            "index": idx,
            "request_id": rec.get("request_id"),
            "problem": ex.get("problem", "")[:200],
            "level": ex.get("level", ""),
            "type": ex.get("type", ""),
            "reference_answer": ex["reference_answer"],
            "model_response": rec.get("response_text", ""),
            "predicted_answer": predicted_boxed,
            "correct": correct,
            "scorable": scorable,
            "ambiguity_reason": ambiguity_reason,
            "latency_ms": rec.get("latency_ms", 0.0),
            "selected_model": rec.get("selected_model"),
            "route_reason": rec.get("reason"),
            "route_confidence": rec.get("confidence"),
            "attempt_count": len(rec.get("attempts") or []),
            "attempts": rec.get("attempts") or [],
            "error": rec.get("attempts", [{}])[-1].get("error") if rec.get("attempts") else None,
        })

    metrics = compute_accuracy(result_rows)
    metrics["benchmark"] = "math"
    metrics["model"] = "cascade_policy"
    return metrics, result_rows


async def _evaluate_arc(
    eval_cfg: Dict[str, Any],
    policy_fn,
    base_ctx: Dict[str, Any],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    bench_cfg = eval_cfg.get("benchmarks", {}).get("arc_challenge", {})
    examples = load_arc(
        dataset_name=bench_cfg.get("dataset_name", "allenai/ai2_arc"),
        dataset_config=bench_cfg.get("dataset_config", "ARC-Challenge"),
        split=bench_cfg.get("dataset_split", "test"),
        subset_size=bench_cfg.get("subset_size"),
    )

    prompts = [ex["prompt"] for ex in examples]
    records = await _route_prompts(
        prompts,
        request_prefix="arc",
        policy_fn=policy_fn,
        base_ctx=base_ctx,
        concurrency=eval_cfg.get("request", {}).get("batch_size", 16),
    )

    result_rows: List[Dict[str, Any]] = []
    for idx, (ex, rec) in enumerate(zip(examples, records)):
        final_error = _is_final_error(rec)
        pred_letter = None if final_error else extract_mc_answer(rec.get("response_text", ""))
        scorable = pred_letter is not None
        correct = bool(scorable and pred_letter == ex["reference_answer"])

        result_rows.append({
            "benchmark": "arc_challenge",
            "index": idx,
            "request_id": rec.get("request_id"),
            "id": ex.get("id", ""),
            "reference_answer": ex["reference_answer"],
            "model_response": rec.get("response_text", ""),
            "predicted_answer": pred_letter,
            "correct": correct,
            "scorable": scorable,
            "category": ex.get("category", ""),
            "latency_ms": rec.get("latency_ms", 0.0),
            "selected_model": rec.get("selected_model"),
            "route_reason": rec.get("reason"),
            "route_confidence": rec.get("confidence"),
            "attempt_count": len(rec.get("attempts") or []),
            "attempts": rec.get("attempts") or [],
            "error": rec.get("attempts", [{}])[-1].get("error") if rec.get("attempts") else None,
        })

    metrics = compute_accuracy(result_rows)
    metrics["benchmark"] = "arc_challenge"
    metrics["model"] = "cascade_policy"
    return metrics, result_rows


def _save_benchmark_artifacts(
    run_dir: Path,
    benchmark: str,
    metrics: Dict[str, Any],
    rows: List[Dict[str, Any]],
) -> None:
    out_dir = run_dir / benchmark
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(rows, out_dir / f"{benchmark}_results.jsonl")
    save_json(metrics, out_dir / f"{benchmark}_metrics.json")
    _write_metrics_csv(metrics, out_dir / f"{benchmark}_metrics.csv")

    unscorable = [r for r in rows if not r.get("scorable", True)]
    if unscorable:
        _write_jsonl(unscorable, out_dir / f"{benchmark}_unscorable.jsonl")

    attempt_rows = _flatten_attempts(rows)
    if attempt_rows:
        _write_jsonl(attempt_rows, out_dir / f"{benchmark}_attempts.jsonl")


async def _run_async(
    eval_cfg: Dict[str, Any],
    policy_name: str,
    policy_fn,
    base_ctx: Dict[str, Any],
    run_dir: Path,
) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    benchmarks_cfg = eval_cfg.get("benchmarks", {})

    if benchmarks_cfg.get("gsm8k", {}).get("enabled", False):
        logger.info("Starting cascade GSM8K evaluation", extra={"policy": policy_name})
        metrics, rows = await _evaluate_gsm8k(eval_cfg, policy_fn, base_ctx)
        _save_benchmark_artifacts(run_dir, "gsm8k", metrics, rows)
        summary_rows.append(metrics)

    if benchmarks_cfg.get("math", {}).get("enabled", False):
        logger.info("Starting cascade MATH evaluation", extra={"policy": policy_name})
        metrics, rows = await _evaluate_math(eval_cfg, policy_fn, base_ctx)
        _save_benchmark_artifacts(run_dir, "math", metrics, rows)
        summary_rows.append(metrics)

    if benchmarks_cfg.get("arc_challenge", {}).get("enabled", False):
        logger.info("Starting cascade ARC-Challenge evaluation", extra={"policy": policy_name})
        metrics, rows = await _evaluate_arc(eval_cfg, policy_fn, base_ctx)
        _save_benchmark_artifacts(run_dir, "arc_challenge", metrics, rows)
        summary_rows.append(metrics)

    return summary_rows


def run(
    eval_config_path: str = "configs/eval.yaml",
    routing_config_path: str = "configs/routing_phase2.yaml",
    role: str = "cascade_phase2",
    policy_override: str | None = None,
) -> Path:
    eval_cfg = load_yaml(eval_config_path)
    routing_cfg = load_yaml(routing_config_path)

    common = eval_cfg.get("common", {})
    seed = int(common.get("seed", 42))
    set_seed(seed)
    setup_logging()

    policy_name = policy_override or routing_cfg.get("active_policy", "cascade_three_tier")
    if policy_name not in POLICIES:
        raise ValueError(
            f"Unknown policy '{policy_name}'. Available: {list(POLICIES.keys())}"
        )
    policy_fn = POLICIES[policy_name]

    run_dir = make_run_dir(
        common.get("results_base_dir", "results/quality"),
        tag=f"quality-{role}",
    )
    snapshot_configs(
        [eval_config_path, routing_config_path, "configs/models.yaml", "configs/serving.yaml"],
        run_dir,
    )
    save_metadata(
        collect_metadata(seed, {"eval": eval_cfg, "routing": routing_cfg}),
        run_dir,
    )

    base_ctx: Dict[str, Any] = {
        "endpoints": routing_cfg.get("endpoints", {}),
        "max_tokens": common.get("max_tokens", 1024),
        "temperature": common.get("temperature", 0.0),
        "policy_name": policy_name,
    }
    base_ctx.update(routing_cfg.get("policies", {}).get(policy_name, {}))

    summary_rows = asyncio.run(
        _run_async(
            eval_cfg=eval_cfg,
            policy_name=policy_name,
            policy_fn=policy_fn,
            base_ctx=base_ctx,
            run_dir=run_dir,
        )
    )

    save_json(summary_rows, run_dir / "quality_summary.json")
    _write_summary_csv(summary_rows, run_dir / "quality_summary.csv")

    logger.info(
        "Cascade quality evaluation complete",
        extra={
            "run_dir": str(run_dir),
            "policy": policy_name,
            "benchmarks_run": len(summary_rows),
        },
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Cascade quality evaluation")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--routing-config", default="configs/routing_phase2.yaml")
    parser.add_argument("--role", default="cascade_phase2")
    parser.add_argument("--policy", default=None)
    args = parser.parse_args()

    run(
        eval_config_path=args.eval_config,
        routing_config_path=args.routing_config,
        role=args.role,
        policy_override=args.policy,
    )


if __name__ == "__main__":
    main()
