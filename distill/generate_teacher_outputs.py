"""
distill/generate_teacher_outputs.py
====================================
Query the teacher model via its OpenAI-compatible ``/v1/completions``
endpoint and persist (prompt, response, metadata) triples as JSONL.

The output file becomes the training dataset for the student SFT step.
Optionally, records can be quality-filtered (e.g. keep only teacher-correct
answers per benchmark) before writing the training JSONL.

**Train/test separation:**  Prompts are sourced from the **train** splits
of each benchmark (GSM8K train, MATH train) to prevent data leakage.
The *test* splits (GSM8K test, MATH-500) are reserved exclusively for
post-distillation quality evaluation. A configurable
``extra_prompts_file`` can add supplementary prompts beyond the benchmarks.

JSONL output schema (one record per line)::

    {
        "id":                     "<benchmark>-<index>",
        "benchmark":              "gsm8k" | "math" | "extra",
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
from collections import Counter
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from distill.dataset_utils import write_jsonl
from eval.scoring import (
    extract_boxed_answer,
    extract_numeric_answer,
    math_answer_match,
    numeric_match,
)
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
    """Collect prompts from GSM8K **train** set (avoids test-set leakage)."""
    from datasets import load_dataset

    bench = cfg.get("benchmarks", {}).get("gsm8k", {})
    if not bench.get("enabled", True):
        return []

    ds = load_dataset(
        bench.get("dataset_name", "openai/gsm8k"),
        "main",
        split=bench.get("dataset_split", "train"),  # TRAIN split for KD
    )
    template = bench.get("prompt_template",
        "Solve the following math problem step by step.\n"
        "Put your final numeric answer after ####.\n\n"
        "Question: {question}\n\nAnswer:")
    answer_pattern = bench.get("answer_extraction_pattern", r"####\s*([\-\d,\.]+)")
    prompts = []
    for i, row in enumerate(ds):
        reference_answer = extract_numeric_answer(str(row.get("answer", "")), answer_pattern)
        prompts.append({
            "id": f"gsm8k-{i}",
            "benchmark": "gsm8k",
            "prompt": template.format(question=row["question"]),
            "reference_answer": reference_answer,
        })
    subset = bench.get("subset_size")
    if subset and subset > 0:
        prompts = prompts[:subset]
    logger.info("Collected GSM8K prompts", extra={"n": len(prompts)})
    return prompts


def _collect_math_prompts(cfg: dict) -> List[Dict[str, Any]]:
    """Collect prompts from MATH **train** set (avoids test-set leakage).

    Uses ``EleutherAI/hendrycks_math`` train split (~7500 items) by default.
    All 7 subject configs are concatenated to reproduce the full MATH dataset
    (``lighteval/MATH`` was removed from the Hub).
    The eval-only ``HuggingFaceH4/MATH-500`` test split is reserved
    exclusively for post-distillation quality evaluation.
    """
    from datasets import concatenate_datasets, load_dataset

    bench = cfg.get("benchmarks", {}).get("math", {})
    if not bench.get("enabled", True):
        return []

    dataset_name = bench.get("dataset_name", "EleutherAI/hendrycks_math")
    dataset_split = bench.get("dataset_split", "train")
    if dataset_name in {"hendrycks/competition_math", "EleutherAI/hendrycks_math"}:
        # No top-level config — load and concatenate all 7 subjects
        _SUBJECTS = [
            "algebra", "counting_and_probability", "geometry",
            "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
        ]
        ds = concatenate_datasets([
            load_dataset(dataset_name, subj, split=dataset_split)
            for subj in _SUBJECTS
        ])
    else:
        ds = load_dataset(dataset_name, split=dataset_split)
    template = bench.get("prompt_template",
        "Solve the following mathematics problem.\n"
        "Put your final answer inside \\boxed{{}}.\n\n"
        "Problem: {problem}\n\nSolution:")
    prompts = []
    for i, row in enumerate(ds):
        reference_answer = extract_boxed_answer(str(row.get("solution", "")))
        prompts.append({
            "id": f"math-{i}",
            "benchmark": "math",
            "prompt": template.format(problem=row["problem"]),
            "reference_answer": reference_answer,
            "reference_is_boxed": bool(reference_answer),
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
    """Gather prompts from all configured benchmark TRAIN sources + extras.

    Only GSM8K and MATH train splits are used for KD data generation.
    ARC-Challenge is not used in this project's evaluation pipeline.
    """
    all_prompts: List[Dict[str, Any]] = []
    all_prompts.extend(_collect_gsm8k_prompts(cfg))
    all_prompts.extend(_collect_math_prompts(cfg))

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


def _clean_completion_for_length(text: str) -> str:
    """Normalise whitespace to compare candidate completion lengths."""
    return " ".join(text.split())


def _pick_consensus_target(
    samples: List[Dict[str, Any]],
    consensus_value: Optional[str],
    strategy: str,
) -> Tuple[Optional[int], str]:
    """Choose a target completion among sampled teacher outputs."""
    indexed = list(enumerate(samples))

    if consensus_value is not None:
        candidates = [
            (idx, s)
            for idx, s in indexed
            if s.get("completion") and s.get("predicted_answer") == consensus_value
        ]
    else:
        candidates = [(idx, s) for idx, s in indexed if s.get("completion")]

    if not candidates:
        return None, "none_available"

    if strategy not in {"shortest_clean", "first"}:
        strategy = "shortest_clean"

    if strategy == "first":
        return candidates[0][0], "first"

    best_idx, _ = min(
        candidates,
        key=lambda it: (
            len(_clean_completion_for_length(str(it[1].get("completion") or ""))),
            len(str(it[1].get("completion") or "")),
            it[0],
        ),
    )
    return best_idx, "shortest_clean"


async def _generate_single_record(
    client: httpx.AsyncClient,
    item: Dict[str, Any],
    url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    gen_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate one teacher completion (default path for non-GSM8K consensus)."""
    res = await _query_teacher(
        client, url, item["prompt"], model, max_tokens, temperature, top_p, timeout,
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
        "teacher_accepted_for_kd": True,
    }
    for key, value in item.items():
        if key not in {"id", "benchmark", "prompt"}:
            record[key] = value
    return record


async def _generate_gsm8k_consensus_record(
    client: httpx.AsyncClient,
    item: Dict[str, Any],
    cfg: Dict[str, Any],
    url: str,
    model: str,
    timeout: float,
    base_gen_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate multiple teacher samples for GSM8K and derive consensus target."""
    gen = cfg.get("generation", {})
    scfg = gen.get("gsm8k_self_consistency", {}) or {}

    num_samples = max(1, int(scfg.get("num_samples", 5)))
    min_consensus = max(1, int(scfg.get("min_consensus", 3)))
    sample_temperature = float(scfg.get("sample_temperature", 0.6))
    sample_top_p = float(scfg.get("sample_top_p", 0.95))
    sample_max_tokens = int(scfg.get("sample_max_tokens", gen.get("max_tokens", 1024)))
    require_consensus_correct = bool(scfg.get("require_consensus_correct", True))
    target_selection = str(scfg.get("target_selection", "shortest_clean"))

    answer_pattern = _gsm8k_answer_pattern(cfg)
    teacher_samples: List[Dict[str, Any]] = []

    for sample_idx in range(num_samples):
        res = await _query_teacher(
            client,
            url,
            item["prompt"],
            model,
            sample_max_tokens,
            sample_temperature,
            sample_top_p,
            timeout,
        )
        completion = str(res.get("teacher_completion") or "")
        predicted = None
        if completion and res.get("error") is None:
            predicted = extract_numeric_answer(completion, answer_pattern)
        teacher_samples.append({
            "sample_index": sample_idx,
            "completion": completion,
            "predicted_answer": predicted,
            "latency_ms": res.get("latency_ms"),
            "finish_reason": res.get("finish_reason"),
            "error": res.get("error"),
        })

    counts = Counter(
        str(s["predicted_answer"])
        for s in teacher_samples
        if s.get("predicted_answer") is not None
    )
    consensus_value: Optional[str] = None
    consensus_size = 0
    if counts:
        consensus_value, consensus_size = counts.most_common(1)[0]

    consensus_reached = bool(consensus_value is not None and consensus_size >= min_consensus)
    reference = item.get("reference_answer")
    consensus_correct = bool(
        consensus_reached
        and reference is not None
        and consensus_value is not None
        and numeric_match(consensus_value, str(reference))
    )
    accepted_for_kd = consensus_correct if require_consensus_correct else consensus_reached

    selected_idx, selected_by = _pick_consensus_target(
        teacher_samples, consensus_value, target_selection,
    )
    selected_completion = None
    selected_predicted = None
    selected_finish_reason = None
    selected_latency = None
    if selected_idx is not None:
        chosen = teacher_samples[selected_idx]
        selected_completion = chosen.get("completion")
        selected_predicted = chosen.get("predicted_answer")
        selected_finish_reason = chosen.get("finish_reason")
        selected_latency = chosen.get("latency_ms")

    record_error = None
    if selected_completion is None:
        record_error = "all_teacher_samples_failed"

    record = {
        "id": item["id"],
        "benchmark": item["benchmark"],
        "prompt": item["prompt"],
        "teacher_completion": selected_completion,
        "teacher_model": model,
        "generation_parameters": {
            **base_gen_params,
            "self_consistency": {
                "enabled": True,
                "num_samples": num_samples,
                "min_consensus": min_consensus,
                "sample_temperature": sample_temperature,
                "sample_top_p": sample_top_p,
                "sample_max_tokens": sample_max_tokens,
                "target_selection": target_selection,
            },
        },
        "latency_ms": selected_latency,
        "finish_reason": selected_finish_reason,
        "error": record_error,
        "teacher_samples": teacher_samples,
        "teacher_sampled_completions": [s.get("completion") for s in teacher_samples],
        "teacher_extracted_candidates": [s.get("predicted_answer") for s in teacher_samples],
        "teacher_consensus_counts": dict(counts),
        "teacher_consensus_value": consensus_value,
        "teacher_consensus_size": consensus_size,
        "teacher_consensus_reached": consensus_reached,
        "teacher_consensus_min_required": min_consensus,
        "teacher_consensus_correct": consensus_correct,
        "teacher_accepted_for_kd": accepted_for_kd,
        "teacher_selected_sample_index": selected_idx,
        "teacher_selected_by": selected_by,
        "teacher_predicted_answer": consensus_value if consensus_value is not None else selected_predicted,
        "teacher_scorable": consensus_reached,
        "teacher_correct": consensus_correct,
    }
    for key, value in item.items():
        if key not in {"id", "benchmark", "prompt"}:
            record[key] = value
    return record


async def _generate_all(
    prompts: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Batch-query the teacher asynchronously."""
    gen = cfg.get("generation", {})
    base_url = gen.get("teacher_base_url", "http://localhost:8000")
    url = f"{base_url.rstrip('/')}/v1/completions"
    model = gen.get("teacher_model", "")
    max_tokens = int(gen.get("max_tokens", 1024))
    temperature = float(gen.get("temperature", 0.0))
    top_p = float(gen.get("top_p", 1.0))
    batch_size = int(gen.get("batch_size", 32))
    timeout = float(gen.get("timeout_seconds", 180))

    gsm8k_consensus_cfg = gen.get("gsm8k_self_consistency", {}) or {}
    gsm8k_consensus_enabled = bool(gsm8k_consensus_cfg.get("enabled", False))

    gen_params = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    sem = asyncio.Semaphore(batch_size)
    results: List[Dict[str, Any]] = []

    async def _bounded(item: Dict[str, Any]) -> None:
        async with sem:
            if item.get("benchmark") == "gsm8k" and gsm8k_consensus_enabled:
                record = await _generate_gsm8k_consensus_record(
                    client,
                    item,
                    cfg,
                    url,
                    model,
                    timeout,
                    gen_params,
                )
            else:
                record = await _generate_single_record(
                    client,
                    item,
                    url,
                    model,
                    max_tokens,
                    temperature,
                    top_p,
                    timeout,
                    gen_params,
                )
        results.append(record)

    async with httpx.AsyncClient() as client:
        tasks = [asyncio.create_task(_bounded(item)) for item in prompts]
        await asyncio.gather(*tasks)

    return results


def _gsm8k_answer_pattern(cfg: Dict[str, Any]) -> str:
    bench = cfg.get("benchmarks", {}).get("gsm8k", {})
    return str(bench.get("answer_extraction_pattern", r"####\s*([\-\d,\.]+)"))


def _default_final_line_pattern(benchmark: str) -> str:
    if benchmark == "gsm8k":
        return r"(?m)^####\s*([\-\d,\.]+)\s*$"
    return r"(?m)^####\s*(.+?)\s*$"


def _final_line_pattern(cfg: Dict[str, Any], benchmark: str) -> str:
    bench_cfg = cfg.get("benchmarks", {}).get(benchmark, {})
    return str(bench_cfg.get("final_line_pattern", _default_final_line_pattern(benchmark)))


def _final_answer_parser(cfg: Dict[str, Any], benchmark: str) -> str:
    bench_cfg = cfg.get("benchmarks", {}).get(benchmark, {})
    parser = bench_cfg.get("final_answer_parser")
    if parser:
        return str(parser).lower()
    if benchmark == "gsm8k":
        return "numeric"
    if benchmark == "math":
        return "boxed"
    return "final_line"


def _extract_final_line_components(text: str, pattern: str) -> Dict[str, Any]:
    components: Dict[str, Any] = {
        "reasoning": text.strip() if text else "",
        "final_line": None,
        "final_answer_raw": None,
        "final_line_is_last": False,
        "has_trailing_text": False,
    }
    if not text:
        return components

    try:
        matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    except re.error:
        return components

    if not matches:
        return components

    match = matches[-1]
    line_start = text.rfind("\n", 0, match.start()) + 1
    line_end = text.find("\n", match.end())
    if line_end == -1:
        line_end = len(text)

    line_text = text[line_start:line_end].strip()
    components["final_line"] = line_text
    components["reasoning"] = text[:line_start].rstrip()

    captured: Optional[str] = None
    if match.lastindex:
        for idx in range(match.lastindex, 0, -1):
            group = match.group(idx)
            if group is not None and str(group).strip():
                captured = str(group).strip()
                break
    if captured is None and line_text:
        if line_text.startswith("####"):
            captured = line_text[4:].strip()
        else:
            captured = line_text
    components["final_answer_raw"] = captured

    trailing_region = text[line_end:]
    components["has_trailing_text"] = bool(trailing_region and trailing_region.strip())

    non_empty_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    last_nonempty = non_empty_lines[-1] if non_empty_lines else ""
    components["final_line_is_last"] = bool(line_text and last_nonempty == line_text)
    return components


def _extract_math_answer(completion: str, cfg: Dict[str, Any]) -> Optional[str]:
    bench = cfg.get("benchmarks", {}).get("math", {})
    parser = str(bench.get("distill_answer_parser", "boxed")).lower()

    if parser == "boxed":
        return extract_boxed_answer(completion)

    if parser in {"final_line", "hash_line", "line"}:
        pattern = str(bench.get("final_line_pattern", _default_final_line_pattern("math")))
        return _extract_final_line_components(completion, pattern).get("final_answer_raw")

    if parser == "numeric":
        pattern = str(bench.get("answer_extraction_pattern", r"####\s*([\-\d,\.]+)"))
        return extract_numeric_answer(completion, pattern)

    logger.warning(
        "Unknown math distill parser; defaulting to boxed",
        extra={"parser": parser},
    )
    return extract_boxed_answer(completion)


def _annotate_teacher_structure(results: List[Dict[str, Any]], cfg: Dict[str, Any]) -> None:
    """Annotate records with final-line structure fields for traceable filtering."""
    for row in results:
        row.setdefault("teacher_reasoning", None)
        row.setdefault("teacher_final_line", None)
        row.setdefault("teacher_final_answer_raw", None)
        row.setdefault("teacher_final_answer_parsed", None)
        row.setdefault("teacher_final_line_parseable", None)
        row.setdefault("teacher_final_line_is_last", None)
        row.setdefault("teacher_has_trailing_text", None)

        if row.get("error") is not None:
            continue

        completion = str(row.get("teacher_completion") or "")
        if not completion:
            continue

        benchmark = str(row.get("benchmark") or "")
        pattern = _final_line_pattern(cfg, benchmark)
        structure = _extract_final_line_components(completion, pattern)

        row["teacher_reasoning"] = structure.get("reasoning")
        row["teacher_final_line"] = structure.get("final_line")
        row["teacher_final_answer_raw"] = structure.get("final_answer_raw")
        row["teacher_final_line_is_last"] = bool(structure.get("final_line_is_last"))
        row["teacher_has_trailing_text"] = bool(structure.get("has_trailing_text"))

        parser = _final_answer_parser(cfg, benchmark)
        parsed_answer: Optional[str] = None

        if parser == "numeric":
            answer_pattern = str(
                cfg.get("benchmarks", {}).get(benchmark, {}).get(
                    "answer_extraction_pattern",
                    _gsm8k_answer_pattern(cfg),
                )
            )
            final_line_text = str(structure.get("final_line") or "")
            parsed_answer = extract_numeric_answer(final_line_text, answer_pattern)
            if parsed_answer is None:
                parsed_answer = extract_numeric_answer(completion, answer_pattern)
        elif parser == "boxed":
            parsed_answer = extract_boxed_answer(completion)
        elif parser in {"final_line", "hash_line", "line"}:
            parsed_answer = structure.get("final_answer_raw")
        else:
            parsed_answer = structure.get("final_answer_raw")

        row["teacher_final_answer_parsed"] = parsed_answer
        row["teacher_final_line_parseable"] = bool(parsed_answer is not None and str(parsed_answer).strip())


def _annotate_teacher_quality(results: List[Dict[str, Any]], cfg: Dict[str, Any]) -> None:
    """Annotate each record with teacher_scorable/teacher_correct when possible."""
    gsm8k_pattern = _gsm8k_answer_pattern(cfg)

    for row in results:
        row.setdefault("teacher_predicted_answer", None)
        row.setdefault("teacher_scorable", None)
        row.setdefault("teacher_correct", None)

        # Keep precomputed consensus labels when present.
        if row.get("teacher_scorable") is not None and row.get("teacher_correct") is not None:
            continue

        if row.get("error") is not None:
            continue

        completion = str(row.get("teacher_completion") or "")
        benchmark = str(row.get("benchmark") or "")
        reference = row.get("reference_answer")
        if reference is None:
            continue

        if benchmark == "gsm8k":
            predicted = extract_numeric_answer(completion, gsm8k_pattern)
            scorable = predicted is not None
            correct = bool(predicted is not None and numeric_match(predicted, str(reference)))
            row["teacher_predicted_answer"] = predicted
            row["teacher_scorable"] = scorable
            row["teacher_correct"] = correct
            continue

        if benchmark == "math":
            predicted = _extract_math_answer(completion, cfg)
            scorable = predicted is not None
            correct = bool(predicted is not None and math_answer_match(predicted, str(reference)))
            row["teacher_predicted_answer"] = predicted
            row["teacher_scorable"] = scorable
            row["teacher_correct"] = correct


def _resolve_quality_policy(filter_cfg: Dict[str, Any], benchmark: str) -> str:
    """Return the policy for a benchmark: all | scorable | correct_only."""
    default_policy = str(filter_cfg.get("default_policy", "all"))
    per_benchmark = filter_cfg.get("benchmark_policies", {}) or {}
    policy = str(per_benchmark.get(benchmark, default_policy))
    if policy not in {"all", "scorable", "correct_only"}:
        logger.warning(
            "Unknown quality-filter policy; falling back to 'all'",
            extra={"benchmark": benchmark, "policy": policy},
        )
        return "all"
    return policy


def _apply_quality_filter(results: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Filter teacher outputs according to generation.quality_filter settings."""
    filter_cfg = cfg.get("generation", {}).get("quality_filter", {}) or {}
    filter_enabled = bool(filter_cfg.get("enabled", False))
    respect_teacher_acceptance = bool(filter_cfg.get("respect_teacher_acceptance", True))
    require_final_line = bool(filter_cfg.get("require_final_line", False))
    require_final_line_last = bool(filter_cfg.get("require_final_line_last", False))
    require_parseable_final_answer = bool(filter_cfg.get("require_parseable_final_answer", False))

    min_kept = max(1, int(filter_cfg.get("min_kept_per_benchmark", 1)))
    eligible_by_benchmark: Dict[str, List[Dict[str, Any]]] = {}
    filtered: List[Dict[str, Any]] = []

    for row in results:
        benchmark = str(row.get("benchmark") or "unknown")
        policy = _resolve_quality_policy(filter_cfg, benchmark) if filter_enabled else "all"

        reasons: List[str] = []
        completion = row.get("teacher_completion")

        if row.get("error") is not None:
            reasons.append("request_error")
        if not completion:
            reasons.append("empty_completion")

        if respect_teacher_acceptance and row.get("teacher_accepted_for_kd") is False:
            reasons.append("teacher_rejected_for_kd")

        structurally_eligible = not reasons
        if structurally_eligible:
            eligible_by_benchmark.setdefault(benchmark, []).append(row)

        if filter_enabled and structurally_eligible:
            if require_final_line and not row.get("teacher_final_line"):
                reasons.append("missing_final_line")
            if require_final_line_last and row.get("teacher_final_line_is_last") is not True:
                reasons.append("final_line_not_last")
            if require_parseable_final_answer and row.get("teacher_final_line_parseable") is not True:
                reasons.append("unparseable_final_answer")

            if policy == "scorable" and not bool(row.get("teacher_scorable")):
                reasons.append("policy_scorable_failed")
            elif policy == "correct_only" and not bool(row.get("teacher_correct")):
                reasons.append("policy_correct_only_failed")

        keep = len(reasons) == 0
        row["teacher_filter_policy"] = policy
        row["teacher_filter_reasons"] = reasons
        row["teacher_filter_decision"] = "kept" if keep else "dropped"

        if keep:
            filtered.append(row)

    kept_keys = {(r.get("id"), r.get("benchmark")) for r in filtered}

    if not filter_enabled:
        return filtered

    for benchmark, valid_rows in eligible_by_benchmark.items():
        kept_count = sum(1 for r in valid_rows if (r.get("id"), r.get("benchmark")) in kept_keys)
        if kept_count >= min_kept:
            continue

        logger.warning(
            "Quality filter kept too few rows for benchmark; backfilling best-effort.",
            extra={"benchmark": benchmark, "kept": kept_count, "min_kept": min_kept},
        )
        for row in valid_rows:
            key = (row.get("id"), row.get("benchmark"))
            if key in kept_keys:
                continue

            filtered.append(row)
            kept_keys.add(key)
            row["teacher_filter_decision"] = "kept_backfill"
            row_reasons = list(row.get("teacher_filter_reasons") or [])
            if "backfill_min_kept" not in row_reasons:
                row_reasons.append("backfill_min_kept")
            row["teacher_filter_reasons"] = row_reasons
            kept_count += 1
            if kept_count >= min_kept:
                break

    return filtered


def _write_gsm8k_consensus_audit(
    results: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    run_dir: Path,
) -> None:
    """Persist consensus diagnostics and previews for auditability."""
    gen = cfg.get("generation", {})
    scfg = gen.get("gsm8k_self_consistency", {}) or {}
    if not bool(scfg.get("enabled", False)):
        return

    gsm_rows = [r for r in results if r.get("benchmark") == "gsm8k"]
    if not gsm_rows:
        return

    size_hist: Dict[str, int] = {}
    accepted = 0
    reached = 0
    correct = 0
    for row in gsm_rows:
        size = int(row.get("teacher_consensus_size") or 0)
        size_hist[str(size)] = size_hist.get(str(size), 0) + 1
        if bool(row.get("teacher_consensus_reached")):
            reached += 1
        if bool(row.get("teacher_consensus_correct")):
            correct += 1
        if bool(row.get("teacher_accepted_for_kd")):
            accepted += 1

    audit = {
        "total_gsm8k_prompts": len(gsm_rows),
        "consensus_reached": reached,
        "consensus_correct": correct,
        "accepted_for_kd": accepted,
        "consensus_size_histogram": size_hist,
        "config": {
            "num_samples": int(scfg.get("num_samples", 5)),
            "min_consensus": int(scfg.get("min_consensus", 3)),
            "sample_temperature": float(scfg.get("sample_temperature", 0.6)),
            "sample_top_p": float(scfg.get("sample_top_p", 0.95)),
            "require_consensus_correct": bool(scfg.get("require_consensus_correct", True)),
            "target_selection": str(scfg.get("target_selection", "shortest_clean")),
        },
    }
    with (run_dir / "gsm8k_consensus_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)

    preview_n = max(1, int(scfg.get("preview_examples", 25)))
    accepted_rows = [r for r in gsm_rows if bool(r.get("teacher_accepted_for_kd"))][:preview_n]
    rejected_rows = [r for r in gsm_rows if not bool(r.get("teacher_accepted_for_kd"))][:preview_n]
    write_jsonl(accepted_rows, run_dir / "gsm8k_consensus_preview_accepted.jsonl")
    write_jsonl(rejected_rows, run_dir / "gsm8k_consensus_preview_rejected.jsonl")


# ── Public entry-point ──────────────────────────────────────

def run(config_path: str = "configs/distill.yaml",
        max_samples: Optional[int] = None) -> Path:
    cfg = load_yaml(config_path)
    gen = cfg.get("generation", {})
    seed = gen.get("seed", 42)
    set_seed(seed)
    setup_logging()

    experiment_tag = str(gen.get("experiment_tag", "")).strip()
    run_tag = "teacher-gen"
    if experiment_tag:
        run_tag = f"teacher-gen-{experiment_tag}"

    run_dir = make_run_dir("results/distill", tag=run_tag)
    snapshot_configs([config_path, "configs/eval.yaml"], run_dir)
    save_metadata(collect_metadata(seed, cfg), run_dir)

    # Use distill config's own benchmarks (TRAIN splits) — NOT eval.yaml's
    # test splits. This prevents data leakage between KD training data and
    # evaluation test sets.
    kd_benchmarks = cfg.get("benchmarks")
    if kd_benchmarks:
        # Use the distill config's benchmark definitions (train splits)
        prompt_cfg = {"benchmarks": kd_benchmarks, "generation": gen}
    else:
        # Fallback: load eval config (legacy path — NOT recommended)
        logger.warning(
            "No 'benchmarks' section in distill.yaml; falling back to eval.yaml. "
            "This may cause data leakage if eval.yaml uses test splits!"
        )
        prompt_cfg = load_yaml(gen.get("eval_config", "configs/eval.yaml"))

    prompts = collect_all_prompts(prompt_cfg)
    if max_samples is not None and max_samples > 0:
        prompts = prompts[:max_samples]
        logger.info("Smoke-test mode: truncated prompt list",
                     extra={"max_samples": max_samples, "num_prompts": len(prompts)})
    logger.info("Generating teacher outputs",
                 extra={"num_prompts": len(prompts)})

    results = asyncio.run(_generate_all(prompts, cfg))
    _annotate_teacher_structure(results, cfg)
    _annotate_teacher_quality(results, cfg)
    filtered_results = _apply_quality_filter(results, cfg)

    # Stats
    ok = [r for r in results if r["error"] is None]
    failed = len(results) - len(ok)
    logger.info("Generation complete",
                 extra={"total": len(results), "success": len(ok),
                        "failed": failed})
    if failed > 0:
        logger.warning(f"{failed} prompts failed — stored with error field")

    filter_cfg = cfg.get("generation", {}).get("quality_filter", {}) or {}
    if filter_cfg.get("enabled", False):
        logger.info(
            "Quality filter applied",
            extra={
                "kept": len(filtered_results),
                "dropped": len(results) - len(filtered_results),
                "default_policy": filter_cfg.get("default_policy", "all"),
                "benchmark_policies": filter_cfg.get("benchmark_policies", {}),
            },
        )
        output_rows = filtered_results
    else:
        output_rows = results

    # Save output rows and always persist full audit trace.
    out_path = Path(gen.get("output_file",
                            "results/distill/teacher_outputs.jsonl"))
    write_jsonl(output_rows, out_path)
    write_jsonl(output_rows, run_dir / "teacher_outputs.jsonl")
    write_jsonl(results, run_dir / "teacher_outputs_all.jsonl")
    _write_gsm8k_consensus_audit(results, cfg, run_dir)

    # Summary stats
    summary = {
        "total_prompts": len(results),
        "successful": len(ok),
        "failed": failed,
        "kept_after_filter": len(output_rows),
        "filter_decisions": {},
        "filter_reason_counts": {},
        "by_benchmark": {},
    }
    kept_keys = {(r.get("id"), r.get("benchmark")) for r in output_rows}
    for r in results:
        b = r["benchmark"]
        if b not in summary["by_benchmark"]:
            summary["by_benchmark"][b] = {
                "total": 0,
                "ok": 0,
                "teacher_scorable": 0,
                "teacher_correct": 0,
                "kept": 0,
                "dropped": 0,
                "consensus_reached": 0,
                "consensus_correct": 0,
                "accepted_for_kd": 0,
                "final_line_present": 0,
                "final_line_parseable": 0,
                "final_line_is_last": 0,
            }
        summary["by_benchmark"][b]["total"] += 1
        if r["error"] is None:
            summary["by_benchmark"][b]["ok"] += 1
        if bool(r.get("teacher_scorable")):
            summary["by_benchmark"][b]["teacher_scorable"] += 1
        if bool(r.get("teacher_correct")):
            summary["by_benchmark"][b]["teacher_correct"] += 1
        if bool(r.get("teacher_consensus_reached")):
            summary["by_benchmark"][b]["consensus_reached"] += 1
        if bool(r.get("teacher_consensus_correct")):
            summary["by_benchmark"][b]["consensus_correct"] += 1
        if bool(r.get("teacher_accepted_for_kd", True)):
            summary["by_benchmark"][b]["accepted_for_kd"] += 1
        if bool(r.get("teacher_final_line")):
            summary["by_benchmark"][b]["final_line_present"] += 1
        if bool(r.get("teacher_final_line_parseable")):
            summary["by_benchmark"][b]["final_line_parseable"] += 1
        if bool(r.get("teacher_final_line_is_last")):
            summary["by_benchmark"][b]["final_line_is_last"] += 1
        if (r.get("id"), r.get("benchmark")) in kept_keys:
            summary["by_benchmark"][b]["kept"] += 1
        else:
            summary["by_benchmark"][b]["dropped"] += 1

        decision = str(r.get("teacher_filter_decision") or "unknown")
        summary["filter_decisions"][decision] = summary["filter_decisions"].get(decision, 0) + 1
        for reason in r.get("teacher_filter_reasons") or []:
            key = str(reason)
            summary["filter_reason_counts"][key] = summary["filter_reason_counts"].get(key, 0) + 1

    with (run_dir / "generation_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return run_dir


# ── CLI ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate teacher outputs")
    parser.add_argument("--config", default="configs/distill.yaml")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Truncate total prompts (smoke test). E.g. --max-samples 15")
    args = parser.parse_args()
    run(args.config, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
