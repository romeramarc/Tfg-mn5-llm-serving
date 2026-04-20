"""
distill/refine_student_grpo.py
==============================
Minimal GRPO-like refinement for student models.

This stage is intentionally lightweight and practical for the current codebase:
1. Sample multiple candidate completions per training prompt from a running
   policy model endpoint.
2. Score each sample with a composite reward:
   - correctness reward (GSM8K/MATH answer match)
   - format reward (strict final-line contract: #### <answer>)
3. Compute group-normalised advantages per prompt and convert them into
   non-negative sample weights.
4. Run weighted completion-only SFT with LoRA.

This is not a full PPO/GRPO optimizer, but a robust approximation that keeps
existing infra (Transformers Trainer + PEFT) and is easy to launch on SLURM.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import math
import os
import re
import signal
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from distill.dataset_utils import write_jsonl
from distill.generate_teacher_outputs import collect_all_prompts
from eval.scoring import (
    extract_boxed_answer,
    extract_numeric_answer,
    math_answer_match,
    numeric_match,
)
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


def _maybe_stop_external_policy_server() -> None:
    """Stop policy server if a parent launcher exposed its PID via env var.

    This allows running sampling against vLLM and then freeing GPU memory
    before merge+train in the same job.
    """
    raw_pid = os.environ.get("GRPO_POLICY_SERVER_PID", "").strip()
    if not raw_pid:
        return

    try:
        pid = int(raw_pid)
    except ValueError:
        logger.warning("Ignoring invalid GRPO_POLICY_SERVER_PID", extra={"value": raw_pid})
        return

    if pid <= 0:
        return

    try:
        os.kill(pid, signal.SIGTERM)
        logger.info("Requested external policy server shutdown", extra={"pid": pid})
    except ProcessLookupError:
        logger.info("External policy server already stopped", extra={"pid": pid})
        return
    except PermissionError:
        logger.warning("No permission to stop external policy server", extra={"pid": pid})
        return

    # Give the server a short grace period to release GPU allocations.
    for _ in range(40):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.25)

    try:
        os.kill(pid, signal.SIGKILL)
        logger.warning("Force-killed external policy server after timeout", extra={"pid": pid})
    except ProcessLookupError:
        pass
    except PermissionError:
        logger.warning("Could not force-kill external policy server", extra={"pid": pid})


def _resolve_path_or_glob(path_or_pattern: str) -> str:
    if any(ch in path_or_pattern for ch in "*?["):
        matches = sorted(glob.glob(path_or_pattern), reverse=True)
        if not matches:
            raise FileNotFoundError(f"No files matched pattern: {path_or_pattern}")
        return matches[0]
    return path_or_pattern


def _default_final_line_pattern(benchmark: str) -> str:
    if benchmark == "gsm8k":
        return r"(?m)^####\s*([\-\d,\.]+)\s*$"
    return r"(?m)^####\s*(.+?)\s*$"


def _extract_final_line_components(text: str, pattern: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "final_line": None,
        "final_answer_raw": None,
        "final_line_is_last": False,
    }
    if not text:
        return out

    try:
        matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    except re.error:
        return out

    if not matches:
        return out

    match = matches[-1]
    line_start = text.rfind("\n", 0, match.start()) + 1
    line_end = text.find("\n", match.end())
    if line_end == -1:
        line_end = len(text)

    line = text[line_start:line_end].strip()
    out["final_line"] = line

    captured: Optional[str] = None
    if match.lastindex:
        for idx in range(match.lastindex, 0, -1):
            grp = match.group(idx)
            if grp is not None and str(grp).strip():
                captured = str(grp).strip()
                break
    if captured is None and line:
        if line.startswith("####"):
            captured = line[4:].strip()
        else:
            captured = line
    out["final_answer_raw"] = captured

    non_empty_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out["final_line_is_last"] = bool(non_empty_lines and non_empty_lines[-1] == line)
    return out


def _extract_predicted_answer(
    benchmark: str,
    completion: str,
    bench_cfg: Dict[str, Any],
    final_components: Dict[str, Any],
) -> Optional[str]:
    if benchmark == "gsm8k":
        pattern = str(bench_cfg.get("answer_extraction_pattern", r"####\s*([\-\d,\.]+)"))
        final_line = str(final_components.get("final_line") or "")
        parsed = extract_numeric_answer(final_line, pattern)
        if parsed is not None:
            return parsed
        return extract_numeric_answer(completion, pattern)

    if benchmark == "math":
        parser = str(bench_cfg.get("reward_answer_parser", bench_cfg.get("distill_answer_parser", "final_line"))).lower()
        if parser == "boxed":
            return extract_boxed_answer(completion)
        if parser in {"final_line", "hash_line", "line"}:
            raw = final_components.get("final_answer_raw")
            if raw is None:
                return None
            return str(raw).strip()
        if parser == "numeric":
            pattern = str(bench_cfg.get("answer_extraction_pattern", r"####\s*([\-\d,\.]+)"))
            return extract_numeric_answer(completion, pattern)
        return str(final_components.get("final_answer_raw") or "").strip() or None

    return None


def _is_correct(
    benchmark: str,
    predicted: Optional[str],
    reference: Optional[str],
) -> bool:
    if predicted is None or reference is None:
        return False

    if benchmark == "gsm8k":
        return numeric_match(str(predicted), str(reference))
    if benchmark == "math":
        return math_answer_match(str(predicted), str(reference))
    return False


async def _query_policy(
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
        finish_reason = None
        if body.get("choices"):
            text = body["choices"][0].get("text", "")
            finish_reason = body["choices"][0].get("finish_reason")

        return {
            "completion": text,
            "latency_ms": latency_ms,
            "finish_reason": finish_reason,
            "error": None,
        }
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "completion": "",
            "latency_ms": latency_ms,
            "finish_reason": None,
            "error": str(exc),
        }


def _build_prompt_cfg(cfg: Dict[str, Any], grpo_cfg: Dict[str, Any]) -> Dict[str, Any]:
    benchmarks = grpo_cfg.get("benchmarks") or cfg.get("benchmarks", {})
    extra_prompts_file = grpo_cfg.get("extra_prompts_file")
    return {
        "benchmarks": benchmarks,
        "generation": {
            "extra_prompts_file": extra_prompts_file,
        },
    }


def _collect_refinement_prompts(
    cfg: Dict[str, Any],
    grpo_cfg: Dict[str, Any],
    max_prompts_override: Optional[int],
) -> List[Dict[str, Any]]:
    prompt_cfg = _build_prompt_cfg(cfg, grpo_cfg)
    prompts = collect_all_prompts(prompt_cfg)

    # Keep only benchmarks with reference answers needed for correctness reward.
    prompts = [
        p for p in prompts
        if p.get("benchmark") in {"gsm8k", "math"} and p.get("reference_answer") is not None
    ]

    sampling_cfg = grpo_cfg.get("sampling", {}) or {}
    max_prompts_cfg = sampling_cfg.get("max_prompts")
    max_prompts = max_prompts_override
    if max_prompts is None and max_prompts_cfg not in (None, "null"):
        max_prompts = int(max_prompts_cfg)

    if max_prompts is not None and max_prompts > 0:
        prompts = prompts[:max_prompts]

    if not prompts:
        raise ValueError("No prompts available for GRPO refinement")
    return prompts


async def _sample_and_score(
    prompts: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    grpo_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    sampling_cfg = grpo_cfg.get("sampling", {}) or {}
    rewards_cfg = grpo_cfg.get("rewards", {}) or {}
    bench_cfg_map = grpo_cfg.get("benchmarks") or cfg.get("benchmarks", {})

    base_url = str(grpo_cfg.get("base_url", "http://localhost:8002")).rstrip("/")
    url = f"{base_url}/v1/completions"
    model = str(grpo_cfg.get("policy_model_name", grpo_cfg.get("policy_model_path", "")))
    if not model:
        raise ValueError("grpo_refinement.policy_model_name or policy_model_path is required")

    num_samples = max(1, int(sampling_cfg.get("num_samples_per_prompt", 4)))
    max_tokens = int(sampling_cfg.get("max_tokens", 768))
    temperature = float(sampling_cfg.get("temperature", 0.7))
    top_p = float(sampling_cfg.get("top_p", 0.95))
    timeout = float(sampling_cfg.get("timeout_seconds", 180))
    batch_size = int(sampling_cfg.get("batch_size", 16))

    correctness_weight = float(rewards_cfg.get("correctness_weight", 1.0))
    format_weight = float(rewards_cfg.get("format_weight", 0.4))
    require_final_line_last = bool(rewards_cfg.get("require_final_line_last", True))

    sem = asyncio.Semaphore(batch_size)
    samples: List[Dict[str, Any]] = []

    async def _one(prompt_row: Dict[str, Any], sample_index: int) -> None:
        benchmark = str(prompt_row.get("benchmark") or "")
        bench_cfg = bench_cfg_map.get(benchmark, {}) if isinstance(bench_cfg_map, dict) else {}
        final_line_pattern = str(bench_cfg.get("final_line_pattern", _default_final_line_pattern(benchmark)))

        async with sem:
            response = await _query_policy(
                client,
                url,
                model,
                str(prompt_row.get("prompt") or ""),
                max_tokens,
                temperature,
                top_p,
                timeout,
            )

        completion = str(response.get("completion") or "")
        final_components = _extract_final_line_components(completion, final_line_pattern)
        predicted = _extract_predicted_answer(benchmark, completion, bench_cfg, final_components)
        reference = prompt_row.get("reference_answer")

        correct = False
        format_ok = False
        if response.get("error") is None and completion:
            correct = _is_correct(benchmark, predicted, reference)
            parseable = bool(predicted is not None and str(predicted).strip())
            line_present = bool(final_components.get("final_line"))
            line_last = bool(final_components.get("final_line_is_last"))
            format_ok = line_present and parseable and (line_last if require_final_line_last else True)

        reward_correctness = correctness_weight * (1.0 if correct else 0.0)
        reward_format = format_weight * (1.0 if format_ok else 0.0)
        reward_total = reward_correctness + reward_format

        samples.append({
            "id": prompt_row.get("id"),
            "benchmark": benchmark,
            "prompt": prompt_row.get("prompt"),
            "reference_answer": reference,
            "sample_index": sample_index,
            "completion": completion,
            "predicted_answer": predicted,
            "final_line": final_components.get("final_line"),
            "final_line_is_last": bool(final_components.get("final_line_is_last")),
            "correct": bool(correct),
            "format_ok": bool(format_ok),
            "reward_correctness": reward_correctness,
            "reward_format": reward_format,
            "reward_total": reward_total,
            "latency_ms": response.get("latency_ms"),
            "finish_reason": response.get("finish_reason"),
            "error": response.get("error"),
        })

    async with httpx.AsyncClient() as client:
        tasks = []
        for prompt_row in prompts:
            for sample_index in range(num_samples):
                tasks.append(asyncio.create_task(_one(prompt_row, sample_index)))
        await asyncio.gather(*tasks)

    return samples


def _compute_advantages_and_select(
    samples: List[Dict[str, Any]],
    grpo_cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    adv_cfg = grpo_cfg.get("group_advantage", {}) or {}
    eps = float(adv_cfg.get("epsilon", 1e-6))
    clip_min = float(adv_cfg.get("clip_min", 0.0))
    min_weight_to_keep = float(adv_cfg.get("min_weight_to_keep", 0.01))
    keep_top_if_all_clipped = bool(adv_cfg.get("keep_top_if_all_clipped", True))

    by_prompt: Dict[str, List[Dict[str, Any]]] = {}
    for row in samples:
        key = f"{row.get('benchmark')}::{row.get('id')}"
        by_prompt.setdefault(key, []).append(row)

    selected: List[Dict[str, Any]] = []

    for _, group in by_prompt.items():
        rewards = [float(x.get("reward_total") or 0.0) for x in group]
        mean_reward = statistics.fmean(rewards) if rewards else 0.0
        std_reward = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        denom = max(std_reward, eps)

        eligible: List[Dict[str, Any]] = []
        for row in group:
            reward = float(row.get("reward_total") or 0.0)
            advantage = (reward - mean_reward) / denom
            weight = max(clip_min, advantage)

            row["group_reward_mean"] = mean_reward
            row["group_reward_std"] = std_reward
            row["advantage"] = advantage
            row["sample_weight"] = weight

            if row.get("error") is None and row.get("completion"):
                eligible.append(row)

        group_selected = [
            row for row in eligible
            if float(row.get("sample_weight") or 0.0) >= min_weight_to_keep
        ]
        if group_selected:
            for row in group_selected:
                row["selection_reason"] = "weight_threshold"
            selected.extend(group_selected)
            continue

        if keep_top_if_all_clipped and eligible:
            top = max(eligible, key=lambda r: (float(r.get("reward_total") or 0.0), float(r.get("advantage") or 0.0)))
            top_copy = dict(top)
            top_copy["sample_weight"] = max(1.0, float(top_copy.get("sample_weight") or 0.0))
            top_copy["selection_reason"] = "fallback_top_reward"
            selected.append(top_copy)

    if not selected:
        raise ValueError("No GRPO samples selected for weighted SFT")

    summary = {
        "total_samples": len(samples),
        "selected_samples": len(selected),
        "selection_rate": len(selected) / len(samples) if samples else 0.0,
        "mean_reward_all": statistics.fmean([float(s.get("reward_total") or 0.0) for s in samples]) if samples else 0.0,
        "mean_reward_selected": statistics.fmean([float(s.get("reward_total") or 0.0) for s in selected]) if selected else 0.0,
    }

    by_benchmark: Dict[str, Dict[str, Any]] = {}
    for row in samples:
        bench = str(row.get("benchmark") or "unknown")
        if bench not in by_benchmark:
            by_benchmark[bench] = {
                "total": 0,
                "correct": 0,
                "format_ok": 0,
                "mean_reward": 0.0,
            }
        by_benchmark[bench]["total"] += 1
        if bool(row.get("correct")):
            by_benchmark[bench]["correct"] += 1
        if bool(row.get("format_ok")):
            by_benchmark[bench]["format_ok"] += 1

    for bench, bench_stats in by_benchmark.items():
        rewards = [float(s.get("reward_total") or 0.0) for s in samples if str(s.get("benchmark") or "") == bench]
        bench_stats["mean_reward"] = statistics.fmean(rewards) if rewards else 0.0

    summary["by_benchmark"] = by_benchmark
    return selected, summary


def _prepare_weighted_dataset(
    rows: List[Dict[str, Any]],
    tokeniser,
    max_seq_length: int,
    oversample_benchmarks: Optional[Dict[str, int]] = None,
):
    from datasets import Dataset as HFDataset

    ignore_index = -100

    train_rows: List[Dict[str, Any]] = []
    for row in rows:
        prompt = str(row.get("prompt") or "")
        completion = str(row.get("completion") or "")
        if not prompt or not completion:
            continue
        sample_weight = float(row.get("sample_weight") or 0.0)
        if sample_weight <= 0:
            continue
        train_rows.append({
            "prompt": prompt,
            "completion": completion,
            "benchmark": row.get("benchmark", ""),
            "sample_weight": sample_weight,
        })

    if not train_rows:
        raise ValueError("No valid weighted rows after filtering")

    if oversample_benchmarks:
        expanded: List[Dict[str, Any]] = []
        for row in train_rows:
            bench = str(row.get("benchmark") or "unknown")
            repeat = max(1, int(oversample_benchmarks.get(bench, 1)))
            for _ in range(repeat):
                expanded.append(dict(row))
        train_rows = expanded

    ds = HFDataset.from_list(train_rows)

    def _tokenise(example: Dict[str, Any]) -> Dict[str, Any]:
        prompt_text = str(example.get("prompt") or "")
        completion_text = str(example.get("completion") or "")

        prompt_enc = tokeniser(
            prompt_text,
            add_special_tokens=True,
            truncation=False,
        )
        prompt_len = len(prompt_enc["input_ids"])

        full_text = prompt_text + completion_text
        full_enc = tokeniser(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        labels = [ignore_index] * len(input_ids)
        for idx in range(prompt_len, len(input_ids)):
            if attention_mask[idx] == 1:
                labels[idx] = input_ids[idx]

        full_enc["labels"] = labels
        full_enc["sample_weight"] = float(example.get("sample_weight") or 0.0)
        return full_enc

    ds = ds.map(_tokenise, remove_columns=["prompt", "completion", "benchmark"])
    ds.set_format("torch")
    return ds


def _build_weighted_trainer():
    import torch
    from transformers import Trainer

    class WeightedCompletionTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            sample_weight = inputs.pop("sample_weight", None)

            outputs = model(**inputs)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            token_losses = token_losses.view(shift_labels.size())

            mask = (shift_labels != -100).float()
            seq_token_counts = mask.sum(dim=1).clamp_min(1.0)
            seq_losses = (token_losses * mask).sum(dim=1) / seq_token_counts

            if sample_weight is None:
                loss = seq_losses.mean()
            else:
                weights = sample_weight.float().to(seq_losses.device)
                weights = torch.clamp(weights, min=0.0)
                weight_sum = weights.sum()
                if float(weight_sum.item()) <= 0.0:
                    loss = seq_losses.mean()
                else:
                    loss = (seq_losses * weights).sum() / weight_sum

            if return_outputs:
                return loss, outputs
            return loss

    return WeightedCompletionTrainer


def run(
    config_path: str = "configs/distill_1p5b_structured_grpo.yaml",
    max_prompts: Optional[int] = None,
    max_train_samples: Optional[int] = None,
) -> Path:
    cfg = load_yaml(config_path)
    grpo_cfg = cfg.get("grpo_refinement", {}) or {}
    if not bool(grpo_cfg.get("enabled", False)):
        raise ValueError("grpo_refinement.enabled=false; enable it before running")

    seed = int(grpo_cfg.get("seed", cfg.get("training", {}).get("seed", 42)))
    set_seed(seed)
    setup_logging()

    student_model = str(grpo_cfg.get("student_model", cfg.get("training", {}).get("student_model", "")))
    if not student_model:
        raise ValueError("No student model configured for GRPO refinement")

    model_short = student_model.split("/")[-1].lower().replace("-instruct", "")
    experiment_tag = str(grpo_cfg.get("experiment_tag", "grpo")).strip() or "grpo"
    run_tag = f"sft-{model_short}-{experiment_tag}"

    run_dir = make_run_dir(grpo_cfg.get("output_dir", "results/distill"), tag=run_tag)
    snapshot_configs([config_path], run_dir)
    save_metadata(collect_metadata(seed, cfg), run_dir)

    prompts = _collect_refinement_prompts(cfg, grpo_cfg, max_prompts)
    logger.info("Collected GRPO prompts", extra={"count": len(prompts)})

    samples = asyncio.run(_sample_and_score(prompts, cfg, grpo_cfg))
    selected, reward_summary = _compute_advantages_and_select(samples, grpo_cfg)

    write_jsonl(samples, run_dir / "grpo_samples_all.jsonl")
    write_jsonl(selected, run_dir / "grpo_samples_selected.jsonl")

    # Sampling finished: optionally stop externally-managed policy vLLM to
    # free GPU memory before adapter merge and weighted SFT.
    _maybe_stop_external_policy_server()

    if max_train_samples is not None and max_train_samples > 0 and max_train_samples < len(selected):
        selected = selected[:max_train_samples]
        logger.info("Truncated selected samples", extra={"max_train_samples": max_train_samples})

    # Lazy heavy imports
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        default_data_collator,
    )

    tokeniser = AutoTokenizer.from_pretrained(student_model)
    if tokeniser.pad_token is None:
        tokeniser.pad_token = tokeniser.eos_token

    has_gpu = torch.cuda.is_available()
    tr_cfg = grpo_cfg.get("training", {}) or {}
    use_bf16 = bool(tr_cfg.get("bf16", True)) and has_gpu and torch.cuda.is_bf16_supported()
    use_fp16 = bool(tr_cfg.get("fp16", False)) and has_gpu and not use_bf16
    if not has_gpu:
        logger.warning("No GPU detected; falling back to fp32")

    load_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        student_model,
        dtype=load_dtype,
        device_map="auto",
    )

    init_adapter_path = grpo_cfg.get("init_adapter_path")
    resolved_init_adapter: Optional[str] = None
    if init_adapter_path:
        resolved_init_adapter = _resolve_path_or_glob(str(init_adapter_path))
        logger.info("Merging init adapter before GRPO weighted SFT", extra={"adapter": resolved_init_adapter})
        model = PeftModel.from_pretrained(model, resolved_init_adapter)
        model = model.merge_and_unload()

    lora_cfg = tr_cfg.get("lora", {}) or {}
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 64)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 16)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        bias=str(lora_cfg.get("bias", "none")),
        task_type=TaskType.CAUSAL_LM,
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    max_seq_length = int(tr_cfg.get("max_seq_length", 2048))
    ds = _prepare_weighted_dataset(
        selected,
        tokeniser,
        max_seq_length,
        oversample_benchmarks=tr_cfg.get("oversample_benchmarks"),
    )

    val_ratio = float(tr_cfg.get("val_ratio", 0.1))
    if val_ratio > 0:
        split = ds.train_test_split(test_size=val_ratio, seed=seed)
        train_ds = split["train"]
        val_ds = split["test"]
    else:
        train_ds = ds
        val_ds = None

    logging_cfg = tr_cfg.get("logging", {}) or {}
    eval_strategy_default = "steps" if val_ds is not None else "no"
    eval_strategy = str(logging_cfg.get("eval_strategy", eval_strategy_default))

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=float(tr_cfg.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(tr_cfg.get("per_device_train_batch_size", 4)),
        gradient_accumulation_steps=int(tr_cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(tr_cfg.get("learning_rate", 5e-5)),
        lr_scheduler_type=str(tr_cfg.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(tr_cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(tr_cfg.get("weight_decay", 0.0)),
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=bool(tr_cfg.get("gradient_checkpointing", True)),
        max_grad_norm=float(tr_cfg.get("max_grad_norm", 1.0)),
        logging_steps=int(logging_cfg.get("logging_steps", 10)),
        save_strategy=str(logging_cfg.get("save_strategy", "steps")),
        save_steps=int(logging_cfg.get("save_steps", 200)),
        eval_strategy=eval_strategy,
        eval_steps=int(logging_cfg.get("eval_steps", 100)) if eval_strategy != "no" else None,
        load_best_model_at_end=bool(logging_cfg.get("load_best_model_at_end", val_ds is not None)),
        metric_for_best_model="eval_loss" if val_ds is not None else None,
        greater_is_better=False if val_ds is not None else None,
        report_to=str(logging_cfg.get("report_to", "none")),
        seed=seed,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    WeightedCompletionTrainer = _build_weighted_trainer()
    trainer = WeightedCompletionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
    )

    logger.info(
        "Starting weighted GRPO-like refinement",
        extra={
            "train_samples": len(train_ds),
            "val_samples": len(val_ds) if val_ds is not None else 0,
            "student_model": student_model,
        },
    )
    trainer.train()

    if trainer.state.log_history:
        with (run_dir / "training_log.json").open("w", encoding="utf-8") as fh:
            json.dump(trainer.state.log_history, fh, indent=2)

    final_dir = run_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokeniser.save_pretrained(str(final_dir))

    manifest = {
        "student_model": student_model,
        "adapter_dir": str(final_dir),
        "init_adapter_path": resolved_init_adapter,
        "num_prompts": len(prompts),
        "num_samples": len(samples),
        "num_selected_samples": len(selected),
        "num_train_samples": len(train_ds),
        "num_val_samples": len(val_ds) if val_ds is not None else 0,
        "reward_summary": reward_summary,
        "experiment_tag": experiment_tag,
    }
    with (run_dir / "grpo_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    with (run_dir / "grpo_reward_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(reward_summary, fh, indent=2)

    logger.info("GRPO-like refinement complete", extra={"adapter_dir": str(final_dir)})
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal GRPO-like student refinement")
    parser.add_argument("--config", default="configs/distill_1p5b_structured_grpo.yaml")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Limit prompt count before sampling")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Limit selected weighted samples before training")
    args = parser.parse_args()

    run(
        config_path=args.config,
        max_prompts=args.max_prompts,
        max_train_samples=args.max_train_samples,
    )


if __name__ == "__main__":
    main()
