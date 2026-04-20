"""
distill/train_student_sft.py
=============================
Supervised fine-tuning (SFT) of a student model on hard-label teacher
outputs using HuggingFace Transformers + PEFT / LoRA.

Key design decisions
--------------------
* **Completion-only masking:** the loss is computed only on the *teacher
  completion* tokens.  Prompt tokens have ``labels=-100`` (ignored by
  ``CrossEntropyLoss``).  This prevents the student from wasting capacity
  memorising the prompt phrasing.
* **LoRA (not full fine-tuning):** keeps memory tractable on a single H100
  and avoids catastrophic forgetting of the pre-trained capabilities.
* **Same dataset for both students:** the 7B and 1.5B students are trained
  on the identical ``teacher_outputs.jsonl``, with only the student model
  name and output directory changing via ``--student`` override or config.

Workflow
--------
1. Load ``configs/distill.yaml``.
2. Read the teacher-generated dataset (JSONL produced by
   ``generate_teacher_outputs.py``).
3. Tokenise with the student tokeniser, applying completion-only masking.
4. Wrap the student model with LoRA adapters.
5. Train with ``Trainer``.
6. Save final adapter + tokeniser and log metrics.

Usage
-----
    python -m distill.train_student_sft \\
        --config configs/distill.yaml \\
        [--student Qwen/Qwen2.5-1.5B-Instruct]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, Optional

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


# ── Dataset preparation with completion-only masking ────────

def _prepare_dataset(
    dataset_path: str,
    tokeniser,
    max_seq_length: int,
    oversample_benchmarks: Optional[Dict[str, int]] = None,
):
    """Load JSONL teacher outputs, tokenise, and return a HF Dataset.

    Labels are set to -100 for prompt tokens so the loss is computed
    **only** on the teacher completion tokens.

    ``oversample_benchmarks``: optional map ``benchmark_name -> repeat_count``
    (e.g. ``math: 2`` duplicates each MATH row once) to up-weight harder tasks.
    """
    from datasets import Dataset as HFDataset

    IGNORE_INDEX = -100

    rows: list[dict] = []
    with open(dataset_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("error") is not None:
                continue  # skip failed teacher queries
            if obj.get("teacher_accepted_for_kd") is False:
                continue
            prompt = obj.get("prompt", "")
            completion = obj.get("teacher_completion", "")
            if not completion:
                continue
            rows.append({
                "prompt": prompt,
                "completion": completion,
                "benchmark": obj.get("benchmark", ""),
            })

    if not rows:
        raise ValueError(f"No valid training samples found in {dataset_path}")

    if oversample_benchmarks:
        expanded: list[dict] = []
        for row in rows:
            b = row.get("benchmark") or "unknown"
            repeat = max(1, int(oversample_benchmarks.get(b, 1)))
            for _ in range(repeat):
                expanded.append({
                    "prompt": row["prompt"],
                    "completion": row["completion"],
                })
        rows = expanded
        logger.info(
            "Oversampling applied",
            extra={
                "oversample_benchmarks": oversample_benchmarks,
                "rows_after": len(rows),
            },
        )

    logger.info("Loaded training samples",
                extra={"total_rows": len(rows), "path": dataset_path})

    ds = HFDataset.from_list(rows)

    def _tokenise(example: dict) -> dict:
        """Tokenise prompt + completion and mask prompt tokens in labels."""
        prompt_text = example["prompt"]
        completion_text = example["completion"]

        # Tokenise prompt alone to find its length
        prompt_enc = tokeniser(
            prompt_text,
            add_special_tokens=True,
            truncation=False,
        )
        prompt_len = len(prompt_enc["input_ids"])

        # Tokenise full sequence: prompt + completion
        full_text = prompt_text + completion_text
        full_enc = tokeniser(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        # Build labels: -100 for prompt tokens + padding, real ids for completion
        labels = [IGNORE_INDEX] * len(input_ids)
        for i in range(prompt_len, len(input_ids)):
            if attention_mask[i] == 1:  # not padding
                labels[i] = input_ids[i]

        full_enc["labels"] = labels
        return full_enc

    ds = ds.map(_tokenise, remove_columns=["prompt", "completion"])
    ds.set_format("torch")
    return ds


def _resolve_path_or_glob(path_or_pattern: str) -> str:
    """Resolve a concrete path from an explicit path or a glob pattern."""
    if any(ch in path_or_pattern for ch in "*?["):
        matches = sorted(glob.glob(path_or_pattern), reverse=True)
        if not matches:
            raise FileNotFoundError(
                f"No files matched init adapter pattern: {path_or_pattern}"
            )
        return matches[0]
    return path_or_pattern


def run(
    config_path: str = "configs/distill.yaml",
    student_override: Optional[str] = None,
    dataset_override: Optional[str] = None,
    stage2_residual: bool = False,
    max_samples: Optional[int] = None,
) -> Path:
    cfg = load_yaml(config_path)
    tcfg = dict(cfg.get("training", {}))
    residual_cfg = cfg.get("residual_distillation", {}) or {}
    stage2_cfg = residual_cfg.get("stage2_training", {}) or {}

    if stage2_residual and stage2_cfg and not bool(stage2_cfg.get("enabled", True)):
        raise ValueError(
            "--stage2-residual was requested but residual_distillation.stage2_training.enabled=false"
        )

    # Allow CLI override of the student model; stage2 may provide its own default.
    default_student = tcfg.get("student_model", "")
    if stage2_residual:
        default_student = stage2_cfg.get("student_model", default_student)

    student_model = student_override or default_student
    if not student_model:
        raise ValueError(
            "No student model specified. Set training.student_model in config or pass --student."
        )

    # ── Apply model-specific overrides (e.g. different lr for 7B vs 1.5B) ──
    model_overrides = cfg.get("model_overrides", {}).get(student_model, {})
    if model_overrides:
        logger.info(
            "Applying model-specific overrides",
            extra={"model": student_model, "overrides": str(model_overrides)},
        )
        for k, v in model_overrides.items():
            if isinstance(v, dict) and isinstance(tcfg.get(k), dict):
                merged = dict(tcfg.get(k, {}))
                merged.update(v)
                tcfg[k] = merged
            else:
                tcfg[k] = v

    # Stage-2 residual training keeps the core pipeline but uses explicit,
    # auditable overrides from the residual_distillation section.
    if stage2_residual and stage2_cfg:
        for k, v in stage2_cfg.items():
            if k in {"enabled", "dataset_path", "student_model", "init_adapter_path"}:
                continue
            if isinstance(v, dict) and isinstance(tcfg.get(k), dict):
                merged = dict(tcfg.get(k, {}))
                merged.update(v)
                tcfg[k] = merged
            else:
                tcfg[k] = v

    seed = int(tcfg.get("seed", 42))
    set_seed(seed)
    setup_logging()

    # Derive a short tag for directory naming
    model_short = student_model.split("/")[-1].lower().replace("-instruct", "")
    tag = f"sft-{model_short}"
    if stage2_residual:
        tag = f"{tag}-residual-stage2"

    experiment_tag = str(tcfg.get("experiment_tag", "")).strip()
    if stage2_residual:
        stage2_experiment_tag = str(stage2_cfg.get("experiment_tag", "")).strip()
        if stage2_experiment_tag:
            experiment_tag = stage2_experiment_tag
    if experiment_tag:
        tag = f"{tag}-{experiment_tag}"

    # ── Run directory ───────────────────────────────────────
    run_dir = make_run_dir(
        tcfg.get("output_dir", "results/distill"),
        tag=tag,
    )
    snapshot_configs([config_path], run_dir)
    save_metadata(collect_metadata(seed, cfg), run_dir)

    # ── Lazy imports (heavy) ────────────────────────────────
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        default_data_collator,
    )
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    # ── Model + tokeniser ───────────────────────────────────
    logger.info("Loading student model", extra={"model": student_model})

    tokeniser = AutoTokenizer.from_pretrained(student_model)
    if tokeniser.pad_token is None:
        tokeniser.pad_token = tokeniser.eos_token

    # Detect actual GPU/bf16 support (login nodes have no GPU)
    has_gpu = torch.cuda.is_available()
    use_bf16 = tcfg.get("bf16", True) and has_gpu and torch.cuda.is_bf16_supported()
    use_fp16 = tcfg.get("fp16", False) and has_gpu and not use_bf16
    if not has_gpu:
        logger.warning("No GPU detected — running in fp32 (smoke-test / CPU mode)")
    load_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        student_model,
        dtype=load_dtype,
        device_map="auto",
    )

    init_adapter_path = None
    if stage2_residual:
        init_adapter_path = stage2_cfg.get("init_adapter_path")
    if init_adapter_path:
        resolved_adapter = _resolve_path_or_glob(str(init_adapter_path))
        logger.info(
            "Loading residual stage initializer adapter",
            extra={"init_adapter_path": resolved_adapter},
        )
        model = PeftModel.from_pretrained(model, resolved_adapter)
        model = model.merge_and_unload()
        logger.info("Merged stage initializer adapter into base model")

    # ── LoRA ────────────────────────────────────────────────
    lora_cfg = tcfg.get("lora", {})
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("lora_alpha", 16),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"]),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    # Required for gradient checkpointing to work with PEFT/LoRA
    model.enable_input_require_grads()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ── Dataset ─────────────────────────────────────────────
    dataset_path = dataset_override or tcfg.get(
        "dataset_path", "results/distill/teacher_outputs.jsonl",
    )
    if stage2_residual and not dataset_override:
        dataset_path = stage2_cfg.get("dataset_path", dataset_path)
    max_seq_length = tcfg.get("max_seq_length", 2048)
    oversample = tcfg.get("oversample_benchmarks")
    ds = _prepare_dataset(
        dataset_path,
        tokeniser,
        max_seq_length,
        oversample_benchmarks=oversample,
    )
    if max_samples is not None and max_samples > 0 and max_samples < len(ds):
        ds = ds.select(range(max_samples))
        logger.info("Smoke-test mode: dataset truncated",
                     extra={"max_samples": max_samples})

    # ── Train / validation split ────────────────────────────
    val_ratio = tcfg.get("val_ratio", 0.15)
    if val_ratio > 0:
        split = ds.train_test_split(test_size=val_ratio, seed=seed)
        train_ds = split["train"]
        val_ds = split["test"]
        logger.info("Dataset split into train/val",
                     extra={"train": len(train_ds), "val": len(val_ds),
                            "val_ratio": val_ratio})
    else:
        train_ds = ds
        val_ds = None
        logger.info("Validation disabled (val_ratio=0)")

    logger.info("Dataset ready", extra={"samples": len(train_ds)})

    # ── Training arguments ──────────────────────────────────
    log_cfg = tcfg.get("logging", {})

    # Eval strategy: use eval_strategy from config, default to "steps" if val set exists
    eval_strategy_default = "steps" if val_ds is not None else "no"
    eval_strategy = log_cfg.get("eval_strategy", eval_strategy_default)
    eval_steps = log_cfg.get("eval_steps", 50)
    load_best = log_cfg.get("load_best_model_at_end", val_ds is not None)

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=tcfg.get("num_train_epochs", 3),
        per_device_train_batch_size=tcfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=tcfg.get("gradient_accumulation_steps", 4),
        learning_rate=tcfg.get("learning_rate", 2e-4),
        lr_scheduler_type=tcfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=tcfg.get("warmup_ratio", 0.03),
        weight_decay=tcfg.get("weight_decay", 0.0),
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        max_grad_norm=tcfg.get("max_grad_norm", 1.0),
        logging_steps=log_cfg.get("logging_steps", 10),
        save_strategy=log_cfg.get("save_strategy", "steps"),
        save_steps=log_cfg.get("save_steps", 200),
        eval_strategy=eval_strategy,
        eval_steps=eval_steps if eval_strategy != "no" else None,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss" if load_best else None,
        greater_is_better=False if load_best else None,
        report_to=log_cfg.get("report_to", "none"),
        seed=seed,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # ── Callbacks ────────────────────────────────────────────
    callbacks = []
    if load_best and val_ds is not None:
        from transformers import EarlyStoppingCallback
        early_stop_patience = tcfg.get("early_stopping_patience", 3)
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=early_stop_patience,
        ))
        logger.info("Early stopping enabled",
                     extra={"patience": early_stop_patience})

    # ── Trainer ─────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )

    logger.info("Starting SFT training",
                extra={"student": student_model, "samples": len(train_ds),
                       "val_samples": len(val_ds) if val_ds else 0,
                       "epochs": tcfg.get("num_train_epochs", 3)})
    trainer.train()

    # Save training log (loss curves, eval metrics per step)
    if trainer.state.log_history:
        with (run_dir / "training_log.json").open("w") as fh:
            json.dump(trainer.state.log_history, fh, indent=2)
        logger.info("Training log saved",
                     extra={"entries": len(trainer.state.log_history)})

    # ── Save final adapter + tokeniser ──────────────────────
    final_dir = run_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokeniser.save_pretrained(str(final_dir))
    logger.info("Training complete",
                extra={"adapter_dir": str(final_dir),
                       "student": student_model})

    # Save a small manifest for downstream pipeline steps
    manifest = {
        "student_model": student_model,
        "adapter_dir": str(final_dir),
        "dataset_path": dataset_path,
        "stage2_residual": stage2_residual,
        "init_adapter_path": init_adapter_path,
        "num_train_samples": len(train_ds),
        "num_val_samples": len(val_ds) if val_ds else 0,
        "val_ratio": val_ratio,
        "epochs": tcfg.get("num_train_epochs", 3),
        "lora_r": lora_cfg.get("r", 64),
    }
    with (run_dir / "training_manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)

    return run_dir


# ── CLI ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Student SFT training")
    parser.add_argument("--config", default="configs/distill.yaml")
    parser.add_argument("--student", default=None,
                        help="Override training.student_model (e.g. Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--dataset-path", default=None,
                        help="Override training.dataset_path at runtime")
    parser.add_argument("--stage2-residual", action="store_true",
                        help="Apply residual_distillation.stage2_training overrides")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Truncate dataset (smoke test). E.g. --max-samples 20")
    args = parser.parse_args()
    run(
        args.config,
        student_override=args.student,
        dataset_override=args.dataset_path,
        stage2_residual=args.stage2_residual,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
