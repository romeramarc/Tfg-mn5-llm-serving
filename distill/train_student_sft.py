"""
distill/train_student_sft.py
=============================
Supervised fine-tuning (SFT) of a student model on hard-label teacher
outputs using HuggingFace Transformers + PEFT / LoRA.

Workflow
--------
1. Load ``configs/distill.yaml``.
2. Read the teacher-generated dataset (JSONL produced by
   ``generate_teacher_outputs.py``).
3. Tokenise with the student tokeniser.
4. Wrap the student model with LoRA adapters.
5. Train with ``Trainer``.
6. Save final adapter + tokeniser and log metrics.

Usage
-----
    python -m distill.train_student_sft [--config configs/distill.yaml]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

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


def _prepare_dataset(dataset_path: str, tokeniser, max_seq_length: int):
    """Load JSONL teacher outputs, tokenise, and return a HF Dataset."""
    from datasets import Dataset as HFDataset

    rows: list[dict] = []
    with open(dataset_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("error") is not None:
                continue  # skip failed teacher queries
            prompt = obj.get("prompt", "")
            response = obj.get("response", "")
            # Concatenate prompt + response as a single training example
            rows.append({"text": f"{prompt}\n{response}"})

    if not rows:
        raise ValueError(f"No valid training samples found in {dataset_path}")

    ds = HFDataset.from_list(rows)

    def _tokenise(example: dict) -> dict:
        enc = tokeniser(
            example["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    ds = ds.map(_tokenise, remove_columns=["text"])
    ds.set_format("torch")
    return ds


def run(config_path: str = "configs/distill.yaml") -> Path:
    cfg = load_yaml(config_path)
    tcfg = cfg.get("training", {})
    seed = tcfg.get("seed", 42)
    set_seed(seed)
    setup_logging()

    # ── Run directory ───────────────────────────────────────
    run_dir = make_run_dir(
        tcfg.get("output_dir", "results/distill/checkpoints"),
        tag="sft",
    )
    snapshot_configs([config_path], run_dir)
    save_metadata(collect_metadata(seed, cfg), run_dir)

    # ── Lazy imports (heavy) ────────────────────────────────
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    # ── Model + tokeniser ───────────────────────────────────
    model_name = tcfg.get("student_model", "gpt2")
    logger.info("Loading student model", extra={"model": model_name})

    tokeniser = AutoTokenizer.from_pretrained(model_name)
    if tokeniser.pad_token is None:
        tokeniser.pad_token = tokeniser.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if tcfg.get("bf16") else torch.float16,
    )

    # ── LoRA ────────────────────────────────────────────────
    lora_cfg = tcfg.get("lora", {})
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("lora_alpha", 16),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ── Dataset ─────────────────────────────────────────────
    dataset_path = tcfg.get(
        "dataset_path", "results/distill/teacher_outputs.jsonl",
    )
    max_seq_length = tcfg.get("max_seq_length", 2048)
    ds = _prepare_dataset(dataset_path, tokeniser, max_seq_length)
    logger.info("Dataset ready", extra={"samples": len(ds)})

    # ── Training arguments ──────────────────────────────────
    log_cfg = tcfg.get("logging", {})
    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=tcfg.get("num_train_epochs", 3),
        per_device_train_batch_size=tcfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=tcfg.get("gradient_accumulation_steps", 4),
        learning_rate=tcfg.get("learning_rate", 2e-4),
        lr_scheduler_type=tcfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=tcfg.get("warmup_ratio", 0.03),
        weight_decay=tcfg.get("weight_decay", 0.0),
        fp16=tcfg.get("fp16", False),
        bf16=tcfg.get("bf16", True),
        gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        logging_steps=log_cfg.get("logging_steps", 10),
        save_strategy=log_cfg.get("save_strategy", "steps"),
        save_steps=log_cfg.get("save_steps", 200),
        evaluation_strategy=log_cfg.get("eval_strategy", "no"),
        report_to=log_cfg.get("report_to", "none"),
        seed=seed,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokeniser, mlm=False,
    )

    # ── Trainer ─────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
    )

    logger.info("Starting SFT training")
    trainer.train()

    # ── Save final adapter + tokeniser ──────────────────────
    final_dir = run_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokeniser.save_pretrained(str(final_dir))
    logger.info("Training complete", extra={"adapter_dir": str(final_dir)})

    return run_dir


# ── CLI ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Student SFT training")
    parser.add_argument("--config", default="configs/distill.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
