#!/usr/bin/env python3
"""Rebuild predictor datasets with ``model_tier`` when raw trace JSONL is missing.

Uses the same JSONL files that trained the original RF bundles on BSC:
  - results/phase_a/datasets/service_cost_phase_a.jsonl
  - results/phase_b/datasets/quality_ex_ante_phase_b.jsonl
  - results/phase_b/datasets/quality_post_hoc_phase_b.jsonl

Maps ``model_name`` (metadata column) → ``model_tier`` via configs/models.yaml.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from distill.dataset_utils import read_jsonl
from predictors.dataset_common import write_dataset_artifacts
from utils.config_loader import load_yaml


def _tier_from_model_name(model_name: str, name_to_tier: Dict[str, str]) -> str:
    text = (model_name or "").strip()
    if not text:
        return "unknown"
    if text in name_to_tier:
        return name_to_tier[text]
    low = text.lower()
    if "0.5b" in low or "0.5b-instruct" in low:
        return "student_tiny"
    if "sft-full-qwen2.5-1.5b" in low or "/distill/" in low and "1.5b" in low:
        return "student_small"
    if "1.5b-instruct" in low:
        return "student_small_base"
    if "3b-instruct" in low or "qwen2.5-3b" in low:
        return "student_q3b"
    if "7b-instruct" in low or "qwen2.5-7b" in low:
        return "student_mid"
    if "14b-instruct" in low or "qwen2.5-14b" in low:
        return "teacher"
    return "unknown"


def _load_name_to_tier(models_yaml: Path) -> Dict[str, str]:
    models = load_yaml(str(models_yaml))
    out: Dict[str, str] = {}
    for tier, node in (models or {}).items():
        if tier in ("dev",) or not isinstance(node, dict):
            continue
        name = str(node.get("name", "")).strip()
        if name:
            out[name] = str(tier)
    return out


def enrich_rows(rows: List[Dict[str, Any]], name_to_tier: Dict[str, str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    unknown = 0
    for row in rows:
        new_row = dict(row)
        tier = _tier_from_model_name(str(row.get("model_name", "")), name_to_tier)
        if tier == "unknown":
            unknown += 1
        new_row["model_tier"] = tier
        out.append(new_row)
    if unknown:
        print(f"  warning: {unknown} rows with unknown model_tier")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/predictors_model_tier/datasets")
    parser.add_argument("--models-yaml", default="configs/models.yaml")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    name_to_tier = _load_name_to_tier(Path(args.models_yaml))

    jobs = [
        ("service_cost", Path("results/phase_a/datasets/service_cost_phase_a.jsonl"), "target_service_cost"),
        ("quality_ex_ante", Path("results/phase_b/datasets/quality_ex_ante_phase_b.jsonl"), "target_correct"),
        ("quality_post_hoc", Path("results/phase_b/datasets/quality_post_hoc_phase_b.jsonl"), "target_correct"),
    ]

    for dataset_name, src, target in jobs:
        if not src.is_file():
            raise FileNotFoundError(f"Missing source dataset (needed on BSC): {src}")
        rows = read_jsonl(src)
        print(f"{dataset_name}: {len(rows)} rows from {src}")
        enriched = enrich_rows(rows, name_to_tier)
        artifacts = write_dataset_artifacts(
            rows=enriched,
            dataset_name=f"{dataset_name}_model_tier",
            target_column=target,
            output_dir=out_dir,
        )
        print(f"  -> {artifacts['jsonl']}")

    print("Done. Datasets ready for refine_phase_*_model_tier.yaml")


if __name__ == "__main__":
    main()
