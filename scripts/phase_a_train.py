"""
scripts/phase_a_train.py
========================
End-to-end driver for Phase A's dataset build + service-cost predictor
training. Reads ``configs/phase_a.yaml`` and chains:

    1) predictors.builders.build_cost_dataset.build_cost_dataset
    2) predictors.training.common.run_training

into one CLI so the SLURM template stays trivial. Both stages already
exist in the repository; this script only orchestrates them.

Usage
-----
    python scripts/phase_a_train.py --config configs/phase_a.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/phase_a_train.py`
    sys.path.insert(0, str(REPO_ROOT))

from predictors.builders.build_cost_dataset import build_cost_dataset
from predictors.training.common import run_training
from utils.config_loader import load_yaml
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def _expand_globs(patterns: List[str]) -> List[str]:
    """Expand recursive globs and return paths that actually exist."""
    out: List[str] = []
    for pat in patterns:
        for p in sorted(Path().glob(pat)):
            if p.is_file():
                out.append(str(p))
        # Glob did not match → still pass the pattern so build_cost_dataset
        # surfaces a clear "no traces found" error.
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase A — dataset build + train")
    parser.add_argument("--config", default="configs/phase_a.yaml")
    args = parser.parse_args()

    setup_logging()
    cfg = load_yaml(args.config)

    ds_cfg: Dict[str, Any] = cfg.get("dataset", {})
    pred_cfg: Dict[str, Any] = cfg.get("predictor", {})

    patterns = ds_cfg.get("trace_glob_patterns") or []
    if not patterns:
        raise ValueError(
            "dataset.trace_glob_patterns is empty — nothing to ingest."
        )

    # ── 1) Build dataset ──────────────────────────────────────
    output_dir = Path(ds_cfg.get("output_dir", "results/phase_a/datasets"))
    dataset_name = ds_cfg.get("dataset_name", "service_cost_phase_a")
    report_path = output_dir / f"{dataset_name}_build_report.json"

    expanded = _expand_globs(list(patterns))
    logger.info(
        "Building Phase A dataset",
        extra={
            "patterns": patterns,
            "trace_files_resolved": len(expanded),
            "output_dir": str(output_dir),
        },
    )

    build_report = build_cost_dataset(
        input_patterns=expanded or list(patterns),
        output_dir=output_dir,
        dataset_name=dataset_name,
        report_path=report_path,
        feature_policy=str(ds_cfg.get("feature_policy", "extended_operational")),
        cost_mode=str(ds_cfg.get("cost_mode", "latency_ms")),
        latency_weight=float(ds_cfg.get("latency_weight", 1.0)),
        gpu_seconds_weight=float(ds_cfg.get("gpu_seconds_weight", 0.0)),
        energy_weight=float(ds_cfg.get("energy_weight", 0.0)),
    )
    logger.info(
        "Phase A dataset built",
        extra={"kept_rows": build_report["kept_rows"]},
    )

    if int(build_report.get("kept_rows", 0)) == 0:
        raise RuntimeError(
            "No rows kept by build_cost_dataset — predictor cannot be trained. "
            "Check that capture jobs produced trace.jsonl files."
        )

    dataset_jsonl = output_dir / f"{dataset_name}.jsonl"
    dataset_meta = output_dir / f"{dataset_name}_meta.json"

    # ── 2) Train predictor ───────────────────────────────────
    train_report = run_training(
        predictor_id="service_cost",
        task="regression",
        dataset_jsonl=dataset_jsonl,
        dataset_meta_json=dataset_meta,
        target_column="target_service_cost",
        model_family=str(pred_cfg.get("family", "gradient_boosting")),
        output_root=Path(pred_cfg.get("output_root", "results/phase_a/predictors")),
        seed=int(pred_cfg.get("seed", 42)),
        train_ratio=float(pred_cfg.get("train_ratio", 0.7)),
        val_ratio=float(pred_cfg.get("val_ratio", 0.15)),
    )

    final = {
        "build": build_report,
        "train": train_report,
    }
    print(json.dumps(final, indent=2, default=str))


if __name__ == "__main__":
    main()
