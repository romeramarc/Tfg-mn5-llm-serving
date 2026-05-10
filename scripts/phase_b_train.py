"""
scripts/phase_b_train.py
========================
End-to-end driver for Phase B's dataset build + quality predictor training.
Reads ``configs/phase_b.yaml`` and chains:

    1) predictors.builders.build_ex_ante_dataset
    2) predictors.builders.build_post_hoc_dataset
    3) predictors.training.run_training (quality_ex_ante, classification)
    4) predictors.training.run_training (quality_post_hoc, classification)

into one CLI so the SLURM template stays trivial. Both stages already exist
in the repository; this script only orchestrates them and emits a single
combined report.

Usage
-----
    python scripts/phase_b_train.py --config configs/phase_b.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/phase_b_train.py`
    sys.path.insert(0, str(REPO_ROOT))

from predictors.builders.build_ex_ante_dataset import build_ex_ante_dataset
from predictors.builders.build_post_hoc_dataset import build_post_hoc_dataset
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
        # Glob did not match → still pass the pattern so the builders
        # surface a clear "no traces found" error.
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase B — dataset build + train (×2)")
    parser.add_argument("--config", default="configs/phase_b.yaml")
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

    output_dir = Path(ds_cfg.get("output_dir", "results/phase_b/datasets"))
    ex_ante_name = str(ds_cfg.get("ex_ante_dataset_name", "quality_ex_ante_phase_b"))
    post_hoc_name = str(ds_cfg.get("post_hoc_dataset_name", "quality_post_hoc_phase_b"))

    expanded = _expand_globs(list(patterns))
    logger.info(
        "Resolving Phase B trace files",
        extra={
            "patterns": patterns,
            "trace_files_resolved": len(expanded),
            "output_dir": str(output_dir),
        },
    )

    if not expanded:
        raise FileNotFoundError(
            f"No trace files found for Phase B with patterns: {patterns}. "
            "Check that capture jobs produced results/phase_b/captures/**/trace.jsonl."
        )

    # ── 1) Build ex-ante dataset ──────────────────────────────
    ex_ante_report_path = output_dir / f"{ex_ante_name}_build_report.json"
    ex_ante_report = build_ex_ante_dataset(
        input_patterns=expanded,
        output_dir=output_dir,
        dataset_name=ex_ante_name,
        report_path=ex_ante_report_path,
    )
    logger.info(
        "Phase B ex-ante dataset built",
        extra={"kept_rows": ex_ante_report["kept_rows"]},
    )
    if int(ex_ante_report.get("kept_rows", 0)) == 0:
        raise RuntimeError(
            "No rows kept by build_ex_ante_dataset — quality_ex_ante predictor "
            "cannot be trained. Check that capture traces have correct=True/False."
        )

    # ── 2) Build post-hoc dataset (same source, different feature set) ──
    post_hoc_report_path = output_dir / f"{post_hoc_name}_build_report.json"
    post_hoc_report = build_post_hoc_dataset(
        input_patterns=expanded,
        output_dir=output_dir,
        dataset_name=post_hoc_name,
        report_path=post_hoc_report_path,
    )
    logger.info(
        "Phase B post-hoc dataset built",
        extra={"kept_rows": post_hoc_report["kept_rows"]},
    )
    if int(post_hoc_report.get("kept_rows", 0)) == 0:
        raise RuntimeError(
            "No rows kept by build_post_hoc_dataset — quality_post_hoc predictor "
            "cannot be trained."
        )

    family = str(pred_cfg.get("family", "gradient_boosting"))
    output_root = Path(pred_cfg.get("output_root", "results/phase_b/predictors"))
    seed = int(pred_cfg.get("seed", 42))
    train_ratio = float(pred_cfg.get("train_ratio", 0.7))
    val_ratio = float(pred_cfg.get("val_ratio", 0.15))

    # ── 3) Train quality_ex_ante predictor ───────────────────
    ex_ante_jsonl = output_dir / f"{ex_ante_name}.jsonl"
    ex_ante_meta = output_dir / f"{ex_ante_name}_meta.json"
    train_report_ex_ante = run_training(
        predictor_id="quality_ex_ante",
        task="classification",
        dataset_jsonl=ex_ante_jsonl,
        dataset_meta_json=ex_ante_meta,
        target_column="target_correct",
        model_family=family,
        output_root=output_root,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    # ── 4) Train quality_post_hoc predictor ──────────────────
    post_hoc_jsonl = output_dir / f"{post_hoc_name}.jsonl"
    post_hoc_meta = output_dir / f"{post_hoc_name}_meta.json"
    train_report_post_hoc = run_training(
        predictor_id="quality_post_hoc",
        task="classification",
        dataset_jsonl=post_hoc_jsonl,
        dataset_meta_json=post_hoc_meta,
        target_column="target_correct",
        model_family=family,
        output_root=output_root,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    final = {
        "build": {
            "ex_ante": ex_ante_report,
            "post_hoc": post_hoc_report,
        },
        "train": {
            "ex_ante": train_report_ex_ante,
            "post_hoc": train_report_post_hoc,
        },
    }
    print(json.dumps(final, indent=2, default=str))


if __name__ == "__main__":
    main()
