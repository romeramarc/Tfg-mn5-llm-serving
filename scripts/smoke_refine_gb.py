#!/usr/bin/env python3
"""Quick smoke test: GB estimators + one refine step (no SLURM)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from predictors.training.refine import _empty_estimator, refine_classifier, refine_regressor  # noqa: E402


def main() -> int:
    _empty_estimator(task="regression", family="gradient_boosting", seed=42)
    _empty_estimator(task="classification", family="gradient_boosting", seed=42)
    print("estimators: OK")

    root = Path(tempfile.mkdtemp(prefix="smoke_gb_"))
    n = 80
    rows = []
    for i in range(n):
        rows.append(
            {
                "query_id": f"q{i:03d}",
                "benchmark": "b",
                "model_name": "m",
                "run_id": str(i),
                "feat_a": float(i % 7),
                "feat_b": f"c{i % 3}",
                "target_service_cost": 0.1 * i,
                "target_correct": i % 2,
            }
        )
    import json

    reg_p = root / "reg.jsonl"
    cls_p = root / "cls.jsonl"
    with reg_p.open("w", encoding="utf-8") as fh:
        for r in rows:
            o = {k: v for k, v in r.items() if k != "target_correct"}
            fh.write(json.dumps(o) + "\n")
    with cls_p.open("w", encoding="utf-8") as fh:
        for r in rows:
            o = {k: v for k, v in r.items() if k != "target_service_cost"}
            fh.write(json.dumps(o) + "\n")
    meta = {"feature_columns": ["feat_a", "feat_b"]}
    (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    tiny = {
        "learning_rate": [0.1],
        "max_iter": [50],
        "max_depth": [6],
        "max_leaf_nodes": [31],
        "l2_regularization": [0.0],
    }
    refine_regressor(
        predictor_id="service_cost",
        dataset_jsonl=reg_p,
        dataset_meta_json=root / "meta.json",
        target_column="target_service_cost",
        family="gradient_boosting",
        output_root=root / "out_reg",
        seed=42,
        train_ratio=0.7,
        val_ratio=0.15,
        n_iter=2,
        n_splits=3,
        scoring="neg_mean_absolute_error",
        search_space=tiny,
    )
    refine_classifier(
        predictor_id="quality_ex_ante",
        dataset_jsonl=cls_p,
        dataset_meta_json=root / "meta.json",
        target_column="target_correct",
        family="gradient_boosting",
        output_root=root / "out_cls",
        seed=42,
        train_ratio=0.7,
        val_ratio=0.15,
        n_iter=2,
        n_splits=3,
        scoring="roc_auc",
        search_space=tiny,
        threshold_criterion="f1",
        threshold_kwargs={},
    )
    print("refine GB smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
