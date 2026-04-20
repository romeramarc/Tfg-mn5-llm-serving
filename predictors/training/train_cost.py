from __future__ import annotations

import argparse
import json
from pathlib import Path

from predictors.training.common import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train service-cost predictor")
    parser.add_argument("--dataset", default="results/predictors/datasets/service_cost_iter2_real_hardened.jsonl")
    parser.add_argument("--meta", default="results/predictors/datasets/service_cost_iter2_real_hardened_meta.json")
    parser.add_argument("--family", choices=["linear", "random_forest", "gradient_boosting"], default="random_forest")
    parser.add_argument("--output-root", default="results/predictors/models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    result = run_training(
        predictor_id="service_cost",
        task="regression",
        dataset_jsonl=Path(args.dataset),
        dataset_meta_json=Path(args.meta),
        target_column="target_service_cost",
        model_family=args.family,
        output_root=Path(args.output_root),
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
