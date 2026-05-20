#!/usr/bin/env python3
"""Materialize a routing-eval YAML from refined predictor selections.

The refinement jobs write one ``REFINEMENT_SELECTION.json`` per phase. This
script picks the winning bundles, reads classification thresholds from the
joblib payloads, and writes a concrete holdout config ready for BSC evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import yaml


PREDICTOR_TO_PHASE = {
    "quality_ex_ante": "phase_b",
    "quality_post_hoc": "phase_b",
    "service_cost": "phase_a",
}


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _bundle_from_selection(selection_path: Path, predictor_id: str) -> Path:
    payload = _read_json(selection_path)
    node = ((payload.get("selection") or {}).get(predictor_id) or {})
    model_dir = node.get("best_model_dir")
    if not model_dir:
        raise KeyError(f"Missing best_model_dir for {predictor_id} in {selection_path}")
    bundle = Path(model_dir) / "model_bundle.joblib"
    if not bundle.is_file():
        raise FileNotFoundError(f"Selected bundle does not exist: {bundle}")
    return bundle


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _threshold(bundle_path: Path, default: float) -> float:
    payload = joblib.load(bundle_path)
    value = payload.get("threshold", default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_config(
    *,
    base_config: Path,
    phase_a_selection: Path,
    phase_b_selection: Path,
    output_config: Path,
    project_root: Path,
) -> Dict[str, Any]:
    with base_config.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    selections = {
        "phase_a": phase_a_selection,
        "phase_b": phase_b_selection,
    }
    bundles = {
        predictor_id: _bundle_from_selection(selections[phase], predictor_id)
        for predictor_id, phase in PREDICTOR_TO_PHASE.items()
    }

    pred_cfg = cfg.setdefault("predictors", {})
    pred_cfg["bundles"] = {
        "quality_ex_ante": _rel(bundles["quality_ex_ante"], project_root),
        "quality_post_hoc": _rel(bundles["quality_post_hoc"], project_root),
        "service_cost": _rel(bundles["service_cost"], project_root),
    }
    pred_cfg["thresholds"] = {
        "quality_ex_ante": _threshold(bundles["quality_ex_ante"], 0.632),
        "quality_post_hoc": _threshold(bundles["quality_post_hoc"], 0.716),
    }
    pred_cfg["source"] = {
        "phase_a_selection": _rel(phase_a_selection, project_root),
        "phase_b_selection": _rel(phase_b_selection, project_root),
        "requires_model_tier_feature": True,
    }

    output_config.parent.mkdir(parents=True, exist_ok=True)
    with output_config.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
    return {
        "output_config": str(output_config),
        "bundles": pred_cfg["bundles"],
        "thresholds": pred_cfg["thresholds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write routing holdout config from refined predictor selections")
    parser.add_argument("--base-config", default="configs/routing_eval_holdout_v2.yaml")
    parser.add_argument("--phase-a-selection", default="results/predictors_model_tier/phase_a/REFINEMENT_SELECTION.json")
    parser.add_argument("--phase-b-selection", default="results/predictors_model_tier/phase_b/REFINEMENT_SELECTION.json")
    parser.add_argument("--output-config", default="configs/routing_eval_holdout_v2_retrained.yaml")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    result = build_config(
        base_config=Path(args.base_config),
        phase_a_selection=Path(args.phase_a_selection),
        phase_b_selection=Path(args.phase_b_selection),
        output_config=Path(args.output_config),
        project_root=Path(args.project_root),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
