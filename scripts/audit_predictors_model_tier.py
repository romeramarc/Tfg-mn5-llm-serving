#!/usr/bin/env python3
"""Audit predictor retrain with model_tier (pred-tier job outputs).

Run on BSC after slurm/retrain_predictors_model_tier.sbatch completes:

  python scripts/audit_predictors_model_tier.py
  python scripts/audit_predictors_model_tier.py --config configs/routing_eval_holdout_v2_retrained.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from routing.predictor_runtime import EvalPredictorSuite, build_trace
from utils.config_loader import load_yaml

DEFAULT_CONFIG = "configs/routing_eval_holdout_v2_retrained.yaml"
DATASET_DIR = Path("results/predictors_model_tier/datasets")
PHASE_A_SEL = Path("results/predictors_model_tier/phase_a/REFINEMENT_SELECTION.json")
PHASE_B_SEL = Path("results/predictors_model_tier/phase_b/REFINEMENT_SELECTION.json")
RUNG_ORDER = ["student_small", "student_q3b", "student_mid", "teacher"]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _tier_counts(jsonl: Path, limit: int = 500_000) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    with jsonl.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= limit:
                break
            row = json.loads(line)
            tier = str(row.get("model_tier") or "MISSING")
            counts[tier] = counts.get(tier, 0) + 1
    return counts


def _importance_rank(bundle_path: Path, feature: str) -> Tuple[float, int, int]:
    imp_csv = bundle_path.parent / "feature_importance.csv"
    if not imp_csv.is_file():
        return 0.0, -1, 0
    rows: List[Dict[str, str]] = []
    with imp_csv.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return 0.0, -1, 0
    ranked = sorted(rows, key=lambda r: float(r.get("importance", 0) or 0), reverse=True)
    names = [r.get("feature", "") for r in ranked]
    try:
        idx = names.index(feature)
    except ValueError:
        return 0.0, -1, len(names)
    return float(ranked[idx].get("importance", 0) or 0), idx + 1, len(names)


def _bundle_has_feature(bundle_path: Path, feature: str) -> bool:
    payload = joblib.load(bundle_path)
    cols = payload.get("feature_columns") or []
    return feature in cols


def audit(config_path: Path, lambda_smoke: float) -> int:
    errors: List[str] = []
    warnings: List[str] = []

    print("=== AUDIT predictors (model_tier) ===\n")

    if not config_path.is_file():
        errors.append(f"Missing routing config: {config_path}")
        _report(errors, warnings)
        return 1

    cfg = load_yaml(str(config_path))
    pred = cfg.get("predictors") or {}
    bundles = pred.get("bundles") or {}
    thresholds = pred.get("thresholds") or {}

    print(f"Config: {config_path}")
    print(f"Thresholds: {json.dumps(thresholds, indent=2)}\n")

    # --- datasets ---
    print("--- Datasets ---")
    for name in (
        "quality_ex_ante_model_tier",
        "quality_post_hoc_model_tier",
        "service_cost_model_tier",
    ):
        meta_path = DATASET_DIR / f"{name}_meta.json"
        jsonl_path = DATASET_DIR / f"{name}.jsonl"
        if not meta_path.is_file():
            errors.append(f"Missing {meta_path}")
            continue
        if not jsonl_path.is_file():
            errors.append(f"Missing {jsonl_path}")
            continue
        meta = _load_json(meta_path)
        feats = set(meta.get("feature_columns") or [])
        if "model_tier" not in feats:
            errors.append(f"model_tier not in features for {name}")
        rows = int(meta.get("row_count") or 0)
        tiers = _tier_counts(jsonl_path)
        unknown = tiers.get("unknown", 0)
        print(f"  {name}: rows={rows} tiers={dict(sorted(tiers.items()))}")
        if unknown:
            warnings.append(f"{name}: {unknown} rows with model_tier=unknown")
        tiny = tiers.get("student_tiny", 0)
        if tiny and name.startswith("quality"):
            warnings.append(f"{name}: still has {tiny} student_tiny rows (legacy 0.5B in training data)")

    # --- refinement selections ---
    print("\n--- Refinement (test metrics) ---")
    metric_expect = {
        "service_cost": ("test_mae", "min", 5000.0),
        "quality_ex_ante": ("test_roc_auc", "max", 0.55),
        "quality_post_hoc": ("test_roc_auc", "max", 0.60),
    }
    for label, sel_path in (("phase_a", PHASE_A_SEL), ("phase_b", PHASE_B_SEL)):
        if not sel_path.is_file():
            errors.append(f"Missing {sel_path}")
            continue
        payload = _load_json(sel_path)
        for pid, node in (payload.get("selection") or {}).items():
            metric = node.get("best_metric_value")
            mdir = node.get("best_model_dir")
            fam = node.get("best_family")
            print(f"  [{label}] {pid}: family={fam} metric={metric} dir={mdir}")
            if metric is None or not mdir:
                errors.append(f"No winner for {pid} in {sel_path}")
                continue
            bundle = Path(mdir) / "model_bundle.joblib"
            if not bundle.is_file():
                errors.append(f"Missing bundle: {bundle}")
            elif not _bundle_has_feature(bundle, "model_tier"):
                errors.append(f"model_tier not in feature_columns of {bundle}")
            else:
                imp, rank, nfeat = _importance_rank(bundle, "model_tier")
                print(f"         model_tier importance rank {rank}/{nfeat} (value={imp:.6f})")
                if rank < 0:
                    warnings.append(f"No feature_importance.csv for {pid}")
                elif rank > max(5, nfeat // 4):
                    warnings.append(
                        f"{pid}: model_tier rank {rank}/{nfeat} is low; routing may still tie often"
                    )
            key = metric_expect.get(pid)
            if key and metric is not None:
                mname, direction, bound = key
                val = float(metric)
                if direction == "min" and val > bound:
                    warnings.append(f"{pid}: {mname}={val:.4f} > soft bound {bound} (weak cost model)")
                if direction == "max" and val < bound:
                    warnings.append(f"{pid}: {mname}={val:.4f} < soft bound {bound} (weak quality model)")

    # --- bundles wired in YAML ---
    print("\n--- Bundles in YAML ---")
    for key, rel in bundles.items():
        path = Path(rel)
        if not path.is_file():
            errors.append(f"Bundle missing ({key}): {path}")
            continue
        payload = joblib.load(path)
        nfeat = len(payload.get("feature_columns") or [])
        print(f"  {key}: {rel} ({nfeat} features, model_tier={'model_tier' in (payload.get('feature_columns') or [])})")

    # --- routing smoke (utilities must differ across rungs) ---
    print(f"\n--- Routing smoke (lambda={lambda_smoke}) ---")
    try:
        suite = EvalPredictorSuite.from_config(pred)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"EvalPredictorSuite load failed: {exc}")
        _report(errors, warnings)
        return 1

    prompt = "Explain the difference between BFS and DFS on a graph."
    z = {"running": 2.0, "waiting": 4.0}
    scores: List[Tuple[str, float, float, float]] = []
    for role in RUNG_ORDER:
        trace = build_trace(
            prompt=prompt,
            benchmark="audit",
            example_id="0",
            request_id="audit-0",
            role=role,
            model_name=role,
            z_metrics=z,
            inflight_at_send=4.0,
            recent_p50_latency_ms=120.0,
        )
        q = suite.ex_ante_probability(trace)
        c = suite.predicted_cost(trace)
        u = q - lambda_smoke * c
        scores.append((role, q, c, u))
        print(f"  {role:14s} Q_ex={q:.4f}  C_hat={c:.1f} ms  U={u:.4f}")

    q_vals = [s[1] for s in scores]
    c_vals = [s[2] for s in scores]
    u_vals = [s[3] for s in scores]
    if len(set(round(v, 4) for v in q_vals)) < 2:
        errors.append("Q_ex identical for all rungs — model_tier not affecting quality predictor")
    if len(set(round(v, 1) for v in c_vals)) < 2:
        errors.append("C_hat identical for all rungs — model_tier not affecting cost predictor")
    if len(set(round(v, 4) for v in u_vals)) < 2:
        errors.append("Utility U identical for all rungs — routing would degenerate (tie-break by order)")
    else:
        best = max(scores, key=lambda x: x[3])
        print(f"\n  Argmax U -> {best[0]} (OK if not always student_small)")

    _report(errors, warnings)
    return 1 if errors else 0


def _report(errors: List[str], warnings: List[str]) -> None:
    print("\n=== SUMMARY ===")
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  [warn] {w}")
    if errors:
        print(f"FAILED ({len(errors)}):")
        for e in errors:
            print(f"  [FAIL] {e}")
    else:
        print("PASS — predictors look ready for holdout v2 routing.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit model_tier predictor retrain outputs")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--lambda-smoke", type=float, default=0.05, help="lambda for utility smoke test")
    args = parser.parse_args()
    return audit(Path(args.config), args.lambda_smoke)


if __name__ == "__main__":
    raise SystemExit(main())
