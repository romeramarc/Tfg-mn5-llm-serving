"""Calibrate per-rung post-hoc thresholds for the smart-cascade policy.

For each ``model_tier`` (rung) the post-hoc predictor uses, we sweep the
acceptance threshold over its Phase B validation traces and pick the
threshold that minimises a per-rung cost function

    cost(thr) = fp_cost * FP(thr) + fn_cost * FN(thr)

where FP = "accept an incorrect answer at this rung" (under-escalate),
FN = "escalate a correct answer at this rung" (over-escalate). The default
``fp_cost=1, fn_cost=3`` mirrors the original Phase B selection logic but
**flipped**: the existing global threshold (0.7728) was calibrated with
``fp_cost=3, fn_cost=1`` to be paranoid about accepting wrong answers,
which is exactly why the cascade escalates >70% of traffic on holdout.

The (still-supported) ``--criterion precision_target`` mode finds the
smallest threshold meeting a per-rung precision target.

Why this exists:
    The single global threshold (0.7728) was calibrated on the aggregated
    Phase B corpus. The post-hoc predictor includes ``model_tier`` as a
    feature, and consistently assigns lower confidence to smaller rungs
    even when they are correct. Applying the same threshold across rungs
    pushes >70% of traffic into the cascade unnecessarily.

Usage (recommended, cost-aware)::

    python scripts/calibrate_per_rung_thresholds.py \
        --dataset results/predictors_model_tier/datasets/quality_post_hoc_model_tier.jsonl \
        --bundle  results/predictors_model_tier/phase_b/refined-qph-rf-20260521T141032Z/model_bundle.joblib \
        --criterion cost --fp-cost 1.0 --fn-cost 3.0 \
        --out results/predictors_model_tier/phase_b/per_rung_thresholds.yaml \
        --report-out results/predictors_model_tier/phase_b/per_rung_thresholds_report.json

The output YAML can be copied into the ``post_hoc_threshold_per_rung``
block of the relevant system in ``routing_eval_holdout_v2_routing_real.yaml``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predictors.inference import PredictorBundle  # noqa: E402
from predictors.training.common import predict_probability  # noqa: E402


def _load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _bucket_by_tier(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        tier = str(r.get("model_tier") or "").strip()
        if tier:
            buckets[tier].append(r)
    return buckets


def _score(bundle: PredictorBundle, records: List[Dict[str, Any]]) -> np.ndarray:
    feats = [{c: r.get(c) for c in bundle.feature_columns} for r in records]
    xt = bundle.vectorizer.transform(feats)
    return predict_probability(bundle.estimator, xt)


def _metrics(probs: np.ndarray, labels: np.ndarray, thr: float) -> Dict[str, float]:
    accept = probs >= thr
    tp = int(((labels == 1) & accept).sum())
    fp = int(((labels == 0) & accept).sum())
    fn = int(((labels == 1) & ~accept).sum())
    tn = int(((labels == 0) & ~accept).sum())
    n_accept = tp + fp
    precision = tp / max(n_accept, 1) if n_accept else 0.0
    recall = tp / max(tp + fn, 1) if (tp + fn) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_accept": n_accept,
        "fraction_accept": n_accept / max(len(labels), 1),
        "precision": precision,
        "recall": recall,
    }


def _calibrate_cost(
    probs: np.ndarray,
    labels: np.ndarray,
    fp_cost: float,
    fn_cost: float,
) -> Dict[str, Any]:
    """Minimise fp_cost * FP + fn_cost * FN over the probability grid."""
    grid = np.linspace(0.0, 1.0, 1001)
    best = None
    best_cost = float("inf")
    for thr in grid:
        m = _metrics(probs, labels, float(thr))
        cost = fp_cost * m["fp"] + fn_cost * m["fn"]
        if cost < best_cost:
            best_cost = cost
            best = {
                "threshold": float(thr),
                "method": "cost",
                "fp_cost": fp_cost,
                "fn_cost": fn_cost,
                "cost_at_threshold": float(cost),
                **{k: (float(v) if isinstance(v, float) else int(v)) for k, v in m.items()},
            }
    return best or {"threshold": 0.5, "method": "cost", "cost_at_threshold": float("inf")}


def _calibrate_precision_target(
    probs: np.ndarray,
    labels: np.ndarray,
    target_precision: float,
    min_recall: float,
) -> Dict[str, Any]:
    """Smallest threshold meeting precision >= target with recall >= min_recall."""
    grid = np.linspace(0.0, 1.0, 1001)
    for thr in grid:
        m = _metrics(probs, labels, float(thr))
        if m["n_accept"] == 0:
            continue
        if m["precision"] >= target_precision and m["recall"] >= min_recall:
            return {
                "threshold": float(thr),
                "method": "precision_target",
                "target_precision": target_precision,
                "min_recall": min_recall,
                **{k: (float(v) if isinstance(v, float) else int(v)) for k, v in m.items()},
            }
    # Fallback to best F1.
    f1_best: Dict[str, Any] = {"f1": -1.0, "threshold": 0.5, "method": "f1_fallback"}
    for thr in grid:
        m = _metrics(probs, labels, float(thr))
        if m["n_accept"] == 0:
            continue
        f1 = (2 * m["precision"] * m["recall"] /
              (m["precision"] + m["recall"])) if (m["precision"] + m["recall"]) else 0.0
        if f1 > f1_best["f1"]:
            f1_best = {
                "threshold": float(thr), "method": "f1_fallback", "f1": f1,
                **{k: (float(v) if isinstance(v, float) else int(v)) for k, v in m.items()},
            }
    f1_best.pop("f1", None)
    return f1_best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Phase B post-hoc jsonl dataset")
    parser.add_argument("--bundle", required=True, help="Phase B post-hoc model bundle (.joblib)")
    parser.add_argument(
        "--criterion",
        choices=["cost", "precision_target"],
        default="cost",
        help="Calibration criterion. Default 'cost' minimises fp_cost*FP + fn_cost*FN.",
    )
    parser.add_argument(
        "--fp-cost",
        type=float,
        default=1.0,
        help="Cost of accepting an incorrect answer (under-escalation). Default 1.0.",
    )
    parser.add_argument(
        "--fn-cost",
        type=float,
        default=3.0,
        help="Cost of escalating a correct answer (over-escalation). Default 3.0.",
    )
    parser.add_argument("--target-precision", type=float, default=0.80)
    parser.add_argument("--min-recall", type=float, default=0.05)
    parser.add_argument(
        "--teacher-threshold",
        type=float,
        default=1.00,
        help="Threshold for the teacher rung (default 1.00 = always accept as final)",
    )
    parser.add_argument("--out", required=True, help="Output YAML path")
    parser.add_argument(
        "--report-out",
        default=None,
        help="Optional JSON report with per-rung diagnostics",
    )
    args = parser.parse_args()

    bundle = PredictorBundle(args.bundle)
    records = _load_records(Path(args.dataset))
    buckets = _bucket_by_tier(records)

    if not buckets:
        print("[calibrate] no records bucketed by model_tier. Aborting.", file=sys.stderr)
        return 2

    per_rung: Dict[str, float] = {}
    report: Dict[str, Any] = {
        "criterion": args.criterion,
        "fp_cost": args.fp_cost,
        "fn_cost": args.fn_cost,
        "target_precision": args.target_precision,
        "min_recall": args.min_recall,
        "bundle": args.bundle,
        "dataset": args.dataset,
        "rungs": {},
    }

    for tier, recs in sorted(buckets.items()):
        if tier == "teacher":
            per_rung[tier] = float(args.teacher_threshold)
            report["rungs"][tier] = {
                "threshold": float(args.teacher_threshold),
                "note": "teacher is accepted_final; threshold not applied",
            }
            continue
        labels = np.array(
            [int(r.get("target_correct", r.get("correct", 0)) or 0) for r in recs]
        )
        probs = _score(bundle, recs)
        if args.criterion == "cost":
            info = _calibrate_cost(probs, labels, args.fp_cost, args.fn_cost)
        else:
            info = _calibrate_precision_target(
                probs, labels, args.target_precision, args.min_recall
            )
        per_rung[tier] = float(info["threshold"])
        info["n_samples"] = len(recs)
        info["base_rate_correct"] = float(labels.mean()) if len(labels) else 0.0
        report["rungs"][tier] = info
        print(
            f"[calibrate] {tier}: n={len(recs)} base_rate={info['base_rate_correct']:.3f} "
            f"-> threshold={info['threshold']:.4f} "
            f"prec={info.get('precision', 0.0):.3f} "
            f"rec={info.get('recall', 0.0):.3f} "
            f"frac_accept={info.get('fraction_accept', 0.0):.3f} "
            f"({info['method']})"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump({"post_hoc_threshold_per_rung": per_rung}, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[calibrate] wrote {out_path}")

    if args.report_out:
        rp = Path(args.report_out)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[calibrate] wrote {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
