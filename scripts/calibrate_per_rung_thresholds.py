"""Calibrate per-rung post-hoc thresholds for the smart-cascade policy.

For each ``model_tier`` (rung) the post-hoc predictor uses, we sweep the
acceptance threshold over its Phase B validation traces and pick the
smallest threshold that meets a per-rung precision target (default 0.80).
That threshold is the minimum confidence at which we are willing to stop
the cascade at that rung without escalating.

Why this exists:
    The single global threshold (0.7728) was calibrated on the aggregated
    Phase B corpus. The post-hoc predictor includes ``model_tier`` as a
    feature, and consistently assigns lower confidence to smaller rungs
    even when they are correct. Applying the same threshold across rungs
    pushes >70% of traffic into the cascade unnecessarily.

Usage::

    python scripts/calibrate_per_rung_thresholds.py \
        --dataset results/predictors_model_tier/datasets/quality_post_hoc_model_tier.jsonl \
        --bundle  results/predictors_model_tier/phase_b/refined-qph-rf-20260521T141032Z/model_bundle.joblib \
        --target-precision 0.80 \
        --min-recall 0.05 \
        --out configs/per_rung_thresholds.yaml

The output YAML can be ``include``-d or copied into the policy_overrides
block of ``routing_eval_holdout_v2_routing_real.yaml``.
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


def _calibrate(
    probs: np.ndarray,
    labels: np.ndarray,
    target_precision: float,
    min_recall: float,
) -> Dict[str, Any]:
    """Return the smallest threshold meeting precision >= target with
    recall >= min_recall. Falls back to the threshold maximising F1 if
    no value satisfies the constraint."""
    grid = np.linspace(0.0, 1.0, 1001)
    best = None
    for thr in grid:
        accept = probs >= thr
        n_accept = int(accept.sum())
        if n_accept == 0:
            continue
        prec = float((labels[accept] == 1).mean())
        rec = float((labels[accept] == 1).sum() / max(int((labels == 1).sum()), 1))
        if prec >= target_precision and rec >= min_recall:
            best = {
                "threshold": float(thr),
                "n_accept": n_accept,
                "fraction_accept": float(n_accept / len(labels)),
                "precision_at_threshold": prec,
                "recall_at_threshold": rec,
                "method": "precision_target",
            }
            break
    if best is not None:
        return best

    f1_best = {"f1": -1.0}
    for thr in grid:
        accept = probs >= thr
        n_accept = int(accept.sum())
        if n_accept == 0:
            continue
        tp = int(((labels == 1) & accept).sum())
        fp = int(((labels == 0) & accept).sum())
        fn = int(((labels == 1) & ~accept).sum())
        if tp + fp == 0 or tp + fn == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > f1_best["f1"]:
            f1_best = {
                "threshold": float(thr),
                "n_accept": n_accept,
                "fraction_accept": float(n_accept / len(labels)),
                "precision_at_threshold": prec,
                "recall_at_threshold": rec,
                "f1": f1,
                "method": "f1_fallback",
            }
    f1_best.pop("f1", None)
    return f1_best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Phase B post-hoc jsonl dataset")
    parser.add_argument("--bundle", required=True, help="Phase B post-hoc model bundle (.joblib)")
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
        info = _calibrate(probs, labels, args.target_precision, args.min_recall)
        per_rung[tier] = float(info["threshold"])
        info["n_samples"] = len(recs)
        info["base_rate_correct"] = float(labels.mean()) if len(labels) else 0.0
        report["rungs"][tier] = info
        print(
            f"[calibrate] {tier}: n={len(recs)} base_rate={info['base_rate_correct']:.3f} "
            f"-> threshold={info['threshold']:.4f} "
            f"prec={info['precision_at_threshold']:.3f} "
            f"rec={info['recall_at_threshold']:.3f} "
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
