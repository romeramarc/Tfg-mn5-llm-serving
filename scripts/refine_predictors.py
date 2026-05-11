#!/usr/bin/env python3
"""
Refine RF / GB models for one or more predictors and pick the best per predictor.

Driver script around ``predictors.training.refine``. Reads a YAML with:

  output_root: results/<phase>/predictors_refined
  primary_metric: <test_mae|test_roc_auc|test_f1|...>
  primary_metric_direction: <min|max>
  predictors:
    - id: service_cost
      task: regression
      dataset_jsonl: results/phase_a/dataset/service_cost.dataset.jsonl
      dataset_meta_json: results/phase_a/dataset/service_cost.dataset_meta.json
      target_column: latency_s
      seed: 42
      train_ratio: 0.7
      val_ratio: 0.15
      n_iter: 30
      n_splits: 3
      scoring: neg_mean_absolute_error
      families: [random_forest, gradient_boosting]
      search_space: {...}            # optional, falls back to defaults
    - id: quality_ex_ante
      task: classification
      ...
      threshold:
        criterion: cost               # f1 | youden | cost | min_recall | min_precision
        fp_cost: 2.0
        fn_cost: 1.0
        min_recall: 0.8               # only used when criterion == min_recall
        min_precision: 0.8            # only used when criterion == min_precision

Writes:
  * One refined model directory per (predictor, family).
  * ``refined_<predictor_id>_comparison.csv`` summarising both families.
  * ``REFINEMENT_SELECTION.json`` listing the chosen model per predictor.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predictors.training.refine import (  # noqa: E402
    DEFAULT_SCORING,
    DEFAULT_SEARCH_SPACES,
    refine_classifier,
    refine_regressor,
)


REFINEMENT_RATIONALE = """
The refinement pipeline targets RandomForest and HistGradientBoosting because
the baseline battery (results/baseline_comparison/SUMMARY_FOR_MEMORIA.md) shows
they dominate every predictor on the held-out test split. For each predictor we
run RandomizedSearchCV with GroupKFold over query_id, so the same query never
appears in both training and validation folds. For classification predictors we
additionally pick the operating threshold on the validation split using the
criterion configured in YAML (F1, Youden's J, asymmetric FP/FN cost,
recall-floor or precision-floor); the test split is only used to report final
metrics. Selection across families is driven by ``primary_metric``.
""".strip()


def _resolve_search_space(task: str, family: str, override: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    base = DEFAULT_SEARCH_SPACES.get(task, {}).get(family, {})
    if not override:
        return {k: list(v) for k, v in base.items()}
    merged = {k: list(v) for k, v in base.items()}
    for key, vals in override.items():
        merged[str(key)] = list(vals) if isinstance(vals, (list, tuple)) else [vals]
    return merged


def _metric_lookup(report: Mapping[str, Any], metric: str) -> Optional[float]:
    """Resolve a dotted-or-flat metric name from a refine report."""
    flat = {
        # Regression
        "test_mae": (report.get("test_global") or {}).get("mae"),
        "test_rmse": (report.get("test_global") or {}).get("rmse"),
        "test_r2": (report.get("test_global") or {}).get("r2"),
        "test_mape": (report.get("test_global") or {}).get("mape"),
        # Classification (probability-level)
        "test_roc_auc": (report.get("test_global") or {}).get("roc_auc"),
        "test_average_precision": (report.get("test_global") or {}).get("average_precision"),
        "test_brier": (report.get("test_global") or {}).get("brier"),
        "test_log_loss": (report.get("test_global") or {}).get("log_loss"),
        "test_ece_abs": (report.get("test_global") or {}).get("ece_abs"),
        # Classification (threshold-level)
        "test_f1_at_threshold": (report.get("test_at_threshold") or {}).get("f1"),
        "test_precision_at_threshold": (report.get("test_at_threshold") or {}).get("precision"),
        "test_recall_at_threshold": (report.get("test_at_threshold") or {}).get("recall"),
        "test_accuracy_at_threshold": (report.get("test_at_threshold") or {}).get("accuracy"),
        # Timing
        "predict_us_per_row": (report.get("timing") or {}).get("predict_us_per_row"),
        "fit_seconds": (report.get("search") or {}).get("search_seconds"),
    }
    val = flat.get(metric)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _row_for_comparison(predictor_id: str, family: str, status: str, report: Optional[Dict[str, Any]], error: Optional[str]) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "predictor_id": predictor_id,
        "model_family": family,
        "status": status,
        "error": error,
        "model_dir": None,
        "metrics_json": None,
        "best_cv_score": None,
        "best_params": None,
        "threshold": None,
        "threshold_criterion": None,
    }
    if not report:
        return base
    base.update(
        {
            "model_dir": report.get("model_dir"),
            "metrics_json": report.get("metrics_json"),
            "best_cv_score": report.get("best_cv_score"),
            "best_params": json.dumps(report.get("best_params") or {}, sort_keys=True),
            "threshold": report.get("threshold"),
            "threshold_criterion": report.get("threshold_criterion"),
        }
    )
    test_global = report.get("test_global") or {}
    test_thr = report.get("test_at_threshold") or {}
    val_global = report.get("val_global") or {}
    timing = report.get("timing") or {}
    for key in (
        "mae", "rmse", "r2", "mape",
        "roc_auc", "average_precision", "brier", "log_loss", "ece_abs",
        "accuracy", "precision", "recall", "f1", "positive_rate",
    ):
        base[f"test_{key}"] = test_global.get(key)
        base[f"val_{key}"] = val_global.get(key)
    for key in ("threshold", "tp", "fp", "fn", "tn",
                "accuracy", "precision", "recall", "f1", "fpr", "tnr"):
        base[f"test_thr_{key}"] = test_thr.get(key)
    for key in ("predict_time_test_s", "predict_us_per_row", "test_rows"):
        base[key] = timing.get(key)
    return base


def _write_comparison_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k in seen:
                continue
            seen.add(k)
            cols.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in cols})


def _select_best(reports: List[Dict[str, Any]], primary_metric: str, direction: str) -> Optional[Dict[str, Any]]:
    eligible = [r for r in reports if r.get("status") == "ok"]
    if not eligible:
        return None
    direction = direction.lower().strip()
    sign = 1.0 if direction == "max" else -1.0
    best = None
    best_value = None
    for r in eligible:
        report = r.get("_report") or {}
        val = _metric_lookup(report, primary_metric)
        if val is None:
            continue
        score = sign * val
        if best_value is None or score > best_value:
            best_value = score
            best = r
    return best


def run_refinement(cfg_path: Path) -> Dict[str, Any]:
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    output_root = Path(cfg.get("output_root", "results/refined")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    primary_metric = str(cfg.get("primary_metric", "test_roc_auc"))
    primary_direction = str(cfg.get("primary_metric_direction", "max"))
    predictors_cfg = cfg.get("predictors") or []

    selection: Dict[str, Any] = {}
    all_reports: List[Dict[str, Any]] = []

    for pred_cfg in predictors_cfg:
        predictor_id = str(pred_cfg["id"])
        task = str(pred_cfg["task"]).lower()
        dataset_jsonl = Path(pred_cfg["dataset_jsonl"]).resolve()
        meta = pred_cfg.get("dataset_meta_json")
        dataset_meta_json = Path(meta).resolve() if meta else None
        target_column = str(pred_cfg["target_column"])
        families = [str(f).lower() for f in (pred_cfg.get("families") or ["random_forest", "gradient_boosting"])]
        seed = int(pred_cfg.get("seed", 42))
        train_ratio = float(pred_cfg.get("train_ratio", 0.7))
        val_ratio = float(pred_cfg.get("val_ratio", 0.15))
        n_iter = int(pred_cfg.get("n_iter", 30))
        n_splits = int(pred_cfg.get("n_splits", 3))
        scoring = str(pred_cfg.get("scoring", DEFAULT_SCORING[task]))
        threshold_cfg = pred_cfg.get("threshold") or {}
        threshold_criterion = str(threshold_cfg.get("criterion", "f1"))
        threshold_kwargs = {
            "fp_cost": threshold_cfg.get("fp_cost", 1.0),
            "fn_cost": threshold_cfg.get("fn_cost", 1.0),
            "min_recall": threshold_cfg.get("min_recall"),
            "min_precision": threshold_cfg.get("min_precision"),
        }

        per_predictor_reports: List[Dict[str, Any]] = []
        for family in families:
            if family not in ("random_forest", "gradient_boosting"):
                print(f"[WARN] {predictor_id}: skipping unsupported family '{family}' for refinement.")
                continue
            search_space = _resolve_search_space(task, family, pred_cfg.get("search_space", {}).get(family) if isinstance(pred_cfg.get("search_space"), dict) else None)
            print(
                f"[refine] {predictor_id} :: {family} :: n_iter={n_iter} cv={n_splits} scoring={scoring}",
                flush=True,
            )
            try:
                if task == "classification":
                    report = refine_classifier(
                        predictor_id=predictor_id,
                        dataset_jsonl=dataset_jsonl,
                        dataset_meta_json=dataset_meta_json,
                        target_column=target_column,
                        family=family,
                        output_root=output_root,
                        seed=seed,
                        train_ratio=train_ratio,
                        val_ratio=val_ratio,
                        n_iter=n_iter,
                        n_splits=n_splits,
                        scoring=scoring,
                        search_space=search_space,
                        threshold_criterion=threshold_criterion,
                        threshold_kwargs=threshold_kwargs,
                    )
                elif task == "regression":
                    report = refine_regressor(
                        predictor_id=predictor_id,
                        dataset_jsonl=dataset_jsonl,
                        dataset_meta_json=dataset_meta_json,
                        target_column=target_column,
                        family=family,
                        output_root=output_root,
                        seed=seed,
                        train_ratio=train_ratio,
                        val_ratio=val_ratio,
                        n_iter=n_iter,
                        n_splits=n_splits,
                        scoring=scoring,
                        search_space=search_space,
                    )
                else:
                    raise ValueError(f"Unknown task '{task}'")
                row = _row_for_comparison(predictor_id, family, "ok", report, None)
                row["_report"] = report
                per_predictor_reports.append(row)
                print(
                    f"  -> ok :: {report.get('model_dir')} :: best_cv_score={report.get('best_cv_score'):.4f}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                err = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                print(f"  -> FAILED :: {err}", flush=True)
                per_predictor_reports.append(_row_for_comparison(predictor_id, family, "failed", None, err))

        # Write per-predictor comparison CSV
        comparison_csv = output_root / f"refined_{predictor_id}_comparison.csv"
        rows_for_csv = [{k: v for k, v in r.items() if k != "_report"} for r in per_predictor_reports]
        _write_comparison_csv(comparison_csv, rows_for_csv)
        print(f"[refine] comparison -> {comparison_csv}")

        # Select best
        best = _select_best(per_predictor_reports, primary_metric, primary_direction)
        selection[predictor_id] = {
            "primary_metric": primary_metric,
            "primary_metric_direction": primary_direction,
            "comparison_csv": str(comparison_csv),
            "candidates": [
                {
                    "family": r.get("model_family"),
                    "status": r.get("status"),
                    "metric_value": _metric_lookup(r.get("_report") or {}, primary_metric),
                    "model_dir": r.get("model_dir"),
                }
                for r in per_predictor_reports
            ],
            "best_family": (best or {}).get("model_family"),
            "best_model_dir": (best or {}).get("model_dir"),
            "best_metric_value": _metric_lookup((best or {}).get("_report") or {}, primary_metric) if best else None,
        }
        all_reports.extend(per_predictor_reports)

    summary_payload = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "config_path": str(cfg_path),
        "rationale": REFINEMENT_RATIONALE,
        "primary_metric": primary_metric,
        "primary_metric_direction": primary_direction,
        "selection": selection,
    }
    selection_path = output_root / "REFINEMENT_SELECTION.json"
    selection_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"[refine] selection -> {selection_path}")
    return summary_payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Refine RF/GB predictors and choose the best per predictor.")
    parser.add_argument("--config", required=True, type=Path, help="YAML config file with predictors to refine.")
    args = parser.parse_args(argv)
    cfg_path = args.config.resolve()
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2
    run_refinement(cfg_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
