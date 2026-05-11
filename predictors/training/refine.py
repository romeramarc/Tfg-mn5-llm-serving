"""
Refinement stage for the most promising predictor families (RF and GB).

Builds on top of ``predictors.training.common``:
  * Same dataset, same JSONL, same group-aware split by ``query_id`` and seed.
  * Adds per-family random hyperparameter search with ``GroupKFold`` so
    queries cannot leak across CV folds.
  * For classification, picks an operating threshold on the validation
    split with a configurable criterion (F1, Youden's J, asymmetric
    FP/FN cost, recall floor, precision floor) — never on test.
  * Reports baseline-at-0.5 vs refined-at-threshold metrics on test so
    the gain (or absence of it) is explicit.

Outputs (per refined model dir):
  * ``metrics.json``            — full report (search, threshold, val/test)
  * ``model_bundle.joblib``     — refined estimator + vectorizer + threshold
  * ``search_results.csv``      — RandomizedSearchCV cv_results_ flattened
  * ``threshold_sweep.csv``     — only for classification
  * ``predictions_test.csv``    — y_true, y_prob, y_pred at chosen threshold
  * ``refine_config.json``      — search space + criterion snapshot

A combined ``refined_<predictor_id>_comparison.csv`` is also written next
to the output root so the post-refinement decision is trivially auditable.
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from distill.dataset_utils import read_jsonl
from predictors.training.common import (
    _resolve_feature_columns,
    assign_splits,
    build_vectorizer,
    infer_feature_types,
    predict_probability,
    to_feature_matrix,
    write_predictions_csv,
)
from predictors.training.metrics import (
    classification_metrics,
    grouped_metrics_classification,
    grouped_metrics_regression,
    regression_metrics,
)


# ----- Default search spaces (modest, no exotic grids) -------------------
DEFAULT_SEARCH_SPACES: Dict[str, Dict[str, Dict[str, List[Any]]]] = {
    "classification": {
        "random_forest": {
            "n_estimators": [200, 400, 800],
            "max_depth": [None, 10, 20, 40],
            "min_samples_leaf": [1, 2, 5, 10],
            "max_features": ["sqrt", "log2", 0.5],
            "class_weight": ["balanced", "balanced_subsample"],
        },
        "gradient_boosting": {
            "learning_rate": [0.03, 0.05, 0.1],
            "max_iter": [200, 400, 800],
            "max_depth": [None, 6, 8, 12],
            "max_leaf_nodes": [15, 31, 63],
            "l2_regularization": [0.0, 1e-4, 1e-2],
        },
    },
    "regression": {
        "random_forest": {
            "n_estimators": [200, 500, 1000],
            "max_depth": [None, 10, 20, 40],
            "min_samples_leaf": [1, 2, 5],
            "max_features": ["sqrt", 0.5, 0.8, 1.0],
        },
        "gradient_boosting": {
            "learning_rate": [0.03, 0.05, 0.1],
            "max_iter": [200, 500, 1000],
            "max_depth": [None, 6, 8, 12],
            "max_leaf_nodes": [15, 31, 63],
            "l2_regularization": [0.0, 1e-4, 1e-2],
        },
    },
}

DEFAULT_SCORING: Dict[str, str] = {
    "classification": "roc_auc",
    "regression": "neg_mean_absolute_error",
}


def _empty_estimator(*, task: str, family: str, seed: int) -> Any:
    """Fresh estimator stub used as the RandomizedSearchCV base.

    Kept here (instead of reusing ``build_estimator`` from ``common``)
    so the refined run has explicit, audit-friendly defaults for n_jobs
    and random_state.
    """
    if task == "classification":
        if family == "random_forest":
            return RandomForestClassifier(random_state=seed, n_jobs=-1)
        if family == "gradient_boosting":
            return HistGradientBoostingClassifier(random_state=seed)
    else:
        if family == "random_forest":
            return RandomForestRegressor(random_state=seed, n_jobs=-1)
        if family == "gradient_boosting":
            return HistGradientBoostingRegressor(random_state=seed)
    raise ValueError(f"Refinement only supports RF/GB; got family={family} task={task}")


# ----- Threshold tuning --------------------------------------------------
def _binary_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    yt = y_true.astype(int)
    yp = y_pred.astype(int)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    return tp, fp, fn, tn


def _candidate_thresholds(y_prob: np.ndarray, grid: int = 199) -> np.ndarray:
    base = np.linspace(0.01, 0.99, grid)
    extras = np.unique(np.clip(y_prob, 0.01, 0.99))
    return np.unique(np.concatenate([base, extras]))


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, *, grid: int = 199) -> List[Dict[str, float]]:
    """Sweep thresholds and return per-threshold confusion + headline metrics."""
    out: List[Dict[str, float]] = []
    yt = y_true.astype(int)
    for t in _candidate_thresholds(y_prob, grid=grid):
        yp = (y_prob >= t).astype(int)
        tp, fp, fn, tn = _binary_confusion(yt, yp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        tpr = rec
        fpr = 1.0 - tnr
        acc = (tp + tn) / max(len(yt), 1)
        out.append(
            {
                "threshold": float(t),
                "tp": float(tp),
                "fp": float(fp),
                "fn": float(fn),
                "tn": float(tn),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "accuracy": float(acc),
                "tpr": float(tpr),
                "fpr": float(fpr),
                "youden_j": float(tpr - fpr),
            }
        )
    return out


def pick_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    criterion: str,
    fp_cost: float = 1.0,
    fn_cost: float = 1.0,
    min_recall: Optional[float] = None,
    min_precision: Optional[float] = None,
) -> Dict[str, Any]:
    """Choose an operating threshold on val data with an explicit criterion."""
    sweep = threshold_sweep(y_true, y_prob)
    if not sweep:
        raise ValueError("Empty sweep — y_prob/y_true incompatible.")

    crit = criterion.strip().lower()
    if crit == "f1":
        best = max(sweep, key=lambda r: r["f1"])
        rationale = "Maximise F1 over a 199-point threshold grid."
    elif crit in ("youden", "youdens_j"):
        best = max(sweep, key=lambda r: r["youden_j"])
        rationale = "Maximise Youden's J = TPR − FPR (balanced for ROC)."
    elif crit == "cost":
        def cost(r: Dict[str, float]) -> float:
            return r["fp"] * float(fp_cost) + r["fn"] * float(fn_cost)
        best = min(sweep, key=cost)
        rationale = (
            f"Minimise asymmetric cost: c_fp={fp_cost}·FP + c_fn={fn_cost}·FN."
        )
    elif crit == "min_recall":
        floor = float(min_recall if min_recall is not None else 0.8)
        eligible = [r for r in sweep if r["recall"] >= floor]
        if not eligible:
            # Fall back to the closest recall threshold so we never return nothing.
            best = max(sweep, key=lambda r: r["recall"])
            rationale = (
                f"No threshold met recall ≥ {floor}; fell back to maximum recall point."
            )
        else:
            best = max(eligible, key=lambda r: r["precision"])
            rationale = (
                f"Maximise precision subject to recall ≥ {floor} on validation."
            )
    elif crit == "min_precision":
        floor = float(min_precision if min_precision is not None else 0.8)
        eligible = [r for r in sweep if r["precision"] >= floor]
        if not eligible:
            best = max(sweep, key=lambda r: r["precision"])
            rationale = (
                f"No threshold met precision ≥ {floor}; fell back to maximum precision."
            )
        else:
            best = max(eligible, key=lambda r: r["recall"])
            rationale = (
                f"Maximise recall subject to precision ≥ {floor} on validation."
            )
    else:
        raise ValueError(
            f"Unsupported threshold criterion '{criterion}'. "
            "Use one of: f1, youden, cost, min_recall, min_precision."
        )

    return {
        "criterion": crit,
        "fp_cost": float(fp_cost) if crit == "cost" else None,
        "fn_cost": float(fn_cost) if crit == "cost" else None,
        "min_recall": float(min_recall) if (crit == "min_recall" and min_recall is not None) else None,
        "min_precision": float(min_precision) if (crit == "min_precision" and min_precision is not None) else None,
        "rationale": rationale,
        "chosen_threshold": float(best["threshold"]),
        "chosen_point": best,
        "sweep": sweep,
    }


# ----- Hyperparameter search --------------------------------------------
def _run_random_search(
    *,
    task: str,
    family: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: Sequence[str],
    search_space: Mapping[str, Sequence[Any]],
    n_iter: int,
    n_splits: int,
    scoring: str,
    seed: int,
) -> Tuple[Any, Dict[str, Any]]:
    base = _empty_estimator(task=task, family=family, seed=seed)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions={k: list(v) for k, v in search_space.items()},
        n_iter=int(n_iter),
        scoring=scoring,
        cv=GroupKFold(n_splits=int(n_splits)),
        refit=True,
        random_state=seed,
        return_train_score=False,
        n_jobs=-1,
        error_score="raise",
    )
    t0 = time.perf_counter()
    search.fit(X_train, y_train, groups=np.asarray(groups_train))
    elapsed = float(time.perf_counter() - t0)

    cv_results = search.cv_results_
    n = len(cv_results.get("mean_test_score", []))
    flat_results: List[Dict[str, Any]] = []
    for i in range(n):
        row: Dict[str, Any] = {
            "rank_test_score": int(cv_results["rank_test_score"][i]),
            "mean_test_score": float(cv_results["mean_test_score"][i]),
            "std_test_score": float(cv_results["std_test_score"][i]),
            "mean_fit_time_s": float(cv_results["mean_fit_time"][i]),
            "mean_score_time_s": float(cv_results["mean_score_time"][i]),
        }
        for key, vals in cv_results.items():
            if key.startswith("param_"):
                val = vals[i]
                # numpy.ma.masked appears for missing params: store as None.
                row[key] = None if val is None or repr(val) == "masked" else val
        flat_results.append(row)

    summary = {
        "scoring": scoring,
        "n_iter": int(n_iter),
        "n_splits": int(n_splits),
        "search_seconds": elapsed,
        "best_score": float(search.best_score_),
        "best_params": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in search.best_params_.items()},
        "cv_results": flat_results,
    }
    return search.best_estimator_, summary


# ----- Public entry points ---------------------------------------------
def refine_classifier(
    *,
    predictor_id: str,
    dataset_jsonl: Path,
    dataset_meta_json: Optional[Path],
    target_column: str,
    family: str,
    output_root: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    n_iter: int,
    n_splits: int,
    scoring: str,
    search_space: Mapping[str, Sequence[Any]],
    threshold_criterion: str,
    threshold_kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    rows = read_jsonl(dataset_jsonl)
    if not rows:
        raise ValueError(f"Dataset is empty: {dataset_jsonl}")
    feature_columns = _resolve_feature_columns(rows, target_column, dataset_meta_json)
    feature_types = infer_feature_types(rows, feature_columns)
    split_labels = assign_splits(
        rows,
        group_column="query_id",
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    train_rows = [r for r, s in zip(rows, split_labels) if s == "train"]
    val_rows = [r for r, s in zip(rows, split_labels) if s == "val"]
    test_rows = [r for r, s in zip(rows, split_labels) if s == "test"]
    if not train_rows or not val_rows or not test_rows:
        raise ValueError("Empty partition. Adjust split ratios or seed.")

    vectorizer = build_vectorizer(feature_types)
    X_train_raw = to_feature_matrix(train_rows, feature_columns, feature_types)
    X_train = vectorizer.fit_transform(X_train_raw)
    y_train = np.asarray(
        [int(r[target_column]) for r in train_rows], dtype=int
    )
    groups_train = [str(r.get("query_id", "")) for r in train_rows]

    best_estimator, search_summary = _run_random_search(
        task="classification",
        family=family,
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        search_space=search_space,
        n_iter=n_iter,
        n_splits=n_splits,
        scoring=scoring,
        seed=seed,
    )

    X_val = vectorizer.transform(to_feature_matrix(val_rows, feature_columns, feature_types))
    y_val = np.asarray([int(r[target_column]) for r in val_rows], dtype=int)
    y_val_prob = predict_probability(best_estimator, X_val)

    threshold_report = pick_threshold(
        y_true=y_val.astype(float),
        y_prob=y_val_prob,
        criterion=threshold_criterion,
        fp_cost=float(threshold_kwargs.get("fp_cost", 1.0)),
        fn_cost=float(threshold_kwargs.get("fn_cost", 1.0)),
        min_recall=threshold_kwargs.get("min_recall"),
        min_precision=threshold_kwargs.get("min_precision"),
    )
    chosen_threshold = float(threshold_report["chosen_threshold"])

    X_test = vectorizer.transform(to_feature_matrix(test_rows, feature_columns, feature_types))
    y_test = np.asarray([int(r[target_column]) for r in test_rows], dtype=int)

    t0 = time.perf_counter()
    y_test_prob = predict_probability(best_estimator, X_test)
    predict_time_test_s = float(time.perf_counter() - t0)
    y_test_pred = (y_test_prob >= chosen_threshold).astype(int)
    tp, fp, fn, tn = _binary_confusion(y_test, y_test_pred)

    test_global = classification_metrics(y_test, y_test_prob)
    val_global = classification_metrics(y_val, y_val_prob)
    test_at_threshold = {
        "threshold": chosen_threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": float((tp + tn) / max(len(y_test), 1)),
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "tnr": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
    }
    test_by_benchmark = grouped_metrics_classification(
        y_test.astype(float),
        y_test_prob,
        [str(r.get("benchmark", "")) for r in test_rows],
    )
    test_by_model = grouped_metrics_classification(
        y_test.astype(float),
        y_test_prob,
        [str(r.get("model_name", "")) for r in test_rows],
    )

    # ── Output paths ────────────────────────────────────────────
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    family_alias = {"random_forest": "rf", "gradient_boosting": "gb"}.get(family, family[:4])
    predictor_alias = {
        "quality_ex_ante": "qea",
        "quality_post_hoc": "qph",
        "service_cost": "sc",
    }.get(predictor_id, predictor_id.replace("_", "-")[:12])
    model_dir = output_root / f"refined-{predictor_alias}-{family_alias}-{stamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # search results CSV
    sr_path = model_dir / "search_results.csv"
    _write_dict_rows_csv(sr_path, search_summary["cv_results"])

    # threshold sweep CSV
    ts_path = model_dir / "threshold_sweep.csv"
    _write_dict_rows_csv(ts_path, threshold_report["sweep"])

    # predictions CSV (test, with chosen threshold)
    preds = []
    for r, yt, yp, yhat in zip(test_rows, y_test, y_test_prob, y_test_pred):
        preds.append(
            {
                "query_id": r.get("query_id"),
                "benchmark": r.get("benchmark"),
                "model_name": r.get("model_name"),
                "run_id": r.get("run_id"),
                "y_true": int(yt),
                "y_prob": float(yp),
                "y_pred": int(yhat),
                "threshold": chosen_threshold,
            }
        )
    write_predictions_csv(model_dir / "predictions_test.csv", preds)

    # metrics.json
    metrics_payload = {
        "predictor_id": predictor_id,
        "task": "classification",
        "model_family": family,
        "stage": "refined",
        "dataset": {
            "jsonl": str(dataset_jsonl),
            "meta_json": str(dataset_meta_json) if dataset_meta_json else None,
            "row_count": len(rows),
            "target_column": target_column,
            "feature_count": len(feature_columns),
        },
        "split": {
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "counts": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        },
        "search": search_summary,
        "threshold": threshold_report,
        "validation": {"global": val_global},
        "test": {
            "global": test_global,
            "at_threshold": test_at_threshold,
            "by_benchmark": test_by_benchmark,
            "by_model": test_by_model,
        },
        "timing": {
            "predict_time_test_s": predict_time_test_s,
            "predict_us_per_row": float(predict_time_test_s / max(len(test_rows), 1) * 1e6),
            "test_rows": len(test_rows),
        },
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    # refine_config.json (audit trail)
    (model_dir / "refine_config.json").write_text(
        json.dumps(
            {
                "predictor_id": predictor_id,
                "family": family,
                "search_space": {k: list(v) for k, v in search_space.items()},
                "n_iter": int(n_iter),
                "n_splits": int(n_splits),
                "scoring": scoring,
                "threshold_criterion": threshold_criterion,
                "threshold_kwargs": dict(threshold_kwargs),
                "seed": seed,
                "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # model bundle
    bundle = {
        "predictor_id": predictor_id,
        "task": "classification",
        "model_family": family,
        "stage": "refined",
        "estimator": best_estimator,
        "vectorizer": vectorizer,
        "feature_columns": feature_columns,
        "feature_types": feature_types,
        "target_column": target_column,
        "threshold": chosen_threshold,
        "threshold_criterion": threshold_criterion,
    }
    joblib.dump(bundle, model_dir / "model_bundle.joblib")

    return {
        "model_dir": str(model_dir),
        "metrics_json": str(model_dir / "metrics.json"),
        "best_params": search_summary["best_params"],
        "best_cv_score": search_summary["best_score"],
        "threshold": chosen_threshold,
        "threshold_criterion": threshold_criterion,
        "test_at_threshold": test_at_threshold,
        "test_global": test_global,
        "val_global": val_global,
        "timing": metrics_payload["timing"],
    }


def refine_regressor(
    *,
    predictor_id: str,
    dataset_jsonl: Path,
    dataset_meta_json: Optional[Path],
    target_column: str,
    family: str,
    output_root: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    n_iter: int,
    n_splits: int,
    scoring: str,
    search_space: Mapping[str, Sequence[Any]],
) -> Dict[str, Any]:
    rows = read_jsonl(dataset_jsonl)
    if not rows:
        raise ValueError(f"Dataset is empty: {dataset_jsonl}")
    feature_columns = _resolve_feature_columns(rows, target_column, dataset_meta_json)
    feature_types = infer_feature_types(rows, feature_columns)
    split_labels = assign_splits(
        rows,
        group_column="query_id",
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    train_rows = [r for r, s in zip(rows, split_labels) if s == "train"]
    val_rows = [r for r, s in zip(rows, split_labels) if s == "val"]
    test_rows = [r for r, s in zip(rows, split_labels) if s == "test"]
    if not train_rows or not val_rows or not test_rows:
        raise ValueError("Empty partition. Adjust split ratios or seed.")

    vectorizer = build_vectorizer(feature_types)
    X_train_raw = to_feature_matrix(train_rows, feature_columns, feature_types)
    X_train = vectorizer.fit_transform(X_train_raw)
    y_train = np.asarray([float(r[target_column]) for r in train_rows], dtype=float)
    groups_train = [str(r.get("query_id", "")) for r in train_rows]

    best_estimator, search_summary = _run_random_search(
        task="regression",
        family=family,
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        search_space=search_space,
        n_iter=n_iter,
        n_splits=n_splits,
        scoring=scoring,
        seed=seed,
    )

    X_val = vectorizer.transform(to_feature_matrix(val_rows, feature_columns, feature_types))
    y_val = np.asarray([float(r[target_column]) for r in val_rows], dtype=float)
    y_val_pred = np.asarray(best_estimator.predict(X_val), dtype=float)

    X_test = vectorizer.transform(to_feature_matrix(test_rows, feature_columns, feature_types))
    y_test = np.asarray([float(r[target_column]) for r in test_rows], dtype=float)

    t0 = time.perf_counter()
    y_test_pred = np.asarray(best_estimator.predict(X_test), dtype=float)
    predict_time_test_s = float(time.perf_counter() - t0)

    val_global = regression_metrics(y_val, y_val_pred)
    test_global = regression_metrics(y_test, y_test_pred)
    test_by_benchmark = grouped_metrics_regression(
        y_test,
        y_test_pred,
        [str(r.get("benchmark", "")) for r in test_rows],
    )
    test_by_model = grouped_metrics_regression(
        y_test,
        y_test_pred,
        [str(r.get("model_name", "")) for r in test_rows],
    )

    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    family_alias = {"random_forest": "rf", "gradient_boosting": "gb"}.get(family, family[:4])
    predictor_alias = {
        "service_cost": "sc",
        "quality_ex_ante": "qea",
        "quality_post_hoc": "qph",
    }.get(predictor_id, predictor_id.replace("_", "-")[:12])
    model_dir = output_root / f"refined-{predictor_alias}-{family_alias}-{stamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    sr_path = model_dir / "search_results.csv"
    _write_dict_rows_csv(sr_path, search_summary["cv_results"])

    preds = []
    for r, yt, yp in zip(test_rows, y_test, y_test_pred):
        preds.append(
            {
                "query_id": r.get("query_id"),
                "benchmark": r.get("benchmark"),
                "model_name": r.get("model_name"),
                "run_id": r.get("run_id"),
                "y_true": float(yt),
                "y_pred": float(yp),
                "abs_error": float(abs(yt - yp)),
            }
        )
    write_predictions_csv(model_dir / "predictions_test.csv", preds)

    metrics_payload = {
        "predictor_id": predictor_id,
        "task": "regression",
        "model_family": family,
        "stage": "refined",
        "dataset": {
            "jsonl": str(dataset_jsonl),
            "meta_json": str(dataset_meta_json) if dataset_meta_json else None,
            "row_count": len(rows),
            "target_column": target_column,
            "feature_count": len(feature_columns),
        },
        "split": {
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "counts": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        },
        "search": search_summary,
        "validation": {"global": val_global},
        "test": {
            "global": test_global,
            "by_benchmark": test_by_benchmark,
            "by_model": test_by_model,
        },
        "timing": {
            "predict_time_test_s": predict_time_test_s,
            "predict_us_per_row": float(predict_time_test_s / max(len(test_rows), 1) * 1e6),
            "test_rows": len(test_rows),
        },
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    (model_dir / "refine_config.json").write_text(
        json.dumps(
            {
                "predictor_id": predictor_id,
                "family": family,
                "search_space": {k: list(v) for k, v in search_space.items()},
                "n_iter": int(n_iter),
                "n_splits": int(n_splits),
                "scoring": scoring,
                "seed": seed,
                "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    bundle = {
        "predictor_id": predictor_id,
        "task": "regression",
        "model_family": family,
        "stage": "refined",
        "estimator": best_estimator,
        "vectorizer": vectorizer,
        "feature_columns": feature_columns,
        "feature_types": feature_types,
        "target_column": target_column,
    }
    joblib.dump(bundle, model_dir / "model_bundle.joblib")

    return {
        "model_dir": str(model_dir),
        "metrics_json": str(model_dir / "metrics.json"),
        "best_params": search_summary["best_params"],
        "best_cv_score": search_summary["best_score"],
        "val_global": val_global,
        "test_global": test_global,
        "timing": metrics_payload["timing"],
    }


# ----- Utilities --------------------------------------------------------
def _write_dict_rows_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
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


__all__ = [
    "DEFAULT_SEARCH_SPACES",
    "DEFAULT_SCORING",
    "pick_threshold",
    "threshold_sweep",
    "refine_classifier",
    "refine_regressor",
]
