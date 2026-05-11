from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_extraction import DictVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from distill.dataset_utils import read_jsonl
from predictors.dataset_common import META_COLUMNS
from predictors.training.metrics import (
    classification_metrics,
    grouped_metrics_classification,
    grouped_metrics_regression,
    regression_metrics,
)

SUPPORTED_MODEL_FAMILIES = frozenset(
    {
        # Linear baselines
        "linear",          # plain LinearRegression / LogisticRegression default (kept for back-compat)
        "logistic",        # LogisticRegression w/o explicit penalty arg (classification only)
        "logistic_l2",     # LogisticRegression with L2 penalty (classification only)
        "logistic_l1",     # LogisticRegression with L1 penalty (classification only)
        "ridge",           # Ridge regression (regression only)
        "lasso",           # Lasso regression (regression only)
        # Trees & ensembles
        "decision_tree",   # single CART tree baseline
        "random_forest",   # bagging
        "gradient_boosting",  # boosting (HistGradientBoosting*)
        # Neural baseline
        "mlp",
    }
)

# Default baseline batteries (kept simple, no tuning yet).
DEFAULT_CLASSIFICATION_BASELINES: Tuple[str, ...] = (
    "logistic",
    "logistic_l2",
    "logistic_l1",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "mlp",
)
DEFAULT_REGRESSION_BASELINES: Tuple[str, ...] = (
    "linear",
    "ridge",
    "lasso",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "mlp",
)


def resolve_model_families(pred_cfg: Dict[str, Any]) -> List[str]:
    """Return model families to train from YAML ``predictor`` block.

    If ``families`` is a non-empty list, each entry is trained in order
    (same dataset, same group-based split). Otherwise a single-element
    list containing ``family`` (default ``gradient_boosting``).
    """
    raw = pred_cfg.get("families")
    if isinstance(raw, list) and len(raw) > 0:
        out = [str(x).strip() for x in raw if str(x).strip()]
        if out:
            bad = sorted(set(out) - SUPPORTED_MODEL_FAMILIES)
            if bad:
                raise ValueError(
                    f"Unsupported predictor.families entries: {bad}. "
                    f"Supported: {sorted(SUPPORTED_MODEL_FAMILIES)}"
                )
            return out
    fam = str(pred_cfg.get("family", "gradient_boosting")).strip()
    if fam not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(
            f"Unsupported predictor.family '{fam}'. "
            f"Supported: {sorted(SUPPORTED_MODEL_FAMILIES)}"
        )
    return [fam]


def run_training(
    *,
    predictor_id: str,
    task: str,
    dataset_jsonl: Path,
    dataset_meta_json: Optional[Path],
    target_column: str,
    model_family: str,
    output_root: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, Any]:
    if task not in {"classification", "regression"}:
        raise ValueError(f"Unsupported task: {task}")
    if model_family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(
            f"Unsupported model_family '{model_family}'. "
            f"Supported: {sorted(SUPPORTED_MODEL_FAMILIES)}"
        )

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

    train_rows = [row for row, split in zip(rows, split_labels) if split == "train"]
    val_rows = [row for row, split in zip(rows, split_labels) if split == "val"]
    test_rows = [row for row, split in zip(rows, split_labels) if split == "test"]

    if not train_rows or not val_rows or not test_rows:
        raise ValueError("Split produced an empty partition. Adjust split ratios or seed.")

    vectorizer = build_vectorizer(feature_types)
    estimator = build_estimator(task=task, family=model_family, seed=seed)

    X_train = to_feature_matrix(train_rows, feature_columns, feature_types)
    y_train = np.asarray([float(row[target_column]) for row in train_rows], dtype=float)

    X_train_enc = vectorizer.fit_transform(X_train)
    y_train_fit = y_train if task == "regression" else y_train.astype(int)
    t_fit_start = time.perf_counter()
    estimator.fit(X_train_enc, y_train_fit)
    fit_time_s = float(time.perf_counter() - t_fit_start)

    val_eval = evaluate_partition(
        task=task,
        estimator=estimator,
        vectorizer=vectorizer,
        rows=val_rows,
        feature_columns=feature_columns,
        feature_types=feature_types,
        target_column=target_column,
    )
    test_eval = evaluate_partition(
        task=task,
        estimator=estimator,
        vectorizer=vectorizer,
        rows=test_rows,
        feature_columns=feature_columns,
        feature_types=feature_types,
        target_column=target_column,
    )

    # Standalone test-set predict timing (averages out metric/transform overhead
    # so per-row inference time is reported on the model itself).
    X_test_raw = to_feature_matrix(test_rows, feature_columns, feature_types)
    X_test_enc = vectorizer.transform(X_test_raw)
    t_pred_start = time.perf_counter()
    estimator.predict(X_test_enc)
    predict_time_test_s = float(time.perf_counter() - t_pred_start)
    n_test_rows = len(test_rows)
    predict_us_per_row = (
        float(predict_time_test_s / n_test_rows * 1e6) if n_test_rows > 0 else None
    )

    model_dir = make_model_dir(output_root=output_root, predictor_id=predictor_id, model_family=model_family)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Feature importance:
    # - Tree models like RandomForest expose `feature_importances_`.
    # - Linear models expose `coef_`.
    # - HistGradientBoosting* exposes neither, so we fall back to permutation importance on the
    #   validation split to keep a stable, model-agnostic importance report.
    X_val_raw = to_feature_matrix(val_rows, feature_columns, feature_types)
    X_val_enc = vectorizer.transform(X_val_raw)
    y_val = np.asarray([float(row[target_column]) for row in val_rows], dtype=float)
    feature_importance_rows = extract_feature_importance(
        estimator,
        vectorizer,
        X_ref=X_val_enc,
        y_ref=y_val if task == "regression" else y_val.astype(int),
        seed=seed,
    )

    metrics_payload = {
        "predictor_id": predictor_id,
        "task": task,
        "model_family": model_family,
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
            "counts": {
                "train": len(train_rows),
                "val": len(val_rows),
                "test": len(test_rows),
            },
        },
        "validation": val_eval["metrics"],
        "test": test_eval["metrics"],
        "timing": {
            "fit_time_s": fit_time_s,
            "predict_time_test_s": predict_time_test_s,
            "predict_us_per_row": predict_us_per_row,
            "test_rows": n_test_rows,
        },
    }

    metrics_path = model_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2)

    predictions_csv = model_dir / "predictions_test.csv"
    write_predictions_csv(predictions_csv, test_eval["prediction_rows"])

    feature_importance_csv = model_dir / "feature_importance.csv"
    write_feature_importance_csv(feature_importance_csv, feature_importance_rows)

    split_rows_out = []
    for row, split in zip(rows, split_labels):
        split_rows_out.append({
            "query_id": row.get("query_id"),
            "benchmark": row.get("benchmark"),
            "model_name": row.get("model_name"),
            "split": split,
        })
    split_csv = model_dir / "split_assignments.csv"
    write_predictions_csv(split_csv, split_rows_out)

    bundle = {
        "predictor_id": predictor_id,
        "task": task,
        "model_family": model_family,
        "estimator": estimator,
        "vectorizer": vectorizer,
        "feature_columns": feature_columns,
        "feature_types": feature_types,
        "target_column": target_column,
        "split_config": {
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
        },
    }
    bundle_path = model_dir / "model_bundle.joblib"
    joblib.dump(bundle, bundle_path)

    config_path = model_dir / "train_config.json"
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "predictor_id": predictor_id,
                "task": task,
                "dataset_jsonl": str(dataset_jsonl),
                "dataset_meta_json": str(dataset_meta_json) if dataset_meta_json else None,
                "target_column": target_column,
                "model_family": model_family,
                "seed": seed,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
            },
            fh,
            indent=2,
        )

    return {
        "model_dir": str(model_dir),
        "metrics_json": str(metrics_path),
        "predictions_test_csv": str(predictions_csv),
        "feature_importance_csv": str(feature_importance_csv),
        "split_assignments_csv": str(split_csv),
        "bundle": str(bundle_path),
        "config": str(config_path),
        "fit_time_s": fit_time_s,
        "predict_time_test_s": predict_time_test_s,
        "predict_us_per_row": predict_us_per_row,
        "test_rows": n_test_rows,
    }


def run_baseline_battery(
    *,
    predictor_id: str,
    task: str,
    dataset_jsonl: Path,
    dataset_meta_json: Optional[Path],
    target_column: str,
    families: Sequence[str],
    output_root: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    comparison_csv: Optional[Path] = None,
    logger: Any = None,
) -> Dict[str, Any]:
    """Train each family in ``families`` and aggregate into a single report.

    Per-family failures are caught and recorded so a single broken model
    does not abort the rest of the baseline comparison. When more than one
    family is requested (or ``comparison_csv`` is given) a tabular summary
    with one row per family is written to disk.
    """
    by_family: Dict[str, Any] = {}
    errors: Dict[str, Any] = {}
    rows_summary: List[Dict[str, Any]] = []
    for fam in families:
        if logger is not None:
            try:
                logger.info(
                    "Training baseline",
                    extra={"predictor_id": predictor_id, "family": fam},
                )
            except Exception:  # logging extras should never abort training
                pass
        try:
            report = run_training(
                predictor_id=predictor_id,
                task=task,
                dataset_jsonl=dataset_jsonl,
                dataset_meta_json=dataset_meta_json,
                target_column=target_column,
                model_family=fam,
                output_root=output_root,
                seed=seed,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
            )
            by_family[fam] = report
            rows_summary.append(
                _comparison_row(
                    predictor_id=predictor_id,
                    task=task,
                    family=fam,
                    report=report,
                    status="ok",
                    error=None,
                )
            )
        except Exception as exc:  # noqa: BLE001 — baseline isolation by design
            err = {"type": type(exc).__name__, "message": str(exc)}
            errors[fam] = err
            rows_summary.append(
                _comparison_row(
                    predictor_id=predictor_id,
                    task=task,
                    family=fam,
                    report=None,
                    status="failed",
                    error=err,
                )
            )
            if logger is not None:
                try:
                    logger.exception(
                        "Baseline family failed",
                        extra={"predictor_id": predictor_id, "family": fam},
                    )
                except Exception:
                    pass

    if comparison_csv is not None and rows_summary:
        write_comparison_table_csv(comparison_csv, rows_summary)

    return {
        "predictor_id": predictor_id,
        "task": task,
        "families": list(families),
        "by_family": by_family,
        "errors": errors,
        "comparison_csv": str(comparison_csv) if comparison_csv else None,
    }


def _comparison_row(
    *,
    predictor_id: str,
    task: str,
    family: str,
    report: Optional[Dict[str, Any]],
    status: str,
    error: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "predictor_id": predictor_id,
        "task": task,
        "model_family": family,
        "status": status,
        "error_type": (error or {}).get("type"),
        "error_message": (error or {}).get("message"),
        "fit_time_s": None,
        "predict_time_test_s": None,
        "predict_us_per_row": None,
        "test_rows": None,
        "model_dir": None,
        "metrics_json": None,
    }
    if report is None:
        return row
    row.update(
        {
            "fit_time_s": report.get("fit_time_s"),
            "predict_time_test_s": report.get("predict_time_test_s"),
            "predict_us_per_row": report.get("predict_us_per_row"),
            "test_rows": report.get("test_rows"),
            "model_dir": report.get("model_dir"),
            "metrics_json": report.get("metrics_json"),
        }
    )
    metrics_path = report.get("metrics_json")
    if not metrics_path:
        return row
    try:
        payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    except Exception:
        return row
    test_global = (payload.get("test") or {}).get("global") or {}
    val_global = (payload.get("validation") or {}).get("global") or {}
    if task == "classification":
        for key in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "average_precision",
            "brier",
            "log_loss",
            "ece_abs",
            "positive_rate",
        ):
            row[f"test_{key}"] = test_global.get(key)
            row[f"val_{key}"] = val_global.get(key)
    else:
        for key in ("mae", "rmse", "r2", "mape"):
            row[f"test_{key}"] = test_global.get(key)
            row[f"val_{key}"] = val_global.get(key)
    return row


def write_comparison_table_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write a baseline comparison table with one row per model family."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def evaluate_partition(
    *,
    task: str,
    estimator: Any,
    vectorizer: DictVectorizer,
    rows: Sequence[Dict[str, Any]],
    feature_columns: Sequence[str],
    feature_types: Dict[str, str],
    target_column: str,
) -> Dict[str, Any]:
    X_raw = to_feature_matrix(rows, feature_columns, feature_types)
    X = vectorizer.transform(X_raw)

    y_true = np.asarray([float(row[target_column]) for row in rows], dtype=float)

    if task == "classification":
        y_prob = predict_probability(estimator, X)
        y_pred = (y_prob >= 0.5).astype(int)

        metrics_global = classification_metrics(y_true, y_prob)
        metrics_by_benchmark = grouped_metrics_classification(
            y_true,
            y_prob,
            [str(row.get("benchmark", "")) for row in rows],
        )
        metrics_by_model = grouped_metrics_classification(
            y_true,
            y_prob,
            [str(row.get("model_name", "")) for row in rows],
        )
        metrics_by_benchmark_model = grouped_metrics_classification(
            y_true,
            y_prob,
            [f"{row.get('benchmark', '')}::{row.get('model_name', '')}" for row in rows],
        )

        prediction_rows = []
        for row, yt, yp, yhat in zip(rows, y_true, y_prob, y_pred):
            out = _meta_payload(row)
            out.update(
                {
                    "y_true": float(yt),
                    "y_prob": float(yp),
                    "y_pred": int(yhat),
                }
            )
            prediction_rows.append(out)

        return {
            "metrics": {
                "global": metrics_global,
                "by_benchmark": metrics_by_benchmark,
                "by_model": metrics_by_model,
                "by_benchmark_model": metrics_by_benchmark_model,
            },
            "prediction_rows": prediction_rows,
        }

    y_pred = np.asarray(estimator.predict(X), dtype=float)
    metrics_global = regression_metrics(y_true, y_pred)
    metrics_by_benchmark = grouped_metrics_regression(
        y_true,
        y_pred,
        [str(row.get("benchmark", "")) for row in rows],
    )
    metrics_by_model = grouped_metrics_regression(
        y_true,
        y_pred,
        [str(row.get("model_name", "")) for row in rows],
    )
    metrics_by_benchmark_model = grouped_metrics_regression(
        y_true,
        y_pred,
        [f"{row.get('benchmark', '')}::{row.get('model_name', '')}" for row in rows],
    )

    prediction_rows = []
    for row, yt, yp in zip(rows, y_true, y_pred):
        out = _meta_payload(row)
        out.update(
            {
                "y_true": float(yt),
                "y_pred": float(yp),
                "abs_error": float(abs(yt - yp)),
            }
        )
        prediction_rows.append(out)

    return {
        "metrics": {
            "global": metrics_global,
            "by_benchmark": metrics_by_benchmark,
            "by_model": metrics_by_model,
            "by_benchmark_model": metrics_by_benchmark_model,
        },
        "prediction_rows": prediction_rows,
    }


def infer_feature_types(rows: Sequence[Dict[str, Any]], feature_columns: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for col in feature_columns:
        values = [row.get(col) for row in rows]
        out[col] = "numeric" if _mostly_numeric(values) else "categorical"
    return out


def assign_splits(
    rows: Sequence[Dict[str, Any]],
    *,
    group_column: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> List[str]:
    if train_ratio <= 0 or val_ratio <= 0 or (train_ratio + val_ratio) >= 1.0:
        raise ValueError("Invalid split ratios. Must satisfy train>0, val>0 and train+val<1")

    out: List[str] = []
    for row in rows:
        group_value = str(row.get(group_column, ""))
        u = stable_fraction(group_value, seed=seed)
        if u < train_ratio:
            out.append("train")
        elif u < train_ratio + val_ratio:
            out.append("val")
        else:
            out.append("test")
    return out


def stable_fraction(value: str, *, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{value}".encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(16 ** len(digest))


def build_vectorizer(feature_types: Dict[str, str]) -> DictVectorizer:
    if not feature_types:
        raise ValueError("Feature type map is empty")
    return DictVectorizer(sparse=False)


def build_estimator(*, task: str, family: str, seed: int) -> Any:
    """Construct a fresh estimator for a given baseline family.

    Hyperparameters are deliberately fixed and modest: this is a baseline
    comparison phase, not a tuned final model. Anything heavier would need
    a separate grid/Bayesian search step elsewhere.
    """
    if task == "classification":
        if family in ("linear", "logistic", "logistic_l2"):
            # Default logistic regression (L2-regularised). ``linear`` kept as
            # back-compat alias for the previous classification baseline.
            return LogisticRegression(
                penalty="l2",
                C=1.0,
                max_iter=2000,
                class_weight="balanced",
                random_state=seed,
                solver="lbfgs",
                n_jobs=-1,
            )
        if family == "logistic_l1":
            # L1 logistic regression — liblinear is the canonical small-data solver
            # for L1 and supports class_weight="balanced".
            return LogisticRegression(
                penalty="l1",
                C=1.0,
                max_iter=2000,
                class_weight="balanced",
                random_state=seed,
                solver="liblinear",
            )
        if family == "decision_tree":
            return DecisionTreeClassifier(
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=seed,
            )
        if family == "random_forest":
            return RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )
        if family == "gradient_boosting":
            return HistGradientBoostingClassifier(
                max_depth=8,
                learning_rate=0.05,
                max_iter=400,
                random_state=seed,
            )
        if family == "mlp":
            # Small feed-forward net + scaled dense inputs (DictVectorizer is dense).
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "mlp",
                        MLPClassifier(
                            hidden_layer_sizes=(128, 64),
                            activation="relu",
                            alpha=1e-4,
                            max_iter=500,
                            early_stopping=True,
                            validation_fraction=0.1,
                            n_iter_no_change=25,
                            random_state=seed,
                        ),
                    ),
                ]
            )
    else:
        if family == "linear":
            # Plain (unregularised) least squares — baseline floor.
            return LinearRegression(n_jobs=-1)
        if family == "ridge":
            return Ridge(alpha=1.0, random_state=seed)
        if family == "lasso":
            return Lasso(alpha=1e-3, random_state=seed, max_iter=10000)
        if family == "decision_tree":
            return DecisionTreeRegressor(
                max_depth=12,
                min_samples_leaf=5,
                random_state=seed,
            )
        if family == "random_forest":
            return RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=-1,
            )
        if family == "gradient_boosting":
            return HistGradientBoostingRegressor(
                max_depth=8,
                learning_rate=0.05,
                max_iter=500,
                random_state=seed,
            )
        if family == "mlp":
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "mlp",
                        MLPRegressor(
                            hidden_layer_sizes=(128, 64),
                            activation="relu",
                            alpha=1e-4,
                            max_iter=500,
                            early_stopping=True,
                            validation_fraction=0.1,
                            n_iter_no_change=25,
                            random_state=seed,
                        ),
                    ),
                ]
            )

    raise ValueError(f"Unsupported estimator family '{family}' for task '{task}'")


def extract_feature_importance(
    estimator: Any,
    vectorizer: DictVectorizer,
    *,
    X_ref: Optional[np.ndarray] = None,
    y_ref: Optional[np.ndarray] = None,
    seed: int = 42,
) -> List[Dict[str, float]]:
    names = vectorizer_feature_names(vectorizer)
    values: Optional[np.ndarray] = None

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        if coef.ndim == 1:
            values = np.abs(coef)
        else:
            values = np.mean(np.abs(coef), axis=0)

    if values is None:
        # Model-agnostic fallback (e.g., HistGradientBoosting*).
        # Use permutation importance if reference data is available.
        if X_ref is not None and y_ref is not None and len(names) > 0:
            try:
                result = permutation_importance(
                    estimator,
                    X_ref,
                    y_ref,
                    n_repeats=10,
                    random_state=seed,
                    n_jobs=-1,
                )
                values = np.asarray(result.importances_mean, dtype=float)
            except Exception:
                return []

    if values is None or len(values) != len(names):
        return []

    rows = [
        {
            "feature": str(name),
            "importance": float(val),
        }
        for name, val in zip(names, values)
    ]
    rows.sort(key=lambda r: r["importance"], reverse=True)
    return rows


def vectorizer_feature_names(vectorizer: DictVectorizer) -> List[str]:
    try:
        return [str(name) for name in vectorizer.get_feature_names_out()]
    except Exception:
        return []


def predict_probability(estimator: Any, X: Any) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(X), dtype=float)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.reshape(-1)

    if hasattr(estimator, "decision_function"):
        raw = np.asarray(estimator.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-raw))

    preds = np.asarray(estimator.predict(X), dtype=float)
    return np.clip(preds, 0.0, 1.0)


def make_model_dir(*, output_root: Path, predictor_id: str, model_family: str) -> Path:
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    predictor_aliases = {
        "quality_ex_ante": "qea",
        "quality_post_hoc": "qph",
        "service_cost": "sc",
    }
    family_aliases = {
        "linear": "lin",
        "logistic": "log",
        "logistic_l2": "logl2",
        "logistic_l1": "logl1",
        "ridge": "ridge",
        "lasso": "lasso",
        "decision_tree": "dt",
        "random_forest": "rf",
        "gradient_boosting": "gb",
        "mlp": "mlp",
    }
    safe_predictor = predictor_aliases.get(predictor_id, predictor_id.replace("_", "-")[:12])
    safe_family = family_aliases.get(model_family, model_family.replace("_", "-")[:12])
    return output_root / f"{safe_predictor}-{safe_family}-{stamp}"


def to_feature_matrix(
    rows: Sequence[Dict[str, Any]],
    feature_columns: Sequence[str],
    feature_types: Dict[str, str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        item: Dict[str, Any] = {}
        for col in feature_columns:
            value = row.get(col)
            if feature_types.get(col) == "numeric":
                item[col] = _numeric_or_zero(value)
            else:
                item[col] = "" if value is None else str(value)
        out.append(item)
    return out


def write_predictions_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    all_columns: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            all_columns.append(key)

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_feature_importance_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    write_predictions_csv(path, rows)


def _resolve_feature_columns(
    rows: Sequence[Dict[str, Any]],
    target_column: str,
    meta_json: Optional[Path],
) -> List[str]:
    if meta_json and meta_json.exists():
        with meta_json.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        feature_columns = payload.get("feature_columns")
        if isinstance(feature_columns, list) and feature_columns:
            return [str(x) for x in feature_columns]

    excluded = set(META_COLUMNS + [target_column])
    cols: List[str] = []
    seen = set(excluded)
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            cols.append(key)
    return cols


def _mostly_numeric(values: Sequence[Any]) -> bool:
    total = 0
    ok = 0
    for value in values:
        if value is None or value == "":
            continue
        total += 1
        try:
            float(value)
            ok += 1
        except (TypeError, ValueError):
            pass
    if total == 0:
        return True
    return (ok / total) >= 0.95


def _meta_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "query_id": row.get("query_id"),
        "benchmark": row.get("benchmark"),
        "model_name": row.get("model_name"),
        "run_id": row.get("run_id"),
    }


def _numeric_or_zero(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
