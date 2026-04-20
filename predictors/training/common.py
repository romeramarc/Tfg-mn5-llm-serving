from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
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
from sklearn.linear_model import LogisticRegression, Ridge

from distill.dataset_utils import read_jsonl
from predictors.dataset_common import META_COLUMNS
from predictors.training.metrics import (
    classification_metrics,
    grouped_metrics_classification,
    grouped_metrics_regression,
    regression_metrics,
)


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
    estimator.fit(X_train_enc, y_train if task == "regression" else y_train.astype(int))

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

    model_dir = make_model_dir(output_root=output_root, predictor_id=predictor_id, model_family=model_family)
    model_dir.mkdir(parents=True, exist_ok=True)

    feature_importance_rows = extract_feature_importance(estimator, vectorizer)

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
    }


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
    if task == "classification":
        if family == "linear":
            return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
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
    else:
        if family == "linear":
            return Ridge(alpha=1.0, random_state=seed)
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

    raise ValueError(f"Unsupported estimator family '{family}' for task '{task}'")


def extract_feature_importance(estimator: Any, vectorizer: DictVectorizer) -> List[Dict[str, float]]:
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
        "random_forest": "rf",
        "gradient_boosting": "gb",
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
