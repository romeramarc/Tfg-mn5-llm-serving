from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: Sequence[float], y_prob: Sequence[float]) -> Dict[str, Optional[float]]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_prob, dtype=float)

    yp = np.clip(yp, 1e-6, 1.0 - 1e-6)
    yhat = (yp >= 0.5).astype(int)
    yb = yt.astype(int)

    metrics: Dict[str, Optional[float]] = {
        "count": float(len(yb)),
        "positive_rate": float(np.mean(yb)) if len(yb) > 0 else None,
        "accuracy": float(accuracy_score(yb, yhat)) if len(yb) > 0 else None,
        "precision": float(precision_score(yb, yhat, zero_division=0)) if len(yb) > 0 else None,
        "recall": float(recall_score(yb, yhat, zero_division=0)) if len(yb) > 0 else None,
        "f1": float(f1_score(yb, yhat, zero_division=0)) if len(yb) > 0 else None,
        "roc_auc": None,
        "average_precision": None,
        "brier": None,
        "log_loss": None,
    }

    unique = np.unique(yb)
    if len(unique) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(yb, yp))
        metrics["average_precision"] = float(average_precision_score(yb, yp))
        metrics["brier"] = float(brier_score_loss(yb, yp))
        metrics["log_loss"] = float(log_loss(yb, np.column_stack([1.0 - yp, yp]), labels=[0, 1]))

    cal = calibration_report(yb, yp, n_bins=10)
    metrics["ece_abs"] = cal["ece_abs"]
    metrics["calibration_bins"] = cal["bins"]
    return metrics


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, Optional[float]]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    if len(yt) == 0:
        return {
            "count": 0.0,
            "mae": None,
            "rmse": None,
            "r2": None,
            "mape": None,
        }

    mae = float(mean_absolute_error(yt, yp))
    rmse = float(math.sqrt(mean_squared_error(yt, yp)))

    mape = None
    denom = np.abs(yt)
    mask = denom > 1e-9
    if np.any(mask):
        mape = float(np.mean(np.abs((yt[mask] - yp[mask]) / denom[mask])))

    r2 = None
    if len(yt) >= 2:
        r2 = float(r2_score(yt, yp))

    return {
        "count": float(len(yt)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }


def grouped_metrics_classification(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    groups: Sequence[str],
) -> Dict[str, Dict[str, Optional[float]]]:
    return _grouped_metrics(
        task="classification",
        y_true=y_true,
        y_pred=y_prob,
        groups=groups,
    )


def grouped_metrics_regression(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    groups: Sequence[str],
) -> Dict[str, Dict[str, Optional[float]]]:
    return _grouped_metrics(
        task="regression",
        y_true=y_true,
        y_pred=y_pred,
        groups=groups,
    )


def calibration_report(y_true: Sequence[int], y_prob: Sequence[float], *, n_bins: int = 10) -> Dict[str, object]:
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=float)
    yp = np.clip(yp, 1e-6, 1.0 - 1e-6)

    if len(yt) == 0:
        return {"ece_abs": None, "bins": []}

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bucket_idx = np.digitize(yp, bins, right=True)

    rows: List[Dict[str, float]] = []
    ece = 0.0
    total = float(len(yt))

    for i in range(1, n_bins + 1):
        mask = bucket_idx == i
        count = int(np.sum(mask))
        if count == 0:
            rows.append(
                {
                    "bin": float(i),
                    "low": float(bins[i - 1]),
                    "high": float(bins[i]),
                    "count": 0.0,
                    "mean_confidence": 0.0,
                    "accuracy": 0.0,
                }
            )
            continue

        conf = float(np.mean(yp[mask]))
        acc = float(np.mean(yt[mask]))
        gap = abs(acc - conf)
        ece += gap * (count / total)

        rows.append(
            {
                "bin": float(i),
                "low": float(bins[i - 1]),
                "high": float(bins[i]),
                "count": float(count),
                "mean_confidence": conf,
                "accuracy": acc,
            }
        )

    return {
        "ece_abs": float(ece),
        "bins": rows,
    }


def _grouped_metrics(
    *,
    task: str,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    groups: Sequence[str],
) -> Dict[str, Dict[str, Optional[float]]]:
    if len(y_true) != len(y_pred) or len(y_true) != len(groups):
        raise ValueError("y_true, y_pred and groups must have the same length")

    out: Dict[str, Dict[str, Optional[float]]] = {}
    unique_groups = sorted({str(g) for g in groups})
    for group in unique_groups:
        idx = [i for i, value in enumerate(groups) if str(value) == group]
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        if task == "classification":
            out[group] = classification_metrics(yt, yp)
        else:
            out[group] = regression_metrics(yt, yp)
    return out
