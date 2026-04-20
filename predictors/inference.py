from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

from predictors.dataset_common import cost_feature_row, ex_ante_feature_row, post_hoc_feature_row
from predictors.schemas import ModelExecutionTrace
from predictors.training.common import predict_probability


class PredictorBundle:
    def __init__(self, bundle_path: str | Path):
        self.bundle_path = Path(bundle_path)
        self.payload = joblib.load(self.bundle_path)
        self.predictor_id = str(self.payload.get("predictor_id"))
        self.task = str(self.payload.get("task"))
        self.model_family = str(self.payload.get("model_family"))
        self.estimator = self.payload["estimator"]
        self.vectorizer = self.payload["vectorizer"]
        self.feature_columns = list(self.payload.get("feature_columns") or [])

    def predict(self, feature_row: Dict[str, Any]) -> Dict[str, float]:
        x = [{col: feature_row.get(col) for col in self.feature_columns}]
        xt = self.vectorizer.transform(x)

        if self.task == "classification":
            prob = float(predict_probability(self.estimator, xt)[0])
            label = 1 if prob >= 0.5 else 0
            return {
                "probability": prob,
                "label": float(label),
            }

        pred = float(np.asarray(self.estimator.predict(xt), dtype=float)[0])
        return {"value": pred}


class OfflinePredictorSuite:
    def __init__(
        self,
        *,
        ex_ante_bundle: str | Path,
        post_hoc_bundle: str | Path,
        cost_bundle: str | Path,
        cost_policy: str = "strict_ex_ante",
    ):
        self.ex_ante = PredictorBundle(ex_ante_bundle)
        self.post_hoc = PredictorBundle(post_hoc_bundle)
        self.cost = PredictorBundle(cost_bundle)
        self.cost_policy = cost_policy

    def predict_from_trace(self, trace: ModelExecutionTrace) -> Dict[str, Any]:
        ex_ante_row = ex_ante_feature_row(trace)
        post_hoc_row = post_hoc_feature_row(trace)
        cost_row = cost_feature_row(trace, policy=self.cost_policy)

        ex_ante_pred = self.ex_ante.predict(ex_ante_row)
        post_hoc_pred = self.post_hoc.predict(post_hoc_row)
        cost_pred = self.cost.predict(cost_row)

        return {
            "query_id": trace.query_id,
            "benchmark": trace.benchmark,
            "model_name": trace.model_name,
            "quality_ex_ante": ex_ante_pred,
            "quality_post_hoc": post_hoc_pred,
            "service_cost": cost_pred,
        }


def suite_from_selection_json(
    selection_json: str | Path,
    *,
    cost_policy: str = "strict_ex_ante",
) -> OfflinePredictorSuite:
    payload = json.loads(Path(selection_json).read_text(encoding="utf-8"))
    winners = payload.get("winners") or {}

    ex_ante = _require_bundle(winners, "quality_ex_ante")
    post_hoc = _require_bundle(winners, "quality_post_hoc")
    cost = _require_bundle(winners, "service_cost")

    return OfflinePredictorSuite(
        ex_ante_bundle=ex_ante,
        post_hoc_bundle=post_hoc,
        cost_bundle=cost,
        cost_policy=cost_policy,
    )


def _require_bundle(winners: Dict[str, Any], predictor_id: str) -> str:
    node = winners.get(predictor_id)
    if not isinstance(node, dict):
        raise KeyError(f"Missing winner for predictor '{predictor_id}'")
    bundle = node.get("bundle")
    if not bundle:
        raise KeyError(f"Missing bundle path for predictor '{predictor_id}'")
    return str(bundle)
