"""Build traces and query offline predictors during online policy evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from predictors.dataset_common import cost_feature_row, ex_ante_feature_row, post_hoc_feature_row
from predictors.inference import OfflinePredictorSuite, PredictorBundle
from predictors.schemas import ModelExecutionTrace, SystemStateSnapshot, UncertaintySnapshot
from predictors.training.common import predict_probability


class EvalPredictorSuite:
  def __init__(
      self,
      *,
      ex_ante: PredictorBundle,
      post_hoc: PredictorBundle,
      cost: PredictorBundle,
      cost_policy: str = "strict_ex_ante",
      ex_ante_threshold: float = 0.632,
      post_hoc_threshold: float = 0.716,
  ):
      self.ex_ante = ex_ante
      self.post_hoc = post_hoc
      self.cost = cost
      self.cost_policy = cost_policy
      self.ex_ante_threshold = ex_ante_threshold
      self.post_hoc_threshold = post_hoc_threshold

  @classmethod
  def from_config(cls, pred_cfg: Dict[str, Any]) -> "EvalPredictorSuite":
      bundles = pred_cfg.get("bundles") or {}
      thresholds = pred_cfg.get("thresholds") or {}
      return cls(
          ex_ante=PredictorBundle(bundles["quality_ex_ante"]),
          post_hoc=PredictorBundle(bundles["quality_post_hoc"]),
          cost=PredictorBundle(bundles["service_cost"]),
          cost_policy=str(pred_cfg.get("cost_policy", "strict_ex_ante")),
          ex_ante_threshold=float(
              thresholds.get("quality_ex_ante", pred_cfg.get("ex_ante_threshold", 0.632))
          ),
          post_hoc_threshold=float(
              thresholds.get("quality_post_hoc", pred_cfg.get("post_hoc_threshold", 0.716))
          ),
      )

  def ex_ante_probability(self, trace: ModelExecutionTrace) -> float:
      row = ex_ante_feature_row(trace)
      x = [{col: row.get(col) for col in self.ex_ante.feature_columns}]
      xt = self.ex_ante.vectorizer.transform(x)
      return float(predict_probability(self.ex_ante.estimator, xt)[0])

  def post_hoc_probability(self, trace: ModelExecutionTrace) -> float:
      row = post_hoc_feature_row(trace)
      x = [{col: row.get(col) for col in self.post_hoc.feature_columns}]
      xt = self.post_hoc.vectorizer.transform(x)
      return float(predict_probability(self.post_hoc.estimator, xt)[0])

  def predicted_cost(self, trace: ModelExecutionTrace) -> float:
      row = cost_feature_row(trace, policy=self.cost_policy)
      x = [{col: row.get(col) for col in self.cost.feature_columns}]
      xt = self.cost.vectorizer.transform(x)
      return float(self.cost.estimator.predict(xt)[0])

  def post_hoc_accepts(self, trace: ModelExecutionTrace) -> bool:
      return self.post_hoc_probability(trace) >= self.post_hoc_threshold


def build_trace(
    *,
    prompt: str,
    benchmark: str,
    example_id: str,
    request_id: str,
    role: str,
    model_name: str,
    z_metrics: Dict[str, Any],
    inflight_at_send: Optional[float] = None,
    recent_p50_latency_ms: Optional[float] = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    response_text: Optional[str] = None,
    output_tokens: Optional[int] = None,
    latency_ms: Optional[float] = None,
    uncertainty: Optional[Dict[str, float]] = None,
) -> ModelExecutionTrace:
    now = datetime.now(tz=timezone.utc).isoformat()
    running = z_metrics.get("running")
    waiting = z_metrics.get("waiting")

    return ModelExecutionTrace(
        query_id=f"eval:{benchmark}:{example_id}:{role}",
        benchmark=benchmark,
        model_name=model_name,
        run_id="routing_eval_holdout",
        timestamp_utc=now,
        example_id=example_id,
        model_tier=role,
        prompt_text=prompt,
        response_text=response_text,
        input_tokens=len(prompt.split()),
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        system_state=SystemStateSnapshot(
            queue_depth=_f(waiting),
            pending_requests=_f(inflight_at_send),
            throughput_rps_recent=_f(recent_p50_latency_ms),
            active_workers=_f(running),
        ),
        uncertainty=UncertaintySnapshot(
            avg_logprob=_f((uncertainty or {}).get("avg_logprob")),
            logprob_std=_f((uncertainty or {}).get("logprob_std")),
            entropy_mean=_f((uncertainty or {}).get("entropy_mean")),
        ),
        tags={
            "request": {"max_tokens": max_tokens, "temperature": temperature},
            "request_id": request_id,
        },
    )


def _f(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
