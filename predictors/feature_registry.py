from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class FeatureRule:
    name: str
    source: str
    availability: str
    leakage_risk: str
    recommendation: str
    allow_quality_ex_ante: bool
    allow_quality_post_hoc: bool
    allow_cost_strict_ex_ante: bool
    allow_cost_extended_operational: bool


CORE_RULES: List[FeatureRule] = [
    FeatureRule("prompt_char_len", "prompt_text", "ex_ante", "low", "keep", True, True, True, True),
    FeatureRule("prompt_word_count", "prompt_text", "ex_ante", "low", "keep", True, True, True, True),
    FeatureRule("prompt_unique_word_ratio", "prompt_text", "ex_ante", "low", "keep", True, True, True, True),
    FeatureRule("prompt_avg_word_len", "prompt_text", "ex_ante", "low", "keep", True, True, True, True),
    FeatureRule("prompt_numeric_token_count", "prompt_text", "ex_ante", "low", "keep", True, True, True, True),
    FeatureRule("prompt_symbolic_token_count", "prompt_text", "ex_ante", "low", "keep", True, True, True, True),
    FeatureRule("prompt_has_equation_marker", "prompt_text", "ex_ante", "low", "keep", True, True, True, True),
    FeatureRule("prompt_has_multiple_choice_marker", "prompt_text", "ex_ante", "low", "keep", True, True, True, True),

    FeatureRule("system_queue_depth", "system_state", "ex_ante", "low", "keep", True, True, True, True),
    FeatureRule("system_pending_requests", "system_state", "ex_ante", "low", "keep", True, True, True, True),
    FeatureRule("system_throughput_rps_recent", "system_state", "ex_ante", "low", "keep", True, True, True, True),
    FeatureRule("system_active_workers", "system_state", "ex_ante", "low", "keep", True, True, True, True),

    FeatureRule("input_tokens", "tokenizer/runtime", "ex_ante", "low", "keep", True, True, True, True),
    FeatureRule("model_tier", "model_registry", "ex_ante", "low", "keep", True, True, True, True),
    FeatureRule("request_hour_utc", "timestamp_utc", "ex_ante", "low", "keep", True, True, True, True),
    FeatureRule("request_weekday_utc", "timestamp_utc", "ex_ante", "low", "keep", True, True, True, True),

    FeatureRule("request_max_tokens", "request_config", "ex_ante", "low", "keep", False, False, True, True),
    FeatureRule("request_temperature", "request_config", "ex_ante", "low", "keep", False, False, True, True),

    FeatureRule("response_numeric_token_count", "response_text", "post_hoc", "medium", "keep", False, True, False, False),
    FeatureRule("response_boxed_count", "response_text", "post_hoc", "medium", "keep", False, True, False, False),
    FeatureRule("response_final_marker_count", "response_text", "post_hoc", "medium", "keep", False, True, False, False),
    FeatureRule("response_has_refusal_marker", "response_text", "post_hoc", "medium", "keep", False, True, False, False),
    FeatureRule("output_tokens", "runtime_output", "post_hoc", "medium", "keep", False, True, False, False),
    FeatureRule("output_length_chars", "response_text", "post_hoc", "medium", "keep", False, True, False, False),
    FeatureRule("avg_logprob", "uncertainty", "post_hoc", "medium", "keep", False, True, False, False),
    FeatureRule("logprob_std", "uncertainty", "post_hoc", "medium", "keep", False, True, False, False),
    FeatureRule("entropy_mean", "uncertainty", "post_hoc", "medium", "keep", False, True, False, False),
    FeatureRule("parseable_numeric", "response_text", "post_hoc", "medium", "keep", False, True, False, False),
    FeatureRule("parseable_boxed", "response_text", "post_hoc", "medium", "keep", False, True, False, False),
    FeatureRule("parseable_multiple_choice", "response_text", "post_hoc", "medium", "keep", False, True, False, False),
    FeatureRule("has_final_answer_marker", "response_text", "post_hoc", "medium", "keep", False, True, False, False),
    FeatureRule("latency_ms", "runtime_output", "post_hoc", "high", "keep", False, True, False, False),
    FeatureRule("ttft_ms", "runtime_output", "post_hoc", "high", "keep", False, True, False, False),
    FeatureRule("output_to_input_token_ratio", "derived_output", "post_hoc", "high", "keep", False, True, False, False),
    FeatureRule("output_to_input_token_ratio_missing", "derived_output", "post_hoc", "high", "keep", False, True, False, False),

    FeatureRule("resource_gpu_seconds", "runtime_output", "post_hoc", "critical", "drop", False, False, False, False),
    FeatureRule("resource_gpu_utilization_pct", "runtime_output", "post_hoc", "high", "drop", False, False, False, True),

    FeatureRule("telemetry_online_latency_mean_ms", "telemetry_online", "historical_ops", "high", "drop", False, False, False, False),
    FeatureRule("telemetry_online_ttfb_mean_ms", "telemetry_online", "historical_ops", "high", "drop", False, False, False, False),
    FeatureRule("telemetry_online_success_rate", "telemetry_online", "historical_ops", "medium", "replace", False, False, False, True),
    FeatureRule("telemetry_online_effective_throughput_tps_peak", "telemetry_online", "historical_ops", "medium", "replace", False, False, False, True),
    FeatureRule("telemetry_online_estimated_request_rps_peak", "telemetry_online", "historical_ops", "medium", "replace", False, False, False, True),

    FeatureRule("telemetry_throughput_request_rps", "telemetry_throughput", "historical_ops", "medium", "replace", False, False, False, True),
    FeatureRule("telemetry_throughput_output_tps", "telemetry_throughput", "historical_ops", "medium", "replace", False, False, False, True),
    FeatureRule("telemetry_throughput_mean_ttft_ms", "telemetry_throughput", "historical_ops", "high", "drop", False, False, False, False),
    FeatureRule("telemetry_throughput_mean_tpot_ms", "telemetry_throughput", "historical_ops", "high", "drop", False, False, False, False),
    FeatureRule("telemetry_throughput_mean_itl_ms", "telemetry_throughput", "historical_ops", "high", "drop", False, False, False, False),

    FeatureRule("telemetry_engine_running_mean", "telemetry_engine", "historical_ops", "medium", "replace", False, False, False, True),
    FeatureRule("telemetry_engine_waiting_mean", "telemetry_engine", "historical_ops", "medium", "replace", False, False, False, True),
    FeatureRule("telemetry_engine_prompt_throughput_mean", "telemetry_engine", "historical_ops", "medium", "replace", False, False, False, True),
    FeatureRule("telemetry_engine_generation_throughput_mean", "telemetry_engine", "historical_ops", "medium", "replace", False, False, False, True),
    FeatureRule("telemetry_engine_kv_cache_usage_pct_mean", "telemetry_engine", "historical_ops", "medium", "replace", False, False, False, True),
]


def all_rules() -> List[FeatureRule]:
    expanded: List[FeatureRule] = []
    for rule in CORE_RULES:
        expanded.append(rule)
        if _supports_missing_indicator(rule):
            expanded.append(
                FeatureRule(
                    name=f"{rule.name}_missing",
                    source=rule.source,
                    availability=rule.availability,
                    leakage_risk=rule.leakage_risk,
                    recommendation=rule.recommendation,
                    allow_quality_ex_ante=rule.allow_quality_ex_ante,
                    allow_quality_post_hoc=rule.allow_quality_post_hoc,
                    allow_cost_strict_ex_ante=rule.allow_cost_strict_ex_ante,
                    allow_cost_extended_operational=rule.allow_cost_extended_operational,
                )
            )
    return expanded


def rule_map() -> Dict[str, FeatureRule]:
    return {rule.name: rule for rule in all_rules()}


def allowed_for_predictor(predictor_id: str, *, cost_policy: str = "strict_ex_ante") -> List[str]:
    out: List[str] = []
    for rule in all_rules():
        if predictor_id == "quality_ex_ante" and rule.allow_quality_ex_ante:
            out.append(rule.name)
        elif predictor_id == "quality_post_hoc" and rule.allow_quality_post_hoc:
            out.append(rule.name)
        elif predictor_id == "service_cost":
            if cost_policy == "strict_ex_ante" and rule.allow_cost_strict_ex_ante:
                out.append(rule.name)
            elif cost_policy == "extended_operational" and rule.allow_cost_extended_operational:
                out.append(rule.name)
    return out


def _supports_missing_indicator(rule: FeatureRule) -> bool:
    return (
        "char_len" not in rule.name
        and "word_count" not in rule.name
        and "avg_word_len" not in rule.name
        and "ratio" not in rule.name
        and "marker" not in rule.name
        and not rule.name.endswith("_count")
        and not rule.name.startswith("parseable_")
        and rule.name not in {"request_hour_utc", "request_weekday_utc"}
    )
