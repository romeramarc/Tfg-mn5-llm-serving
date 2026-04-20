from __future__ import annotations

from typing import Any, Dict, Optional

from predictors.schemas import (
    ModelExecutionTrace,
    ResourceSnapshot,
    SystemStateSnapshot,
    UncertaintySnapshot,
)


def legacy_quality_row_to_trace(
    row: Dict[str, Any],
    *,
    benchmark: str,
    model_name: str,
    run_id: str,
    source_file: str,
    source_record_index: int,
) -> ModelExecutionTrace:
    prompt_text = _pick_str(row, "prompt", "question", "input", "query")
    response_text = _pick_str(row, "response", "completion", "output", "prediction", "generated_text")

    trace = ModelExecutionTrace(
        query_id=_build_query_id(benchmark, row, source_record_index),
        benchmark=benchmark,
        model_name=model_name,
        run_id=run_id,
        timestamp_utc=_pick_str(row, "timestamp_utc", "timestamp"),
        source_file=source_file,
        source_record_index=source_record_index,
        prompt_text=prompt_text,
        response_text=response_text,
        input_tokens=_safe_int(_pick(row, "input_tokens", "prompt_tokens", "num_input_tokens")),
        output_tokens=_safe_int(_pick(row, "output_tokens", "completion_tokens", "num_output_tokens")),
        latency_ms=_safe_float(_pick(row, "latency_ms", "latency", "response_latency_ms")),
        ttft_ms=_safe_float(_pick(row, "ttft_ms", "ttft", "time_to_first_token_ms")),
        correct=_safe_bool(_pick(row, "correct", "is_correct")),
        score=_safe_float(_pick(row, "score", "accuracy", "normalized_score")),
        system_state=SystemStateSnapshot(
            queue_depth=_safe_float(_pick(row, "queue_depth", "system_queue_depth")),
            pending_requests=_safe_float(_pick(row, "pending_requests", "system_pending_requests")),
            throughput_rps_recent=_safe_float(_pick(row, "throughput_rps_recent", "system_throughput_rps_recent")),
            active_workers=_safe_float(_pick(row, "active_workers", "system_active_workers")),
        ),
        resources=ResourceSnapshot(
            gpu_seconds=_safe_float(_pick(row, "gpu_seconds", "resource_gpu_seconds")),
            energy_joules=_safe_float(_pick(row, "energy_joules", "resource_energy_joules")),
            gpu_utilization_pct=_safe_float(_pick(row, "gpu_utilization_pct", "resource_gpu_utilization_pct")),
        ),
        uncertainty=UncertaintySnapshot(
            avg_logprob=_safe_float(_pick(row, "avg_logprob", "mean_logprob")),
            logprob_std=_safe_float(_pick(row, "logprob_std", "std_logprob")),
            entropy_mean=_safe_float(_pick(row, "entropy_mean", "mean_entropy")),
        ),
        tags={"raw_quality_row": _strip_large_fields(row)},
        missing_signals=[],
    )

    # Drop empty nested payloads to keep trace rows compact.
    if _all_none(trace.system_state):
        trace.system_state = None
    if _all_none(trace.resources):
        trace.resources = None
    if _all_none(trace.uncertainty):
        trace.uncertainty = None

    return trace


def _build_query_id(benchmark: str, row: Dict[str, Any], fallback_index: int) -> str:
    idx = _pick(row, "index", "query_id", "id")
    if idx is None:
        idx = fallback_index
    return f"{benchmark}:{idx}"


def _strip_large_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    keys_to_skip = {
        "prompt",
        "question",
        "input",
        "response",
        "completion",
        "output",
        "generated_text",
    }
    cleaned: Dict[str, Any] = {}
    for key, value in row.items():
        if key in keys_to_skip:
            continue
        if isinstance(value, (dict, list)):
            continue
        cleaned[key] = value
    return cleaned


def _pick(row: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row.get(key)
    return None


def _pick_str(row: Dict[str, Any], *keys: str) -> Optional[str]:
    value = _pick(row, *keys)
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _all_none(snapshot: Any) -> bool:
    if snapshot is None:
        return True
    for value in snapshot.__dict__.values():
        if value is not None:
            return False
    return True
