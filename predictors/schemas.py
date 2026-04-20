from __future__ import annotations

from dataclasses import asdict, dataclass
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from distill.dataset_utils import read_jsonl


@dataclass
class SystemStateSnapshot:
    queue_depth: Optional[float] = None
    pending_requests: Optional[float] = None
    throughput_rps_recent: Optional[float] = None
    active_workers: Optional[float] = None


@dataclass
class ResourceSnapshot:
    gpu_seconds: Optional[float] = None
    energy_joules: Optional[float] = None
    gpu_utilization_pct: Optional[float] = None


@dataclass
class UncertaintySnapshot:
    avg_logprob: Optional[float] = None
    logprob_std: Optional[float] = None
    entropy_mean: Optional[float] = None


@dataclass
class ModelExecutionTrace:
    query_id: str
    benchmark: str
    model_name: str
    run_id: str
    timestamp_utc: Optional[str] = None

    source_file: Optional[str] = None
    source_record_index: Optional[int] = None
    example_id: Optional[str] = None
    model_tier: Optional[str] = None

    prompt_text: Optional[str] = None
    response_text: Optional[str] = None

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    latency_ms: Optional[float] = None
    ttft_ms: Optional[float] = None

    correct: Optional[bool] = None
    score: Optional[float] = None

    system_state: Optional[SystemStateSnapshot] = None
    resources: Optional[ResourceSnapshot] = None
    uncertainty: Optional[UncertaintySnapshot] = None

    tags: Optional[Dict[str, Any]] = None
    missing_signals: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.system_state is None:
            payload["system_state"] = None
        if self.resources is None:
            payload["resources"] = None
        if self.uncertainty is None:
            payload["uncertainty"] = None
        return payload

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "ModelExecutionTrace":
        system_state_raw = row.get("system_state")
        resources_raw = row.get("resources")
        uncertainty_raw = row.get("uncertainty")

        return cls(
            query_id=str(row.get("query_id", "")),
            benchmark=str(row.get("benchmark", "")),
            model_name=str(row.get("model_name", "")),
            run_id=str(row.get("run_id", "")),
            timestamp_utc=row.get("timestamp_utc"),
            source_file=row.get("source_file"),
            source_record_index=row.get("source_record_index"),
            example_id=row.get("example_id"),
            model_tier=row.get("model_tier"),
            prompt_text=row.get("prompt_text") if row.get("prompt_text") is not None else row.get("prompt"),
            response_text=row.get("response_text") if row.get("response_text") is not None else row.get("response"),
            input_tokens=_safe_int(row.get("input_tokens")),
            output_tokens=_safe_int(row.get("output_tokens")),
            latency_ms=_safe_float(row.get("latency_ms")),
            ttft_ms=_safe_float(row.get("ttft_ms")),
            correct=_safe_bool(row.get("correct") if row.get("correct") is not None else row.get("is_correct")),
            score=_safe_float(row.get("score")),
            system_state=_snapshot_from_dict(SystemStateSnapshot, system_state_raw),
            resources=_snapshot_from_dict(ResourceSnapshot, resources_raw),
            uncertainty=_snapshot_from_dict(UncertaintySnapshot, uncertainty_raw),
            tags=row.get("tags") if isinstance(row.get("tags"), dict) else None,
            missing_signals=list(row.get("missing_signals") or []),
        )


def load_trace_rows_from_patterns(patterns: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            rows.extend(read_jsonl(path))
    return rows


def load_traces_from_patterns(patterns: Sequence[str]) -> List[ModelExecutionTrace]:
    return [ModelExecutionTrace.from_dict(row) for row in load_trace_rows_from_patterns(patterns)]


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


def _snapshot_from_dict(snapshot_cls: Any, raw: Any) -> Optional[Any]:
    if not isinstance(raw, dict):
        return None

    allowed = set(snapshot_cls.__dataclass_fields__.keys())
    filtered = {k: raw.get(k) for k in allowed if k in raw}
    return snapshot_cls(**filtered)
