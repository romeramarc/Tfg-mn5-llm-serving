from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from distill.dataset_utils import write_jsonl
from predictors.schemas import ModelExecutionTrace


META_COLUMNS = [
    "query_id",
    "benchmark",
    "model_name",
    "run_id",
    "timestamp_utc",
    "source_file",
    "source_record_index",
]

PROMPT_EQUATION_MARKERS = (
    "=",
    "\\frac",
    "\\sqrt",
    "^",
)

PROMPT_MULTIPLE_CHOICE_MARKERS = (
    "(a)",
    "(b)",
    "(c)",
    "(d)",
    "option",
    "choices",
)

RESPONSE_REFUSAL_MARKERS = (
    "i can't",
    "i cannot",
    "as an ai",
    "sorry",
)

RESPONSE_FINAL_MARKERS = (
    "final answer",
    "answer:",
)

TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
NUMBER_RE = re.compile(r"\d")
BOXED_RE = re.compile(r"\\boxed\s*\{")
FINAL_RE = re.compile(r"(?i)final\s+answer")
MULTI_CHOICE_RE = re.compile(r"\([A-Da-d]\)")


def metadata_from_trace(trace: ModelExecutionTrace) -> Dict[str, Any]:
    return {
        "query_id": trace.query_id,
        "benchmark": trace.benchmark,
        "model_name": trace.model_name,
        "run_id": trace.run_id,
        "timestamp_utc": trace.timestamp_utc,
        "source_file": trace.source_file,
        "source_record_index": trace.source_record_index,
    }


def request_context_from_trace(trace: ModelExecutionTrace) -> Dict[str, Optional[float]]:
    request_tags = (trace.tags or {}).get("request") or {}
    return {
        "request_max_tokens": _safe_float(request_tags.get("max_tokens")),
        "request_temperature": _safe_float(request_tags.get("temperature")),
    }


def prompt_feature_row(prompt_text: Optional[str]) -> Dict[str, float]:
    text = (prompt_text or "").strip()
    tokens = TOKEN_RE.findall(text)
    word_tokens = [tok for tok in tokens if tok.isalpha()]

    unique_ratio = 0.0
    avg_word_len = 0.0
    if word_tokens:
        unique_ratio = len(set(t.lower() for t in word_tokens)) / float(len(word_tokens))
        avg_word_len = sum(len(t) for t in word_tokens) / float(len(word_tokens))

    numeric_token_count = sum(1 for tok in tokens if NUMBER_RE.search(tok))
    symbolic_token_count = sum(1 for tok in tokens if not tok.isalnum())

    lowered = text.lower()
    has_equation_marker = float(any(marker in text for marker in PROMPT_EQUATION_MARKERS))
    has_multiple_choice_marker = float(any(marker in lowered for marker in PROMPT_MULTIPLE_CHOICE_MARKERS))

    return {
        "prompt_char_len": float(len(text)),
        "prompt_word_count": float(len(word_tokens)),
        "prompt_unique_word_ratio": float(unique_ratio),
        "prompt_avg_word_len": float(avg_word_len),
        "prompt_numeric_token_count": float(numeric_token_count),
        "prompt_symbolic_token_count": float(symbolic_token_count),
        "prompt_has_equation_marker": has_equation_marker,
        "prompt_has_multiple_choice_marker": has_multiple_choice_marker,
    }


def response_feature_row(response_text: Optional[str]) -> Dict[str, float]:
    text = (response_text or "").strip()
    tokens = TOKEN_RE.findall(text)
    numeric_token_count = sum(1 for tok in tokens if NUMBER_RE.search(tok))
    lowered = text.lower()

    return {
        "response_numeric_token_count": float(numeric_token_count),
        "response_boxed_count": float(len(BOXED_RE.findall(text))),
        "response_final_marker_count": float(sum(1 for marker in RESPONSE_FINAL_MARKERS if marker in lowered)),
        "response_has_refusal_marker": float(any(marker in lowered for marker in RESPONSE_REFUSAL_MARKERS)),
        "output_length_chars": float(len(text)),
        "parseable_numeric": float(bool(NUMBER_RE.search(text))),
        "parseable_boxed": float(bool(BOXED_RE.search(text))),
        "parseable_multiple_choice": float(bool(MULTI_CHOICE_RE.search(text))),
        "has_final_answer_marker": float(bool(FINAL_RE.search(text))),
    }


def time_feature_row(timestamp_utc: Optional[str]) -> Dict[str, float]:
    dt = _parse_timestamp(timestamp_utc)
    if dt is None:
        return {
            "request_hour_utc": 0.0,
            "request_weekday_utc": 0.0,
        }
    return {
        "request_hour_utc": float(dt.hour),
        "request_weekday_utc": float(dt.weekday()),
    }


def ex_ante_feature_row(trace: ModelExecutionTrace) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    row.update(prompt_feature_row(trace.prompt_text))

    queue_depth = _safe_float(trace.system_state.queue_depth if trace.system_state else None)
    pending = _safe_float(trace.system_state.pending_requests if trace.system_state else None)
    throughput_recent = _safe_float(trace.system_state.throughput_rps_recent if trace.system_state else None)
    active_workers = _safe_float(trace.system_state.active_workers if trace.system_state else None)

    row.update(
        {
            "system_queue_depth": _fill(queue_depth),
            "system_queue_depth_missing": _missing(queue_depth),
            "system_pending_requests": _fill(pending),
            "system_pending_requests_missing": _missing(pending),
            "system_throughput_rps_recent": _fill(throughput_recent),
            "system_throughput_rps_recent_missing": _missing(throughput_recent),
            "system_active_workers": _fill(active_workers),
            "system_active_workers_missing": _missing(active_workers),
            "input_tokens": _fill(_safe_float(trace.input_tokens)),
            "input_tokens_missing": _missing(_safe_float(trace.input_tokens)),
        }
    )

    row.update(time_feature_row(trace.timestamp_utc))

    # Model identity for training / inference alignment. ``model_name`` is kept in
    # META_COLUMNS and excluded from learnt features in ``infer_feature_columns``,
    # which made ex-ante and cost predictors blind to the candidate rung at routing
    # time unless we expose tier explicitly as a normal feature column.
    tier = (trace.model_tier or "").strip()
    row["model_tier"] = tier if tier else "unknown"

    return row


def post_hoc_feature_row(trace: ModelExecutionTrace) -> Dict[str, Any]:
    row = ex_ante_feature_row(trace)
    row.update(response_feature_row(trace.response_text))

    output_tokens = _safe_float(trace.output_tokens)
    output_chars = row.get("output_length_chars", 0.0)
    avg_logprob = _safe_float(trace.uncertainty.avg_logprob if trace.uncertainty else None)
    logprob_std = _safe_float(trace.uncertainty.logprob_std if trace.uncertainty else None)
    entropy_mean = _safe_float(trace.uncertainty.entropy_mean if trace.uncertainty else None)
    latency_ms = _safe_float(trace.latency_ms)
    ttft_ms = _safe_float(trace.ttft_ms)

    input_tokens = row.get("input_tokens", 0.0)
    ratio = None
    if output_tokens is not None and input_tokens and input_tokens > 0:
        ratio = output_tokens / input_tokens

    row.update(
        {
            "output_tokens": _fill(output_tokens),
            "output_tokens_missing": _missing(output_tokens),
            "output_length_chars": _fill(output_chars),
            "output_length_chars_missing": _missing(output_chars if output_chars > 0 else None),
            "avg_logprob": _fill(avg_logprob),
            "avg_logprob_missing": _missing(avg_logprob),
            "logprob_std": _fill(logprob_std),
            "logprob_std_missing": _missing(logprob_std),
            "entropy_mean": _fill(entropy_mean),
            "entropy_mean_missing": _missing(entropy_mean),
            "latency_ms": _fill(latency_ms),
            "latency_ms_missing": _missing(latency_ms),
            "ttft_ms": _fill(ttft_ms),
            "ttft_ms_missing": _missing(ttft_ms),
            "output_to_input_token_ratio": _fill(ratio),
            "output_to_input_token_ratio_missing": _missing(ratio),
        }
    )
    return row


def cost_feature_row(trace: ModelExecutionTrace, *, policy: str = "strict_ex_ante") -> Dict[str, Any]:
    row = ex_ante_feature_row(trace)

    request_ctx = request_context_from_trace(trace)
    max_tokens = request_ctx.get("request_max_tokens")
    temperature = request_ctx.get("request_temperature")

    row.update(
        {
            "request_max_tokens": _fill(max_tokens),
            "request_max_tokens_missing": _missing(max_tokens),
            "request_temperature": _fill(temperature),
            "request_temperature_missing": _missing(temperature),
        }
    )

    # Explicitly prevent direct target leakage in strict policy.
    if policy == "strict_ex_ante":
        return row

    telemetry = ((trace.tags or {}).get("telemetry") or {})
    online = telemetry.get("online") or {}
    throughput = telemetry.get("throughput") or {}
    engine = telemetry.get("engine") or {}

    feature_map = {
        "telemetry_online_success_rate": _safe_float(online.get("success_rate")),
        "telemetry_online_effective_throughput_tps_peak": _safe_float(online.get("effective_throughput_tps_peak")),
        "telemetry_online_estimated_request_rps_peak": _safe_float(online.get("estimated_request_throughput_rps_peak")),
        "telemetry_throughput_request_rps": _safe_float(throughput.get("request_throughput_rps")),
        "telemetry_throughput_output_tps": _safe_float(throughput.get("output_throughput_tps")),
        "telemetry_engine_running_mean": _safe_float(engine.get("running_mean")),
        "telemetry_engine_waiting_mean": _safe_float(engine.get("waiting_mean")),
        "telemetry_engine_prompt_throughput_mean": _safe_float(engine.get("prompt_throughput_mean")),
        "telemetry_engine_generation_throughput_mean": _safe_float(engine.get("generation_throughput_mean")),
        "telemetry_engine_kv_cache_usage_pct_mean": _safe_float(engine.get("kv_cache_usage_pct_mean")),
        "resource_gpu_utilization_pct": _safe_float(trace.resources.gpu_utilization_pct if trace.resources else None),
    }

    for feature_name, feature_value in feature_map.items():
        row[feature_name] = _fill(feature_value)
        row[f"{feature_name}_missing"] = _missing(feature_value)

    return row


def target_correct(trace: ModelExecutionTrace) -> Optional[float]:
    if trace.correct is None:
        return None
    return 1.0 if bool(trace.correct) else 0.0


def target_service_cost(
    trace: ModelExecutionTrace,
    *,
    cost_mode: str = "latency_ms",
    latency_weight: float = 1.0,
    gpu_seconds_weight: float = 0.0,
    energy_weight: float = 0.0,
) -> Optional[float]:
    latency_ms = _safe_float(trace.latency_ms)
    gpu_seconds = _safe_float(trace.resources.gpu_seconds if trace.resources else None)
    energy_joules = _safe_float(trace.resources.energy_joules if trace.resources else None)

    if cost_mode == "latency_ms":
        return latency_ms

    if cost_mode == "composite":
        if latency_ms is None and gpu_seconds is None and energy_joules is None:
            return None
        score = 0.0
        if latency_ms is not None:
            score += latency_weight * latency_ms
        if gpu_seconds is not None:
            score += gpu_seconds_weight * gpu_seconds
        if energy_joules is not None:
            score += energy_weight * energy_joules
        return float(score)

    raise ValueError(f"Unsupported cost_mode: {cost_mode}")


def write_dataset_artifacts(
    *,
    rows: Sequence[Dict[str, Any]],
    dataset_name: str,
    target_column: str,
    output_dir: Path,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / f"{dataset_name}.jsonl"
    csv_path = output_dir / f"{dataset_name}.csv"
    meta_path = output_dir / f"{dataset_name}_meta.json"

    write_jsonl(list(rows), jsonl_path)
    _write_csv(rows, csv_path)

    feature_columns = infer_feature_columns(rows, target_column=target_column)
    meta = {
        "dataset_name": dataset_name,
        "target_column": target_column,
        "row_count": len(rows),
        "metadata_columns": list(META_COLUMNS),
        "feature_columns": feature_columns,
        "benchmark_counts": _count_by(rows, "benchmark"),
        "model_counts": _count_by(rows, "model_name"),
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    return {
        "jsonl": jsonl_path,
        "csv": csv_path,
        "meta": meta_path,
    }


def infer_feature_columns(rows: Sequence[Dict[str, Any]], *, target_column: str) -> List[str]:
    cols: List[str] = []
    seen = set(META_COLUMNS + [target_column])
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            cols.append(key)
    return cols


def _write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    import csv

    all_cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            all_cols.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _count_by(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter[str(row.get(key, ""))] += 1
    return dict(counter)


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _fill(value: Optional[float]) -> float:
    return float(value) if value is not None else 0.0


def _missing(value: Optional[float]) -> float:
    return 0.0 if value is not None else 1.0
