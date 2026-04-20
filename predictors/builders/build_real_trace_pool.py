"""
Build an auditable multi-model real trace pool for predictor training.

This script:
- discovers quality runs for student_small, student_mid, teacher,
- aligns examples by exact benchmark + index intersection across the 3 roles,
- enriches traces with nearest online/throughput telemetry,
- optionally enriches with vLLM engine snapshots from job logs,
- writes canonical trace JSONL + detailed ingest report.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import glob
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from distill.dataset_utils import read_jsonl
from predictors.schemas import ModelExecutionTrace
from predictors.trace_logging import legacy_quality_row_to_trace


ROLE_ORDER: Tuple[str, ...] = ("student_small", "student_mid", "teacher")
TIMESTAMP_SUFFIX_RE = re.compile(r"^(?P<prefix>[a-z_]+)-(?P<role>.+)-(?P<stamp>\d{8}T\d{6}Z)$")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
FLOAT_RE = r"[-+]?\d+(?:\.\d+)?"
ENGINE_LINE_RE = re.compile(
    r"Avg prompt throughput:\s*(?P<prompt>"
    + FLOAT_RE
    + r")\s*tokens/s,\s*Avg generation throughput:\s*(?P<generation>"
    + FLOAT_RE
    + r")\s*tokens/s,\s*Running:\s*(?P<running>\d+)\s*reqs,\s*Waiting:\s*(?P<waiting>\d+)\s*reqs,\s*GPU KV cache usage:\s*(?P<kv>"
    + FLOAT_RE
    + r")%"
)
BASELINE_RE = re.compile(r"Baseline:\s*(?P<role>[A-Za-z0-9_]+)\s*\((?P<model>[^)]+)\)")
MAX_TOKENS_RE = re.compile(r"^\s*max_tokens\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$")
TEMPERATURE_RE = re.compile(r"^\s*temperature\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$")


@dataclass
class RunInfo:
    role: str
    run_id: str
    timestamp: datetime
    timestamp_utc: str
    path: Path
    run_meta: Dict[str, Any]
    slurm_job_id: Optional[str]
    benchmark_files: Dict[str, Path] = field(default_factory=dict)
    request_max_tokens: Optional[float] = None
    request_temperature: Optional[float] = None
    summary_file: Optional[Path] = None


@dataclass
class TelemetryMatch:
    role: str
    quality_run_id: str
    quality_timestamp_utc: str
    online_run_id: Optional[str]
    online_delta_hours: Optional[float]
    throughput_run_id: Optional[str]
    throughput_delta_hours: Optional[float]
    log_file: Optional[str]
    log_from_job_id: Optional[str]


def _now_utc_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _parse_iso_utc(value: Optional[str]) -> Optional[datetime]:
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


def _parse_suffix_timestamp(value: str) -> Optional[datetime]:
    try:
        dt = datetime.strptime(value, "%Y%m%dT%H%M%SZ")
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _clean_line(text: str) -> str:
    return ANSI_RE.sub("", text)


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    idx = max(0, min(idx, len(ordered) - 1))
    return float(ordered[idx])


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _discover_quality_runs(quality_root: Path, roles: Sequence[str]) -> Dict[str, List[RunInfo]]:
    roles_set = set(roles)
    by_role: Dict[str, List[RunInfo]] = {role: [] for role in roles}

    for item in sorted(quality_root.glob("quality-*")):
        if not item.is_dir():
            continue
        m = TIMESTAMP_SUFFIX_RE.match(item.name)
        if not m or m.group("prefix") != "quality":
            continue

        role = m.group("role")
        if role not in roles_set:
            continue

        run_meta_path = item / "run_meta.json"
        run_meta: Dict[str, Any] = {}
        if run_meta_path.exists():
            run_meta = _load_json(run_meta_path)

        timestamp = _parse_iso_utc(run_meta.get("timestamp_utc"))
        if timestamp is None:
            timestamp = _parse_suffix_timestamp(m.group("stamp"))
        if timestamp is None:
            continue

        benchmark_files: Dict[str, Path] = {}
        for bench_dir in sorted(item.iterdir()):
            if not bench_dir.is_dir():
                continue
            result_files = sorted(bench_dir.glob("*_results.jsonl"))
            if not result_files:
                continue
            benchmark_files[bench_dir.name.lower()] = result_files[0]

        max_tokens, temperature = _extract_request_config(item / "eval.yaml")
        run = RunInfo(
            role=role,
            run_id=item.name,
            timestamp=timestamp,
            timestamp_utc=timestamp.isoformat(),
            path=item,
            run_meta=run_meta,
            slurm_job_id=str(run_meta.get("slurm_job_id")) if run_meta.get("slurm_job_id") else None,
            benchmark_files=benchmark_files,
            request_max_tokens=max_tokens,
            request_temperature=temperature,
        )
        by_role[role].append(run)

    for role in by_role:
        by_role[role].sort(key=lambda r: r.timestamp)

    return by_role


def _discover_perf_runs(
    root: Path,
    *,
    prefix: str,
    summary_filename: str,
    roles: Sequence[str],
) -> Dict[str, List[RunInfo]]:
    roles_set = set(roles)
    by_role: Dict[str, List[RunInfo]] = {role: [] for role in roles}

    for item in sorted(root.glob(f"{prefix}-*")):
        if not item.is_dir():
            continue
        m = TIMESTAMP_SUFFIX_RE.match(item.name)
        if not m or m.group("prefix") != prefix:
            continue

        role = m.group("role")
        if role not in roles_set:
            continue

        run_meta_path = item / "run_meta.json"
        summary_path = item / summary_filename
        if not run_meta_path.exists() or not summary_path.exists():
            continue

        run_meta = _load_json(run_meta_path)

        timestamp = _parse_iso_utc(run_meta.get("timestamp_utc"))
        if timestamp is None:
            timestamp = _parse_suffix_timestamp(m.group("stamp"))
        if timestamp is None:
            continue

        run = RunInfo(
            role=role,
            run_id=item.name,
            timestamp=timestamp,
            timestamp_utc=timestamp.isoformat(),
            path=item,
            run_meta=run_meta,
            slurm_job_id=str(run_meta.get("slurm_job_id")) if run_meta.get("slurm_job_id") else None,
            summary_file=summary_path,
        )
        by_role[role].append(run)

    for role in by_role:
        by_role[role].sort(key=lambda r: r.timestamp)

    return by_role


def _extract_request_config(eval_yaml_path: Path) -> Tuple[Optional[float], Optional[float]]:
    if not eval_yaml_path.exists():
        return None, None

    max_tokens: Optional[float] = None
    temperature: Optional[float] = None

    with eval_yaml_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if max_tokens is None:
                m_max = MAX_TOKENS_RE.match(line)
                if m_max:
                    max_tokens = _safe_float(m_max.group(1))
                    continue
            if temperature is None:
                m_temp = TEMPERATURE_RE.match(line)
                if m_temp:
                    temperature = _safe_float(m_temp.group(1))
                    continue

    return max_tokens, temperature


def _nearest_unused(
    target_ts: datetime,
    candidates: Sequence[RunInfo],
    used_ids: set[str],
    max_gap: timedelta,
) -> Optional[RunInfo]:
    best: Optional[RunInfo] = None
    best_gap: Optional[timedelta] = None
    for run in candidates:
        if run.run_id in used_ids:
            continue
        gap = abs(run.timestamp - target_ts)
        if gap > max_gap:
            continue
        if best is None or gap < best_gap:
            best = run
            best_gap = gap
    return best


def _build_triplets(
    quality_runs: Dict[str, List[RunInfo]],
    *,
    max_gap_hours: float,
) -> List[Dict[str, RunInfo]]:
    required = set(ROLE_ORDER)
    if set(quality_runs.keys()) != required:
        raise ValueError(f"quality_runs keys must match {sorted(required)}")

    teachers = quality_runs["teacher"]
    smalls = quality_runs["student_small"]
    mids = quality_runs["student_mid"]

    used_small: set[str] = set()
    used_mid: set[str] = set()
    max_gap = timedelta(hours=max_gap_hours)

    triplets: List[Dict[str, RunInfo]] = []
    for teacher in teachers:
        small = _nearest_unused(teacher.timestamp, smalls, used_small, max_gap)
        mid = _nearest_unused(teacher.timestamp, mids, used_mid, max_gap)
        if small is None or mid is None:
            continue
        used_small.add(small.run_id)
        used_mid.add(mid.run_id)
        triplets.append(
            {
                "teacher": teacher,
                "student_small": small,
                "student_mid": mid,
            }
        )

    triplets.sort(key=lambda t: t["teacher"].timestamp)
    return triplets


def _nearest_run(target_ts: datetime, candidates: Sequence[RunInfo], max_gap: timedelta) -> Tuple[Optional[RunInfo], Optional[float]]:
    best: Optional[RunInfo] = None
    best_gap: Optional[timedelta] = None
    for run in candidates:
        gap = abs(run.timestamp - target_ts)
        if gap > max_gap:
            continue
        if best is None or gap < best_gap:
            best = run
            best_gap = gap
    if best is None or best_gap is None:
        return None, None
    return best, float(best_gap.total_seconds() / 3600.0)


def _summarize_online(run: RunInfo) -> Optional[Dict[str, Any]]:
    if not run.summary_file:
        return None
    rows = _load_json(run.summary_file)
    if not isinstance(rows, list) or not rows:
        return None

    total_requests = 0.0
    successful_requests = 0.0
    failed_requests = 0.0
    latency_weighted_num = 0.0
    latency_weighted_den = 0.0
    ttfb_weighted_num = 0.0
    ttfb_weighted_den = 0.0

    peak_row: Optional[Dict[str, Any]] = None
    max_rate_row: Optional[Dict[str, Any]] = None

    for row in rows:
        total_req = _safe_float(row.get("total_requests")) or 0.0
        succ_req = _safe_float(row.get("successful_requests")) or 0.0
        fail_req = _safe_float(row.get("failed_requests")) or 0.0

        total_requests += total_req
        successful_requests += succ_req
        failed_requests += fail_req

        latency_mean = _safe_float(row.get("latency_mean_ms"))
        latency_count = _safe_float(row.get("latency_count"))
        if latency_mean is not None and latency_count is not None and latency_count > 0:
            latency_weighted_num += latency_mean * latency_count
            latency_weighted_den += latency_count

        ttfb_mean = _safe_float(row.get("ttfb_mean_ms"))
        ttfb_count = _safe_float(row.get("ttfb_count"))
        if ttfb_mean is not None and ttfb_count is not None and ttfb_count > 0:
            ttfb_weighted_num += ttfb_mean * ttfb_count
            ttfb_weighted_den += ttfb_count

        if peak_row is None:
            peak_row = row
        else:
            prev = _safe_float(peak_row.get("effective_throughput_tps")) or float("-inf")
            cur = _safe_float(row.get("effective_throughput_tps")) or float("-inf")
            if cur > prev:
                peak_row = row

        if max_rate_row is None:
            max_rate_row = row
        else:
            prev_rate = _safe_float(max_rate_row.get("request_rate")) or float("-inf")
            cur_rate = _safe_float(row.get("request_rate")) or float("-inf")
            if cur_rate > prev_rate:
                max_rate_row = row

    peak_effective_tps = _safe_float(peak_row.get("effective_throughput_tps")) if peak_row else None
    peak_total_out = _safe_float(peak_row.get("total_output_tokens")) if peak_row else None
    peak_success = _safe_float(peak_row.get("successful_requests")) if peak_row else None

    mean_output_tokens_peak = None
    est_request_rps_peak = None
    if peak_total_out is not None and peak_success and peak_success > 0:
        mean_output_tokens_peak = peak_total_out / peak_success
        if peak_effective_tps is not None and mean_output_tokens_peak > 0:
            est_request_rps_peak = peak_effective_tps / mean_output_tokens_peak

    return {
        "rate_points": len(rows),
        "request_rate_min": _safe_float(min((row.get("request_rate") for row in rows), default=None)),
        "request_rate_max": _safe_float(max((row.get("request_rate") for row in rows), default=None)),
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "success_rate": (successful_requests / total_requests) if total_requests > 0 else None,
        "latency_mean_ms_weighted": (latency_weighted_num / latency_weighted_den) if latency_weighted_den > 0 else None,
        "ttfb_mean_ms_weighted": (ttfb_weighted_num / ttfb_weighted_den) if ttfb_weighted_den > 0 else None,
        "latency_p95_ms_at_max_rate": _safe_float(max_rate_row.get("latency_p95_ms")) if max_rate_row else None,
        "latency_p99_ms_at_max_rate": _safe_float(max_rate_row.get("latency_p99_ms")) if max_rate_row else None,
        "ttfb_p95_ms_at_max_rate": _safe_float(max_rate_row.get("ttfb_p95_ms")) if max_rate_row else None,
        "effective_throughput_tps_peak": peak_effective_tps,
        "request_rate_at_peak_tps": _safe_float(peak_row.get("request_rate")) if peak_row else None,
        "estimated_request_throughput_rps_peak": est_request_rps_peak,
        "mean_output_tokens_at_peak": mean_output_tokens_peak,
    }


def _summarize_throughput(run: RunInfo) -> Optional[Dict[str, Any]]:
    if not run.summary_file:
        return None
    payload = _load_json(run.summary_file)
    if not isinstance(payload, dict):
        return None

    return {
        "completed_requests": _safe_float(payload.get("completed_requests")),
        "total_time_s": _safe_float(payload.get("total_time_s")),
        "request_throughput_rps": _safe_float(payload.get("request_throughput_rps")),
        "output_throughput_tps": _safe_float(payload.get("output_throughput_tps")),
        "mean_ttft_ms": _safe_float(payload.get("mean_ttft_ms")),
        "p99_ttft_ms": _safe_float(payload.get("p99_ttft_ms")),
        "mean_tpot_ms": _safe_float(payload.get("mean_tpot_ms")),
        "p99_tpot_ms": _safe_float(payload.get("p99_tpot_ms")),
        "mean_itl_ms": _safe_float(payload.get("mean_itl_ms")),
        "p99_itl_ms": _safe_float(payload.get("p99_itl_ms")),
        "wall_time_s": _safe_float(payload.get("wall_time_s")),
        "return_code": _safe_int(payload.get("return_code")),
    }


def _index_logs(logs_dir: Path) -> Dict[str, List[Path]]:
    by_job_id: Dict[str, List[Path]] = {}
    for path in sorted(logs_dir.glob("*.out")):
        m = re.search(r"-(\d+)\.out$", path.name)
        if not m:
            continue
        by_job_id.setdefault(m.group(1), []).append(path)
    return by_job_id


def _select_log(log_index: Dict[str, List[Path]], job_id: Optional[str], role: str) -> Optional[Path]:
    if not job_id:
        return None
    candidates = log_index.get(job_id, [])
    if not candidates:
        return None

    role_hyphen = role.replace("_", "-")
    role_hits = [p for p in candidates if role in p.name or role_hyphen in p.name]
    target = role_hits if role_hits else candidates
    return sorted(target)[-1]


def _extract_log_header_info(log_path: Path) -> Dict[str, Any]:
    role = None
    model = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh):
            cleaned = _clean_line(line)
            m = BASELINE_RE.search(cleaned)
            if m:
                role = m.group("role")
                model = m.group("model")
                break
            if i > 240:
                break
    return {
        "role_from_log": role,
        "model_from_log": model,
    }


def _summarize_engine_metrics(log_path: Path) -> Optional[Dict[str, Any]]:
    prompt_vals: List[float] = []
    generation_vals: List[float] = []
    running_vals: List[float] = []
    waiting_vals: List[float] = []
    kv_vals: List[float] = []

    with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            cleaned = _clean_line(line)
            m = ENGINE_LINE_RE.search(cleaned)
            if not m:
                continue
            prompt_vals.append(float(m.group("prompt")))
            generation_vals.append(float(m.group("generation")))
            running_vals.append(float(m.group("running")))
            waiting_vals.append(float(m.group("waiting")))
            kv_vals.append(float(m.group("kv")))

    if not prompt_vals:
        return None

    header = _extract_log_header_info(log_path)

    return {
        "samples": len(prompt_vals),
        "prompt_throughput_mean": _mean(prompt_vals),
        "prompt_throughput_p95": _quantile(prompt_vals, 0.95),
        "generation_throughput_mean": _mean(generation_vals),
        "generation_throughput_p95": _quantile(generation_vals, 0.95),
        "running_mean": _mean(running_vals),
        "running_p95": _quantile(running_vals, 0.95),
        "waiting_mean": _mean(waiting_vals),
        "waiting_p95": _quantile(waiting_vals, 0.95),
        "kv_cache_usage_pct_mean": _mean(kv_vals),
        "kv_cache_usage_pct_p95": _quantile(kv_vals, 0.95),
        "log_file": str(log_path),
        **header,
    }


def _make_telemetry_bundle(
    *,
    quality_run: RunInfo,
    role: str,
    online_runs: Dict[str, List[RunInfo]],
    throughput_runs: Dict[str, List[RunInfo]],
    max_gap_hours: float,
    log_index: Dict[str, List[Path]],
) -> Tuple[Dict[str, Any], TelemetryMatch]:
    max_gap = timedelta(hours=max_gap_hours)

    online_run, online_gap_h = _nearest_run(quality_run.timestamp, online_runs.get(role, []), max_gap)
    throughput_run, throughput_gap_h = _nearest_run(quality_run.timestamp, throughput_runs.get(role, []), max_gap)

    online_summary = _summarize_online(online_run) if online_run else None
    throughput_summary = _summarize_throughput(throughput_run) if throughput_run else None

    job_id = None
    if throughput_run and throughput_run.slurm_job_id:
        job_id = throughput_run.slurm_job_id
    elif online_run and online_run.slurm_job_id:
        job_id = online_run.slurm_job_id

    log_file = _select_log(log_index, job_id, role)
    engine_summary = _summarize_engine_metrics(log_file) if log_file else None

    telemetry = {
        "online": online_summary,
        "throughput": throughput_summary,
        "engine": engine_summary,
        "provenance": {
            "quality_run_id": quality_run.run_id,
            "online_run_id": online_run.run_id if online_run else None,
            "throughput_run_id": throughput_run.run_id if throughput_run else None,
            "online_summary_file": str(online_run.summary_file) if online_run and online_run.summary_file else None,
            "throughput_summary_file": str(throughput_run.summary_file) if throughput_run and throughput_run.summary_file else None,
            "log_file": str(log_file) if log_file else None,
            "slurm_job_id_for_log": job_id,
            "online_delta_hours": online_gap_h,
            "throughput_delta_hours": throughput_gap_h,
        },
    }

    match = TelemetryMatch(
        role=role,
        quality_run_id=quality_run.run_id,
        quality_timestamp_utc=quality_run.timestamp_utc,
        online_run_id=online_run.run_id if online_run else None,
        online_delta_hours=online_gap_h,
        throughput_run_id=throughput_run.run_id if throughput_run else None,
        throughput_delta_hours=throughput_gap_h,
        log_file=str(log_file) if log_file else None,
        log_from_job_id=job_id,
    )
    return telemetry, match


def _row_key(row: Dict[str, Any], fallback_index: int) -> str:
    value = row.get("index")
    if value is None:
        value = row.get("query_id")
    if value is None:
        value = fallback_index
    return str(value)


def _sorted_keys(keys: Iterable[str]) -> List[str]:
    def sort_key(v: str) -> Tuple[int, Any]:
        if re.fullmatch(r"\d+", v):
            return (0, int(v))
        return (1, v)

    return sorted(keys, key=sort_key)


def _collect_missing_signals(trace: ModelExecutionTrace) -> List[str]:
    missing: List[str] = []

    if trace.input_tokens is None:
        missing.append("input_tokens")
    if trace.output_tokens is None:
        missing.append("output_tokens")
    if trace.ttft_ms is None:
        missing.append("ttft_ms")

    if trace.system_state is None:
        missing.extend(
            [
                "system_state.queue_depth",
                "system_state.pending_requests",
                "system_state.throughput_rps_recent",
                "system_state.active_workers",
            ]
        )
    else:
        if trace.system_state.queue_depth is None:
            missing.append("system_state.queue_depth")
        if trace.system_state.pending_requests is None:
            missing.append("system_state.pending_requests")
        if trace.system_state.throughput_rps_recent is None:
            missing.append("system_state.throughput_rps_recent")
        if trace.system_state.active_workers is None:
            missing.append("system_state.active_workers")

    if trace.resources is None:
        missing.extend(
            [
                "resources.gpu_seconds",
                "resources.energy_joules",
                "resources.gpu_utilization_pct",
            ]
        )
    else:
        if trace.resources.gpu_seconds is None:
            missing.append("resources.gpu_seconds")
        if trace.resources.energy_joules is None:
            missing.append("resources.energy_joules")
        if trace.resources.gpu_utilization_pct is None:
            missing.append("resources.gpu_utilization_pct")

    if trace.uncertainty is None:
        missing.extend(
            [
                "uncertainty.avg_logprob",
                "uncertainty.logprob_std",
                "uncertainty.entropy_mean",
            ]
        )
    else:
        if trace.uncertainty.avg_logprob is None:
            missing.append("uncertainty.avg_logprob")
        if trace.uncertainty.logprob_std is None:
            missing.append("uncertainty.logprob_std")
        if trace.uncertainty.entropy_mean is None:
            missing.append("uncertainty.entropy_mean")

    return sorted(set(missing))


def _inject_telemetry(trace: ModelExecutionTrace, telemetry: Dict[str, Any], run: RunInfo) -> None:
    online = telemetry.get("online") or {}
    throughput = telemetry.get("throughput") or {}
    engine = telemetry.get("engine") or {}

    queue_depth = _safe_float(engine.get("waiting_mean"))
    pending_requests = _safe_float(engine.get("running_mean"))

    throughput_rps_recent = _safe_float(throughput.get("request_throughput_rps"))
    if throughput_rps_recent is None:
        throughput_rps_recent = _safe_float(online.get("estimated_request_throughput_rps_peak"))

    active_workers = _safe_float(run.run_meta.get("gpu_count"))
    if active_workers is None:
        active_workers = _safe_float(run.run_meta.get("slurm_gpus_on_node"))

    if trace.system_state is None:
        from predictors.schemas import SystemStateSnapshot

        trace.system_state = SystemStateSnapshot()

    if trace.system_state.queue_depth is None:
        trace.system_state.queue_depth = queue_depth
    if trace.system_state.pending_requests is None:
        trace.system_state.pending_requests = pending_requests
    if trace.system_state.throughput_rps_recent is None:
        trace.system_state.throughput_rps_recent = throughput_rps_recent
    if trace.system_state.active_workers is None:
        trace.system_state.active_workers = active_workers

    if trace.resources is None:
        from predictors.schemas import ResourceSnapshot

        trace.resources = ResourceSnapshot()

    gpu_util_proxy = _safe_float(engine.get("kv_cache_usage_pct_mean"))

    if trace.resources.gpu_utilization_pct is None:
        trace.resources.gpu_utilization_pct = gpu_util_proxy


def _build_traces_from_triplet(
    *,
    triplet_id: str,
    triplet: Dict[str, RunInfo],
    telemetry_by_role: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    benchmarks = None
    for role in ROLE_ORDER:
        role_bench = set(triplet[role].benchmark_files.keys())
        benchmarks = role_bench if benchmarks is None else benchmarks.intersection(role_bench)

    common_benchmarks = sorted(benchmarks or [])
    traces: List[Dict[str, Any]] = []
    bench_report: Dict[str, Any] = {}

    for benchmark in common_benchmarks:
        rows_by_role: Dict[str, Dict[str, Tuple[Dict[str, Any], int]]] = {}
        raw_counts: Dict[str, int] = {}

        for role in ROLE_ORDER:
            file_path = triplet[role].benchmark_files[benchmark]
            rows = read_jsonl(file_path)
            keyed: Dict[str, Tuple[Dict[str, Any], int]] = {}
            for idx, row in enumerate(rows):
                keyed[_row_key(row, idx)] = (row, idx)
            rows_by_role[role] = keyed
            raw_counts[role] = len(rows)

        common_keys = set(rows_by_role[ROLE_ORDER[0]].keys())
        for role in ROLE_ORDER[1:]:
            common_keys = common_keys.intersection(rows_by_role[role].keys())
        common_keys_sorted = _sorted_keys(common_keys)

        bench_report[benchmark] = {
            "raw_rows": raw_counts,
            "aligned_rows": len(common_keys_sorted),
            "dropped_due_to_alignment": {
                role: raw_counts[role] - len(common_keys_sorted)
                for role in ROLE_ORDER
            },
        }

        for key in common_keys_sorted:
            query_id = f"{benchmark}:{key}"
            for role in ROLE_ORDER:
                run = triplet[role]
                row, source_idx = rows_by_role[role][key]
                trace = legacy_quality_row_to_trace(
                    row,
                    benchmark=benchmark,
                    model_name=role,
                    run_id=run.run_id,
                    source_file=str(run.benchmark_files[benchmark]),
                    source_record_index=source_idx,
                )

                trace.query_id = query_id
                trace.model_tier = role
                trace.example_id = str(key)
                trace.timestamp_utc = run.timestamp_utc

                tags = dict(trace.tags or {})
                tags["request"] = {
                    "max_tokens": run.request_max_tokens,
                    "temperature": run.request_temperature,
                }
                tags["real_ingest"] = {
                    "triplet_id": triplet_id,
                    "quality_run_id": run.run_id,
                    "quality_run_timestamp_utc": run.timestamp_utc,
                    "alignment_key": key,
                    "alignment_benchmark": benchmark,
                }
                tags["telemetry"] = telemetry_by_role.get(role, {})
                trace.tags = tags

                _inject_telemetry(trace, telemetry_by_role.get(role, {}), run)
                trace.missing_signals = _collect_missing_signals(trace)

                traces.append(trace.to_dict())

    return traces, {
        "triplet_id": triplet_id,
        "quality_runs": {
            role: {
                "run_id": triplet[role].run_id,
                "timestamp_utc": triplet[role].timestamp_utc,
                "slurm_job_id": triplet[role].slurm_job_id,
                "benchmarks": sorted(triplet[role].benchmark_files.keys()),
            }
            for role in ROLE_ORDER
        },
        "benchmarks": bench_report,
    }


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _summarize_missing_signals(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        missing = row.get("missing_signals") or []
        for key in missing:
            counter[str(key)] += 1
    return dict(counter)


def _build_report(
    *,
    output_trace: Path,
    output_report: Path,
    rows: Sequence[Dict[str, Any]],
    triplet_reports: Sequence[Dict[str, Any]],
    telemetry_matches: Sequence[TelemetryMatch],
    quality_runs: Dict[str, List[RunInfo]],
    online_runs: Dict[str, List[RunInfo]],
    throughput_runs: Dict[str, List[RunInfo]],
    args_dict: Dict[str, Any],
) -> Dict[str, Any]:
    benchmark_counter: Counter[str] = Counter()
    model_counter: Counter[str] = Counter()

    telemetry_presence = {
        "with_online": 0,
        "with_throughput": 0,
        "with_engine": 0,
    }

    for row in rows:
        benchmark_counter[str(row.get("benchmark", ""))] += 1
        model_counter[str(row.get("model_name", ""))] += 1

        telemetry = (row.get("tags") or {}).get("telemetry") or {}
        if telemetry.get("online"):
            telemetry_presence["with_online"] += 1
        if telemetry.get("throughput"):
            telemetry_presence["with_throughput"] += 1
        if telemetry.get("engine"):
            telemetry_presence["with_engine"] += 1

    report = {
        "generated_at_utc": _now_utc_iso(),
        "config": args_dict,
        "artifacts": {
            "trace_jsonl": str(output_trace),
            "report_json": str(output_report),
        },
        "discovery": {
            "quality_runs": {role: [run.run_id for run in quality_runs.get(role, [])] for role in ROLE_ORDER},
            "online_runs": {role: [run.run_id for run in online_runs.get(role, [])] for role in ROLE_ORDER},
            "throughput_runs": {role: [run.run_id for run in throughput_runs.get(role, [])] for role in ROLE_ORDER},
        },
        "triplet_count": len(triplet_reports),
        "triplets": list(triplet_reports),
        "telemetry_matches": [match.__dict__ for match in telemetry_matches],
        "output_row_count": len(rows),
        "rows_per_benchmark": dict(benchmark_counter),
        "rows_per_model": dict(model_counter),
        "telemetry_presence_rows": telemetry_presence,
        "missing_signal_counts": _summarize_missing_signals(rows),
    }
    return report


def build_real_trace_pool(
    *,
    quality_root: Path,
    online_root: Path,
    throughput_root: Path,
    logs_dir: Path,
    output_trace: Path,
    output_report: Path,
    max_triplet_gap_hours: float,
    max_telemetry_gap_hours: float,
    max_triplets: Optional[int],
) -> Dict[str, Any]:
    quality_runs = _discover_quality_runs(quality_root, ROLE_ORDER)
    online_runs = _discover_perf_runs(
        online_root,
        prefix="online",
        summary_filename="online_results.json",
        roles=ROLE_ORDER,
    )
    throughput_runs = _discover_perf_runs(
        throughput_root,
        prefix="throughput",
        summary_filename="throughput_results.json",
        roles=ROLE_ORDER,
    )

    triplets = _build_triplets(quality_runs, max_gap_hours=max_triplet_gap_hours)
    if max_triplets is not None:
        triplets = triplets[-max_triplets:]

    log_index = _index_logs(logs_dir)

    all_rows: List[Dict[str, Any]] = []
    triplet_reports: List[Dict[str, Any]] = []
    telemetry_matches: List[TelemetryMatch] = []

    for idx, triplet in enumerate(triplets):
        triplet_id = f"triplet_{idx:03d}"
        telemetry_by_role: Dict[str, Dict[str, Any]] = {}
        for role in ROLE_ORDER:
            telemetry, match = _make_telemetry_bundle(
                quality_run=triplet[role],
                role=role,
                online_runs=online_runs,
                throughput_runs=throughput_runs,
                max_gap_hours=max_telemetry_gap_hours,
                log_index=log_index,
            )
            telemetry_by_role[role] = telemetry
            telemetry_matches.append(match)

        triplet_rows, triplet_report = _build_traces_from_triplet(
            triplet_id=triplet_id,
            triplet=triplet,
            telemetry_by_role=telemetry_by_role,
        )
        all_rows.extend(triplet_rows)
        triplet_reports.append(triplet_report)

    _write_jsonl(output_trace, all_rows)

    report = _build_report(
        output_trace=output_trace,
        output_report=output_report,
        rows=all_rows,
        triplet_reports=triplet_reports,
        telemetry_matches=telemetry_matches,
        quality_runs=quality_runs,
        online_runs=online_runs,
        throughput_runs=throughput_runs,
        args_dict={
            "quality_root": str(quality_root),
            "online_root": str(online_root),
            "throughput_root": str(throughput_root),
            "logs_dir": str(logs_dir),
            "output_trace": str(output_trace),
            "output_report": str(output_report),
            "max_triplet_gap_hours": max_triplet_gap_hours,
            "max_telemetry_gap_hours": max_telemetry_gap_hours,
            "max_triplets": max_triplets,
        },
    )

    output_report.parent.mkdir(parents=True, exist_ok=True)
    with output_report.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    return {
        "trace_jsonl": str(output_trace),
        "report_json": str(output_report),
        "rows_written": len(all_rows),
        "triplets_used": len(triplets),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build real multi-model trace pool with telemetry enrichment")
    parser.add_argument("--quality-root", default="results/quality")
    parser.add_argument("--online-root", default="results/online")
    parser.add_argument("--throughput-root", default="results/throughput")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--output-trace", default="results/predictors/traces/iter2_real_multimodel_trace.jsonl")
    parser.add_argument("--output-report", default="results/predictors/traces/iter2_real_multimodel_trace_report.json")
    parser.add_argument("--max-triplet-gap-hours", type=float, default=36.0)
    parser.add_argument("--max-telemetry-gap-hours", type=float, default=96.0)
    parser.add_argument("--max-triplets", type=int, default=None)
    args = parser.parse_args()

    result = build_real_trace_pool(
        quality_root=Path(args.quality_root),
        online_root=Path(args.online_root),
        throughput_root=Path(args.throughput_root),
        logs_dir=Path(args.logs_dir),
        output_trace=Path(args.output_trace),
        output_report=Path(args.output_report),
        max_triplet_gap_hours=args.max_triplet_gap_hours,
        max_telemetry_gap_hours=args.max_telemetry_gap_hours,
        max_triplets=args.max_triplets,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
