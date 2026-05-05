"""
bench/run_load_capture.py
=========================
Phase A — Poisson-arrival load capture against a vLLM endpoint.

Generates an *open-loop* workload with exponential inter-arrival times
(λ configurable) and records, for every request:

* timestamps                  — t_send, t_first_token, t_last_token
* generation outcome          — output_tokens, error
* prompt-side features        — input_tokens, prompt_text
* **decision-time z**         — client-side queue, server-side /metrics
* resource z (best effort)    — mean GPU utilisation during the request

The output is a single ``trace.jsonl`` per role/rate combination using the
:class:`predictors.schemas.ModelExecutionTrace` schema, so the existing
``predictors/builders/build_cost_dataset.py`` and
``predictors/training/train_cost.py`` can ingest it directly without any
schema bridge.

This module deliberately mirrors the conventions of
``bench/run_online_load.py`` (same async client, same warm-up pattern) but
adds:

* True Poisson arrivals (numpy exponential), not fixed-interval pacing.
* Streaming SSE parsing to capture true TTFT.
* Background scrape of the vLLM ``/metrics`` Prometheus endpoint.
* Optional out-of-process ``nvidia-smi`` sidecar.
* Per-request enrichment with the most recent server / GPU sample.

Usage
-----
    python -m bench.run_load_capture \\
        --config configs/phase_a.yaml \\
        --role teacher \\
        --base-url http://compute-node:8000

A typical SLURM invocation runs through ``slurm/launch_phase_a.sh`` so
that one server job is co-allocated with a capture client per role.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

from bench.metrics import save_csv, save_json, summarise_latencies
from distill.dataset_utils import load_prompts, write_jsonl
from utils.config_loader import load_yaml
from utils.logging import get_logger, setup_logging
from utils.reproducibility import (
    collect_metadata, make_run_dir, save_metadata, set_seed, snapshot_configs,
)

logger = get_logger(__name__)


# ── Helpers ─────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _approx_word_count(text: str) -> int:
    return sum(1 for _ in re.finditer(r"\S+", text or ""))


def _read_endpoint_file(role: str, base_dir: str = "results/routing/endpoints") -> Optional[str]:
    """Return the URL published by ``server_role_phase2.sbatch`` for *role*."""
    p = Path(base_dir) / f"{role}.url"
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8").strip()
    return text or None


# ── Server-metrics scraper ──────────────────────────────────

class _MetricsScraper:
    """Periodic poll of the vLLM ``/metrics`` Prometheus endpoint.

    Captures a small subset of gauges that map onto the
    :class:`predictors.schemas.SystemStateSnapshot` fields (queue depth,
    pending requests, rolling throughput) so that traces can be ingested
    by the existing predictor pipeline without translation.
    """

    _RUNNING_RE = re.compile(r"^vllm:num_requests_running\b.*?\s+(?P<v>[0-9.eE+-]+)")
    _WAITING_RE = re.compile(r"^vllm:num_requests_waiting\b.*?\s+(?P<v>[0-9.eE+-]+)")
    _GEN_TPS_RE = re.compile(r"^vllm:avg_generation_throughput_toks_per_s\b.*?\s+(?P<v>[0-9.eE+-]+)")
    _KV_RE = re.compile(r"^vllm:gpu_cache_usage_perc\b.*?\s+(?P<v>[0-9.eE+-]+)")

    def __init__(self, base_url: str, interval_s: float, output: Optional[Path]) -> None:
        self._url = f"{base_url.rstrip('/')}/metrics"
        self._interval_s = float(interval_s)
        self._output = Path(output) if output else None
        self._task: Optional[asyncio.Task[None]] = None
        self._stop = asyncio.Event()
        self._last: Dict[str, Optional[float]] = {
            "running": None,
            "waiting": None,
            "gen_tps": None,
            "kv_usage_pct": None,
            "ts_monotonic": None,
        }
        self._available: Optional[bool] = None

    @property
    def last(self) -> Dict[str, Optional[float]]:
        return dict(self._last)

    @property
    def available(self) -> bool:
        return bool(self._available)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _run(self) -> None:
        fh = self._output.open("w", encoding="utf-8") if self._output else None
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                while not self._stop.is_set():
                    sample = await self._poll(client)
                    if sample is not None and fh is not None:
                        fh.write(json.dumps(sample, default=str) + "\n")
                        fh.flush()
                    try:
                        await asyncio.wait_for(self._stop.wait(), timeout=self._interval_s)
                    except asyncio.TimeoutError:
                        pass
        finally:
            if fh is not None:
                fh.close()

    async def _poll(self, client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
        try:
            resp = await client.get(self._url)
        except httpx.RequestError:
            self._available = self._available if self._available is not None else False
            return None
        if resp.status_code != 200:
            self._available = False
            return None
        self._available = True

        body = resp.text
        running = self._extract(self._RUNNING_RE, body)
        waiting = self._extract(self._WAITING_RE, body)
        gen_tps = self._extract(self._GEN_TPS_RE, body)
        kv = self._extract(self._KV_RE, body)
        ts_mono = time.monotonic()

        self._last = {
            "running": running,
            "waiting": waiting,
            "gen_tps": gen_tps,
            "kv_usage_pct": kv,
            "ts_monotonic": ts_mono,
        }
        return {
            "ts": _now_iso(),
            "ts_monotonic": ts_mono,
            "running": running,
            "waiting": waiting,
            "gen_tps": gen_tps,
            "kv_usage_pct": kv,
        }

    @staticmethod
    def _extract(pattern: re.Pattern[str], body: str) -> Optional[float]:
        for line in body.splitlines():
            m = pattern.match(line)
            if m is not None:
                try:
                    return float(m.group("v"))
                except ValueError:
                    return None
        return None


# ── GPU sidecar wrapper ─────────────────────────────────────

class _GpuSidecar:
    """Spawn ``python -m bench.gpu_sampler`` as an out-of-process sidecar."""

    def __init__(self, output: Path, interval_s: float) -> None:
        self._output = Path(output)
        self._interval_s = float(interval_s)
        self._proc: Optional[subprocess.Popen[bytes]] = None

    def start(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "bench.gpu_sampler",
            "--output",
            str(self._output),
            "--interval",
            str(self._interval_s),
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("GPU sidecar started", extra={"pid": self._proc.pid})
        except (FileNotFoundError, OSError) as exc:
            logger.warning("Could not start GPU sidecar", extra={"error": str(exc)})
            self._proc = None

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=2.0)
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logger.warning("GPU sidecar shutdown problem", extra={"error": str(exc)})
        finally:
            self._proc = None


# ── In-flight tracker ───────────────────────────────────────

class _InflightTracker:
    """Counts requests issued but not yet finished (closed-loop proxy)."""

    def __init__(self) -> None:
        self._count = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> int:
        async with self._lock:
            self._count += 1
            return self._count

    async def release(self) -> int:
        async with self._lock:
            self._count = max(0, self._count - 1)
            return self._count

    @property
    def value(self) -> int:
        return self._count


# ── Single request ──────────────────────────────────────────

async def _send_streaming(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    """Issue one ``/v1/completions`` request and time the SSE stream.

    Falls back to non-streaming timing for older vLLM builds that ignore
    ``"stream": true`` (in which case ttft equals latency).
    """
    t_start = time.perf_counter()
    ttft_ms: Optional[float] = None
    last_event_ms: Optional[float] = None
    output_tokens: Optional[int] = None
    body_text_parts: List[str] = []
    error: Optional[str] = None
    status: Optional[int] = None

    try:
        async with client.stream("POST", url, json=payload, timeout=timeout) as resp:
            status = resp.status_code
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    body_text_parts.append(line)
                    continue
                payload_text = line[len("data:"):].strip()
                if payload_text == "[DONE]":
                    last_event_ms = (time.perf_counter() - t_start) * 1000.0
                    break
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_start) * 1000.0
                last_event_ms = (time.perf_counter() - t_start) * 1000.0
                try:
                    obj = json.loads(payload_text)
                except json.JSONDecodeError:
                    continue
                usage = obj.get("usage")
                if isinstance(usage, dict):
                    ct = usage.get("completion_tokens")
                    if isinstance(ct, (int, float)):
                        output_tokens = int(ct)
    except Exception as exc:  # pragma: no cover - networking hard to exercise in tests
        error = str(exc)

    latency_ms = (time.perf_counter() - t_start) * 1000.0
    if ttft_ms is None:
        ttft_ms = latency_ms  # no stream => TTFB ≈ latency

    return {
        "status": status,
        "ttft_ms": ttft_ms,
        "latency_ms": latency_ms,
        "last_event_ms": last_event_ms,
        "output_tokens": output_tokens,
        "error": error,
    }


# ── Capture session for one (role, rate) pair ───────────────

async def _run_session(
    *,
    role: str,
    model_name: str,
    base_url: str,
    arrival_rate_rps: float,
    num_requests: int,
    num_warmup: int,
    max_inflight: int,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    stream: bool,
    include_usage: bool,
    prompts: List[str],
    rng: np.random.Generator,
    metrics_scraper: Optional[_MetricsScraper],
    out_dir: Path,
) -> Dict[str, Any]:
    """Run one Poisson session and write trace + sidecar files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    url = f"{base_url.rstrip('/')}/v1/completions"
    sem = asyncio.Semaphore(max_inflight)
    inflight = _InflightTracker()

    # Warm-up first (results discarded)
    if num_warmup > 0:
        await _warm_up(
            url=url, model_name=model_name, prompts=prompts, rng=rng,
            num=num_warmup, sem=sem, max_tokens=max_tokens,
            temperature=temperature, timeout=timeout_s,
            stream=stream, include_usage=include_usage,
        )

    # Recent latency window for the rolling-p50 z feature.
    recent: List[float] = []
    recent_window = 50

    results: List[Dict[str, Any]] = []
    t0_mono = time.monotonic()

    async with httpx.AsyncClient(http2=False) as client:
        async def _one_request(req_idx: int, prompt_idx: int) -> None:
            prompt = prompts[prompt_idx]
            payload: Dict[str, Any] = {
                "model": model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if stream:
                payload["stream"] = True
                if include_usage:
                    payload["stream_options"] = {"include_usage": True}

            inflight_at_send = await inflight.acquire()
            t_send_mono = time.monotonic()
            t_send_iso = _now_iso()

            metrics_at_send = (
                metrics_scraper.last if metrics_scraper is not None else {}
            )

            recent_p50 = (
                float(np.percentile(recent[-recent_window:], 50))
                if recent else None
            )

            async with sem:
                outcome = await _send_streaming(client, url, payload, timeout_s)

            await inflight.release()
            if outcome["error"] is None and isinstance(outcome["latency_ms"], (int, float)):
                recent.append(float(outcome["latency_ms"]))

            results.append({
                "req_idx": req_idx,
                "prompt_idx": prompt_idx,
                "prompt": prompt,
                "input_word_count": _approx_word_count(prompt),
                "t_send_iso": t_send_iso,
                "t_send_monotonic": t_send_mono,
                **outcome,
                "z_inflight_at_send": inflight_at_send,
                "z_recent_p50_latency_ms": recent_p50,
                "z_lambda_target_rps": arrival_rate_rps,
                "z_metrics_running": metrics_at_send.get("running"),
                "z_metrics_waiting": metrics_at_send.get("waiting"),
                "z_metrics_gen_tps": metrics_at_send.get("gen_tps"),
                "z_metrics_kv_usage_pct": metrics_at_send.get("kv_usage_pct"),
            })

        tasks: List[asyncio.Task[None]] = []
        for i in range(num_requests):
            prompt_idx = int(rng.integers(0, len(prompts)))
            tasks.append(asyncio.create_task(_one_request(i, prompt_idx)))
            interval = float(rng.exponential(1.0 / arrival_rate_rps))
            await asyncio.sleep(interval)

        await asyncio.gather(*tasks, return_exceptions=False)

    t_end_mono = time.monotonic()
    wall_s = max(t_end_mono - t0_mono, 1e-9)

    ok = [r for r in results if r["error"] is None]
    latencies = [r["latency_ms"] for r in ok if r["latency_ms"] is not None]
    ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
    out_tokens = [int(r["output_tokens"]) for r in ok if isinstance(r["output_tokens"], (int, float))]

    summary = {
        "role": role,
        "model_name": model_name,
        "arrival_rate_rps": arrival_rate_rps,
        "num_requests": len(results),
        "successful_requests": len(ok),
        "failed_requests": len(results) - len(ok),
        "total_output_tokens": sum(out_tokens),
        "wall_clock_seconds": wall_s,
        "achieved_request_rps": len(results) / wall_s,
        **{f"latency_{k}": v for k, v in summarise_latencies(latencies).items()},
        **{f"ttft_{k}": v for k, v in summarise_latencies(ttfts).items()},
    }

    # Per-session artefacts
    save_json(results, out_dir / "raw_requests.json")
    save_json(summary, out_dir / "summary.json")
    save_csv(results, out_dir / "raw_requests.csv")

    return {
        "summary": summary,
        "results": results,
    }


async def _warm_up(
    *,
    url: str,
    model_name: str,
    prompts: List[str],
    rng: np.random.Generator,
    num: int,
    sem: asyncio.Semaphore,
    max_tokens: int,
    temperature: float,
    timeout: float,
    stream: bool,
    include_usage: bool,
) -> None:
    """Burst-mode warm-up. Results discarded."""
    async with httpx.AsyncClient(http2=False) as client:
        async def _one(prompt: str) -> None:
            payload: Dict[str, Any] = {
                "model": model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if stream:
                payload["stream"] = True
                if include_usage:
                    payload["stream_options"] = {"include_usage": True}
            async with sem:
                await _send_streaming(client, url, payload, timeout)

        tasks = []
        for _ in range(num):
            prompt = prompts[int(rng.integers(0, len(prompts)))]
            tasks.append(asyncio.create_task(_one(prompt)))
        await asyncio.gather(*tasks, return_exceptions=True)


# ── GPU samples post-processing ─────────────────────────────

def _attach_gpu_samples(
    rows: List[Dict[str, Any]],
    gpu_jsonl: Path,
) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Annotate trace rows with mean GPU utilisation during their lifetime.

    Returns ``(records_with_gpu, summary_dict)`` for diagnostics.
    """
    if not gpu_jsonl.exists():
        return 0, None

    samples: List[Dict[str, Any]] = []
    with gpu_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not samples:
        return 0, None

    # Pre-compute per-sample mean util across GPUs (single-GPU is the common
    # serving setup but we keep the average general).
    times = np.asarray([float(s["ts_monotonic"]) for s in samples], dtype=float)
    util = np.asarray(
        [
            float(np.mean([
                (g.get("utilization_gpu_pct") or 0.0) for g in s.get("gpus") or []
            ])) if s.get("gpus") else 0.0
            for s in samples
        ],
        dtype=float,
    )

    n_attached = 0
    for row in rows:
        t_send = float(row.get("t_send_monotonic") or 0.0)
        latency_ms = row.get("latency_ms")
        if latency_ms is None:
            continue
        t_end = t_send + float(latency_ms) / 1000.0

        # Closest-window mean — vectorised mask
        mask = (times >= t_send) & (times <= t_end)
        if not mask.any():
            # Fall back to nearest sample
            idx = int(np.argmin(np.abs(times - t_send)))
            row["gpu_utilization_pct_mean"] = float(util[idx])
        else:
            row["gpu_utilization_pct_mean"] = float(util[mask].mean())

        gpu_seconds_proxy = (row["gpu_utilization_pct_mean"] / 100.0) * (float(latency_ms) / 1000.0)
        row["gpu_seconds_proxy"] = gpu_seconds_proxy
        n_attached += 1

    summary = {
        "samples_loaded": len(samples),
        "rows_annotated": n_attached,
        "mean_util_pct": float(util.mean()) if util.size else None,
        "max_util_pct": float(util.max()) if util.size else None,
    }
    return n_attached, summary


# ── Trace-file emission (ModelExecutionTrace schema) ────────

def _emit_trace_jsonl(
    *,
    role: str,
    model_name: str,
    benchmark_label: str,
    run_id: str,
    rows: List[Dict[str, Any]],
    arrival_rate_rps: float,
    out_path: Path,
) -> int:
    """Materialise rows into the predictors/ trace schema."""
    out_records: List[Dict[str, Any]] = []
    for r in rows:
        running = _none_to_float(r.get("z_metrics_running"))
        waiting = _none_to_float(r.get("z_metrics_waiting"))
        active_workers: Optional[float] = None
        if running is not None and waiting is not None:
            active_workers = running

        record = {
            "query_id": f"phase_a:{benchmark_label}:{role}:{int(r['req_idx'])}",
            "benchmark": benchmark_label,
            "model_name": model_name,
            "run_id": run_id,
            "timestamp_utc": r.get("t_send_iso"),
            "source_file": str(out_path),
            "source_record_index": int(r["req_idx"]),
            "model_tier": role,
            "prompt_text": r.get("prompt"),
            "response_text": None,
            "input_tokens": int(r.get("input_word_count") or 0) or None,
            "output_tokens": _none_to_int(r.get("output_tokens")),
            "latency_ms": _none_to_float(r.get("latency_ms")),
            "ttft_ms": _none_to_float(r.get("ttft_ms")),
            "correct": None,
            "score": None,
            "system_state": {
                "queue_depth": _none_to_float(r.get("z_metrics_waiting")),
                "pending_requests": _none_to_float(r.get("z_inflight_at_send")),
                "throughput_rps_recent": _none_to_float(r.get("z_metrics_gen_tps")),
                "active_workers": active_workers,
            },
            "resources": {
                "gpu_seconds": _none_to_float(r.get("gpu_seconds_proxy")),
                "energy_joules": None,
                "gpu_utilization_pct": _none_to_float(r.get("gpu_utilization_pct_mean")),
            },
            "uncertainty": None,
            "tags": {
                "phase": "A",
                "request": {
                    "max_tokens": None,
                    "temperature": None,
                },
                "telemetry": {
                    "online": {
                        "lambda_target_rps": arrival_rate_rps,
                        "achieved_rps": None,
                    },
                    "engine": {
                        "running_mean": _none_to_float(r.get("z_metrics_running")),
                        "waiting_mean": _none_to_float(r.get("z_metrics_waiting")),
                        "kv_cache_usage_pct_mean": _none_to_float(r.get("z_metrics_kv_usage_pct")),
                        "generation_throughput_mean": _none_to_float(r.get("z_metrics_gen_tps")),
                    },
                },
                "z_inflight_at_send": _none_to_float(r.get("z_inflight_at_send")),
                "z_recent_p50_latency_ms": _none_to_float(r.get("z_recent_p50_latency_ms")),
            },
            "missing_signals": [],
        }
        out_records.append(record)

    write_jsonl(out_records, out_path)
    return len(out_records)


def _none_to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _none_to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


# ── Orchestrator ────────────────────────────────────────────

def run(
    *,
    config_path: str,
    role: str,
    base_url_override: Optional[str] = None,
    model_override: Optional[str] = None,
    benchmark_label: str = "phase_a_workload",
) -> Path:
    """Execute the Phase A capture for one role across all configured rates."""
    cfg = load_yaml(config_path)
    common = cfg.get("common", {})
    capt = cfg.get("capture", {})
    samp_cfg = (capt.get("samplers") or {})

    seed = int(common.get("seed", 42))
    set_seed(seed)
    setup_logging()

    base_url = (
        base_url_override
        or _read_endpoint_file(role)
        or common.get("base_url", "http://localhost:8000")
    )

    # Resolve model name from configs/models.yaml unless overridden.
    model_name = model_override
    if not model_name:
        try:
            models_cfg = load_yaml("configs/models.yaml")
            model_name = (models_cfg.get(role) or {}).get("name", "")
        except FileNotFoundError:
            model_name = ""
    if not model_name:
        raise ValueError(
            f"Could not resolve model name for role='{role}'. "
            "Pass --model or ensure configs/models.yaml has the role entry."
        )

    prompts_file = capt.get("prompts_file", "configs/prompts.jsonl")
    prompts = load_prompts(prompts_file)
    if not prompts:
        raise ValueError(f"No prompts loaded from {prompts_file}")

    rates: List[float] = [float(x) for x in capt.get("arrival_rates_rps", [10])]
    num_requests = int(capt.get("num_requests_per_rate", 600))
    num_warmup = int(capt.get("num_warmup_requests", 50))
    max_inflight = int(capt.get("max_inflight", 256))
    max_tokens = int(capt.get("max_tokens", 256))
    temperature = float(capt.get("temperature", 0.0))
    timeout_s = float(capt.get("request_timeout_s", 120.0))
    stream = bool(capt.get("stream", True))
    include_usage = bool(capt.get("include_usage", True))

    base_dir = common.get("results_base_dir", "results/phase_a")
    run_dir = make_run_dir(f"{base_dir}/captures", tag=f"capture-{role}")
    snapshot_configs(
        [config_path, "configs/models.yaml", "configs/serving.yaml"],
        run_dir,
    )
    meta = collect_metadata(seed, cfg)
    meta.update({
        "role": role,
        "model_name": model_name,
        "base_url": base_url,
        "rates_rps": rates,
        "phase": "A",
    })
    save_metadata(meta, run_dir)

    # ── Background samplers (one set, shared across rates) ────
    metrics_sidecar: Optional[_MetricsScraper] = None
    gpu_sidecar: Optional[_GpuSidecar] = None
    if (samp_cfg.get("server_metrics") or {}).get("enabled", True):
        metrics_sidecar = _MetricsScraper(
            base_url=base_url,
            interval_s=float(samp_cfg["server_metrics"].get("interval_s", 0.25)),
            output=run_dir / "server_metrics.jsonl",
        )
    if (samp_cfg.get("gpu_smi") or {}).get("enabled", True):
        gpu_sidecar = _GpuSidecar(
            output=run_dir / "gpu_samples.jsonl",
            interval_s=float(samp_cfg["gpu_smi"].get("interval_s", 0.2)),
        )

    rng = np.random.default_rng(seed)

    async def _run_all() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if metrics_sidecar is not None:
            metrics_sidecar.start()
        if gpu_sidecar is not None:
            gpu_sidecar.start()

        try:
            summaries: List[Dict[str, Any]] = []
            all_rows: List[Dict[str, Any]] = []
            for rate in rates:
                logger.info(
                    "Phase A session start",
                    extra={"role": role, "rate_rps": rate, "num_requests": num_requests},
                )
                rate_dir = run_dir / f"rate_{rate}"
                session = await _run_session(
                    role=role,
                    model_name=model_name,
                    base_url=base_url,
                    arrival_rate_rps=rate,
                    num_requests=num_requests,
                    num_warmup=num_warmup,
                    max_inflight=max_inflight,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout_s=timeout_s,
                    stream=stream,
                    include_usage=include_usage,
                    prompts=prompts,
                    rng=rng,
                    metrics_scraper=metrics_sidecar,
                    out_dir=rate_dir,
                )
                summaries.append(session["summary"])
                all_rows.extend(session["results"])
                logger.info(
                    "Phase A session done",
                    extra={"role": role, "rate_rps": rate,
                            "ok": session["summary"]["successful_requests"]},
                )
            return summaries, all_rows
        finally:
            if metrics_sidecar is not None:
                await metrics_sidecar.stop()
            if gpu_sidecar is not None:
                gpu_sidecar.stop()

    summaries, all_rows = asyncio.run(_run_all())

    # ── Attach GPU utilisation post-hoc ─────────────────────
    gpu_summary: Optional[Dict[str, Any]] = None
    if gpu_sidecar is not None:
        n_attached, gpu_summary = _attach_gpu_samples(
            rows=all_rows,
            gpu_jsonl=run_dir / "gpu_samples.jsonl",
        )
        logger.info("GPU samples attached", extra={"rows": n_attached})

    # ── Aggregate summaries ─────────────────────────────────
    save_json(summaries, run_dir / "summaries.json")
    save_csv(summaries, run_dir / "summaries.csv")
    if gpu_summary is not None:
        save_json(gpu_summary, run_dir / "gpu_samples_summary.json")

    # ── Trace JSONL (predictors-compatible, single file) ────
    run_id = run_dir.name
    trace_path = run_dir / "trace.jsonl"

    # Group rows by lambda label so query_ids are stable per workload.
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_rows:
        bench_label = f"phase_a_lambda_{r['z_lambda_target_rps']}"
        r["__benchmark"] = bench_label
        grouped.setdefault(bench_label, []).append(r)

    # Re-index req_idx within each benchmark group for stable query_ids,
    # then materialise everything into a single trace.jsonl so that the
    # default glob in configs/phase_a.yaml picks it up.
    consolidated: List[Dict[str, Any]] = []
    for bench_label, rows in grouped.items():
        for new_idx, row in enumerate(rows):
            row["req_idx"] = new_idx
        consolidated.extend(rows)

    consolidated.sort(key=lambda r: (str(r["__benchmark"]), int(r["req_idx"])))

    n_records = 0
    for bench_label, rows in grouped.items():
        n_records += _emit_trace_jsonl(
            role=role,
            model_name=model_name,
            benchmark_label=bench_label,
            run_id=run_id,
            rows=rows,
            arrival_rate_rps=float(rows[0]["z_lambda_target_rps"]) if rows else 0.0,
            out_path=run_dir / f"_partial_{bench_label}.jsonl",
        )

    # Concatenate partials into the canonical trace.jsonl, then drop them.
    with trace_path.open("w", encoding="utf-8") as out_fh:
        for partial in sorted(run_dir.glob("_partial_*.jsonl")):
            out_fh.write(partial.read_text(encoding="utf-8"))
            partial.unlink(missing_ok=True)
    logger.info(
        "Phase A capture complete",
        extra={
            "run_dir": str(run_dir),
            "records_written": n_records,
            "successful_sessions": len([s for s in summaries if s["successful_requests"] > 0]),
        },
    )
    return run_dir


# ── CLI ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase A — Poisson load capture")
    parser.add_argument("--config", default="configs/phase_a.yaml")
    parser.add_argument("--role", required=True,
                        help="Role key in configs/models.yaml (teacher | student_mid | student_q3b | student_small | student_tiny)")
    parser.add_argument("--base-url", default=None,
                        help="Override base URL (else read from results/routing/endpoints/<role>.url)")
    parser.add_argument("--model", default=None,
                        help="Override model name (else taken from configs/models.yaml[role].name)")
    parser.add_argument("--benchmark-label", default="phase_a_workload",
                        help="Label written into the ModelExecutionTrace.benchmark field")
    args = parser.parse_args()

    run(
        config_path=args.config,
        role=args.role,
        base_url_override=args.base_url,
        model_override=args.model,
        benchmark_label=args.benchmark_label,
    )


if __name__ == "__main__":
    main()
