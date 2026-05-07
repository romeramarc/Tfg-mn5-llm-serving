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


def _hostname() -> str:
    try:
        import socket
        return socket.gethostname()
    except Exception:
        return ""


def _approx_word_count(text: str) -> int:
    return sum(1 for _ in re.finditer(r"\S+", text or ""))


def _read_endpoint_file(role: str, base_dir: str = "results/routing/endpoints") -> Optional[str]:
    """Return the URL published by ``server_role_phase2.sbatch`` for *role*."""
    p = Path(base_dir) / f"{role}.url"
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8").strip()
    return text or None


def _read_gpu_path_file(role: str, base_dir: str = "results/routing/endpoints") -> Optional[Path]:
    """Return the GPU samples path published by the server (or None)."""
    p = Path(base_dir) / f"{role}.gpu"
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8").strip()
    return Path(text) if text else None


# ── Server-metrics scraper ──────────────────────────────────

class _MetricsScraper:
    """Periodic poll of the vLLM ``/metrics`` Prometheus endpoint.

    Captures a small subset of gauges that map onto the
    :class:`predictors.schemas.SystemStateSnapshot` fields (queue depth,
    pending requests, rolling throughput) so that traces can be ingested
    by the existing predictor pipeline without translation.
    """

    # vLLM metric names have changed across releases (':' vs '_' namespaces, and some suffixes).
    # We therefore match a small set of known variants.
    _RUNNING_PATTERNS = [
        re.compile(r"^(?:vllm:|vllm_)?num_requests_running\b.*?\s+(?P<v>[0-9.eE+-]+)"),
        re.compile(r"^vllm_num_requests_running\b.*?\s+(?P<v>[0-9.eE+-]+)"),
    ]
    _WAITING_PATTERNS = [
        re.compile(r"^(?:vllm:|vllm_)?num_requests_waiting\b.*?\s+(?P<v>[0-9.eE+-]+)"),
        re.compile(r"^vllm_num_requests_waiting\b.*?\s+(?P<v>[0-9.eE+-]+)"),
    ]
    _GEN_TPS_PATTERNS = [
        re.compile(r"^(?:vllm:|vllm_)?avg_generation_throughput.*?\s+(?P<v>[0-9.eE+-]+)"),
        re.compile(r"^(?:vllm:|vllm_)?generation_throughput.*?\s+(?P<v>[0-9.eE+-]+)"),
    ]
    _KV_PATTERNS = [
        re.compile(r"^(?:vllm:|vllm_)?gpu_cache_usage_perc\b.*?\s+(?P<v>[0-9.eE+-]+)"),
        re.compile(r"^(?:vllm:|vllm_)?kv_cache_usage_perc\b.*?\s+(?P<v>[0-9.eE+-]+)"),
        re.compile(r"^(?:vllm:|vllm_)?gpu_kv_cache_usage_perc\b.*?\s+(?P<v>[0-9.eE+-]+)"),
    ]
    # Counters (used to compute generation throughput when the gauge is
    # not exposed in newer vLLM versions).
    _GEN_TOK_TOTAL_PATTERNS = [
        re.compile(r"^(?:vllm:|vllm_)?generation_tokens_total\b.*?\s+(?P<v>[0-9.eE+-]+)"),
    ]
    _PROMPT_TOK_TOTAL_PATTERNS = [
        re.compile(r"^(?:vllm:|vllm_)?prompt_tokens_total\b.*?\s+(?P<v>[0-9.eE+-]+)"),
    ]

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
            "prompt_tps": None,
            "kv_usage_pct": None,
            "ts_monotonic": None,
        }
        self._prev_gen_total: Optional[float] = None
        self._prev_prompt_total: Optional[float] = None
        self._prev_ts: Optional[float] = None
        self._diag_path: Optional[Path] = None
        self._diag_written: bool = False
        self._available: Optional[bool] = None

    @property
    def last(self) -> Dict[str, Optional[float]]:
        return dict(self._last)

    @property
    def available(self) -> bool:
        return bool(self._available)

    def set_diagnostics_path(self, path: Path) -> None:
        self._diag_path = Path(path)

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
        running = self._extract_any(self._RUNNING_PATTERNS, body)
        waiting = self._extract_any(self._WAITING_PATTERNS, body)
        gen_tps_gauge = self._extract_any(self._GEN_TPS_PATTERNS, body)
        kv = self._extract_any(self._KV_PATTERNS, body)
        gen_tot = self._extract_any(self._GEN_TOK_TOTAL_PATTERNS, body)
        prompt_tot = self._extract_any(self._PROMPT_TOK_TOTAL_PATTERNS, body)
        ts_mono = time.monotonic()

        gen_tps = gen_tps_gauge
        prompt_tps: Optional[float] = None
        if (
            self._prev_ts is not None
            and gen_tot is not None
            and self._prev_gen_total is not None
        ):
            dt = max(ts_mono - self._prev_ts, 1e-9)
            d_gen = max(gen_tot - self._prev_gen_total, 0.0)
            if gen_tps_gauge is None:
                gen_tps = d_gen / dt
            if prompt_tot is not None and self._prev_prompt_total is not None:
                d_pr = max(prompt_tot - self._prev_prompt_total, 0.0)
                prompt_tps = d_pr / dt

        if gen_tot is not None:
            self._prev_gen_total = gen_tot
        if prompt_tot is not None:
            self._prev_prompt_total = prompt_tot
        self._prev_ts = ts_mono

        if (not self._diag_written) and self._diag_path is not None:
            try:
                self._diag_path.parent.mkdir(parents=True, exist_ok=True)
                first_lines = [
                    ln for ln in body.splitlines() if ln and not ln.startswith("#")
                ][:200]
                with self._diag_path.open("w", encoding="utf-8") as dh:
                    json.dump(
                        {
                            "url": self._url,
                            "matched": {
                                "running": running,
                                "waiting": waiting,
                                "gen_tps_gauge": gen_tps_gauge,
                                "gen_tokens_total": gen_tot,
                                "prompt_tokens_total": prompt_tot,
                                "kv_usage_pct": kv,
                            },
                            "first_metric_lines": first_lines,
                        },
                        dh,
                        indent=2,
                        default=str,
                    )
                self._diag_written = True
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("metrics diagnostics write failed", extra={"error": str(exc)})

        self._last = {
            "running": running,
            "waiting": waiting,
            "gen_tps": gen_tps,
            "prompt_tps": prompt_tps,
            "kv_usage_pct": kv,
            "ts_monotonic": ts_mono,
        }
        return {
            "ts": _now_iso(),
            "ts_monotonic": ts_mono,
            "running": running,
            "waiting": waiting,
            "gen_tps": gen_tps,
            "prompt_tps": prompt_tps,
            "kv_usage_pct": kv,
            "gen_tokens_total": gen_tot,
            "prompt_tokens_total": prompt_tot,
        }

    @staticmethod
    def _extract_any(patterns: List[re.Pattern[str]], body: str) -> Optional[float]:
        for line in body.splitlines():
            for pattern in patterns:
                m = pattern.match(line)
                if m is None:
                    continue
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
                "z_metrics_prompt_tps": metrics_at_send.get("prompt_tps"),
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
    # serving setup but we keep the average general). If a sample contains no
    # GPU rows (e.g. nvidia-smi unavailable), mark it as unusable so we don't
    # silently turn missing telemetry into zeros.
    times = np.asarray([float(s["ts_monotonic"]) for s in samples], dtype=float)
    util_list: List[Optional[float]] = []
    for s in samples:
        gpus = s.get("gpus") or []
        if not gpus:
            util_list.append(None)
            continue
        vals = [(g.get("utilization_gpu_pct")) for g in gpus]
        vals = [float(v) for v in vals if isinstance(v, (int, float))]
        util_list.append(float(np.mean(vals)) if vals else None)

    usable_mask = np.asarray([u is not None for u in util_list], dtype=bool)
    if not usable_mask.any():
        return 0, {
            "samples_loaded": len(samples),
            "rows_annotated": 0,
            "mean_util_pct": None,
            "max_util_pct": None,
            "note": "No usable GPU samples (gpus empty or missing utilization).",
        }

    times_usable = times[usable_mask]
    util_usable = np.asarray([u for u in util_list if u is not None], dtype=float)

    n_attached = 0
    for row in rows:
        t_send = float(row.get("t_send_monotonic") or 0.0)
        latency_ms = row.get("latency_ms")
        if latency_ms is None:
            continue
        t_end = t_send + float(latency_ms) / 1000.0

        # Closest-window mean — vectorised mask
        mask = (times_usable >= t_send) & (times_usable <= t_end)
        if not mask.any():
            # Fall back to nearest sample
            idx = int(np.argmin(np.abs(times_usable - t_send)))
            row["gpu_utilization_pct_mean"] = float(util_usable[idx])
        else:
            row["gpu_utilization_pct_mean"] = float(util_usable[mask].mean())

        gpu_seconds_proxy = (row["gpu_utilization_pct_mean"] / 100.0) * (float(latency_ms) / 1000.0)
        row["gpu_seconds_proxy"] = gpu_seconds_proxy
        n_attached += 1

    summary = {
        "samples_loaded": len(samples),
        "rows_annotated": n_attached,
        "mean_util_pct": float(util_usable.mean()) if util_usable.size else None,
        "max_util_pct": float(util_usable.max()) if util_usable.size else None,
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
        # `active_workers` should reflect the number of requests currently
        # running on the engine; if running was scraped successfully we use
        # it regardless of whether waiting was also captured.
        active_workers: Optional[float] = running

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
                # NOTE: vLLM exposes generation throughput in tokens/s; we keep the field name for schema
                # compatibility but interpret it as "recent throughput proxy" (not strict req/s).
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
                        "prompt_throughput_mean": _none_to_float(r.get("z_metrics_prompt_tps")),
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
        metrics_sidecar.set_diagnostics_path(run_dir / "server_metrics_diag.json")
    # If the server has published a shared GPU samples path, prefer that —
    # it is taken on the GPU node where the model actually runs, so it is
    # always correct, even if the capture is allocated to a different node.
    server_gpu_path = _read_gpu_path_file(role)
    use_external_gpu = server_gpu_path is not None
    if use_external_gpu:
        logger.info(
            "Using server-side GPU samples (no local sidecar)",
            extra={"server_gpu_path": str(server_gpu_path)},
        )

    if (samp_cfg.get("gpu_smi") or {}).get("enabled", True) and not use_external_gpu:
        gpu_sidecar = _GpuSidecar(
            output=run_dir / "gpu_samples.jsonl",
            interval_s=float(samp_cfg["gpu_smi"].get("interval_s", 0.2)),
        )
        # Probe nvidia-smi up-front so we fail-loud if the capture node has no
        # visible GPU (capture must be co-located with the server node).
        probe_path = run_dir / "gpu_probe.json"
        try:
            probe_out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5.0, check=False,
            )
            probe_payload = {
                "returncode": probe_out.returncode,
                "stdout": probe_out.stdout.strip(),
                "stderr": probe_out.stderr.strip(),
                "hostname": _hostname(),
            }
            probe_path.write_text(json.dumps(probe_payload, indent=2), encoding="utf-8")
            if probe_out.returncode != 0 or not probe_out.stdout.strip():
                logger.warning(
                    "GPU probe FAILED — capture node appears to have no visible GPU; "
                    "GPU utilization features will be missing.",
                    extra=probe_payload,
                )
        except Exception as exc:
            probe_path.write_text(
                json.dumps({"error": str(exc), "hostname": _hostname()}, indent=2),
                encoding="utf-8",
            )
            logger.warning("GPU probe error", extra={"error": str(exc)})

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
    # Priority:
    #   1. Server-side path (if published by server_role_phase2.sbatch).
    #   2. Local sidecar output (legacy fallback).
    gpu_summary: Optional[Dict[str, Any]] = None
    gpu_jsonl_to_use: Optional[Path] = None
    if use_external_gpu and server_gpu_path is not None and server_gpu_path.exists():
        gpu_jsonl_to_use = server_gpu_path
        # Copy a snapshot into run_dir so the trace is self-contained.
        try:
            (run_dir / "gpu_samples.jsonl").write_text(
                server_gpu_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Failed to snapshot server GPU samples", extra={"error": str(exc)})
    elif gpu_sidecar is not None:
        gpu_jsonl_to_use = run_dir / "gpu_samples.jsonl"

    if gpu_jsonl_to_use is not None:
        n_attached, gpu_summary = _attach_gpu_samples(
            rows=all_rows,
            gpu_jsonl=gpu_jsonl_to_use,
        )
        logger.info(
            "GPU samples attached",
            extra={"rows": n_attached, "source": str(gpu_jsonl_to_use)},
        )

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

    # ── Post-capture telemetry validator (fails the job loudly on bad data) ──
    _validate_capture_telemetry(
        trace_path=trace_path,
        gpu_summary=gpu_summary,
        run_dir=run_dir,
        require_gpu=bool((samp_cfg.get("gpu_smi") or {}).get("enabled", True)),
        require_metrics=bool((samp_cfg.get("server_metrics") or {}).get("enabled", True)),
    )
    return run_dir


def _validate_capture_telemetry(
    *,
    trace_path: Path,
    gpu_summary: Optional[Dict[str, Any]],
    run_dir: Path,
    require_gpu: bool,
    require_metrics: bool,
) -> None:
    """Walk the trace and ensure key telemetry fields are populated.

    Writes ``telemetry_validation.json`` for diagnostics, and raises
    ``RuntimeError`` if a required signal is missing — so SLURM marks the
    capture as FAILED and we do NOT silently re-train on empty data.
    """
    n_total = 0
    n_running_nonzero = 0
    n_waiting_nonnull = 0
    n_gen_tps_nonnull = 0
    n_kv_nonnull = 0
    n_gpu_util_nonzero = 0

    if not trace_path.exists():
        raise RuntimeError(f"trace.jsonl missing: {trace_path}")

    with trace_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1
            engine = (((rec.get("tags") or {}).get("telemetry") or {}).get("engine") or {})
            if (rec.get("system_state") or {}).get("active_workers"):
                n_running_nonzero += 1
            if engine.get("waiting_mean") is not None:
                n_waiting_nonnull += 1
            if engine.get("generation_throughput_mean") is not None:
                n_gen_tps_nonnull += 1
            if engine.get("kv_cache_usage_pct_mean") is not None:
                n_kv_nonnull += 1
            res = rec.get("resources") or {}
            util = res.get("gpu_utilization_pct")
            if util is not None and float(util) > 0.0:
                n_gpu_util_nonzero += 1

    report = {
        "trace_rows": n_total,
        "running_nonzero": n_running_nonzero,
        "waiting_nonnull": n_waiting_nonnull,
        "gen_tps_nonnull": n_gen_tps_nonnull,
        "kv_nonnull": n_kv_nonnull,
        "gpu_util_nonzero": n_gpu_util_nonzero,
        "gpu_summary": gpu_summary,
    }
    (run_dir / "telemetry_validation.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    errors: List[str] = []
    if n_total == 0:
        errors.append("trace.jsonl is empty.")
    if require_metrics:
        if n_running_nonzero == 0:
            errors.append("active_workers (vllm running) is always zero — server /metrics not scraped correctly.")
        if n_gen_tps_nonnull == 0 and n_kv_nonnull == 0:
            errors.append("Both generation throughput and KV usage are missing — /metrics scrape produced nothing useful.")
    if require_gpu and n_gpu_util_nonzero == 0:
        errors.append(
            "GPU utilization is zero across the entire trace — the capture node "
            "probably has no visible GPU (capture must be co-located with the "
            "server node)."
        )

    if errors:
        raise RuntimeError(
            "Telemetry validation FAILED:\n  - " + "\n  - ".join(errors)
            + f"\nSee {run_dir / 'telemetry_validation.json'}"
        )


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
