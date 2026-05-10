"""
bench/run_quality_capture.py
============================
Phase B — Poisson-arrival quality capture against a vLLM endpoint.

Generates an *open-loop* Poisson workload (same arrival law as Phase A) and
issues GSM8K + MATH-500 problems with ``logprobs=k`` so we can record, per
request:

* timestamps                  — t_send, latency
* generation outcome          — response_text, output_tokens, error
* prompt-side features        — prompt_text, input_tokens, benchmark
* gold-based label            — correct (scored via eval/scoring.py)
* uncertainty                 — avg_logprob, logprob_std, entropy_mean
* decision-time z             — server-side /metrics, recent latency
* resource z                  — mean GPU utilisation during the request

The output is a single ``trace.jsonl`` per role that uses the
:class:`predictors.schemas.ModelExecutionTrace` schema, so the existing
``predictors/builders/build_ex_ante_dataset.py`` and
``predictors/builders/build_post_hoc_dataset.py`` can ingest it directly
without any schema bridge.

The module deliberately mirrors ``bench/run_load_capture.py`` and reuses
its background samplers (``_MetricsScraper``, ``_GpuSidecar``) and
post-processing helpers (``_attach_gpu_samples``, telemetry validation).
The new code paths are quality-specific:

* non-streaming POST that returns full text + ``logprobs`` payload,
* per-request answer extraction + ``correct`` scoring,
* per-request uncertainty stats from token-level logprobs,
* trace emission populates ``correct``, ``response_text`` and
  ``uncertainty.*`` (which Phase A leaves as ``None``).

Usage
-----
    python -m bench.run_quality_capture \\
        --config configs/phase_b.yaml \\
        --role teacher \\
        --base-url http://compute-node:8000

Normally invoked through ``slurm/launch_phase_b.sh``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

from bench.metrics import save_csv, save_json, summarise_latencies
from bench.run_load_capture import (
    _GpuSidecar,
    _InflightTracker,
    _MetricsScraper,
    _attach_gpu_samples,
    _hostname,
    _read_endpoint_file,
    _read_gpu_path_file,
)
from distill.dataset_utils import write_jsonl
from eval.scoring import (
    extract_boxed_answer,
    extract_numeric_answer,
    math_answer_match,
    numeric_match,
)
from utils.config_loader import load_yaml
from utils.logging import get_logger, setup_logging
from utils.reproducibility import (
    collect_metadata, make_run_dir, save_metadata, set_seed, snapshot_configs,
)

logger = get_logger(__name__)


def _wait_for_routing_endpoint_files(
    role: str,
    *,
    require_gpu_publish: bool,
    timeout_s: float = 900.0,
    poll_s: float = 3.0,
) -> None:
    """Block until the server-side launcher publishes URL (and optionally GPU path) files.

    Avoids a race where the capture job starts before ``server_role_phase2.sbatch``
    has written ``results/routing/endpoints/<role>.{url,gpu}`` on the shared FS.
    """
    base = Path("results/routing/endpoints")
    url_f = base / f"{role}.url"
    gpu_f = base / f"{role}.gpu"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        url_ok = url_f.is_file() and bool(url_f.read_text(encoding="utf-8").strip())
        gpu_ok = not require_gpu_publish or (
            gpu_f.is_file() and bool(gpu_f.read_text(encoding="utf-8").strip())
        )
        if url_ok and gpu_ok:
            logger.info(
                "Routing endpoint files ready",
                extra={"url_file": str(url_f), "gpu_file": str(gpu_f)},
            )
            return
        time.sleep(poll_s)
    need = f"{url_f}" + (f" and {gpu_f}" if require_gpu_publish else "")
    raise RuntimeError(
        f"Timed out after {timeout_s:.0f}s waiting for {need}. "
        "Is the matching vLLM server job running and past its startup banner?"
    )


# ── Helpers ─────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _approx_word_count(text: str) -> int:
    return sum(1 for _ in re.finditer(r"\S+", text or ""))


# ── Dataset loading (gold-labelled) ─────────────────────────

def _load_gsm8k(
    *,
    dataset_name: str,
    split: str,
    subset_size: Optional[int],
    prompt_template: str,
) -> List[Dict[str, Any]]:
    """Load GSM8K problems and return a list of evaluation examples."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, "main", split=split)
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        raw_answer = row.get("answer", "")
        m = re.search(r"####\s*([\-\d,\.]+)", raw_answer)
        ref = m.group(1).strip() if m else raw_answer.strip()
        question = row["question"]
        out.append({
            "benchmark": "gsm8k",
            "example_id": str(idx),
            "question": question,
            "reference_answer": ref,
            "prompt_text": prompt_template.replace("{question}", question),
        })
    if subset_size is not None and subset_size > 0:
        out = out[:subset_size]
    logger.info("Loaded GSM8K", extra={"split": split, "examples": len(out)})
    return out


def _load_math(
    *,
    dataset_name: str,
    split: str,
    subset_size: Optional[int],
    prompt_template: str,
) -> List[Dict[str, Any]]:
    """Load MATH-500 problems and return a list of evaluation examples."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        problem = row.get("problem", "")
        solution = row.get("solution", "")
        ref_boxed = extract_boxed_answer(solution)
        ref = ref_boxed if ref_boxed else solution.strip()
        out.append({
            "benchmark": "math",
            "example_id": str(idx),
            "question": problem,
            "reference_answer": ref,
            "reference_is_boxed": ref_boxed is not None,
            "prompt_text": prompt_template.replace("{problem}", problem),
        })
    if subset_size is not None and subset_size > 0:
        out = out[:subset_size]
    logger.info("Loaded MATH-500", extra={"split": split, "examples": len(out)})
    return out


def _build_example_pool(
    benchmarks_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build the union of all enabled benchmark examples."""
    pool: List[Dict[str, Any]] = []
    if benchmarks_cfg.get("gsm8k", {}).get("enabled", False):
        gcfg = benchmarks_cfg["gsm8k"]
        pool.extend(
            _load_gsm8k(
                dataset_name=gcfg.get("dataset_name", "openai/gsm8k"),
                split=gcfg.get("dataset_split", "test"),
                subset_size=gcfg.get("subset_size"),
                prompt_template=gcfg.get(
                    "prompt_template",
                    "Question: {question}\n\nAnswer:",
                ),
            )
        )
    if benchmarks_cfg.get("math", {}).get("enabled", False):
        mcfg = benchmarks_cfg["math"]
        pool.extend(
            _load_math(
                dataset_name=mcfg.get("dataset_name", "HuggingFaceH4/MATH-500"),
                split=mcfg.get("dataset_split", "test"),
                subset_size=mcfg.get("subset_size"),
                prompt_template=mcfg.get(
                    "prompt_template",
                    "Problem: {problem}\n\nSolution:",
                ),
            )
        )
    if not pool:
        raise ValueError(
            "Phase B example pool is empty — enable at least one benchmark "
            "in configs/phase_b.yaml capture.benchmarks."
        )
    return pool


# ── Scoring helpers ─────────────────────────────────────────

def _score_response(
    *,
    benchmark: str,
    response_text: str,
    example: Dict[str, Any],
    answer_extraction_pattern: str,
) -> Tuple[bool, bool, Optional[str], Optional[str]]:
    """Return (correct, scorable, predicted_answer, ambiguity_reason).

    `correct` is False when the answer is unscorable, so all rows are
    eligible as training examples for the quality predictor.
    """
    if benchmark == "gsm8k":
        predicted = extract_numeric_answer(response_text or "", answer_extraction_pattern)
        if predicted is None:
            return False, False, None, "no_extractable_answer_in_response"
        ok = numeric_match(predicted, example["reference_answer"])
        return ok, True, predicted, None

    if benchmark == "math":
        boxed = extract_boxed_answer(response_text or "")
        if boxed is None:
            return False, False, None, "no_boxed_answer_in_response"
        if not example.get("reference_is_boxed", True):
            return False, False, boxed, "reference_not_boxed"
        ok = math_answer_match(boxed, example["reference_answer"])
        return ok, True, boxed, None

    return False, False, None, f"unknown_benchmark:{benchmark}"


# ── Uncertainty extraction from vLLM logprobs ───────────────

def _uncertainty_from_logprobs(
    payload: Optional[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    """Compute avg_logprob / logprob_std / entropy_mean from a vLLM logprobs payload.

    The payload has the OpenAI shape::
        {
            "tokens": [...],
            "token_logprobs": [lp_chosen_per_position, ...],
            "top_logprobs": [{tok: lp, ...}, ...]
        }

    Missing fields → ``None`` (predictor handles via _missing flags).
    """
    out: Dict[str, Optional[float]] = {
        "avg_logprob": None,
        "logprob_std": None,
        "entropy_mean": None,
    }
    if not isinstance(payload, dict):
        return out

    chosen: List[float] = []
    token_logprobs = payload.get("token_logprobs")
    if isinstance(token_logprobs, list):
        for x in token_logprobs:
            if isinstance(x, (int, float)):
                chosen.append(float(x))

    top = payload.get("top_logprobs")
    # Fallback for chosen-token logprob: max per position from top_logprobs.
    if not chosen and isinstance(top, list):
        for pos in top:
            if not isinstance(pos, dict) or not pos:
                continue
            try:
                chosen.append(float(max(pos.values())))
            except (TypeError, ValueError):
                continue

    if chosen:
        arr = np.asarray(chosen, dtype=float)
        out["avg_logprob"] = float(arr.mean())
        if arr.size >= 2:
            out["logprob_std"] = float(arr.std(ddof=0))
        else:
            out["logprob_std"] = 0.0

    if isinstance(top, list) and top:
        entropies: List[float] = []
        for pos in top:
            if not isinstance(pos, dict) or not pos:
                continue
            lps = [float(v) for v in pos.values() if isinstance(v, (int, float))]
            if not lps:
                continue
            # Convert to a probability distribution and renormalise on the
            # captured top-k support (this is the standard top-k entropy proxy).
            ps = [math.exp(lp) for lp in lps]
            z = sum(ps)
            if z <= 0:
                continue
            ps = [p / z for p in ps]
            entropies.append(-sum(p * math.log(p + 1e-12) for p in ps))
        if entropies:
            out["entropy_mean"] = float(np.mean(entropies))

    return out


# ── Single non-streaming request with logprobs ──────────────

async def _send_completion(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    """Issue one non-streaming /v1/completions request and time it."""
    t_start = time.perf_counter()
    error: Optional[str] = None
    status: Optional[int] = None
    text: str = ""
    output_tokens: Optional[int] = None
    logprobs_payload: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None

    try:
        resp = await client.post(url, json=payload, timeout=timeout)
        status = resp.status_code
        try:
            body = resp.json()
        except json.JSONDecodeError:
            body = {}
        if status >= 400:
            snippet = ""
            try:
                snippet = resp.text[:240]
            except Exception:
                snippet = ""
            error = f"http_{status}: {snippet}".strip()
        else:
            choices = body.get("choices") or []
            if choices:
                first = choices[0]
                text = str(first.get("text", "") or "")
                finish_reason = first.get("finish_reason")
                lp = first.get("logprobs")
                if isinstance(lp, dict):
                    logprobs_payload = lp
            usage = body.get("usage") or {}
            ct = usage.get("completion_tokens")
            if isinstance(ct, (int, float)):
                output_tokens = int(ct)
    except Exception as exc:  # pragma: no cover - networking is hard to test
        error = str(exc)

    latency_ms = (time.perf_counter() - t_start) * 1000.0
    return {
        "status": status,
        "latency_ms": latency_ms,
        "ttft_ms": None,                # non-streaming → TTFT not measured
        "output_tokens": output_tokens,
        "response_text": text,
        "logprobs": logprobs_payload,
        "finish_reason": finish_reason,
        "error": error,
    }


# ── Capture session for one (role, rate) pair ───────────────

async def _run_session(
    *,
    role: str,
    model_name: str,
    base_url: str,
    arrival_rate_rps: float,
    examples: List[Dict[str, Any]],
    benchmarks_cfg: Dict[str, Any],
    num_warmup: int,
    max_inflight: int,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    logprobs_top_k: int,
    rng: np.random.Generator,
    metrics_scraper: Optional[_MetricsScraper],
    out_dir: Path,
) -> Dict[str, Any]:
    """Run one Poisson session and write per-rate artefacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    url = f"{base_url.rstrip('/')}/v1/completions"
    sem = asyncio.Semaphore(max_inflight)
    inflight = _InflightTracker()

    if num_warmup > 0:
        await _warm_up(
            url=url, model_name=model_name, examples=examples, rng=rng,
            num=num_warmup, sem=sem, max_tokens=max_tokens,
            temperature=temperature, timeout=timeout_s,
            logprobs_top_k=logprobs_top_k,
        )

    recent: List[float] = []
    recent_window = 50

    # Deterministic shuffle of the example pool for this rate.
    order = list(range(len(examples)))
    rng.shuffle(order)

    results: List[Dict[str, Any]] = []
    t0_mono = time.monotonic()

    async with httpx.AsyncClient(http2=False) as client:
        async def _one_request(req_idx: int, ex_idx: int) -> None:
            example = examples[ex_idx]
            benchmark = example["benchmark"]
            payload: Dict[str, Any] = {
                "model": model_name,
                "prompt": example["prompt_text"],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if logprobs_top_k and logprobs_top_k > 0:
                payload["logprobs"] = int(logprobs_top_k)

            inflight_at_send = await inflight.acquire()
            t_send_mono = time.monotonic()
            t_send_iso = _now_iso()

            metrics_at_send = (
                metrics_scraper.last if metrics_scraper is not None else {}
            )
            recent_p50 = (
                float(np.percentile(recent[-recent_window:], 50)) if recent else None
            )

            async with sem:
                outcome = await _send_completion(client, url, payload, timeout_s)

            await inflight.release()
            if outcome["error"] is None and isinstance(outcome["latency_ms"], (int, float)):
                recent.append(float(outcome["latency_ms"]))

            answer_extraction_pattern = (
                benchmarks_cfg.get(benchmark, {}).get("answer_extraction_pattern", "")
            )
            correct, scorable, predicted, ambiguity = _score_response(
                benchmark=benchmark,
                response_text=outcome["response_text"] or "",
                example=example,
                answer_extraction_pattern=str(answer_extraction_pattern),
            )

            uncertainty = _uncertainty_from_logprobs(outcome.get("logprobs"))

            results.append({
                "req_idx": req_idx,
                "example_idx": ex_idx,
                "benchmark": benchmark,
                "example_id": example["example_id"],
                "prompt_text": example["prompt_text"],
                "input_word_count": _approx_word_count(example["prompt_text"]),
                "reference_answer": example["reference_answer"],
                "predicted_answer": predicted,
                "scorable": scorable,
                "correct": bool(correct),
                "ambiguity_reason": ambiguity,
                "t_send_iso": t_send_iso,
                "t_send_monotonic": t_send_mono,
                "status": outcome["status"],
                "latency_ms": outcome["latency_ms"],
                "ttft_ms": outcome["ttft_ms"],
                "output_tokens": outcome["output_tokens"],
                "response_text": outcome["response_text"],
                "finish_reason": outcome["finish_reason"],
                "error": outcome["error"],
                "avg_logprob": uncertainty["avg_logprob"],
                "logprob_std": uncertainty["logprob_std"],
                "entropy_mean": uncertainty["entropy_mean"],
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
        for i, ex_idx in enumerate(order):
            tasks.append(asyncio.create_task(_one_request(i, ex_idx)))
            interval = float(rng.exponential(1.0 / arrival_rate_rps))
            await asyncio.sleep(interval)

        await asyncio.gather(*tasks, return_exceptions=False)

    t_end_mono = time.monotonic()
    wall_s = max(t_end_mono - t0_mono, 1e-9)

    ok = [r for r in results if r["error"] is None]
    latencies = [r["latency_ms"] for r in ok if r["latency_ms"] is not None]
    out_tokens = [int(r["output_tokens"]) for r in ok if isinstance(r["output_tokens"], (int, float))]
    correct_total = sum(1 for r in results if r.get("correct"))
    scorable_total = sum(1 for r in results if r.get("scorable"))

    summary = {
        "role": role,
        "model_name": model_name,
        "arrival_rate_rps": arrival_rate_rps,
        "num_requests": len(results),
        "successful_requests": len(ok),
        "failed_requests": len(results) - len(ok),
        "scorable_requests": scorable_total,
        "correct_requests": correct_total,
        "accuracy_total_pct": (correct_total / max(len(results), 1)) * 100.0,
        "accuracy_scorable_pct": (
            (correct_total / scorable_total * 100.0) if scorable_total else 0.0
        ),
        "total_output_tokens": sum(out_tokens),
        "wall_clock_seconds": wall_s,
        "achieved_request_rps": len(results) / wall_s,
        **{f"latency_{k}": v for k, v in summarise_latencies(latencies).items()},
    }

    save_json(results, out_dir / "raw_requests.json")
    save_json(summary, out_dir / "summary.json")
    save_csv(
        [
            {k: v for k, v in r.items() if k not in {"response_text", "prompt_text"}}
            for r in results
        ],
        out_dir / "raw_requests.csv",
    )

    return {
        "summary": summary,
        "results": results,
    }


async def _warm_up(
    *,
    url: str,
    model_name: str,
    examples: List[Dict[str, Any]],
    rng: np.random.Generator,
    num: int,
    sem: asyncio.Semaphore,
    max_tokens: int,
    temperature: float,
    timeout: float,
    logprobs_top_k: int,
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
            if logprobs_top_k and logprobs_top_k > 0:
                payload["logprobs"] = int(logprobs_top_k)
            async with sem:
                await _send_completion(client, url, payload, timeout)

        tasks = []
        for _ in range(num):
            ex = examples[int(rng.integers(0, len(examples)))]
            tasks.append(asyncio.create_task(_one(ex["prompt_text"])))
        await asyncio.gather(*tasks, return_exceptions=True)


# ── Trace-file emission (ModelExecutionTrace schema) ────────

def _emit_trace_jsonl(
    *,
    role: str,
    model_name: str,
    run_id: str,
    rows: List[Dict[str, Any]],
    arrival_rate_rps: float,
    out_path: Path,
) -> int:
    """Materialise rows into the predictors/ trace schema (one record per request)."""
    out_records: List[Dict[str, Any]] = []
    for r in rows:
        running = _none_to_float(r.get("z_metrics_running"))
        waiting = _none_to_float(r.get("z_metrics_waiting"))
        # Prefer the engine-side `running` count for active_workers; same
        # convention as Phase A.
        active_workers: Optional[float] = running

        record = {
            "query_id": f"phase_b:{r['benchmark']}:{r['example_id']}:{role}",
            "benchmark": str(r["benchmark"]),
            "model_name": model_name,
            "run_id": run_id,
            "timestamp_utc": r.get("t_send_iso"),
            "source_file": str(out_path),
            "source_record_index": int(r["req_idx"]),
            "model_tier": role,
            "example_id": str(r.get("example_id", "")),
            "prompt_text": r.get("prompt_text"),
            "response_text": r.get("response_text"),
            "input_tokens": int(r.get("input_word_count") or 0) or None,
            "output_tokens": _none_to_int(r.get("output_tokens")),
            "latency_ms": _none_to_float(r.get("latency_ms")),
            "ttft_ms": _none_to_float(r.get("ttft_ms")),
            "correct": bool(r.get("correct")),
            "score": 1.0 if r.get("correct") else 0.0,
            "system_state": {
                "queue_depth": waiting,
                "pending_requests": _none_to_float(r.get("z_inflight_at_send")),
                # vLLM exposes generation throughput in tokens/s; we keep the
                # field name for schema compatibility but interpret it as a
                # recent throughput proxy (not strict req/s).
                "throughput_rps_recent": _none_to_float(r.get("z_metrics_gen_tps")),
                "active_workers": active_workers,
            },
            "resources": {
                "gpu_seconds": _none_to_float(r.get("gpu_seconds_proxy")),
                "energy_joules": None,
                "gpu_utilization_pct": _none_to_float(r.get("gpu_utilization_pct_mean")),
            },
            "uncertainty": {
                "avg_logprob": _none_to_float(r.get("avg_logprob")),
                "logprob_std": _none_to_float(r.get("logprob_std")),
                "entropy_mean": _none_to_float(r.get("entropy_mean")),
            },
            "tags": {
                "phase": "B",
                "request": {
                    "max_tokens": None,
                    "temperature": None,
                },
                "scoring": {
                    "scorable": bool(r.get("scorable")),
                    "predicted_answer": r.get("predicted_answer"),
                    "reference_answer": r.get("reference_answer"),
                    "ambiguity_reason": r.get("ambiguity_reason"),
                },
                "telemetry": {
                    "online": {
                        "lambda_target_rps": arrival_rate_rps,
                        "achieved_rps": None,
                    },
                    "engine": {
                        "running_mean": running,
                        "waiting_mean": waiting,
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
) -> Path:
    """Execute the Phase B quality capture for one role across all configured rates."""
    cfg = load_yaml(config_path)
    common = cfg.get("common", {})
    capt = cfg.get("capture", {})
    samp_cfg = (capt.get("samplers") or {})
    benchmarks_cfg = capt.get("benchmarks", {})

    seed = int(common.get("seed", 42))
    set_seed(seed)
    setup_logging()

    require_gpu_sidecar = bool((samp_cfg.get("gpu_smi") or {}).get("enabled", True))
    if not base_url_override:
        _wait_for_routing_endpoint_files(
            role,
            require_gpu_publish=require_gpu_sidecar,
            timeout_s=float(os.environ.get("PHASE_B_ENDPOINT_WAIT_S", "900")),
        )

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

    examples = _build_example_pool(benchmarks_cfg)
    if not examples:
        raise ValueError("Phase B example pool is empty after filtering.")

    rates: List[float] = [float(x) for x in capt.get("arrival_rates_rps", [10])]
    num_warmup = int(capt.get("num_warmup_requests", 30))
    max_inflight = int(capt.get("max_inflight", 64))
    max_tokens = int(capt.get("max_tokens", 512))
    temperature = float(capt.get("temperature", 0.0))
    timeout_s = float(capt.get("request_timeout_s", 180.0))
    logprobs_top_k = int(capt.get("logprobs_top_k", 5))

    base_dir = common.get("results_base_dir", "results/phase_b")
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
        "phase": "B",
        "examples_total": len(examples),
        "examples_per_benchmark": {
            b: sum(1 for e in examples if e["benchmark"] == b) for b in {e["benchmark"] for e in examples}
        },
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
                    "Phase B session start",
                    extra={"role": role, "rate_rps": rate, "num_examples": len(examples)},
                )
                rate_dir = run_dir / f"rate_{rate}"
                session = await _run_session(
                    role=role,
                    model_name=model_name,
                    base_url=base_url,
                    arrival_rate_rps=rate,
                    examples=examples,
                    benchmarks_cfg=benchmarks_cfg,
                    num_warmup=num_warmup,
                    max_inflight=max_inflight,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout_s=timeout_s,
                    logprobs_top_k=logprobs_top_k,
                    rng=rng,
                    metrics_scraper=metrics_sidecar,
                    out_dir=rate_dir,
                )
                summaries.append(session["summary"])
                all_rows.extend(session["results"])
                logger.info(
                    "Phase B session done",
                    extra={
                        "role": role,
                        "rate_rps": rate,
                        "ok": session["summary"]["successful_requests"],
                        "correct": session["summary"]["correct_requests"],
                    },
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
    gpu_jsonl_to_use: Optional[Path] = None
    if use_external_gpu and server_gpu_path is not None and server_gpu_path.exists():
        gpu_jsonl_to_use = server_gpu_path
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

    # Group rows by benchmark so query_ids are stable per workload.
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_rows:
        grouped.setdefault(str(r["benchmark"]), []).append(r)

    # Re-index req_idx within each group for stable trace records, then
    # consolidate into a single trace.jsonl that the dataset glob picks up.
    n_records = 0
    with trace_path.open("w", encoding="utf-8") as out_fh:
        for benchmark_label, rows in grouped.items():
            partial = run_dir / f"_partial_{benchmark_label}.jsonl"
            for new_idx, row in enumerate(rows):
                row["req_idx"] = new_idx
            n_records += _emit_trace_jsonl(
                role=role,
                model_name=model_name,
                run_id=run_id,
                rows=rows,
                arrival_rate_rps=float(rows[0]["z_lambda_target_rps"]) if rows else 0.0,
                out_path=partial,
            )
            out_fh.write(partial.read_text(encoding="utf-8"))
            partial.unlink(missing_ok=True)

    logger.info(
        "Phase B capture complete",
        extra={
            "run_dir": str(run_dir),
            "records_written": n_records,
            "successful_sessions": len([s for s in summaries if s["successful_requests"] > 0]),
        },
    )

    # ── Post-capture validator (fails loudly on bad data) ───
    _validate_quality_capture(
        trace_path=trace_path,
        gpu_summary=gpu_summary,
        run_dir=run_dir,
        require_gpu=bool((samp_cfg.get("gpu_smi") or {}).get("enabled", True)),
        require_metrics=bool((samp_cfg.get("server_metrics") or {}).get("enabled", True)),
        require_logprobs=int(capt.get("logprobs_top_k", 5)) > 0,
    )
    return run_dir


def _validate_quality_capture(
    *,
    trace_path: Path,
    gpu_summary: Optional[Dict[str, Any]],
    run_dir: Path,
    require_gpu: bool,
    require_metrics: bool,
    require_logprobs: bool,
) -> None:
    """Walk the Phase B trace and ensure key fields are populated.

    Writes ``telemetry_validation.json`` for diagnostics, and raises
    ``RuntimeError`` if a required signal is missing — so SLURM marks the
    capture as FAILED and we do NOT silently re-train on empty data.
    """
    n_total = 0
    n_correct_present = 0
    n_correct_true = 0
    n_response_nonempty = 0
    n_running_nonzero = 0
    n_waiting_nonnull = 0
    n_gen_tps_nonnull = 0
    n_kv_nonnull = 0
    n_gpu_util_nonzero = 0
    n_logprob_nonnull = 0
    n_entropy_nonnull = 0
    benchmark_counter: Dict[str, int] = {}

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
            benchmark_counter[str(rec.get("benchmark", ""))] = (
                benchmark_counter.get(str(rec.get("benchmark", "")), 0) + 1
            )
            if rec.get("correct") is not None:
                n_correct_present += 1
                if bool(rec.get("correct")):
                    n_correct_true += 1
            if rec.get("response_text"):
                n_response_nonempty += 1

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
            unc = rec.get("uncertainty") or {}
            if unc.get("avg_logprob") is not None:
                n_logprob_nonnull += 1
            if unc.get("entropy_mean") is not None:
                n_entropy_nonnull += 1

    accuracy_pct = (n_correct_true / n_total * 100.0) if n_total else 0.0
    report = {
        "trace_rows": n_total,
        "rows_per_benchmark": benchmark_counter,
        "correct_present": n_correct_present,
        "correct_true": n_correct_true,
        "accuracy_pct": accuracy_pct,
        "response_nonempty": n_response_nonempty,
        "running_nonzero": n_running_nonzero,
        "waiting_nonnull": n_waiting_nonnull,
        "gen_tps_nonnull": n_gen_tps_nonnull,
        "kv_nonnull": n_kv_nonnull,
        "gpu_util_nonzero": n_gpu_util_nonzero,
        "logprob_nonnull": n_logprob_nonnull,
        "entropy_nonnull": n_entropy_nonnull,
        "gpu_summary": gpu_summary,
    }
    (run_dir / "telemetry_validation.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    errors: List[str] = []
    if n_total == 0:
        errors.append("trace.jsonl is empty.")
    if n_correct_present < n_total:
        errors.append(
            f"correct flag missing on {n_total - n_correct_present} rows — scoring failed."
        )
    if n_response_nonempty == 0:
        errors.append("All response_text fields are empty — server returned nothing.")
    if require_metrics:
        if n_running_nonzero == 0:
            errors.append(
                "active_workers (vllm running) is always zero — server /metrics not "
                "scraped correctly."
            )
        if n_gen_tps_nonnull == 0 and n_kv_nonnull == 0:
            errors.append(
                "Both generation throughput and KV usage are missing — /metrics "
                "scrape produced nothing useful."
            )
    if require_gpu and n_gpu_util_nonzero == 0:
        errors.append(
            "GPU utilization is zero across the entire trace — capture node "
            "probably has no visible GPU (must be co-located with the server "
            "node, OR server-side gpu_sampler must publish its path)."
        )
    if require_logprobs and n_logprob_nonnull == 0:
        errors.append(
            "uncertainty.avg_logprob is null on every row — vLLM did not return "
            "logprobs (check server build / payload). Without logprobs the "
            "quality_post_hoc predictor cannot use uncertainty features."
        )

    if errors:
        raise RuntimeError(
            "Phase B telemetry validation FAILED:\n  - " + "\n  - ".join(errors)
            + f"\nSee {run_dir / 'telemetry_validation.json'}"
        )


# ── CLI ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase B — Poisson quality capture")
    parser.add_argument("--config", default="configs/phase_b.yaml")
    parser.add_argument(
        "--role",
        required=True,
        help=(
            "Role key in configs/models.yaml "
            "(teacher | student_mid | student_q3b | student_small | student_tiny)"
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override base URL (else read from results/routing/endpoints/<role>.url)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model name (else taken from configs/models.yaml[role].name)",
    )
    args = parser.parse_args()

    run(
        config_path=args.config,
        role=args.role,
        base_url_override=args.base_url,
        model_override=args.model,
    )


if __name__ == "__main__":
    main()
