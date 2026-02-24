"""
bench/run_online_load.py
========================
Online load benchmark — async HTTP client that measures:

  * **TTFT** (time to first token)
  * **End-to-end latency** with p50, p95, p99
  * **Effective throughput** (tokens / wall-clock second)

The benchmark sweeps over one or more request rates defined in
``configs/benchmark.yaml`` → ``online.request_rates``.

Usage
-----
    python -m bench.run_online_load [--config configs/benchmark.yaml]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from bench.metrics import save_csv, save_json, summarise_latencies
from utils.config_loader import load_yaml
from utils.logging import setup_logging, get_logger
from utils.reproducibility import (
    collect_metadata, make_run_dir, save_metadata, set_seed, snapshot_configs,
)

logger = get_logger(__name__)


# ── Single request ──────────────────────────────────────────

async def _send_request(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    """Send one completion request and measure timing."""
    t_start = time.perf_counter()
    ttfb: Optional[float] = None
    chunks: list[bytes] = []
    tokens_received = 0

    try:
        async with client.stream(
            "POST", url, json=payload, timeout=timeout,
        ) as resp:
            async for chunk in resp.aiter_bytes():
                if ttfb is None:
                    ttfb = (time.perf_counter() - t_start) * 1000.0  # ms
                chunks.append(chunk)

        body_text = b"".join(chunks).decode("utf-8", errors="replace")
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        # Attempt to count output tokens from OpenAI-compat usage field
        try:
            body_json = json.loads(body_text)
            tokens_received = (
                body_json.get("usage", {}).get("completion_tokens", 0)
            )
        except json.JSONDecodeError:
            tokens_received = 0

        return {
            "status": resp.status_code,
            "latency_ms": latency_ms,
            "ttfb_ms": ttfb,
            "output_tokens": tokens_received,
            "error": None,
        }

    except Exception as exc:
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "status": None,
            "latency_ms": latency_ms,
            "ttfb_ms": None,
            "output_tokens": 0,
            "error": str(exc),
        }


# ── Load driver ─────────────────────────────────────────────

async def _run_at_rate(
    prompts: List[str],
    base_url: str,
    model: str,
    request_rate: float,
    num_requests: int,
    max_tokens: int,
    temperature: float,
    timeout: float,
    concurrency: int,
) -> List[Dict[str, Any]]:
    """Issue *num_requests* at approximately *request_rate* req/s."""
    url = f"{base_url.rstrip('/')}/v1/completions"
    sem = asyncio.Semaphore(concurrency)
    results: List[Dict[str, Any]] = []
    interval = 1.0 / request_rate if request_rate > 0 else 0.0

    async def _worker(prompt: str) -> None:
        async with sem:
            payload = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            async with httpx.AsyncClient() as client:
                res = await _send_request(client, url, payload, timeout)
            results.append(res)

    tasks: list[asyncio.Task] = []
    for i in range(num_requests):
        prompt = prompts[i % len(prompts)]
        tasks.append(asyncio.create_task(_worker(prompt)))
        if interval > 0:
            await asyncio.sleep(interval)

    await asyncio.gather(*tasks)
    return results


# ── Orchestrator ────────────────────────────────────────────

def run(config_path: str = "configs/benchmark.yaml") -> Path:
    cfg = load_yaml(config_path)
    common = cfg.get("common", {})
    ocfg = cfg.get("online", {})
    seed = common.get("seed", 42)
    set_seed(seed)
    setup_logging()

    run_dir = make_run_dir(
        common.get("results_base_dir", "results") + "/online",
        tag="online",
    )
    snapshot_configs([config_path, "configs/serving.yaml"], run_dir)
    meta = collect_metadata(seed, cfg)
    save_metadata(meta, run_dir)

    # Load prompts
    prompts_file = ocfg.get("prompts_file", "configs/prompts.jsonl")
    prompts: list[str] = []
    with open(prompts_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                prompts.append(obj.get("prompt", line))
            except json.JSONDecodeError:
                prompts.append(line)
    if not prompts:
        raise ValueError(f"No prompts loaded from {prompts_file}")

    base_url = common.get("base_url", "http://localhost:8000")
    model = common.get("model", "")
    request_rates = ocfg.get("request_rates", [10])
    num_requests = ocfg.get("num_requests", 500)
    max_tokens = ocfg.get("max_tokens", 256)
    temperature = ocfg.get("temperature", 0.0)
    timeout = ocfg.get("timeout_seconds", 120)
    concurrency = ocfg.get("concurrency", 64)

    all_summaries: list[dict] = []

    for rate in request_rates:
        logger.info("Starting online benchmark",
                     extra={"request_rate": rate, "num_requests": num_requests})

        raw = asyncio.run(_run_at_rate(
            prompts, base_url, model, rate, num_requests,
            max_tokens, temperature, timeout, concurrency,
        ))

        ok = [r for r in raw if r["error"] is None]
        latencies = [r["latency_ms"] for r in ok]
        ttfbs = [r["ttfb_ms"] for r in ok if r["ttfb_ms"] is not None]
        total_tokens = sum(r["output_tokens"] for r in ok)
        wall_s = (max(r["latency_ms"] for r in raw) / 1000.0) if raw else 0.0

        summary = {
            "request_rate": rate,
            "total_requests": len(raw),
            "successful_requests": len(ok),
            "failed_requests": len(raw) - len(ok),
            "total_output_tokens": total_tokens,
            "effective_throughput_tps": total_tokens / wall_s if wall_s > 0 else 0,
            **{f"latency_{k}": v for k, v in summarise_latencies(latencies).items()},
            **{f"ttfb_{k}": v for k, v in summarise_latencies(ttfbs).items()},
        }
        all_summaries.append(summary)

        # Save per-rate raw data
        rate_dir = run_dir / f"rate_{rate}"
        rate_dir.mkdir(parents=True, exist_ok=True)
        save_json(raw, rate_dir / "raw_results.json")
        save_json(summary, rate_dir / "summary.json")

    # Aggregate CSV across rates
    save_json(all_summaries, run_dir / "online_results.json")
    save_csv(all_summaries, run_dir / "online_results.csv")
    logger.info("Online benchmark complete", extra={"run_dir": str(run_dir)})

    return run_dir


# ── CLI ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Online load benchmark")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
