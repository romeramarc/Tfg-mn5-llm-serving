"""
routing/run_eval_holdout.py
===========================
Online evaluation of routing systems on a shared holdout prompt pool
under Poisson open-loop load (configs/routing_eval_holdout.yaml).

Usage (MN5, endpoints already published):
    python -m bench.holdout_pool --config configs/routing_eval_holdout.yaml
    python -m routing.run_eval_holdout \\
        --config configs/routing_eval_holdout.yaml \\
        --system sysD_cascade_distilled
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np

from bench.holdout_pool import export_prompt_pool, load_prompt_pool
from bench.metrics import save_csv, save_json, summarise_latencies
from bench.run_load_capture import _GpuSidecar, _InflightTracker, _MetricsScraper
from bench.run_quality_capture import _score_response, _wait_for_routing_endpoint_files
from routing.endpoints import resolve_endpoints
from routing.policies import POLICIES, RoutingDecision
from routing.predictor_runtime import EvalPredictorSuite
from utils.config_loader import load_yaml
from utils.logging import get_logger, setup_logging
from utils.reproducibility import (
    collect_metadata,
    make_run_dir,
    save_metadata,
    set_seed,
    snapshot_configs,
)

logger = get_logger(__name__)

_POLICY_NEEDS_PREDICTORS = {
    "routing_predictive",
    "cascade_five_rung",
    "routing_plus_cascade",
}


def _find_system(cfg: Dict[str, Any], system_id: str) -> Dict[str, Any]:
    for node in cfg.get("systems") or []:
        if node.get("id") == system_id:
            return node
    raise KeyError(f"Unknown system id '{system_id}'")


def _merge_policy_ctx(
    base: Dict[str, Any],
    policy_name: str,
    policy_params: Dict[str, Any],
) -> Dict[str, Any]:
    ctx = dict(base)
    params = (policy_params or {}).get(policy_name) or {}
    ctx.update(params)
    if policy_name == "routing_plus_cascade":
        rp = (policy_params or {}).get("routing_predictive") or {}
        for key in ("cost_weight_lambda", "candidate_rungs", "min_quality_floor"):
            ctx.setdefault(key, rp.get(key))
    return ctx


async def _run_poisson_session(
    *,
    examples: List[Dict[str, Any]],
    policy_fn,
    ctx: Dict[str, Any],
    benchmarks_cfg: Dict[str, Any],
    arrival_rate_rps: float,
    num_warmup: int,
    max_inflight: int,
    rng: np.random.Generator,
    metrics_scraper: Optional[_MetricsScraper],
) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(max_inflight)
    inflight = _InflightTracker()
    recent: List[float] = []
    recent_window = 50

    order = list(range(len(examples)))
    rng.shuffle(order)

    records: List[Dict[str, Any]] = []

    async with httpx.AsyncClient(http2=False) as client:

        async def _handle(req_idx: int, ex_idx: int) -> None:
            example = examples[ex_idx]
            benchmark = str(example["benchmark"])
            prompt = example["prompt_text"]

            inflight_at_send = await inflight.acquire()
            t0 = time.perf_counter()

            z_metrics = metrics_scraper.last if metrics_scraper is not None else {}
            recent_p50 = (
                float(np.percentile(recent[-recent_window:], 50)) if recent else None
            )

            request_ctx = dict(ctx)
            request_ctx["request_id"] = example["request_id"]
            request_ctx["prompt_metadata"] = {
                "benchmark": benchmark,
                "example_id": example["example_id"],
                "length_bucket": example.get("length_bucket"),
                "prompt_char_len": example.get("prompt_char_len"),
            }
            request_ctx["z_metrics"] = z_metrics
            request_ctx["inflight_at_send"] = inflight_at_send
            request_ctx["recent_p50_latency_ms"] = recent_p50

            async with sem:
                decision: RoutingDecision = await policy_fn(client, prompt, request_ctx)

            await inflight.release()
            wall_ms = (time.perf_counter() - t0) * 1000.0
            if decision.latency_ms:
                recent.append(float(decision.latency_ms))

            bench_key = "gsm8k_holdout" if benchmark == "gsm8k" else "hendrycks_math_holdout"
            pattern = str(
                (benchmarks_cfg.get(bench_key) or {}).get("answer_extraction_pattern", "")
            )

            correct, scorable, predicted, ambiguity = _score_response(
                benchmark=benchmark,
                response_text=decision.response_text or "",
                example=example,
                answer_extraction_pattern=pattern,
            )

            n_attempts = len(decision.attempts or [])
            records.append({
                "request_id": example["request_id"],
                "pool_index": example.get("pool_index"),
                "benchmark": benchmark,
                "example_id": example["example_id"],
                "length_bucket": example.get("length_bucket"),
                "prompt_char_len": example.get("prompt_char_len"),
                "arrival_rate_rps": arrival_rate_rps,
                "req_idx": req_idx,
                "policy": ctx.get("policy_name"),
                "system_id": ctx.get("system_id"),
                "selected_model": decision.selected_model,
                "reason": decision.reason,
                "confidence": decision.confidence,
                "latency_ms": decision.latency_ms,
                "client_wall_ms": wall_ms,
                "num_attempts": n_attempts,
                "correct": bool(correct),
                "scorable": bool(scorable),
                "predicted_answer": predicted,
                "ambiguity_reason": ambiguity,
                "metadata": decision.metadata,
            })

        tasks: List[asyncio.Task[None]] = []
        for i, ex_idx in enumerate(order):
            tasks.append(asyncio.create_task(_handle(i, ex_idx)))
            await asyncio.sleep(float(rng.exponential(1.0 / arrival_rate_rps)))
        await asyncio.gather(*tasks)

    return records


def _build_summary(records: List[Dict[str, Any]], *, system_id: str, policy: str) -> Dict[str, Any]:
    latencies = [float(r["latency_ms"]) for r in records if r.get("latency_ms")]
    correct = sum(1 for r in records if r.get("correct"))
    scorable = sum(1 for r in records if r.get("scorable"))
    reason_counts = Counter(str(r.get("reason")) for r in records)
    bench_correct: Dict[str, List[bool]] = {}
    for r in records:
        if r.get("scorable"):
            bench_correct.setdefault(str(r["benchmark"]), []).append(bool(r["correct"]))

    return {
        "system_id": system_id,
        "policy": policy,
        "total_requests": len(records),
        "scorable_requests": scorable,
        "correct_requests": correct,
        "accuracy_total_pct": (correct / max(len(records), 1)) * 100.0,
        "accuracy_scorable_pct": (correct / scorable * 100.0) if scorable else 0.0,
        "accuracy_by_benchmark": {
            b: (sum(v) / len(v) * 100.0) if v else 0.0 for b, v in bench_correct.items()
        },
        "length_bucket_counts": dict(Counter(str(r.get("length_bucket")) for r in records)),
        "mean_attempts": float(np.mean([r.get("num_attempts", 1) for r in records])),
        "reason_counts": dict(reason_counts),
        **{f"latency_{k}": v for k, v in summarise_latencies(latencies).items()},
    }


def run(
    *,
    config_path: str,
    system_id: str,
    prompt_pool_path: Optional[str] = None,
) -> Path:
    cfg = load_yaml(config_path)
    common = cfg.get("common") or {}
    seed = int(common.get("seed", 42))
    set_seed(seed)
    setup_logging()

    system = _find_system(cfg, system_id)
    policy_name = str(system["policy"])
    if policy_name not in POLICIES:
        raise ValueError(f"Unknown policy '{policy_name}'")

    pool_path = Path(
        prompt_pool_path
        or (cfg.get("execution_plan") or {}).get("shared_prompt_pool")
        or "results/routing_eval_holdout/prompt_pool.jsonl"
    )
    if not pool_path.is_file():
        export_prompt_pool(cfg, pool_path, seed=seed)
    examples = load_prompt_pool(pool_path)

    exec_cfg = cfg.get("execution_plan") or {}
    endpoint_dir = str(exec_cfg.get("endpoint_dir", "results/routing/endpoints"))
    roles = list(system.get("roles") or [])
    for role in roles:
        _wait_for_routing_endpoint_files(
            role,
            require_gpu_publish=False,
            timeout_s=float(exec_cfg.get("endpoint_wait_s", 900.0)),
        )
    endpoints = resolve_endpoints(roles, endpoint_dir=endpoint_dir)

    workload = cfg.get("workload") or {}
    policy_params = cfg.get("policy_params") or {}

    predictor_suite = None
    if policy_name in _POLICY_NEEDS_PREDICTORS:
        pred_cfg = cfg.get("predictors") or {}
        predictor_suite = EvalPredictorSuite.from_config(pred_cfg)
        if policy_name == "cascade_five_rung":
            policy_params.setdefault("cascade_five_rung", {})["post_hoc_threshold"] = (
                predictor_suite.post_hoc_threshold
            )

    run_dir = make_run_dir(
        common.get("results_base_dir", "results/routing_eval_holdout"),
        tag=f"{system_id}-{policy_name}",
    )
    snapshot_configs([config_path], run_dir)
    save_metadata(
        collect_metadata(seed, {"system_id": system_id, "policy": policy_name, **cfg}),
        run_dir,
    )
    (run_dir / "prompt_pool_path.txt").write_text(str(pool_path.resolve()), encoding="utf-8")

    metrics_url = endpoints[roles[0]]["base_url"]
    scraper_cfg = (workload.get("samplers") or {}).get("server_metrics") or {}
    metrics_scraper: Optional[_MetricsScraper] = None
    if scraper_cfg.get("enabled", True):
        metrics_scraper = _MetricsScraper(
            base_url=metrics_url,
            interval_s=float(scraper_cfg.get("interval_s", 0.25)),
            output=run_dir / "server_metrics.jsonl",
        )

    base_ctx: Dict[str, Any] = {
        "endpoints": endpoints,
        "max_tokens": int(workload.get("max_tokens", 512)),
        "temperature": float(workload.get("temperature", 0.0)),
        "request_timeout_s": float(workload.get("request_timeout_s", 180.0)),
        "logprobs_top_k": int(workload.get("logprobs_top_k", 5)),
        "policy_name": policy_name,
        "system_id": system_id,
        "predictor_suite": predictor_suite,
    }
    ctx = _merge_policy_ctx(base_ctx, policy_name, policy_params)
    policy_fn = POLICIES[policy_name]

    rng = np.random.default_rng(seed)
    rates = [float(x) for x in workload.get("arrival_rates_rps", [5])]

    async def _run_all() -> List[Dict[str, Any]]:
        if metrics_scraper is not None:
            metrics_scraper.start()
        out: List[Dict[str, Any]] = []
        try:
            for rate in rates:
                session_dir = run_dir / f"lambda_{rate:g}"
                session_dir.mkdir(parents=True, exist_ok=True)
                recs = await _run_poisson_session(
                    examples=examples,
                    policy_fn=policy_fn,
                    ctx=ctx,
                    benchmarks_cfg=cfg.get("benchmarks") or {},
                    arrival_rate_rps=rate,
                    num_warmup=int(workload.get("num_warmup_requests", 15)),
                    max_inflight=int(workload.get("max_inflight", 48)),
                    rng=rng,
                    metrics_scraper=metrics_scraper,
                )
                out.extend(recs)
                sess_summary = _build_summary(recs, system_id=system_id, policy=policy_name)
                sess_summary["arrival_rate_rps"] = rate
                save_json(recs, session_dir / "per_request.json")
                save_json(sess_summary, session_dir / "summary.json")
        finally:
            if metrics_scraper is not None:
                await metrics_scraper.stop()
        return out

    all_records = asyncio.run(_run_all())

    summary = _build_summary(all_records, system_id=system_id, policy=policy_name)
    summary["prompt_pool"] = str(pool_path)
    summary["prompt_count"] = len(examples)
    out_cfg = cfg.get("output") or {}
    save_json(all_records, run_dir / out_cfg.get("per_request_jsonl", "per_request.json"))
    save_json(summary, run_dir / out_cfg.get("summary_json", "summary.json"))
    save_csv(all_records, run_dir / "per_request.csv")
    save_csv([summary], run_dir / "summary.csv")
    logger.info("Holdout eval complete", extra={"run_dir": str(run_dir), "summary": summary})
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run holdout routing evaluation")
    parser.add_argument("--config", default="configs/routing_eval_holdout.yaml")
    parser.add_argument("--system", required=True, help="system.id from config")
    parser.add_argument("--prompt-pool", default=None)
    args = parser.parse_args()
    run(config_path=args.config, system_id=args.system, prompt_pool_path=args.prompt_pool)


if __name__ == "__main__":
    main()
