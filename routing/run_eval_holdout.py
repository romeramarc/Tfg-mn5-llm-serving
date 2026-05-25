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
import json
import time
from collections import Counter, defaultdict
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


def _build_holdout_request_record(
    *,
    example: Dict[str, Any],
    benchmark: str,
    arrival_rate_rps: float,
    req_idx: int,
    ctx: Dict[str, Any],
    decision: RoutingDecision,
    wall_ms: float,
    inflight_at_send: Any,
    recent_p50: Optional[float],
    z_metrics: Dict[str, Any],
    correct: bool,
    scorable: bool,
    predicted: Any,
    ambiguity: Optional[str],
) -> Dict[str, Any]:
    """One row for per_request.jsonl / CSV (nested structures in JSON fields for CSV)."""
    meta = dict(decision.metadata or {})
    attempts = list(decision.attempts or [])
    total_out = sum(int(a.get("output_tokens") or 0) for a in attempts)
    first_stage = str(attempts[0].get("stage", "")) if attempts else ""
    return {
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
        "num_attempts": len(attempts),
        "correct": bool(correct),
        "scorable": bool(scorable),
        "predicted_answer": predicted,
        "ambiguity_reason": ambiguity,
        "inflight_at_send": inflight_at_send,
        "recent_p50_latency_ms": recent_p50,
        "z_metrics_at_send": dict(z_metrics) if z_metrics else {},
        "route_path": meta.get("route_path"),
        "entry_rung": meta.get("entry_rung"),
        "selected_rung": meta.get("selected_rung"),
        "routing_scores": meta.get("routing_scores"),
        "routing_selection": meta.get("routing_selection"),
        "post_hoc_probability": meta.get("post_hoc_probability"),
        "first_attempt_stage": first_stage,
        "total_output_tokens": total_out,
        "response_char_len": len(decision.response_text or ""),
        "attempts": attempts,
        "metadata": meta,
    }


def _flatten_holdout_rows_for_csv(
    records: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[str]]:
    """JSON-serialize dict/list columns so CSV is rectangular and portable."""
    flat_rows: List[Dict[str, Any]] = []
    all_keys: set[str] = set()
    for r in records:
        flat: Dict[str, Any] = {}
        for k, v in r.items():
            if isinstance(v, (dict, list)):
                flat[k] = json.dumps(v, ensure_ascii=False, default=str)
            elif v is None:
                flat[k] = ""
            else:
                flat[k] = v
            all_keys.add(k)
        flat_rows.append(flat)
    keys_sorted = sorted(all_keys)
    for flat in flat_rows:
        for k in keys_sorted:
            flat.setdefault(k, "")
    ordered = [{k: flat[k] for k in keys_sorted} for flat in flat_rows]
    return ordered, keys_sorted


def _flatten_summary_for_csv(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Single-row summary: nested aggregates as JSON strings for spreadsheet tools."""
    row: Dict[str, Any] = {}
    for k, v in summary.items():
        if isinstance(v, (dict, list)):
            row[k] = json.dumps(v, ensure_ascii=False, default=str)
        elif v is None:
            row[k] = ""
        else:
            row[k] = v
    return row

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
    system_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ctx = dict(base)
    params = (policy_params or {}).get(policy_name) or {}
    ctx.update(params)
    if policy_name == "routing_plus_cascade":
        rp = (policy_params or {}).get("routing_predictive") or {}
        for key in (
            "cost_weight_lambda",
            "candidate_rungs",
            "routing_candidates",
            "min_quality_floor",
            "routing_selection",
        ):
            ctx.setdefault(key, rp.get(key))
    if system_overrides:
        override = system_overrides.get(policy_name) or {}
        if isinstance(override, dict):
            ctx.update(override)
        if policy_name == "routing_plus_cascade":
            rp_override = system_overrides.get("routing_predictive") or {}
            if isinstance(rp_override, dict):
                for key in (
                    "cost_weight_lambda",
                    "candidate_rungs",
                    "routing_candidates",
                    "min_quality_floor",
                    "routing_selection",
                ):
                    if key in rp_override:
                        ctx[key] = rp_override[key]
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

            records.append(
                _build_holdout_request_record(
                    example=example,
                    benchmark=benchmark,
                    arrival_rate_rps=arrival_rate_rps,
                    req_idx=req_idx,
                    ctx=request_ctx,
                    decision=decision,
                    wall_ms=wall_ms,
                    inflight_at_send=inflight_at_send,
                    recent_p50=recent_p50,
                    z_metrics=z_metrics if isinstance(z_metrics, dict) else {},
                    correct=correct,
                    scorable=scorable,
                    predicted=predicted,
                    ambiguity=ambiguity,
                ),
            )

        tasks: List[asyncio.Task[None]] = []
        for i, ex_idx in enumerate(order):
            tasks.append(asyncio.create_task(_handle(i, ex_idx)))
            await asyncio.sleep(float(rng.exponential(1.0 / arrival_rate_rps)))
        await asyncio.gather(*tasks)

    return records


def _build_summary(records: List[Dict[str, Any]], *, system_id: str, policy: str) -> Dict[str, Any]:
    latencies = [float(r["latency_ms"]) for r in records if r.get("latency_ms")]
    wall_latencies = [float(r["client_wall_ms"]) for r in records if r.get("client_wall_ms")]
    correct = sum(1 for r in records if r.get("correct"))
    scorable = sum(1 for r in records if r.get("scorable"))
    reason_counts = Counter(str(r.get("reason")) for r in records)
    bench_correct: Dict[str, List[bool]] = {}
    for r in records:
        if r.get("scorable"):
            bench_correct.setdefault(str(r["benchmark"]), []).append(bool(r["correct"]))

    selected_model_counts = dict(
        Counter(str(r["selected_model"]) for r in records if r.get("selected_model")),
    )

    by_model_lat: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        if r.get("latency_ms") and r.get("selected_model"):
            by_model_lat[str(r["selected_model"])].append(float(r["latency_ms"]))
    latency_by_selected_model = {
        m: summarise_latencies(v) for m, v in by_model_lat.items()
    }

    by_bench_lat: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        if r.get("latency_ms"):
            by_bench_lat[str(r.get("benchmark") or "")].append(float(r["latency_ms"]))
    latency_by_benchmark = {
        b: summarise_latencies(v) for b, v in by_bench_lat.items() if b
    }

    by_len_lat: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        if r.get("latency_ms"):
            lb = str(r.get("length_bucket") or "")
            by_len_lat[lb].append(float(r["latency_ms"]))
    latency_by_length_bucket = {
        k: summarise_latencies(v) for k, v in by_len_lat.items() if k
    }

    entry_rung_counts = dict(
        Counter(str(r.get("entry_rung")) for r in records if r.get("entry_rung")),
    )
    selected_rung_counts = dict(
        Counter(str(r.get("selected_rung")) for r in records if r.get("selected_rung")),
    )
    first_attempt_stage_counts = dict(
        Counter(
            str(r.get("first_attempt_stage"))
            for r in records
            if r.get("first_attempt_stage")
        ),
    )

    acc_by_model: Dict[str, List[bool]] = defaultdict(list)
    for r in records:
        if r.get("scorable") and r.get("selected_model"):
            acc_by_model[str(r["selected_model"])].append(bool(r.get("correct")))
    accuracy_scorable_pct_by_selected_model = {
        m: (sum(v) / len(v) * 100.0) if v else 0.0 for m, v in acc_by_model.items()
    }
    scorable_counts_by_selected_model = {m: len(v) for m, v in acc_by_model.items()}

    out_tokens = [int(r.get("total_output_tokens") or 0) for r in records]
    mean_total_output_tokens = (
        float(sum(out_tokens) / len(out_tokens)) if out_tokens else 0.0
    )

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
        "mean_total_output_tokens": mean_total_output_tokens,
        "reason_counts": dict(reason_counts),
        "selected_model_counts": selected_model_counts,
        "entry_rung_counts": entry_rung_counts,
        "selected_rung_counts": selected_rung_counts,
        "first_attempt_stage_counts": first_attempt_stage_counts,
        "latency_by_selected_model": latency_by_selected_model,
        "latency_by_benchmark": latency_by_benchmark,
        "latency_by_length_bucket": latency_by_length_bucket,
        "accuracy_scorable_pct_by_selected_model": accuracy_scorable_pct_by_selected_model,
        "scorable_counts_by_selected_model": scorable_counts_by_selected_model,
        **{f"latency_{k}": v for k, v in summarise_latencies(latencies).items()},
        **{
            f"client_wall_{k}": v
            for k, v in summarise_latencies(wall_latencies).items()
        },
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
    models_config = str(
        system.get("models_config")
        or exec_cfg.get("models_config")
        or "configs/models.yaml"
    )
    endpoints = resolve_endpoints(
        roles,
        endpoint_dir=endpoint_dir,
        models_config=models_config,
    )
    logger.info(
        "Resolved endpoints",
        extra={
            "models_config": models_config,
            "roles": roles,
            "selected_models": {r: endpoints[r]["model"] for r in roles},
        },
    )

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
    system_overrides = system.get("policy_overrides") or {}
    if not isinstance(system_overrides, dict):
        system_overrides = {}
    ctx = _merge_policy_ctx(base_ctx, policy_name, policy_params, system_overrides)
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
    csv_rows, csv_fields = _flatten_holdout_rows_for_csv(all_records)
    save_csv(csv_rows, run_dir / "per_request.csv", fieldnames=csv_fields)
    summary_csv_row = _flatten_summary_for_csv(summary)
    save_csv(
        [summary_csv_row],
        run_dir / "summary.csv",
        fieldnames=sorted(summary_csv_row.keys()),
    )
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
