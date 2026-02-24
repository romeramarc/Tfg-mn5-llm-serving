"""
routing/policies.py
===================
Routing policy implementations.

Each policy is a callable with the signature::

    async def policy(prompt, context) -> RoutingDecision

where ``context`` carries endpoint clients, thresholds, etc.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import httpx

from routing.confidence import confidence_from_logprobs, heuristic_confidence
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RoutingDecision:
    """Immutable record of a single routing decision."""
    selected_model: str
    latency_ms: float
    response_text: str
    confidence: Optional[float]
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Helpers ─────────────────────────────────────────────────

async def _query_endpoint(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    logprobs: Optional[int] = None,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Query a vLLM endpoint and return the parsed JSON + timing."""
    url = f"{base_url.rstrip('/')}/v1/completions"
    body: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if logprobs is not None:
        body["logprobs"] = logprobs

    t0 = time.perf_counter()
    resp = await client.post(url, json=body, timeout=timeout)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    data = resp.json()
    data["_latency_ms"] = latency_ms
    return data


# ── Policy A: Always teacher ───────────────────────────────

async def always_teacher(
    client: httpx.AsyncClient,
    prompt: str,
    ctx: Dict[str, Any],
) -> RoutingDecision:
    """Route every request to the teacher."""
    ep = ctx["endpoints"]["teacher"]
    data = await _query_endpoint(
        client, ep["base_url"], ep["model"], prompt,
        ctx.get("max_tokens", 256), ctx.get("temperature", 0.0),
    )
    text = data["choices"][0]["text"] if data.get("choices") else ""
    return RoutingDecision(
        selected_model=ep["model"],
        latency_ms=data["_latency_ms"],
        response_text=text,
        confidence=None,
        reason="always_teacher",
    )


# ── Policy B: Cascading / forced escalation ────────────────

async def cascading(
    client: httpx.AsyncClient,
    prompt: str,
    ctx: Dict[str, Any],
) -> RoutingDecision:
    """Try the student first; escalate to teacher on timeout / error."""
    student_ep = ctx["endpoints"]["student"]
    teacher_ep = ctx["endpoints"]["teacher"]
    student_timeout_s = ctx.get("student_timeout_ms", 3000) / 1000.0

    try:
        data = await _query_endpoint(
            client, student_ep["base_url"], student_ep["model"], prompt,
            ctx.get("max_tokens", 256), ctx.get("temperature", 0.0),
            timeout=student_timeout_s,
        )
        text = data["choices"][0]["text"] if data.get("choices") else ""
        return RoutingDecision(
            selected_model=student_ep["model"],
            latency_ms=data["_latency_ms"],
            response_text=text,
            confidence=None,
            reason="student_ok",
        )
    except Exception as exc:
        logger.info("Student failed; escalating to teacher",
                     extra={"error": str(exc)})
        data = await _query_endpoint(
            client, teacher_ep["base_url"], teacher_ep["model"], prompt,
            ctx.get("max_tokens", 256), ctx.get("temperature", 0.0),
        )
        text = data["choices"][0]["text"] if data.get("choices") else ""
        return RoutingDecision(
            selected_model=teacher_ep["model"],
            latency_ms=data["_latency_ms"],
            response_text=text,
            confidence=None,
            reason="escalated_after_student_failure",
        )


# ── Policy C: Confidence-based routing ─────────────────────

async def confidence_routing(
    client: httpx.AsyncClient,
    prompt: str,
    ctx: Dict[str, Any],
) -> RoutingDecision:
    """Query student with logprobs.  Escalate if confidence < threshold."""
    student_ep = ctx["endpoints"]["student"]
    teacher_ep = ctx["endpoints"]["teacher"]
    logprobs_k = ctx.get("logprobs_top_k", 5)
    threshold = ctx.get("confidence_threshold", 0.70)
    fallback_method = ctx.get("fallback_method", "entropy")

    # Query student with logprobs
    data = await _query_endpoint(
        client, student_ep["base_url"], student_ep["model"], prompt,
        ctx.get("max_tokens", 256), ctx.get("temperature", 0.0),
        logprobs=logprobs_k,
    )
    text = data["choices"][0]["text"] if data.get("choices") else ""
    student_latency = data["_latency_ms"]

    # Compute confidence
    raw_logprobs = (
        data.get("choices", [{}])[0]
        .get("logprobs", {})
        .get("top_logprobs", [])
    )
    if raw_logprobs:
        conf = confidence_from_logprobs(raw_logprobs, method=fallback_method)
    else:
        conf = heuristic_confidence(text)

    if conf >= threshold:
        return RoutingDecision(
            selected_model=student_ep["model"],
            latency_ms=student_latency,
            response_text=text,
            confidence=conf,
            reason="student_confident",
        )

    # Escalate
    logger.info("Low confidence; escalating to teacher",
                 extra={"confidence": conf, "threshold": threshold})
    data = await _query_endpoint(
        client, teacher_ep["base_url"], teacher_ep["model"], prompt,
        ctx.get("max_tokens", 256), ctx.get("temperature", 0.0),
    )
    text = data["choices"][0]["text"] if data.get("choices") else ""
    total_latency = student_latency + data["_latency_ms"]
    return RoutingDecision(
        selected_model=teacher_ep["model"],
        latency_ms=total_latency,
        response_text=text,
        confidence=conf,
        reason="escalated_low_confidence",
    )


# ── Registry ───────────────────────────────────────────────

POLICIES = {
    "always_teacher": always_teacher,
    "cascading": cascading,
    "confidence": confidence_routing,
}
