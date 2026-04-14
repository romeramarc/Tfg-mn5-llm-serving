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
    request_id: str
    selected_model: str
    latency_ms: float
    response_text: str
    confidence: Optional[float]
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    attempts: list[Dict[str, Any]] = field(default_factory=list)


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
    data["_status_code"] = resp.status_code
    return data


async def _query_endpoint_with_status(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    logprobs: Optional[int] = None,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Query a vLLM endpoint and always return a structured response."""
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
    try:
        resp = await client.post(url, json=body, timeout=timeout)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Best-effort JSON parsing even for non-2xx responses.
        try:
            data = resp.json()
        except Exception:
            data = {"choices": [], "usage": {}}

        data["_latency_ms"] = latency_ms
        data["_status_code"] = resp.status_code
        data["_error"] = None

        if resp.status_code >= 400:
            snippet = ""
            try:
                snippet = resp.text[:240]
            except Exception:
                snippet = ""
            data["_error"] = f"http_{resp.status_code}: {snippet}".strip()
        return data
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "choices": [],
            "usage": {},
            "_latency_ms": latency_ms,
            "_status_code": None,
            "_error": str(exc),
        }


def _extract_choice_fields(data: Dict[str, Any]) -> tuple[str, int, Optional[str], list[Dict[str, float]]]:
    """Extract completion text, token usage, finish reason, and top-logprobs."""
    choices = data.get("choices") or []
    first = choices[0] if choices else {}
    text = str(first.get("text", ""))
    usage = data.get("usage") or {}
    output_tokens = int(usage.get("completion_tokens", 0) or 0)
    finish_reason = first.get("finish_reason")
    top_logprobs = (first.get("logprobs") or {}).get("top_logprobs", [])
    return text, output_tokens, finish_reason, top_logprobs


def _estimate_confidence(
    top_logprobs: list[Dict[str, float]],
    response_text: str,
    method: str,
) -> float:
    """Estimate confidence from logprobs when available, otherwise fallback."""
    if top_logprobs:
        return confidence_from_logprobs(top_logprobs, method=method)
    return heuristic_confidence(response_text)


def _build_attempt(
    stage: str,
    endpoint: Dict[str, Any],
    data: Dict[str, Any],
    decision: str,
    confidence: Optional[float] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a normalised per-attempt trace record."""
    text, output_tokens, finish_reason, top_logprobs = _extract_choice_fields(data)
    return {
        "stage": stage,
        "model": endpoint.get("model", ""),
        "base_url": endpoint.get("base_url", ""),
        "status_code": data.get("_status_code"),
        "latency_ms": data.get("_latency_ms", 0.0),
        "output_tokens": output_tokens,
        "finish_reason": finish_reason,
        "confidence": confidence,
        "threshold": threshold,
        "decision": decision,
        "error": data.get("_error"),
        "used_logprobs": bool(top_logprobs),
        "response_preview": text[:160],
    }


# ── Policy A: Always teacher ───────────────────────────────

async def always_teacher(
    client: httpx.AsyncClient,
    prompt: str,
    ctx: Dict[str, Any],
) -> RoutingDecision:
    """Route every request to the teacher."""
    request_id = ctx.get("request_id", "")
    ep = ctx["endpoints"]["teacher"]
    data = await _query_endpoint_with_status(
        client, ep["base_url"], ep["model"], prompt,
        ctx.get("max_tokens", 256), ctx.get("temperature", 0.0),
    )
    text, _, _, _ = _extract_choice_fields(data)
    reason = "always_teacher" if not data.get("_error") else "teacher_error"
    return RoutingDecision(
        request_id=request_id,
        selected_model=ep["model"],
        latency_ms=data.get("_latency_ms", 0.0),
        response_text=text,
        confidence=None,
        reason=reason,
        metadata={"error": data.get("_error")},
        attempts=[_build_attempt("teacher", ep, data, reason)],
    )


# ── Policy B: Cascading / forced escalation ────────────────

async def cascading(
    client: httpx.AsyncClient,
    prompt: str,
    ctx: Dict[str, Any],
) -> RoutingDecision:
    """Try the student first; escalate to teacher on timeout / error."""
    request_id = ctx.get("request_id", "")
    student_ep = ctx["endpoints"]["student"]
    teacher_ep = ctx["endpoints"]["teacher"]
    student_timeout_s = ctx.get("student_timeout_ms", 3000) / 1000.0
    attempts: list[Dict[str, Any]] = []

    data = await _query_endpoint_with_status(
        client, student_ep["base_url"], student_ep["model"], prompt,
        ctx.get("max_tokens", 256), ctx.get("temperature", 0.0),
        timeout=student_timeout_s,
    )
    text, _, _, _ = _extract_choice_fields(data)

    if not data.get("_error"):
        attempts.append(_build_attempt("student", student_ep, data, "student_ok"))
        return RoutingDecision(
            request_id=request_id,
            selected_model=student_ep["model"],
            latency_ms=data.get("_latency_ms", 0.0),
            response_text=text,
            confidence=None,
            reason="student_ok",
            attempts=attempts,
        )

    logger.info("Student failed; escalating to teacher",
                extra={"error": data.get("_error")})
    attempts.append(_build_attempt("student", student_ep, data, "student_error"))
    teacher_data = await _query_endpoint_with_status(
        client, teacher_ep["base_url"], teacher_ep["model"], prompt,
        ctx.get("max_tokens", 256), ctx.get("temperature", 0.0),
    )
    teacher_text, _, _, _ = _extract_choice_fields(teacher_data)
    teacher_reason = (
        "escalated_after_student_failure"
        if not teacher_data.get("_error")
        else "teacher_error_after_student_failure"
    )
    attempts.append(_build_attempt("teacher", teacher_ep, teacher_data, teacher_reason))
    return RoutingDecision(
        request_id=request_id,
        selected_model=teacher_ep["model"],
        latency_ms=data.get("_latency_ms", 0.0) + teacher_data.get("_latency_ms", 0.0),
        response_text=teacher_text,
        confidence=None,
        reason=teacher_reason,
        metadata={
            "student_error": data.get("_error"),
            "teacher_error": teacher_data.get("_error"),
        },
        attempts=attempts,
    )


# ── Policy C: Confidence-based routing ─────────────────────

async def confidence_routing(
    client: httpx.AsyncClient,
    prompt: str,
    ctx: Dict[str, Any],
) -> RoutingDecision:
    """Query student with logprobs.  Escalate if confidence < threshold."""
    request_id = ctx.get("request_id", "")
    student_ep = ctx["endpoints"]["student"]
    teacher_ep = ctx["endpoints"]["teacher"]
    logprobs_k = ctx.get("logprobs_top_k", 5)
    threshold = ctx.get("confidence_threshold", 0.70)
    fallback_method = ctx.get("fallback_method", "entropy")
    attempts: list[Dict[str, Any]] = []

    data = await _query_endpoint_with_status(
        client, student_ep["base_url"], student_ep["model"], prompt,
        ctx.get("max_tokens", 256), ctx.get("temperature", 0.0),
        logprobs=logprobs_k,
    )
    text, _, _, top_logprobs = _extract_choice_fields(data)
    student_latency = data.get("_latency_ms", 0.0)

    if data.get("_error"):
        logger.info("Student failed in confidence policy; escalating",
                    extra={"error": data.get("_error")})
        attempts.append(_build_attempt("student", student_ep, data, "student_error"))
        teacher_data = await _query_endpoint_with_status(
            client, teacher_ep["base_url"], teacher_ep["model"], prompt,
            ctx.get("max_tokens", 256), ctx.get("temperature", 0.0),
        )
        teacher_text, _, _, _ = _extract_choice_fields(teacher_data)
        attempts.append(_build_attempt(
            "teacher",
            teacher_ep,
            teacher_data,
            "escalated_after_student_error",
        ))
        return RoutingDecision(
            request_id=request_id,
            selected_model=teacher_ep["model"],
            latency_ms=student_latency + teacher_data.get("_latency_ms", 0.0),
            response_text=teacher_text,
            confidence=None,
            reason="escalated_after_student_error",
            metadata={
                "student_error": data.get("_error"),
                "teacher_error": teacher_data.get("_error"),
            },
            attempts=attempts,
        )

    # Compute confidence
    conf = _estimate_confidence(top_logprobs, text, fallback_method)

    if conf >= threshold:
        attempts.append(_build_attempt(
            "student", student_ep, data, "student_confident", conf, threshold,
        ))
        return RoutingDecision(
            request_id=request_id,
            selected_model=student_ep["model"],
            latency_ms=student_latency,
            response_text=text,
            confidence=conf,
            reason="student_confident",
            attempts=attempts,
        )

    # Escalate
    logger.info("Low confidence; escalating to teacher",
                 extra={"confidence": conf, "threshold": threshold})
    attempts.append(_build_attempt(
        "student", student_ep, data, "low_confidence", conf, threshold,
    ))
    teacher_data = await _query_endpoint_with_status(
        client, teacher_ep["base_url"], teacher_ep["model"], prompt,
        ctx.get("max_tokens", 256), ctx.get("temperature", 0.0),
    )
    teacher_text, _, _, _ = _extract_choice_fields(teacher_data)
    total_latency = student_latency + teacher_data.get("_latency_ms", 0.0)
    attempts.append(_build_attempt(
        "teacher", teacher_ep, teacher_data, "escalated_low_confidence",
    ))
    return RoutingDecision(
        request_id=request_id,
        selected_model=teacher_ep["model"],
        latency_ms=total_latency,
        response_text=teacher_text,
        confidence=conf,
        reason="escalated_low_confidence",
        metadata={"teacher_error": teacher_data.get("_error")},
        attempts=attempts,
    )


# ── Policy D: Fixed 3-tier cascade (1.5B -> 7B -> teacher) ─

async def cascade_three_tier(
    client: httpx.AsyncClient,
    prompt: str,
    ctx: Dict[str, Any],
) -> RoutingDecision:
    """Run fixed cascade student_small -> student_mid -> teacher.

    Acceptance criteria:
    1) Small student accepted when confidence >= small threshold.
    2) Otherwise mid student accepted when confidence >= mid threshold.
    3) Otherwise escalate to teacher as final fallback.
    """
    request_id = ctx.get("request_id", "")
    endpoints = ctx.get("endpoints", {})
    small_ep = endpoints.get("student_small") or endpoints.get("student")
    mid_ep = endpoints.get("student_mid")
    teacher_ep = endpoints.get("teacher")

    if not small_ep or not mid_ep or not teacher_ep:
        raise ValueError(
            "cascade_three_tier requires endpoints.student_small, "
            "endpoints.student_mid, and endpoints.teacher"
        )

    max_tokens = ctx.get("max_tokens", 256)
    temperature = ctx.get("temperature", 0.0)
    logprobs_k = ctx.get("logprobs_top_k", 5)
    confidence_method = ctx.get("fallback_method", "entropy")
    small_threshold = ctx.get("small_confidence_threshold", 0.45)
    mid_threshold = ctx.get("mid_confidence_threshold", 0.60)
    small_timeout = ctx.get("small_timeout_ms", 3000) / 1000.0
    mid_timeout = ctx.get("mid_timeout_ms", 5000) / 1000.0
    teacher_timeout = ctx.get("teacher_timeout_ms", 120000) / 1000.0

    total_latency = 0.0
    attempts: list[Dict[str, Any]] = []
    small_conf: Optional[float] = None
    mid_conf: Optional[float] = None

    # Stage 1: 1.5B (student_small)
    small_data = await _query_endpoint_with_status(
        client,
        small_ep["base_url"],
        small_ep["model"],
        prompt,
        max_tokens,
        temperature,
        logprobs=logprobs_k,
        timeout=small_timeout,
    )
    total_latency += small_data.get("_latency_ms", 0.0)
    small_text, _, _, small_top_logprobs = _extract_choice_fields(small_data)

    if not small_data.get("_error"):
        small_conf = _estimate_confidence(
            small_top_logprobs,
            small_text,
            confidence_method,
        )
        if small_conf >= small_threshold:
            attempts.append(_build_attempt(
                "student_small", small_ep, small_data,
                "accepted_student_small", small_conf, small_threshold,
            ))
            return RoutingDecision(
                request_id=request_id,
                selected_model=small_ep["model"],
                latency_ms=total_latency,
                response_text=small_text,
                confidence=small_conf,
                reason="accepted_student_small",
                metadata={
                    "route_path": "student_small",
                    "small_confidence": small_conf,
                    "mid_confidence": None,
                },
                attempts=attempts,
            )

        attempts.append(_build_attempt(
            "student_small", small_ep, small_data,
            "escalate_low_confidence", small_conf, small_threshold,
        ))
    else:
        attempts.append(_build_attempt(
            "student_small", small_ep, small_data,
            "escalate_error", small_conf, small_threshold,
        ))

    # Stage 2: 7B (student_mid)
    mid_data = await _query_endpoint_with_status(
        client,
        mid_ep["base_url"],
        mid_ep["model"],
        prompt,
        max_tokens,
        temperature,
        logprobs=logprobs_k,
        timeout=mid_timeout,
    )
    total_latency += mid_data.get("_latency_ms", 0.0)
    mid_text, _, _, mid_top_logprobs = _extract_choice_fields(mid_data)

    if not mid_data.get("_error"):
        mid_conf = _estimate_confidence(
            mid_top_logprobs,
            mid_text,
            confidence_method,
        )
        if mid_conf >= mid_threshold:
            attempts.append(_build_attempt(
                "student_mid", mid_ep, mid_data,
                "accepted_student_mid", mid_conf, mid_threshold,
            ))
            return RoutingDecision(
                request_id=request_id,
                selected_model=mid_ep["model"],
                latency_ms=total_latency,
                response_text=mid_text,
                confidence=mid_conf,
                reason="accepted_student_mid",
                metadata={
                    "route_path": "student_small->student_mid",
                    "small_confidence": small_conf,
                    "mid_confidence": mid_conf,
                },
                attempts=attempts,
            )

        attempts.append(_build_attempt(
            "student_mid", mid_ep, mid_data,
            "escalate_low_confidence", mid_conf, mid_threshold,
        ))
    else:
        attempts.append(_build_attempt(
            "student_mid", mid_ep, mid_data,
            "escalate_error", mid_conf, mid_threshold,
        ))

    # Stage 3: Teacher (final fallback)
    teacher_data = await _query_endpoint_with_status(
        client,
        teacher_ep["base_url"],
        teacher_ep["model"],
        prompt,
        max_tokens,
        temperature,
        timeout=teacher_timeout,
    )
    total_latency += teacher_data.get("_latency_ms", 0.0)
    teacher_text, _, _, _ = _extract_choice_fields(teacher_data)

    final_reason = (
        "accepted_teacher"
        if not teacher_data.get("_error")
        else "teacher_error"
    )
    attempts.append(_build_attempt("teacher", teacher_ep, teacher_data, final_reason))

    return RoutingDecision(
        request_id=request_id,
        selected_model=teacher_ep["model"],
        latency_ms=total_latency,
        response_text=teacher_text,
        confidence=mid_conf if mid_conf is not None else small_conf,
        reason=final_reason,
        metadata={
            "route_path": "student_small->student_mid->teacher",
            "small_confidence": small_conf,
            "mid_confidence": mid_conf,
            "teacher_error": teacher_data.get("_error"),
        },
        attempts=attempts,
    )


# ── Registry ───────────────────────────────────────────────

POLICIES = {
    "always_teacher": always_teacher,
    "cascading": cascading,
    "confidence": confidence_routing,
    "cascade_three_tier": cascade_three_tier,
}
