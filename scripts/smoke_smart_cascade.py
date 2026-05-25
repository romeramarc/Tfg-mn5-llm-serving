"""Smoke-test the smart-cascade knobs on routing/policies.cascade_five_rung.

Mocks _generate_at_stage and the predictor suite to validate that:
  - threshold-per-rung makes a small-rung accept even with confidence 0.55.
  - max_attempts=2 caps cascade depth (after one student, jump to teacher).
  - accept_if_parseable bypasses the threshold when the response carries
    a parseable final answer and confidence clears the soft floor.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import routing.policies as P
from routing.policies import cascade_five_rung


class _MockSuite:
    """Predictor suite whose post_hoc returns a scripted sequence per rung."""

    def __init__(self, by_stage):
        self.by_stage = by_stage
        self.calls = []

    def post_hoc_probability(self, trace):
        stage = trace.model_tier
        self.calls.append(stage)
        return float(self.by_stage[stage])


def _mock_generate_factory(responses):
    """Return a coroutine matching _generate_at_stage signature."""

    async def fake_generate(client, prompt, ctx, stage, with_logprobs):
        payload = responses[stage]
        data = {
            "_latency_ms": payload.get("latency_ms", 100.0),
            "_status_code": 200,
            "_error": None,
            "choices": [
                {
                    "text": payload["text"],
                    "finish_reason": "stop",
                    "logprobs": {"top_logprobs": []},
                }
            ],
            "usage": {"completion_tokens": payload.get("tokens", 32)},
        }
        ep = {"model": f"mock-{stage}", "base_url": "http://mock"}
        return (
            data,
            ep,
            payload["text"],
            payload.get("tokens", 32),
            [],
        )

    return fake_generate


def _build_ctx(suite, **overrides):
    ctx = {
        "request_id": "req-1",
        "endpoints": {
            "student_small": {"model": "mock-1.5", "base_url": "http://mock"},
            "student_q3b": {"model": "mock-3", "base_url": "http://mock"},
            "student_mid": {"model": "mock-7", "base_url": "http://mock"},
            "teacher": {"model": "mock-14", "base_url": "http://mock"},
        },
        "prompt_metadata": {"benchmark": "gsm8k", "example_id": "ex-1"},
        "predictor_suite": suite,
        "rung_order": ["student_small", "student_q3b", "student_mid", "teacher"],
        "post_hoc_threshold": 0.7728,
        "max_tokens": 64,
        "temperature": 0.0,
    }
    ctx.update(overrides)
    return ctx


async def case_per_rung_threshold():
    print("\n== Case 1: per-rung threshold lets student_small accept at 0.55 ==")
    suite = _MockSuite({"student_small": 0.55, "student_q3b": 0.9, "student_mid": 0.9})
    P._generate_at_stage = _mock_generate_factory(
        {
            "student_small": {"text": "answer is #### 7", "latency_ms": 100},
            "student_q3b": {"text": "x", "latency_ms": 200},
            "student_mid": {"text": "x", "latency_ms": 300},
            "teacher": {"text": "x", "latency_ms": 800},
        }
    )
    ctx = _build_ctx(
        suite,
        post_hoc_threshold_per_rung={
            "student_small": 0.50,
            "student_q3b": 0.60,
            "student_mid": 0.70,
            "teacher": 1.00,
        },
    )
    dec = await cascade_five_rung(None, "p?", ctx)
    n = len(dec.attempts)
    final = dec.attempts[-1]
    assert n == 1, f"expected 1 attempt, got {n}: {[a['stage'] for a in dec.attempts]}"
    assert final["stage"] == "student_small", f"expected accept at student_small, got {final['stage']}"
    assert final["decision"] == "accepted_post_hoc"
    print(f"  OK -> {dec.reason} at {final['stage']} conf={final['confidence']:.3f}")


async def case_max_attempts_cap():
    print("\n== Case 2: max_attempts=2 keeps order [small, q3b] (no teacher jump) ==")
    suite = _MockSuite({"student_small": 0.1, "student_q3b": 0.1, "student_mid": 0.9})
    P._generate_at_stage = _mock_generate_factory(
        {
            "student_small": {"text": "garbage1", "latency_ms": 100},
            "student_q3b": {"text": "garbage2 #### 7", "latency_ms": 200},
            "student_mid": {"text": "good", "latency_ms": 300},
            "teacher": {"text": "teacher", "latency_ms": 800},
        }
    )
    ctx = _build_ctx(suite, max_attempts=2)
    dec = await cascade_five_rung(None, "p?", ctx)
    stages = [a["stage"] for a in dec.attempts]
    assert stages == ["student_small", "student_q3b"], f"expected [small, q3b], got {stages}"
    assert dec.reason == "cascade_exhausted", f"expected exhausted, got {dec.reason}"
    print(f"  OK -> {dec.reason} path={'->'.join(stages)}")


async def case_max_attempts_with_teacher():
    print("\n== Case 2b: max_attempts=4 with teacher at end works as before ==")
    suite = _MockSuite({"student_small": 0.1, "student_q3b": 0.1, "student_mid": 0.1})
    P._generate_at_stage = _mock_generate_factory(
        {
            "student_small": {"text": "x", "latency_ms": 100},
            "student_q3b": {"text": "x", "latency_ms": 200},
            "student_mid": {"text": "x", "latency_ms": 300},
            "teacher": {"text": "teacher answer #### 42", "latency_ms": 800},
        }
    )
    ctx = _build_ctx(suite, max_attempts=4)
    dec = await cascade_five_rung(None, "p?", ctx)
    stages = [a["stage"] for a in dec.attempts]
    assert stages == ["student_small", "student_q3b", "student_mid", "teacher"], (
        f"expected full path, got {stages}"
    )
    print(f"  OK -> {dec.reason} path={'->'.join(stages)}")


async def case_max_attempts_3_no_teacher():
    print("\n== Case 2c: max_attempts=3 stops at student_mid (NO teacher) ==")
    suite = _MockSuite({"student_small": 0.1, "student_q3b": 0.1, "student_mid": 0.1})
    P._generate_at_stage = _mock_generate_factory(
        {
            "student_small": {"text": "x", "latency_ms": 100},
            "student_q3b": {"text": "x", "latency_ms": 200},
            "student_mid": {"text": "x", "latency_ms": 300},
            "teacher": {"text": "teacher", "latency_ms": 800},
        }
    )
    ctx = _build_ctx(suite, max_attempts=3)
    dec = await cascade_five_rung(None, "p?", ctx)
    stages = [a["stage"] for a in dec.attempts]
    assert stages == ["student_small", "student_q3b", "student_mid"], (
        f"expected [small,q3b,mid] (no teacher), got {stages}"
    )
    assert "teacher" not in stages, "teacher must not be probed with max_attempts=3"
    print(f"  OK -> {dec.reason} path={'->'.join(stages)}")


async def case_parseable_bypass():
    print("\n== Case 3: parseable bypass accepts borderline response ==")
    suite = _MockSuite({"student_small": 0.45, "student_q3b": 0.9, "student_mid": 0.9})
    P._generate_at_stage = _mock_generate_factory(
        {
            "student_small": {"text": "calculation ... #### 7", "latency_ms": 100},
            "student_q3b": {"text": "x", "latency_ms": 200},
            "student_mid": {"text": "x", "latency_ms": 300},
            "teacher": {"text": "x", "latency_ms": 800},
        }
    )
    ctx = _build_ctx(
        suite,
        post_hoc_threshold_per_rung={
            "student_small": 0.70,
            "student_q3b": 0.70,
            "student_mid": 0.70,
            "teacher": 1.00,
        },
        accept_if_parseable=True,
        parseable_min_confidence=0.40,
    )
    dec = await cascade_five_rung(None, "p?", ctx)
    final = dec.attempts[-1]
    assert final["stage"] == "student_small"
    assert final["decision"] == "accepted_parseable_bypass", final["decision"]
    print(f"  OK -> {dec.reason} at {final['stage']} conf={final['confidence']:.3f}")


async def case_legacy_threshold():
    print("\n== Case 4: legacy single threshold still works ==")
    suite = _MockSuite({"student_small": 0.30, "student_q3b": 0.40, "student_mid": 0.85})
    P._generate_at_stage = _mock_generate_factory(
        {
            "student_small": {"text": "x", "latency_ms": 100},
            "student_q3b": {"text": "x", "latency_ms": 200},
            "student_mid": {"text": "good #### 42", "latency_ms": 300},
            "teacher": {"text": "x", "latency_ms": 800},
        }
    )
    ctx = _build_ctx(suite)
    dec = await cascade_five_rung(None, "p?", ctx)
    stages = [a["stage"] for a in dec.attempts]
    assert stages[-1] == "student_mid", f"expected end at student_mid, got {stages}"
    print(f"  OK -> {dec.reason} path={'->'.join(stages)}")


async def main():
    await case_per_rung_threshold()
    await case_max_attempts_cap()
    await case_max_attempts_with_teacher()
    await case_max_attempts_3_no_teacher()
    await case_parseable_bypass()
    await case_legacy_threshold()
    print("\nALL OK")


if __name__ == "__main__":
    asyncio.run(main())
