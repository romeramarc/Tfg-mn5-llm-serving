"""
routing/confidence.py
=====================
Confidence estimators for routing decisions.

Two strategies are implemented:

1. **max_logprob** — if the server returns ``logprobs``, confidence is
   the maximum probability assigned to any token in the generated
   sequence.

2. **entropy** — normalised entropy of the top-k probability
   distribution.  Lower entropy → higher confidence.

Both functions return a float in [0, 1].
"""

from __future__ import annotations

import math
from typing import Dict, List, Any, Optional


def confidence_from_logprobs(
    logprobs: List[Dict[str, float]],
    method: str = "max_logprob",
) -> float:
    """Estimate confidence from a list of per-token logprob dicts.

    Parameters
    ----------
    logprobs : list[dict[str, float]]
        Each element maps token strings to their log-probabilities,
        as returned by vLLM when ``logprobs=k`` is requested.
    method : str
        ``"max_logprob"`` or ``"entropy"``.

    Returns
    -------
    float
        Confidence score in [0, 1].  Higher is more confident.
    """
    if not logprobs:
        return 0.0

    if method == "max_logprob":
        return _max_logprob_confidence(logprobs)
    elif method == "entropy":
        return _entropy_confidence(logprobs)
    else:
        raise ValueError(f"Unknown confidence method: {method}")


# ── Strategy implementations ───────────────────────────────

def _max_logprob_confidence(logprobs: List[Dict[str, float]]) -> float:
    """Average of the maximum token probability at each position."""
    scores: list[float] = []
    for pos in logprobs:
        if not pos:
            continue
        max_lp = max(pos.values())
        scores.append(math.exp(max_lp))   # convert logprob → prob
    return sum(scores) / len(scores) if scores else 0.0


def _entropy_confidence(logprobs: List[Dict[str, float]]) -> float:
    """1 − normalised entropy averaged across positions."""
    confidences: list[float] = []
    for pos in logprobs:
        if not pos:
            continue
        probs = [math.exp(lp) for lp in pos.values()]
        total = sum(probs)
        if total == 0:
            continue
        probs = [p / total for p in probs]

        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        max_entropy = math.log(len(probs)) if len(probs) > 1 else 1.0
        norm = entropy / (max_entropy + 1e-12)
        confidences.append(1.0 - norm)
    return sum(confidences) / len(confidences) if confidences else 0.0


# ── Heuristic fallback ─────────────────────────────────────

def heuristic_confidence(response_text: str) -> float:
    """Very simple heuristic when logprobs are unavailable.

    Shorter, more decisive answers get a higher confidence score.
    This is a *fallback only* — prefer logprob-based estimates.
    """
    if not response_text:
        return 0.0
    words = response_text.split()
    hedging = sum(
        1 for w in words
        if w.lower() in {"maybe", "perhaps", "possibly", "uncertain",
                          "unsure", "might", "could"}
    )
    hedging_ratio = hedging / max(len(words), 1)
    # Penalise hedging and very short answers
    base = 0.85
    penalty = hedging_ratio * 0.5
    if len(words) < 3:
        penalty += 0.3
    return max(0.0, min(1.0, base - penalty))
