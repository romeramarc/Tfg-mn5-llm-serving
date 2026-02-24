"""
eval/scoring.py
===============
Answer extraction and scoring utilities shared by all quality benchmarks.

Responsibilities
----------------
* Extract final answers from model-generated text using regex patterns.
* Normalise numeric and mathematical expressions for comparison.
* Compute exact-match accuracy over a list of predictions / references.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple


# ── Numeric answer extraction (GSM8K style) ────────────────

def extract_numeric_answer(text: str, pattern: str) -> Optional[str]:
    """Extract a numeric answer from *text* using *pattern*.

    The regex must contain one capturing group whose content is the
    answer string.  The match is taken from the **last** occurrence
    (models often restate the answer at the end).
    """
    matches = re.findall(pattern, text)
    if not matches:
        return None
    raw = matches[-1].strip()
    return raw


def normalise_numeric(value: str) -> Optional[float]:
    """Parse a numeric string into a float, ignoring commas and
    surrounding whitespace.  Returns ``None`` on failure."""
    try:
        cleaned = value.replace(",", "").replace(" ", "").strip()
        # Handle percentages
        if cleaned.endswith("%"):
            return float(cleaned[:-1]) / 100.0
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def numeric_match(prediction: str, reference: str) -> bool:
    """Return ``True`` when *prediction* and *reference* represent the
    same number (within floating-point tolerance)."""
    pred_val = normalise_numeric(prediction)
    ref_val = normalise_numeric(reference)
    if pred_val is None or ref_val is None:
        return False
    if ref_val == 0.0:
        return abs(pred_val) < 1e-6
    return math.isclose(pred_val, ref_val, rel_tol=1e-4)


# ── Math answer extraction (MATH / boxed style) ────────────

def extract_boxed_answer(text: str) -> Optional[str]:
    r"""Extract the content inside the last ``\boxed{...}`` in *text*.

    Handles nested braces up to two levels deep.
    """
    # Find all \boxed{...} occurrences — greedy on content
    # We manually track brace depth for robustness.
    results: list[str] = []
    idx = 0
    while idx < len(text):
        start = text.find("\\boxed{", idx)
        if start == -1:
            break
        # Walk from the opening brace
        brace_start = start + len("\\boxed{")
        depth = 1
        pos = brace_start
        while pos < len(text) and depth > 0:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            results.append(text[brace_start : pos - 1])
        idx = pos
    return results[-1].strip() if results else None


def normalise_math_answer(answer: str) -> str:
    r"""Normalise a mathematical answer string for comparison.

    Strips whitespace, removes ``\text{}``, ``\mathrm{}``, etc.
    """
    s = answer.strip()
    # Remove common LaTeX wrappers
    for cmd in (r"\text", r"\mathrm", r"\textbf", r"\mathbf"):
        s = s.replace(cmd + "{", "").rstrip("}")
    # Remove dollar signs
    s = s.replace("$", "")
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def math_answer_match(prediction: str, reference: str) -> bool:
    """Return ``True`` when the *prediction* matches the *reference*
    after normalisation.  Tries numeric comparison first, then
    exact normalised string comparison.
    """
    pred_norm = normalise_math_answer(prediction)
    ref_norm = normalise_math_answer(reference)

    # Try numeric comparison
    pred_num = normalise_numeric(pred_norm)
    ref_num = normalise_numeric(ref_norm)
    if pred_num is not None and ref_num is not None:
        return numeric_match(pred_norm, ref_norm)

    # Fall back to exact string match
    return pred_norm == ref_norm


# ── Aggregate metrics ──────────────────────────────────────

def compute_accuracy(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute accuracy and breakdown from a list of per-example results.

    Each result dict must contain ``"correct"`` (bool) and may contain
    ``"scorable"`` (bool, default ``True``).
    """
    total = len(results)
    scorable = [r for r in results if r.get("scorable", True)]
    unscorable = total - len(scorable)
    correct = sum(1 for r in scorable if r.get("correct", False))
    accuracy = correct / len(scorable) if scorable else 0.0

    return {
        "total_examples": total,
        "scorable_examples": len(scorable),
        "unscorable_examples": unscorable,
        "correct": correct,
        "incorrect": len(scorable) - correct,
        "accuracy": round(accuracy, 6),
        "accuracy_pct": round(accuracy * 100, 2),
    }
