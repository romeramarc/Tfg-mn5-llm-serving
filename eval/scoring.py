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

from decimal import Decimal, InvalidOperation
import math
import re
from typing import Any, Dict, List, Optional, Tuple


# ── Numeric answer extraction (GSM8K style) ────────────────

_NUMERIC_TOKEN_RE = re.compile(
    r"[-+]?\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?",
    flags=re.IGNORECASE,
)


def _canonicalise_numeric_token(raw: str) -> Optional[str]:
    """Normalise a numeric token to a canonical string.

    Rules:
    - remove currency symbols and separators (e.g. "$43,500" -> "43500")
    - drop trailing percent sign for extraction (matching handles value semantics)
    - collapse trailing .0 when the value is integral
    """
    if raw is None:
        return None

    s = raw.strip()
    if not s:
        return None

    s = s.replace("$", "")
    s = s.replace(",", "")
    s = s.replace(" ", "")
    s = s.rstrip(".?!;:")
    if s.endswith("%"):
        s = s[:-1]
    if s.startswith("+"):
        s = s[1:]

    if not re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        return None

    try:
        dec = Decimal(s)
    except InvalidOperation:
        return None

    if dec == dec.to_integral_value():
        return str(int(dec))

    norm = format(dec.normalize(), "f")
    if norm == "-0":
        return "0"
    return norm


def _extract_last_numeric_from_pattern(text: str, pattern: str) -> Optional[str]:
    """Apply a regex pattern and canonicalise the last captured value."""
    try:
        matches = re.findall(pattern, text)
    except re.error:
        return None

    if not matches:
        return None

    candidate = matches[-1]
    if isinstance(candidate, tuple):
        candidate = next((m for m in reversed(candidate) if m), "")
    return _canonicalise_numeric_token(str(candidate))


def _extract_with_priority_patterns(text: str) -> Optional[str]:
    """Robust final-answer extraction with explicit-context priority."""
    patterns = [
        r"####\s*([-+]?\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?)",
        r"\\boxed\{\s*([-+]?\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?)\s*\}",
        r"(?:final\s+answer|final\s+result)\s*(?:is|=|:)?\s*([-+]?\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?)",
        r"(?:therefore|thus|hence)[^\n]{0,80}?(?:answer\s*(?:is|=|:)?\s*)?([-+]?\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?)",
        r"(?:the\s+)?answer\s*(?:is|=|:)\s*([-+]?\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?)",
    ]

    for pattern in patterns:
        m = re.findall(pattern, text, flags=re.IGNORECASE)
        if not m:
            continue
        extracted = _canonicalise_numeric_token(str(m[-1]))
        if extracted is not None:
            return extracted

    return None


def _extract_from_short_final_line(text: str) -> Optional[str]:
    """Conservative fallback: numeric-only final line in the tail of output."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None

    for line in reversed(lines[-3:]):
        tokens = _NUMERIC_TOKEN_RE.findall(line)
        if len(tokens) != 1:
            continue
        # Avoid pulling numbers from long reasoning lines.
        alpha_words = len(re.findall(r"[A-Za-z]+", line))
        if alpha_words > 3:
            continue
        parsed = _canonicalise_numeric_token(tokens[0])
        if parsed is not None:
            return parsed

    return None

def extract_numeric_answer(text: str, pattern: str) -> Optional[str]:
    """Extract a numeric answer from *text* using *pattern*.

    The regex must contain one capturing group whose content is the
    answer string. Extraction now prioritises explicit final-answer
    markers before falling back to the provided pattern.
    """
    if not text:
        return None

    # 1) Prefer explicit final-answer contexts.
    parsed = _extract_with_priority_patterns(text)
    if parsed is not None:
        return parsed

    # 2) Backward-compatible custom pattern from config.
    parsed = _extract_last_numeric_from_pattern(text, pattern)
    if parsed is not None:
        return parsed

    # 3) Conservative tail-line fallback.
    return _extract_from_short_final_line(text)


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

    Converts common LaTeX constructs to a sympy-parseable form and
    strips cosmetic formatting.
    """
    s = answer.strip()
    # Remove common LaTeX wrappers
    for cmd in (r"\text", r"\mathrm", r"\textbf", r"\mathbf", r"\displaystyle"):
        s = re.sub(re.escape(cmd) + r"\{([^}]*)\}", r"\1", s)
    # Remove dollar signs and \left \right
    s = s.replace("$", "").replace(r"\left", "").replace(r"\right", "")
    # \frac{a}{b}  →  (a)/(b)
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
    # \sqrt{x}  →  sqrt(x)
    s = re.sub(r"\\sqrt\{([^}]*)\}", r"sqrt(\1)", s)
    # \sqrt[n]{x}  →  (x)**(1/n)
    s = re.sub(r"\\sqrt\[([^\]]*)\]\{([^}]*)\}", r"(\2)**(1/(\1))", s)
    # x^{n}  →  x**(n)
    s = re.sub(r"\^\{([^}]*)\}", r"**(\1)", s)
    # x^n  →  x**n
    s = re.sub(r"\^(\w)", r"**\1", s)
    # \pi  →  pi   \infty  →  oo
    s = s.replace(r"\pi", "pi").replace(r"\infty", "oo")
    # \cdot \times  →  *
    s = s.replace(r"\cdot", "*").replace(r"\times", "*")
    # Remove remaining backslash commands (e.g. \, \! \; spacing)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = re.sub(r"\\[,;!: ]", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _sympy_equal(pred: str, ref: str) -> Optional[bool]:
    """Try symbolic equality check via sympy.

    Returns ``True`` / ``False`` if sympy can parse both, else ``None``.
    """
    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
        )
        transformations = standard_transformations + (implicit_multiplication_application,)

        p = parse_expr(pred, transformations=transformations, evaluate=True)
        r = parse_expr(ref,  transformations=transformations, evaluate=True)
        diff = sympy.simplify(p - r)
        return diff == 0
    except Exception:
        return None


def math_answer_match(prediction: str, reference: str) -> bool:
    """Return ``True`` when the *prediction* matches the *reference*.

    Evaluation order:
    1. Numeric floating-point comparison (fast, handles most numeric answers).
    2. Symbolic sympy comparison (handles fractions, sqrt, pi, algebraic exprs).
    3. Normalised exact string comparison (fallback for non-numeric answers).
    """
    pred_norm = normalise_math_answer(prediction)
    ref_norm  = normalise_math_answer(reference)

    # 1. Numeric comparison
    pred_num = normalise_numeric(pred_norm)
    ref_num  = normalise_numeric(ref_norm)
    if pred_num is not None and ref_num is not None:
        return numeric_match(pred_norm, ref_norm)

    # 2. Symbolic comparison via sympy
    sym_result = _sympy_equal(pred_norm, ref_norm)
    if sym_result is not None:
        return sym_result

    # 3. Exact string fallback
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
    accuracy_scorable = correct / len(scorable) if scorable else 0.0
    accuracy_total = correct / total if total else 0.0
    unscorable_rate = unscorable / total if total else 0.0

    return {
        "total_examples": total,
        "scorable_examples": len(scorable),
        "unscorable_examples": unscorable,
        "correct": correct,
        "incorrect": len(scorable) - correct,
        # Backward-compatible fields (scorable-based accuracy)
        "accuracy": round(accuracy_scorable, 6),
        "accuracy_pct": round(accuracy_scorable * 100, 2),
        # Explicit metrics for honest pre/post comparison
        "accuracy_scorable": round(accuracy_scorable, 6),
        "accuracy_scorable_pct": round(accuracy_scorable * 100, 2),
        "accuracy_total": round(accuracy_total, 6),
        "accuracy_total_pct": round(accuracy_total * 100, 2),
        "unscorable_rate": round(unscorable_rate, 6),
        "unscorable_rate_pct": round(unscorable_rate * 100, 2),
    }
