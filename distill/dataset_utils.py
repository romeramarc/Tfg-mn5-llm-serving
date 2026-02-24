"""
distill/dataset_utils.py
========================
I/O helpers for JSONL datasets used by the distillation pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Read a JSONL file and return a list of dicts."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {p.resolve()}")
    items: list[dict] = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {lineno} of {p}: {exc}"
                ) from exc
    return items


def write_jsonl(items: List[Dict[str, Any]], path: str | Path) -> Path:
    """Write a list of dicts as a JSONL file, creating parents as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    return p


def load_prompts(path: str | Path) -> List[str]:
    """Load prompt strings from a JSONL file.

    Each line must be a JSON object with a ``"prompt"`` key.
    """
    rows = read_jsonl(path)
    prompts: list[str] = []
    for row in rows:
        text = row.get("prompt")
        if text is None:
            raise KeyError(f"Missing 'prompt' key in row: {row}")
        prompts.append(text)
    return prompts
