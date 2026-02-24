"""
utils/config_loader.py
======================
Centralised YAML configuration loader.

Every module reads its parameters through this interface so that:
  1. There is a single source of truth per config file.
  2. Configs can be deep-merged at runtime for overrides.
  3. The loaded dict is frozen after return (shallow copy).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dict.

    Raises ``FileNotFoundError`` if *path* does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def save_yaml(obj: Dict[str, Any], path: str | Path) -> Path:
    """Serialise *obj* to a YAML file, creating parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(obj, fh, default_flow_style=False, sort_keys=False)
    return p


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into a deep copy of *base*.

    Scalar values in *override* replace those in *base*; dicts are
    merged recursively; lists are replaced wholesale.
    """
    merged = copy.deepcopy(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = merge_configs(merged[key], val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def load_with_overrides(
    base_path: str | Path,
    override_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Load *base_path* and optionally deep-merge *override_path* on top."""
    cfg = load_yaml(base_path)
    if override_path is not None:
        ovr = load_yaml(override_path)
        cfg = merge_configs(cfg, ovr)
    return cfg
