"""Resolve vLLM endpoint URLs and model names for routing evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from utils.config_loader import load_yaml


def resolve_endpoints(
    roles: List[str],
    *,
    models_config: str = "configs/models.yaml",
    endpoint_dir: str = "results/routing/endpoints",
) -> Dict[str, Dict[str, str]]:
    models = load_yaml(models_config)
    base = Path(endpoint_dir)
    out: Dict[str, Dict[str, str]] = {}

    for role in roles:
        url_file = base / f"{role}.url"
        if not url_file.is_file():
            raise FileNotFoundError(f"Missing endpoint URL file: {url_file}")
        url = url_file.read_text(encoding="utf-8").strip()
        if not url.startswith("http"):
            raise ValueError(f"Invalid URL in {url_file}: {url!r}")
        model_entry = models.get(role)
        if not model_entry:
            raise KeyError(f"Role '{role}' not found in {models_config}")
        out[role] = {
            "base_url": url,
            "model": str(model_entry["name"]),
            "role": role,
        }
    return out
