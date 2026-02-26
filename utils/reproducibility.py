"""
utils/reproducibility.py
========================
Helpers that enforce deterministic, auditable experiment runs.

Responsibilities
----------------
* Set global seeds (Python, NumPy, PyTorch).
* Create timestamped run directories.
* Copy active configs into the run directory.
* Collect hardware / environment metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch


# ── Seed management ─────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── Run directory ───────────────────────────────────────────

def make_run_dir(base: str | Path, tag: Optional[str] = None) -> Path:
    """Create ``<base>/<tag>-<ISO-timestamp>`` and return the path.

    The directory name uses UTC and is unique to the second.
    """
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{tag}-{ts}" if tag else ts
    run_dir = Path(base) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def snapshot_configs(config_paths: Sequence[str | Path], run_dir: str | Path) -> None:
    """Copy every listed config file into *run_dir* for reproducibility."""
    dst = Path(run_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for src in config_paths:
        src = Path(src)
        if src.is_file():
            shutil.copy2(src, dst / src.name)


# ── Git metadata ────────────────────────────────────────────

def git_commit_hash() -> Optional[str]:
    """Return the short HEAD hash, or ``None`` outside a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


# ── Hardware / environment metadata ─────────────────────────

def collect_metadata(seed: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """Gather a snapshot of software + hardware state for the run log."""
    meta: Dict[str, Any] = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "hostname": platform.node(),
        "seed": seed,
        "git_commit": git_commit_hash(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "config_hash": hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest()[:12],
    }

    # Package versions (best-effort)
    for pkg_name in ("vllm", "transformers", "datasets", "peft", "accelerate"):
        try:
            mod = __import__(pkg_name)
            meta[f"{pkg_name}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            meta[f"{pkg_name}_version"] = "not_installed"

    # SLURM metadata (available inside jobs)
    for var in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_NODELIST",
                "SLURM_GPUS_ON_NODE", "SLURM_NTASKS"):
        val = os.environ.get(var)
        if val is not None:
            meta[var.lower()] = val

    # GPU details — names and VRAM
    if torch.cuda.is_available():
        meta["gpu_count"] = torch.cuda.device_count()
        gpu_info: List[Dict[str, Any]] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                "index": i,
                "name": props.name,
                "total_memory_mb": round(props.total_memory / (1024 ** 2)),
            })
        meta["gpus"] = gpu_info
        meta["gpu_names"] = [g["name"] for g in gpu_info]

    return meta


def save_metadata(meta: Dict[str, Any], run_dir: str | Path) -> Path:
    """Persist metadata dict as ``run_meta.json`` inside *run_dir*."""
    p = Path(run_dir) / "run_meta.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    return p
