"""
serving/server_utils.py
=======================
Helpers consumed by the serving entry-point.

* Build the metadata dict logged on every server start.
* Construct the ``vllm serve`` CLI command from a config dict.
"""

from __future__ import annotations

import os
import socket
import time
from typing import Any, Dict, List, Optional

from utils.reproducibility import git_commit_hash


def server_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a dict of contextual metadata for structured logging."""
    import subprocess as _sp

    meta: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "timestamp": time.time(),
        "git_commit": git_commit_hash(),
        "model": config.get("vllm", {}).get("model"),
        "tensor_parallel_size": config.get("vllm", {}).get("tensor_parallel_size"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }

    # GPU info via nvidia-smi (best-effort)
    try:
        nv_out = _sp.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=_sp.DEVNULL, text=True,
        )
        gpus = []
        for line in nv_out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                gpus.append({"name": parts[0], "vram_mb": int(parts[1])})
        meta["gpus"] = gpus
    except Exception:
        meta["gpus"] = []

    return meta


def build_vllm_serve_cmd(config: Dict[str, Any]) -> List[str]:
    """Translate *config* into a ``vllm serve …`` argument list.

    The returned list is ready for ``subprocess.Popen`` / ``os.execvp``.
    """
    vllm_cfg = config.get("vllm", {})
    srv_cfg = config.get("server", {})
    api_cfg = config.get("api", {})

    cmd: List[str] = [
        "vllm", "serve", vllm_cfg["model"],
        "--host", str(srv_cfg.get("host", "0.0.0.0")),
        "--port", str(srv_cfg.get("port", 8000)),
        "--tensor-parallel-size", str(vllm_cfg.get("tensor_parallel_size", 1)),
        "--max-model-len", str(vllm_cfg.get("max_model_len", 4096)),
        "--gpu-memory-utilization", str(vllm_cfg.get("gpu_memory_utilization", 0.90)),
        "--dtype", str(vllm_cfg.get("dtype", "auto")),
        "--swap-space", str(vllm_cfg.get("swap_space", 4)),
        "--max-num-seqs", str(vllm_cfg.get("max_num_seqs", 256)),
        "--seed", str(vllm_cfg.get("seed", 0)),
    ]

    if vllm_cfg.get("enforce_eager"):
        cmd.append("--enforce-eager")

    if vllm_cfg.get("trust_remote_code"):
        cmd.append("--trust-remote-code")

    if api_cfg.get("api_key"):
        cmd.extend(["--api-key", api_cfg["api_key"]])

    if api_cfg.get("chat_template"):
        cmd.extend(["--chat-template", api_cfg["chat_template"]])

    return cmd
