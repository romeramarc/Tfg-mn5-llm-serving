"""
serving/start_server.py
=======================
Entry-point for launching a vLLM OpenAI-compatible API server.

This script:
  1. Loads ``configs/serving.yaml`` **and** ``configs/models.yaml``.
  2. Resolves the model identifier from the model registry.
  3. Logs structured metadata (timestamp, SLURM job ID, GPU info, git hash).
  4. Constructs the ``vllm serve …`` command.
  5. Replaces the current process with ``vllm serve`` via ``os.execvp``.

Usage
-----
    python -m serving.start_server [--config configs/serving.yaml]
                                   [--models configs/models.yaml]
                                   [--role teacher]

The server exposes ``/v1/completions``, ``/v1/chat/completions``, and
``/health`` endpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from utils.config_loader import load_yaml
from utils.logging import setup_logging, get_logger
from serving.server_utils import build_vllm_serve_cmd, server_metadata


def _resolve_model(serving_cfg: dict, models_cfg: dict, role: str) -> dict:
    """Override ``vllm.model`` from the model registry if available."""
    role_entry = models_cfg.get(role, {})
    model_name = role_entry.get("name")
    if model_name:
        serving_cfg.setdefault("vllm", {})["model"] = model_name
        # Propagate all vllm-relevant fields from the model registry entry
        for key in ("dtype", "trust_remote_code", "tensor_parallel_size"):
            if role_entry.get(key) is not None:
                serving_cfg["vllm"][key] = role_entry[key]
    return serving_cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch vLLM OpenAI-compatible serving endpoint",
    )
    parser.add_argument(
        "--config", default="configs/serving.yaml",
        help="Path to the serving YAML configuration file.",
    )
    parser.add_argument(
        "--models", default="configs/models.yaml",
        help="Path to the model registry YAML.",
    )
    parser.add_argument(
        "--role", default="teacher",
        choices=["teacher", "student_mid", "student_small", "dev"],
        help="Which model role to serve (resolved from models.yaml).",
    )
    args = parser.parse_args()

    # ── Load config & init logging ──────────────────────────
    cfg = load_yaml(args.config)
    models_cfg = load_yaml(args.models)
    cfg = _resolve_model(cfg, models_cfg, args.role)

    log_cfg = cfg.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        fmt=log_cfg.get("format", "json"),
        log_dir=log_cfg.get("log_dir"),
    )
    logger = get_logger("serving.start_server")

    # ── Log metadata ────────────────────────────────────────
    meta = server_metadata(cfg)
    logger.info("Launching vLLM server",
                extra={"role": args.role,
                       "metadata": json.dumps(meta, default=str)})

    # ── Build command and exec ──────────────────────────────
    cmd = build_vllm_serve_cmd(cfg)
    logger.info("Exec command", extra={"cmd": " ".join(cmd)})

    # Replace the process image with vllm serve
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
