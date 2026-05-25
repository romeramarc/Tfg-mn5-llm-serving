#!/usr/bin/env python3
"""Preflight checks for the phase 2 holdout (1.5B base vs distilled).

Validates that:
  1. The eval YAML pins ``models_config`` (or the system blocks do) so the
     client does not silently fall back to ``configs/models.yaml`` (which
     contains the distilled model path).
  2. The ``models_config`` file resolves the expected model name per rung.
  3. Each endpoint URL file already exists (servers running) and the vLLM
     server answers a tiny ``/v1/completions`` request using the model
     name from ``models_config``.

Designed to run on MN5 login node before ``slurm/submit_holdout_phase2.sh``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import urllib.error
import urllib.request

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.config_loader import load_yaml  # noqa: E402


EXPECTED_PHASE2_MODELS = {
    "student_small": "Qwen/Qwen2.5-1.5B-Instruct",
    "student_q3b": "Qwen/Qwen2.5-3B-Instruct",
    "student_mid": "Qwen/Qwen2.5-7B-Instruct",
    "teacher": "Qwen/Qwen2.5-14B-Instruct",
}


def _read_endpoint_url(endpoint_dir: Path, role: str) -> str | None:
    f = endpoint_dir / f"{role}.url"
    if not f.is_file():
        return None
    url = f.read_text(encoding="utf-8").strip()
    if not url.startswith("http"):
        return None
    return url


def _ping_completion(base_url: str, model: str, timeout: float = 30.0) -> Dict[str, Any]:
    """Send a tiny completion request and return ``{ok, status, error}``."""
    payload = {
        "model": model,
        "prompt": "ping",
        "max_tokens": 1,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return {
                "ok": resp.status == 200,
                "status": resp.status,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "preview": body[:200],
            }
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return {
            "ok": False,
            "status": exc.code,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "error": str(exc),
            "preview": body[:300],
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": None,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 holdout preflight (1.5B base)")
    parser.add_argument(
        "--config",
        default="configs/routing_eval_holdout_v2_phase2.yaml",
    )
    parser.add_argument(
        "--ping",
        action="store_true",
        help="Send a real /v1/completions request to each endpoint",
    )
    args = parser.parse_args()

    cfg_path = _ROOT / args.config
    if not cfg_path.is_file():
        print(f"PREFLIGHT FAILED: config not found: {cfg_path}")
        return 1

    cfg = load_yaml(str(cfg_path))
    errors: List[str] = []
    warnings: List[str] = []

    exec_cfg = cfg.get("execution_plan") or {}
    endpoint_dir = _ROOT / str(exec_cfg.get("endpoint_dir", "results/routing/endpoints"))
    global_models_config = exec_cfg.get("models_config")

    systems = cfg.get("systems") or []
    if not systems:
        errors.append("config has no systems")

    for system in systems:
        sid = system.get("id")
        roles = list(system.get("roles") or [])
        sys_models_config = system.get("models_config") or global_models_config
        if not sys_models_config:
            errors.append(
                f"{sid}: missing models_config (in system block or execution_plan); "
                "client would default to configs/models.yaml (distilled student_small)"
            )
            continue

        models_path = _ROOT / str(sys_models_config)
        if not models_path.is_file():
            errors.append(f"{sid}: models_config not found: {models_path}")
            continue

        models = load_yaml(str(models_path))
        for role in roles:
            entry = models.get(role)
            if not entry or not entry.get("name"):
                errors.append(f"{sid}: role '{role}' missing in {models_path}")
                continue
            expected = EXPECTED_PHASE2_MODELS.get(role)
            if expected and entry["name"] != expected:
                warnings.append(
                    f"{sid}: role '{role}' resolves to {entry['name']!r} "
                    f"(expected {expected!r})"
                )

            url = _read_endpoint_url(endpoint_dir, role)
            if url is None:
                errors.append(
                    f"{sid}: endpoint {endpoint_dir/role}.url missing or invalid; "
                    "server not running"
                )
                continue

            if args.ping:
                res = _ping_completion(url, entry["name"])
                marker = "OK" if res["ok"] else "FAIL"
                line = (
                    f"  [{marker}] {sid} {role} {entry['name']} -> "
                    f"{url} status={res.get('status')} ({res.get('elapsed_ms', 0):.0f}ms)"
                )
                print(line)
                if not res["ok"]:
                    preview = res.get("preview") or res.get("error") or ""
                    errors.append(
                        f"{sid}: ping failed for {role} at {url} with model "
                        f"{entry['name']!r}: status={res.get('status')} "
                        f"body={preview[:200]!r}"
                    )

    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    if errors:
        print("PREFLIGHT FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("PREFLIGHT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
