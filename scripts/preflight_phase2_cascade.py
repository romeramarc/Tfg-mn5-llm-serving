#!/usr/bin/env python3
"""
Preflight checks for phase-2 cascade launch on BSC.

Usage:
  python scripts/preflight_phase2_cascade.py
"""

from __future__ import annotations

import importlib
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config_loader import load_yaml


def ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    raise SystemExit(1)


def check_files() -> None:
    required = [
        ROOT / "configs" / "routing_phase2.yaml",
        ROOT / "slurm" / "server_role_phase2.sbatch",
        ROOT / "slurm" / "eval_cascade_phase2.sbatch",
        ROOT / "slurm" / "launch_cascade_phase2.sh",
        ROOT / "routing" / "router.py",
        ROOT / "routing" / "cascade_quality.py",
        ROOT / "routing" / "policies.py",
    ]
    for p in required:
        if not p.exists():
            fail(f"Missing required file: {p}")
    ok("All required phase-2 files exist")


def check_config_schema() -> None:
    cfg = load_yaml(ROOT / "configs" / "routing_phase2.yaml")

    if cfg.get("active_policy") != "cascade_three_tier":
        fail("configs/routing_phase2.yaml active_policy must be cascade_three_tier")

    for ep in ("teacher", "student_mid", "student_small"):
        if ep not in cfg.get("endpoints", {}):
            fail(f"Missing endpoint '{ep}' in routing_phase2.yaml")
        if "model" not in cfg["endpoints"][ep]:
            fail(f"Missing model for endpoint '{ep}'")

    p = cfg.get("policies", {}).get("cascade_three_tier", {})
    for key in (
        "small_confidence_threshold",
        "mid_confidence_threshold",
        "small_timeout_ms",
        "mid_timeout_ms",
        "teacher_timeout_ms",
    ):
        if key not in p:
            fail(f"Missing cascade_three_tier.{key}")

    ok("routing_phase2.yaml schema looks valid")


def check_python_deps() -> None:
    deps = ["yaml", "httpx", "numpy", "torch", "datasets", "sympy"]
    missing = []
    for mod in deps:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)

    if missing:
        fail(
            "Missing Python modules in current environment: "
            + ", ".join(missing)
            + "\nInstall requirements in the target environment before launch."
        )
    ok("Required Python modules import correctly")


def check_launcher_mode() -> None:
    p = ROOT / "slurm" / "launch_cascade_phase2.sh"
    mode = p.stat().st_mode
    if mode & stat.S_IXUSR:
        ok("launch_cascade_phase2.sh is executable")
    else:
        warn("launch_cascade_phase2.sh is not executable (run: chmod +x slurm/launch_cascade_phase2.sh)")


def check_project_path_hint() -> None:
    expected = "/gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving"
    txt = (ROOT / "slurm" / "eval_cascade_phase2.sbatch").read_text(encoding="utf-8")
    if expected in txt:
        ok("BSC project path is configured in eval_cascade_phase2.sbatch")
    else:
        warn("Expected BSC project path not found in eval_cascade_phase2.sbatch")


def main() -> None:
    print("=== Phase-2 Cascade Preflight ===")
    check_files()
    check_config_schema()
    check_python_deps()
    check_launcher_mode()
    check_project_path_hint()
    print("=== Preflight complete: ready to push/pull/launch ===")


if __name__ == "__main__":
    main()
