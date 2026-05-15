#!/usr/bin/env python3
"""Preflight for 1.5B-base capture (student_small_base). Run before sbatch."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.config_loader import load_yaml  # noqa: E402


def main() -> int:
    ok = True
    models = load_yaml(REPO / "configs" / "models.yaml")
    role = "student_small_base"
    entry = models.get(role)
    if not entry or not str(entry.get("name", "")).strip():
        print(f"[FAIL] configs/models.yaml missing role '{role}'")
        ok = False
    else:
        print(f"[OK]   {role} -> {entry['name']}")

    for cfg_name in (
        "phase_a_capture_base_1p5b.yaml",
        "phase_b_capture_base_1p5b.yaml",
        "phase_a_train_ladder_base_1p5b.yaml",
        "phase_b_train_ladder_base_1p5b.yaml",
    ):
        p = REPO / "configs" / cfg_name
        if not p.is_file():
            print(f"[FAIL] missing {p}")
            ok = False
            continue
        cfg = load_yaml(p)
        roles = [r["name"] for r in (cfg.get("capture") or {}).get("roles") or []]
        if cfg_name.startswith("phase_a_capture") or cfg_name.startswith("phase_b_capture"):
            if roles != [role]:
                print(f"[FAIL] {cfg_name} capture.roles expected [{role}], got {roles}")
                ok = False
            else:
                print(f"[OK]   {cfg_name} roles={roles}")

    for sb in (
        "slurm/server_role_phase2.sbatch",
        "slurm/phase_a_capture.sbatch",
        "slurm/phase_b_capture.sbatch",
        "slurm/launch_capture_base_1p5b.sh",
    ):
        if not (REPO / sb).is_file():
            print(f"[FAIL] missing {sb}")
            ok = False
        else:
            print(f"[OK]   {sb}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
