#!/usr/bin/env python3
"""Quick checks before launching holdout evaluation on MN5."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.config_loader import load_yaml  # noqa: E402


def main() -> int:
    root = _ROOT
    cfg_path = root / "configs" / "routing_eval_holdout.yaml"
    cfg = load_yaml(str(cfg_path))

    errors: list[str] = []

    from routing.policies import POLICIES

    for system in cfg.get("systems") or []:
        policy = system.get("policy")
        if policy not in POLICIES:
            errors.append(f"Unknown policy '{policy}' for {system.get('id')}")

    pred = cfg.get("predictors") or {}
    for key, rel in (pred.get("bundles") or {}).items():
        path = root / rel
        if path.is_file():
            continue
        alt = path.parent / "bundle.joblib"
        if alt.is_file():
            print(f"[WARN] {key}: using {alt} (update config to match)")
            continue
        found = sorted((root / "results").rglob("model_bundle.joblib"))
        hint = ""
        if found:
            hint = f"  found elsewhere, e.g. {found[0]}"
        errors.append(f"Missing predictor bundle ({key}): {path}{hint}")

    for script in (
        "bench/holdout_pool.py",
        "routing/run_eval_holdout.py",
        "slurm/eval_holdout.sbatch",
        "slurm/launch_eval_holdout.sh",
    ):
        if not (root / script).is_file():
            errors.append(f"Missing file: {script}")

    try:
        import httpx  # noqa: F401
        import numpy  # noqa: F401
        import sklearn  # noqa: F401
        import joblib  # noqa: F401
    except ImportError as exc:
        errors.append(f"Missing Python dependency: {exc}")

    if errors:
        print("PREFLIGHT FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("PREFLIGHT OK")
    print(f"  config: {cfg_path}")
    print(f"  systems: {[s['id'] for s in cfg.get('systems', [])]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
