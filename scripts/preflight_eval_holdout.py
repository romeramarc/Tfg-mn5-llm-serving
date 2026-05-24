#!/usr/bin/env python3
"""Quick checks before launching holdout evaluation on MN5."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.config_loader import load_yaml  # noqa: E402

V2_EXPECTED_SYSTEMS = [
    "sysC_l010",
    "sysC_l025",
    "sysC_l050",
    "sysC_l100",
    "sysD_cascade4",
    "sysE_l010",
    "sysE_l025",
    "sysE_l050",
    "sysE_l100",
]

LAMBDA_PAIRS = [
    ("sysC_l010", "sysE_l010", 0.010),
    ("sysC_l025", "sysE_l025", 0.025),
    ("sysC_l050", "sysE_l050", 0.050),
    ("sysC_l100", "sysE_l100", 0.100),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight checks for holdout evaluation")
    parser.add_argument(
        "--config",
        default="configs/routing_eval_holdout.yaml",
        help="Eval YAML (use configs/routing_eval_holdout_v2_retrained.yaml for definitive v2 run)",
    )
    args = parser.parse_args()

    root = _ROOT
    cfg_path = root / args.config
    if not cfg_path.is_file():
        print(f"PREFLIGHT FAILED: config not found: {cfg_path}")
        return 1

    cfg = load_yaml(str(cfg_path))
    errors: list[str] = []

    from routing.policies import POLICIES

    systems = cfg.get("systems") or []
    system_ids = [str(s.get("id")) for s in systems if s.get("id")]

    for system in systems:
        policy = system.get("policy")
        sid = system.get("id")
        if policy not in POLICIES:
            errors.append(f"Unknown policy '{policy}' for {sid}")
        overrides = system.get("policy_overrides") or {}
        if policy in ("routing_predictive", "routing_plus_cascade"):
            block = overrides.get(policy) or {}
            lam = block.get("cost_weight_lambda")
            if lam is None and sid and "_l" in sid:
                errors.append(f"{sid}: missing cost_weight_lambda override")
            if lam is not None and sid:
                suffix = sid.rsplit("_l", 1)[-1] if "_l" in sid else ""
                if suffix.isdigit():
                    expected = int(suffix) / 1000.0
                    if abs(float(lam) - expected) > 1e-9:
                        errors.append(
                            f"{sid}: lambda override {lam} does not match id suffix (expected {expected})"
                        )

    # Legacy λ={0.01..0.1} grid only — not routing_real / no_distill / definitive runs.
    _LEGACY_V2_NAMES = {
        "routing_eval_holdout_v2.yaml",
        "routing_eval_holdout_v2_retrained.yaml",
    }
    if cfg_path.name in _LEGACY_V2_NAMES:
        missing = [s for s in V2_EXPECTED_SYSTEMS if s not in system_ids]
        if missing:
            errors.append(f"v2 config missing systems: {missing}")
        for c_id, e_id, lam in LAMBDA_PAIRS:
            if c_id not in system_ids or e_id not in system_ids:
                continue
            c = next(s for s in systems if s.get("id") == c_id)
            e = next(s for s in systems if s.get("id") == e_id)
            c_lam = ((c.get("policy_overrides") or {}).get("routing_predictive") or {}).get(
                "cost_weight_lambda"
            )
            e_lam = ((e.get("policy_overrides") or {}).get("routing_plus_cascade") or {}).get(
                "cost_weight_lambda"
            )
            if c_lam is None or e_lam is None or abs(float(c_lam) - float(e_lam)) > 1e-9:
                errors.append(f"{c_id} and {e_id}: lambda overrides are not matched")
            if c_lam is not None and abs(float(c_lam) - lam) > 1e-9:
                errors.append(f"{c_id}: expected lambda {lam}, got {c_lam}")
            if e_lam is not None and abs(float(e_lam) - lam) > 1e-9:
                errors.append(f"{e_id}: expected lambda {lam}, got {e_lam}")

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

    pool = (cfg.get("execution_plan") or {}).get("shared_prompt_pool")
    if pool and not (root / pool).is_file():
        errors.append(f"Prompt pool missing (build first or copy from v1): {root / pool}")

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
    print(f"  systems ({len(system_ids)}): {system_ids}")
    print(f"  results_base_dir: {(cfg.get('common') or {}).get('results_base_dir')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
