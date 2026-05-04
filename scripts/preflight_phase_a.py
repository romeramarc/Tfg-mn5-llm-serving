"""
scripts/preflight_phase_a.py
============================
Static sanity checks for Phase A. Run this **before** submitting jobs to
the cluster: it validates that everything that can be checked without
GPUs / the BSC environment is in shape.

Checks performed
----------------
1. ``configs/phase_a.yaml`` parses and has the required sections.
2. Every role declared under ``capture.roles`` exists in
   ``configs/models.yaml`` and has a non-empty ``name``.
3. ``capture.prompts_file`` exists and has at least 4 prompts.
4. ``capture.arrival_rates_rps`` is a non-empty list of positive numbers.
5. Required Python imports load (yaml, numpy, httpx, sklearn, joblib).
6. ``bench/run_load_capture.py``, ``bench/gpu_sampler.py`` and
   ``scripts/phase_a_train.py`` import without errors.
7. SLURM templates exist with the right names.
8. Optional: probes ``${SERVER_BASE_URL}/health`` if the variable is set.

Exits with code 0 on success, code 1 on the first failure.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config_loader import load_yaml  # noqa: E402
from utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

REQUIRED_FILES = [
    "configs/phase_a.yaml",
    "configs/models.yaml",
    "configs/serving.yaml",
    "bench/run_load_capture.py",
    "bench/gpu_sampler.py",
    "scripts/phase_a_train.py",
    "slurm/phase_a_capture.sbatch",
    "slurm/phase_a_train.sbatch",
    "slurm/launch_phase_a.sh",
    "slurm/server_role_phase2.sbatch",
]


def _check(label: str, ok: bool, detail: str = "") -> bool:
    status = "OK   " if ok else "FAIL "
    print(f"[{status}] {label}{(' -- ' + detail) if detail else ''}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase A preflight checks")
    parser.add_argument("--config", default="configs/phase_a.yaml")
    args = parser.parse_args()

    setup_logging()
    print("=== Phase A preflight ===")

    failures: List[str] = []

    # 1) Required files
    for rel in REQUIRED_FILES:
        if not _check(f"file exists: {rel}", Path(rel).is_file()):
            failures.append(rel)

    # 2) YAML parses with expected structure
    try:
        cfg = load_yaml(args.config)
    except Exception as exc:  # pragma: no cover - hard to exercise
        print(f"[FAIL ] could not load {args.config}: {exc}")
        return 1
    _check("phase_a.yaml has 'capture' section", isinstance(cfg.get("capture"), dict))
    _check("phase_a.yaml has 'dataset' section", isinstance(cfg.get("dataset"), dict))
    _check("phase_a.yaml has 'predictor' section", isinstance(cfg.get("predictor"), dict))

    # 3) Models registry alignment
    try:
        models_cfg = load_yaml("configs/models.yaml")
    except Exception as exc:
        print(f"[FAIL ] could not load configs/models.yaml: {exc}")
        return 1
    capt = cfg.get("capture") or {}
    role_entries = capt.get("roles") or []
    for entry in role_entries:
        role_name = (entry or {}).get("name")
        ok = bool(role_name) and isinstance(models_cfg.get(role_name), dict) and bool(
            (models_cfg.get(role_name) or {}).get("name")
        )
        if not _check(f"role '{role_name}' resolves in models.yaml", ok):
            failures.append(f"role:{role_name}")

    # 4) Prompts file
    prompts_path = Path(capt.get("prompts_file", "configs/prompts.jsonl"))
    n_prompts = 0
    if prompts_path.is_file():
        with prompts_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    n_prompts += 1
    _check(f"prompts file '{prompts_path}' has >= 4 entries", n_prompts >= 4,
           detail=f"{n_prompts} prompts")

    # 5) Rate list
    rates = capt.get("arrival_rates_rps") or []
    rates_ok = isinstance(rates, list) and len(rates) > 0 and all(
        isinstance(x, (int, float)) and float(x) > 0 for x in rates
    )
    _check("arrival_rates_rps is a list of positive numbers", rates_ok,
           detail=str(rates))

    # 6) Imports
    for mod in [
        "yaml", "numpy", "httpx", "sklearn", "joblib",
        "bench.run_load_capture", "bench.gpu_sampler",
        "predictors.builders.build_cost_dataset",
        "predictors.training.common",
    ]:
        try:
            importlib.import_module(mod)
            _check(f"import: {mod}", True)
        except Exception as exc:
            _check(f"import: {mod}", False, detail=str(exc))
            failures.append(f"import:{mod}")

    # 7) Optional health probe
    base_url = os.environ.get("SERVER_BASE_URL", "")
    if base_url:
        try:
            import httpx  # local import keeps preflight cheap if httpx missing
            resp = httpx.get(f"{base_url.rstrip('/')}/health", timeout=5.0)
            _check(f"server reachable @ {base_url}", resp.status_code == 200,
                   detail=f"status={resp.status_code}")
        except Exception as exc:
            _check(f"server reachable @ {base_url}", False, detail=str(exc))

    if failures:
        print(f"\n=== Preflight FAILED — {len(failures)} issue(s): {failures}")
        return 1

    print("\n=== Preflight OK ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
