"""
scripts/preflight_phase_b.py
============================
Static sanity checks for Phase B. Run this **before** submitting jobs to
the cluster: it validates that everything that can be checked without
GPUs / the BSC environment is in shape.

Checks performed
----------------
1. ``configs/phase_b.yaml`` parses and has the required sections.
2. Every role declared under ``capture.roles`` exists in
   ``configs/models.yaml`` and has a non-empty ``name``.
3. At least one benchmark in ``capture.benchmarks`` is enabled, has a
   prompt template, and has a positive subset_size (or null).
4. ``capture.arrival_rates_rps`` is a non-empty list of positive numbers.
5. ``capture.logprobs_top_k`` is a positive integer (uncertainty features
   for the post-hoc predictor depend on it).
6. Required Python imports load (yaml, numpy, httpx, sklearn, joblib,
   datasets, sympy if MATH benchmark enabled).
7. ``bench/run_quality_capture.py``, ``bench/gpu_sampler.py`` and
   ``scripts/phase_b_train.py`` import without errors.
8. SLURM templates exist with the right names.
9. Optional: probes ``${SERVER_BASE_URL}/health`` if the variable is set.

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
    "configs/phase_b.yaml",
    "configs/models.yaml",
    "configs/serving.yaml",
    "bench/run_quality_capture.py",
    "bench/run_load_capture.py",
    "bench/gpu_sampler.py",
    "eval/scoring.py",
    "scripts/phase_b_train.py",
    "slurm/phase_b_capture.sbatch",
    "slurm/phase_b_train.sbatch",
    "slurm/launch_phase_b.sh",
    "slurm/server_role_phase2.sbatch",
]


def _check(label: str, ok: bool, detail: str = "") -> bool:
    status = "OK   " if ok else "FAIL "
    print(f"[{status}] {label}{(' -- ' + detail) if detail else ''}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase B preflight checks")
    parser.add_argument("--config", default="configs/phase_b.yaml")
    args = parser.parse_args()

    setup_logging()
    print("=== Phase B preflight ===")

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
    _check("phase_b.yaml has 'capture' section", isinstance(cfg.get("capture"), dict))
    _check("phase_b.yaml has 'dataset' section", isinstance(cfg.get("dataset"), dict))
    _check("phase_b.yaml has 'predictor' section", isinstance(cfg.get("predictor"), dict))

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

    # 4) Benchmarks
    benchmarks_cfg = capt.get("benchmarks") or {}
    enabled_benchmarks: List[str] = []
    for bname in ("gsm8k", "math"):
        bcfg = benchmarks_cfg.get(bname) or {}
        if not bcfg.get("enabled", False):
            _check(f"benchmark '{bname}' enabled", False, detail="disabled")
            continue
        enabled_benchmarks.append(bname)
        template_ok = bool(str(bcfg.get("prompt_template", "")).strip())
        _check(f"benchmark '{bname}' has prompt_template", template_ok)
        subset = bcfg.get("subset_size")
        ok_subset = subset is None or (isinstance(subset, int) and subset > 0)
        _check(
            f"benchmark '{bname}' subset_size valid",
            ok_subset,
            detail=str(subset),
        )
        if not (template_ok and ok_subset):
            failures.append(f"benchmark:{bname}")

    if not enabled_benchmarks:
        print("[FAIL ] at least one benchmark must be enabled in capture.benchmarks")
        failures.append("benchmarks:none_enabled")

    # 5) Rate list
    rates = capt.get("arrival_rates_rps") or []
    rates_ok = isinstance(rates, list) and len(rates) > 0 and all(
        isinstance(x, (int, float)) and float(x) > 0 for x in rates
    )
    _check("arrival_rates_rps is a list of positive numbers", rates_ok,
           detail=str(rates))
    if not rates_ok:
        failures.append("rates")

    # 6) Logprobs (uncertainty features for quality_post_hoc)
    logprobs_k = capt.get("logprobs_top_k")
    lp_ok = isinstance(logprobs_k, int) and logprobs_k > 0
    _check(
        "logprobs_top_k is a positive integer",
        lp_ok,
        detail=str(logprobs_k),
    )
    if not lp_ok:
        failures.append("logprobs_top_k")

    # 7) Imports — common Python stack
    base_imports = [
        "yaml", "numpy", "httpx", "sklearn", "joblib", "datasets",
    ]
    if "math" in enabled_benchmarks:
        base_imports.append("sympy")
    for mod in base_imports:
        try:
            importlib.import_module(mod)
            _check(f"import: {mod}", True)
        except Exception as exc:
            _check(f"import: {mod}", False, detail=str(exc))
            failures.append(f"import:{mod}")

    # 8) Imports — Phase B project modules
    for mod in [
        "bench.run_quality_capture",
        "bench.run_load_capture",
        "bench.gpu_sampler",
        "eval.scoring",
        "predictors.builders.build_ex_ante_dataset",
        "predictors.builders.build_post_hoc_dataset",
        "predictors.training.common",
    ]:
        try:
            importlib.import_module(mod)
            _check(f"import: {mod}", True)
        except Exception as exc:
            _check(f"import: {mod}", False, detail=str(exc))
            failures.append(f"import:{mod}")

    # 9) Optional health probe
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
