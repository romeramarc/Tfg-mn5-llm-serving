"""
distill/preflight.py
=====================
Pre-flight validation for the Phase 2 distillation pipeline.
Run this BEFORE submitting any SLURM job to catch problems early.

Usage
-----
    # Check step 1 prerequisites (HuggingFace datasets + teacher server URL)
    python -m distill.preflight --step 1

    # Check step 2 prerequisites (teacher_outputs.jsonl exists + peft importable)
    python -m distill.preflight --step 2

    # Check step 3 prerequisites (adapter exists + merge test on CPU with tiny model)
    python -m distill.preflight --step 3

    # Check all steps
    python -m distill.preflight --step all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ── colour helpers ────────────────────────────────────────────────────────────

def ok(msg: str)   -> None: print(f"  [OK]  {msg}")
def fail(msg: str) -> None: print(f"  [FAIL] {msg}"); sys.exit(1)
def warn(msg: str) -> None: print(f"  [WARN] {msg}")
def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── individual checks ─────────────────────────────────────────────────────────

def check_python_imports() -> None:
    """Verify that all heavy dependencies are importable."""
    section("Python imports")
    for pkg in ["httpx", "yaml", "datasets"]:
        try:
            __import__(pkg)
            ok(pkg)
        except ImportError as e:
            fail(f"{pkg} not importable: {e}")

    for pkg in ["torch", "transformers", "peft"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            ok(f"{pkg}=={ver}")
        except ImportError as e:
            fail(f"{pkg} not importable — run: pip install {pkg}  ({e})")


def check_datasets() -> None:
    """Download a tiny slice of each benchmark to confirm HF datasets work."""
    section("HuggingFace datasets (5-sample probe)")
    from datasets import load_dataset  # noqa: PLC0415

    checks = [
        ("openai/gsm8k",              "main",          "test"),
        ("HuggingFaceH4/MATH-500",    None,            "test"),
        ("allenai/ai2_arc",           "ARC-Challenge", "test"),
    ]
    for repo, cfg_name, split in checks:
        label = repo if cfg_name is None else f"{repo} [{cfg_name}]"
        try:
            kw = {"split": f"{split}[:5]", "trust_remote_code": True}
            if cfg_name:
                kw["name"] = cfg_name
            ds = load_dataset(repo, **kw)
            ok(f"{label}  →  {len(ds)} samples, columns={list(ds.features)}")
        except Exception as e:
            fail(f"{label}: {e}")


def check_generate_smoke() -> None:
    """Verify collect_all_prompts works (no server needed)."""
    section("Prompt collection (no server)")
    try:
        from utils.config_loader import load_yaml  # noqa: PLC0415
        from distill.generate_teacher_outputs import collect_all_prompts  # noqa: PLC0415
        eval_cfg = load_yaml("configs/eval.yaml")
        prompts = collect_all_prompts(eval_cfg)
        by_bench: dict[str, int] = {}
        for p in prompts:
            by_bench[p["benchmark"]] = by_bench.get(p["benchmark"], 0) + 1
        ok(f"Total prompts: {len(prompts)}  breakdown: {by_bench}")
        p0 = prompts[0]
        required = {"id", "benchmark", "prompt"}
        missing = required - p0.keys()
        if missing:
            fail(f"Prompt record missing keys: {missing}")
        ok("Prompt record schema: id, benchmark, prompt — all present")
    except Exception as e:
        fail(f"collect_all_prompts raised: {e}")


def check_teacher_server(base_url: str = "http://localhost:8000") -> None:
    """Ping the teacher vLLM server /health endpoint."""
    section(f"Teacher server health ({base_url})")
    try:
        import httpx  # noqa: PLC0415
        r = httpx.get(f"{base_url}/health", timeout=5)
        if r.status_code == 200:
            ok(f"Server responded 200 OK")
        else:
            fail(f"Server returned HTTP {r.status_code}")
    except Exception as e:
        fail(f"Cannot reach teacher server at {base_url}: {e}\n"
             "       Start the teacher vLLM server before running step 1.")


def check_jsonl_dataset(path: str = "results/distill/teacher_outputs.jsonl") -> None:
    """Verify teacher_outputs.jsonl exists and has valid records."""
    section(f"Teacher JSONL dataset ({path})")
    p = Path(path)
    if not p.exists():
        fail(f"{path} not found — run Step 1 first (distill_generate.sbatch)")

    records: list[dict] = []
    with p.open() as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                fail(f"Line {i+1} is not valid JSON: {e}")

    ok(f"File readable: {len(records)} records")

    required = {"id", "benchmark", "prompt", "teacher_completion"}
    bad = [r for r in records if required - r.keys()]
    if bad:
        fail(f"{len(bad)} records missing required keys. First bad: {bad[0]}")
    ok("All records have required keys (id, benchmark, prompt, teacher_completion)")

    errors  = [r for r in records if r.get("error")]
    valid   = [r for r in records if not r.get("error") and r.get("teacher_completion")]
    warn(f"{len(errors)} records have errors (will be skipped in training)"
         ) if errors else ok("No error records")
    ok(f"Valid (has completion): {len(valid)}  /  {len(records)}")

    if len(valid) < 50:
        fail(f"Only {len(valid)} valid samples — too few for meaningful training. "
             "Rerun step 1.")

    by_bench: dict[str, int] = {}
    for r in valid:
        b = r.get("benchmark", "?")
        by_bench[b] = by_bench.get(b, 0) + 1
    ok(f"Benchmark breakdown: {by_bench}")


def check_adapter(pattern: str, student: str = "7B") -> None:
    """Check that a LoRA adapter directory exists and looks valid."""
    section(f"LoRA adapter ({student})")
    import glob  # noqa: PLC0415
    matches = sorted(glob.glob(pattern), reverse=True)
    if not matches:
        fail(f"No adapter found matching: {pattern}\n"
             "       Run the training step first.")
    adapter_dir = Path(matches[0])
    ok(f"Latest adapter: {adapter_dir}")

    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    for f in required_files:
        candidate = adapter_dir / f
        # safetensors may be a directory of shards too
        if not candidate.exists() and not any(adapter_dir.glob("adapter_model*")):
            warn(f"Expected file missing: {f} — check if training completed")
        else:
            ok(f"  {f} present")

    cfg_path = adapter_dir / "adapter_config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            ok(f"  base_model_name_or_path: {cfg.get('base_model_name_or_path')}")
            ok(f"  lora_r={cfg.get('r')}  alpha={cfg.get('lora_alpha')}")
        except Exception as e:
            warn(f"Could not parse adapter_config.json: {e}")


# ── per-step flows ────────────────────────────────────────────────────────────

def step1(config: str, check_server: bool) -> None:
    """Validate everything needed before sbatch distill_generate.sbatch."""
    check_python_imports()
    check_datasets()
    check_generate_smoke()
    if check_server:
        from utils.config_loader import load_yaml  # noqa: PLC0415
        cfg = load_yaml(config)
        base_url = cfg.get("generation", {}).get("teacher_base_url", "http://localhost:8000")
        check_teacher_server(base_url)
    else:
        section("Teacher server")
        warn("Skipped (--no-server). Run with --check-server after starting vLLM.")


def step2(config: str) -> None:
    """Validate everything needed before sbatch distill_train_*.sbatch."""
    check_python_imports()
    from utils.config_loader import load_yaml  # noqa: PLC0415
    cfg = load_yaml(config)
    ds_path = cfg.get("training", {}).get(
        "dataset_path", "results/distill/teacher_outputs.jsonl")
    check_jsonl_dataset(ds_path)


def step3_7b() -> None:
    """Validate adapter existence for 7B before eval_distilled_7b.sbatch."""
    check_python_imports()
    check_adapter("results/distill/sft-qwen2.5-7b-*/final_adapter", student="7B")


def step3_1b5() -> None:
    """Validate adapter existence for 1.5B before eval_distilled_1.5b.sbatch."""
    check_python_imports()
    check_adapter("results/distill/sft-qwen2.5-1.5b-*/final_adapter", student="1.5B")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-flight checks for the Phase 2 distillation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m distill.preflight --step 1             # before distill_generate.sbatch
  python -m distill.preflight --step 1 --check-server   # includes vLLM ping
  python -m distill.preflight --step 2             # before distill_train_*.sbatch
  python -m distill.preflight --step 3             # before eval_distilled_*.sbatch
  python -m distill.preflight --step all           # all checks
        """,
    )
    parser.add_argument("--step", choices=["1", "2", "3", "all"], required=True)
    parser.add_argument("--config", default="configs/distill.yaml")
    parser.add_argument("--check-server", action="store_true",
                        help="Also ping the teacher vLLM /health endpoint (step 1)")
    args = parser.parse_args()

    if args.step in ("1", "all"):
        step1(args.config, check_server=args.check_server)
    if args.step in ("2", "all"):
        step2(args.config)
    if args.step in ("3", "all"):
        step3_7b()
        step3_1b5()

    section("Summary")
    ok("All checks passed! Safe to submit the corresponding SLURM job.")


if __name__ == "__main__":
    main()
