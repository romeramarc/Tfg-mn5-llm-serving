"""
distill/preflight.py
=====================
Pre-flight validation for the Phase 2 distillation pipeline.
Run this BEFORE submitting any SLURM job to catch problems early.

Usage
-----
    # Check config-level wiring for structured + GRPO line
    python -m distill.preflight --step config --config configs/distill_1p5b_structured_grpo.yaml --require-grpo

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
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse


# ── colour helpers ────────────────────────────────────────────────────────────

def ok(msg: str)   -> None: print(f"  [OK]  {msg}")
def fail(msg: str) -> None: print(f"  [FAIL] {msg}"); sys.exit(1)
def warn(msg: str) -> None: print(f"  [WARN] {msg}")
def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── individual checks ─────────────────────────────────────────────────────────

def check_python_imports(include_grpo: bool = False) -> None:
    """Verify that all heavy dependencies are importable."""
    section("Python imports")
    for pkg in ["httpx", "yaml", "datasets", "huggingface_hub"]:
        try:
            __import__(pkg)
            ok(pkg)
        except ImportError as e:
            fail(f"{pkg} not importable: {e}")

    heavy = ["torch", "transformers", "peft"]

    for pkg in heavy:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            ok(f"{pkg}=={ver}")
        except ImportError as e:
            fail(f"{pkg} not importable — run: pip install {pkg}  ({e})")

    if include_grpo:
        try:
            mod = __import__("vllm")
            ver = getattr(mod, "__version__", "?")
            ok(f"vllm=={ver}")
        except ImportError:
            warn("vllm not importable on this node (expected on login nodes without GPU; "
                 "will be available on compute nodes)")


def check_datasets(config_path: str = "configs/distill.yaml") -> None:
    """Probe only KD TRAIN datasets enabled in distill config."""
    section("HuggingFace datasets for KD (train-split probe)")
    from datasets import load_dataset  # noqa: PLC0415
    from utils.config_loader import load_yaml  # noqa: PLC0415

    cfg = load_yaml(config_path)
    benches = cfg.get("benchmarks", {})

    # GSM8K train
    gsm_cfg = benches.get("gsm8k", {})
    if gsm_cfg.get("enabled", True):
        try:
            ds = load_dataset(
                gsm_cfg.get("dataset_name", "openai/gsm8k"),
                "main",
                split=f"{gsm_cfg.get('dataset_split', 'train')}[:5]",
            )
            ok(
                f"{gsm_cfg.get('dataset_name', 'openai/gsm8k')} [main {gsm_cfg.get('dataset_split', 'train')}]"
                f"  →  {len(ds)} samples, columns={list(ds.features)}"
            )
        except Exception as e:
            fail(f"GSM8K probe failed: {e}")
    else:
        warn("GSM8K disabled in config; skipping probe")

    # MATH train
    math_cfg = benches.get("math", {})
    if math_cfg.get("enabled", True):
        dataset_name = math_cfg.get("dataset_name", "EleutherAI/hendrycks_math")
        dataset_split = math_cfg.get("dataset_split", "train")
        subjects = [
            "algebra", "counting_and_probability", "geometry",
            "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
        ]
        if dataset_name in {"hendrycks/competition_math", "EleutherAI/hendrycks_math"}:
            for subj in subjects:
                try:
                    ds = load_dataset(dataset_name, subj, split=f"{dataset_split}[:1]")
                    ok(f"{dataset_name} [{subj} {dataset_split}]  →  {len(ds)} sample")
                except Exception as e:
                    fail(f"{dataset_name} [{subj} {dataset_split}]: {e}")
        else:
            try:
                ds = load_dataset(dataset_name, split=f"{dataset_split}[:5]")
                ok(
                    f"{dataset_name} [{dataset_split}]  →  {len(ds)} samples, columns={list(ds.features)}"
                )
            except Exception as e:
                fail(f"{dataset_name} [{dataset_split}]: {e}")
    else:
        warn("MATH disabled in config; skipping probe")


def check_generate_smoke(config_path: str = "configs/distill.yaml") -> None:
    """Verify collect_all_prompts works (no server needed)."""
    section("Prompt collection (no server)")
    try:
        from utils.config_loader import load_yaml  # noqa: PLC0415
        from distill.generate_teacher_outputs import collect_all_prompts  # noqa: PLC0415
        distill_cfg = load_yaml(config_path)
        prompts = collect_all_prompts(distill_cfg)
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


def check_teacher_model_cache(config_path: str = "configs/distill.yaml") -> None:
    """Verify teacher model weights exist in local HF cache (offline-safe)."""
    section("Teacher model cache")
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from utils.config_loader import load_yaml  # noqa: PLC0415

    cfg = load_yaml(config_path)
    model = cfg.get("generation", {}).get("teacher_model", "Qwen/Qwen2.5-14B-Instruct")
    try:
        local_path = snapshot_download(repo_id=model, local_files_only=True)
        ok(f"{model} cached at: {local_path}")
    except Exception as e:
        fail(
            f"Teacher model not found in local HF cache: {model}\n"
            f"       Cache check error: {e}\n"
            "       Prime cache on a login node first (internet), then rerun preflight."
        )


@contextmanager
def offline_mode(enabled: bool):
    """Temporarily force offline mode to mimic compute-node constraints."""
    if not enabled:
        yield
        return

    old_hf = os.environ.get("HF_HUB_OFFLINE")
    old_tf = os.environ.get("TRANSFORMERS_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        yield
    finally:
        if old_hf is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old_hf
        if old_tf is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = old_tf


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


def check_jsonl_dataset(
    path: str = "results/distill/teacher_outputs.jsonl",
    required_benchmarks: tuple[str, ...] = ("gsm8k", "math"),
    min_valid_samples: int = 50,
) -> None:
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

    if len(valid) < min_valid_samples:
        fail(
            f"Only {len(valid)} valid samples — requires at least {min_valid_samples}. "
            "Rerun step 1 or reduce preflight.min_valid_samples for smoke mode."
        )

    by_bench: dict[str, int] = {}
    for r in valid:
        b = r.get("benchmark", "?")
        by_bench[b] = by_bench.get(b, 0) + 1
    ok(f"Benchmark breakdown: {by_bench}")

    for bench in required_benchmarks:
        if by_bench.get(bench, 0) == 0:
            fail(f"No valid samples for benchmark='{bench}' in {path}")


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


def _require_nonempty_str(cfg: dict, key: str, scope: str) -> str:
    value = str(cfg.get(key, "")).strip()
    if not value:
        fail(f"{scope}.{key} is missing or empty")
    ok(f"{scope}.{key}: {value}")
    return value


def _check_parent_dir(path_str: str, label: str, treat_as_dir: bool = False) -> None:
    path_obj = Path(path_str)
    parent = path_obj if treat_as_dir else path_obj.parent
    if str(parent).strip() == "":
        fail(f"{label} has invalid parent path: '{path_str}'")
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        fail(f"Cannot create/access parent directory for {label}: {parent} ({e})")
    ok(f"{label} parent directory accessible: {parent}")


def _check_base_url(url: str, label: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        fail(f"{label} must be a valid http(s) URL, got: {url}")
    ok(f"{label}: {url}")


def _validate_model_reference(
    model_ref: str,
    label: str,
    require_local_cache: bool,
) -> None:
    value = str(model_ref).strip()
    if not value:
        fail(f"{label} is empty")

    if Path(value).exists():
        ok(f"{label}: local path exists ({value})")
        return

    if " " in value or "/" not in value:
        fail(f"{label} is neither an existing path nor a valid HF repo id: {value}")

    if require_local_cache:
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        try:
            cache_path = snapshot_download(repo_id=value, local_files_only=True)
            ok(f"{label}: cached in local HF store ({cache_path})")
        except Exception as e:
            fail(
                f"{label} not available in local HF cache while offline mode is active: {value}\n"
                f"       Cache check error: {e}"
            )
    else:
        ok(f"{label}: remote model reference ({value})")


def _expected_sft_tag(student_model: str, experiment_tag: str) -> str:
    model_short = student_model.split("/")[-1].lower().replace("-instruct", "")
    tag = f"sft-{model_short}"
    if experiment_tag:
        tag = f"{tag}-{experiment_tag}"
    return tag


def _expected_grpo_tag(student_model: str, experiment_tag: str) -> str:
    model_short = student_model.split("/")[-1].lower().replace("-instruct", "")
    return f"sft-{model_short}-{experiment_tag or 'grpo'}"


def check_pipeline_config(
    config_path: str,
    simulate_offline: bool = False,
    require_grpo: bool = False,
    smoke: bool = False,
) -> None:
    """Validate config fields, paths, and chain wiring for distill/GRPO flows."""
    section("Pipeline config and chain wiring")
    from utils.config_loader import load_yaml  # noqa: PLC0415

    cfg = load_yaml(config_path)
    if not isinstance(cfg, dict):
        fail(f"Invalid YAML payload in {config_path}: expected a mapping")
    ok(f"Config loaded: {config_path}")

    for key in ("benchmarks", "generation", "training"):
        if not isinstance(cfg.get(key), dict):
            fail(f"Missing required top-level section: {key}")
        ok(f"Section present: {key}")

    benches = cfg.get("benchmarks", {}) or {}
    enabled_benchmarks: list[str] = []
    for bench_name in ("gsm8k", "math"):
        bench_cfg = benches.get(bench_name)
        if not isinstance(bench_cfg, dict):
            if smoke:
                fail(f"Smoke mode requires benchmarks.{bench_name} section")
            warn(f"benchmarks.{bench_name} section missing")
            continue

        if bool(bench_cfg.get("enabled", False)):
            enabled_benchmarks.append(bench_name)
            _require_nonempty_str(bench_cfg, "dataset_name", f"benchmarks.{bench_name}")
            _require_nonempty_str(bench_cfg, "dataset_split", f"benchmarks.{bench_name}")
            _require_nonempty_str(bench_cfg, "prompt_template", f"benchmarks.{bench_name}")

            subset_size = bench_cfg.get("subset_size")
            if smoke:
                if subset_size in (None, "null"):
                    fail(f"Smoke mode requires benchmarks.{bench_name}.subset_size > 0")
                try:
                    subset_int = int(str(subset_size))
                except Exception:
                    fail(f"benchmarks.{bench_name}.subset_size must be an integer in smoke mode")
                if subset_int <= 0:
                    fail(f"benchmarks.{bench_name}.subset_size must be > 0 in smoke mode")
                ok(f"benchmarks.{bench_name}.subset_size={subset_int}")

    if not enabled_benchmarks:
        fail("No enabled benchmarks found under 'benchmarks'")
    ok(f"Enabled benchmarks: {enabled_benchmarks}")

    if smoke:
        for bench_name in ("gsm8k", "math"):
            if bench_name not in enabled_benchmarks:
                fail(f"Smoke mode requires both gsm8k and math enabled; missing: {bench_name}")

    gen_cfg = cfg.get("generation", {}) or {}
    train_cfg = cfg.get("training", {}) or {}

    teacher_model = _require_nonempty_str(gen_cfg, "teacher_model", "generation")
    teacher_base_url = _require_nonempty_str(gen_cfg, "teacher_base_url", "generation")
    generation_output = _require_nonempty_str(gen_cfg, "output_file", "generation")

    student_model = _require_nonempty_str(train_cfg, "student_model", "training")
    training_dataset = _require_nonempty_str(train_cfg, "dataset_path", "training")
    training_output_dir = _require_nonempty_str(train_cfg, "output_dir", "training")

    _check_base_url(teacher_base_url, "generation.teacher_base_url")
    _check_parent_dir(generation_output, "generation.output_file")
    _check_parent_dir(training_dataset, "training.dataset_path")
    _check_parent_dir(training_output_dir, "training.output_dir", treat_as_dir=True)

    if Path(training_dataset) != Path(generation_output):
        fail(
            "Chain mismatch: training.dataset_path must match generation.output_file for"
            f" a coherent teacher -> distill flow. Got training.dataset_path={training_dataset}"
            f" vs generation.output_file={generation_output}"
        )
    ok("Chain link OK: generation.output_file -> training.dataset_path")

    _validate_model_reference(
        teacher_model,
        "generation.teacher_model",
        require_local_cache=simulate_offline,
    )
    _validate_model_reference(
        student_model,
        "training.student_model",
        require_local_cache=simulate_offline,
    )

    preflight_cfg = cfg.get("preflight", {}) or {}
    min_valid_samples = int(preflight_cfg.get("min_valid_samples", 50))
    if min_valid_samples <= 0:
        fail("preflight.min_valid_samples must be > 0")
    ok(f"preflight.min_valid_samples={min_valid_samples}")

    grpo_cfg = cfg.get("grpo_refinement", {}) or {}
    grpo_enabled = bool(grpo_cfg.get("enabled", False))
    if require_grpo and not grpo_enabled:
        fail("--require-grpo specified but grpo_refinement.enabled=false")

    if grpo_enabled:
        grpo_student = _require_nonempty_str(grpo_cfg, "student_model", "grpo_refinement")
        grpo_init_adapter = _require_nonempty_str(grpo_cfg, "init_adapter_path", "grpo_refinement")
        grpo_base_url = _require_nonempty_str(grpo_cfg, "base_url", "grpo_refinement")
        grpo_output_glob = _require_nonempty_str(grpo_cfg, "output_adapter_glob", "grpo_refinement")
        grpo_merged_output = _require_nonempty_str(grpo_cfg, "merged_output_path", "grpo_refinement")
        grpo_eval_role = _require_nonempty_str(grpo_cfg, "eval_role", "grpo_refinement")
        grpo_policy_path = _require_nonempty_str(grpo_cfg, "policy_model_path", "grpo_refinement")

        if grpo_student != student_model:
            fail(
                "training.student_model and grpo_refinement.student_model must match for"
                f" adapter compatibility. Got training={student_model} vs grpo={grpo_student}"
            )
        ok("Chain link OK: training.student_model == grpo_refinement.student_model")

        _check_base_url(grpo_base_url, "grpo_refinement.base_url")
        _check_parent_dir(grpo_policy_path, "grpo_refinement.policy_model_path")
        _check_parent_dir(grpo_merged_output, "grpo_refinement.merged_output_path")

        _validate_model_reference(
            grpo_student,
            "grpo_refinement.student_model",
            require_local_cache=simulate_offline,
        )

        expected_init_tag = _expected_sft_tag(
            student_model,
            str(train_cfg.get("experiment_tag", "")).strip(),
        )
        if expected_init_tag not in grpo_init_adapter:
            fail(
                "grpo_refinement.init_adapter_path does not match expected SFT adapter tag: "
                f"expected to contain '{expected_init_tag}', got '{grpo_init_adapter}'"
            )
        ok("Chain link OK: GRPO init adapter pattern matches SFT output tag")

        expected_grpo_run_tag = _expected_grpo_tag(
            student_model,
            str(grpo_cfg.get("experiment_tag", "grpo")).strip() or "grpo",
        )
        if expected_grpo_run_tag not in grpo_output_glob:
            fail(
                "grpo_refinement.output_adapter_glob does not match expected GRPO run tag: "
                f"expected to contain '{expected_grpo_run_tag}', got '{grpo_output_glob}'"
            )
        ok("Chain link OK: GRPO output adapter glob matches GRPO run tag")

        ok(f"GRPO eval role configured: {grpo_eval_role}")

        grpo_benches = grpo_cfg.get("benchmarks", {}) or {}
        for bench_name in ("gsm8k", "math"):
            bench_cfg = grpo_benches.get(bench_name)
            if not isinstance(bench_cfg, dict) or not bool(bench_cfg.get("enabled", False)):
                if smoke:
                    fail(f"Smoke mode requires grpo_refinement.benchmarks.{bench_name}.enabled=true")
                warn(f"grpo_refinement benchmark not enabled: {bench_name}")
                continue

            if smoke:
                subset_size = bench_cfg.get("subset_size")
                if subset_size in (None, "null"):
                    fail(f"Smoke mode requires grpo_refinement.benchmarks.{bench_name}.subset_size > 0")
                try:
                    subset_int = int(str(subset_size))
                except Exception:
                    fail(
                        "grpo_refinement.benchmarks."
                        f"{bench_name}.subset_size must be an integer in smoke mode"
                    )
                if subset_int <= 0:
                    fail(f"grpo_refinement.benchmarks.{bench_name}.subset_size must be > 0")
                ok(f"grpo_refinement.benchmarks.{bench_name}.subset_size={subset_int}")

        sampling_cfg = grpo_cfg.get("sampling", {}) or {}
        if smoke and sampling_cfg.get("max_prompts") in (None, "null"):
            warn(
                "Smoke mode: grpo_refinement.sampling.max_prompts is null; "
                "all smoke prompts will be used"
            )


def step_config(
    config: str,
    simulate_offline: bool = False,
    require_grpo: bool = False,
    smoke: bool = False,
) -> None:
    """Validate config wiring and environment without requiring generated artifacts."""
    from utils.config_loader import load_yaml  # noqa: PLC0415

    cfg = load_yaml(config)
    include_grpo_imports = require_grpo or bool((cfg.get("grpo_refinement") or {}).get("enabled", False))
    check_python_imports(include_grpo=include_grpo_imports)
    with offline_mode(simulate_offline):
        check_pipeline_config(
            config,
            simulate_offline=simulate_offline,
            require_grpo=require_grpo,
            smoke=smoke,
        )


# ── per-step flows ────────────────────────────────────────────────────────────

def step1(config: str, check_server: bool, simulate_offline: bool = False) -> None:
    """Validate everything needed before sbatch distill_generate.sbatch."""
    check_python_imports()
    if simulate_offline:
        section("Mode")
        warn("Simulating compute-node offline mode (HF_HUB_OFFLINE=1)")
    with offline_mode(simulate_offline):
        check_datasets(config)
        check_generate_smoke(config)
        check_teacher_model_cache(config)
    if check_server:
        from utils.config_loader import load_yaml  # noqa: PLC0415
        cfg = load_yaml(config)
        base_url = cfg.get("generation", {}).get("teacher_base_url", "http://localhost:8000")
        check_teacher_server(base_url)
    else:
        section("Teacher server")
        warn("Skipped (--no-server). Run with --check-server after starting vLLM.")


def step2(
    config: str,
    stage2_residual: bool = False,
    min_valid_samples_override: int | None = None,
) -> None:
    """Validate everything needed before sbatch distill_train_*.sbatch."""
    check_python_imports()
    from utils.config_loader import load_yaml  # noqa: PLC0415
    cfg = load_yaml(config)
    preflight_cfg = cfg.get("preflight", {}) or {}
    min_valid_samples = int(preflight_cfg.get("min_valid_samples", 50))
    if min_valid_samples_override is not None:
        min_valid_samples = int(min_valid_samples_override)
    if min_valid_samples <= 0:
        fail("Minimum valid samples must be > 0")

    if stage2_residual:
        residual_cfg = cfg.get("residual_distillation", {}) or {}
        stage2_cfg = residual_cfg.get("stage2_training", {}) or {}
        ds_path = stage2_cfg.get(
            "dataset_path",
            residual_cfg.get("output_file", "results/distill/teacher_outputs_residual_stage2.jsonl"),
        )
        required = ("gsm8k",)
    else:
        ds_path = cfg.get("training", {}).get(
            "dataset_path", "results/distill/teacher_outputs.jsonl")
        benches = cfg.get("benchmarks", {})
        required = tuple(
            name for name, bcfg in benches.items() if bcfg.get("enabled", True)
        )
    check_jsonl_dataset(
        ds_path,
        required_benchmarks=required,
        min_valid_samples=min_valid_samples,
    )


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
  python -m distill.preflight --step config --config configs/distill_1p5b_structured_grpo.yaml --require-grpo
  python -m distill.preflight --step 1             # before distill_generate.sbatch
  python -m distill.preflight --step 1 --check-server   # includes vLLM ping
  python -m distill.preflight --step 2             # before distill_train_*.sbatch
  python -m distill.preflight --step 3             # before eval_distilled_*.sbatch
  python -m distill.preflight --step all           # all checks
        """,
    )
    parser.add_argument("--step", choices=["config", "1", "2", "3", "all"], required=True)
    parser.add_argument("--config", default="configs/distill.yaml")
    parser.add_argument("--check-server", action="store_true",
                        help="Also ping the teacher vLLM /health endpoint (step 1)")
    parser.add_argument("--simulate-offline", action="store_true",
                        help="Force HF/Transformers offline flags during dataset/model checks")
    parser.add_argument("--stage2-residual", action="store_true",
                        help="For --step 2, validate residual_distillation.stage2_training dataset")
    parser.add_argument("--require-grpo", action="store_true",
                        help="For --step config/all, require grpo_refinement.enabled=true and validate GRPO wiring")
    parser.add_argument("--smoke", action="store_true",
                        help="For --step config/all, enforce smoke-mode config constraints")
    parser.add_argument("--min-valid-samples", type=int, default=None,
                        help="Override minimum valid rows required by step 2 dataset validation")
    args = parser.parse_args()

    if args.step in ("config", "all"):
        step_config(
            args.config,
            simulate_offline=args.simulate_offline,
            require_grpo=args.require_grpo,
            smoke=args.smoke,
        )
    if args.step in ("1", "all"):
        step1(args.config, check_server=args.check_server,
              simulate_offline=args.simulate_offline)
    if args.step in ("2", "all"):
        step2(
            args.config,
            stage2_residual=args.stage2_residual,
            min_valid_samples_override=args.min_valid_samples,
        )
    if args.step in ("3", "all"):
        step3_7b()
        step3_1b5()

    section("Summary")
    ok("All checks passed! Safe to submit the corresponding SLURM job.")


if __name__ == "__main__":
    main()
