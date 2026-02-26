"""
bench/run_throughput.py
=======================
Offline throughput benchmark using ``vllm bench serve``.

Workflow
--------
1. Load ``configs/benchmark.yaml``.
2. Create a timestamped run directory under ``results/throughput/``.
3. Snapshot active configs.
4. Run ``vllm bench serve`` as a subprocess.
5. Parse the JSON / stdout output.
6. Save normalised results as JSON **and** CSV.

Usage
-----
    python -m bench.run_throughput [--config configs/benchmark.yaml]

Requires a running vLLM server (see ``serving/start_server.py``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from bench.metrics import save_json, save_csv
from bench.parse_results import parse_json_result, parse_stdout
from utils.config_loader import load_yaml
from utils.logging import setup_logging, get_logger
from utils.reproducibility import (
    collect_metadata, make_run_dir, save_metadata, set_seed, snapshot_configs,
)

logger = get_logger(__name__)


def _build_bench_cmd(cfg: dict, result_file: Path) -> list[str]:
    """Construct the ``vllm bench serve`` CLI invocation."""
    common = cfg.get("common", {})
    tp = cfg.get("throughput", {})

    cmd: list[str] = [
        "vllm", "bench", "serve",
        "--base-url", common.get("base_url", "http://localhost:8000"),
        "--model", common.get("model", ""),
        "--dataset-name", tp.get("dataset_name", "sharegpt"),
        "--num-prompts", str(tp.get("num_prompts", 1000)),
        "--request-rate", str(tp.get("request_rate", "inf")),
        "--seed", str(common.get("seed", 42)),
        "--result-filename", str(result_file),
    ]

    if tp.get("dataset_path"):
        cmd.extend(["--dataset-path", tp["dataset_path"]])
    if tp.get("input_len") is not None:
        cmd.extend(["--input-len", str(tp["input_len"])])
    if tp.get("output_len") is not None:
        cmd.extend(["--output-len", str(tp["output_len"])])
    if tp.get("max_concurrency") is not None:
        cmd.extend(["--max-concurrency", str(tp["max_concurrency"])])

    return cmd


def run(
    config_path: str = "configs/benchmark.yaml",
    model_override: str | None = None,
    role: str | None = None,
) -> Path:
    """Execute the throughput benchmark and return the run directory.

    Parameters
    ----------
    model_override:
        If given, overrides the ``common.model`` in the config.
    role:
        Short label (e.g. ``teacher``, ``student_mid``, ``student_small``) used
        to name the run directory, making results easy to identify and compare.
    """
    cfg = load_yaml(config_path)
    if model_override:
        cfg.setdefault("common", {})["model"] = model_override
    common = cfg.get("common", {})
    seed = common.get("seed", 42)
    set_seed(seed)

    log_cfg = cfg.get("logging", {})
    setup_logging(level=log_cfg.get("level", "INFO"))

    # ── Run directory ───────────────────────────────────────
    # Tag format: throughput-<role>  so results/throughput/throughput-teacher-<ts>/
    # makes it immediately obvious which model produced each result.
    tag = f"throughput-{role}" if role else "throughput"
    run_dir = make_run_dir(
        common.get("results_base_dir", "results") + "/throughput",
        tag=tag,
    )
    snapshot_configs([config_path, "configs/serving.yaml"], run_dir)

    # ── Metadata ────────────────────────────────────────────
    meta = collect_metadata(seed, cfg)
    save_metadata(meta, run_dir)

    # ── Execute benchmark ───────────────────────────────────
    result_file = run_dir / "vllm_bench_raw.json"
    cmd = _build_bench_cmd(cfg, result_file)
    logger.info("Running throughput benchmark", extra={"cmd": " ".join(cmd)})

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall_time = time.time() - t0

    # Persist raw stdout / stderr for debugging
    (run_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")

    # ── Parse results ───────────────────────────────────────
    if result_file.exists():
        results = parse_json_result(result_file)
    else:
        logger.warning("JSON result file missing; falling back to stdout parse")
        results = parse_stdout(proc.stdout)

    results["wall_time_s"] = wall_time
    results["return_code"] = proc.returncode

    save_json(results, run_dir / "throughput_results.json")
    save_csv([results], run_dir / "throughput_results.csv")
    logger.info("Throughput benchmark complete", extra={"run_dir": str(run_dir)})

    return run_dir


# ── CLI ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Offline throughput benchmark")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--model", default=None,
                        help="Override the model name in the config.")
    parser.add_argument("--role", default=None,
                        help="Role label for the run directory (e.g. teacher, student_mid).")
    args = parser.parse_args()
    run(args.config, model_override=args.model, role=args.role)


if __name__ == "__main__":
    main()
