# RULES.md — Engineering Governance

This document defines the mandatory procedures, coding standards, and operational constraints that apply to every contributor — human or automated — interacting with this repository. Violations of these rules invalidate experiment results and must be corrected before merging.

---

## 1. Coding Standards

1.1. All code targets Python 3.10 or later. Language features available only in newer versions must not be used unless the minimum version is formally raised.

1.2. Type annotations are required on all public function signatures. Internal helper functions should be annotated when the types are not immediately obvious.

1.3. Docstrings follow the NumPy/Scipy convention. Every module, every public class, and every public function must have a docstring that states its purpose, parameters, return value, and any side effects.

1.4. Line length must not exceed 100 characters in source files and 120 characters in YAML files. Exceptions are permitted only for URLs and long string literals that cannot be broken without introducing bugs.

1.5. Imports are grouped in the standard order: standard library, third-party packages, then project-local modules — separated by blank lines. Wildcard imports (`from x import *`) are prohibited.

1.6. No code may use `print()` for runtime output. All runtime messages must pass through `utils.logging.get_logger()`. The only exception is the `__main__` guard of a CLI entry-point that sets up logging before any other work.

---

## 2. Folder Discipline

2.1. `configs/` contains exclusively YAML configuration files and the prompt JSONL file. No Python code, no binary data, no generated output.

2.2. `results/` contains only timestamped experiment directories created by the framework at runtime. Manual files must not be placed here. Large binary artefacts (model checkpoints, datasets exceeding 100 MB) should be symlinked from scratch storage rather than committed.

2.3. `docs/` contains human-written documentation, architecture diagrams, and experiment notes. Auto-generated API documentation, if produced, goes into `docs/api/`.

2.4. Each Python package (`serving/`, `bench/`, `eval/`, `distill/`, `routing/`, `utils/`) must contain an `__init__.py`. Module names must be lowercase with underscores and must not shadow standard-library modules.

2.5. SLURM job templates reside in `slurm/` and nowhere else. They must never be generated dynamically by Python code; they are static shell scripts parameterised by configuration files.

---

## 3. Logging Requirements

3.1. Every process must initialise logging via `utils.logging.setup_logging()` before performing any work. The default format is JSON.

3.2. Structured log entries must include, at a minimum: timestamp, log level, logger name, and message. Context-specific fields (`model`, `latency_ms`, `seed`, `job_id`) must be passed via the `extra` dict, not interpolated into the message string.

3.3. Log files are written to the `logs/` directory (for interactive runs) or to the SLURM-managed `--output` / `--error` paths (for batch runs). Log files must never be written to `results/`.

3.4. Exceptions must be logged with `logger.exception()` so that the traceback is captured in the structured log stream.

---

## 4. Prohibition of Hidden State

4.1. No module may depend on mutable global state that is not set via a configuration file or explicitly passed as a function argument. Module-level caches and singletons must not influence experiment outcomes.

4.2. Temporary files, if needed, must be created inside the run directory (`results/<run>/tmp/`) and cleaned up before the run completes. Use of `/tmp`, `$TMPDIR`, or other system-managed temporary directories is permitted only for transient subprocess artefacts and must be documented in the code.

4.3. Environment variables may be read (e.g., SLURM variables) but must never be the sole source of a parameter that affects experiment output. Any consumed environment variable must be logged in `run_meta.json`.

---

## 5. Mandatory Configuration Usage

5.1. Every script that produces experiment output must accept a `--config` argument pointing to a YAML file. Default values must live in the YAML, not in the Python code.

5.2. Config files must be loaded through `utils.config_loader.load_yaml()` or `load_with_overrides()`. Direct calls to `yaml.safe_load()` are prohibited outside `utils/config_loader.py`.

5.3. Before an experiment begins, the active config must be copied into the run directory by calling `utils.reproducibility.snapshot_configs()`.

---

## 6. No Hardcoded Paths

6.1. Absolute filesystem paths are forbidden in all Python source files and YAML configs. The only acceptable absolute reference is `${HOME}` inside SLURM scripts, used to locate the virtual environment.

6.2. Paths to models, datasets, and results directories must be specified in configuration files using paths relative to the repository root.

6.3. If a cluster-specific base path is required (e.g., scratch storage), it must be defined as a YAML key and referenced via the config loader — never embedded directly in code.

---

## 7. SLURM Compliance

7.1. Every SLURM script must declare `--job-name`, `--output`, `--error`, `--partition`, `--gres` (for GPUs), `--ntasks`, `--cpus-per-task`, `--mem`, and `--time`.

7.2. Resource requests must be conservative during development. Production-scale experiments must be justified in an experiment plan documented in `docs/`.

7.3. SLURM scripts must `set -euo pipefail` and echo metadata (date, job ID, node, git commit) at the top of every job for traceability.

7.4. Long-running jobs must include a health-check step (via `serving/healthcheck.py`) before proceeding to the workload.

---

## 8. Determinism Requirements

8.1. Every experiment must call `utils.reproducibility.set_seed(seed)` with the seed drawn from the YAML config before any stochastic operation.

8.2. The seed value must be recorded in `run_meta.json`.

8.3. Known sources of non-determinism (e.g., CUDA convolution algorithm selection, asynchronous data loading) must be documented in the metadata when they apply. Setting `torch.backends.cudnn.deterministic = True` is recommended for training runs.

8.4. Two executions with the same config on the same hardware must produce outputs that agree within documented tolerance.

---

## 9. Documentation Requirements

9.1. Every new module must include a module-level docstring explaining its purpose, inputs, and outputs.

9.2. New YAML config keys must be documented: name, type, default, and description.

9.3. Changes that alter the experiment structure or the output schema must be accompanied by an update to `docs/` and, if the change is architectural, to this `RULES.md`.

9.4. The `README.md` must remain the top-level entry point for all users. It must not duplicate content from this file, but it must link to it.

---

## 10. Experiment Tracking Protocol

10.1. Every experiment creates a directory under `results/` named with an ISO 8601 UTC timestamp (and an optional tag prefix). No two runs may share a directory.

10.2. The run directory must contain:
  - `run_meta.json` — seeds, git hash, SLURM IDs, hardware info, config hash.
  - A copy of every config file consumed during the run.
  - The primary output artefacts (JSON, CSV).

10.3. Experiments may not overwrite previous run directories. If a rerun is needed, a new timestamped directory is created.

10.4. Large artefacts (model checkpoints, raw datasets) should be symlinked into the run directory rather than copied, to avoid filling quotas.

---

## 11. Safety Rules for HPC Resource Usage

11.1. Do not submit multi-node GPU jobs without explicit supervisor approval.

11.2. Validate code on a single-GPU interactive session before submitting a multi-GPU batch job.

11.3. Use `--time` conservatively. Estimate wall-clock time from a small-scale dry run and add a 20 % safety margin.

11.4. Never run training scripts outside of SLURM-managed jobs on shared login nodes.

11.5. Clean up large temporary files after each job completes. Checkpoint directories that are no longer needed must be archived or deleted.

---

## 12. Memory Usage Precautions

12.1. `gpu_memory_utilization` in `configs/serving.yaml` must not exceed 0.95. The default is 0.90.

12.2. Training scripts must enable `gradient_checkpointing` by default for models with more than one billion parameters.

12.3. If a job is killed by the OOM killer, reduce `max_model_len` or `per_device_train_batch_size` before retrying. The OOM event must be documented in the experiment notes.

12.4. Use `nvidia-smi` or `torch.cuda.max_memory_allocated()` to log peak memory usage in the run metadata when feasible.

---

## 13. No Modification of Architecture Without Documentation

13.1. Adding or removing a top-level module (e.g., a new `evaluation/` package) requires updating:
  - The directory tree in `README.md § 4`.
  - The corresponding section of this `RULES.md`.
  - Any SLURM templates that depend on the old structure.

13.2. Renaming a config key is a breaking change. It must be announced, and a migration note must be added to `docs/`.

13.3. Changing the output schema of any results file (JSON or CSV) requires updating `bench/parse_results.py` and any downstream analysis notebooks.

---

## Enforcement

These rules are mandatory. Automated agents must verify compliance before committing code. Human contributors must review this document before their first contribution. Non-compliant code must not be merged, and non-compliant experiment results must not be cited in the thesis.
