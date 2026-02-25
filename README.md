# Efficient Serving and Knowledge Distillation of Large Language Models in Supercomputing Environments

## 1. Project Overview

This repository contains the experimental framework developed for the Bachelor Thesis project on efficient serving and knowledge distillation of large language models (LLMs) in high-performance computing (HPC) environments. The work is designed to run on MareNostrum 5, a SLURM-managed supercomputer at the Barcelona Supercomputing Center (BSC), although the codebase is portable to any cluster that provides NVIDIA GPUs and a standard SLURM scheduler.

The framework is organised into four self-contained but interoperable layers — serving, benchmarking, distillation, and routing — each driven entirely by YAML configuration files and producing timestamped, structured experiment artefacts. Every decision in this repository favours reproducibility, modularity, and transparency over convenience.

> **Phase-1 (current):** Teacher serving + efficiency benchmarks + quality evaluation (GSM8K, MATH, RouterBench).  Knowledge distillation training and student evaluation are planned for Phase-2.

## 2. Research Objective

The central question this thesis investigates is how to reconcile the quality of large teacher models with the latency and resource efficiency requirements of production-grade inference systems. Specifically, the research objectives are:

1. Quantify the throughput and latency characteristics of vLLM-served LLMs across tensor-parallel configurations and GPU memory utilisation settings.
2. Evaluate hard-label (supervised fine-tuning) knowledge distillation as a method for producing a smaller, faster student model that approximates teacher quality.
3. Design and empirically compare routing policies that dynamically dispatch requests between teacher and student endpoints using confidence estimation.
4. Document all findings with sufficient metadata to allow exact reproduction of every experiment on the same or comparable hardware.

## 3. Experimental Methodology

All experiments follow a strict protocol. Each run begins by loading one or more YAML configuration files, creating a timestamped output directory, and copying the active configuration into it. Seeds are set for Python, NumPy, and PyTorch before any stochastic operation. Hardware metadata — including SLURM job identifiers, GPU names, and software versions — is captured automatically and stored alongside results.

The serving layer uses vLLM to expose an OpenAI-compatible API endpoint. Benchmarks are executed either as offline throughput tests — where `bench/run_throughput.py` invokes `vllm bench throughput` as a subprocess and parses its output — or as online load tests driven by an asynchronous HTTP client that measures time-to-first-token (TTFT), end-to-end latency percentiles, and effective throughput. The distillation pipeline first generates teacher completions as plain text for a set of prompts, then re-tokenises them with the student tokeniser and fine-tunes a student model using LoRA adapters via the HuggingFace Transformers and PEFT libraries. The routing layer implements three policies — always-teacher, cascading escalation, and confidence-based routing — to explore the quality–efficiency trade-off at the system level.

## 4. Repository Architecture

```
.
├── configs/                    YAML configuration files (single source of truth)
│   ├── models.yaml             Model registry (Qwen2.5 family)
│   ├── serving.yaml            vLLM server parameters
│   ├── benchmark.yaml          Throughput and online-load settings
│   ├── eval.yaml               Quality evaluation configuration
│   ├── distill.yaml            Distillation hyper-parameters (Phase-2)
│   ├── routing.yaml            Routing policy configuration (Phase-2)
│   └── prompts.jsonl           General-purpose evaluation prompt set
│
├── env/                        Environment setup
│   └── setup_env.sh            MN5 module loads + HF cache + venv activation
│
├── serving/                    vLLM serving layer
│   ├── start_server.py         Entry-point (launches vllm serve)
│   ├── healthcheck.py          Blocking health-check probe
│   └── server_utils.py         Metadata and CLI construction helpers
│
├── bench/                      Efficiency benchmarking tools
│   ├── run_throughput.py       Offline throughput benchmark
│   ├── run_online_load.py      Online load benchmark (async)
│   ├── metrics.py              Percentile computation and I/O
│   └── parse_results.py        Raw vLLM output parser
│
├── eval/                       Quality evaluation benchmarks
│   ├── run_quality.py          Unified entry-point (GSM8K + MATH + RouterBench)
│   ├── gsm8k.py                GSM8K runner and scorer
│   ├── math_eval.py            MATH runner and scorer
│   ├── routerbench.py          RouterBench runner and scorer
│   └── scoring.py              Answer extraction and matching utilities
│
├── distill/                    Knowledge distillation pipeline (Phase-2)
│   ├── generate_teacher_outputs.py   Query teacher → JSONL
│   ├── train_student_sft.py          LoRA SFT training
│   └── dataset_utils.py             JSONL I/O helpers
│
├── routing/                    Request routing layer (Phase-2)
│   ├── router.py               Experiment runner
│   ├── policies.py             Policy implementations (A, B, C)
│   └── confidence.py           Logprob / entropy confidence estimators
│
├── slurm/                      SLURM job templates (MN5)
│   ├── server_teacher.sbatch   Launch teacher vLLM server
│   ├── bench_throughput.sbatch Offline throughput benchmark
│   ├── bench_online.sbatch     Online load benchmark
│   ├── eval_quality.sbatch     Quality evaluation (GSM8K + MATH + RouterBench)
│   ├── server.sbatch           (legacy) Generic server launch
│   ├── throughput.sbatch       (legacy) Generic throughput benchmark
│   ├── online_load.sbatch      (legacy) Generic online benchmark
│   └── distill.sbatch          (Phase-2) Distillation pipeline
│
├── utils/                      Shared utilities
│   ├── logging.py              Structured JSON/text logging
│   ├── config_loader.py        YAML load, save, merge
│   └── reproducibility.py      Seeds, run dirs, metadata
│
├── data/                       External datasets (not committed)
│   └── routerbench/            RouterBench data (see README inside)
│
├── results/                    Experiment output (timestamped)
├── docs/                       Supplementary documentation
│   ├── index.md                Documentation landing page
│   └── experiments.md          Metrics reference and interpretation guide
├── requirements.txt            Python dependencies
├── README.md                   This document
└── RULES.md                    Engineering governance rules
```

Each module is importable as a Python package and executable as a script via `python -m <module>`. Configuration flows downward from `configs/` into every script; command-line flags are limited to config file paths, model role selection (`--role`), and connection parameters (`--url`).

## 5. Installation on HPC

### 5.1 MN5 Module Prerequisites

MareNostrum 5 uses Lmod with a strict prerequisite chain. Load these modules in order before any Python or SLURM operation:

```bash
module purge
module load intel
module load impi
module load mkl
module load hdf5
module load python/3.12.1
```

Then clear the `PYTHONPATH` that the module injects (see §5.2 for why this matters):

```bash
unset PYTHONPATH
```

Sanity check (all three should return sensible output):

```bash
module list
which python
python --version
```

### 5.2 Create a Virtual Environment

```bash
python -m venv "${HOME}/.venvs/tfg"
source "${HOME}/.venvs/tfg/bin/activate"
pip install --upgrade pip
```

**CRITICAL — unset `PYTHONPATH` before installing or importing anything:**

```bash
unset PYTHONPATH
```

> **Why?** Loading `python/3.12.1` via Lmod sets `PYTHONPATH` to the system site-packages (`/apps/GPP/PYTHON/3.12.1/INTEL/lib/python3.12/site-packages`). Python searches `PYTHONPATH` *before* the virtual environment, so the system torch 2.3 silently shadows the correct version installed in the venv. `unset PYTHONPATH` makes the venv take full precedence. This is already handled automatically in `env/setup_env.sh` (sourced by all SLURM scripts), but must be done manually in interactive sessions.

Install all dependencies (vLLM 0.15.1 pulls in the required torch 2.9.1 automatically):

```bash
pip install -r requirements.txt
```

Verify the installation:

```bash
python -c "import torch; print(torch.__file__)"             # must show ~/.venvs/tfg/...
python -c "from torch.library import infer_schema; print('torch OK')"
python -c "import vllm; print('vLLM', vllm.__version__)"
```

> **Version note:** The offline throughput benchmark (`bench/run_throughput.py`) invokes `vllm bench throughput` as a subprocess. This CLI sub-command requires vLLM ≥ 0.6. Verify the installed version with `vllm --version` after installation.

### 5.3 Model Access

Models are downloaded from the HuggingFace Hub. MareNostrum 5 compute nodes do not have outbound internet access, so model weights must be pre-downloaded on a login or transfer node. Set the `HF_HOME` environment variable to a directory on the high-capacity SCRATCH filesystem (`HF_HOME` supersedes the deprecated `TRANSFORMERS_CACHE`):

```bash
export HF_HOME="/gpfs/projects/bsc98/tbsc381408/hf_cache"
mkdir -p "${HF_HOME}"
huggingface-cli login
huggingface-cli download Qwen/Qwen2.5-72B-Instruct
```

> **Path note:** MN5 SCRATCH (`/gpfs/scratch/bsc98/bsc381408/`) is not writable for this account. The TFG project allocation at `/gpfs/projects/bsc98/tbsc381408/` is the correct writable location for large files.

> **Storage note:** Qwen2.5-72B-Instruct requires approximately 145 GB of disk space for full-precision weights. For an initial setup validation, download the dev model first (`Qwen/Qwen2.5-0.5B-Instruct`, ~1 GB) and serve it with `--role dev` before committing to the full teacher download.

## 6. MN5 Quickstart (Phase-1)

Phase-1 covers teacher serving, efficiency benchmarks, and quality evaluation.  All jobs are submitted via SLURM on MareNostrum 5.

### Step 1 — Start the teacher server

```bash
sbatch slurm/server_teacher.sbatch
```

This launches a vLLM instance serving `Qwen/Qwen2.5-72B-Instruct` on 4 GPUs with tensor parallelism.  The server writes logs to `logs/vllm-teacher-<JOBID>.out`.

> **GPU memory:** Serving Qwen2.5-72B-Instruct under bf16 precision with `gpu_memory_utilization: 0.90` typically requires 4× 80 GB GPUs (e.g. A100-80G or H100); exact requirements depend on `max_model_len`, KV cache allocation, and the chosen `dtype`. If fewer GPUs or smaller VRAM is available, use a smaller model (e.g. `Qwen/Qwen2.5-7B-Instruct` with 1 GPU) by editing `configs/models.yaml` and adjusting `--gres` and `tensor_parallel_size` in the SLURM script and `configs/serving.yaml` respectively.

### Step 2 — Run benchmarks on the server node

Benchmark and evaluation scripts connect to `http://localhost:8000`, so they must execute on the same node as the teacher server. Because the server job requests `--exclusive` node access, the recommended approach is to run client commands within the server's SLURM allocation using `srun --overlap`:

```bash
SERVER_JOBID=<JOBID from Step 1>

# Offline throughput benchmark
srun --overlap --jobid=${SERVER_JOBID} \
     python -m bench.run_throughput --config configs/benchmark.yaml

# Online load benchmark (TTFT, latency percentiles)
srun --overlap --jobid=${SERVER_JOBID} \
     python -m bench.run_online_load --config configs/benchmark.yaml
```

Alternatively, remove `--exclusive` from `slurm/server_teacher.sbatch` and submit the benchmark SLURM scripts with `--dependency=after:${SERVER_JOBID}` and `--nodelist=<SERVER_NODE>`. Each script runs a health-check probe that blocks until the server's `/health` endpoint returns HTTP 200.

### Step 3 — Run quality evaluation

Using the same `srun --overlap` method:

```bash
srun --overlap --jobid=${SERVER_JOBID} \
     python -m eval.run_quality --config configs/eval.yaml
```

This runs GSM8K, MATH, and RouterBench (if the dataset file is present) against the teacher endpoint.  Results go to `results/quality/<timestamp>/`.

> **Note:** RouterBench requires a manually placed dataset file at `data/routerbench/routerbench.jsonl`; it is not bundled with this repository.  See `data/routerbench/README.md` for download instructions and expected format.

### Summary of outputs

| Job                    | Output directory                 |
|------------------------|----------------------------------|
| Teacher server         | `logs/vllm-teacher-<JOBID>.out`  |
| Throughput benchmark   | `results/throughput/<timestamp>/`|
| Online load benchmark  | `results/online/<timestamp>/`    |
| Quality evaluation     | `results/quality/<timestamp>/`   |

For detailed metrics descriptions, see `docs/experiments.md`.

## 7. Running the Serving Layer

### 7.1 Interactive Launch

```bash
python -m serving.start_server --config configs/serving.yaml --role teacher
```

This replaces the Python process with a `vllm serve` invocation, which exposes `/v1/completions`, `/v1/chat/completions`, and `/health` endpoints.

### 7.2 SLURM Launch

```bash
sbatch slurm/server_teacher.sbatch
```

Adjust `--gres=gpu:N` and the `tensor_parallel_size` in `configs/serving.yaml` to match the desired degree of parallelism. The value of `tensor_parallel_size` must equal the number of GPUs allocated by `--gres`; a mismatch causes vLLM to fail at startup. The SLURM script logs the job ID, node name, and timestamp into `logs/vllm-teacher-<JOBID>.out`.

### 7.3 Health-Check

Other scripts should wait for the server before proceeding:

```bash
python -m serving.healthcheck --url http://localhost:8000 --retries 60
```

The probe exits with code 0 once `/health` returns HTTP 200.

## 8. Running Benchmarks

### 8.1 Offline Throughput

Edit `configs/benchmark.yaml` → `throughput` section, then:

```bash
python -m bench.run_throughput --config configs/benchmark.yaml
```

The script invokes `vllm bench throughput` as a subprocess, captures its output, and creates `results/throughput/<timestamp>/` containing `throughput_results.json`, `throughput_results.csv`, the raw vLLM output, and a copy of the configuration used.

### 8.2 Online Load

Edit the `online` section of the same config, specifying one or more request rates. Then:

```bash
python -m bench.run_online_load --config configs/benchmark.yaml
```

Output goes to `results/online/<timestamp>/` and includes per-rate subdirectories with raw latency data and a top-level `online_results.csv` that summarises TTFT and latency percentiles across all rates.

## 9. Quality Evaluation (GSM8K, MATH, RouterBench)

The `eval/` module runs math-focused quality benchmarks against the teacher endpoint.

### 9.1 Running all benchmarks

```bash
python -m eval.run_quality --config configs/eval.yaml
```

This sequentially evaluates GSM8K, MATH, and RouterBench (if enabled and data is present).  Each benchmark produces per-example JSONL logs, metrics JSON/CSV, and a top-level summary.

### 9.2 Configuration

Edit `configs/eval.yaml` to:

- Enable/disable individual benchmarks (`enabled: true/false`).
- Set `subset_size` to limit the number of examples (useful for development).
- Configure the prompt template used to query the model.
- Set `has_labels: true` for RouterBench if ground-truth labels are available.

### 9.3 Interpreting scores

**GSM8K:** Accuracy is computed via exact-match on the final numeric answer (after `####`).  See `gsm8k/gsm8k_unscorable.jsonl` for cases where the model did not produce a parseable answer.

**MATH:** Accuracy is computed via normalised exact-match on `\boxed{...}` content.  Unscorable cases (missing boxed answer, ambiguous reference) are logged separately with an `ambiguity_reason` field.

**RouterBench:** If labels are available, simple exact-match accuracy is reported.  Otherwise, raw model outputs are stored for manual or custom scoring. Note that the RouterBench dataset is not bundled with this repository and must be placed manually at `data/routerbench/routerbench.jsonl`.

See `docs/experiments.md` for full details on output format and baselines.

## 10. Running Knowledge Distillation (Phase-2)

Distillation proceeds in two sequential stages.

### 10.1 Generate Teacher Outputs

With the teacher server running:

```bash
python -m distill.generate_teacher_outputs --config configs/distill.yaml
```

This issues asynchronous requests to the teacher endpoint, collects completions, and writes them to the JSONL path specified in `distill.yaml → generation → output_file`.

### 10.2 Train the Student Model

```bash
python -m distill.train_student_sft --config configs/distill.yaml
```

The training script loads the teacher outputs as plain text, re-tokenises them using the student model's tokeniser, wraps the student in LoRA adapters, and runs SFT using the HuggingFace `Trainer`. Checkpoints are saved to `results/distill/checkpoints/<timestamp>/`.

Both stages can be chained automatically via the SLURM template:

```bash
sbatch slurm/distill.sbatch
```

## 11. Routing Policies (Phase-2)

The routing layer currently implements three strategies:

**Policy A — Always Teacher.** Every request is dispatched to the teacher model. This serves as the quality upper bound and the latency/cost upper bound.

**Policy B — Cascading Escalation.** The student receives the request first. If the student times out or returns an error, the request is escalated to the teacher. The student timeout is configurable via `routing.yaml → policies → cascading → student_timeout_ms`.

**Policy C — Confidence-Based Routing.** The student is queried with `logprobs` enabled. A confidence score is computed from the returned log-probability distribution (using either the max-logprob or entropy method). If confidence falls below the threshold, the request is re-sent to the teacher. When logprobs are unavailable, a heuristic fallback based on lexical hedging detection is used.

Run a routing experiment with:

```bash
python -m routing.router --config configs/routing.yaml
```

Results include per-request records with selected model, latency, routing decision, and confidence score, saved as JSON and CSV in `results/routing/<timestamp>/`.

## 12. Reproducibility Protocol

Every executable script in this repository follows the same protocol:

1. Load the YAML config from the path given by `--config` (or the module's default).
2. Set all random seeds (Python, NumPy, PyTorch) to the value specified in the config.
3. Create a timestamped run directory under `results/`.
4. Copy the active config files into the run directory.
5. Collect and persist hardware/environment metadata as `run_meta.json`.
6. Execute the experiment.
7. Write structured results to the run directory.

No experiment modifies state outside its own timestamped directory. No parameter is read from the environment unless it is also logged in `run_meta.json`. Deterministic algorithms are enabled where supported (via `torch.use_deterministic_algorithms` and cuDNN deterministic mode), and all random seeds are fixed. Nevertheless, GPU kernel scheduling order and floating-point reduction order in parallel operations prevent bitwise reproducibility across runs. Two runs with the same config on the same hardware are therefore expected to produce statistically consistent outputs within floating-point tolerance, not bitwise-identical results. The degree and sources of variation are recorded in the run metadata.

## 13. Logging Design

Logging is centralised in `utils/logging.py`. Two modes are available:

- **JSON** (default): Every log line is a JSON object with `ts`, `level`, `logger`, `message`, and optional extra fields such as `job_id`, `model`, `latency`, and `throughput`. This format integrates directly with ELK stacks and other log aggregation systems.
- **Text**: For interactive debugging. Produces human-readable lines with ISO timestamps and aligned level strings.

Modules obtain a logger via `get_logger(__name__)` and attach structured context through the `extra` parameter. The choice between JSON and text mode is controlled by `configs/serving.yaml → logging → format` or by calling `setup_logging(fmt="text")` at the top of a script.

## 14. Extensibility Guidelines

The repository is designed for extension along three axes:

**Adding a new model.** Register it in `configs/models.yaml` and reference the identifier in the relevant serving or distillation config.

**Adding a benchmark type.** Create a new script under `bench/` that reads from `configs/benchmark.yaml`, creates a timestamped run directory, and writes structured output using `bench/metrics.py`.

**Adding a routing policy.** Implement an async function with the signature `async def policy(client, prompt, ctx) -> RoutingDecision` in `routing/policies.py`, register it in the `POLICIES` dict, and document it in `configs/routing.yaml`.

All extensions must follow the rules codified in `RULES.md`.

## 15. Troubleshooting HPC Environments

**CUDA version mismatch.** Ensure the `module load cuda/…` version in the SLURM script matches the CUDA version against which PyTorch and vLLM were compiled. Run `python -c "import torch; print(torch.version.cuda)"` to verify.

**Out-of-GPU-memory.** The most common cause is that the model is too large for the available GPUs (e.g. Qwen2.5-72B under bf16 typically requires 4× 80 GB, though exact usage depends on `max_model_len` and KV cache settings). Switch to a smaller model in `configs/models.yaml`, or reduce `gpu_memory_utilization` in `configs/serving.yaml`, or lower `max_model_len`. For distillation training, enable `gradient_checkpointing` and reduce `per_device_train_batch_size`.

**Server not reachable from the benchmark job.** Client jobs must run on the same node as the vLLM server (they connect to `localhost:8000`). Use `srun --overlap --jobid=<SERVER_JOBID>` to run within the server's allocation, or remove `--exclusive` from the server script and pin client jobs with `--nodelist`. Note that `--dependency` alone does not guarantee node co-location.

**Permission errors on `results/`.** The job environment must have write access to the repository root (or the directory pointed to by `results_base_dir` in the config). Use `$SCRATCH` or `$TMPDIR` on clusters that restrict home-directory writes during jobs.

**vLLM installation issues.** vLLM requires a recent GCC toolchain and CUDA-compatible system libraries. If `pip install vllm` fails, build from source following the upstream instructions and ensure the cluster's system modules expose the correct `libcudart` and `libnccl`.

## 16. License and Attribution

This repository supports a Bachelor Thesis project. Please cite the associated thesis document when using results produced by this framework.
