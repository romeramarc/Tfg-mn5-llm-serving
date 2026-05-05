# Phase A Runbook — Service-state traces and service-cost predictor

This runbook is the operational counterpart to the Phase A plan agreed with
the supervisor (e-mails of April 22nd / 24th, 2026). The goal of Phase A is
twofold:

1. Generate **load traces** under a Poisson arrival process against the same
   vLLM serving stack used elsewhere in the thesis.
2. Train an offline **service-cost predictor** that, given a request, the
   candidate model and the current state of the system `z`, estimates the
   marginal cost (latency by default; optionally a composite of latency,
   GPU-seconds and an energy proxy).

The runbook is intentionally short and prescriptive. For the design
rationale and how this slots into Phases B–D and the online validation,
read the corresponding sections of `memoria/`.

---

## 1. Components

```
configs/phase_a.yaml                    Single source of truth for Phase A
bench/run_load_capture.py               Async Poisson client + z logger
bench/gpu_sampler.py                    Out-of-process nvidia-smi sidecar
scripts/preflight_phase_a.py            Static checks; run before sbatch
scripts/phase_a_train.py                Build dataset + train predictor
slurm/phase_a_capture.sbatch            Capture client SLURM template
slurm/phase_a_train.sbatch              Predictor training SLURM template
slurm/launch_phase_a.sh                 End-to-end launcher
```

## 2. Output layout

Everything Phase A produces lives under `results/phase_a/`:

```
results/phase_a/
├── captures/<role>-<timestamp>/
│   ├── trace.jsonl                 ModelExecutionTrace records (one per request)
│   ├── server_metrics.jsonl        /metrics scrape (per-tick gauges)
│   ├── gpu_samples.jsonl           nvidia-smi per-tick rows
│   ├── rate_<lambda>/
│   │   ├── raw_requests.json
│   │   ├── raw_requests.csv
│   │   └── summary.json
│   ├── summaries.json
│   ├── summaries.csv
│   └── run_meta.json
├── datasets/
│   ├── service_cost_phase_a.jsonl
│   ├── service_cost_phase_a.csv
│   └── service_cost_phase_a_meta.json
└── predictors/sc-gb-<timestamp>/
    ├── model_bundle.joblib
    ├── metrics.json
    ├── feature_importance.csv
    ├── predictions_test.csv
    └── split_assignments.csv
```

## 3. Local checks (run before pushing to BSC)

On **your laptop / WSL**, `env/setup_env.sh` is not the MareNostrum Lmod stack:
it skips `module load` and activates, in order, `VENV_PATH` if set, else
`<repo>/.venv`, else `~/.venvs/tfg`. Create a repo venv once:

```bash
cd /path/to/repo
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy httpx scikit-learn joblib pyyaml python-json-logger
# optional full stack (heavy): pip install -r requirements.txt
```

Then either `source env/setup_env.sh` or stay inside `.venv` and run:

```bash
python scripts/preflight_phase_a.py
```

This validates configs, imports, prompt count and SLURM templates. It does
**not** require a GPU.

If you sourced `setup_env.sh` on WSL **before** creating `.venv`, you will see
`module: command not found` (harmless) and then the script tells you how to
create `.venv`.

## 4. BSC execution

From the repository root on a BSC login node, after sourcing the
environment (`source env/setup_env.sh`) at least once:

```bash
bash slurm/launch_phase_a.sh
```

This submits, in order:

| Job                       | Template                          | Notes                                     |
|---------------------------|-----------------------------------|-------------------------------------------|
| `vllm-role-p2` (×N roles) | `slurm/server_role_phase2.sbatch` | One vLLM endpoint per role                |
| `phase-a-capture` (×N)    | `slurm/phase_a_capture.sbatch`    | Co-located with its server, after RUNNING |
| `phase-a-train`           | `slurm/phase_a_train.sbatch`      | `--dependency=afterok` of all captures    |
| auto-cleanup              | inline `--wrap=scancel …`         | Kills server jobs once captures finish    |

Monitoring:

```bash
squeue -u $USER
tail -f logs/phase-a-capture-*.out
tail -f logs/phase-a-train-*.out
```

If you only need the capture for a single role:

```bash
sbatch --export=ALL,ROLE=teacher,PROJECT_DIR=$PWD slurm/server_role_phase2.sbatch
# wait until results/routing/endpoints/teacher.url appears
sbatch --export=ALL,ROLE=teacher,PROJECT_DIR=$PWD slurm/phase_a_capture.sbatch
```

## 5. Tuning knobs (`configs/phase_a.yaml`)

| Key                                | What it controls                                   |
|------------------------------------|----------------------------------------------------|
| `capture.arrival_rates_rps`        | List of λ values (req/s). One Poisson session each |
| `capture.num_requests_per_rate`    | Sample size per λ (default 600)                    |
| `capture.max_inflight`             | Open-loop concurrency ceiling                      |
| `capture.stream` / `include_usage` | Real TTFT via SSE; `usage` block in last event     |
| `dataset.cost_mode`                | `latency_ms` or `composite`                        |
| `dataset.feature_policy`           | `extended_operational` to use telemetry features    |
| `predictor.family`                 | `linear` / `random_forest` / `gradient_boosting`   |

## 6. What the predictor uses (decision-time `z`)

Captured per request **at send time** and written into the trace:

| Field                              | Source                              |
|------------------------------------|-------------------------------------|
| `system_state.queue_depth`         | `vllm:num_requests_waiting` gauge   |
| `system_state.pending_requests`    | client-side in-flight counter       |
| `system_state.throughput_rps_recent` | `vllm:avg_generation_throughput_toks_per_s` |
| `system_state.active_workers`      | `vllm:num_requests_running`         |
| `tags.engine.kv_cache_usage_pct_mean` | `vllm:gpu_cache_usage_perc`      |
| `tags.z_inflight_at_send`          | client-side in-flight counter       |
| `tags.z_recent_p50_latency_ms`     | rolling p50 over last 50 outcomes   |
| `resources.gpu_utilization_pct`    | nvidia-smi mean during the request  |
| `resources.gpu_seconds`            | `util * latency_s` proxy            |

Targets: `latency_ms` (default) or a composite weighted sum of latency +
gpu_seconds + energy joules, configured via `dataset.cost_*`.

## 7. Re-using captured traces

Once `trace.jsonl` files are in place, you can rebuild the dataset and
train a different predictor family without re-running the capture:

```bash
python scripts/phase_a_train.py --config configs/phase_a.yaml
```

(For ad-hoc experimentation you can edit `configs/phase_a.yaml` to point
`dataset.trace_glob_patterns` at any subset of capture runs.)
