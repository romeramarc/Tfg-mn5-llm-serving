# Experiment Documentation

This document describes the metrics produced by each experimental pipeline,
where output artefacts are stored, and how to interpret the results.

---

## Phase-1 Scope

Phase-1 covers **teacher serving and evaluation only**.  Knowledge
distillation training and student evaluation are deferred to Phase-2.

The Phase-1 experiment suite consists of:

1. **Serving** — Launch the teacher model (Qwen2.5-72B-Instruct) via vLLM.
2. **Efficiency benchmarks** — Offline throughput and online load testing.
3. **Quality evaluation** — GSM8K, MATH, and RouterBench benchmarks.

---

## 1. Efficiency Benchmarks

### 1.1 Offline Throughput

**Script:** `bench/run_throughput.py`
**Config:** `configs/benchmark.yaml → throughput`
**Output:** `results/throughput/<timestamp>/`

| File                       | Description                                   |
|----------------------------|-----------------------------------------------|
| `throughput_results.json`  | Normalised metrics (see below)                |
| `throughput_results.csv`   | Same data in CSV format                       |
| `vllm_bench_raw.json`     | Raw JSON output from `vllm bench serve`       |
| `stdout.txt` / `stderr.txt` | Raw subprocess output for debugging         |
| `run_meta.json`           | Seed, git hash, GPU info, SLURM job ID        |
| `benchmark.yaml`          | Copy of the benchmark config used             |
| `serving.yaml`            | Copy of the serving config used               |

**Key metrics:**

| Metric                     | Unit    | Description                        |
|----------------------------|---------|------------------------------------|
| `request_throughput_rps`   | req/s   | Requests processed per second      |
| `input_throughput_tps`     | tok/s   | Input tokens processed per second  |
| `output_throughput_tps`    | tok/s   | Output tokens generated per second |
| `total_time_s`             | seconds | Total benchmark duration           |
| `mean_ttft_ms`             | ms      | Mean time to first token           |
| `p99_ttft_ms`              | ms      | 99th-percentile TTFT               |

### 1.2 Online Load

**Script:** `bench/run_online_load.py`
**Config:** `configs/benchmark.yaml → online`
**Output:** `results/online/<timestamp>/`

| File                     | Description                                     |
|--------------------------|-------------------------------------------------|
| `online_results.json`   | Summary metrics across all request rates        |
| `online_results.csv`    | Same data in CSV format                         |
| `rate_<N>/raw_results.json` | Per-request raw data at rate N req/s         |
| `rate_<N>/summary.json` | Summary for rate N                              |
| `run_meta.json`         | Seed, git hash, GPU info, SLURM job ID          |

**Key metrics (per rate):**

| Metric                   | Unit    | Description                          |
|--------------------------|---------|--------------------------------------|
| `latency_p50_ms`         | ms      | Median end-to-end latency            |
| `latency_p95_ms`         | ms      | 95th-percentile latency              |
| `latency_p99_ms`         | ms      | 99th-percentile latency              |
| `ttfb_p50_ms`            | ms      | Median time to first byte            |
| `effective_throughput_tps` | tok/s | Effective tokens per second          |

---

## 2. Quality Evaluation

### Overview

**Script:** `eval/run_quality.py`
**Config:** `configs/eval.yaml`
**Output:** `results/quality/<timestamp>/`

All quality benchmarks share the same timestamped run directory.  Each
benchmark writes to its own subdirectory.

| Directory                | Benchmark    |
|--------------------------|--------------|
| `gsm8k/`                | GSM8K        |
| `math/`                 | MATH         |
| `routerbench/`          | RouterBench  |

A top-level `quality_summary.json` and `quality_summary.csv` aggregate
accuracy across all benchmarks.

### 2.1 GSM8K

**Dataset:** `openai/gsm8k` (loaded from HuggingFace, `test` split)

GSM8K consists of grade-school math word problems.  The model is prompted
to solve each problem step by step and produce a final numeric answer
after `####`.

**Scoring:** Exact-match on the numeric value extracted from after `####`.
The extraction uses the last occurrence of the pattern `#### <number>` in
the model's response.  Commas and whitespace are stripped before numeric
comparison (with floating-point tolerance of 1e-4 relative).

**Interpreting results:**

- `accuracy_pct`: percentage of correctly answered problems among scorable
  examples.
- `unscorable_examples`: count of examples where the model failed to
  produce a parseable `####` answer or where the request errored.
  These are excluded from the accuracy denominator.
- Per-example detail is in `gsm8k/gsm8k_results.jsonl`.
- Unscorable cases are logged in `gsm8k/gsm8k_unscorable.jsonl` for
  manual review.

**Typical baselines (approximate):**

| Model                        | GSM8K Accuracy |
|------------------------------|----------------|
| GPT-4                        | ~92%           |
| Qwen2.5-72B-Instruct         | ~88–91%        |
| Qwen2.5-7B-Instruct          | ~75–80%        |

### 2.2 MATH

**Dataset:** `lighteval/MATH` (loaded from HuggingFace, `test` split)

MATH contains competition-level mathematics problems across algebra,
geometry, number theory, combinatorics, etc.  The model is prompted
to place its final answer inside `\boxed{}`.

**Scoring:** Normalised exact-match on the content of `\boxed{...}`.
Normalisation strips `\text{}`, `\mathrm{}`, dollar signs, and collapses
whitespace.  Numeric values are compared with floating-point tolerance.

**Interpreting results:**

- `accuracy_pct`: percentage correct among scorable examples.
- `unscorable_examples`: examples where the model did not produce a
  `\boxed{}` answer, or where the reference itself lacks one.
  These cases are logged separately in `math/math_unscorable.jsonl`
  with an `ambiguity_reason` field explaining why.
- The `subset_size` config key controls how many problems are evaluated
  (the full MATH test set is large; 500 is the default for development).

**Typical baselines (approximate):**

| Model                        | MATH Accuracy |
|------------------------------|---------------|
| GPT-4                        | ~52–58%       |
| Qwen2.5-72B-Instruct         | ~50–55%       |
| Qwen2.5-7B-Instruct          | ~35–40%       |

### 2.3 RouterBench

**Dataset:** Local JSONL file at `data/routerbench/routerbench.jsonl`

RouterBench is an external benchmark.  The dataset is **not bundled** with
this repository.  See `data/routerbench/README.md` for download and
placement instructions.

**Scoring:**

- If labels are available (`has_labels: true` in `configs/eval.yaml`),
  accuracy is computed via exact string match between the model's
  stripped output and the label.
- If labels are **not** available (`has_labels: false`), raw model
  outputs are stored and no accuracy is computed.  A placeholder note
  is included in the metrics JSON.

**Required data path:**

```
data/routerbench/routerbench.jsonl
```

Each line must be a JSON object with at least a `"prompt"` key.
Optional: `"label"`, `"expected_answer"`, `"metadata"`.

---

## 3. Run Metadata

Every experiment writes `run_meta.json` containing:

| Field              | Description                                      |
|--------------------|--------------------------------------------------|
| `timestamp_utc`    | ISO 8601 UTC timestamp of the run                |
| `hostname`         | Machine hostname                                 |
| `seed`             | Random seed used                                 |
| `git_commit`       | Short git commit hash (or null)                  |
| `python_version`   | Python interpreter version                       |
| `torch_version`    | PyTorch version                                  |
| `vllm_version`     | vLLM version                                     |
| `transformers_version` | HuggingFace Transformers version             |
| `datasets_version` | HuggingFace Datasets version                     |
| `slurm_job_id`     | SLURM job ID (if running inside a job)           |
| `gpus`             | List of GPU name + VRAM (MB) per device          |
| `config_hash`      | SHA-256 prefix of the serialised config          |

---

## 4. Directory Structure After Phase-1

```
results/
├── throughput/
│   └── throughput-20260224T120000Z/
│       ├── throughput_results.json
│       ├── throughput_results.csv
│       ├── vllm_bench_raw.json
│       ├── stdout.txt
│       ├── stderr.txt
│       ├── run_meta.json
│       ├── benchmark.yaml
│       └── serving.yaml
├── online/
│   └── online-20260224T130000Z/
│       ├── online_results.json
│       ├── online_results.csv
│       ├── rate_1/  rate_5/  rate_10/ ...
│       └── run_meta.json
└── quality/
    └── quality-20260224T140000Z/
        ├── quality_summary.json
        ├── quality_summary.csv
        ├── run_meta.json
        ├── eval.yaml
        ├── serving.yaml
        ├── models.yaml
        ├── gsm8k/
        │   ├── gsm8k_results.jsonl
        │   ├── gsm8k_metrics.json
        │   ├── gsm8k_metrics.csv
        │   └── gsm8k_unscorable.jsonl
        ├── math/
        │   ├── math_results.jsonl
        │   ├── math_metrics.json
        │   ├── math_metrics.csv
        │   └── math_unscorable.jsonl
        └── routerbench/
            ├── routerbench_results.jsonl
            ├── routerbench_metrics.json
            └── routerbench_metrics.csv
```
