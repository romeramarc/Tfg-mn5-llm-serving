#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — Offline throughput benchmark (vllm bench serve)
# Target cluster: MareNostrum 5   (BSC)
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=bench-throughput
#SBATCH --output=logs/bench-throughput-%j.out
#SBATCH --error=logs/bench-throughput-%j.err
# NOTE: This script is designed for use with `srun --overlap` inside the
# server job, or submitted with --nodelist=<server-node> so that
# localhost:8000 resolves to the running vLLM server.
# For a self-contained run, use slurm/run_phase1.sbatch instead.
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

# ── Environment (shared setup) ─────────────────────────────
source env/setup_env.sh

echo "=== THROUGHPUT BENCHMARK ==="
echo "Date:          $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID:  ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
nvidia-smi
echo "============================="

# ── Wait for the vLLM server to be ready ────────────────────
python -m serving.healthcheck \
    --url "http://localhost:8000" \
    --retries 60 --interval 5

# ── Run benchmark ──────────────────────────────────────────
python -m bench.run_throughput --config configs/benchmark.yaml

echo "=== THROUGHPUT BENCHMARK DONE ==="
