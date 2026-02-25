#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — Offline throughput benchmark (vllm bench serve)
# Target cluster: MareNostrum 5   (BSC)
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=bench-throughput
#SBATCH --output=logs/throughput-%j.out
#SBATCH --error=logs/throughput-%j.err
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

module purge
module load intel
module load impi
module load mkl
module load hdf5
module load python/3.12.1
unset PYTHONPATH   # prevent system torch from shadowing the venv

source "${HOME}/.venvs/tfg/bin/activate"
mkdir -p logs

echo "=== THROUGHPUT BENCHMARK ==="
echo "Date:          $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID:  ${SLURM_JOB_ID}"

# ── Wait for the vLLM server to be ready ────────────────────
python -m serving.healthcheck \
    --url "http://localhost:8000" \
    --retries 60 --interval 5

# ── Run benchmark ──────────────────────────────────────────
python -m bench.run_throughput --config configs/benchmark.yaml

echo "=== THROUGHPUT BENCHMARK DONE ==="
