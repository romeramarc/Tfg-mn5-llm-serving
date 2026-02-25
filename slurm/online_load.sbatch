#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — Online load benchmark (async HTTP client)
# Target cluster: MareNostrum 5   (BSC)
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=bench-online
#SBATCH --output=logs/online-%j.out
#SBATCH --error=logs/online-%j.err
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

source "${HOME}/.venvs/tfg/bin/activate"
mkdir -p logs

echo "=== ONLINE LOAD BENCHMARK ==="
echo "Date:          $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID:  ${SLURM_JOB_ID}"

python -m serving.healthcheck \
    --url "http://localhost:8000" \
    --retries 60 --interval 5

python -m bench.run_online_load --config configs/benchmark.yaml

echo "=== ONLINE LOAD BENCHMARK DONE ==="
