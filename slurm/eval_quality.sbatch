#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — Quality evaluation (GSM8K + MATH + RouterBench)
# Target cluster: MareNostrum 5   (BSC)
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=eval-quality
#SBATCH --output=logs/eval-quality-%j.out
#SBATCH --error=logs/eval-quality-%j.err
# NOTE: This script is designed for use with `srun --overlap` inside the
# server job, or submitted with --nodelist=<server-node> so that
# localhost:8000 resolves to the running vLLM server.
# For a self-contained run, uncomment the eval stage in run_phase1.sbatch.
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail

# ── Environment (shared setup) ─────────────────────────────
source env/setup_env.sh

echo "=== QUALITY EVALUATION ==="
echo "Date:          $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID:  ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "Benchmarks:    GSM8K, MATH, RouterBench"
nvidia-smi
echo "==========================="

# ── Wait for the vLLM server to be ready ────────────────────
python -m serving.healthcheck \
    --url "http://localhost:8000" \
    --retries 60 --interval 5

# ── Run quality evaluation ─────────────────────────────────
python -m eval.run_quality --config configs/eval.yaml

echo "=== QUALITY EVALUATION DONE ==="
