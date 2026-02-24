#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — Quality evaluation (GSM8K + MATH + RouterBench)
# Target cluster: MareNostrum 5   (BSC)
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=eval-quality
#SBATCH --output=logs/eval-quality-%j.out
#SBATCH --error=logs/eval-quality-%j.err
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
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
