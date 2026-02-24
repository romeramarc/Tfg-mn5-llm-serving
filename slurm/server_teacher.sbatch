#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — vLLM Teacher serving endpoint
# Target cluster: MareNostrum 5   (BSC)
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=vllm-teacher
#SBATCH --output=logs/vllm-teacher-%j.out
#SBATCH --error=logs/vllm-teacher-%j.err
#SBATCH --partition=acc                  # GPU accelerated partition
#SBATCH --gres=gpu:4                     # request 4 GPUs (tensor-parallel)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --exclusive

set -euo pipefail

# ── Environment (shared setup) ─────────────────────────────
source env/setup_env.sh

echo "=== TEACHER SERVER START ==="
echo "Date:          $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID:  ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "GPUs:          ${SLURM_GPUS_ON_NODE:-unknown}"
echo "Git commit:    $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
nvidia-smi
echo "=============================="

# ── Launch teacher server ──────────────────────────────────
python -m serving.start_server \
    --config configs/serving.yaml \
    --models configs/models.yaml \
    --role teacher
