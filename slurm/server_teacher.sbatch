#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — vLLM Teacher serving endpoint
# Target cluster: MareNostrum 5   (BSC)
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=vllm-teacher
#SBATCH --output=logs/vllm-teacher-%j.out
#SBATCH --error=logs/vllm-teacher-%j.err
#SBATCH --partition=acc                  # GPU accelerated partition
#SBATCH --account=bsc98
#SBATCH --gres=gpu:1                     # 14B fits in 1 H100-80G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --exclusive
# #SBATCH --qos=acc_bsccs                # uncomment for production QoS

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
