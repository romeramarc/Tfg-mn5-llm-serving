#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — vLLM OpenAI-compatible serving endpoint
# Target cluster: MareNostrum 5   (BSC)
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=vllm-server
#SBATCH --output=logs/server-%j.out
#SBATCH --error=logs/server-%j.err
#SBATCH --partition=acc                  # GPU accelerated partition
#SBATCH --gres=gpu:4                     # request 4 GPUs (tensor-parallel)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --exclusive

set -euo pipefail

# ── Environment ─────────────────────────────────────────────
module purge
module load python/3.11 cuda/12.1 nccl/2.18

# Activate project virtual-env (created during installation)
source "${HOME}/.venvs/tfg/bin/activate"

# Ensure log directory exists
mkdir -p logs

# ── Metadata ────────────────────────────────────────────────
echo "=== SERVER START ==="
echo "Date:          $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID:  ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "GPUs:          ${SLURM_GPUS_ON_NODE:-unknown}"
echo "Git commit:    $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
echo "===================="

# ── Launch server ───────────────────────────────────────────
python -m serving.start_server --config configs/serving.yaml
