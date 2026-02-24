#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# env/setup_env.sh — Environment bootstrap for MareNostrum 5
# ─────────────────────────────────────────────────────────────
# Source this file at the top of every SLURM script:
#
#     source env/setup_env.sh
#
# It loads the required modules, activates the virtual environment,
# and sets HuggingFace cache paths to SCRATCH so that large model
# weights are stored on the high-capacity parallel filesystem.
# ─────────────────────────────────────────────────────────────
set -euo pipefail

# ── Modules ─────────────────────────────────────────────────
module purge
module load python/3.11 cuda/12.1 nccl/2.18

# ── Virtual environment ────────────────────────────────────
VENV_PATH="${VENV_PATH:-${HOME}/.venvs/tfg}"
if [[ ! -d "${VENV_PATH}" ]]; then
    echo "ERROR: Virtual environment not found at ${VENV_PATH}"
    echo "Create it with:  python -m venv ${VENV_PATH} && pip install -r requirements.txt"
    exit 1
fi
source "${VENV_PATH}/bin/activate"

# ── HuggingFace cache on SCRATCH ───────────────────────────
# SCRATCH is a high-capacity parallel filesystem available on MN5.
# Storing model weights and datasets here avoids filling the HOME quota.
HF_SCRATCH="${SCRATCH:-/gpfs/scratch}/${USER}/hf_cache"
export HF_HOME="${HF_SCRATCH}"
export TRANSFORMERS_CACHE="${HF_SCRATCH}/transformers"
export HF_DATASETS_CACHE="${HF_SCRATCH}/datasets"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

# ── Miscellaneous ──────────────────────────────────────────
# Ensure logs/ and results/ directories exist
mkdir -p logs results

# ── Diagnostics ─────────────────────────────────────────────
echo "──────────────────────────────────────────────"
echo "ENV SETUP COMPLETE"
echo "  Date:            $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  Hostname:        $(hostname)"
echo "  SLURM_JOB_ID:    ${SLURM_JOB_ID:-interactive}"
echo "  Python:          $(python --version 2>&1)"
echo "  CUDA:            $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'n/a')"
echo "  vLLM:            $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'n/a')"
echo "  HF_HOME:         ${HF_HOME}"
echo "  Git commit:      $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
echo "──────────────────────────────────────────────"

# Print GPU information if available
if command -v nvidia-smi &>/dev/null; then
    echo ""
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
    echo ""
fi
