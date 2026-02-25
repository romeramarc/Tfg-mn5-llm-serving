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
# MN5 requires this exact chain (Lmod enforces prerequisites):
module purge
module load intel
module load impi
module load mkl
module load hdf5
module load python/3.12.1

# CRITICAL: the python/3.12.1 module injects PYTHONPATH pointing at the
# system site-packages (/apps/GPP/PYTHON/3.12.1/INTEL/lib/python3.12/
# site-packages).  PYTHONPATH is searched BEFORE the venv, so the system
# torch 2.3 would shadow the venv's torch 2.9 without this line.
unset PYTHONPATH

# ── Virtual environment ────────────────────────────────────
VENV_PATH="${VENV_PATH:-${HOME}/.venvs/tfg}"
if [[ ! -d "${VENV_PATH}" ]]; then
    echo "ERROR: Virtual environment not found at ${VENV_PATH}"
    echo "Create it with:  python -m venv ${VENV_PATH} && pip install -r requirements.txt"
    exit 1
fi
source "${VENV_PATH}/bin/activate"

# ── Torch shared libraries ─────────────────────────────────
# MN5 compute nodes load mkl/2025.2 which injects system paths into
# LD_LIBRARY_PATH that can contain a conflicting libc10.so / libtorch build.
# Additionally, vllm/_C.abi3.so may carry an embedded RPATH from its build
# environment that overrides LD_LIBRARY_PATH entirely.
#
# Two-pronged fix:
#   1. Prepend the venv's torch/lib to LD_LIBRARY_PATH (handles the common
#      shadowing case where no RPATH is embedded).
#   2. LD_PRELOAD the critical torch .so files explicitly — LD_PRELOAD is
#      resolved before DT_RPATH/DT_RUNPATH entries in any .so, so this
#      ensures the venv's libc10 / libtorch are found regardless of how the
#      wheel was built.
TORCH_LIB="${VENV_PATH}/lib/python3.12/site-packages/torch/lib"
if [[ -d "${TORCH_LIB}" ]]; then
    # 1: LD_LIBRARY_PATH prepend
    export LD_LIBRARY_PATH="${TORCH_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

    # 2: LD_PRELOAD — only set the libs that actually exist in the venv
    _PRELOAD_LIBS=""
    for _lib in libc10.so libtorch.so libtorch_cpu.so; do
        if [[ -f "${TORCH_LIB}/${_lib}" ]]; then
            _PRELOAD_LIBS="${TORCH_LIB}/${_lib}${_PRELOAD_LIBS:+ ${_PRELOAD_LIBS}}"
        fi
    done
    if [[ -n "${_PRELOAD_LIBS}" ]]; then
        export LD_PRELOAD="${_PRELOAD_LIBS}${LD_PRELOAD:+ ${LD_PRELOAD}}"
    fi
    unset _PRELOAD_LIBS _lib
fi

# ── HuggingFace cache on PROJECTS ──────────────────────────
# /gpfs/projects/bsc98/tbsc381408/ is the TFG project allocation on MN5.
# The SCRATCH filesystem (/gpfs/scratch/bsc98/bsc381408/) exists but is
# not writable by this user; PROJECTS is the correct writable location.
# Storing model weights here avoids filling the HOME quota (limited).
HF_SCRATCH="/gpfs/projects/bsc98/tbsc381408/hf_cache"
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
