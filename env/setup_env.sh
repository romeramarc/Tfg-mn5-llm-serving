#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# env/setup_env.sh — Environment bootstrap
# ─────────────────────────────────────────────────────────────
# MareNostrum 5 (BSC): loads Lmod modules + ~/.venvs/tfg + HF cache on PROJECTS.
# Laptop / WSL / CI: skips modules; prefers <repo>/.venv then ~/.venvs/tfg;
# uses ~/.cache/huggingface unless HF_HOME is already set.
#
#     source env/setup_env.sh
#
# NOTE: set -euo pipefail is intentionally NOT used here because this file
# is sourced into interactive shells.
# ─────────────────────────────────────────────────────────────

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"

# ── Detect BSC MN5-style host ───────────────────────────────
_USE_BSC_MODULES=0
if command -v module &>/dev/null && [[ -d /gpfs/projects/bsc98 ]]; then
  _USE_BSC_MODULES=1
fi

if [[ "${_USE_BSC_MODULES}" -eq 1 ]]; then
  # ═══════════════════════════════════════════════════════════
  # MareNostrum 5 — Lmod + project HF cache + venv torch fixes
  # ═══════════════════════════════════════════════════════════
  module purge
  module load intel
  module load impi
  module load mkl
  module load sqlite3   # required on compute nodes: _sqlite3.so needs sqlite3_deserialize (>=3.23)
  module load hdf5
  module load python/3.12.1

  unset PYTHONPATH

  VENV_PATH="${VENV_PATH:-${HOME}/.venvs/tfg}"
  if [[ ! -d "${VENV_PATH}" ]]; then
    echo "ERROR: Virtual environment not found at ${VENV_PATH}"
    echo "Create it with:  python -m venv ${VENV_PATH} && pip install -r requirements.txt"
    return 1 2>/dev/null || exit 1
  fi
  source "${VENV_PATH}/bin/activate"

  TORCH_LIB="${VENV_PATH}/lib/python3.12/site-packages/torch/lib"
  if [[ -d "${TORCH_LIB}" ]]; then
    export LD_LIBRARY_PATH="${TORCH_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
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

  HF_SCRATCH="/gpfs/projects/bsc98/tbsc381408/hf_cache"
  export HF_HOME="${HF_SCRATCH}"
  unset TRANSFORMERS_CACHE
  export HF_DATASETS_CACHE="${HF_SCRATCH}/datasets"
  mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

  _hostname=$(hostname -s)
  if [[ "${_hostname}" == glogin* ]]; then
    unset HF_HUB_OFFLINE
    unset TRANSFORMERS_OFFLINE
  else
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
  fi
  unset _hostname

else
  # ═══════════════════════════════════════════════════════════
  # Local / WSL / laptop — no Lmod, no /gpfs paths
  # ═══════════════════════════════════════════════════════════
  echo "INFO: entorno local (sin Lmod BSC). Usa venv del repo o ~/.venvs/tfg." >&2
  unset PYTHONPATH 2>/dev/null || true

  if [[ -n "${VENV_PATH:-}" && -d "${VENV_PATH}" ]]; then
    :
  elif [[ -d "${_REPO_ROOT}/.venv" ]]; then
    VENV_PATH="${_REPO_ROOT}/.venv"
  else
    VENV_PATH="${HOME}/.venvs/tfg"
  fi

  if [[ ! -d "${VENV_PATH}" ]]; then
    echo "ERROR: no hay entorno virtual en:" >&2
    echo "  ${_REPO_ROOT}/.venv   (recomendado en portátil)" >&2
    echo "  ni en ${HOME}/.venvs/tfg" >&2
    echo "" >&2
    echo "Crea uno en la raíz del repo y instala dependencias:" >&2
    echo "  cd \"${_REPO_ROOT}\"" >&2
    echo "  python3 -m venv .venv" >&2
    echo "  source .venv/bin/activate" >&2
    echo "  pip install -U pip" >&2
    echo "  pip install numpy httpx scikit-learn joblib pyyaml python-json-logger" >&2
    echo "  # preflight completo: pip install -r requirements.txt" >&2
    return 1 2>/dev/null || exit 1
  fi
  source "${VENV_PATH}/bin/activate"

  # Optional torch lib shadowing (venv may be 3.10–3.12)
  _PY_TAG="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "3.12")"
  TORCH_LIB="${VENV_PATH}/lib/python${_PY_TAG}/site-packages/torch/lib"
  if [[ -d "${TORCH_LIB}" ]]; then
    export LD_LIBRARY_PATH="${TORCH_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
  unset _PY_TAG

  export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
  mkdir -p "${HF_HOME}"
  unset TRANSFORMERS_CACHE 2>/dev/null || true
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
  mkdir -p "${HF_DATASETS_CACHE}"
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE 2>/dev/null || true
fi

unset _USE_BSC_MODULES _SCRIPT_DIR _REPO_ROOT

# ── Miscellaneous ──────────────────────────────────────────
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

if command -v nvidia-smi &>/dev/null; then
  echo ""
  if ! nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv; then
    echo "WARNING: nvidia-smi unavailable on this node (expected on some login nodes)."
  fi
  echo ""
fi
