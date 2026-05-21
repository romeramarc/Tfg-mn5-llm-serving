#!/bin/bash
# ------------------------------------------------------------------
# SLURM — Holdout routing evaluation (one system per job)
# ------------------------------------------------------------------
# Requires vLLM servers for the roles listed in the system config.
# Endpoint URLs must exist under results/routing/endpoints/<role>.url
#
# Example:
#   sbatch --export=ALL,SYSTEM_ID=sysD_cascade_distilled slurm/eval_holdout.sbatch
# ------------------------------------------------------------------
#SBATCH --job-name=eval-holdout
#SBATCH --output=logs/eval-holdout-%x-%j.out
#SBATCH --error=logs/eval-holdout-%x-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=14:00:00

set -euo pipefail

SYSTEM_ID="${SYSTEM_ID:-}"
if [[ -z "${SYSTEM_ID}" ]]; then
  echo "ERROR: set SYSTEM_ID (e.g. sysA_only_teacher, sysD_cascade_distilled)."
  exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-/gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving}"
CONFIG="${EVAL_CONFIG:-configs/routing_eval_holdout.yaml}"
PROMPT_POOL="${PROMPT_POOL:-results/routing_eval_holdout/prompt_pool.jsonl}"

cd "${PROJECT_DIR}"
source env/setup_env.sh
# setup_env.sh unsets PYTHONPATH on BSC; restore repo root so distill/, predictors/,
# routing/ and bench/ are importable for python -m and direct scripts.
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p logs results/routing_eval_holdout

echo "=========================================="
echo " Holdout evaluation: ${SYSTEM_ID}"
echo "=========================================="

python scripts/preflight_eval_holdout.py --config "${CONFIG}"
echo "Date:         $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "CONFIG:       ${CONFIG}"

# Shared prompt pool (built once; safe to rebuild if missing).
if [[ ! -s "${PROMPT_POOL}" ]]; then
  echo "Building shared prompt pool..."
  python -m bench.holdout_pool --config "${CONFIG}" --output "${PROMPT_POOL}"
fi

# Resolve roles for this system from YAML.
mapfile -t ROLES < <(python - <<PY
from utils.config_loader import load_yaml
cfg = load_yaml("${CONFIG}")
sid = "${SYSTEM_ID}"
roles = []
for s in cfg.get("systems", []):
    if s.get("id") == sid:
        roles = list(s.get("roles") or [])
        break
if not roles:
    raise SystemExit(f"Unknown system id: {sid}")
print("\n".join(roles))
PY
)

ENDPOINT_WAIT_S="${ENDPOINT_WAIT_S:-7200}"
ENDPOINT_POLL_S="${ENDPOINT_POLL_S:-5}"
max_attempts=$(( ENDPOINT_WAIT_S / ENDPOINT_POLL_S ))
echo "Waiting for endpoint files: ${ROLES[*]} (up to ${ENDPOINT_WAIT_S}s)"
for role in "${ROLES[@]}"; do
  file="results/routing/endpoints/${role}.url"
  for attempt in $(seq 1 "${max_attempts}"); do
    if [[ -s "${file}" ]]; then
      break
    fi
    sleep "${ENDPOINT_POLL_S}"
  done
  if [[ ! -s "${file}" ]]; then
    echo "ERROR: missing ${file}"
    exit 1
  fi
  url="$(cat "${file}")"
  echo "  ${role} -> ${url}"
  python -m serving.healthcheck --url "${url}" --retries 60 --interval 5
done

python -m routing.run_eval_holdout \
  --config "${CONFIG}" \
  --system "${SYSTEM_ID}" \
  --prompt-pool "${PROMPT_POOL}"

echo "Done: ${SYSTEM_ID}"
