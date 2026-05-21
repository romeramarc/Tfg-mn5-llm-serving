#!/bin/bash
#SBATCH --job-name=holdout-v2-chain
#SBATCH --output=logs/holdout-v2-chain-%j.out
#SBATCH --error=logs/holdout-v2-chain-%j.err
#SBATCH --partition=gpp
#SBATCH --account=bsc98
#SBATCH --qos=gp_bsccs
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/routing_eval_holdout_v2_retrained.yaml}"
PROMPT_POOL="${PROMPT_POOL:-results/routing_eval_holdout/prompt_pool.jsonl}"

cd "${PROJECT_DIR}"
source env/setup_env.sh
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "=== HOLDOUT V2 CHAIN (post retrain) ==="
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "EVAL_CONFIG=${EVAL_CONFIG}"

if [[ ! -f "${EVAL_CONFIG}" ]]; then
  echo "ERROR: ${EVAL_CONFIG} missing; predictor retrain did not finish or failed."
  exit 1
fi

echo "=== EARLY SANITY ==="
python -c "from utils.config_loader import load_yaml; from routing.policies import POLICIES; print('imports OK')"
python -c "from utils.config_loader import load_yaml; c=load_yaml('${EVAL_CONFIG}'); ids=[s['id'] for s in c['systems']]; exp=['sysC_l010','sysC_l025','sysC_l050','sysC_l100','sysD_cascade4','sysE_l010','sysE_l025','sysE_l050','sysE_l100']; assert ids==exp, f'system list mismatch: {ids}'; print('OK 9 systems')"

echo "=== PREFLIGHT ==="
python scripts/preflight_eval_holdout.py --config "${EVAL_CONFIG}"

echo "=== LAUNCH SERVERS + EVAL CLIENTS ==="
export EVAL_CONFIG PROMPT_POOL PROJECT_DIR
bash slurm/launch_eval_holdout.sh all

echo "=== HOLDOUT V2 CHAIN SUBMIT DONE ==="
