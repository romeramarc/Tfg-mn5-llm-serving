#!/usr/bin/env bash
# ------------------------------------------------------------------
# Chain: predictor retrain (pred-tier) -> holdout v2 (servers + 9 evals).
#
# Usage (login node, project root):
#   # Submit retrain + auto-launch holdout when retrain succeeds:
#   bash slurm/chain_retrain_and_holdout_v2.sh
#
#   # Predictors already running; only queue holdout after that job:
#   bash slurm/chain_retrain_and_holdout_v2.sh <PRED_JOB_ID>
#
# Requires on BSC: git pull with routing_eval_holdout_v2.yaml (9 systems) and
# slurm/retrain_predictors_model_tier.sbatch that writes v2_retrained.yaml.
# ------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving}"
cd "${PROJECT_DIR}"
mkdir -p logs

EVAL_CONFIG="${EVAL_CONFIG:-configs/routing_eval_holdout_v2_retrained.yaml}"
PROMPT_POOL="${PROMPT_POOL:-results/routing_eval_holdout/prompt_pool.jsonl}"
PRED_JOB="${1:-}"

if [[ -z "${PRED_JOB}" ]]; then
  PRED_JOB="$(sbatch --parsable slurm/retrain_predictors_model_tier.sbatch)"
  echo "Submitted predictor retrain (pred-tier): job ${PRED_JOB}"
else
  echo "Using existing predictor retrain job: ${PRED_JOB}"
fi

LAUNCH_JID="$(sbatch --parsable \
  --job-name="holdout-v2-chain" \
  --partition=gpp \
  --account=bsc98 \
  --qos=gp_bsccs \
  --cpus-per-task=1 \
  --time=01:00:00 \
  --output="logs/holdout-v2-chain-%j.out" \
  --error="logs/holdout-v2-chain-%j.err" \
  --dependency="afterok:${PRED_JOB}" \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",EVAL_CONFIG="${EVAL_CONFIG}",PROMPT_POOL="${PROMPT_POOL}" \
  slurm/holdout_v2_after_retrain.sbatch)"

echo "Submitted holdout launcher (afterok:${PRED_JOB}): job ${LAUNCH_JID}"
echo "When pred-tier finishes OK, launcher will:"
echo "  1) preflight ${EVAL_CONFIG}"
echo "  2) bash slurm/launch_eval_holdout.sh all  (4 servers + 9 eval clients)"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "          tail -f logs/predictors-model-tier-${PRED_JOB}.out"
echo "          tail -f logs/holdout-v2-chain-${LAUNCH_JID}.out"
