#!/bin/bash
# ------------------------------------------------------------------
# Launch only the 1.5B distillation branch with a dedicated config.
#
# Pipeline:
#   distill_generate.sbatch -> distill_train_1.5b.sbatch -> posteval_1.5b.sbatch
#
# Override config path if needed:
#   DISTILL_CONFIG=configs/distill_1p5b_focus.yaml bash slurm/launch_distill_1p5b_focus.sh
# ------------------------------------------------------------------

set -euo pipefail

cd "$(dirname "$0")/.."
DISTILL_CONFIG="${DISTILL_CONFIG:-configs/distill_1p5b_focus.yaml}"

if [[ ! -f "${DISTILL_CONFIG}" ]]; then
  echo "ERROR: Distillation config not found: ${DISTILL_CONFIG}"
  exit 1
fi

echo "=========================================="
echo " Launch 1.5B-focused distillation"
echo "=========================================="
echo "UTC time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Git HEAD:       $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
echo "Config:         ${DISTILL_CONFIG}"
echo "Pipeline:       generate -> train_1.5b -> posteval_1.5b"
echo "=========================================="

echo "Running preflight step 1 with offline simulation..."
python -m distill.preflight --step 1 --simulate-offline --config "${DISTILL_CONFIG}"

JOB_DG=$(sbatch --parsable --export="ALL,DISTILL_CONFIG=${DISTILL_CONFIG}" slurm/distill_generate.sbatch)
echo "[1/3] distill-generate  : ${JOB_DG}"

JOB_T1=$(sbatch --parsable --dependency=afterok:${JOB_DG} --export="ALL,DISTILL_CONFIG=${DISTILL_CONFIG}" slurm/distill_train_1.5b.sbatch)
echo "[2/3] train-1.5b        : ${JOB_T1} (after ${JOB_DG})"

JOB_P1=$(sbatch --parsable --dependency=afterok:${JOB_T1} slurm/posteval_1.5b.sbatch)
echo "[3/3] posteval-1.5b     : ${JOB_P1} (after ${JOB_T1})"

echo ""
echo "Submitted jobs:"
echo "  ${JOB_DG} distill-generate"
echo "  ${JOB_T1} train-1.5b"
echo "  ${JOB_P1} posteval-1.5b"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/distill-gen-${JOB_DG}.out"
echo "  tail -f logs/distill-1.5b-${JOB_T1}.out"
