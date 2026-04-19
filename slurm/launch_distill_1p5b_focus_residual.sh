#!/bin/bash
# ------------------------------------------------------------------
# Launch 1.5B-focused distillation with optional residual stage-2.
#
# Pipeline:
#   distill_generate -> distill_train_1.5b -> distill_residual_1.5b -> posteval_1.5b
#
# Usage:
#   bash slurm/launch_distill_1p5b_focus_residual.sh
#   DISTILL_CONFIG=configs/distill_1p5b_focus.yaml bash slurm/launch_distill_1p5b_focus_residual.sh
# ------------------------------------------------------------------

set -euo pipefail

cd "$(dirname "$0")/.."
DISTILL_CONFIG="${DISTILL_CONFIG:-configs/distill_1p5b_focus.yaml}"

if [[ ! -f "${DISTILL_CONFIG}" ]]; then
  echo "ERROR: Distillation config not found: ${DISTILL_CONFIG}"
  exit 1
fi

echo "=========================================="
echo " Launch 1.5B focus + residual stage2"
echo "=========================================="
echo "UTC time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Git HEAD:       $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
echo "Config:         ${DISTILL_CONFIG}"
echo "Pipeline:       generate -> train_1.5b -> residual_1.5b -> posteval_1.5b"
echo "=========================================="

python -m distill.preflight --step 1 --simulate-offline --config "${DISTILL_CONFIG}"

JOB_DG=$(sbatch --parsable --export="ALL,DISTILL_CONFIG=${DISTILL_CONFIG}" slurm/distill_generate.sbatch)
echo "[1/4] distill-generate      : ${JOB_DG}"

JOB_T1=$(sbatch --parsable --dependency=afterok:${JOB_DG} --export="ALL,DISTILL_CONFIG=${DISTILL_CONFIG}" slurm/distill_train_1.5b.sbatch)
echo "[2/4] train-1.5b            : ${JOB_T1} (after ${JOB_DG})"

JOB_R2=$(sbatch --parsable --dependency=afterok:${JOB_T1} --export="ALL,DISTILL_CONFIG=${DISTILL_CONFIG}" slurm/distill_residual_1.5b.sbatch)
echo "[3/4] residual-stage2-1.5b  : ${JOB_R2} (after ${JOB_T1})"

JOB_P1=$(sbatch --parsable --dependency=afterok:${JOB_R2} slurm/posteval_1.5b.sbatch)
echo "[4/4] posteval-1.5b         : ${JOB_P1} (after ${JOB_R2})"

echo ""
echo "Submitted jobs:"
echo "  ${JOB_DG} distill-generate"
echo "  ${JOB_T1} train-1.5b"
echo "  ${JOB_R2} residual-stage2-1.5b"
echo "  ${JOB_P1} posteval-1.5b"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/distill-gen-${JOB_DG}.out"
echo "  tail -f logs/distill-1.5b-${JOB_T1}.out"
echo "  tail -f logs/distill-res1.5b-${JOB_R2}.out"
