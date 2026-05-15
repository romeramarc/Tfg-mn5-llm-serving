#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Launch capture for 1.5B pre-KD (student_small_base) — Phase A then Phase B.
#
# Does NOT train predictors (SKIP_TRAIN=1). Does NOT touch distilled captures.
# Phase B starts automatically after Phase A capture completes (via SLURM dep).
#
# Usage (MN5 login node, repo root):
#   bash slurm/launch_capture_base_1p5b.sh
#
# Optional:
#   CHAIN_PHASE_B=0   → only submit Phase A (run Phase B manually later)
#   SKIP_TRAIN=1      → default; leave as-is for capture-only
# ─────────────────────────────────────────────────────────────

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd -P)"
mkdir -p logs

ROLE="student_small_base"
PHASE_A_CONFIG="configs/phase_a_capture_base_1p5b.yaml"
PHASE_B_CONFIG="configs/phase_b_capture_base_1p5b.yaml"
BENCHMARK_LABEL="phase_a_workload_base"
CHAIN_PHASE_B="${CHAIN_PHASE_B:-1}"

echo "=========================================="
echo " Capture 1.5B base (pre-KD) — Phase A + B"
echo "=========================================="
echo "Project:     ${PROJECT_DIR}"
echo "Role:        ${ROLE}"
echo "Phase A cfg: ${PHASE_A_CONFIG}"
echo "Phase B cfg: ${PHASE_B_CONFIG}"
echo "Chain B:     ${CHAIN_PHASE_B}"
echo "=========================================="

python scripts/preflight_capture_base_1p5b.py || {
  echo "Preflight failed — fix before submitting to SLURM."
  exit 1
}

LOG_A="${PROJECT_DIR}/logs/launch-capture-base-1p5b-phase-a.log"
export ROLES="${ROLE}"
export SKIP_TRAIN=1
export PHASE_A_CONFIG
export BENCHMARK_LABEL

echo ""
echo ">>> Phase A launcher (blocks until capture jobs are submitted)..."
bash slurm/launch_phase_a.sh 2>&1 | tee "${LOG_A}"

ENV_A="${PROJECT_DIR}/logs/last_phase_a_capture.env"
if [[ ! -f "${ENV_A}" ]]; then
  echo "ERROR: ${ENV_A} not found after Phase A launch."
  exit 1
fi
# shellcheck source=/dev/null
source "${ENV_A}"
CAP_A="${CAPTURE_JOB_student_small_base:-}"
if [[ -z "${CAP_A}" ]]; then
  echo "ERROR: CAPTURE_JOB_student_small_base missing in ${ENV_A}"
  exit 1
fi
echo "Phase A capture job id: ${CAP_A}"

if [[ "${CHAIN_PHASE_B}" != "1" ]]; then
  echo ""
  echo "CHAIN_PHASE_B=0 — done. When capture ${CAP_A} finishes, run:"
  echo "  ROLES=${ROLE} SKIP_TRAIN=1 PHASE_B_CONFIG=${PHASE_B_CONFIG} bash slurm/launch_phase_b.sh"
  exit 0
fi

# Submit Phase B launcher on GPP after Phase A capture succeeds (needs login-node wait loop inside).
WRAP_LOG="${PROJECT_DIR}/logs/launch-capture-base-1p5b-phase-b-wrap-%j.out"
JOB_B_WRAP=$(sbatch --parsable \
  --job-name=capture-base-1p5b-b \
  --partition=gpp --account=bsc98 --qos=gp_bsccs \
  --dependency=afterok:"${CAP_A}" \
  --time=12:00:00 --cpus-per-task=2 --ntasks=1 \
  --output="${PROJECT_DIR}/logs/launch-capture-base-1p5b-phase-b-wrap-%j.out" \
  --error="${PROJECT_DIR}/logs/launch-capture-base-1p5b-phase-b-wrap-%j.err" \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",ROLE="${ROLE}",PHASE_B_CONFIG="${PHASE_B_CONFIG}" \
  --wrap="cd '${PROJECT_DIR}' && export ROLES='${ROLE}' SKIP_TRAIN=1 PHASE_B_CONFIG='${PHASE_B_CONFIG}' && bash slurm/launch_phase_b.sh")

echo ""
echo "Phase B chained launcher job: ${JOB_B_WRAP} (afterok capture ${CAP_A})"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/phase-a-capture-${CAP_A}.out"
echo "  tail -f logs/launch-capture-base-1p5b-phase-b-wrap-${JOB_B_WRAP}.out"
echo ""
echo "When both captures finish, train the ladder-base datasets:"
echo "  PHASE_A_CONFIG=configs/phase_a_train_ladder_base_1p5b.yaml sbatch slurm/phase_a_train.sbatch"
echo "  PHASE_B_CONFIG=configs/phase_b_train_ladder_base_1p5b.yaml sbatch slurm/phase_b_train.sbatch"
