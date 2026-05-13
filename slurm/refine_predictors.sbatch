#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — Refine RF/GB predictors (Phase A + Phase B)
# Target cluster: MareNostrum 5   (BSC)
# ─────────────────────────────────────────────────────────────
# Runs scripts/refine_predictors.py twice in a single job:
#   1) configs/refine_phase_a.yaml  → service_cost regressor
#   2) configs/refine_phase_b.yaml  → quality_ex_ante + quality_post_hoc
#
# Reuses the same train/val/test split as the baseline (seed=42),
# refines only RandomForest and HistGradientBoosting (the two
# strongest baseline families), and tunes the operating threshold
# on the validation split for classifiers with an asymmetric FP/FN
# cost criterion documented in configs/refine_phase_b.yaml.
#
# CPU-only — refinement is dominated by tree builds, no GPU required.
#
# Wall time: 6 model searches (A: RF+GB, B: 2× RF+GB) with n_iter=30 and
# 3-fold GroupKFold can exceed 4h on busy GPP nodes; a prior MN5 run hit
# SLURM TIMEOUT mid Phase-A GB. Keep this comfortably above worst case.
# Combined job: 8h wall clock (Phase A+B in one submission).
# Scheduling: 16 CPUs matches CV parallelism (n_splits=3, n_jobs=-1) and is
# easier to place on GPP than 32 CPUs for 12h; other CPU jobs in this repo
# typically request 16-20 cores.
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=refine-predictors
#SBATCH --output=logs/refine-predictors-%j.out
#SBATCH --error=logs/refine-predictors-%j.err
#SBATCH --partition=gpp
#SBATCH --account=bsc98
#SBATCH --qos=gp_bsccs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
REFINE_PHASE_A_CONFIG="${REFINE_PHASE_A_CONFIG:-configs/refine_phase_a.yaml}"
REFINE_PHASE_B_CONFIG="${REFINE_PHASE_B_CONFIG:-configs/refine_phase_b.yaml}"

cd "${PROJECT_DIR}"
source env/setup_env.sh

mkdir -p logs \
         results/phase_a/predictors_refined \
         results/phase_b/predictors_refined

echo "=========================================="
echo " Refine RF/GB predictors (Phase A + Phase B)"
echo "=========================================="
echo "Date:           $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID:   ${SLURM_JOB_ID:-interactive}"
echo "Node:           $(hostname)"
echo "CPUs per task:  ${SLURM_CPUS_PER_TASK:-?}"
echo "Phase A config: ${REFINE_PHASE_A_CONFIG}"
echo "Phase B config: ${REFINE_PHASE_B_CONFIG}"
echo "Git commit:     $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
echo "=========================================="

# Sklearn picks up these env vars; keep them aligned with the CPUs we asked for.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "--- Phase A refinement -------------------"
python scripts/refine_predictors.py --config "${REFINE_PHASE_A_CONFIG}"

echo "--- Phase B refinement -------------------"
python scripts/refine_predictors.py --config "${REFINE_PHASE_B_CONFIG}"

echo "=== REFINE PREDICTORS DONE ==="
