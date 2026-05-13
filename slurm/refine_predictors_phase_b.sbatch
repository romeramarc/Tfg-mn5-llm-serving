#!/bin/bash
# SLURM — Refine Phase B only (quality ex-ante + post-hoc RF + GB).
#SBATCH --job-name=refine-ph-b
#SBATCH --output=logs/refine-phase-b-%j.out
#SBATCH --error=logs/refine-phase-b-%j.err
#SBATCH --partition=gpp
#SBATCH --account=bsc98
#SBATCH --qos=gp_bsccs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
REFINE_PHASE_B_CONFIG="${REFINE_PHASE_B_CONFIG:-configs/refine_phase_b.yaml}"
cd "${PROJECT_DIR}"
source env/setup_env.sh
mkdir -p logs results/phase_b/predictors_refined
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
python scripts/refine_predictors.py --config "${REFINE_PHASE_B_CONFIG}"
echo "=== REFINE PHASE B DONE ==="
