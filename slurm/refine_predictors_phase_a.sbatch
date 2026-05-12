#!/bin/bash
# SLURM — Refine Phase A only (service_cost RF + GB). Same resources as full refine.
#SBATCH --job-name=refine-ph-a
#SBATCH --output=logs/refine-phase-a-%j.out
#SBATCH --error=logs/refine-phase-a-%j.err
#SBATCH --partition=gpp
#SBATCH --account=bsc98
#SBATCH --qos=gp_bsccs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
REFINE_PHASE_A_CONFIG="${REFINE_PHASE_A_CONFIG:-configs/refine_phase_a.yaml}"
cd "${PROJECT_DIR}"
source env/setup_env.sh
mkdir -p logs results/phase_a/predictors_refined
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
python scripts/refine_predictors.py --config "${REFINE_PHASE_A_CONFIG}"
echo "=== REFINE PHASE A DONE ==="
