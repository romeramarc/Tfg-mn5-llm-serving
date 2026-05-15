#!/bin/bash
# Refine Gradient Boosting only (Phase A + B). RF results in predictors_refined/ are kept.
#SBATCH --job-name=refine-gb-only
#SBATCH --output=logs/refine-gb-only-%j.out
#SBATCH --error=logs/refine-gb-only-%j.err
#SBATCH --partition=gpp
#SBATCH --account=bsc98
#SBATCH --qos=gp_bsccs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_DIR}"
source env/setup_env.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

mkdir -p logs results/phase_a/predictors_refined_gb results/phase_b/predictors_refined_gb

echo "=== Smoke: HistGradientBoosting estimators (no n_jobs) ==="
python -c "
from predictors.training.refine import _empty_estimator
r = _empty_estimator(task='regression', family='gradient_boosting', seed=42)
c = _empty_estimator(task='classification', family='gradient_boosting', seed=42)
print('OK', type(r).__name__, type(c).__name__)
"

echo "=== Phase A GB only ==="
python scripts/refine_predictors.py --config configs/refine_phase_a_gb_only.yaml

echo "=== Phase B GB only ==="
python scripts/refine_predictors.py --config configs/refine_phase_b_gb_only.yaml

echo "=== REFINE GB ONLY DONE ==="
