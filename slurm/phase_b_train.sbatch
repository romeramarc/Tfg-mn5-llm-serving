#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — Phase B quality predictor build + training
# Target cluster: MareNostrum 5   (BSC)
# ─────────────────────────────────────────────────────────────
# Reads every Phase B capture trace under
#   results/phase_b/captures/**/trace.jsonl
# (configurable via configs/phase_b.yaml → dataset.trace_glob_patterns),
# materialises both the quality_ex_ante and quality_post_hoc datasets under
#   results/phase_b/datasets/
# and trains a classification predictor for each into
#   results/phase_b/predictors/.
#
# CPU-only — gradient_boosting / random_forest / linear all train fine on
# CPU for the dataset sizes produced in Phase B.
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=phase-b-train
#SBATCH --output=logs/phase-b-train-%j.out
#SBATCH --error=logs/phase-b-train-%j.err
#SBATCH --partition=gpp
#SBATCH --account=bsc98
#SBATCH --qos=gp_bsccs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PHASE_B_CONFIG="${PHASE_B_CONFIG:-configs/phase_b.yaml}"

cd "${PROJECT_DIR}"
source env/setup_env.sh

mkdir -p logs results/phase_b/datasets results/phase_b/predictors

echo "=========================================="
echo " Phase B — build datasets + train predictors"
echo "=========================================="
echo "Date:         $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-interactive}"
echo "Node:         $(hostname)"
echo "Config:       ${PHASE_B_CONFIG}"
echo "Git commit:   $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
echo "=========================================="

python scripts/phase_b_train.py --config "${PHASE_B_CONFIG}"

echo "=== PHASE B TRAIN DONE ==="
