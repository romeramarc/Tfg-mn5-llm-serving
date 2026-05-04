#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — Phase A predictor build + training
# Target cluster: MareNostrum 5   (BSC)
# ─────────────────────────────────────────────────────────────
# Reads every Phase A capture trace under
#   results/phase_a/captures/**/trace.jsonl
# (configurable via configs/phase_a.yaml → dataset.trace_glob_patterns),
# materialises the service-cost dataset under
#   results/phase_a/datasets/
# and trains a regression predictor for the chosen target into
#   results/phase_a/predictors/.
#
# CPU-only: gradient_boosting / random_forest / linear all train fine on
# CPU for the dataset sizes produced in Phase A. Switch partition/qos
# to a GPP node if your account does not allow CPU-only jobs on `acc`.
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=phase-a-train
#SBATCH --output=logs/phase-a-train-%j.out
#SBATCH --error=logs/phase-a-train-%j.err
#SBATCH --partition=gpp
#SBATCH --account=bsc98
#SBATCH --qos=gp_bsccs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PHASE_A_CONFIG="${PHASE_A_CONFIG:-configs/phase_a.yaml}"

cd "${PROJECT_DIR}"
source env/setup_env.sh

mkdir -p logs results/phase_a/datasets results/phase_a/predictors

echo "=========================================="
echo " Phase A — build dataset + train predictor"
echo "=========================================="
echo "Date:         $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-interactive}"
echo "Node:         $(hostname)"
echo "Config:       ${PHASE_A_CONFIG}"
echo "Git commit:   $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
echo "=========================================="

python scripts/phase_a_train.py --config "${PHASE_A_CONFIG}"

echo "=== PHASE A TRAIN DONE ==="
