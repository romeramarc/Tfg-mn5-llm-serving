#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — Routing experiment: Teacher evaluation
# Target cluster: MareNostrum 5   (BSC)
#
# Evaluates the teacher model on the routing benchmark prompts
# to establish a quality/latency ceiling for the routing system.
# Then runs the student + teacher confidence-based routing.
#
# Usage:
#   sbatch slurm/eval_routing.sbatch
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=eval-routing
#SBATCH --output=logs/eval-routing-%j.out
#SBATCH --error=logs/eval-routing-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00
#SBATCH --exclusive
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# ── Environment ────────────────────────────────────────────
cd /gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving
source env/setup_env.sh
mkdir -p logs results/routing

TEACHER_MODEL="Qwen/Qwen2.5-14B-Instruct"
SERVER_URL="http://localhost:8000"
_SERVER_PID=""

echo "=========================================="
echo " Routing Experiment: Teacher baseline"
echo "=========================================="
echo "Date:          $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID:  ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "Git commit:    $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "pip freeze (first 60 packages):"
pip freeze 2>/dev/null | head -60
echo "..."
echo "=========================================="

cleanup() {
    if [[ -n "${_SERVER_PID}" ]]; then
        echo "--- Stopping server (PID ${_SERVER_PID}) ---"
        kill "${_SERVER_PID}" 2>/dev/null || true
        wait "${_SERVER_PID}" 2>/dev/null || true
    fi
    echo ""
    echo "--- Final GPU state ---"
    nvidia-smi
    echo "--- Job finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
}
trap cleanup EXIT

# ── [1/3] Start teacher server ─────────────────────────────
echo ""
echo "--- [1/3] Starting teacher server ---"
python -m serving.start_server \
    --config configs/serving.yaml \
    --models configs/models.yaml \
    --role teacher &
_SERVER_PID=$!
echo "Server PID: ${_SERVER_PID}"

# ── [2/3] Wait for /health ────────────────────────────────
echo ""
echo "--- [2/3] Waiting for /health (up to 10 min) ---"
python -m serving.healthcheck \
    --url "${SERVER_URL}" \
    --retries 120 --interval 5
echo "Server is healthy."

# ── [3/3] Run routing experiment ───────────────────────────
echo ""
echo "--- [3/3] Routing experiment ---"

# If a routing module exists, use it; otherwise run quality eval
# with routing prompts as a quality baseline for the teacher.
if python -c "import routing.run_routing" 2>/dev/null; then
    echo "Running routing experiment module…"
    python -m routing.run_routing \
        --config configs/routing.yaml
else
    echo "routing.run_routing not found; running quality eval as fallback…"
    python -m eval.run_quality \
        --config configs/eval.yaml \
        --model  "${TEACHER_MODEL}" \
        --role   "teacher_routing"
fi

echo "Routing experiment done."

echo ""
echo "=========================================="
echo " Routing experiment complete."
echo " results/routing/ or results/quality/quality-teacher_routing-*"
echo "=========================================="
