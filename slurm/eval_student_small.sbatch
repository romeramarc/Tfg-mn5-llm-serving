#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — Quality evaluation: Qwen2.5-1.5B-Instruct (student_small)
# Runs GSM8K + MATH-500. Self-contained: starts its own vLLM server.
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=eval-student-small
#SBATCH --output=logs/eval-student-small-%j.out
#SBATCH --error=logs/eval-student-small-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=02:00:00

set -euo pipefail

# ── Environment ─────────────────────────────────────────────
cd /gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving
source env/setup_env.sh
mkdir -p logs

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
ROLE="student_small"
SERVER_URL="http://localhost:8000"

echo "========================================="
echo " Quality Eval: ${ROLE} (${MODEL})"
echo " Job ID: ${SLURM_JOB_ID}"
echo " Node:   $(hostname)"
echo " Date:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── [1/3] Start vLLM server ─────────────────────────────────
echo ""
echo "--- [1/3] Starting server ---"
python -m serving.start_server \
    --config configs/serving.yaml \
    --models configs/models.yaml \
    --role   "${ROLE}" &
SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

# ── [2/3] Wait for /health ───────────────────────────────────
echo ""
echo "--- [2/3] Waiting for /health (up to 10 min) ---"
python -m serving.healthcheck \
    --url "${SERVER_URL}" \
    --retries 120 --interval 5
echo "Server is healthy."

# ── [3/3] Quality evaluation ────────────────────────────────
echo ""
echo "--- [3/3] Quality evaluation (GSM8K + MATH) ---"
python -m eval.run_quality \
    --config configs/eval.yaml \
    --model  "${MODEL}" \
    --role   "${ROLE}"
echo "Quality evaluation done."

# ── Stop server ─────────────────────────────────────────────
kill "${SERVER_PID}" 2>/dev/null || true
wait  "${SERVER_PID}" 2>/dev/null || true
echo "Server stopped."

echo ""
echo "=========================================="
echo " ${ROLE} quality eval complete."
echo " results/quality/quality-${ROLE}-*/quality_summary.json"
echo "=========================================="
