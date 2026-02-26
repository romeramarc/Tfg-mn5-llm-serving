#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — student_mid efficiency baseline (Qwen2.5-7B)
# Target cluster: MareNostrum 5   (BSC)
#
# Purpose: pre-distillation baseline for Qwen2.5-7B-Instruct.
# These results will be compared against the distilled student
# to demonstrate that knowledge distillation improved quality/speed.
#
# Pipeline:
#   1. Start vLLM server (7B model)
#   2. Wait for /health
#   3. Offline throughput benchmark
#   4. Online load benchmark
#   5. Stop server
#
# Results land in:
#   results/throughput/throughput-student_mid-<ts>/
#   results/online/online-student_mid-<ts>/
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=bench-student-mid
#SBATCH --output=logs/bench-student-mid-%j.out
#SBATCH --error=logs/bench-student-mid-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --exclusive
# ─────────────────────────────────────────────────────────────

set -euo pipefail

source env/setup_env.sh
mkdir -p logs

ROLE="student_mid"
MODEL="Qwen/Qwen2.5-7B-Instruct"
SERVER_URL="http://localhost:8000"
SERVER_PID=""

cleanup() {
    echo ""
    echo "--- Stopping vLLM server (PID ${SERVER_PID:-unknown}) ---"
    if [[ -n "${SERVER_PID}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    echo "--- Job finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
}
trap cleanup EXIT

echo "=========================================="
echo " Baseline: ${ROLE}  (${MODEL})"
echo "=========================================="
echo "Date:         $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=========================================="

# ── [1/4] Start server ─────────────────────────────────────
echo ""
echo "--- [1/4] Starting vLLM server ---"
python -m serving.start_server \
    --config configs/serving.yaml \
    --models configs/models.yaml \
    --role   "${ROLE}" &
SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

# ── [2/4] Wait for health ──────────────────────────────────
echo ""
echo "--- [2/4] Waiting for /health (up to 10 min) ---"
python -m serving.healthcheck \
    --url "${SERVER_URL}" \
    --retries 120 --interval 5
echo "Server is healthy."

# ── [3/4] Throughput benchmark ─────────────────────────────
echo ""
echo "--- [3/4] Throughput benchmark ---"
python -m bench.run_throughput \
    --config configs/benchmark.yaml \
    --model  "${MODEL}" \
    --role   "${ROLE}"
echo "Throughput done."

# ── [4/4] Online load benchmark ────────────────────────────
echo ""
echo "--- [4/4] Online load benchmark ---"
python -m bench.run_online_load \
    --config configs/benchmark.yaml \
    --model  "${MODEL}" \
    --role   "${ROLE}"
echo "Online load done."

echo ""
echo "=========================================="
echo " ${ROLE} baseline complete."
echo " results/throughput/throughput-${ROLE}-*"
echo " results/online/online-${ROLE}-*"
echo "=========================================="
