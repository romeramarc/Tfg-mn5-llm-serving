#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — Phase-1 efficiency baselines: teacher + students
# Target cluster: MareNostrum 5   (BSC)
#
# Runs the complete efficiency benchmark suite for all three
# Qwen2.5 models sequentially on a single H100 node:
#
#   Per model (teacher 14B → student_mid 7B → student_small 1.5B):
#     1. Start vLLM server in the background
#     2. Wait for /health to return HTTP 200
#     3. Run offline throughput benchmark (vllm bench serve)
#     4. Run online load benchmark (async HTTP client)
#     5. Stop the server
#
# All outputs go to timestamped subdirectories under results/.
# Results are labelled by model role in the run directory name.
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=baselines
#SBATCH --output=logs/baselines-%j.out
#SBATCH --error=logs/baselines-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --gres=gpu:1                     # 1× H100-80G — all models fit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00
#SBATCH --exclusive
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# ── Environment ────────────────────────────────────────────
source env/setup_env.sh
mkdir -p logs

# ── Header ─────────────────────────────────────────────────
echo "=========================================="
echo " Phase-1 Efficiency Baselines"
echo " Teacher (14B) + Students (7B, 1.5B)"
echo "=========================================="
echo "Date:          $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID:  ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "GPUs:          ${SLURM_GPUS_ON_NODE:-unknown}"
echo "Git commit:    $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "=========================================="

SERVER_URL="http://localhost:8000"

# ── Cleanup trap ───────────────────────────────────────────
_SERVER_PID=""
cleanup() {
    if [[ -n "${_SERVER_PID}" ]]; then
        echo "--- Stopping server (PID ${_SERVER_PID}) ---"
        kill "${_SERVER_PID}" 2>/dev/null || true
        wait "${_SERVER_PID}" 2>/dev/null || true
    fi
    echo "--- Baselines job finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
}
trap cleanup EXIT

# ── Helper: benchmark one role/model ───────────────────────
bench_model() {
    local ROLE=$1
    local MODEL_NAME=$2

    echo ""
    echo "────────────────────────────────────────"
    echo " Benchmarking: ${ROLE}  (${MODEL_NAME})"
    echo "────────────────────────────────────────"

    # Start server
    echo "[1/3] Starting server…"
    python -m serving.start_server \
        --config configs/serving.yaml \
        --models configs/models.yaml \
        --role "${ROLE}" &
    _SERVER_PID=$!
    echo "      Server PID: ${_SERVER_PID}"

    # Wait for health
    echo "[2/3] Waiting for /health (up to 10 min)…"
    python -m serving.healthcheck \
        --url "${SERVER_URL}" \
        --retries 120 --interval 5
    echo "      Server healthy."

    # Throughput benchmark
    echo "[3a/3] Throughput benchmark…"
    python -m bench.run_throughput \
        --config configs/benchmark.yaml \
        --model "${MODEL_NAME}"
    echo "       Throughput done."

    # Online load benchmark
    echo "[3b/3] Online load benchmark…"
    python -m bench.run_online_load \
        --config configs/benchmark.yaml \
        --model "${MODEL_NAME}"
    echo "       Online load done."

    # Stop server gracefully
    echo "       Stopping server…"
    kill "${_SERVER_PID}" 2>/dev/null || true
    wait "${_SERVER_PID}" 2>/dev/null || true
    _SERVER_PID=""
    echo "       Server stopped."
    sleep 5   # brief cooldown before next model
}

# ── Run all three models ────────────────────────────────────
bench_model "teacher"       "Qwen/Qwen2.5-14B-Instruct"
bench_model "student_mid"   "Qwen/Qwen2.5-7B-Instruct"
bench_model "student_small" "Qwen/Qwen2.5-1.5B-Instruct"

echo ""
echo "=========================================="
echo " All baselines complete. Results in results/"
echo "=========================================="
