#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — Efficiency benchmarks with 3 repetitions
# Target cluster: MareNostrum 5   (BSC)
#
# Runs throughput + online benchmarks 3× per model to compute
# mean ± std.  Each repetition is a separate run directory.
#
# Usage:
#   sbatch slurm/bench_repetitions.sbatch
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=bench-reps
#SBATCH --output=logs/bench-reps-%j.out
#SBATCH --error=logs/bench-reps-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --exclusive
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# ── Environment ────────────────────────────────────────────
cd /gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving
source env/setup_env.sh
mkdir -p logs

NUM_REPS=3
SERVER_URL="http://localhost:8000"
_SERVER_PID=""

echo "=========================================="
echo " Efficiency Benchmarks — 3 repetitions"
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

# ── Cleanup trap ───────────────────────────────────────────
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

# ── Helper: benchmark one role/model with repetitions ──────
bench_model() {
    local ROLE=$1
    local MODEL_NAME=$2

    echo ""
    echo "════════════════════════════════════════"
    echo " Benchmarking: ${ROLE}  (${MODEL_NAME})"
    echo "════════════════════════════════════════"

    # Start server
    echo "[1/2] Starting server…"
    python -m serving.start_server \
        --config configs/serving.yaml \
        --models configs/models.yaml \
        --role "${ROLE}" &
    _SERVER_PID=$!
    echo "      Server PID: ${_SERVER_PID}"

    # Wait for health
    echo "[2/2] Waiting for /health (up to 10 min)…"
    python -m serving.healthcheck \
        --url "${SERVER_URL}" \
        --retries 120 --interval 5
    echo "      Server healthy."

    # Run repetitions
    for REP in $(seq 1 ${NUM_REPS}); do
        echo ""
        echo "──── Repetition ${REP}/${NUM_REPS} ────"

        echo "[${REP}a] Throughput benchmark…"
        python -m bench.run_throughput \
            --config configs/benchmark.yaml \
            --model "${MODEL_NAME}" \
            --role  "${ROLE}"
        echo "      Throughput done (rep ${REP})."

        echo "[${REP}b] Online load benchmark…"
        python -m bench.run_online_load \
            --config configs/benchmark.yaml \
            --model "${MODEL_NAME}" \
            --role  "${ROLE}"
        echo "      Online load done (rep ${REP})."

        echo "──── Repetition ${REP}/${NUM_REPS} complete ────"
    done

    # Stop server
    echo "       Stopping server…"
    kill "${_SERVER_PID}" 2>/dev/null || true
    wait "${_SERVER_PID}" 2>/dev/null || true
    _SERVER_PID=""
    echo "       Server stopped."
    sleep 5
}

# ── Run all three models ────────────────────────────────────
bench_model "teacher"       "Qwen/Qwen2.5-14B-Instruct"
bench_model "student_mid"   "Qwen/Qwen2.5-7B-Instruct"
bench_model "student_small" "Qwen/Qwen2.5-1.5B-Instruct"

echo ""
echo "=========================================="
echo " All efficiency benchmarks complete (${NUM_REPS} reps each)."
echo " Results in results/throughput/ and results/online/"
echo "=========================================="
