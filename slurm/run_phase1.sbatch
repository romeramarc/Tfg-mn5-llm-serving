#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — Phase-1 all-in-one: teacher server + benchmarks
# Target cluster: MareNostrum 5   (BSC)
#
# This script runs the complete Phase-1 pipeline on a single node:
#   1. Start vLLM teacher server in the background
#   2. Wait for /health to return HTTP 200
#   3. Run offline throughput benchmark (vllm bench serve)
#   4. Run online load benchmark (async HTTP client)
#   5. Run quality evaluation (GSM8K + MATH)
#   6. Gracefully stop the server
#
# All outputs go to timestamped subdirectories under results/.
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=phase1-teacher
#SBATCH --output=logs/phase1-%j.out
#SBATCH --error=logs/phase1-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --gres=gpu:1                     # 1× H100 — 14B fits in single GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --exclusive
# ─────────────────────────────────────────────────────────────
# QoS options for this account (bsc98):
#   acc_bsccs → production GPU QoS (used here)
#   acc_debug+ → short debug queue (≤2h, for quick tests)
#   Uncomment and set the appropriate value:
# #SBATCH --qos=acc_bsccs
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# ── Environment ────────────────────────────────────────────
source env/setup_env.sh
mkdir -p logs

# ── Header ─────────────────────────────────────────────────
echo "=========================================="
echo " Phase-1 Pipeline — Teacher + Benchmarks"
echo "=========================================="
echo "Date:          $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID:  ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "GPUs:          ${SLURM_GPUS_ON_NODE:-unknown}"
echo "Git commit:    $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "=========================================="

SERVER_URL="http://localhost:8000"
SERVER_PID=""

# ── Cleanup trap ───────────────────────────────────────────
cleanup() {
    echo ""
    echo "--- Stopping vLLM server (PID ${SERVER_PID:-unknown}) ---"
    if [[ -n "${SERVER_PID}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    echo "--- Phase-1 pipeline finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
}
trap cleanup EXIT

# ── Stage 1: Start teacher server in background ────────────
echo ""
echo "--- [1/4] Starting teacher server ---"
python -m serving.start_server \
    --config configs/serving.yaml \
    --models configs/models.yaml \
    --role teacher &
SERVER_PID=$!
echo "vLLM server PID: ${SERVER_PID}"

# ── Stage 2: Wait for server to be healthy ─────────────────
echo ""
echo "--- [2/4] Waiting for /health (up to 10 min) ---"
python -m serving.healthcheck \
    --url "${SERVER_URL}" \
    --retries 120 --interval 5
echo "Server is healthy at ${SERVER_URL}"

# ── Stage 3: Throughput benchmark ─────────────────────────
echo ""
echo "--- [3/4] Throughput benchmark ---"
python -m bench.run_throughput --config configs/benchmark.yaml
echo "Throughput benchmark complete."

# ── Stage 4: Online load benchmark ────────────────────────
echo ""
echo "--- [4/4] Online load benchmark ---"
python -m bench.run_online_load --config configs/benchmark.yaml
echo "Online load benchmark complete."

# ── Stage 5: Quality evaluation (optional) ────────────────
# Uncomment to include GSM8K + MATH evaluation in the same job.
# Wall-time budget: GSM8K ~30 min, MATH-500 ~40 min at full model speed.
#
# echo ""
# echo "--- [5/5] Quality evaluation ---"
# python -m eval.run_quality --config configs/eval.yaml
# echo "Quality evaluation complete."

echo ""
echo "=========================================="
echo " Phase-1 complete. Results in results/"
echo "=========================================="
