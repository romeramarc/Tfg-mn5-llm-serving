#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — Distillation Step 1: Generate teacher outputs
#   Starts vLLM teacher → collects completions for all benchmark
#   prompts (GSM8K + MATH-500 + ARC-Challenge) → saves JSONL.
#   ~2991 prompts × greedy decode ≈ 2-3 h on 1×H100.
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=distill-gen
#SBATCH --output=logs/distill-gen-%j.out
#SBATCH --error=logs/distill-gen-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=04:00:00

set -euo pipefail

# ── Environment ───────────────────────────────────────────
cd /gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving
source env/setup_env.sh
mkdir -p logs results/distill

MODEL="Qwen/Qwen2.5-14B-Instruct"

echo "========================================="
echo " Distillation Step 1: Teacher generation"
echo " Model:  ${MODEL}"
echo " Job ID: ${SLURM_JOB_ID}"
echo " Node:   $(hostname)"
echo " Date:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── [1/3] Start vLLM teacher server ─────────────────────────
echo ""
echo "--- [1/3] Starting teacher server ---"
python -m serving.start_server \
    --config configs/serving.yaml \
    --models configs/models.yaml \
    --role   teacher &
SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

# ── [2/3] Wait for /health ──────────────────────────────────
echo ""
echo "--- [2/3] Waiting for /health (up to 10 min) ---"
python -m serving.healthcheck \
    --url "http://localhost:8000" \
    --retries 120 --interval 5
echo "Server is healthy."

# ── [3/3] Generate teacher outputs ──────────────────────────
echo ""
echo "--- [3/3] Generating teacher outputs ---"
python -m distill.generate_teacher_outputs \
    --config configs/distill.yaml
echo "Teacher generation done."

# ── Stop server ─────────────────────────────────────────────
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true
echo "Server stopped."

echo ""
echo "========================================="
echo " Teacher outputs saved to:"
echo " results/distill/teacher_outputs.jsonl"
echo "========================================="
