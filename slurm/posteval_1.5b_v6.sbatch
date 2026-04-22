#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — Exp. 6 · Step 3: Posteval for 1.5B (full-FT model)
#   Full-parameter FT ⇒ the training step saves the FULL model directly
#   under `final_model/` — no PEFT merge is needed. Point vLLM at that
#   directory, run quality eval + perf benchmarks.
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=posteval-1.5b-v6
#SBATCH --output=logs/posteval-1.5b-v6-%j.out
#SBATCH --error=logs/posteval-1.5b-v6-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00

set -euo pipefail

cd /gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving
source env/setup_env.sh
mkdir -p logs results/quality results/throughput results/online

DISTILL_CONFIG="${DISTILL_CONFIG:-configs/distill_v6.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/eval.yaml}"
SERVER_URL="http://localhost:8000"
NUM_REPS="${NUM_REPS:-3}"
SKIP_PERF_BENCH="${SKIP_PERF_BENCH:-0}"
EVAL_ROLE="${EVAL_ROLE:-distilled_student_small_v6}"

# By default, pick the newest Exp. 6 full-FT 1.5B run.
MODEL_GLOB="${MODEL_GLOB:-results/distill/sft-full-qwen2.5-1.5b-v6-*/final_model}"
MODEL_DIR="${MODEL_DIR:-}"

_SERVER_PID=""
cleanup() {
    if [[ -n "${_SERVER_PID}" ]]; then
        kill "${_SERVER_PID}" 2>/dev/null || true
        wait "${_SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ -z "${MODEL_DIR}" ]]; then
    MODEL_DIR=$(ls -dt ${MODEL_GLOB} 2>/dev/null | head -1 || true)
fi
if [[ -z "${MODEL_DIR}" || ! -d "${MODEL_DIR}" ]]; then
    echo "ERROR: No Exp. 6 full-FT 1.5B model directory found."
    echo "       MODEL_GLOB=${MODEL_GLOB}"
    echo "       Set MODEL_DIR=/path/to/final_model if auto-resolve fails."
    exit 1
fi

echo "========================================="
echo " Exp. 6 · Step 3: Posteval 1.5B (full-FT)"
echo " Job ID:   ${SLURM_JOB_ID}"
echo " Node:     $(hostname)"
echo " Date:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " Model:    ${MODEL_DIR}"
echo " Eval cfg: ${EVAL_CONFIG}"
echo " Eval role: ${EVAL_ROLE}"
echo " Skip perf: ${SKIP_PERF_BENCH}"
echo "========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Start vLLM on the full-FT model (no merge step) ─────────
echo ""
echo "--- [1/3] Starting vLLM on ${MODEL_DIR} ---"
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_DIR}" \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --dtype auto --seed 42 &
_SERVER_PID=$!
python -m serving.healthcheck --url "${SERVER_URL}" --retries 120 --interval 5
echo "Server healthy."

# ── Quality eval (GSM8K + MATH) ─────────────────────────────
echo ""
echo "--- [2/3] Running quality evaluation ---"
python -m eval.run_quality \
    --config "${EVAL_CONFIG}" \
    --model "${MODEL_DIR}" \
    --role  "${EVAL_ROLE}"

# ── Perf benchmarks (throughput + online) ───────────────────
if [[ "${SKIP_PERF_BENCH}" == "1" ]]; then
    echo "Skipping throughput/online benchmarks (SKIP_PERF_BENCH=1)"
else
    echo ""
    echo "--- [3/3] Running throughput + online benchmarks (${NUM_REPS} reps) ---"
    for REP in $(seq 1 ${NUM_REPS}); do
        python -m bench.run_throughput  --config configs/benchmark.yaml --model "${MODEL_DIR}" --role "${EVAL_ROLE}"
        python -m bench.run_online_load --config configs/benchmark.yaml --model "${MODEL_DIR}" --role "${EVAL_ROLE}"
    done
fi

echo ""
echo "========================================="
echo " Posteval 1.5B (Exp. 6) done."
echo " Model:       ${MODEL_DIR}"
echo " Quality dir: results/quality/quality-${EVAL_ROLE}-*/"
echo "========================================="
