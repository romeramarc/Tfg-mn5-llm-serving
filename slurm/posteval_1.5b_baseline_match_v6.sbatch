#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM — Re-eval 1.5B **pre-KD baseline** usando el **mismo protocolo**
# que `posteval_1.5b_v6.sbatch` (vLLM api_server directo, mismos flags,
# configs/eval.yaml, configs/benchmark.yaml, NUM_REPS).
#
# Diferencia respecto a v6: el modelo es el instruct base de HuggingFace
# (no `final_model` local) y el rol es `student_small` (salidas bajo
# `results/quality/quality-student_small-<timestamp>/`).
#
# Uso (MN5, desde el repo):
#   sbatch slurm/posteval_1.5b_baseline_match_v6.sbatch
# Solo calidad (como si SKIP_PERF en v6):
#   sbatch --export=ALL,SKIP_PERF_BENCH=1 slurm/posteval_1.5b_baseline_match_v6.sbatch
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=posteval-1.5b-base-like-v6
#SBATCH --output=logs/posteval-1.5b-base-like-v6-%j.out
#SBATCH --error=logs/posteval-1.5b-base-like-v6-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00

set -euo pipefail

WORKDIR="${WORKDIR:-/gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving}"
cd "${WORKDIR}"
source env/setup_env.sh
mkdir -p logs results/quality results/throughput results/online

EVAL_CONFIG="${EVAL_CONFIG:-configs/eval.yaml}"
SERVER_URL="http://localhost:8000"
NUM_REPS="${NUM_REPS:-3}"
SKIP_PERF_BENCH="${SKIP_PERF_BENCH:-0}"
EVAL_ROLE="${EVAL_ROLE:-student_small}"
# Mismo identificador que en models.yaml / fase1 (HF hub)
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

_SERVER_PID=""
cleanup() {
    if [[ -n "${_SERVER_PID}" ]]; then
        kill "${_SERVER_PID}" 2>/dev/null || true
        wait "${_SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "========================================="
echo " Posteval baseline 1.5B (protocolo = v6)"
echo " Job ID:    ${SLURM_JOB_ID}"
echo " Node:      $(hostname)"
echo " Date:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " Workdir:   ${WORKDIR}"
echo " Model:     ${MODEL}"
echo " Eval cfg:  ${EVAL_CONFIG}"
echo " Eval role: ${EVAL_ROLE}"
echo " Skip perf: ${SKIP_PERF_BENCH}"
echo " Git:       $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Misma línea de servidor que posteval_1.5b_v6.sbatch (modelo HF) ─
echo ""
echo "--- [1/3] Starting vLLM (same flags as posteval_1.5b_v6) ---"
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --dtype auto --seed 42 &
_SERVER_PID=$!
python -m serving.healthcheck --url "${SERVER_URL}" --retries 120 --interval 5
echo "Server healthy."

echo ""
echo "--- [2/3] Running quality evaluation ---"
python -m eval.run_quality \
    --config "${EVAL_CONFIG}" \
    --model "${MODEL}" \
    --role  "${EVAL_ROLE}"

if [[ "${SKIP_PERF_BENCH}" == "1" ]]; then
    echo "Skipping throughput/online benchmarks (SKIP_PERF_BENCH=1)"
else
    echo ""
    echo "--- [3/3] Running throughput + online benchmarks (${NUM_REPS} reps) ---"
    for REP in $(seq 1 ${NUM_REPS}); do
        python -m bench.run_throughput  --config configs/benchmark.yaml --model "${MODEL}" --role "${EVAL_ROLE}"
        python -m bench.run_online_load --config configs/benchmark.yaml --model "${MODEL}" --role "${EVAL_ROLE}"
    done
fi

echo ""
echo "========================================="
echo " Posteval baseline 1.5B (match v6 protocol) done."
echo " Model:       ${MODEL}"
echo " Quality dir: results/quality/quality-${EVAL_ROLE}-*/"
echo "========================================="
