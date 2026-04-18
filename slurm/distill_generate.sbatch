#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — Distillation Step 1: Generate teacher outputs
#   Starts vLLM teacher → collects completions from TRAIN splits
#   (GSM8K train + MATH train) → saves JSONL.
#   ~15 000 prompts × greedy decode ≈ 4-6 h on 1×H100.
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=distill-gen
#SBATCH --output=logs/distill-gen-%j.out
#SBATCH --error=logs/distill-gen-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00

set -euo pipefail

on_error() {
    local exit_code=$?
    echo ""
    echo "ERROR: distill_generate.sbatch failed"
    echo "  Exit code: ${exit_code}"
    echo "  Line:      ${BASH_LINENO[0]}"
    echo "  Command:   ${BASH_COMMAND}"
    echo ""
}
trap on_error ERR

# ── Environment ───────────────────────────────────────────
cd /gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving
source env/setup_env.sh
mkdir -p logs results/distill

DISTILL_CONFIG="${DISTILL_CONFIG:-configs/distill.yaml}"
DISTILL_OUTPUT=$(python - <<EOF
from utils.config_loader import load_yaml

cfg = load_yaml("${DISTILL_CONFIG}")
print(cfg.get("generation", {}).get("output_file", "results/distill/teacher_outputs.jsonl"))
EOF
)

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

MODEL="Qwen/Qwen2.5-14B-Instruct"

echo "========================================="
echo " Distillation Step 1: Teacher generation"
echo " Model:  ${MODEL}"
echo " Job ID: ${SLURM_JOB_ID}"
echo " Node:   $(hostname)"
echo " Date:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " Config: ${DISTILL_CONFIG}"
echo " Output: ${DISTILL_OUTPUT}"
echo "========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── [0/4] In-node preflight (offline-safe) ─────────────────
echo ""
echo "--- [0/4] Running in-node preflight (offline-safe) ---"
python -m distill.preflight --step 1 --config "${DISTILL_CONFIG}"
echo "In-node preflight OK."

# ── [1/3] Start vLLM teacher server ─────────────────────────
echo ""
echo "--- [1/4] Starting teacher server ---"
python -m serving.start_server \
    --config configs/serving.yaml \
    --models configs/models.yaml \
    --role   teacher &
SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

# ── [2/3] Wait for /health ──────────────────────────────────
echo ""
echo "--- [2/4] Waiting for /health (up to 10 min) ---"
python -m serving.healthcheck \
    --url "http://localhost:8000" \
    --retries 120 --interval 5
echo "Server is healthy."

# ── [3/3] Generate teacher outputs ──────────────────────────
echo ""
echo "--- [3/4] Generating teacher outputs ---"
python -m distill.generate_teacher_outputs \
    --config "${DISTILL_CONFIG}"
echo "Teacher generation done."

if [[ ! -s "${DISTILL_OUTPUT}" ]]; then
    echo "ERROR: ${DISTILL_OUTPUT} missing or empty"
    exit 1
fi

echo "Teacher dataset lines: $(wc -l < "${DISTILL_OUTPUT}")"

# ── [4/4] Validate generated dataset semantics ──────────────
echo ""
echo "--- [4/4] Validating generated JSONL for training ---"
python -m distill.preflight --step 2 --config "${DISTILL_CONFIG}"
echo "Generated JSONL validated."

# ── Stop server ─────────────────────────────────────────────
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true
echo "Server stopped."

# ── Final GPU/VRAM snapshot ──────────────────────────────────
echo ""
echo "--- Final GPU state ---"
nvidia-smi
echo "pip freeze:"
pip freeze 2>/dev/null | head -60
echo "..."

echo ""
echo "========================================="
echo " Teacher outputs saved to:"
echo " ${DISTILL_OUTPUT}"
echo "========================================="
