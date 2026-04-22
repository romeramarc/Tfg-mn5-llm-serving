#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — Exp. 6 · Step 1: Teacher generation (RFT mode)
#   Generates N samples per problem with the 14B teacher, rejects
#   incorrect ones, keeps up to K correct per problem (diverse).
#   Expected volume: ~15 k problems × n=8-10 samples → 120 k+
#   teacher calls. With batched vLLM on 1×H100, budget 12-20 h.
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=distill-gen-v6
#SBATCH --output=logs/distill-gen-v6-%j.out
#SBATCH --error=logs/distill-gen-v6-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00

set -euo pipefail

on_error() {
    local exit_code=$?
    echo ""
    echo "ERROR: distill_generate_v6.sbatch failed"
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

DISTILL_CONFIG="${DISTILL_CONFIG:-configs/distill_v6.yaml}"
TEACHER_MAX_SAMPLES="${TEACHER_MAX_SAMPLES:-}"
MIN_VALID_SAMPLES="${MIN_VALID_SAMPLES:-}"

DISTILL_OUTPUT=$(python - <<EOF
from utils.config_loader import load_yaml
cfg = load_yaml("${DISTILL_CONFIG}")
print(cfg.get("generation", {}).get("output_file", "results/distill/teacher_outputs_v6.jsonl"))
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
echo " Exp. 6 · Step 1: Teacher generation (RFT)"
echo " Model:  ${MODEL}"
echo " Job ID: ${SLURM_JOB_ID}"
echo " Node:   $(hostname)"
echo " Date:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " Config: ${DISTILL_CONFIG}"
echo " Output: ${DISTILL_OUTPUT}"
if [[ -n "${TEACHER_MAX_SAMPLES}" ]]; then
    echo " Teacher max samples override: ${TEACHER_MAX_SAMPLES}"
fi
echo "========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── [0/4] In-node preflight (offline-safe) ─────────────────
echo ""
echo "--- [0/4] Running in-node preflight (offline-safe) ---"
python -m distill.preflight --step 1 --config "${DISTILL_CONFIG}"
echo "In-node preflight OK."

# ── [1/4] Start vLLM teacher server ─────────────────────────
echo ""
echo "--- [1/4] Starting teacher server ---"
python -m serving.start_server \
    --config configs/serving.yaml \
    --models configs/models.yaml \
    --role   teacher &
SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

# ── [2/4] Wait for /health ──────────────────────────────────
echo ""
echo "--- [2/4] Waiting for /health (up to 10 min) ---"
python -m serving.healthcheck \
    --url "http://localhost:8000" \
    --retries 120 --interval 5
echo "Server is healthy."

# ── [3/4] Generate teacher outputs (RFT mode) ───────────────
echo ""
echo "--- [3/4] Generating teacher outputs (RFT keep-all-correct) ---"
GEN_ARGS=(--config "${DISTILL_CONFIG}")
if [[ -n "${TEACHER_MAX_SAMPLES}" ]]; then
    GEN_ARGS+=(--max-samples "${TEACHER_MAX_SAMPLES}")
fi
python -m distill.generate_teacher_outputs "${GEN_ARGS[@]}"
echo "Teacher generation done."

if [[ ! -s "${DISTILL_OUTPUT}" ]]; then
    echo "ERROR: ${DISTILL_OUTPUT} missing or empty"
    exit 1
fi

echo "Teacher dataset lines: $(wc -l < "${DISTILL_OUTPUT}")"

# ── [4/4] Validate generated dataset semantics ──────────────
echo ""
echo "--- [4/4] Validating generated JSONL for training ---"
PREFLIGHT_STEP2_ARGS=(--step 2 --config "${DISTILL_CONFIG}")
if [[ -n "${MIN_VALID_SAMPLES}" ]]; then
    PREFLIGHT_STEP2_ARGS+=(--min-valid-samples "${MIN_VALID_SAMPLES}")
fi
python -m distill.preflight "${PREFLIGHT_STEP2_ARGS[@]}"
echo "Generated JSONL validated."

# ── Stop server ─────────────────────────────────────────────
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true
echo "Server stopped."

echo ""
echo "--- Final GPU state ---"
nvidia-smi

echo ""
echo "========================================="
echo " Exp. 6 teacher outputs saved to:"
echo " ${DISTILL_OUTPUT}"
echo "========================================="
