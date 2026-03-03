#!/bin/bash
# ─────────────────────────────────────────────────────────────
# FASE 3: POST-DISTILLATION EVALUATION
# ─────────────────────────────────────────────────────────────
# 3.1 Merge & eval distilled 7B:   Quality + Throughput×3 + Online×3
# 3.2 Merge & eval distilled 1.5B: Quality + Throughput×3 + Online×3
#
# Chain:
#   JOB1=$(sbatch --parsable slurm/fase1_baselines.sbatch)
#   JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/fase2_distill.sbatch)
#   sbatch --dependency=afterok:$JOB2 slurm/fase3_posteval.sbatch
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=fase3-posteval
#SBATCH --output=logs/fase3-posteval-%j.out
#SBATCH --error=logs/fase3-posteval-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --exclusive
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# ── Environment ────────────────────────────────────────────
cd /gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving
source env/setup_env.sh
mkdir -p logs results/quality results/throughput results/online results/distill

SERVER_URL="http://localhost:8000"
_SERVER_PID=""
NUM_REPS=3

# ── Header ─────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════"
echo "  FASE 3: POST-DISTILLATION EVALUATION"
echo "══════════════════════════════════════════════════════════"
echo "Date:          $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID:  ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "Git commit:    $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "══════════════════════════════════════════════════════════"

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
    echo "--- FASE 3 finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
}
trap cleanup EXIT

# ── Helpers ────────────────────────────────────────────────
start_server_custom() {
    local MODEL_PATH=$1
    echo "Starting server for ${MODEL_PATH}..."
    python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --host 0.0.0.0 --port 8000 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.90 \
        --dtype auto --seed 42 &
    _SERVER_PID=$!
    echo "Server PID: ${_SERVER_PID}"
    python -m serving.healthcheck \
        --url "${SERVER_URL}" \
        --retries 120 --interval 5
    echo "Server healthy."
}

stop_server() {
    if [[ -n "${_SERVER_PID}" ]]; then
        kill "${_SERVER_PID}" 2>/dev/null || true
        wait "${_SERVER_PID}" 2>/dev/null || true
        _SERVER_PID=""
        sleep 5
    fi
}

run_quality() {
    local MODEL=$1
    local ROLE=$2
    echo "Quality eval for ${ROLE}..."
    python -m eval.run_quality \
        --config configs/eval.yaml \
        --model  "${MODEL}" \
        --role   "${ROLE}"
    echo "Quality eval done for ${ROLE}."
}

run_efficiency() {
    local MODEL=$1
    local ROLE=$2
    for REP in $(seq 1 ${NUM_REPS}); do
        echo "── Efficiency rep ${REP}/${NUM_REPS} for ${ROLE} ──"
        echo "[${REP}a] Throughput..."
        python -m bench.run_throughput \
            --config configs/benchmark.yaml \
            --model "${MODEL}" \
            --role  "${ROLE}"
        echo "[${REP}b] Online load..."
        python -m bench.run_online_load \
            --config configs/benchmark.yaml \
            --model "${MODEL}" \
            --role  "${ROLE}"
        echo "── Rep ${REP} done ──"
    done
}

# ── Merge helper ───────────────────────────────────────────
merge_adapter() {
    local BASE_MODEL=$1
    local ADAPTER_PATH=$2
    local OUTPUT_PATH=$3
    echo "Merging LoRA adapter: ${ADAPTER_PATH} → ${OUTPUT_PATH}"
    python3 << EOF
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, shutil, os

base = AutoModelForCausalLM.from_pretrained('${BASE_MODEL}', torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, '${ADAPTER_PATH}')
merged = model.merge_and_unload()

out = '${OUTPUT_PATH}'
if os.path.exists(out): shutil.rmtree(out)
merged.save_pretrained(out)
AutoTokenizer.from_pretrained('${BASE_MODEL}').save_pretrained(out)
print(f'Merged model saved to {out}')
EOF
    echo "Merge complete."
}

# ═══════════════════════════════════════════════════════════
# 3.1 DISTILLED 7B: Merge + Quality + Efficiency ×3
# ═══════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════"
echo " 3.1 DISTILLED 7B: Quality + Efficiency"
echo "════════════════════════════════════════"

ADAPTER_7B=$(ls -dt results/distill/sft-qwen2.5-7b-*/final_adapter 2>/dev/null | head -1)
if [[ -z "${ADAPTER_7B}" ]]; then
    echo "ERROR: No 7B adapter found in results/distill/"
    exit 1
fi
echo "Using 7B adapter: ${ADAPTER_7B}"

MERGED_7B="results/distill/merged-qwen2.5-7b"
merge_adapter "Qwen/Qwen2.5-7B-Instruct" "${ADAPTER_7B}" "${MERGED_7B}"

start_server_custom "${MERGED_7B}"

run_quality "${MERGED_7B}" "distilled_student_mid"
run_efficiency "${MERGED_7B}" "distilled_student_mid"

stop_server
echo "✅ Distilled 7B evaluation complete."

# ═══════════════════════════════════════════════════════════
# 3.2 DISTILLED 1.5B: Merge + Quality + Efficiency ×3
# ═══════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════"
echo " 3.2 DISTILLED 1.5B: Quality + Efficiency"
echo "════════════════════════════════════════"

ADAPTER_15B=$(ls -dt results/distill/sft-qwen2.5-1.5b-*/final_adapter 2>/dev/null | head -1)
if [[ -z "${ADAPTER_15B}" ]]; then
    echo "ERROR: No 1.5B adapter found in results/distill/"
    exit 1
fi
echo "Using 1.5B adapter: ${ADAPTER_15B}"

MERGED_15B="results/distill/merged-qwen2.5-1.5b"
merge_adapter "Qwen/Qwen2.5-1.5B-Instruct" "${ADAPTER_15B}" "${MERGED_15B}"

start_server_custom "${MERGED_15B}"

run_quality "${MERGED_15B}" "distilled_student_small"
run_efficiency "${MERGED_15B}" "distilled_student_small"

stop_server
echo "✅ Distilled 1.5B evaluation complete."

# ═══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  ✅✅✅ COMPLETE PIPELINE FINISHED ✅✅✅"
echo "══════════════════════════════════════════════════════════"
echo ""
echo "FASE 3 — Post-distillation results:"
echo "  results/quality/quality-distilled_student_mid-*/"
echo "  results/quality/quality-distilled_student_small-*/"
echo "  results/throughput/throughput-distilled_student_mid-*/   (×3)"
echo "  results/throughput/throughput-distilled_student_small-*/ (×3)"
echo "  results/online/online-distilled_student_mid-*/           (×3)"
echo "  results/online/online-distilled_student_small-*/         (×3)"
echo ""
echo "ALL PHASES DONE:"
echo "  Total: 5 quality + 15 throughput + 15 online = 35 benchmark runs"
echo "══════════════════════════════════════════════════════════"
