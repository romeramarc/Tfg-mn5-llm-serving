#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — COMPLETE EXPERIMENTAL PIPELINE
# Target cluster: MareNostrum 5   (BSC)
#
# Executes the FULL rigorous experimental pipeline:
#
#   FASE 1: Pre-distillation baselines
#   ───────────────────────────────────
#   1.1 Teacher (14B):
#       - Quality: GSM8K test + MATH-500 test + Routing bench
#       - Efficiency: Throughput ×3 reps + Online ×3 reps
#   1.2 Student mid (7B):
#       - Quality: GSM8K test + MATH-500 test
#       - Efficiency: Throughput ×3 reps + Online ×3 reps
#   1.3 Student small (1.5B):
#       - Quality: GSM8K test + MATH-500 test
#       - Efficiency: Throughput ×3 reps + Online ×3 reps
#
#   FASE 2: Knowledge Distillation
#   ──────────────────────────────
#   2.1 Generate teacher outputs (TRAIN splits — no leakage)
#   2.2 Train 7B student (LoRA + 15% validation + early stopping)
#   2.3 Train 1.5B student (LoRA + 15% validation + early stopping)
#
#   FASE 3: Post-distillation evaluation
#   ─────────────────────────────────────
#   3.1 Distilled 7B:
#       - Quality: GSM8K test + MATH-500 test
#       - Efficiency: Throughput ×3 reps + Online ×3 reps
#   3.2 Distilled 1.5B:
#       - Quality: GSM8K test + MATH-500 test
#       - Efficiency: Throughput ×3 reps + Online ×3 reps
#
# Total time: ~72-96 hours (3-4 days)
#
# Usage:
#   sbatch slurm/run_complete_experiment.sbatch
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=complete-exp
#SBATCH --output=logs/complete-exp-%j.out
#SBATCH --error=logs/complete-exp-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=96:00:00
#SBATCH --exclusive
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# ── Environment ────────────────────────────────────────────
cd /gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving
source env/setup_env.sh
mkdir -p logs results/quality results/distill results/throughput results/online results/routing

SERVER_URL="http://localhost:8000"
_SERVER_PID=""
NUM_REPS=3

# ── Header ─────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════"
echo "  COMPLETE EXPERIMENTAL PIPELINE"
echo "  Teacher + Students (baseline) → Distillation → Post-eval"
echo "══════════════════════════════════════════════════════════"
echo "Date:          $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID:  ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "Git commit:    $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""
echo "pip freeze (first 60 packages):"
pip freeze 2>/dev/null | head -60
echo "..."
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
    echo "--- Pipeline finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
}
trap cleanup EXIT

# ── Helper: start server ───────────────────────────────────
start_server() {
    local ROLE=$1
    echo "Starting server for ${ROLE}..."
    python -m serving.start_server \
        --config configs/serving.yaml \
        --models configs/models.yaml \
        --role "${ROLE}" &
    _SERVER_PID=$!
    echo "Server PID: ${_SERVER_PID}"
    
    python -m serving.healthcheck \
        --url "${SERVER_URL}" \
        --retries 120 --interval 5
    echo "Server healthy."
}

# ── Helper: start server with custom model path ────────────
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

# ── Helper: stop server ────────────────────────────────────
stop_server() {
    if [[ -n "${_SERVER_PID}" ]]; then
        kill "${_SERVER_PID}" 2>/dev/null || true
        wait "${_SERVER_PID}" 2>/dev/null || true
        _SERVER_PID=""
        sleep 5
    fi
}

# ── Helper: run quality eval ───────────────────────────────
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

# ── Helper: run efficiency benchmarks ×3 reps ──────────────
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

# ═══════════════════════════════════════════════════════════
# FASE 1: PRE-DISTILLATION BASELINES
# ═══════════════════════════════════════════════════════════
echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║  FASE 1: PRE-DISTILLATION BASELINES                   ║"
echo "╚═══════════════════════════════════════════════════════╝"

# ────────────────────────────────────────────────────────────
# 1.1 TEACHER (14B): Quality + Routing + Efficiency ×3
# ────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo " 1.1 TEACHER: Quality + Routing + Efficiency"
echo "════════════════════════════════════════"

start_server "teacher"

# Quality eval (GSM8K + MATH-500)
run_quality "Qwen/Qwen2.5-14B-Instruct" "teacher"

# Routing eval (if module exists)
echo "Routing eval for teacher..."
if python -c "import routing.run_routing" 2>/dev/null; then
    python -m routing.run_routing --config configs/routing.yaml
    echo "Routing eval done."
else
    # Fallback: run quality eval with routing label
    python -m eval.run_quality \
        --config configs/eval.yaml \
        --model  "Qwen/Qwen2.5-14B-Instruct" \
        --role   "teacher_routing"
    echo "Routing fallback (quality eval) done."
fi

# Efficiency ×3 reps
run_efficiency "Qwen/Qwen2.5-14B-Instruct" "teacher"

stop_server
echo "✅ Teacher baseline complete."

# ────────────────────────────────────────────────────────────
# 1.2 STUDENT MID (7B): Quality + Efficiency ×3
# ────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo " 1.2 STUDENT MID (7B): Quality + Efficiency"
echo "════════════════════════════════════════"

start_server "student_mid"

run_quality "Qwen/Qwen2.5-7B-Instruct" "student_mid"
run_efficiency "Qwen/Qwen2.5-7B-Instruct" "student_mid"

stop_server
echo "✅ Student mid baseline complete."

# ────────────────────────────────────────────────────────────
# 1.3 STUDENT SMALL (1.5B): Quality + Efficiency ×3
# ────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo " 1.3 STUDENT SMALL (1.5B): Quality + Efficiency"
echo "════════════════════════════════════════"

start_server "student_small"

run_quality "Qwen/Qwen2.5-1.5B-Instruct" "student_small"
run_efficiency "Qwen/Qwen2.5-1.5B-Instruct" "student_small"

stop_server
echo "✅ Student small baseline complete."

echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║  FASE 1 COMPLETE                                      ║"
echo "╚═══════════════════════════════════════════════════════╝"

# ═══════════════════════════════════════════════════════════
# FASE 2: KNOWLEDGE DISTILLATION
# ═══════════════════════════════════════════════════════════
echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║  FASE 2: KNOWLEDGE DISTILLATION                       ║"
echo "╚═══════════════════════════════════════════════════════╝"

# ────────────────────────────────────────────────────────────
# 2.1 Generate teacher outputs (TRAIN splits)
# ────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo " 2.1 Generate teacher outputs (TRAIN splits)"
echo "════════════════════════════════════════"

start_server "teacher"

python -m distill.generate_teacher_outputs \
    --config configs/distill.yaml
echo "Teacher outputs generated (train splits, no leakage)."

stop_server

# ────────────────────────────────────────────────────────────
# 2.2 Train 7B student (LoRA SFT + validation)
# ────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo " 2.2 Train 7B student (LoRA + validation + early stop)"
echo "════════════════════════════════════════"

python -m distill.train_student_sft \
    --config configs/distill.yaml \
    --student "Qwen/Qwen2.5-7B-Instruct"
echo "7B SFT complete."

# ────────────────────────────────────────────────────────────
# 2.3 Train 1.5B student (LoRA SFT + validation)
# ────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo " 2.3 Train 1.5B student (LoRA + validation + early stop)"
echo "════════════════════════════════════════"

python -m distill.train_student_sft \
    --config configs/distill.yaml \
    --student "Qwen/Qwen2.5-1.5B-Instruct"
echo "1.5B SFT complete."

echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║  FASE 2 COMPLETE                                      ║"
echo "╚═══════════════════════════════════════════════════════╝"

# ═══════════════════════════════════════════════════════════
# FASE 3: POST-DISTILLATION EVALUATION
# ═══════════════════════════════════════════════════════════
echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║  FASE 3: POST-DISTILLATION EVALUATION                 ║"
echo "╚═══════════════════════════════════════════════════════╝"

# ────────────────────────────────────────────────────────────
# 3.1 Distilled 7B: Quality + Efficiency ×3
# ────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo " 3.1 DISTILLED 7B: Quality + Efficiency"
echo "════════════════════════════════════════"

# Find and merge adapter
ADAPTER_7B=$(ls -dt results/distill/sft-qwen2.5-7b-*/final_adapter 2>/dev/null | head -1)
if [[ -z "${ADAPTER_7B}" ]]; then
    echo "ERROR: No 7B adapter found."
    exit 1
fi
echo "Using 7B adapter: ${ADAPTER_7B}"

MERGED_7B="results/distill/merged-qwen2.5-7b"
echo "Merging LoRA adapter..."
python3 << EOF
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, shutil, os

base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, '${ADAPTER_7B}')
merged = model.merge_and_unload()

out = '${MERGED_7B}'
if os.path.exists(out): shutil.rmtree(out)
merged.save_pretrained(out)
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct').save_pretrained(out)
print(f'Merged 7B model saved to {out}')
EOF
echo "Merge complete."

start_server_custom "${MERGED_7B}"

run_quality "${MERGED_7B}" "distilled_student_mid"
run_efficiency "${MERGED_7B}" "distilled_student_mid"

stop_server
echo "✅ Distilled 7B evaluation complete."

# ────────────────────────────────────────────────────────────
# 3.2 Distilled 1.5B: Quality + Efficiency ×3
# ────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo " 3.2 DISTILLED 1.5B: Quality + Efficiency"
echo "════════════════════════════════════════"

# Find and merge adapter
ADAPTER_15B=$(ls -dt results/distill/sft-qwen2.5-1.5b-*/final_adapter 2>/dev/null | head -1)
if [[ -z "${ADAPTER_15B}" ]]; then
    echo "ERROR: No 1.5B adapter found."
    exit 1
fi
echo "Using 1.5B adapter: ${ADAPTER_15B}"

MERGED_15B="results/distill/merged-qwen2.5-1.5b"
echo "Merging LoRA adapter..."
python3 << EOF
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, shutil, os

base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, '${ADAPTER_15B}')
merged = model.merge_and_unload()

out = '${MERGED_15B}'
if os.path.exists(out): shutil.rmtree(out)
merged.save_pretrained(out)
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct').save_pretrained(out)
print(f'Merged 1.5B model saved to {out}')
EOF
echo "Merge complete."

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
echo "  ✅✅✅ COMPLETE EXPERIMENTAL PIPELINE FINISHED ✅✅✅"
echo "══════════════════════════════════════════════════════════"
echo ""
echo "Results summary:"
echo ""
echo "FASE 1 — Pre-distillation baselines:"
echo "  results/quality/quality-teacher-*/"
echo "  results/quality/quality-teacher_routing-*/"
echo "  results/quality/quality-student_mid-*/"
echo "  results/quality/quality-student_small-*/"
echo "  results/throughput/throughput-teacher-*/     (×3)"
echo "  results/throughput/throughput-student_mid-*/ (×3)"
echo "  results/throughput/throughput-student_small-*/ (×3)"
echo "  results/online/online-teacher-*/             (×3)"
echo "  results/online/online-student_mid-*/         (×3)"
echo "  results/online/online-student_small-*/       (×3)"
echo ""
echo "FASE 2 — Distillation:"
echo "  results/distill/teacher_outputs.jsonl"
echo "  results/distill/sft-qwen2.5-7b-*/final_adapter/"
echo "  results/distill/sft-qwen2.5-1.5b-*/final_adapter/"
echo ""
echo "FASE 3 — Post-distillation:"
echo "  results/quality/quality-distilled_student_mid-*/"
echo "  results/quality/quality-distilled_student_small-*/"
echo "  results/throughput/throughput-distilled_student_mid-*/   (×3)"
echo "  results/throughput/throughput-distilled_student_small-*/ (×3)"
echo "  results/online/online-distilled_student_mid-*/           (×3)"
echo "  results/online/online-distilled_student_small-*/         (×3)"
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Total benchmarks: 5 models × (quality + 3×throughput + 3×online)"
echo "  = 5 quality evals + 15 throughput runs + 15 online runs"
echo "  = 35 benchmark runs total"
echo "══════════════════════════════════════════════════════════"
