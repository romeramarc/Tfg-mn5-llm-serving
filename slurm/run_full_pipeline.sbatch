#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — Complete experimental pipeline (re-run)
# Target cluster: MareNostrum 5   (BSC)
#
# Executes the full rigorous experimental pipeline:
#
#   Phase A — Quality baselines
#   ──────────────────────────
#   A1. Teacher (14B) quality eval    → GSM8K test + MATH-500 test
#   A2. Student mid (7B) quality eval → GSM8K test + MATH-500 test
#   A3. Student small (1.5B) quality  → GSM8K test + MATH-500 test
#
#   Phase B — Knowledge Distillation
#   ────────────────────────────────
#   B1. Generate teacher outputs (TRAIN splits — no leakage)
#   B2. Train 7B student (LoRA SFT + validation + early stopping)
#   B3. Train 1.5B student (LoRA SFT + validation + early stopping)
#
#   Phase C — Post-distillation quality evaluation
#   ───────────────────────────────────────────────
#   C1. Eval distilled 7B   → GSM8K test + MATH-500 test
#   C2. Eval distilled 1.5B → GSM8K test + MATH-500 test
#
# NOTE: Efficiency benchmarks (throughput + online × 3 reps)
# are in a separate script: slurm/bench_repetitions.sbatch
#
# Usage:
#   sbatch slurm/run_full_pipeline.sbatch
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=full-pipeline
#SBATCH --output=logs/full-pipeline-%j.out
#SBATCH --error=logs/full-pipeline-%j.err
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
mkdir -p logs results/quality results/distill

SERVER_URL="http://localhost:8000"
_SERVER_PID=""

# ── Header ─────────────────────────────────────────────────
echo "══════════════════════════════════════════"
echo "  COMPLETE EXPERIMENTAL PIPELINE"
echo "══════════════════════════════════════════"
echo "Date:          $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID:  ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "Git commit:    $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""
echo "pip freeze (first 60 packages):"
pip freeze 2>/dev/null | head -60
echo "..."
echo "══════════════════════════════════════════"

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

# ── Helper: start model server, eval quality, stop ─────────
eval_quality() {
    local ROLE=$1
    local MODEL=$2

    echo ""
    echo "────────────────────────────────────────"
    echo " Quality Eval: ${ROLE} (${MODEL})"
    echo "────────────────────────────────────────"

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

    python -m eval.run_quality \
        --config configs/eval.yaml \
        --model  "${MODEL}" \
        --role   "${ROLE}"
    echo "Quality eval done for ${ROLE}."

    kill "${_SERVER_PID}" 2>/dev/null || true
    wait "${_SERVER_PID}" 2>/dev/null || true
    _SERVER_PID=""
    sleep 5
}

# ═══════════════════════════════════════════════════════════
# PHASE A — Quality baselines (teacher + students)
# ═══════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════"
echo " PHASE A: Quality baselines"
echo "═══════════════════════════════════════"

eval_quality "teacher"       "Qwen/Qwen2.5-14B-Instruct"
eval_quality "student_mid"   "Qwen/Qwen2.5-7B-Instruct"
eval_quality "student_small" "Qwen/Qwen2.5-1.5B-Instruct"

echo ""
echo "Phase A complete. Results in results/quality/"

# ═══════════════════════════════════════════════════════════
# PHASE B — Knowledge Distillation
# ═══════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════"
echo " PHASE B: Knowledge Distillation"
echo "═══════════════════════════════════════"

# B1 — Generate teacher outputs from TRAIN splits
echo ""
echo "────────────────────────────────────────"
echo " B1: Generate teacher outputs (train splits)"
echo "────────────────────────────────────────"

python -m serving.start_server \
    --config configs/serving.yaml \
    --models configs/models.yaml \
    --role teacher &
_SERVER_PID=$!
echo "Teacher server PID: ${_SERVER_PID}"

python -m serving.healthcheck \
    --url "${SERVER_URL}" \
    --retries 120 --interval 5
echo "Server healthy."

python -m distill.generate_teacher_outputs \
    --config configs/distill.yaml
echo "Teacher outputs generated."

kill "${_SERVER_PID}" 2>/dev/null || true
wait "${_SERVER_PID}" 2>/dev/null || true
_SERVER_PID=""
sleep 5

# B2 — Train 7B student
echo ""
echo "────────────────────────────────────────"
echo " B2: Train 7B student (LoRA SFT)"
echo "────────────────────────────────────────"

python -m distill.train_student_sft \
    --config configs/distill.yaml \
    --student "Qwen/Qwen2.5-7B-Instruct"
echo "7B SFT complete."

# B3 — Train 1.5B student
echo ""
echo "────────────────────────────────────────"
echo " B3: Train 1.5B student (LoRA SFT)"
echo "────────────────────────────────────────"

python -m distill.train_student_sft \
    --config configs/distill.yaml \
    --student "Qwen/Qwen2.5-1.5B-Instruct"
echo "1.5B SFT complete."

echo ""
echo "Phase B complete. Adapters in results/distill/"

# ═══════════════════════════════════════════════════════════
# PHASE C — Post-distillation quality evaluation
# ═══════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════"
echo " PHASE C: Post-distillation evaluation"
echo "═══════════════════════════════════════"

# C1 — Eval distilled 7B
echo ""
echo "────────────────────────────────────────"
echo " C1: Eval distilled 7B"
echo "────────────────────────────────────────"

ADAPTER_7B=$(ls -dt results/distill/sft-qwen2.5-7b-*/final_adapter 2>/dev/null | head -1)
if [[ -z "${ADAPTER_7B}" ]]; then
    echo "ERROR: No 7B adapter found."
    exit 1
fi
echo "Using 7B adapter: ${ADAPTER_7B}"

MERGED_7B="results/distill/merged-qwen2.5-7b"
python -c "
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
"

python -m vllm.entrypoints.openai.api_server \
    --model "${MERGED_7B}" \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --dtype auto --seed 42 &
_SERVER_PID=$!

python -m serving.healthcheck --url "${SERVER_URL}" --retries 120 --interval 5
echo "Server healthy."

python -m eval.run_quality \
    --config configs/eval.yaml \
    --model  "${MERGED_7B}" \
    --role   "distilled_student_mid"
echo "Distilled 7B eval done."

kill "${_SERVER_PID}" 2>/dev/null || true
wait "${_SERVER_PID}" 2>/dev/null || true
_SERVER_PID=""
sleep 5

# C2 — Eval distilled 1.5B
echo ""
echo "────────────────────────────────────────"
echo " C2: Eval distilled 1.5B"
echo "────────────────────────────────────────"

ADAPTER_15B=$(ls -dt results/distill/sft-qwen2.5-1.5b-*/final_adapter 2>/dev/null | head -1)
if [[ -z "${ADAPTER_15B}" ]]; then
    echo "ERROR: No 1.5B adapter found."
    exit 1
fi
echo "Using 1.5B adapter: ${ADAPTER_15B}"

MERGED_15B="results/distill/merged-qwen2.5-1.5b"
python -c "
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
"

python -m vllm.entrypoints.openai.api_server \
    --model "${MERGED_15B}" \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --dtype auto --seed 42 &
_SERVER_PID=$!

python -m serving.healthcheck --url "${SERVER_URL}" --retries 120 --interval 5
echo "Server healthy."

python -m eval.run_quality \
    --config configs/eval.yaml \
    --model  "${MERGED_15B}" \
    --role   "distilled_student_small"
echo "Distilled 1.5B eval done."

kill "${_SERVER_PID}" 2>/dev/null || true
wait "${_SERVER_PID}" 2>/dev/null || true
_SERVER_PID=""

echo ""
echo "══════════════════════════════════════════"
echo "  COMPLETE PIPELINE FINISHED"
echo "══════════════════════════════════════════"
echo ""
echo "Quality results:  results/quality/"
echo "Distill outputs:  results/distill/"
echo ""
echo "Next: submit slurm/bench_repetitions.sbatch for"
echo "efficiency benchmarks with 3 repetitions."
echo "══════════════════════════════════════════"
