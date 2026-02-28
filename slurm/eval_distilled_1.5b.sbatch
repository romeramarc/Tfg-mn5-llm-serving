#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — Post-distillation evaluation: distilled 1.5B student
#   Serves the LoRA-merged 1.5B model with vLLM and runs the same
#   3 benchmarks (GSM8K, MATH-500, ARC-Challenge) as Phase 1.
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=eval-dist1.5b
#SBATCH --output=logs/eval-distilled-1.5b-%j.out
#SBATCH --error=logs/eval-distilled-1.5b-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=03:00:00

set -euo pipefail

# ── Environment ───────────────────────────────────────────
cd /gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving
source env/setup_env.sh
mkdir -p logs

# ── Locate the latest 1.5B adapter ──────────────────────────
ADAPTER_DIR=$(ls -dt results/distill/sft-qwen2.5-1.5b-*/final_adapter 2>/dev/null | head -1)
if [[ -z "${ADAPTER_DIR}" ]]; then
    echo "ERROR: No 1.5B adapter found under results/distill/sft-qwen2.5-1.5b-*"
    exit 1
fi
echo "Using adapter: ${ADAPTER_DIR}"

BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
ROLE="distilled_student_small"
SERVER_URL="http://localhost:8000"

echo "========================================="
echo " Post-distill Eval: ${ROLE}"
echo " Base:    ${BASE_MODEL}"
echo " Adapter: ${ADAPTER_DIR}"
echo " Job ID:  ${SLURM_JOB_ID}"
echo " Node:    $(hostname)"
echo " Date:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── [1/4] Merge LoRA adapter into base model ────────────────
echo ""
echo "--- [1/4] Merging LoRA adapter ---"
MERGED_DIR="results/distill/merged-qwen2.5-1.5b"
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, shutil, os

base = AutoModelForCausalLM.from_pretrained('${BASE_MODEL}', torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, '${ADAPTER_DIR}')
merged = model.merge_and_unload()

out = '${MERGED_DIR}'
if os.path.exists(out):
    shutil.rmtree(out)
merged.save_pretrained(out)
AutoTokenizer.from_pretrained('${BASE_MODEL}').save_pretrained(out)
print(f'Merged model saved to {out}')
"
echo "Merge complete."

# ── [2/4] Start vLLM server with merged model ───────────────
echo ""
echo "--- [2/4] Starting server ---"
python -m vllm.entrypoints.openai.api_server \
    --model "${MERGED_DIR}" \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --dtype auto \
    --seed 42 &
SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

# ── [3/4] Wait for /health ──────────────────────────────────
echo ""
echo "--- [3/4] Waiting for /health ---"
python -m serving.healthcheck \
    --url "${SERVER_URL}" \
    --retries 120 --interval 5
echo "Server is healthy."

# ── [4/4] Quality evaluation ────────────────────────────────
echo ""
echo "--- [4/4] Quality evaluation (GSM8K + MATH-500 + ARC-Challenge) ---"
python -m eval.run_quality \
    --config configs/eval.yaml \
    --model  "${MERGED_DIR}" \
    --role   "${ROLE}"
echo "Quality evaluation done."

# ── Stop server ─────────────────────────────────────────────
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true
echo "Server stopped."

echo ""
echo "========================================="
echo " ${ROLE} post-distill eval complete."
echo " results/quality/quality-${ROLE}-*/quality_summary.json"
echo "========================================="
