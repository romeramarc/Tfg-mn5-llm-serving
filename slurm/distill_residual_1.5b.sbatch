#!/bin/bash
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SLURM job â€” Distillation Stage 2 (Residual) for Qwen2.5-1.5B
#   1) Merge latest stage-1 adapter into a temporary student model
#   2) Build residual KD dataset (teacher reliable, student fails)
#   3) Train stage-2 LoRA on residual-mixed dataset
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#SBATCH --job-name=distill-res1.5b
#SBATCH --output=logs/distill-res1.5b-%j.out
#SBATCH --error=logs/distill-res1.5b-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00

set -euo pipefail

cd /gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving
source env/setup_env.sh
mkdir -p logs results/distill

DISTILL_CONFIG="${DISTILL_CONFIG:-configs/distill_1p5b_focus.yaml}"

INIT_PATTERN=$(python - <<EOF
from utils.config_loader import load_yaml
cfg = load_yaml("${DISTILL_CONFIG}")
print(cfg.get("residual_distillation", {}).get("stage2_training", {}).get("init_adapter_path", "results/distill/sft-qwen2.5-1.5b-*/final_adapter"))
EOF
)

ADAPTER=$(python - <<EOF
import glob
pattern = "${INIT_PATTERN}"
matches = sorted(glob.glob(pattern), reverse=True)
print(matches[0] if matches else "")
EOF
)

if [[ -z "${ADAPTER}" ]]; then
    echo "ERROR: Could not resolve stage1 adapter with pattern: ${INIT_PATTERN}"
    exit 1
fi

MERGED_STUDENT=$(python - <<EOF
from utils.config_loader import load_yaml
cfg = load_yaml("${DISTILL_CONFIG}")
print(cfg.get("residual_distillation", {}).get("student_model", "results/distill/merged-qwen2.5-1.5b-stage1"))
EOF
)

RESIDUAL_DATASET=$(python - <<EOF
from utils.config_loader import load_yaml
cfg = load_yaml("${DISTILL_CONFIG}")
print(cfg.get("residual_distillation", {}).get("stage2_training", {}).get("dataset_path", cfg.get("residual_distillation", {}).get("output_file", "results/distill/teacher_outputs_residual_stage2.jsonl")))
EOF
)

RESIDUAL_URL=$(python - <<EOF
from utils.config_loader import load_yaml
cfg = load_yaml("${DISTILL_CONFIG}")
print(cfg.get("residual_distillation", {}).get("student_base_url", "http://localhost:8001"))
EOF
)

RESIDUAL_PORT=$(python - <<EOF
from urllib.parse import urlparse
url = urlparse("${RESIDUAL_URL}")
print(url.port or 8001)
EOF
)

_SERVER_PID=""
cleanup() {
    if [[ -n "${_SERVER_PID}" ]]; then
        kill "${_SERVER_PID}" 2>/dev/null || true
        wait "${_SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "========================================="
echo " Distillation Stage 2: Residual 1.5B"
echo " Config:          ${DISTILL_CONFIG}"
echo " Stage1 adapter:  ${ADAPTER}"
echo " Merged student:  ${MERGED_STUDENT}"
echo " Residual dataset:${RESIDUAL_DATASET}"
echo " Student URL:     ${RESIDUAL_URL}"
echo "========================================="

# Validate stage1 dataset first
python -m distill.preflight --step 2 --config "${DISTILL_CONFIG}"

# Merge stage1 adapter into a local model for residual scoring server
export ADAPTER MERGED_STUDENT
python - <<'PY'
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os, shutil

base_id = 'Qwen/Qwen2.5-1.5B-Instruct'
adapter = os.environ['ADAPTER']
merged_out = os.environ['MERGED_STUDENT']

base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, adapter)
merged = model.merge_and_unload()
shutil.rmtree(merged_out, ignore_errors=True)
merged.save_pretrained(merged_out)
AutoTokenizer.from_pretrained(base_id).save_pretrained(merged_out)
print(f"Merged model saved to: {merged_out}")
PY

# Start temporary student server for residual detection
python -m vllm.entrypoints.openai.api_server \
    --model "${MERGED_STUDENT}" \
    --host 0.0.0.0 \
    --port "${RESIDUAL_PORT}" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --dtype auto \
    --seed 42 &
_SERVER_PID=$!

python -m serving.healthcheck --url "${RESIDUAL_URL}" --retries 120 --interval 5

# Build stage2 residual dataset
python -m distill.build_residual_dataset --config "${DISTILL_CONFIG}"

if [[ ! -s "${RESIDUAL_DATASET}" ]]; then
    echo "ERROR: Residual dataset missing or empty: ${RESIDUAL_DATASET}"
    exit 1
fi

echo "Residual dataset lines: $(wc -l < "${RESIDUAL_DATASET}")"

# Validate stage2 dataset path from config
python -m distill.preflight --step 2 --stage2-residual --config "${DISTILL_CONFIG}"

# Stop temporary student server before training to free GPU memory.
if [[ -n "${_SERVER_PID}" ]]; then
    kill "${_SERVER_PID}" 2>/dev/null || true
    wait "${_SERVER_PID}" 2>/dev/null || true
    _SERVER_PID=""
fi

# Train stage2 residual LoRA
python -m distill.train_student_sft \
    --config "${DISTILL_CONFIG}" \
    --student "Qwen/Qwen2.5-1.5B-Instruct" \
    --stage2-residual

echo ""
echo "========================================="
echo " Residual stage2 training complete"
echo " Adapter: results/distill/sft-qwen2.5-1.5b-residual-stage2-*/final_adapter"
echo "========================================="
