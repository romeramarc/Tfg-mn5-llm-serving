#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — Distillation Step 2: LoRA SFT for Qwen2.5-1.5B
#   Trains the 1.5B student on teacher_outputs.jsonl with LoRA.
#   Expects Step 1 (distill_generate.sbatch) to have completed.
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=distill-1.5b
#SBATCH --output=logs/distill-1.5b-%j.out
#SBATCH --error=logs/distill-1.5b-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00

set -euo pipefail

# ── Environment ───────────────────────────────────────────
cd /gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving
source env/setup_env.sh
mkdir -p logs

DISTILL_CONFIG="${DISTILL_CONFIG:-configs/distill.yaml}"
SFT_MAX_SAMPLES="${SFT_MAX_SAMPLES:-}"
MIN_VALID_SAMPLES="${MIN_VALID_SAMPLES:-}"

STUDENT="Qwen/Qwen2.5-1.5B-Instruct"
DATASET=$(python - <<EOF
from utils.config_loader import load_yaml

cfg = load_yaml("${DISTILL_CONFIG}")
print(cfg.get("training", {}).get("dataset_path", "results/distill/teacher_outputs.jsonl"))
EOF
)

echo "========================================="
echo " Distillation Step 2: LoRA SFT"
echo " Student: ${STUDENT}"
echo " Dataset: ${DATASET}"
echo " Job ID:  ${SLURM_JOB_ID}"
echo " Node:    $(hostname)"
echo " Date:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " Config:  ${DISTILL_CONFIG}"
if [[ -n "${SFT_MAX_SAMPLES}" ]]; then
    echo " SFT max samples override: ${SFT_MAX_SAMPLES}"
fi
echo "========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Verify dataset exists
if [[ ! -f "${DATASET}" ]]; then
    echo "ERROR: ${DATASET} not found. Run distill_generate.sbatch first."
    exit 1
fi
echo "Dataset lines: $(wc -l < "${DATASET}")"
echo "1.5B training recipe loaded from: ${DISTILL_CONFIG}"

# Optional safety check for the configured training dataset
PREFLIGHT_STEP2_ARGS=(--step 2 --config "${DISTILL_CONFIG}")
if [[ -n "${MIN_VALID_SAMPLES}" ]]; then
    PREFLIGHT_STEP2_ARGS+=(--min-valid-samples "${MIN_VALID_SAMPLES}")
fi
python -m distill.preflight "${PREFLIGHT_STEP2_ARGS[@]}"

# ── Train ───────────────────────────────────────────────────
TRAIN_ARGS=(--config "${DISTILL_CONFIG}" --student "${STUDENT}")
if [[ -n "${SFT_MAX_SAMPLES}" ]]; then
    TRAIN_ARGS+=(--max-samples "${SFT_MAX_SAMPLES}")
fi
python -m distill.train_student_sft "${TRAIN_ARGS[@]}"

# ── Final GPU/VRAM snapshot ──────────────────────────────────
echo ""
echo "--- Final GPU state ---"
nvidia-smi
echo "pip freeze:"
pip freeze 2>/dev/null | head -60
echo "..."

echo ""
echo "========================================="
echo " 1.5B SFT training complete."
echo " Adapter: results/distill/sft-qwen2.5-1.5b-*/final_adapter/"
echo "========================================="
