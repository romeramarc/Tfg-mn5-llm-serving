#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — Exp. 6 · Step 2: Full-parameter FT for Qwen2.5-1.5B
#   Full fine-tune (no LoRA) on the RFT-expanded teacher outputs.
#   Expects Step 1 (distill_generate_v6.sbatch) to have completed.
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=distill-1.5b-v6
#SBATCH --output=logs/distill-1.5b-v6-%j.out
#SBATCH --error=logs/distill-1.5b-v6-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00

set -euo pipefail

# ── Environment ───────────────────────────────────────────
cd /gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving
source env/setup_env.sh
mkdir -p logs results/distill

DISTILL_CONFIG="${DISTILL_CONFIG:-configs/distill_v6.yaml}"
SFT_MAX_SAMPLES="${SFT_MAX_SAMPLES:-}"
MIN_VALID_SAMPLES="${MIN_VALID_SAMPLES:-}"

STUDENT="Qwen/Qwen2.5-1.5B-Instruct"
DATASET=$(python - <<EOF
from utils.config_loader import load_yaml
cfg = load_yaml("${DISTILL_CONFIG}")
print(cfg.get("training", {}).get("dataset_path", "results/distill/teacher_outputs_v6.jsonl"))
EOF
)

echo "========================================="
echo " Exp. 6 · Step 2: Full-parameter FT (1.5B)"
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

# ── Sanity checks ─────────────────────────────────────────
if [[ ! -f "${DATASET}" ]]; then
    echo "ERROR: ${DATASET} not found. Run distill_generate_v6.sbatch first."
    exit 1
fi
echo "Dataset lines: $(wc -l < "${DATASET}")"

PREFLIGHT_STEP2_ARGS=(--step 2 --config "${DISTILL_CONFIG}")
if [[ -n "${MIN_VALID_SAMPLES}" ]]; then
    PREFLIGHT_STEP2_ARGS+=(--min-valid-samples "${MIN_VALID_SAMPLES}")
fi
python -m distill.preflight "${PREFLIGHT_STEP2_ARGS[@]}"

# ── Train (full-parameter FT, mode=full in distill_v6.yaml) ─
TRAIN_ARGS=(--config "${DISTILL_CONFIG}" --student "${STUDENT}")
if [[ -n "${SFT_MAX_SAMPLES}" ]]; then
    TRAIN_ARGS+=(--max-samples "${SFT_MAX_SAMPLES}")
fi
python -m distill.train_student_sft "${TRAIN_ARGS[@]}"

echo ""
echo "--- Final GPU state ---"
nvidia-smi

echo ""
echo "========================================="
echo " Exp. 6 · 1.5B full-FT training complete."
echo " Model dir: results/distill/sft-full-qwen2.5-1.5b-v6-*/final_model/"
echo "========================================="
