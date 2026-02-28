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
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=04:00:00

set -euo pipefail

# ── Environment ───────────────────────────────────────────
cd /gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving
source env/setup_env.sh
mkdir -p logs

STUDENT="Qwen/Qwen2.5-1.5B-Instruct"
DATASET="results/distill/teacher_outputs.jsonl"

echo "========================================="
echo " Distillation Step 2: LoRA SFT"
echo " Student: ${STUDENT}"
echo " Dataset: ${DATASET}"
echo " Job ID:  ${SLURM_JOB_ID}"
echo " Node:    $(hostname)"
echo " Date:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Verify dataset exists
if [[ ! -f "${DATASET}" ]]; then
    echo "ERROR: ${DATASET} not found. Run distill_generate.sbatch first."
    exit 1
fi
echo "Dataset lines: $(wc -l < "${DATASET}")"

# ── Train ───────────────────────────────────────────────────
python -m distill.train_student_sft \
    --config configs/distill.yaml \
    --student "${STUDENT}"

echo ""
echo "========================================="
echo " 1.5B SFT training complete."
echo " Adapter: results/distill/sft-qwen2.5-1.5b-*/final_adapter/"
echo "========================================="
