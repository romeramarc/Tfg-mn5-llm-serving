#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — Knowledge distillation pipeline
#   Step 1: Generate teacher outputs (requires running server)
#   Step 2: Train student SFT with LoRA
# Target cluster: MareNostrum 5   (BSC)
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=distill
#SBATCH --output=logs/distill-%j.out
#SBATCH --error=logs/distill-%j.err
#SBATCH --partition=acc
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --exclusive

set -euo pipefail

module purge
module load intel
module load impi
module load mkl
module load hdf5
module load python/3.12.1
unset PYTHONPATH   # prevent system torch from shadowing the venv

source "${HOME}/.venvs/tfg/bin/activate"
mkdir -p logs

echo "=== DISTILLATION PIPELINE ==="
echo "Date:          $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID:  ${SLURM_JOB_ID}"

# ── Step 1: Generate teacher outputs ────────────────────────
echo "[Step 1] Generating teacher outputs …"
python -m serving.healthcheck \
    --url "http://localhost:8000" \
    --retries 60 --interval 5

python -m distill.generate_teacher_outputs --config configs/distill.yaml

# ── Step 2: Train student SFT ──────────────────────────────
echo "[Step 2] Training student with LoRA SFT …"
python -m distill.train_student_sft --config configs/distill.yaml

echo "=== DISTILLATION PIPELINE DONE ==="
