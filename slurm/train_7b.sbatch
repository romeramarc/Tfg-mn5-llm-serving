#!/bin/bash
#SBATCH --job-name=train-7b
#SBATCH --output=logs/train-7b-%j.out
#SBATCH --error=logs/train-7b-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=08:00:00

set -euo pipefail

cd /gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving
source env/setup_env.sh
unset HF_DATASETS_OFFLINE
unset HF_HUB_OFFLINE
mkdir -p logs results/distill

echo "TRAIN 7B: LoRA SFT"
date

if [[ ! -f "results/distill/teacher_outputs.jsonl" ]]; then
    echo "ERROR: teacher_outputs.jsonl not found"
    exit 1
fi

python -m distill.train_student_sft --config configs/distill.yaml --student "Qwen/Qwen2.5-7B-Instruct"
echo "✅ train-7b done"
