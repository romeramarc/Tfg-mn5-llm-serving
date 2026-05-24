#!/usr/bin/env bash
# Fire-and-forget: queue Option B (capture → train → refine → sysE holdout).
# Does NOT cancel Option A jobs already running.
#
# Usage (glogin4):
#   bash slurm/submit_option_b_base.sh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving}"
cd "${PROJECT_DIR}"
mkdir -p logs

J=$(sbatch --parsable \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}" \
  slurm/chain_option_b_base.sbatch)

echo "Option B chain submitted: job ${J}"
echo "  tail -f logs/chain-optB-base-${J}.out"
echo "  squeue -u \$USER"
echo ""
echo "Skips capture/train automatically if ladder_base artifacts already exist on scratch."
