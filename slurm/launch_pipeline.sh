#!/bin/bash
# ─────────────────────────────────────────────────────────────
# LAUNCHER: Submit all 3 phases with SLURM dependency chain
# ─────────────────────────────────────────────────────────────
# Fase 1 (baselines, 48h) → Fase 2 (distill, 24h) → Fase 3 (post-eval, 48h)
#
# Usage:
#   bash slurm/launch_pipeline.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

cd "$(dirname "$0")/.."

echo "══════════════════════════════════════════════════════════"
echo "  Launching COMPLETE PIPELINE (3 chained jobs)"
echo "══════════════════════════════════════════════════════════"

# Submit Fase 1
JOB1=$(sbatch --parsable slurm/fase1_baselines.sbatch)
echo "  FASE 1 (baselines):   Job ${JOB1}  [48h]"

# Submit Fase 2 — runs only after Fase 1 succeeds
JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} slurm/fase2_distill.sbatch)
echo "  FASE 2 (distill):     Job ${JOB2}  [24h]  (after ${JOB1})"

# Submit Fase 3 — runs only after Fase 2 succeeds
JOB3=$(sbatch --parsable --dependency=afterok:${JOB2} slurm/fase3_posteval.sbatch)
echo "  FASE 3 (post-eval):   Job ${JOB3}  [48h]  (after ${JOB2})"

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Pipeline submitted: ${JOB1} → ${JOB2} → ${JOB3}"
echo "  Max total time: 48 + 24 + 48 = 120h (5 days)"
echo ""
echo "  Monitor with:"
echo "    squeue -u \$USER"
echo "    tail -f logs/fase1-baselines-${JOB1}.out"
echo "══════════════════════════════════════════════════════════"
