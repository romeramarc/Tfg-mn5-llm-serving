#!/bin/bash
# ─────────────────────────────────────────────────────────────
# LAUNCHER v2: Improved distillation pipeline
# ─────────────────────────────────────────────────────────────
#
# Changes from v1:
#   - DataCollator bug fixed (completion-only masking now works)
#   - MATH train data re-enabled (~15K total training examples)
#   - Model-specific hyperparameters (7B: lr=5e-5, 1.5B: lr=2e-4)
#   - LoRA alpha increased (64 base, 128 for 1.5B)
#   - Training logs saved (training_log.json)
#
# Dependency graph:
#
#   distill_gen (8h) ─┬─ train_7b  (10h) ─── posteval_7b  (18h)
#                      └─ train_1.5b( 6h) ─── posteval_1.5b(12h)
#
# Critical path: distill_gen → train_7b → posteval_7b ≈ 36h
#
# NOTE: Baselines are NOT re-run (already available from v1).
#
# Usage:
#   bash slurm/launch_distill_v2.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

cd "$(dirname "$0")/.."

echo "══════════════════════════════════════════════════════════"
echo "  DISTILLATION v2 — Improved Pipeline"
echo "══════════════════════════════════════════════════════════"
echo "  Date:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
echo ""
echo "  Improvements over v1:"
echo "    - DataCollator bug fixed (completion-only masking)"
echo "    - MATH train data re-enabled (~15K examples total)"
echo "    - Model-specific hyperparameters"
echo "    - LoRA alpha increased"
echo "══════════════════════════════════════════════════════════"
echo ""

# ── Step 1: Generate teacher outputs (GSM8K + MATH train) ────
JOB_DG=$(sbatch --parsable slurm/distill_generate.sbatch)
echo "  [1/5] distill-gen   : Job ${JOB_DG}  [ 8h]  (immediate)"

# ── Step 2: Train both students (after generation) ───────────
JOB_T7=$(sbatch --parsable --dependency=afterok:${JOB_DG} slurm/distill_train_7b.sbatch)
echo "  [2/5] train-7b      : Job ${JOB_T7}  [10h]  (after ${JOB_DG})"

JOB_T1=$(sbatch --parsable --dependency=afterok:${JOB_DG} slurm/distill_train_1.5b.sbatch)
echo "  [3/5] train-1.5b    : Job ${JOB_T1}  [ 6h]  (after ${JOB_DG})"

# ── Step 3: Post-evaluation (after each training job) ────────
JOB_P7=$(sbatch --parsable --dependency=afterok:${JOB_T7} slurm/posteval_7b.sbatch)
echo "  [4/5] posteval-7b   : Job ${JOB_P7}  [18h]  (after ${JOB_T7})"

JOB_P1=$(sbatch --parsable --dependency=afterok:${JOB_T1} slurm/posteval_1.5b.sbatch)
echo "  [5/5] posteval-1.5b : Job ${JOB_P1}  [12h]  (after ${JOB_T1})"

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  5 jobs submitted. Dependency chain:"
echo ""
echo "  ${JOB_DG} (distill-gen)"
echo "    ├── ${JOB_T7} (train-7b)    → ${JOB_P7} (posteval-7b)"
echo "    └── ${JOB_T1} (train-1.5b)  → ${JOB_P1} (posteval-1.5b)"
echo ""
echo "  Estimated total wall-clock: ~36h (critical path)"
echo ""
echo "  Monitor:"
echo "    squeue -u \$USER"
echo "    tail -f logs/distill-gen-${JOB_DG}.out"
echo "    tail -f logs/distill-7b-${JOB_T7}.out"
echo "    tail -f logs/distill-1.5b-${JOB_T1}.out"
echo "══════════════════════════════════════════════════════════"
