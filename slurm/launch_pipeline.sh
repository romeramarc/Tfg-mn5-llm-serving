#!/bin/bash
# ─────────────────────────────────────────────────────────────
# LAUNCHER: Parallel fine-grained pipeline (8 jobs)
# ─────────────────────────────────────────────────────────────
#
# Dependency graph:
#
#   t=0  ┌─ eval_teacher  (18h) ──────────────────────────┐
#        ├─ eval_mid      (18h) ──────────────────────────┤ done
#        ├─ eval_small    (12h) ──────────────────────────┘
#        └─ distill_gen   ( 8h) ─┬─ train_7b   (8h) ─── posteval_7b  (18h)
#                                 └─ train_1.5b (6h) ─── posteval_1.5b(12h)
#
# Critical path: distill_gen → train_7b → posteval_7b  ≈ 34h
# All individual jobs fit inside the 24h wall-clock limit.
#
# Usage:
#   bash slurm/launch_pipeline.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

cd "$(dirname "$0")/.."

echo "══════════════════════════════════════════════════════════"
echo "  Launching PARALLEL PIPELINE (8 fine-grained jobs)"
echo "══════════════════════════════════════════════════════════"

# ── Wave 0: all 4 independent jobs start immediately ────────
JOB_ET=$(sbatch --parsable slurm/eval_teacher.sbatch)
echo "  eval-teacher  : Job ${JOB_ET}  [18h]  (immediate)"

JOB_EM=$(sbatch --parsable slurm/eval_mid.sbatch)
echo "  eval-mid      : Job ${JOB_EM}  [18h]  (immediate)"

JOB_ES=$(sbatch --parsable slurm/eval_small.sbatch)
echo "  eval-small    : Job ${JOB_ES}  [12h]  (immediate)"

JOB_DG=$(sbatch --parsable slurm/distill_gen.sbatch)
echo "  distill-gen   : Job ${JOB_DG}  [ 8h]  (immediate)"

# ── Wave 1: training — after distill_gen ────────────────────
JOB_T7=$(sbatch --parsable --dependency=afterok:${JOB_DG} slurm/train_7b.sbatch)
echo "  train-7b      : Job ${JOB_T7}  [ 8h]  (after ${JOB_DG})"

JOB_T1=$(sbatch --parsable --dependency=afterok:${JOB_DG} slurm/train_1.5b.sbatch)
echo "  train-1.5b    : Job ${JOB_T1}  [ 6h]  (after ${JOB_DG})"

# ── Wave 2: post-eval — each after its own training job ─────
JOB_P7=$(sbatch --parsable --dependency=afterok:${JOB_T7} slurm/posteval_7b.sbatch)
echo "  posteval-7b   : Job ${JOB_P7}  [18h]  (after ${JOB_T7})"

JOB_P1=$(sbatch --parsable --dependency=afterok:${JOB_T1} slurm/posteval_1.5b.sbatch)
echo "  posteval-1.5b : Job ${JOB_P1}  [12h]  (after ${JOB_T1})"

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  8 jobs submitted. Dependency graph:"
echo "  ${JOB_ET} (eval-teacher)"
echo "  ${JOB_EM} (eval-mid)"
echo "  ${JOB_ES} (eval-small)"
echo "  ${JOB_DG} (distill-gen) → ${JOB_T7} (train-7b) → ${JOB_P7} (posteval-7b)"
echo "  ${JOB_DG} (distill-gen) → ${JOB_T1} (train-1.5b) → ${JOB_P1} (posteval-1.5b)"
echo ""
echo "  Monitor with:"
echo "    squeue -u \$USER"
echo "    tail -f logs/eval-teacher-${JOB_ET}.out"
echo "    tail -f logs/distill-gen-${JOB_DG}.out"
echo "══════════════════════════════════════════════════════════"
