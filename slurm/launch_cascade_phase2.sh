#!/bin/bash
# ------------------------------------------------------------------
# Launch script - Phase 2 cascade on BSC (3 servers + evaluator)
# ------------------------------------------------------------------
# Submits:
#   1) teacher server job
#   2) student_mid server job
#   3) student_small server job
#   4) cascade evaluator job (depends on all three servers having started)
#
# Server jobs are long-running by design; when evaluation finishes,
# cancel them manually or submit the optional cleanup command shown below.
# ------------------------------------------------------------------

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd -P)"

mkdir -p results/routing/endpoints
rm -f \
	results/routing/endpoints/teacher.url \
	results/routing/endpoints/student_mid.url \
	results/routing/endpoints/student_small.url

echo "=========================================="
echo " Launching Phase 2 cascade (BSC)"
echo "=========================================="
echo "Project dir        : ${PROJECT_DIR}"
echo "Endpoint files     : cleaned"

JOB_TEACHER=$(sbatch --parsable --export="ALL,PROJECT_DIR=${PROJECT_DIR},ROLE=teacher" slurm/server_role_phase2.sbatch)
echo "teacher server     : ${JOB_TEACHER}"

JOB_MID=$(sbatch --parsable --export="ALL,PROJECT_DIR=${PROJECT_DIR},ROLE=student_mid" slurm/server_role_phase2.sbatch)
echo "student_mid server : ${JOB_MID}"

JOB_SMALL=$(sbatch --parsable --export="ALL,PROJECT_DIR=${PROJECT_DIR},ROLE=student_small" slurm/server_role_phase2.sbatch)
echo "student_small srv  : ${JOB_SMALL}"

# Start evaluator after all servers have entered RUNNING state at least once.
JOB_EVAL=$(sbatch --parsable --dependency=after:${JOB_TEACHER}:${JOB_MID}:${JOB_SMALL} --export="ALL,PROJECT_DIR=${PROJECT_DIR}" slurm/eval_cascade_phase2.sbatch)
echo "cascade evaluator  : ${JOB_EVAL}"

echo ""
echo "Submitted jobs:"
echo "  teacher      ${JOB_TEACHER}"
echo "  student_mid  ${JOB_MID}"
echo "  student_small ${JOB_SMALL}"
echo "  evaluator    ${JOB_EVAL}"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/vllm-role-p2-${JOB_TEACHER}.out"
echo "  tail -f logs/eval-cascade-p2-${JOB_EVAL}.out"
echo ""
echo "After evaluator finishes, stop servers:"
echo "  scancel ${JOB_TEACHER} ${JOB_MID} ${JOB_SMALL}"
echo ""
echo "Optional auto-cleanup job submission:"
echo "  sbatch --dependency=afterany:${JOB_EVAL} --wrap=\"scancel ${JOB_TEACHER} ${JOB_MID} ${JOB_SMALL} || true\""
