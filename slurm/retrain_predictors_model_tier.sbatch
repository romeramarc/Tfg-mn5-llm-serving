#!/bin/bash
# SLURM - rebuild datasets and refine predictors with model_tier as a feature.
#SBATCH --job-name=pred-tier
#SBATCH --output=logs/predictors-model-tier-%j.out
#SBATCH --error=logs/predictors-model-tier-%j.err
#SBATCH --partition=gpp
#SBATCH --account=bsc98
#SBATCH --qos=gp_bsccs
#SBATCH --ntasks=1
# RF-only refine: 8 CPUs suffice (sklearn RF uses n_jobs=-1). Walltime 4h is enough
# vs 12h when we also searched gradient_boosting.
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
TRACE_PATTERN="${TRACE_PATTERN:-results/predictors/traces/iter2_real_multimodel_trace.jsonl}"
DATASET_DIR="${DATASET_DIR:-results/predictors_model_tier/datasets}"
PHASE_A_CONFIG="${PHASE_A_CONFIG:-configs/refine_phase_a_model_tier.yaml}"
PHASE_B_CONFIG="${PHASE_B_CONFIG:-configs/refine_phase_b_model_tier.yaml}"
BASE_ROUTING_CONFIG="${BASE_ROUTING_CONFIG:-configs/routing_eval_holdout_v2.yaml}"
OUTPUT_ROUTING_CONFIG="${OUTPUT_ROUTING_CONFIG:-configs/routing_eval_holdout_v2_retrained.yaml}"

cd "${PROJECT_DIR}"
source env/setup_env.sh
# setup_env.sh unsets PYTHONPATH on BSC; repo root must be importable for distill/, predictors/, utils/.
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p logs "${DATASET_DIR}" results/predictors_model_tier/phase_a results/predictors_model_tier/phase_b

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

PHASE_A_SOURCE="${PHASE_A_SOURCE:-results/phase_a/datasets/service_cost_phase_a.jsonl}"
PHASE_B_EX_ANTE_SOURCE="${PHASE_B_EX_ANTE_SOURCE:-results/phase_b/datasets/quality_ex_ante_phase_b.jsonl}"
PHASE_B_POST_HOC_SOURCE="${PHASE_B_POST_HOC_SOURCE:-results/phase_b/datasets/quality_post_hoc_phase_b.jsonl}"

echo "=== EARLY SANITY CHECKS (fail fast if env or sources missing) ==="
python -c "import sys; print('Python', sys.version.split()[0]); print('cwd', __import__('os').getcwd())"
python -c "from distill.dataset_utils import read_jsonl; from predictors.dataset_common import write_dataset_artifacts; from utils.config_loader import load_yaml; print('imports OK')"
for src in \
  "${PHASE_A_SOURCE}" \
  "${PHASE_B_EX_ANTE_SOURCE}" \
  "${PHASE_B_POST_HOC_SOURCE}"; do
  if [[ ! -s "${src}" ]]; then
    echo "ERROR: missing source dataset: ${src}" >&2
    exit 1
  fi
done
for cfg in "${PHASE_A_CONFIG}" "${PHASE_B_CONFIG}" "${BASE_ROUTING_CONFIG}"; do
  if [[ ! -f "${cfg}" ]]; then
    echo "ERROR: missing config: ${cfg}" >&2
    exit 1
  fi
done
echo "sanity OK"

echo "=== BUILD DATASETS WITH model_tier ==="
echo "TRACE_PATTERN=${TRACE_PATTERN}"

if [[ -f "${TRACE_PATTERN}" ]]; then
  echo "Source: raw trace JSONL"
  python -m predictors.builders.build_ex_ante_dataset \
    --input "${TRACE_PATTERN}" \
    --output-dir "${DATASET_DIR}" \
    --dataset-name quality_ex_ante_model_tier \
    --report "${DATASET_DIR}/quality_ex_ante_model_tier_build_report.json"

  python -m predictors.builders.build_post_hoc_dataset \
    --input "${TRACE_PATTERN}" \
    --output-dir "${DATASET_DIR}" \
    --dataset-name quality_post_hoc_model_tier \
    --report "${DATASET_DIR}/quality_post_hoc_model_tier_build_report.json"

  python -m predictors.builders.build_cost_dataset \
    --input "${TRACE_PATTERN}" \
    --output-dir "${DATASET_DIR}" \
    --dataset-name service_cost_model_tier \
    --feature-policy strict_ex_ante \
    --cost-mode latency_ms \
    --report "${DATASET_DIR}/service_cost_model_tier_build_report.json"
else
  echo "Trace not found; using Phase A/B datasets"
  echo "  PHASE_A_SOURCE=${PHASE_A_SOURCE}"
  echo "  PHASE_B_EX_ANTE_SOURCE=${PHASE_B_EX_ANTE_SOURCE}"
  echo "  PHASE_B_POST_HOC_SOURCE=${PHASE_B_POST_HOC_SOURCE}"
  python scripts/build_model_tier_datasets_from_phase_ab.py \
    --output-dir "${DATASET_DIR}" \
    --phase-a-source "${PHASE_A_SOURCE}" \
    --phase-b-ex-ante-source "${PHASE_B_EX_ANTE_SOURCE}" \
    --phase-b-post-hoc-source "${PHASE_B_POST_HOC_SOURCE}"
fi

echo "=== VERIFY model_tier IN ALL DATASET METADATA ==="
DATASET_DIR="${DATASET_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

dataset_dir = Path(os.environ["DATASET_DIR"])
for name in [
    "quality_ex_ante_model_tier",
    "quality_post_hoc_model_tier",
    "service_cost_model_tier",
]:
    meta = json.loads((dataset_dir / f"{name}_meta.json").read_text(encoding="utf-8"))
    features = set(meta.get("feature_columns") or [])
    if "model_tier" not in features:
        raise SystemExit(f"model_tier missing from {name}_meta.json")
    print(f"OK {name}: rows={meta.get('row_count')} features={len(features)}")
PY

echo "=== REFINE PHASE A / B ==="
python scripts/refine_predictors.py --config "${PHASE_A_CONFIG}"
python scripts/refine_predictors.py --config "${PHASE_B_CONFIG}"

echo "=== MATERIALIZE ROUTING CONFIG WITH WINNING BUNDLES ==="
python scripts/write_routing_config_from_refinements.py \
  --base-config "${BASE_ROUTING_CONFIG}" \
  --phase-a-selection results/predictors_model_tier/phase_a/REFINEMENT_SELECTION.json \
  --phase-b-selection results/predictors_model_tier/phase_b/REFINEMENT_SELECTION.json \
  --output-config "${OUTPUT_ROUTING_CONFIG}" \
  --project-root .

echo "=== RETRAINED ROUTING CONFIG ==="
OUTPUT_ROUTING_CONFIG="${OUTPUT_ROUTING_CONFIG}" python - <<'PY'
import os
from utils.config_loader import load_yaml

cfg_path = os.environ["OUTPUT_ROUTING_CONFIG"]
cfg = load_yaml(cfg_path)
print(cfg["predictors"]["bundles"])
print(cfg["predictors"]["thresholds"])
systems = [s["id"] for s in cfg.get("systems") or []]
print(f"OK {len(systems)} eval systems: {systems}")
PY

echo "=== PREDICTOR MODEL_TIER RETRAIN DONE ==="
echo "Next (manual): python scripts/preflight_eval_holdout.py --config ${OUTPUT_ROUTING_CONFIG}"
echo "       EVAL_CONFIG=${OUTPUT_ROUTING_CONFIG} PROMPT_POOL=results/routing_eval_holdout/prompt_pool.jsonl bash slurm/launch_eval_holdout.sh all"
echo "Or chain (auto after retrain): bash slurm/chain_retrain_and_holdout_v2.sh <THIS_JOB_ID>"
