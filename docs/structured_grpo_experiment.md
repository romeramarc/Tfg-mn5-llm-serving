# Structured Distillation + GRPO-like Experiment (1.5B)

This runbook defines a parallel line that does not replace existing flows.

## What this line changes

1. Teacher generation uses strict structured prompts and enforces a final line contract:
   - `#### <answer>`
   - final line must be the last non-empty line
2. Quality filtering is stricter and traceable:
   - decision per sample (`kept`, `dropped`, `kept_backfill`)
   - explicit reason list per sample
3. Student SFT writes to clearly tagged run directories:
   - `sft-qwen2.5-1.5b-structured-*`
4. Optional GRPO-like short refinement stage:
   - multi-sample per prompt
   - reward = correctness + format
   - per-prompt group-normalized advantage
   - weighted completion-only SFT on selected samples
5. Dedicated post-eval script for the GRPO adapter.

## Config files

- Main run config:
  - `configs/distill_1p5b_structured_grpo.yaml`
- Smoke run config (minimal integration):
  - `configs/distill_1p5b_structured_grpo_smoke.yaml`
- Smoke eval config:
  - `configs/eval_smoke.yaml`

## Real preflight (recommended before submit)

Config and chain wiring check:

```bash
python -m distill.preflight \
  --step config \
  --simulate-offline \
  --config configs/distill_1p5b_structured_grpo.yaml \
  --require-grpo
```

Teacher generation prerequisites check:

```bash
python -m distill.preflight \
  --step 1 \
  --simulate-offline \
  --config configs/distill_1p5b_structured_grpo.yaml
```

## End-to-end launch (full line)

```bash
bash slurm/launch_distill_1p5b_structured_grpo.sh
```

Equivalent explicit call:

```bash
DISTILL_CONFIG=configs/distill_1p5b_structured_grpo.yaml \
EVAL_CONFIG=configs/eval.yaml \
bash slurm/launch_distill_1p5b_structured_grpo.sh
```

If `grpo_refinement.enabled: false` in the config, the launcher automatically
skips the GRPO job and runs distill-only post-eval.

## Smoke launch (integration check)

Recommended one-command smoke chain:

```bash
bash slurm/launch_distill_1p5b_structured_grpo_smoke.sh
```

Equivalent explicit call:

```bash
DISTILL_CONFIG=configs/distill_1p5b_structured_grpo_smoke.yaml \
EVAL_CONFIG=configs/eval_smoke.yaml \
SKIP_PERF_BENCH=1 \
NUM_REPS=1 \
bash slurm/launch_distill_1p5b_structured_grpo_smoke.sh
```

## Smoke sequence in manual mode (exact order)

```bash
python -m distill.preflight \
  --step config \
  --simulate-offline \
  --config configs/distill_1p5b_structured_grpo_smoke.yaml \
  --require-grpo \
  --smoke

python -m distill.preflight \
  --step 1 \
  --simulate-offline \
  --config configs/distill_1p5b_structured_grpo_smoke.yaml

BASE_MODEL=$(python - <<'PY'
from utils.config_loader import load_yaml
cfg = load_yaml('configs/distill_1p5b_structured_grpo_smoke.yaml')
print(cfg.get('training', {}).get('student_model', 'Qwen/Qwen2.5-1.5B-Instruct'))
PY
)

ADAPTER_GLOB="results/distill/sft-qwen2.5-1.5b-structured-smoke-*/final_adapter"
MERGED="results/distill/merged-qwen2.5-1.5b-structured-smoke"
EVAL_ROLE="distilled_student_small_structured_smoke"

JOB_DG=$(sbatch --parsable \
  --export=ALL,DISTILL_CONFIG=configs/distill_1p5b_structured_grpo_smoke.yaml \
  slurm/distill_generate.sbatch)

JOB_T1=$(sbatch --parsable --dependency=afterok:${JOB_DG} \
  --export=ALL,DISTILL_CONFIG=configs/distill_1p5b_structured_grpo_smoke.yaml \
  slurm/distill_train_1.5b.sbatch)

JOB_E2=$(sbatch --parsable --dependency=afterok:${JOB_T1} \
  --export=ALL,BASE_MODEL=${BASE_MODEL},ADAPTER_GLOB=${ADAPTER_GLOB},MERGED=${MERGED},EVAL_ROLE=${EVAL_ROLE},EVAL_CONFIG=configs/eval_smoke.yaml,SKIP_PERF_BENCH=1,NUM_REPS=1 \
  slurm/posteval_1.5b.sbatch)

JOB_G3=$(sbatch --parsable --dependency=afterok:${JOB_E2} \
  --export=ALL,DISTILL_CONFIG=configs/distill_1p5b_structured_grpo_smoke.yaml \
  slurm/distill_grpo_1.5b.sbatch)

JOB_P4=$(sbatch --parsable --dependency=afterok:${JOB_G3} \
  --export=ALL,DISTILL_CONFIG=configs/distill_1p5b_structured_grpo_smoke.yaml,EVAL_CONFIG=configs/eval_smoke.yaml,SKIP_PERF_BENCH=1,NUM_REPS=1 \
  slurm/posteval_1.5b_structured_grpo.sbatch)

echo "DG=${JOB_DG} T1=${JOB_T1} E2=${JOB_E2} G3=${JOB_G3} P4=${JOB_P4}"
```

## Main outputs

- Teacher outputs (filtered):
  - `results/distill/teacher_outputs_1p5b_structured.jsonl`
  - `results/distill/teacher_outputs_1p5b_structured_smoke.jsonl`
- Teacher full audit trace:
  - `results/distill/teacher-gen-1p5b-structured-grpo-*/teacher_outputs_all.jsonl`
  - `results/distill/teacher-gen-1p5b-structured-grpo-smoke-*/teacher_outputs_all.jsonl`
  - `results/distill/teacher-gen-*/generation_summary.json`
- Structured SFT adapters:
  - `results/distill/sft-qwen2.5-1.5b-structured-*/final_adapter`
  - `results/distill/sft-qwen2.5-1.5b-structured-smoke-*/final_adapter`
- GRPO-like adapters:
  - `results/distill/sft-qwen2.5-1.5b-structured-grpo-*/final_adapter`
  - `results/distill/sft-qwen2.5-1.5b-structured-grpo-smoke-*/final_adapter`
- GRPO run artifacts:
  - `grpo_samples_all.jsonl`
  - `grpo_samples_selected.jsonl`
  - `grpo_reward_summary.json`
  - `grpo_manifest.json`
