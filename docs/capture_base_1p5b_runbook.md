# Capture 1.5B pre-KD (`student_small_base`) — runbook

Adds operational traces for the **same three predictors** (service cost, quality ex-ante, quality post-hoc) using **`Qwen/Qwen2.5-1.5B-Instruct`** instead of the distilled `student_small` checkpoint.

Existing captures for teacher / 7B / 3B / 0.5B are **reused** when training the “ladder-base” datasets.

## MN5 — launch captures (one command)

```bash
cd /gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving
git pull
source env/setup_env.sh
python scripts/preflight_capture_base_1p5b.py
bash slurm/launch_capture_base_1p5b.sh
```

This submits:

1. vLLM server + Phase A Poisson capture for `student_small_base`
2. After A succeeds, a wrapper job runs Phase B quality capture (GSM8K + MATH-500, logprobs, `z`)

Training jobs are **not** submitted (`SKIP_TRAIN=1`).

## After captures finish — train predictors (ladder with base 1.5B)

```bash
PHASE_A_CONFIG=configs/phase_a_train_ladder_base_1p5b.yaml \
  sbatch slurm/phase_a_train.sbatch

PHASE_B_CONFIG=configs/phase_b_train_ladder_base_1p5b.yaml \
  sbatch slurm/phase_b_train.sbatch
```

Outputs:

- `results/phase_a/datasets/service_cost_phase_a_ladder_base.jsonl`
- `results/phase_b/datasets/quality_ex_ante_phase_b_ladder_base.jsonl`
- `results/phase_b/datasets/quality_post_hoc_phase_b_ladder_base.jsonl`

Then refine (copy `refine_phase_*.yaml` paths to these datasets) and wire routing policies.

## Distilled vs base ladders

| Ladder | 1.5B rung | Dataset suffix |
|--------|-----------|----------------|
| Distilled (current) | `student_small` → v6 FT | `*_phase_a` / `*_phase_b` |
| Base (this run) | `student_small_base` → HF instruct | `*_ladder_base` |

Keep both to compare policies **with / without** distillation at the same rung.
