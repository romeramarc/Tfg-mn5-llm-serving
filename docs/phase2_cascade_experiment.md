# Phase 2 Cascade Experiment Design (Comparable to Baselines)

This document defines the phase-2 cascade experiment with strict comparability to the existing teacher-only and single-model baseline protocol.

## 1) Comparable Experiment Definition

Use the same controls as phase-1 quality and efficiency runs:

- Same benchmarks and splits from `configs/eval.yaml` (GSM8K test, MATH-500 test, optional ARC).
- Same prompt templates from `configs/eval.yaml`.
- Same decoding controls for evaluation (`temperature`, `max_tokens`).
- Same benchmark harness assumptions:
  - offline quality metrics in `results/quality/...`
  - online/service metrics in `results/routing/...`
- Same seed handling and metadata capture (`set_seed`, config snapshots, `run_meta.json`).
- Same aggregation semantics:
  - `accuracy_pct` measured on scorable examples: `correct / scorable_examples`
  - `total_examples` and `unscorable_examples` preserved for diagnostics.

## 2) Technical Cascade Design

Fixed execution path per request:

1. `student_small` (1.5B)
2. `student_mid` (7B)
3. `teacher` (14B)

Decision logic:

- Query 1.5B with logprobs.
- Accept at stage 1 if confidence >= `small_confidence_threshold`.
- Else query 7B with logprobs.
- Accept at stage 2 if confidence >= `mid_confidence_threshold`.
- Else query teacher as final fallback.
- Any stage error/time-out triggers escalation to next stage.

Policy key: `cascade_three_tier` in `routing/policies.py`.

## 3) Confidence Baselines and Recommendation

Supported confidence methods:

- `entropy` (recommended): normalized entropy over top-k token probabilities.
- `max_logprob`: average max token probability.
- lexical heuristic fallback when logprobs are unavailable.

Recommendation for this codebase:

- Keep `entropy` as default (`fallback_method: entropy`), because it is more stable than max-only confidence in long-form generations.
- Start with thresholds:
  - `small_confidence_threshold: 0.45`
  - `mid_confidence_threshold: 0.60`
- Tune thresholds with sweep experiments, but keep all non-threshold controls frozen.

## 4) Instrumentation (Online vs Offline)

Online/service instrumentation (`routing.router`):

- Final model selected per request.
- End-to-end latency per request.
- Hop count (how many stages were used).
- Per-stage attempt metrics (latency, status, error, confidence, threshold).
- Aggregates in `routing_summary.json`:
  - model-selection counts
  - reason counts
  - hop-count distribution
  - attempt-stage counts
  - latency p50/p95/p99

Offline/quality instrumentation (`routing.cascade_quality`):

- Per-example benchmark records with:
  - final answer text
  - extracted prediction
  - correctness
  - scorable/unscorable
  - selected model and route reason
  - attempt traces
- Benchmark-level metrics JSON/CSV compatible with existing `eval/run_quality.py` output style.

## 5) Per-Request Logging Schema

Top-level decision record (`routing_decisions.json`):

- `request_id`
- `selected_model`
- `latency_ms`
- `response_text`
- `confidence`
- `reason`
- `metadata`
- `attempts[]`

Attempt object (`attempts[]` and flattened `routing_attempts.json/csv`):

- `stage` (student_small | student_mid | teacher)
- `model`
- `base_url`
- `status_code`
- `latency_ms`
- `output_tokens`
- `finish_reason`
- `confidence`
- `threshold`
- `decision`
- `error`
- `used_logprobs`
- `response_preview`

## 6) Risks and Biases

- Confidence miscalibration across model sizes can over-escalate or under-escalate.
- Stage timeouts can bias routing decisions toward larger models under transient load.
- Logprob availability differences across serving versions can shift confidence behavior.
- Prompt-template drift between baseline and cascade runs breaks comparability.
- Dataset leakage or split mismatch invalidates quality conclusions.
- Running endpoints on different hardware nodes without controls can confound latency comparisons.

## 7) Phased Implementation Plan

Phase A (completed in this prototype):

- Add 3-tier policy with per-attempt tracing.
- Extend routing runner with request IDs, deterministic ordering, flattened attempts, richer summary.
- Add dedicated phase-2 config.

Phase B (completed in this prototype):

- Add cascade quality runner that reuses existing benchmark loaders and scorers.
- Persist quality outputs in phase-1-compatible structure.

Phase C (next operational step):

- Run threshold sweeps and produce a threshold selection report.
- Add final comparison table/script integrating teacher baseline, 7B baseline, 1.5B baseline, and cascade.

## 8) Code Structure

New/updated files:

- `routing/policies.py`
  - extended `RoutingDecision` schema
  - added `cascade_three_tier`
  - added robust attempt-level helper utilities
- `routing/router.py`
  - request-aware dispatch
  - prompt metadata support
  - attempt flattening and richer summaries
- `routing/cascade_quality.py`
  - offline quality evaluation through the routing policy
- `configs/routing_phase2.yaml`
  - fixed 1.5B -> 7B -> teacher setup
- `slurm/eval_cascade_phase2.sbatch`
  - end-to-end phase-2 eval template (expects running endpoints)

## 9) Initial Prototype Runbook

1. Run preflight checks:

  `python scripts/preflight_phase2_cascade.py`

2. Launch BSC jobs (recommended):

  `bash slurm/launch_cascade_phase2.sh`

  This submits three server jobs (`teacher`, `student_mid`, `student_small`) and one evaluator job.
  The launcher clears stale endpoint files and forwards the current repository path as `PROJECT_DIR`.
  Server jobs publish endpoint URLs to `results/routing/endpoints/*.url`.
  The evaluator generates a runtime routing config with those URLs, so no localhost assumption is required.

3. (Optional manual mode) Run online/service cascade evaluation:

   `python -m routing.router --config configs/routing_phase2.yaml`

4. (Optional manual mode) Run offline quality cascade evaluation:

   `python -m routing.cascade_quality --eval-config configs/eval.yaml --routing-config configs/routing_phase2.yaml --role cascade_phase2`

5. Outputs:

   - `results/routing/cascade_three_tier-<timestamp>/`
   - `results/quality/quality-cascade_phase2-<timestamp>/`

6. Compare against baseline outputs using the same benchmark protocol and summary fields.
