# Handoff For New Codex Session: 2026-06-22

This file is for a newly opened Codex session to catch up quickly and safely.

## First Read Order

1. `/data/cyx/1030/AGENTS.md`
2. `/data/cyx/1030/goal.md`
3. `/data/cyx/1030/docs/PROJECT_REVIEW.md`
4. Current run statuses:
   - `/data/cyx/1030/runs/latentfm_xverse_trackc_routed_distill_20260622/xverse_trackc_route_condprior_w05_replay1_2k_seed42/RUN_STATUS.md`
   - `/data/cyx/1030/runs/latentfm_trackc_routed_distill_1800s_check_20260622/RUN_STATUS.md`

## Non-Negotiable Operating Constraints

- Follow `/data/cyx/1030/AGENTS.md`.
- Long jobs must be detached with `tmux`/`nohup` and must have `RUN_STATUS.md`.
- Do not frequently poll long jobs. A delayed check has already been scheduled.
- Before launching GPU tasks, sample GPU availability at least 3 times with about 10 seconds between samples.
- Empty GPU means `memory.used < 4096 MiB` and `utilization.gpu < 10%` in every sample.
- Use at most 4 physical GPUs for our work.
- If at least 5 GPUs are empty, up to 4 physical GPUs may be used; if fewer than 5 are empty, leave at least 1 empty GPU unused.
- LatentFM strategy probes may colocate up to 4 jobs per GPU only if CPU/RAM/I/O are safe.
- Use subagents only for clear read-only audits, planning reviews, code/metric audits, or well-scoped disjoint work. Do not duplicate work across agents.

## Current Scientific Policy

- Current Track A bottom line is single/background first:
  - `cross_background_seen_gene`
  - `all_test_single`
  - `family_gene`
- Canonical `split_seed42.json` remains frozen.
- Do not move canonical `test_multi*` into Track A training.
- Canonical multi has selection weight `0`; it is only a zero-shot composition diagnostic/failure-analysis signal.
- Formal multi capability requires separate Track C true-multi support adaptation:
  - train/fine-tune on true multi support;
  - select only using support-val via `split_seed42_multi_support_v2_trainselect.json`;
  - evaluate full `split_seed42_multi_support_v2.json` query exactly once after route/checkpoint freeze.
- Track C claims must be phrased as true-multi support adaptation, not replacement of the canonical zero-shot/single-background claim.

## Active/Recent Work

Track C routed-distill smoke was launched:

- Run status:
  `/data/cyx/1030/runs/latentfm_xverse_trackc_routed_distill_20260622/xverse_trackc_route_condprior_w05_replay1_2k_seed42/RUN_STATUS.md`
- Launcher:
  `/data/cyx/1030/ops/launch_latentfm_xverse_trackc_routed_distill_smoke_20260622.sh`
- Summary/gate script:
  `/data/cyx/1030/ops/summarize_latentfm_trackc_routed_distill_smoke_20260622.py`
- Expected decision report after posthoc:
  `/data/cyx/1030/reports/LATENTFM_TRACKC_ROUTED_DISTILL_SMOKE_DECISION_xverse_trackc_route_condprior_w05_replay1_2k_seed42.md`

The smoke uses:

- xverse 8k seed42 anchor warm-start;
- safe trainselect split:
  `/data/cyx/1030/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json`;
- route artifact:
  `/data/cyx/1030/reports/latentfm_trackc_support_route_teacher_20260622.json`;
- Norman route = `additive_single_sum`;
- Wessels route = `dataset_multi_mean`;
- trainable scope = `condition_prior_adapter`;
- Track C routed-distill weight = `0.5`;
- anchor replay = `1.0`;
- posthoc support-val + canonical no-harm only;
- held-out query intentionally not evaluated.

A delayed 1800-second single check was scheduled:

- Run status:
  `/data/cyx/1030/runs/latentfm_trackc_routed_distill_1800s_check_20260622/RUN_STATUS.md`
- tmux session:
  `trackc_route_1800s_check_20260622`

Do not add frequent manual polling on top of this unless the user explicitly asks.

## What To Do Next

If the Track C smoke is still running:

- Do not keep polling.
- Work on non-blocking tasks: decision scripts, documentation cleanup, or a CPU-only next-gate proposal.

If the smoke/posthoc decision report exists:

1. Read the decision report.
2. If support/canonical gate fails, close the routed-distill smoke and record negative evidence in `goal.md` and `PROJECT_REVIEW.md`.
3. If it passes, do not immediately read held-out query.
4. First run uncapped canonical no-harm evaluation.
5. Only if uncapped canonical no-harm passes, freeze route/checkpoint and run one-shot Track C query eval.

## Self-Check Questions For The New Session

Before launching anything, the new session should be able to answer:

1. What are the current Track A model-selection metrics?
2. Why is canonical multi selection weight `0`?
3. Which split is safe for Track C training-selection, and why is full v2 unsafe during training?
4. Which Track C run is active, and where is its `RUN_STATUS.md`?
5. What report should appear after posthoc?
6. What must happen before held-out query is evaluated?
7. What GPU rules must be followed before launching any new training job?

If the session cannot answer these from files, it has not caught up.
