# Mainline Session Handoff: 2026-06-24

Use this file as the initial prompt for a new Codex session if the user wants a
separate session to execute the LatentFM mainline while the current session acts
as coordinator/user-facing monitor.

## Role

You are the mainline execution session for `/data/cyx/1030`.

The existing coordinator session remains responsible for user-facing summary,
resource-policy arbitration, resource-utilization monitoring, macro
project-progress monitoring, exception handling, and final decision integration.
Because sessions cannot directly message each other, coordinate through files:

- `/data/cyx/1030/goal.md`
- `/data/cyx/1030/docs/PROJECT_REVIEW.md`
- `/data/cyx/1030/docs/EXPERIMENT_INDEX.md`
- `/data/cyx/1030/docs/MAINLINE_AGENT_STATUS_20260624.md`
- each run's `RUN_STATUS.md`

Before launching any GPU job, reread these files and current resource state.
Never assume old conversation memory is authoritative.

## Required Rules

1. Read and obey `/data/cyx/1030/AGENTS.md` first.
2. Long tasks must be detached to `tmux`/`nohup` and have `RUN_STATUS.md`.
3. Do not frequently poll long jobs. After one sanity check, return to a long
   job only after about 1800 seconds unless there is crash/exit evidence.
4. Current hard cap: at most 4 physical GPUs total for active LatentFM work; at
   most 48 CPU cores for this project; at most 4 LatentFM training jobs per GPU
   only when CPU/RAM/I/O are safe.
5. Empty GPU definition for planning: at least 3 samples, about 10 seconds apart,
   with `memory.used < 4096 MiB` and util `< 10%` in every sample.
6. Canonical `split_seed42.json` is not recut. Canonical multi is diagnostic
   only for Track A and never used for checkpoint selection. Track C query is
   held out until route/checkpoint are frozen.
7. All experiments need hypothesis, resource plan, launcher/RUN_STATUS, gate,
   and failure-close rule.
8. During active exploration, record a lightweight resource-utilization and
   macro-progress checkpoint about once per hour or at each major decision.
   Do not use this as permission to poll long-job logs more often than allowed.

## Status Reporting Contract

When acting as the mainline execution agent, update:

`/data/cyx/1030/docs/MAINLINE_AGENT_STATUS_20260624.md`

after major decisions, launches, branch closures, summarizer completion, or an
hourly active-exploration checkpoint. Include:

- timestamp;
- active tmux sessions and active LatentFM physical GPUs;
- CPU/RAM summary and whether use stays within the 48-core project budget;
- whether GPU usage matches the intended exploration portfolio;
- current best model/settings;
- active branches and closed branches;
- next gate and expected decision point;
- blockers or anomaly requiring coordinator attention.

## Current Mainline State

Latest pointer should be verified from `/data/cyx/1030/goal.md`, but as of
2026-06-24 14:45 CST:

- Static best-vs-latest update audit is closed as `no_latest_rescue`.
  Report:
  `/data/cyx/1030/reports/LATENTFM_STATIC_BEST_LATEST_UPDATE_AUDIT_20260624.md`.
- Dataset-specific MMD/replay hook is implemented and validated.
  Report:
  `/data/cyx/1030/reports/LATENTFM_RISK_CONDITIONED_DATASET_HOOK_VALIDATION_20260624.md`.
- Singer subagent audited the hook and found no wiring blocker.
- Active risk-conditioned general-exposure portfolio:
  run root
  `/data/cyx/1030/runs/latentfm_risk_conditioned_general_exposure_smoke_20260624/`.

Four arms are part of this portfolio:

| Run | Intent | Status at handoff |
|---|---|---|
| `xverse_general_exposure_tian_mmd20_replayall_3k_seed42` | Tian-targeted MMD + all-dataset replay | train/posthoc exit `0` |
| `xverse_general_exposure_tian_mmd20_replaytian_3k_seed42` | Tian-targeted MMD + Tian-only replay | train/posthoc exit `0` |
| `xverse_general_exposure_tian_mmd20_noreplay_3k_seed42` | Tian-targeted MMD, no replay | train exit `0`; posthoc pending/running |
| `xverse_general_exposure_tian_norman_mmd20_replayall_3k_seed42` | Tian+Norman targeted MMD + all-dataset replay | running |

The batch status file is:
`/data/cyx/1030/runs/latentfm_risk_conditioned_general_exposure_smoke_20260624/RUN_STATUS.md`.

## Immediate Next Action

Do not read the running logs repeatedly.

If at least ~1800 seconds have elapsed since the last check or if tmux/exit
markers show natural completion, inspect only markers first:

```bash
for d in /data/cyx/1030/runs/latentfm_risk_conditioned_general_exposure_smoke_20260624/xverse_general_exposure_*_3k_seed42; do
  [ -d "$d" ] || continue
  echo "$(basename "$d")"
  for f in "$d"/*.EXIT_CODE "$d"/POSTHOC_EXIT_CODE; do
    [ -e "$f" ] && printf '  %s: ' "$(basename "$f")" && cat "$f"
  done
done
```

When all expected posthoc markers are `0`, run:

```bash
python /data/cyx/1030/ops/summarize_latentfm_risk_conditioned_general_exposure_smoke_20260624.py
sed -n '1,220p' /data/cyx/1030/reports/LATENTFM_RISK_CONDITIONED_GENERAL_EXPOSURE_SMOKE_DECISION_20260624.md
```

Then update:

- `/data/cyx/1030/goal.md`
- `/data/cyx/1030/docs/PROJECT_REVIEW.md`
- `/data/cyx/1030/docs/EXPERIMENT_INDEX.md`
- relevant `RUN_STATUS.md`

## Gate

A risk-conditioned arm only remains interesting if it:

- preserves cross-background/internal Pearson signal;
- avoids family Pearson harm;
- controls `TianActivation` mean MMD tail;
- does not expand risk-dataset MMD harm;
- has config provenance showing the intended dataset filters were active.

If an arm passes internal gate, do not promote directly. Request external
review first, then decide whether frozen canonical no-harm is justified.

If all arms fail, close this branch as targeted-MMD negative evidence and pivot
to the next documented hypothesis instead of launching scalar gamma/replay
sweeps.

## Coordination Note

Before launching a new GPU job, check whether another session has updated
`goal.md` or this run root after your session started. If yes, reconcile first.
Do not exceed the shared 4-physical-GPU cap across all active sessions.
