# CC Audit & Active Codex Handoff — 2026-07-01

Author: CC (local Windows coordination/audit agent). Scope: audit verdict +
three-way sync record + the active goal handed to remote Codex for
`/data/cyx/1030/scLatent`. `goal.md` remains the top steering authority.

## Three-Way Sync (verified 2026-07-01)

| side | ref | state |
|---|---|---|
| local `E:\cc_workspace\scLatent` | main `56a9bd2` | clean |
| GitHub `cfy2yue/scRepresentation` | origin/main `56a9bd2` | in sync |
| server `/data/cyx/1030/scLatent` | main `56a9bd2` | clean, 0 dirty |

All three sides aligned; nothing to reconcile. No tmux session was running at
audit time (the doc-referenced Track-C seed jobs were not live).

## Audit Verdict

- **Goal**: reasonable and current. Leakage-safe latent-space perturbation
  prediction + scaling-axis/failure-map audit, no-harm gates, preserved negative
  evidence; default `xverse_8k_anchor`. **Keep as-is.**
- **Direction & evidence**: strongest = per-condition true-cell *support* signal;
  weakest = monotonic scaling-law assumption (empirically non-monotonic);
  bottleneck = LatentFM has not cleared the biological-insight bar, and the
  multi-condition (Track-C *query*) route is unsolved.
- **Live gate status (from docs)**: Track-C pair-type **support-only** robustness
  used a predeclared **2/3-seed + no-hard-fail** gate. seed43 **passed**
  (pp/MMD +0.1013/−0.0088); **seed45 hard-failed**; seed44 posthoc was pending.
  Because the gate requires *no hard fail*, the branch is very likely **not
  promotable** → the right move is to close it (preserving negative evidence) and
  pivot to packaging the CPU-only scaling-axis/failure-map manuscript artifact
  that already exists, rather than launching more GPU work.

## Top Optimization Directions

1. **(Active Codex goal)** Evaluate the gate from completed posthoc; if the
   no-hard-fail condition is violated, close the support-only branch and assemble
   the CPU-only scaling-axis + failure-map report into a manuscript-ready
   artifact. No new GPU training.
2. (Future, separate hypothesis) Define the multi-condition Track-C *query* route
   with its own leakage-safe split, written gate, and stop rule — do not launch
   yet.

## Ownership For Parallel Work

- **CC owns**: `goal.md`, `docs/EXPERIMENT_INDEX.md`, `docs/PROJECT_REVIEW.md`,
  `docs/DECISIONS.md`, this handoff doc.
- **Codex owns**: `runs/`, `reports/`, manuscript artifacts, `RUN_STATUS.md`.

## Active Codex Goal (handed off 2026-07-01)

```
Project: scLatent
Server path: /data/cyx/1030/scLatent
Goal: Evaluate the predeclared Track-C support-only 2/3-seed + no-hard-fail gate
  using the completed posthoc (seed43 pass, seed45 hard-fail, seed44 posthoc). If
  the no-hard-fail condition is violated, CLOSE the support-only branch (preserve
  negative evidence) and assemble the existing CPU-only scaling-axis + failure-map
  report into a manuscript-ready artifact (report manifest + reproduction manifest
  + narrative skeleton). Do NOT launch any new GPU training.
Why now: seed45 hard-failed the no-hard-fail gate, so the support-only branch is
  very likely not promotable; the real deliverable is the CPU-only
  scaling-axis/failure-map manuscript, not more GPU chasing.
Read first: goal.md, docs/EXPERIMENT_INDEX.md, docs/PROJECT_REVIEW.md,
  docs/DECISIONS.md, runs/latentfm_trackc_support_only_robustness_20260624/*/RUN_STATUS.md
Codex owns: runs/, reports/, manuscript artifacts, RUN_STATUS
CC owns: goal.md, docs/EXPERIMENT_INDEX.md, docs/PROJECT_REVIEW.md, docs/DECISIONS.md
Permissions: workspace-write (CPU-only)
Forbidden: launch ANY new GPU job; touch datasets/checkpoints destructively;
  delete results/negative evidence; print/commit secrets.
Success criteria: gate evaluated from the posthoc state; close/keep decision
  recorded in RUN_STATUS.md + proposed for docs/DECISIONS.md and
  docs/EXPERIMENT_INDEX.md; manuscript report manifest assembled from existing
  CPU artifacts with exact paths.
Stop rules: if seed44 posthoc PASSES cleanly AND seed45 re-scores as NOT a hard
  fail, do NOT close — append a DECISION NEEDED block to RUN_STATUS.md and escalate
  to CC instead. ANTI-SPIN: if the posthoc state is ambiguous or assembly stalls
  after two attempts, STOP and escalate to CC with the specific question.
Expected output paths: runs/.../RUN_STATUS.md + a manuscript report manifest path.
Progress reporting: append dated lines to RUN_STATUS.md; final summary via
  codex --output-last-message.
```

## Budget / Monitoring (CC)

CC/Claude daily spend gate < $80 (Codex spend separate). This goal is CPU-only
and decision-driven; ~1h *strategic* check that the close/keep decision is sound
and the manuscript manifest is coherent. Escalate-or-correct rather than re-run.
