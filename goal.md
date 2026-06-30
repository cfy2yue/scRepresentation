# scLatent Current Goal

Last slimmed: 2026-07-01.

This is the actionable project steering note for the server workspace
`/data/cyx/1030/scLatent` and GitHub repository
`https://github.com/cfy2yue/scRepresentation`.

The full pre-slim chronological goal log was preserved on the server at:

```text
docs/local_archive/20260630_pre_slim/goal.md
```

That archive is intentionally ignored by Git. It is provenance, not the first
document a new CC/Codex agent should read.

## Current Objective

Maintain scLatent as the monorepo entrypoint for LatentFM, scFMBench,
CoupledFM-derived experiments, benchmark infrastructure, scaling-law audits,
and owned scripts/reports/docs. The current scientific deliverable is a
leakage-safe latent-space perturbation-prediction and scaling-axis audit with
clear no-harm gates and preserved negative evidence.

## Current Workspace Contract

- Project root: `/data/cyx/1030/scLatent`.
- Shared data root: `/data/cyx/1030/dataset`.
- Shared runtime root: `/data/cyx/1030/software`.
- Current scdfm runtime entry: `source /data/cyx/1030/scLatent/init-scdfm.sh`.
- Remote: `https://github.com/cfy2yue/scRepresentation`.
- Server login: `ssh cyx-server-proxy-cfy`.

Project assets should use `/data/cyx/1030/scLatent/...`. Shared datasets should
stay under `/data/cyx/1030/dataset/...`. Historical documents may contain older
root-level paths as provenance; new scripts and entry docs should not.

## Current Research State

- Current default/baseline model state remains the last documented default in
  project evidence (`xverse_8k_anchor`) until a newer strict gate supersedes it.
- The 2026-06-25 scaling/NM package supports a manuscript-style
  scaling-axis/failure-map audit, not a claim that a deployable monotonic scaling
  law has been solved.
- Recent scaling artifacts were produced by short CPU/report builders only. They
  do not authorize new GPU training by themselves.
- Chemical V2 or any other new GPU branch needs a fresh resource audit,
  leakage-safe split boundary, written hypothesis, stop rule, and RUN_STATUS
  before launch.

## Do Next

1. Keep GitHub publication focused on source, small configs, and high-signal
   docs. Do not track datasets, checkpoints, runs, reports, logs, venvs, or
   local archives.
2. Before any future experiment, update the relevant run status and project
   review with the hypothesis, gate, resource plan, and expected outputs.
3. When reviewing old evidence, use the local archive and server reports as
   provenance, but summarize only the current conclusion in Git-tracked docs.
4. If CC prepares goals locally on Windows, it should hand the final goal to
   Codex for server execution rather than pretending local Windows can run
   GPU/data-heavy work.

## CC Audit + Active Handoff (2026-07-01)

- Three-way sync verified: local = GitHub = server, all at `56a9bd2`, clean.
- Audit verdict: goal + direction reasonable; keep leakage-safe latent +
  scaling-axis audit framing. Track-C support-only gate: seed45 hard-failed the
  no-hard-fail condition → branch likely not promotable.
- Active Codex goal: evaluate the gate from posthoc; if no-hard-fail is violated,
  close the support-only branch (preserve negative evidence) and package the
  CPU-only scaling-axis/failure-map manuscript artifact — no new GPU. Details +
  ownership in `docs/CC_AUDIT_AND_HANDOFF_20260701.md`.
- Ownership: CC owns goal/index/review/decision/handoff docs; Codex owns runs/
  reports/RUN_STATUS.

## Read First

```text
README.md
AGENTS.md
docs/WORKSPACE_ORGANIZATION.md
docs/CODEX_CC_COLLABORATION.md
docs/GIT_AND_COLLABORATION.md
docs/GITHUB_FILE_MAP.md
docs/PROJECT_OVERVIEW.md
docs/PROJECT_REVIEW.md
docs/EXPERIMENT_INDEX.md
```

## Non-Goals For This Cleanup

No experiments, GPU tasks, training, inference, large data processing, result
deletion, or checkpoint movement were requested or performed during the
2026-06-30 workspace/Git cleanup.
