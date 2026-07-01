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
- Server login: `ssh cyx-server-cfy`.

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

## Manual Local Audit State (2026-07-01)

- Active local-to-remote planning now lives in `local_goal.md`,
  `local_audit.md`, and `local_suggestion.md`.
- Historical CC/Codex handoff docs from the 2026-07-01 auto-coordination dry run
  are archived under `docs/archive/legacy_auto_coordination_20260701/`.
- Remote Codex should use archived handoffs as evidence only, not as active
  execution instructions.
- Current ownership: local CC/Codex audits strategy and updates the three
  `local_*.md` files; remote Codex executes after the user manually pulls
  GitHub and starts goal mode.
- Result context still stands: Track-C support-only CLOSED; CPU-only manuscript
  manifest built at
  `reports/trackc_support_only_closed_scaling_manuscript_manifest_20260701/`.

## Higher-Priority Direction (2026-07-01, insight-driven)

Per user steering, prioritize computational + biological INSIGHT that constrains the
method over direct flow-matching metric tuning. Two analysis-first threads are
recorded as historical archived route notes; active next steps should be
rewritten into `local_goal.md`, `local_audit.md`, and `local_suggestion.md`
before remote execution.

- **Scaling unit** (`docs/archive/legacy_auto_coordination_20260701/CC_AUDIT_AND_HANDOFF_20260701_scaling_unit.md`): cell count is
  the wrong minimal unit. Test an information / effective-state axis (Vendi N_eff) and an
  abundance/response-energy-weighted effective-gene-count G_eff vs cell count, by CPU
  regression over existing runs. HVG thesis is half-validated: HVG *concentration* is real
  (top-2k ~84% response energy) but the HVG-specific signal collapses to abundance - so
  weight by abundance/response-energy, not a bespoke HVG score. Prereq: materialize
  per-arm geometry (current join collapses best runs to one parent geometry).
- **Zebrafish dynamic-law flow-regularizer** (`docs/archive/legacy_auto_coordination_20260701/CC_AUDIT_AND_HANDOFF_20260701_zebrafish_regularizer.md`):
  mine the wild-type developmental reference atlas (GT dynamic transitions) for
  generalizable geometric response laws (L2 developmental-tangent split; L1 state
  preservation), then spec each as a differentiable flow regularizer. UCE/species-latent
  route is CLOSED - mine geometry in expression + encoder-agnostic latent. An
  expression-space prior must attach in the raw-expression trainer (no latent->gene decoder).

De-prioritized/closed: flow-matching endpoint tuning; UCE/species-latent; Track-C
support-only GPU (manuscript polish only). Default model stays xverse_8k_anchor.

## Read First

```text
README.md
AGENTS.md
local_goal.md
local_audit.md
local_suggestion.md
docs/WORKSPACE_ORGANIZATION.md
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
