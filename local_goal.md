# local_goal.md

Updated: 2026-07-01.

This is the standing local-audit packet for scLatent / scRepresentation. It is
not a remote execution goal by itself. Remote Codex has an active goal only
after local CC/Codex has completed a fresh audit and filled the `Exact Next
Task` section below with a concrete task, outputs, limits, and stop rules.

## Durable Final Goal

Maintain `/data/cyx/1030/scLatent` as the reproducible monorepo entrypoint for:

- LatentFM latent-space perturbation prediction and no-harm model audits;
- scFMBench representation benchmarking and figure/metric infrastructure;
- CoupledFM-derived model code used by LatentFM;
- scaling-axis / failure-map analysis for single-cell perturbation data;
- zebrafish / ZSCAPE dynamic-response analysis when it constrains the method;
- project-owned scripts, reports, run status files, prompts, and docs.

The scientific acceptance target is a leakage-safe, publication-ready workflow:
strong benchmark evidence, clearly bounded LatentFM claims, preserved negative
evidence, and no model/scaling claim without strict gates.

## Current Direction And Boundaries

- Current default/deployable LatentFM state remains `xverse_8k_anchor` until a
  newer strict gate supersedes it.
- Track-C support-only GPU work is CLOSED. Its 2026-07-01 result was 2/3 seeds
  passing but seed45 hard-failing the predeclared no-hard-fail rule; do not
  reopen it as a support-only GPU branch.
- The CPU-only manuscript package for the Track-C closure and scaling-axis /
  failure-map evidence has been verified. Further work here is narrative,
  reviewer-package, or provenance polish unless a new hypothesis is audited.
- Scaling is an active insight direction, but not a deployable scaling-law
  claim. The current next scientific blocker is per-arm geometry materialization
  for information/effective-state axes: Vendi `N_eff`, effective rank,
  participation ratio, pair-mode diversity, and abundance/response-energy
  weighted `G_eff`.
- Zebrafish/ZSCAPE remains an insight source. The narrow dynamic-law regularizer
  mining run was negative: no validated differentiable regularizer should be
  launched from that coverage. Any future zebrafish task must broaden the
  discovery question or verify new coverage before proposing model constraints.
- Closed or de-prioritized routes: UCE/species-latent zebrafish route,
  flow-matching endpoint tuning as the main priority, Track-C query use without
  a new frozen support-val/no-harm protocol, and GPU scaling replays from the
  current evidence.

## Local-To-Remote Operating Model

1. Local CC/Codex reads the current documents and any relevant server report
   indexes, audits contradictions and risk, then updates `local_goal.md`,
   `local_audit.md`, and `local_suggestion.md`.
2. The user manually syncs/pulls the remote repository and starts remote Codex.
3. Remote Codex reads this packet plus `goal.md` and `docs/START_HERE.md`.
4. Remote Codex executes only the concrete `Exact Next Task`; it does not infer
   extra experiments from archived handoffs or broad research vision text.
5. When blocked or after completion, remote Codex reports structured results so
   the next local audit can decide the next exact task.

Archive note: files under `docs/archive/legacy_auto_coordination_20260701/` and
`prompts/archive/legacy_auto_coordination_20260701/` are historical evidence
only. They are not active instructions.

## Files Remote Codex Must Read First

Remote Codex must read these before acting:

- `README.md`
- `AGENTS.md`
- `goal.md`
- `local_goal.md`
- `local_audit.md`
- `local_suggestion.md`
- `docs/START_HERE.md`
- `docs/PROJECT_OVERVIEW.md`
- `docs/PROJECT_REVIEW.md`
- `docs/EXPERIMENT_INDEX.md`
- `docs/DECISIONS.md`
- `docs/RESULTS_SUMMARY.md`
- `docs/LATENTFM_ARCHITECTURE_AUDIT_20260701.md` if model architecture,
  regularizer attachment, or metric defects are relevant.

For any run-specific task, also read the relevant `runs/<run>/RUN_STATUS.md`,
`reports/<report>/...`, and source script paths named in `local_audit.md` or
`local_suggestion.md`.

## Exact Next Task

Status: NOT ACTIVE.

Local audit must replace this block before the user starts remote execution.
Until this block is filled with a single bounded task, this document is only a
standing packet and remote Codex must wait.

Required fields for the next remote task:

- Goal:
- Why now:
- Inputs and files to read:
- Allowed commands:
- Expected outputs:
- DONE criteria:
- Resource limits:
- Forbidden actions:
- Stop rules:

## Default Resource Rules

- Prefer CPU-only audit/report work unless the exact task explicitly authorizes
  GPU use.
- No long training, inference, scaling replay, query evaluation, or dataset-wide
  processing without a fresh resource audit, written hypothesis, leakage
  boundary, expected outputs, promotion gate, fail-close rule, and
  `runs/<run>/RUN_STATUS.md`.
- Large data stay under `/data/cyx/1030/dataset`; project outputs stay under
  `/data/cyx/1030/scLatent`.
- Generated `runs/`, `reports/`, logs, checkpoints, pretrained weights, caches,
  venvs, secrets, and local archives must not be added to Git.
- Use detached execution and status files for any long remote job.

## Forbidden Actions

- Do not use archived legacy handoffs as active instructions.
- Do not reopen Track-C support-only GPU work or the UCE/species-latent route.
- Do not claim a promoted model, scaling law, or validated zebrafish regularizer
  without a predeclared gate that passes.
- Do not read or use held-out Track-C query/canonical-multi data unless a fresh
  local audit explicitly authorizes the exact protocol.
- Do not move/delete datasets, checkpoints, reports, runs, logs, local archives,
  caches, secrets, or other project roots.
- Do not commit, push, reset, force-push, or clean files unless the user
  explicitly asks for that Git operation.

## Stop Rules

Remote Codex must stop and output a structured local-audit request when:

- the exact task is missing, ambiguous, or contradicted by current docs;
- required data/report/run artifacts are absent or provenance does not match;
- a gate hard-fails, metrics are missing, or a result would require changing
  claim scope;
- resource use would exceed the task limits or require GPU/network/data access
  not authorized in the exact task;
- the task needs code changes outside the approved scope;
- there is evidence of leakage, split mismatch, overwritten outputs, or
  duplicated/generated artifacts entering Git.

## Expected Remote Output Shape

At completion or block, remote Codex should report:

- files read;
- commands run and whether they were CPU/GPU/network/disk-heavy;
- changed files and generated server-local outputs;
- key metrics, gates, negative controls, and anomalies;
- whether the task is positive, negative, blocked, or ambiguous;
- recommended updates to `local_goal.md`, `local_audit.md`, and
  `local_suggestion.md` for the next local audit.
