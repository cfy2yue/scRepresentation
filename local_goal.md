# local_goal.md

Updated: 2026-07-01.

This is the local-authored remote execution packet for scLatent /
scRepresentation. Remote Codex reads this file, `local_audit.md`, and
`local_suggestion.md` as the authoritative task package when the user starts a
remote goal.

Remote Codex must not edit these three `local_*.md` files during execution.
They are updated only by local CC/Codex audit and then pushed to GitHub for the
remote to pull.

This packet becomes executable only after local CC/Codex fills the `Exact Next
Task` section below with a concrete task, outputs, limits, and stop rules. If
that section is not filled, remote Codex waits rather than inventing work.

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

Active long-horizon acceptance target for the current scaling line:

- reveal whether effective-state / information axes such as Vendi `N_eff`,
  effective rank, participation ratio, pair-mode diversity, `G_eff`, or
  `N_eff x G_eff` explain perturbation-model performance better than raw cell
  count or condition count;
- use leakage-safe train-only per-arm geometry, not collapsed dataset-level
  means;
- pass held-out/confound-aware checks such as LODO or source-held-out fit,
  dataset/abundance controls, and fail-close power floors before claiming any
  scaling law;
- preserve a negative or underpowered result as useful evidence rather than
  promoting a model, checkpoint, or scaling claim.

The `Exact Next Task` below is only the current materialization stage toward
that final scientific target. Completing materialization or a preflight report
does not complete the scaling line. Remote Codex must end the stage with either
validated inputs for the next regression/analysis stage, a precise
`DATA_BLOCKED` report, or a local-audit decision request tied to the
long-horizon acceptance target.

## Remote Long-Run Operating Rule

When the user starts remote goal mode, remote Codex should treat the durable
final goal and active long-horizon acceptance target as the objective. `Exact
Next Task` is the current priority stage and starting direction, not a
short-job completion condition.

The current route is a hypothesis under the same final goal, not a step in a
step-to-step decomposition. If the route fails, underperforms, or becomes
unpromising, the right action is to record the evidence, audit/optimize the
route, and continue toward the same final target or request local route
optimization. Do not mark the goal complete merely because the current stage is
done.

Remote Codex should keep progressing until one of these happens:

- `ACHIEVED`: the long-horizon acceptance target is actually met with evidence,
  metrics, controls, output paths, and claim boundaries recorded;
- `BLOCKED`: a hard blocker requires changing final target, resource boundary,
  data source, held-out/query permission, destructive operation, or other
  user-owned decision;
- `LOCAL_AUDIT_REQUEST`: repeated negative/ambiguous results, suspected bug or
  split issue, or route drift makes local strategy audit the right next step;
- user interrupts manually.

Within the written resource and safety boundaries, remote Codex may make
`AUTONOMOUS_DECISION` route choices, add lightweight controls, run bounded
diagnostics, and pre-explore the next stage after the current stage completes.
It may also launch remote subagents for independent read-only/code-review style
audits when available. Subagents must read the same `goal.md` and
`local_*.md` files, stay inside the project/resource boundaries, avoid editing
the three `local_*.md` files, and summarize their evidence in RUN_STATUS or the
report before the main remote agent continues.

Remote Codex should not stop merely because a preflight, materialization, or
rerun summary is written. If the final target is not achieved and there is no
hard block, it should record the result, choose the next bounded stage inside
the same final goal, and continue or clearly explain why local audit is
required.

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
   indexes, audits contradictions and risk, then writes the next remote task
   package into `local_goal.md`, `local_audit.md`, and `local_suggestion.md`.
2. Local CC/Codex commits/pushes these files. The user makes remote Codex pull.
3. Remote Codex reads this packet plus `goal.md` and `docs/START_HERE.md`.
4. Remote Codex starts from and prioritizes the concrete `Exact Next Task`.
   Further autonomous stages must stay inside the final goal, resource limits,
   and long-run operating rule; remote must not infer extra experiments from
   archived handoffs or broad research vision text.
5. Remote Codex records execution results in run/report/status outputs, not in
   the three `local_*.md` files.
6. When blocked or after completion, remote Codex reports structured results and
   suggested changes so the next local audit can update the three local files.

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

Date: 2026-07-01

Goal: CPU-only materialization preflight for leakage-safe per-train-condition
geometry, then materialize the smallest valid per-condition mean artifacts if
the preflight passes.

Why now: two remote CPU audits found the same hard blocker. All 17 scaling rows
have referenced `*_pert_means.npz` artifacts, but those NPZs are dataset-label
means, not train-condition vectors (`0/17` rows have condition-level vectors).
The scaling-law question cannot be answered fairly until this prerequisite is
repaired. Local audit chooses the first repair route rather than asking the
user to decide.

Inputs and files to read:

- the first-read files listed above;
- `runs/scaling_unit_cpu_regression_20260701/RUN_STATUS.md`;
- `reports/scaling_unit_regression_20260701/scaling_unit_decision.md`;
- `runs/scaling_perarm_regression_20260701/RUN_STATUS.md`;
- `reports/scaling_perarm_regression_20260701/scaling_perarm_decision.md`;
- `reports/multiaxis_information_scaling_incremental_gate_20260629/multiaxis_information_scaling_join_rows.csv`;
- `reports/downstream_information_scaling_preflight_20260628/split_information_metrics.csv`;
- split JSON files and referenced dataset/cache artifacts named by those CSVs;
- existing scripts:
  - `ops/analyze_scaling_unit_regression_20260701.py`;
  - `ops/analyze_scaling_perarm_regression_20260701.py`.

Allowed commands:

- read-only path/provenance checks;
- CPU-only Python preflight to identify where train-condition source matrices
  can be loaded from;
- if and only if the preflight proves source paths are complete, create one
  small materializer script such as
  `ops/materialize_scaling_condition_means_20260701.py` or extend an existing
  untracked ops script;
- run a two-arm CPU-only materializer/preflight smoke first; if successful,
  materialize all 17 rows within the limits below;
- rerun only the prerequisite audit portion or a clearly labeled CPU regression
  rerun after materialization, without changing model/checkpoint code.

Expected outputs:

- `runs/scaling_condition_mean_materialization_20260701/RUN_STATUS.md`;
- `reports/scaling_condition_mean_materialization_20260701/PREFLIGHT.md`;
- `reports/scaling_condition_mean_materialization_20260701/materialized_condition_mean_inventory.csv`;
- materialized NPZ outputs under a generated server-local artifact directory
  named in the report, not added to Git. The directory must be new and
  run-scoped, for example under
  `runs/scaling_condition_mean_materialization_20260701/artifacts/`; never
  overwrite existing `*_pert_means.npz`;
- if rerun is valid:
  - `reports/scaling_condition_mean_materialization_20260701/scaling_rerun_summary.md`.

DONE criteria:

- report branch/HEAD/dirty state and confirm no `local_*.md` edits;
- prove whether each of 17 rows has loadable train-condition source data;
- for every materialized artifact, record split name, arm, source path,
  condition count, vector dimension, train-only boundary, and checksum/size;
- demonstrate that materialized keys are train-condition keys/names rather than
  dataset-label keys;
- preflight must record split train condition keys, observed condition/name
  columns, source matrix path, train-only mask, and the exact mapping from split
  rows to materialized NPZ keys;
- if materialization cannot be done safely, output `DATA_BLOCKED` with exact
  missing paths and do not rerun regression;
- if materialization succeeds, rerun the CPU gate and decide whether any
  information/geometry axis should be tested further, while preserving the
  no-scaling-law claim unless gates actually pass.

Resource limits:

- CPU only; no GPU, no training, no inference, no checkpoint selection, no
  held-out Track-C query/canonical-multi use.
- Target 2 hours; hard stop 3 hours with a partial report.
- Start with 2 arms; stop if a single arm requires more than 20 minutes,
  unbounded memory, or dataset-wide processing beyond existing caches.
- Keep generated artifacts server-local and out of Git.

Forbidden actions:

- do not edit `local_goal.md`, `local_audit.md`, or `local_suggestion.md`;
- do not reopen Track-C support-only GPU work, UCE/species-latent, or the
  narrow ZSCAPE regularizer route;
- do not fabricate per-condition geometry from dataset-level vectors;
- do not claim a scaling law from collapsed or underpowered data.

Stop rules:

- stop and output `LOCAL_AUDIT_REQUEST` if source matrices/splits cannot be
  found, if train-only boundaries are ambiguous, if outputs would overwrite
  prior artifacts, or if the repair needs GPU/large data processing beyond the
  resource limit.

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
- Do not edit `local_goal.md`, `local_audit.md`, or `local_suggestion.md`
  during remote execution; propose updates in the final/block report instead.
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
  `local_suggestion.md` for the next local audit, without editing those files.
