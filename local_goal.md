# local_goal.md

Updated: 2026-07-02.

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

Current research priority is insight-first. Scaling-law discovery and
zebrafish/ZSCAPE dynamic-response biology have priority over direct
flow-matching endpoint metric chasing, because their purpose is to reveal the
training information and biological constraints that should shape LatentFM.
LatentFM architecture audit remains an important parallel engineering thread,
but architecture findings are not permission to train or promote a new default
model without a separate gate.

Active long-horizon acceptance target for the current scaling line:

- reveal whether effective-state / information axes such as Vendi `N_eff`,
  effective rank, participation ratio, pair-mode diversity, `G_eff`, or
  `N_eff x G_eff` explain perturbation-model performance better than raw cell
  count or condition count;
- define scaling `x` as effective perturbation-training information rather
  than raw cell count: state coverage, cluster/centroid coverage,
  condition/pair-mode diversity, transition coverage, gene-token/HVG
  information, and information density are all candidate axes;
- use leakage-safe train-only per-arm geometry, not collapsed dataset-level
  means;
- pass held-out/confound-aware checks such as LODO or source-held-out fit,
  dataset/abundance controls, and fail-close power floors before claiming any
  scaling law;
- preserve a negative or underpowered result as useful evidence rather than
  promoting a model, checkpoint, or scaling claim.

The `Exact Next Task` below is only the current materialization stage toward
that final scientific target. Completing materialization or a preflight report
does not complete the scaling line. Remote Codex should end the stage with
validated inputs for the next regression/analysis stage, a precise subroute
`DATA_BLOCKED` report, or a route pivot recorded in `remote_decision.md`; if the
final target is not achieved and no hard boundary is hit, it should continue
with the next safe scaling, zebrafish, or architecture audit route.

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
- user interrupts manually.

`LOCAL_AUDIT_REQUEST` is a soft audit marker, not a stop condition. Remote
Codex may write it in RUN_STATUS/reports/`remote_decision.md` to help future
local audit, but should not mark the long goal blocked unless all reasonable
next routes require changing the final target, resource boundary, data source,
held-out/query permission, destructive operation, or other user-owned decision.

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
- Scaling is an active insight direction, but the plain per-train-condition
  mean geometry sub-route is now CLOSED as an underpowered negative (see
  `local_audit.md` 2026-07-02). The 2026-07-01 materialization prerequisite is
  REPAIRED (17/17 arms, 11737 train-condition vectors, dim 384, keys
  `dataset::condition`), but the CPU rerun returned `none_no_scaling_law_claim`:
  no geometry/information axis beat condition count under LODO/confound control.
  Decisive evidence that this is a POWER/CONFOUND floor, not an axis-selection
  failure: condition count is itself the strongest full-table correlate
  (spearman ~0.60, p~0.01 vs `family_mmd_delta`/`tail_score`) yet its own LODO
  out-of-sample R2 is negative; every geometry/information axis sits at partial
  rho ~|0.4| with negative LODO OOS R2. On 17 dataset-level rows no axis,
  including the baseline, can leave-one-dataset-out generalize.
- The next scientific blocker is therefore NOT another axis on the same 17 rows.
  It is (a) whether a genuinely new mechanism axis (explicit train-only OT
  pair-mode diversity, cluster/centroid coverage, gene-token/HVG information) is
  even recoverable from existing artifacts, and (b) whether a power-adequate,
  confound-controlled scaling DESIGN exists (more arms/rows, or within-dataset
  budget sweeps that vary the axis while holding dataset/source fixed). Do NOT
  re-run the condition-mean geometry regression on the current 17 rows.
- Zebrafish/ZSCAPE remains an insight source. The narrow dynamic-law regularizer
  mining run was negative: no validated differentiable regularizer should be
  launched from that coverage. Any future zebrafish task must broaden the
  discovery question or verify new coverage before proposing model constraints.
- Zebrafish is treated as rare perturbation time-series ground truth. Future
  work should look for reproducible perturbation dynamic laws in expression and
  latent spaces before proposing regularizers: distribution shifts across time,
  multi-timepoint OT pseudo-tracking, target/marker gene propagation,
  GRN/pathway cascade structure, and latent geometry/direction.
- Closed or de-prioritized routes: UCE/species-latent zebrafish route,
  flow-matching endpoint tuning as the main priority, Track-C query use without
  a new frozen support-val/no-harm protocol, and GPU scaling replays from the
  current evidence.

## Local-To-Remote Operating Model

1. Local CC/Codex reads the current documents and any relevant server report
   indexes, audits contradictions and risk, then writes the next remote task
   package into `local_goal.md`, `local_audit.md`, and `local_suggestion.md`.
2. Local CC/Codex commits/pushes these files. The user makes remote Codex pull.
3. Remote Codex reads this packet plus `goal.md`, `remote_decision.md`, and
   `docs/START_HERE.md`.
4. Remote Codex starts from and prioritizes the concrete `Exact Next Task`.
   Further autonomous stages must stay inside the final goal, resource limits,
   and long-run operating rule; remote must not infer extra experiments from
   archived handoffs or broad research vision text.
5. Remote Codex records execution results in run/report/status outputs, not in
   the three `local_*.md` files.
6. When a subroute is blocked or after a stage completes, remote Codex reports
   structured results, writes the decision/pivot to `remote_decision.md`, and
   continues with the next safe route unless a hard boundary is reached.

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
- `remote_decision.md`
- `docs/START_HERE.md`
- `docs/PROJECT_OVERVIEW.md`
- `docs/PROJECT_REVIEW.md`
- `docs/EXPERIMENT_INDEX.md`
- `docs/DECISIONS.md`
- `docs/RESULTS_SUMMARY.md`
- `docs/LATENTFM_ARCHITECTURE_AUDIT_20260701.md` if model architecture,
  regularizer attachment, or metric defects are relevant.
- `docs/literature/SCALING_ZSCAPE_SQUIDIFF_NOTES_20260701.md` and
  `ref/zebrafish_dataset.pdf` if zebrafish/ZSCAPE dynamic analysis is relevant
  and the paths exist on the remote.

For any run-specific task, also read the relevant `runs/<run>/RUN_STATUS.md`,
`reports/<report>/...`, and source script paths named in `local_audit.md` or
`local_suggestion.md`.

## Exact Next Task

Date: 2026-07-02

Task name: `next_route_audit_20260702` — CPU-only route-selection audit after the
condition-mean geometry negative. It decides which of three tracks becomes the
next execution stage: (A) a power-adequate + new-mechanism scaling redesign,
(B) broader zebrafish/ZSCAPE discovery, or (C) architecture-hygiene planning.
This supersedes the completed `scaling_condition_mean_materialization_20260701`
task; do NOT re-run the condition-mean geometry regression.

Why a route audit and not a new regression: local audit verified the 17-row
condition-mean geometry route is an underpowered negative. Condition count is the
strongest full-table axis yet fails leave-one-dataset-out; every geometry/
information axis has negative LODO OOS R2. Swapping in a new axis on the same 17
rows repeats the error. Before spending any regression/analysis run, the remote
must establish (A) whether a genuinely new axis is recoverable AND whether a
higher-power design exists, and in parallel scope the two other prioritized
tracks the user cares about (zebrafish discovery, architecture hygiene). User
priority is insight-first: scaling and zebrafish outrank flow-matching endpoint
tuning; architecture hygiene runs in parallel as low-risk engineering.

Prior-stage results confirmed by local audit (build on, do not redo):

- materialization REPAIRED: 17/17 arms, 11737 train-condition vectors, dim 384,
  NPZ keys `dataset::condition` (inventory
  `reports/scaling_condition_mean_materialization_20260701/materialized_condition_mean_inventory.csv`);
- rerun decision `none_no_scaling_law_claim`, winning axis `none`
  (`.../scaling_rerun_summary.md`);
- failed candidates: `condition_mean_pairwise_l2_mass` vs `cross_pp_delta`
  (partial rho -0.4641, LODO OOS R2 -0.0007, margin -0.0022, pass False);
  `condition_mean_vendi_rbf_effective_count` vs `family_pp_delta` (partial rho
  0.4444, LODO OOS R2 -0.0174, margin -0.0476, pass False);
- decisive power evidence: condition count spearman ~0.60 (p~0.01) but its own
  LODO OOS R2 negative;
- benign anomaly RESOLVED locally: run-start HEAD `90fd97a` is an ancestor of
  current `54e9520` (forward docs commits after run start), not an integrity
  issue;
- `budget64` true-cell rows were mis-pathed to a `budget128` artifact in
  `split_information_metrics.csv`; remote's `AUTONOMOUS_DECISION` corrected via
  exact manifest split-name match. This correction must be persisted in a
  tracked doc this round (sub-goal C0).

Sub-goals (ALL CPU-only, read-mostly; no GPU/training/inference/checkpoint; no
held-out Track-C query/canonical-multi; never overwrite `*_pert_means.npz`):

Track A — scaling salvageability (primary):

- A1. Formally close the condition-mean geometry route: write a failure-map entry
  summarizing the LODO/power evidence above (negative evidence to PRESERVE, not a
  route to rerun).
- A2. OT pair-mode / cluster artifact inventory: search existing runs/reports for
  RECOVERABLE train-only OT coupling / pair-mode assignments and cluster/centroid
  assignments (files mapping train cells/conditions to explicit pair modes or
  clusters, with traceable provenance). Read `CoupledFM/model/latent/fm_ot.py` and
  `CoupledFM/model/latent/../utils/data/ot_pairer.py` only as static schema
  context. Classify each candidate as (i) usable train-only assignment with
  traceable mapping, or (ii) collapsed/dataset-level only.
- A3. Power/design feasibility: enumerate how many independent scaling arms/rows
  are actually available beyond 17, and whether a within-dataset budget-sweep
  design (vary the axis while holding dataset/source fixed) can raise effective N
  and control the dataset/abundance confound that sinks LODO today. State the
  minimum arm count for a meaningful LODO/source-held-out test.
- A4. Gene-token/HVG information axis feasibility: check whether HVG masks / gene-
  token importance are computable from existing caches for a gene-information-
  weighted axis, without new dataset-wide processing.

Track B — zebrafish/ZSCAPE discovery readiness (parallel; likely primary pivot if
Track A is power-blocked):

- B1. Inventory ZSCAPE assets on the server: timepoints, lineages, perturbations,
  cell counts, and which are already encodable/loadable on CPU. Read
  `docs/literature/SCALING_ZSCAPE_SQUIDIFF_NOTES_20260701.md` and
  `ref/zebrafish_dataset.pdf` if present (remote-only, untracked; summarize, do
  not add to Git).
- B2. Define the FIRST broader-discovery CPU smoke that avoids repeating the
  rejected narrow regularizer-mining coverage: a macro distribution-dynamics
  analysis across timepoints (e-distance / mean-shift trajectories) with
  wrong-time, wrong-lineage, and permutation nulls, and/or a multi-timepoint OT
  pseudo-tracking scaffold (one cell/prototype per stage). Specify inputs,
  controls, and the fail-close rule. Do NOT launch a differentiable regularizer.

Track C — architecture-hygiene localization (parallel; planning only, NO code
edits this round):

- C0. Persist provenance corrections into a tracked doc (e.g. `docs/DECISIONS.md`
  or `docs/EXPERIMENT_INDEX.md`): the `budget64`->`budget128` split-name
  correction and the `/data/cyx/1030/runs` -> `/data/cyx/1030/scLatent/runs`
  old-root drift, so future audits do not re-trip them.
- C1. Localize and write a concrete fix PLAN (do NOT edit model/training code this
  round) for the two confirmed metric-only defects:
  - P4 eval velocity-MSE random pairing (`CoupledFM/model/latent/train.py:3500-3501`):
    eval permutes src/gt independently instead of OT-pairing, so `test_mse` is not
    comparable to `train_mse` and biases model selection; plan the R1 fix to
    OT-pair src/gt in eval, mirroring `OTPrefetchIter` / `ot_pairer.py`;
  - P1 train/eval estimator mismatch (`train.py:2941,2958,2969,3003` single Euler
    step vs 20-step eval): plan the R2 alignment (short `ode_integrate_diff` for the
    aux endpoint estimator).
  - For each, write the exact no-harm gate: headline ODE-MMD/Pearson (pairing-free)
    unchanged; only the diagnostic metric changes; default model stays
    `xverse_8k_anchor`. APPLYING these fixes needs an explicit user greenlight in
    the goal-start message (code edits are gated); this round is localization +
    plan + gate only.
  - Note the biggest structural gap for the user's "regularize in expression AND
    latent space" plan: the latent trainer has NO latent->expression decoder
    (emb_dim 2058, latent-only); an expression-space prior must attach in
    `CoupledFM/model/train.py` (~1643-1644) or requires a frozen decoder. Record
    this as a constraint, not a task.

Decision output (REQUIRED):

- write `reports/next_route_audit_20260702/NEXT_AXIS_DECISION.md` recommending
  EXACTLY ONE next execution stage with justification tied to the long-horizon
  acceptance target, chosen from:
  - `SCALING_REDESIGN`: Track A found BOTH a recoverable new train-only axis AND a
    power-adequate/confound-controlled design -> next stage materializes+tests that
    axis with LODO/source-held-out controls;
  - `ZEBRAFISH_DISCOVERY`: Track A is power-blocked or axis-blocked -> promote the
    Track B first discovery smoke as the next execution stage;
  - `ARCH_HYGIENE`: only if the user greenlights code edits -> apply the R1
    metric-only fix under its no-harm gate as the next stage;
  - `DATA_BLOCKED`: name the exact missing artifact and the minimal materialize
    task if no track is runnable.
- append a dated `AUTONOMOUS_DECISION` entry to `remote_decision.md` recording this
  route audit AND back-filling the un-logged
  `scaling_condition_mean_materialization_20260701` result.

Allowed inputs: the first-read files; the materialization run/report dir; split
JSONs; existing OT/coupling/cluster artifacts if discoverable;
`CoupledFM/model/latent/fm_ot.py`, `utils/data/ot_pairer.py`, `train.py` (static
read for C1 localization only); ZSCAPE assets and literature/ref (read-only).

Expected outputs:

- `runs/next_route_audit_20260702/RUN_STATUS.md`;
- `reports/next_route_audit_20260702/OT_PAIRMODE_CLUSTER_INVENTORY.md` (+ a
  `pairmode_artifact_inventory.csv`);
- `reports/next_route_audit_20260702/SCALING_POWER_FEASIBILITY.md`;
- `reports/next_route_audit_20260702/ZSCAPE_ASSET_INVENTORY.md`;
- `reports/next_route_audit_20260702/ARCH_HYGIENE_FIX_PLAN.md`;
- `reports/next_route_audit_20260702/NEXT_AXIS_DECISION.md`;
- `remote_decision.md` appended entry; provenance-correction note in a tracked doc
  (C0).

DONE criteria:

- report branch/HEAD/dirty state and confirm no `local_*.md` edits;
- condition-mean geometry route explicitly closed as preserved negative;
- OT pair-mode/cluster artifacts classified usable vs collapsed with exact paths;
- scaling power-feasibility states available arm count and the minimum for a
  meaningful LODO test;
- ZSCAPE asset inventory + one fail-closed first discovery smoke spec;
- P4/P1 localized with exact file:line, an R1/R2 fix plan, and no-harm gates;
- `NEXT_AXIS_DECISION.md` picks exactly one next stage;
- provenance corrections persisted to a tracked doc.

Conditional next-stage authorization (remote may proceed WITHOUT a new local round
IF its own DONE gate holds):

- if `ZEBRAFISH_DISCOVERY`: run the single specified CPU distribution-dynamics /
  OT-pseudo-tracking smoke with wrong-time/wrong-lineage/permutation nulls;
  promotion requires the regularity to survive ALL nulls in expression space (and
  an encoder-agnostic latent view if used); a surviving result is discovery
  evidence only, NOT a regularizer launch;
- if `SCALING_REDESIGN`: materialize the one recovered train-only axis into a
  fresh run-scoped dir (never overwrite NPZs) and rerun the CPU gate ONLY with the
  power-adequate design; promotion requires the axis to beat BOTH raw cell count
  AND condition count under LODO/source-held-out with confound controls; otherwise
  preserve as negative;
- if `ARCH_HYGIENE` AND the user greenlit code edits: apply ONLY the R1 metric-only
  fix; no-harm gate = headline ODE-MMD/Pearson unchanged, default model unchanged;
- any negative/ambiguous/blocked outcome -> log honestly in `remote_decision.md`
  and request local audit rather than sweeping variants.

Resource limits:

- CPU only; no GPU, no training, no inference, no checkpoint selection, no
  held-out Track-C query/canonical-multi use.
- Writes only under the two `next_route_audit_20260702/` dirs, the authorized
  `remote_decision.md` append, and the single C0 tracked-doc provenance note.
- Never overwrite `*_pert_means.npz` or any prior artifact.
- Target 2 hours; hard stop 3 hours with a partial report.
- No dataset-wide processing beyond existing caches; if any sub-goal needs a large
  recompute, mark it `DATA_BLOCKED` and continue with the others.

Forbidden actions:

- do not edit `local_goal.md`, `local_audit.md`, or `local_suggestion.md`;
- do not reopen Track-C support-only GPU work, UCE/species-latent, or the narrow
  ZSCAPE regularizer route;
- do not fabricate per-condition geometry from dataset-level vectors;
- do not claim a scaling law from collapsed or underpowered data;
- do not edit model/training code this round (Track C is planning only) unless the
  user explicitly greenlights the R1 metric-only fix.

Stop rules:

- soft-block a single track and continue the others if its inputs are missing;
  record `SOFT_BLOCK`/`ROUTE_PIVOT` in `remote_decision.md`;
- mark hard `BLOCKED` only if every track needs GPU/large data processing beyond
  the resource limit, held-out/query permission, new data source, destructive
  operations, or final-goal changes.

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

Remote Codex must hard-stop only when:

- the exact task is missing, ambiguous, or contradicted by current docs;
- required data/report/run artifacts are absent and all safe reconstruction or
  alternative audit routes are exhausted;
- a result would require changing final claim scope rather than selecting a new
  bounded research route;
- resource use would exceed the task limits or require GPU/network/data access
  not authorized in the exact task;
- every useful next route needs code changes outside the approved scope;
- there is evidence of leakage, split mismatch, overwritten outputs, or
  duplicated/generated artifacts entering Git.

## Expected Remote Output Shape

At stage completion, soft block, hard block, or achievement, remote Codex should
report:

- files read;
- commands run and whether they were CPU/GPU/network/disk-heavy;
- changed files and generated server-local outputs;
- key metrics, gates, negative controls, and anomalies;
- whether the task is positive, negative, blocked, or ambiguous;
- recommended updates to `local_goal.md`, `local_audit.md`, and
  `local_suggestion.md` for the next local audit, without editing those files.
Remote should also append the key decision to `remote_decision.md` in Chinese.
