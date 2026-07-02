# local_suggestion.md

Updated: 2026-07-02.

This file is part of the local-authored remote execution packet. Remote Codex
must read it with `local_goal.md` and `local_audit.md`; it gives priorities,
fallback routes, gates, and the decision tree for interpreting execution
results.

Remote Codex must not edit this file during execution. If a suggestion becomes
wrong, incomplete, or softly blocked, remote Codex should report that in
RUN_STATUS/reports/`remote_decision.md`, choose a new safe route when possible,
and recommend changes for the next local audit. Local CC/Codex updates this
file and pushes it.

The concrete remote task still lives in `local_goal.md` under `Exact Next
Task`; this file explains how to execute and interpret that task.

## Suggestion Generation Rules

Each suggested remote task must be:

- one bounded goal, not a bundle of speculative experiments;
- grounded in current docs, run status, and report provenance;
- explicit about data/split boundaries and forbidden inputs;
- CPU-only unless a fresh GPU resource audit is part of the task;
- measurable by predeclared metrics, negative controls, and fail-close rules;
- safe to stop with a useful local-audit request if artifacts are missing;
- clear about which files may be edited and which outputs are server-local.

Do not suggest tasks that reopen closed branches merely because archived notes
contain older handoff language.

## Third-Round Priorities & Strategy Portfolio - 2026-07-02 (GOVERNING)

This supersedes the priority order below. See `local_goal.md` Exact Next Task for the
full multi-stage spec. Governing priorities:

1. Program Z (zebrafish/ZSCAPE dynamics) — HIGH #1.
2. Program S (scaling two-level formula) — HIGH.
3. Program B (scFM reconstruction benchmark) — IMPORTANT.
4. Program M (Pearson-control metric audit) — LOW / quick parallel.
LatentFM end-to-end endpoint modeling — LOW / paused (trainings for `scaling_y` and a
validated Program Z regularizer are still allowed). Target quality: Nature Methods / NBT.

Gates are CLAIM-BARS, not stop conditions — gate-fail => log + switch strategy, never
`BLOCKED`:

- Program S: claim a law only if `f`/`g` predict `scaling_y` OUT-OF-SAMPLE better than
  raw cell count AND condition count under source-held-out/LODO + confound controls;
  report `control_info`/`perturb_info`/`OT_pair_info` definitions + provenance.
- Program Z: any dynamic law must survive wrong-time / wrong-lineage / permutation nulls
  in expression space (+ encoder-agnostic latent view); no regularizer launch without
  the no-harm gate.
- Program B: fair cross-scFM decoder reconstruction with a random/mean baseline; per-gene
  + global error (MSE/Pearson/EV), HVG-restricted.
- Program M: `PEARSON_CTRL_VERDICT` in {METRIC_ARTIFACT (correct + unblock), REAL_TRADEOFF
  (pick a direction-preserving objective), INCONCLUSIVE}.

Strategy portfolio (parallel directions so goal mode never runs dry — pick / interleave /
spawn subagents; a wall on one => move to another, never stop):

- Program S: latent-space info (primary) vs gene/HVG original-space (cross-check);
  cluster-based info via multiple clusterings (kmeans / leiden / GMM), effective-cluster
  counts, entropy, Vendi `N_eff`; `OT_pair_info` via #pair-modes, cost dispersion,
  direction diversity, Sinkhorn vs Hungarian; within-dataset down-sampling curves; set-set
  similarity via pseudobulk-delta cosine / MMD / overlap-condition Jaccard.
- Program Z: macro (e-distance / MMD / mean-shift trajectories) + individual (OT
  pseudo-tracking, branching, velocity, delay) + expression (target/marker propagation,
  GRN via CellOracle / GEARS, pathway cascade/enrichment, pathway-pathway association
  graphs) + latent (geometry / direction / curvature; expression-vs-latent agreement).
- Program B: Tabula Sapiens fidelity (primary) + perturb-OOD reconstruction; linear vs MLP
  decoder; full-gene vs HVG.
- Modeling (low-pri, only via science): direction-preserving / no-harm objectives if the
  Pearson tradeoff is real; conditioning upgrades (FiLM / cross-attention / CFG);
  curvature / straightness regularizer from Program Z; OT-cost robustification.

Decision tree (all branches KEEP ADVANCING; only user-owned decisions => hard block):

- positive => harden, add controls, write up toward publication; pick the next program.
- negative / ambiguous => preserve evidence in `remote_decision.md`, switch to the next
  portfolio item or the other program; spawn a subagent to re-audit the route.
- "data insufficient" => download / augment more data (authorized), NOT a block.
- gate-fail => log + new strategy, NOT `BLOCKED`.

## Current Priority Order

Updated 2026-07-02 after the condition-mean geometry negative. The immediate next
remote task is the CPU-only `next_route_audit_20260702` (see `local_goal.md`),
which picks the next execution stage among the tracks below.

1. Scaling REDESIGN (not a rerun): recover explicit train-only OT pair-mode /
   cluster-centroid / gene-token-HVG assignments AND establish a power-adequate,
   confound-controlled design (more arms, or within-dataset budget sweeps).
   Rationale: the 17-row condition-mean geometry route is a closed underpowered
   negative — even condition count fails LODO — so the fix is a new axis PLUS
   power, not another axis on the same rows. The plain per-condition mean geometry
   regression is DEMOTED and must not be rerun.
2. Broader zebrafish/ZSCAPE discovery: raised toward primary pivot because the
   scaling line is power-blocked. Must introduce a broader lens than the rejected
   narrow regularizer-mining run: macro distribution dynamics, multi-timepoint OT
   pseudo-tracking, GRN/pathway cascade, latent geometry/direction — all under
   wrong-time/wrong-lineage/permutation nulls. Discovery only; no regularizer
   launch until controls pass.
3. LatentFM architecture hygiene (parallel, planning-first): localize the two
   confirmed metric-only defects (P4 eval velocity-MSE random pairing at
   `train.py:3500-3501`; P1 estimator mismatch at `train.py:2941,2958,2969,3003`)
   and write the R1/R2 fix plan + no-harm gate. APPLYING even the near-zero-risk R1
   fix requires explicit user approval (code edits are gated).
4. Manuscript/reviewer-package polish, only if the user wants packaging with no new
   experiment.
5. Chemical V2 or any GPU branch only after exact ACK, resource audit, split
   boundary, written hypothesis, stop rule, and `RUN_STATUS.md`.

## Second-Round Guidance - 2026-07-02

The condition-mean geometry route is a closed underpowered negative (condition
count itself fails LODO; all axes have negative LODO OOS R2). Do not rerun it. The
next remote task `next_route_audit_20260702` is a CPU-only route-selection audit;
its `NEXT_AXIS_DECISION.md` must pick exactly one next execution stage:

- `SCALING_REDESIGN` only if BOTH a recoverable train-only new axis (OT pair-mode /
  cluster / gene-token-HVG) AND a power-adequate confound-controlled design are
  found. Promotion of any scaling axis then requires it to beat BOTH raw cell count
  AND condition count under LODO/source-held-out with confound controls; otherwise
  preserve as negative.
- `ZEBRAFISH_DISCOVERY` if scaling is power/axis-blocked. First smoke is macro
  distribution dynamics and/or multi-timepoint OT pseudo-tracking with
  wrong-time/wrong-lineage/permutation nulls; a regularity is discovery evidence
  only and must survive ALL nulls before any regularizer is even proposed.
- `ARCH_HYGIENE` only if the user greenlights code edits; then apply ONLY the R1
  metric-only eval-pairing fix under its no-harm gate (headline ODE-MMD/Pearson
  unchanged, default model `xverse_8k_anchor` unchanged).
- `DATA_BLOCKED` with the exact missing artifact if no track is runnable.

Standing gates this round: no scaling-law claim without beating cell/condition
count under held-out/confound control; no zebrafish regularizer without passing all
nulls; no code edit or model promotion without an explicit gate; never overwrite
`*_pert_means.npz`; persist the budget64/128 and old-root provenance corrections to
a tracked doc.

## Candidate Task Templates

### A. Scaling Per-Arm Geometry CPU Task

Note (2026-07-02): the plain per-condition MEAN geometry version of this task is
CLOSED as an underpowered negative. The live variant is OT pair-mode / cluster /
gene-token-HVG axes PLUS a power-adequate design; see `Second-Round Guidance` above
and `next_route_audit_20260702` in `local_goal.md`.

Hypothesis: effective-state / information axes explain performance better than raw
cell count once a RECOVERABLE new axis is materialized AND the design has enough
independent arms to leave-one-dataset-out generalize.

Minimum task:

- find the existing per-arm inputs without reading held-out query data;
- before materialization, preflight split train condition keys, observed
  condition/name columns, source matrix path, train-only mask, and the mapping
  from split rows to planned NPZ keys;
- materialize per-arm rows for Vendi `N_eff`, effective rank, participation
  ratio, cluster/centroid coverage, pair-mode diversity if available,
  gene-token/HVG information, information density, and abundance/response-
  energy weighted `G_eff`;
- write all new NPZs to a fresh run-scoped artifact directory; never overwrite
  existing `*_pert_means.npz`;
- write a report explaining which arms were materialized, which were missing,
  and why;
- only rerun regression if the per-arm table passes completeness/provenance
  checks.

Promotion gate:

- per-arm table covers the predeclared true-cell/scaling arms;
- no collapsed-parent geometry remains in the joined regression table;
- information axis improves out-of-sample or leave-one-dataset-out fit versus
  raw cell count and condition count;
- abundance/source/dataset confounds are controlled or reported as blockers.
- next-stage schema explicitly says which scaling `x` axes are meaningful
  perturbation-training information and which are abundance proxies.

Fail-close:

- if inputs are missing or collapse cannot be repaired, output a subroute
  `DATA_BLOCKED`/soft-block report with exact missing paths, do not claim a
  scaling law, record the pivot in `remote_decision.md`, and continue with a
  safe alternative insight route if available.

### B. Manuscript Package Polish CPU Task

Hypothesis: current negative and scaling-axis evidence is ready for reviewer
interpretation without more experiments.

Promotion gate:

- manifest validates;
- referenced paths exist;
- claims stay within "scaling-axis/failure-map, no-harm-gated, negative
  evidence preserved";
- no wording says promoted checkpoint, solved Track-C query, or monotonic
  scaling law.

Fail-close:

- if provenance is incomplete or figures/scripts do not validate, report the
  exact missing artifacts and stop.

### C. Broader Zebrafish Discovery CPU Task

Hypothesis: broader time-series/distribution/pseudo-single-cell tracking or
GRN/pathway-cascade analysis can find a reproducible biological regularity,
even though the narrow dynamic-law regularizer run was negative.

Preferred lenses:

- macro distribution dynamics across timepoints, perturbations, lineages, and
  cell types;
- multi-timepoint OT pseudo-tracking, with one cell/prototype sampled per
  stage to form synthetic trajectories;
- expression-space target/marker propagation, optional CellOracle/GEARS-style
  GRN checks, and pathway cascade/enrichment relationships;
- latent-space geometry, direction, branching, delay, and curvature.

Promotion gate:

- regularity survives wrong-time, wrong-lineage, permutation, abundance/support,
  and coverage controls;
- result is reproducible in expression space and, if used, an encoder-agnostic
  latent view;
- proposed regularizer has a clear attachment point and no claim is made before
  validation.

Fail-close:

- if controls fail, preserve the negative as biological insight and do not
  launch a model regularizer.

### D. Architecture Hygiene Planning

Hypothesis: a known metric/protocol defect can be corrected without changing
scientific claim scope.

Possible first targets from the audit:

- eval velocity-MSE OT pairing caveat;
- aux one-step endpoint estimator versus multi-step eval mismatch;
- condition-dropout/CFG or gradient-conflict handling only after gates are
  specified.

Promotion gate:

- small tests pass;
- old claims are not rewritten;
- default model remains `xverse_8k_anchor` unless a separate strict no-harm
  gate passes.

Fail-close:

- if fixing it changes metric comparability or requires training, stop and ask
  for local audit.

## Metrics And Gates

Every suggestion must state at least one primary metric and one control.

Allowed metric families:

- LatentFM: ODE-MMD, Pearson perturbation (`pp`), Pearson cell (`pc`), family
  gene/drug metrics, split-group metrics, bootstrap/CI, tail/no-harm rows.
- Scaling: cell count, condition count, Vendi `N_eff`, effective rank,
  participation ratio, Kish/state entropy, pair-mode diversity, `G_eff`,
  `N_eff x G_eff`, LODO or source-held-out regression fit, confound controls.
- Zebrafish: time/lineage/direction consistency, wrong-time and wrong-lineage
  nulls, permutation nulls, abundance/support residualization, expression-vs-
  latent agreement, coverage and pairability checks.
- Manuscript: manifest validity, path existence, figure QA, reproduction-script
  count, claim-scope lint.

Hard gates:

- no hard fail when a no-hard-fail rule is predeclared;
- no held-out query use without fresh authorization;
- no model promotion without canonical no-harm and promotion gates;
- no scaling-law claim unless the information axis beats raw cell count under
  held-out/confound-controlled evaluation;
- no zebrafish regularizer launch unless dynamic law controls pass.

## Decision Tree After Remote Results

Positive:

- verify outputs and paths;
- update `docs/DECISIONS.md`, `docs/EXPERIMENT_INDEX.md`, and the relevant
  project review summary if the user authorizes documentation updates;
- decide whether the next step is confirmatory validation, manuscript polish,
  or a stricter gate;
- do not promote defaults until strict promotion/no-harm criteria pass.

Negative:

- preserve the negative evidence;
- close or narrow the branch in the next local audit;
- update suggested claim language to avoid overreach;
- choose a different mechanism only if it is not a cosmetic variant of the
  failed branch.

Blocked:

- require a `remote_decision.md` entry with exact missing files, commands
  attempted, partial outputs, and at least three possible next directions;
- do not let remote Codex invent a replacement heavy experiment, but do allow a
  safe replacement audit/diagnostic route inside the same final goal;
- local audit later decides whether to repair prerequisites, switch tasks, or
  close the route.

Ambiguous:

- ask for the smallest CPU-only disambiguation check;
- prefer provenance and negative controls over more training;
- if ambiguity touches split/query leakage, stop that unsafe subroute, record
  it in `remote_decision.md`, and pivot to a safe route that does not use the
  ambiguous split/query data.

## Remote Prompt Snippet

Use only after `local_goal.md` has a filled `Exact Next Task`:

```text
Read `goal.md`, `local_goal.md`, `local_audit.md`, `local_suggestion.md`, and
`docs/START_HERE.md` first. Execute only the filled `Exact Next Task`; do not
infer extra experiments from archived legacy handoffs. Respect resource limits,
forbidden actions, and hard stop rules. If a subroute is blocked, write
`remote_decision.md` with files read, commands run, changed/generated paths,
metrics, anomalies, suspected bottlenecks, and suggested updates to the three
local docs, then continue with a safe route if available.
Do not edit the three local docs on the remote side.
```
