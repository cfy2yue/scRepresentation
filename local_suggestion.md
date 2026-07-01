# local_suggestion.md

Updated: 2026-07-01.

This file is part of the local-authored remote execution packet. Remote Codex
must read it with `local_goal.md` and `local_audit.md`; it gives priorities,
fallback routes, gates, and the decision tree for interpreting execution
results.

Remote Codex must not edit this file during execution. If a suggestion becomes
wrong, incomplete, or blocked, remote Codex should report that and recommend
changes for the next local audit. Local CC/Codex updates this file and pushes
it.

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

## Current Priority Order

1. Scaling per-arm geometry materialization, if the user wants the next
   scientific insight step. This unblocks fair regression of effective
   information axes, information density, and mode coverage against model
   performance.
2. Broader zebrafish discovery audit, because ZSCAPE time-series perturbation
   ground truth can reveal dynamic biological constraints for LatentFM before
   any new regularizer is launched. The task must introduce new coverage or a
   broader analysis lens beyond the rejected narrow regularizer-mining run.
3. LatentFM architecture audit in parallel with the insight tracks, especially
   bottlenecks suggested by scaling or zebrafish results; code edits or model
   promotion still require a separate explicit gate.
4. Manuscript/reviewer-package polish, if the user wants publication packaging
   with no new experiment.
5. Architecture hygiene planning or a tiny metric-only code fix proposal, only
   after local audit separates it from the insight tracks and defines no-harm
   gates. Code edits require explicit user approval.
6. Chemical V2 or any GPU branch only after exact ACK, resource audit, split
   boundary, written hypothesis, stop rule, and `RUN_STATUS.md`.

## Candidate Task Templates

### A. Scaling Per-Arm Geometry CPU Task

Hypothesis: effective-state / information axes explain performance better than
raw cell count once per-arm geometry is materialized.

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

- if inputs are missing or collapse cannot be repaired, output a blocked report
  with exact missing paths and do not claim a scaling law.

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

- require a `LOCAL_AUDIT_REQUEST` with exact missing files, commands attempted,
  partial outputs, and at least three possible next directions;
- do not let remote Codex invent a replacement heavy experiment;
- local audit decides whether to repair prerequisites, switch tasks, or close
  the route.

Ambiguous:

- ask for the smallest CPU-only disambiguation check;
- prefer provenance and negative controls over more training;
- if ambiguity touches split/query leakage, stop as blocked.

## Remote Prompt Snippet

Use only after `local_goal.md` has a filled `Exact Next Task`:

```text
Read `goal.md`, `local_goal.md`, `local_audit.md`, `local_suggestion.md`, and
`docs/START_HERE.md` first. Execute only the filled `Exact Next Task`; do not
infer extra experiments from archived legacy handoffs. Respect resource limits,
forbidden actions, and stop rules. If blocked, output `LOCAL_AUDIT_REQUEST`
with files read, commands run, changed/generated paths, metrics, anomalies,
suspected bottlenecks, and suggested updates to the three local docs.
Do not edit the three local docs on the remote side.
```
