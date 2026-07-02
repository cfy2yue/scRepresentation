# local_goal.md

Updated: 2026-07-02 (third local round — post Pearson-control no-harm block).

This is the local-authored remote execution packet for scLatent /
scRepresentation. Remote Codex reads this file, `local_audit.md`, and
`local_suggestion.md` as the authoritative task package when the user starts a
remote goal.

Remote code baseline this round: HEAD `6b4c591` (local clean, in sync with
GitHub). The remote working tree carries valuable UNCOMMITTED overnight work in
`CoupledFM/model/latent/train.py`, `config.py`, several tests, and docs. Already
LANDED in that tree and confirmed by local head-verify: (1) R1 / P4 eval-pairing
repair — `evaluate()` now OT/assignment-pairs src/gt instead of independent
random permutation; (2) CUDA AMP fix — `_amp_autocast_ctx` at `train.py:61-71`,
`config.use_amp=True`, `amp_dtype="bf16"`; (3) a differentiable aux-endpoint ODE
path, DEFAULT-OFF (`config.aux_endpoint_ode_steps=0`). Do NOT pull/reset/clean
this tree; it is real, in-boundary work to be preserved as evidence.

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

Current research priority (2026-07-02, refined by user) is insight-first and
SCIENCE-PROGRAM-driven, at Nature Methods / Nature Biotechnology research quality
and acceptance threshold. The TWO HIGH-PRIORITY programs are:

- (Z) the zebrafish / ZSCAPE dynamic-response discovery program — the #1 goal:
  find the laws and propose the insight — how cells actually change across the
  perturbation time-course, how pathways cascade and react, what the pathway-to-
  pathway associations are, and exactly how the latent space changes;
- (S) the systematic scaling / information-density program built around the user's
  explicit TWO-LEVEL FORMULA (single-dataset `scaling_singleset`, then a
  cross-dataset law; see Program S) — define what "information" a single-cell /
  perturb-seq dataset carries and how it drives model performance.

A THIRD important benchmark piece is (B) the scFM RECONSTRUCTION benchmark
(embedding -> expression), previously missed. These three are run to mature-methods
depth (systematic, reproducible, theory-grounded, null-controlled) and are the
publishable core.

LatentFM end-to-end MODELING is DEPRIORITIZED this round (LOW priority; paused as a
standalone endpoint-improvement goal). IMPORTANT nuance: LatentFM TRAININGS remain
AUTHORIZED where they serve the science — Program S needs MULTIPLE LatentFM
trainings to measure the performance outcome `scaling_y`, and a validated Program Z
law may be tested as a regularizer. So "pause LatentFM modeling" means do not pour
overnight GPU into blind endpoint / MMD-weight tuning; it does NOT forbid the
trainings Program S / Z require. The Pearson-control metric audit (Program M) is now
a QUICK, LOW-priority parallel check (not the gating front-end): it tells us whether
the overnight modeling block was real, but must not delay the two HIGH-priority
science programs.

Data-scaling is no longer treated as permanently power-blocked. The 17
outcome-linked arms are a data-supply problem, not a scientific dead end:
DOWNLOADING / EXPANDING more perturb-seq datasets to lift the arm count is
AUTHORIZED this round (see Program S below). GPU training toward the final goal
is authorized within the project budget after a fresh low-util audit. Only
science-grounded safety gates remain.

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

## Top-Priority Operating Principles (this round — READ FIRST)

These override any softer language below and correct last night's failure mode.

1. ANTI-STOP / OWN AGENCY. The remote's first-order objective is to keep
   ADVANCING the final goal. Hitting a claim gate, a route underperforming, a
   current strategy not passing, or thin data are NOT `blocked`. When that
   happens the remote must exercise agency: think, spawn read-only/code-review
   subagents to audit and re-plan the route, switch strategy / sub-problem /
   generate a new hypothesis, and CONTINUE — logging each decision in
   `remote_decision.md`. Do NOT bounce back to local for help on a gate. Last
   night the remote hit the Pearson-control gate and STOPPED to request a local
   audit (`LOCAL_AUDIT_REQUEST_PEARSON_CONTROL_DIFFUSE_HARM`), wasting overnight
   GPU. That specific behavior is the thing to eliminate.
2. HARD `BLOCKED` is reserved for genuine user-owned decisions only: changing the
   final goal / threshold, new keys or permissions, or destructive operations.
   A negative result, a failing gate, or "current data insufficient" never
   qualifies.
3. NO UNGROUNDED RESOURCE/DATA LIMITS. The scaling sample-size wall (only 17
   outcome-linked arms) is now to be solved by DOWNLOADING / AUGMENTING more
   perturb-seq datasets to raise the arm count — this is AUTHORIZED (network +
   disk under `/data/cyx/1030/dataset`, provenance-tracked). GPU training inside
   the project boundary is AUTHORIZED. Keep only safety gates that have a
   scientific basis.
4. TARGETS AND THRESHOLDS ARE DURABLE; GATES ARE CLAIM-GATES, NOT STOP
   CONDITIONS. The final goal and its acceptance thresholds must not be lowered
   or dropped; local audit changes only route/strategy. A promotion/no-harm gate
   is a bar for CLAIMING success, not a reason to halt. Still-falsified /
   still-closed branches (do NOT reopen): Track-C support-only GPU;
   UCE/species-latent zebrafish; held-out Track-C query / canonical-multi
   selection; promoting a new default model with no passing gate.

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

Gate semantics (collaboration mode): the final goal and its acceptance thresholds
are DURABLE — the remote must not lower or drop them, and local audit changes only
the route, implementation, and strategy, not the target. The routes and gates in
this packet are local SUGGESTIONS/HYPOTHESES, not the only path; the remote is
expected to have its own problem-solving ability. A promotion/claim gate (beating
cell/condition count under held-out; surviving all nulls; no-harm) is a quality bar
for CLAIMING success, NOT a stop condition. If a suggested route or gate does not
work out, do NOT declare `BLOCKED`: log the honest negative in `remote_decision.md`,
find a new in-boundary strategy yourself, and keep advancing across the
scaling/zebrafish/architecture tracks toward the same target — the recorded
decisions are read by the next local audit. Reserve hard `BLOCKED` only for genuine
user-owned decisions (changing the final target/threshold/resource
boundary/permission/data source, or a destructive operation).

## Current Direction And Boundaries

- Current default/deployable LatentFM state remains `xverse_8k_anchor` until a
  newer strict gate supersedes it.
- LANDED remote code this round (uncommitted, preserved): R1 / P4 eval-pairing
  repair in `evaluate()`; CUDA AMP fix (`_amp_autocast_ctx`); default-off
  differentiable aux-endpoint ODE path (`aux_endpoint_ode_steps=0`). These are
  evidence, not a promotion.
- P1 endpoint aux-ODE4 (and `lowweight`) IMPROVED distribution metrics but FAILED
  the Pearson-control no-harm gate; R1 Pearson-repair also not promotable; P2
  harm-localization was negative (harm is diffuse, AUC 0.605 < 0.70). None of
  these promote a model. The endpoint / MMD-weight tuning micro-route is
  EXHAUSTED as a blind path — do NOT keep re-tuning endpoint or MMD weight. The
  next move is the Pearson-control metric-validity audit (Program M) and the two
  science programs, NOT another endpoint smoke.
- Track-C support-only GPU work is CLOSED. Its 2026-07-01 result was 2/3 seeds
  passing but seed45 hard-failing the predeclared no-hard-fail rule; do not
  reopen it as a support-only GPU branch.
- The CPU-only manuscript package for the Track-C closure and scaling-axis /
  failure-map evidence has been verified. Further work here is narrative,
  reviewer-package, or provenance polish unless a new hypothesis is audited.
- Scaling is a PRIMARY multi-step science program (Program S below), no longer a
  closed dead end. The plain per-train-condition MEAN geometry sub-route stays
  CLOSED as an underpowered negative (preserve, do not rerun on the same rows):
  materialization was REPAIRED (17/17 arms, 11737 train-condition vectors, dim
  384, keys `dataset::condition`) but the CPU rerun returned
  `none_no_scaling_law_claim`, and even condition count — the strongest
  full-table correlate (spearman ~0.60, p~0.01) — has NEGATIVE LODO OOS R2; every
  geometry/information axis sits at partial rho ~|0.4| with negative LODO OOS R2.
  On 17 dataset-level rows nothing, including the baseline, leaves-one-dataset-out
  generalizes.
- The correct resolution of that power wall is DATA, not another axis on 17 rows:
  download / augment more perturb-seq datasets to raise the outcome-linked arm
  count well above 17, then define and validate information-density axes with
  source-held-out / LODO / confound controls. Do NOT re-run the condition-mean
  geometry regression on the current 17 rows; do build the larger corpus.
- Zebrafish/ZSCAPE is a PRIMARY multi-step science program (Program Z below), run
  to mature-methods discovery depth, THEN used to inform/regularize LatentFM. The
  earlier narrow dynamic-law regularizer-mining run was negative: no validated
  differentiable regularizer may be launched from that coverage. Any zebrafish
  work must run the broader systematic discovery (macro distribution dynamics +
  individual multi-timepoint OT pseudo-tracking + expression/latent regularities)
  with wrong-time / wrong-lineage / permutation nulls before proposing model
  constraints, and only claim/launch a regularizer through the no-harm gate.
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

Date: 2026-07-02 (third round). This is an OVERNIGHT AUTONOMOUS, MULTI-STAGE goal,
not a single short job. It has one front-end audit (Program M) and two long-run
science programs (Program S = scaling; Program Z = zebrafish) that the remote
advances with its own route-finding and subagents until the acceptance target is
met, a genuine hard blocker appears, or the user interrupts. Hitting a gate,
producing a negative, or "current data insufficient" are NOT stop conditions —
log the decision in `remote_decision.md`, switch strategy / spawn a subagent, and
continue. Do NOT re-run the closed condition-mean geometry regression, and do NOT
keep re-tuning endpoint / MMD weight (that micro-route is exhausted).

Prior-stage results confirmed by local head-verify (build on, do not redo):

- materialization REPAIRED: 17/17 arms, 11737 train-condition vectors, dim 384,
  NPZ keys `dataset::condition`; condition-mean geometry rerun =
  `none_no_scaling_law_claim`; even condition count has NEGATIVE LODO OOS R2 (the
  17-row table is power/confound-limited — preserved negative, do not rerun).
- LANDED remote code (uncommitted): R1 / P4 eval-pairing repair in `evaluate()`;
  CUDA AMP fix (`_amp_autocast_ctx`, `train.py:61-71`); default-off differentiable
  aux-endpoint ODE path (`config.aux_endpoint_ode_steps=0`).
- P1 aux-ODE4 / lowweight IMPROVED distribution metrics but FAILED Pearson-control
  no-harm (see `local_audit.md` for exact numbers); R1 Pearson-repair not
  promotable; P2 harm-localization negative (harm diffuse, AUC 0.605 < 0.70).

### Program M (QUICK, LOW-PRIORITY parallel audit): Pearson-control metric-validity audit

Goal: decide whether the `pearson_ctrl` no-harm gate that stopped the overnight
run is a REAL direction-preserving no-harm gate or a metric / reference-frame
ARTIFACT. This is cheap (CPU, read/recompute over existing posthoc rows and the
`evaluate()` Pearson code path). It is LOW priority this round (LatentFM end-to-end
modeling is deprioritized) but worth doing in PARALLEL because it tells us whether
the overnight block was spurious; it must NOT delay Programs Z and S.

- M1. Read the Pearson computation in `CoupledFM/model/latent/train.py`
  `evaluate()` (the per-condition -> per-dataset -> overall Pearson pipeline) and
  `eval_condition_residuals.py` / `eval_background_groups.py`. Extract EXACTLY:
  what vectors `pearson_ctrl` correlates (predicted-vs-control? delta-vs-control?
  which reference frame), whether values are normalized / standardized before
  correlation, and how conditions are aggregated (mean-of-per-condition vs pooled).
- M2. Check the three suspected artifact sources and state pass/fail for each:
  (i) NORMALIZATION — is `pearson_ctrl` computed on a scale where an MMD-improving
  shift of the whole distribution can mechanically lower a control-referenced
  correlation? (ii) REFERENCE FRAME — is "control" the right anchor, and is the
  same frame used for anchor vs candidate so the delta is fair? (iii) CONDITION
  AGGREGATION — does mean-over-condition with unequal n / low-condition datasets
  (Schmidt, Jiang_INS, Wessels) dominate the overall number?
- M3. Recompute a DIRECTION-PRESERVING no-harm diagnostic on the SAME closed-branch
  posthoc rows (no training): e.g. cosine / sign-agreement of predicted vs true
  perturbation DELTA (pert - ctrl), or Pearson on delta vectors, per condition,
  anchor vs `lowweight`/`auxode4`. Compare its verdict to `pearson_ctrl`'s.
- M4. Decision `PEARSON_CTRL_VERDICT` in a report + `remote_decision.md`:
  - `METRIC_ARTIFACT`: the gate is an evaluation-framing artifact -> the overnight
    stop was spurious; specify the corrected no-harm metric and UNBLOCK the P1/aux
    direction under the corrected gate (may resume bounded GPU training on that
    direction). Persist the metric correction to a tracked doc.
  - `REAL_TRADEOFF`: the gate is real (MMD gain genuinely costs direction/no-harm)
    -> do NOT blindly re-tune endpoint/MMD weight; pick a DIRECTION-PRESERVING /
    no-harm training objective (e.g. delta-consistency-regularized) as the next
    modeling route, or pivot to Program S/Z. Either way, keep advancing.
  - `INCONCLUSIVE`: run one more cheap disambiguation, then choose the more likely
    branch and continue (do NOT stop for local audit).

### Program S (HIGH PRIORITY, multi-stage science): systematic scaling / information law

Aim: a QUANTITATIVE, REPRODUCIBLE, THEORY-GROUNDED definition of single-cell /
perturb-seq "training information", validated against the model-performance outcome
`scaling_y`, at Nature Methods / NBT depth. The user's explicit TWO-LEVEL hypothesis
drives the design — implement and test it specifically:

LEVEL 1 — single-dataset `scaling_singleset`:
`scaling_y ≈ f( control_info + perturb_info + OT_pair_info )`, where
- `control_info` = information content of the control-cell population, CLUSTER-based
  (e.g. #effective clusters / cluster coverage / entropy over control-cell clusters);
- `perturb_info` = information content of the perturbed-cell populations, CLUSTER-based
  (same family of measures over treated cells / conditions);
- `OT_pair_info` = the per-condition control->treated OT pairing structure AVERAGED
  over conditions, PLUS pair diversity/heterogeneity (how many distinct pair modes,
  dispersion of pair costs / directions).
FIT by DOWN-SAMPLING conditions and samples WITHIN a single dataset: a within-dataset
sweep yields many `(info, scaling_y)` points while holding dataset/source FIXED — this
is how the power wall is beaten at level 1. `scaling_y` = a LatentFM performance
outcome trained on each down-sampled subset (MULTIPLE LatentFM trainings AUTHORIZED
here). Fit and report `f` per dataset.

LEVEL 2 — cross-dataset scaling law:
`scaling_y_crossset ≈ g( {scaling_singleset}, set-set association )`, where set-to-set
association may depend on: the NUMBER OF OVERLAPPING conditions between datasets, and
the PERTURBATION-EFFECT SIMILARITY between datasets. Candidate construction for the
latter: pseudobulk per condition, optional binning / regularization, and the
perturbed-minus-control DELTA, then a similarity over these. Fit `g` across datasets.

SPACE FOR THE INFO ANALYSIS: compute the information measures PRIMARILY in the LATENT
space (LatentFM / scFM embeddings already inferred — cheaper, and coherent with
`scaling_y` being a LatentFM outcome; likely fits the law better), with a GENE / HVG
original-space CROSS-CHECK for the gene-token/HVG-weighted axis. Report both where
feasible and state which space the law fits better.

- S-Stage 1 — corpus & within-dataset sweep design: pick datasets with enough
  conditions/cells for a within-dataset down-sampling sweep; download / augment more
  perturb-seq datasets (AUTHORIZED; provenance-tracked under `/data/cyx/1030/dataset`)
  to widen level-2 coverage. Fresh run-scoped artifacts; never overwrite
  `*_pert_means.npz`.
- S-Stage 2 — measure `control_info`, `perturb_info`, `OT_pair_info` per subset in
  LATENT space (+ HVG original-space cross-check); precise reproducible definitions and
  provenance.
- S-Stage 3 — train LatentFM on each down-sampled subset to obtain `scaling_y` (GPU
  AUTHORIZED); fit level-1 `f`; then fit level-2 `g` with the set-set association terms,
  under source-held-out / LODO and confound controls, with fail-close power floors.
- S-Stage 4 — claim a scaling law ONLY if `f` / `g` predict `scaling_y` OUT-OF-SAMPLE
  (held-out datasets/subsets) better than raw cell count AND condition count; else
  preserve as a stronger, better-powered negative. Translate the validated law into
  "what data / information to prioritize for LatentFM training".

### Program B (IMPORTANT benchmark, was missed): scFM reconstruction benchmark

Aim: evaluate each scFM's RECONSTRUCTION ability (embedding -> original gene-
expression vector) — a previously-missing but important part of the scFMBench
comparison. Use EXISTING already-inferred embeddings (no re-inference needed).

Two scenarios (run both; recommendation on the primary comparability metric):
- Scenario B-tabula (PRIMARY, cleaner cross-scFM comparability): train a
  reconstruction decoder (embedding -> expression) on a subset of Tabula Sapiens that
  already has inferred embeddings (the same subset used for de-batch testing), then
  measure HELD-OUT reconstruction fidelity. Fair, standard information-retention /
  invertibility benchmark, directly comparable across scFMs.
- Scenario B-ood (SECONDARY, perturbation-relevant): train the decoder on control +
  a sampled set of conditions, then reconstruct expression on NEW (held-out, OOD-like)
  conditions. Tests perturbation-relevant generalization of the embedding — more
  aligned with LatentFM's use case.

Recommendation: run BOTH (compute allows); lead with B-tabula as the primary
cross-scFM reconstruction-fidelity number, report B-ood as the OOD-generalization
companion. Metrics: per-gene and global reconstruction error (MSE / Pearson /
explained variance), HVG-restricted variants, plus a random / mean-baseline control.
Rank scFMs by how much reconstructable expression information their embeddings retain.

### Program Z (HIGH PRIORITY #1, multi-stage science): zebrafish / ZSCAPE dynamics

Aim: systematically discover GENERALIZABLE perturbation dynamic-response laws to
mature-methods depth, ALL with wrong-time / wrong-lineage / permutation nulls,
THEN use them to inform / regularize LatentFM (only claim/launch a regularizer if
it passes the no-harm gate). Not endpoint-metric chasing — discovery/insight first.

- Z-Stage 1 — ASSET INVENTORY. Inventory ZSCAPE assets on the server (timepoints,
  lineages, perturbations, cell counts, CPU-loadable encodings). Read
  `docs/literature/SCALING_ZSCAPE_SQUIDIFF_NOTES_20260701.md` and
  `ref/zebrafish_dataset.pdf` if present (remote-only, untracked; summarize, do
  NOT add to Git).
- Z-Stage 2 — MACRO DISTRIBUTION DYNAMICS. Distribution-level dynamics across
  timepoints: e-distance / mean-shift trajectories per perturbation × lineage,
  with wrong-time, wrong-lineage, and permutation nulls and a fail-close rule.
  Broaden beyond the rejected narrow regularizer-mining coverage.
- Z-Stage 3 — INDIVIDUAL OT PSEUDO-TRACKING. Multi-timepoint OT pseudo-tracking
  (one cell / prototype per stage) to build synthetic single-cell trajectories;
  characterize consistency, delay, branching under the same nulls.
- Z-Stage 4 — EXPRESSION & LATENT REGULARITIES. Target/marker propagation,
  optional GRN (CellOracle / GEARS) / pathway cascade, and latent-space geometry /
  direction / curvature. Every claim null-controlled and reproducible in expression
  space plus an encoder-agnostic latent view.
- Z-Stage 5 — FEEDBACK TO MODELING. If a regularity survives ALL nulls, propose a
  differentiable regularizer at the exact attachment point (velocity /
  straightness / curvature / OT-coupling; expression-space priors need the
  `CoupledFM/model/train.py` gene-space path or a frozen decoder — the latent
  trainer has NO latent->expression decoder), TRAIN a variant (GPU authorized), and
  compare to `xverse_8k_anchor` under the no-harm gate. A surviving law with no
  trained gain is still preserved as discovery evidence.

### Execution order, autonomy, and continuation

1. HIGH PRIORITY, run in parallel as the publishable core: Program Z (zebrafish
   dynamics / insight, #1) and Program S (scaling two-level formula). The remote
   chooses which to push based on tractability and evidence; it may interleave, spawn
   subagents, and pre-explore the next stage. Each is a multi-stage long-run job with
   a detached RUN_STATUS.
2. IMPORTANT, schedule alongside: Program B (scFM reconstruction benchmark) — uses
   existing inferred embeddings, largely independent, good to run in parallel.
3. LOW PRIORITY / quick: Program M (Pearson-control metric audit) — cheap parallel
   check; must not delay Z/S. LatentFM end-to-end endpoint modeling stays PAUSED
   except the trainings that Program S (`scaling_y`) and a validated Program Z
   regularizer require.
4. GPU is AUTHORIZED for NEW work toward the final goal after a fresh low-util audit
   (Program S LatentFM trainings for `scaling_y`, Program Z-Stage 5 regularizer test,
   Program B decoder training, Program M resume-if-artifact). Data
   download/augmentation is AUTHORIZED for Program S-Stage 1.
5. A negative / ambiguous outcome on one program -> log honestly in
   `remote_decision.md` and push the next best in-boundary program (Z <-> S <-> B).
   Return to local audit ONLY when all reasonable in-boundary routes are exhausted or
   a genuine user-owned blocker appears. Do NOT stop merely because a stage completed
   or a gate failed.

Experiment-driven exploration: remote GPU is abundant. Do NOT force the full CPU
analysis to be exhausted before any GPU work. If a program's analysis is
inconclusive or there is no clearly better CPU path, the remote is encouraged to
RUN a bounded, informative GPU experiment (a LatentFM variant with a candidate
information axis, a direction-preserving-objective variant, a zebrafish-informed
regularizer, or a diagnostic run) and let results guide the next plan. Safety
still holds: no reopening closed branches, no held-out Track-C query, and no
scaling-law / regularizer / new-default-model CLAIM without its gate.

Required outputs (each program writes its own dir; do not overwrite prior
artifacts):

- `runs/<program>_<date>/RUN_STATUS.md` for each long-run program;
- Program M: `reports/pearson_control_metric_audit_<date>/PEARSON_CTRL_VERDICT.md`
  (+ recompute CSVs) and, if `METRIC_ARTIFACT`, a tracked-doc metric correction;
- Program S: dataset-expansion inventory + provenance, fresh per-arm information
  NPZs, and a regression/validation report with the LODO/confound results;
- Program Z: `ZSCAPE_ASSET_INVENTORY.md` and per-stage discovery reports with
  null-control tables;
- an appended `AUTONOMOUS_DECISION` / verdict entry in `remote_decision.md` per
  stage, and back-fill the un-logged
  `scaling_condition_mean_materialization_20260701` result if still missing.

Resource limits:

- Program M is CPU-only. Programs S/Z may use GPU up to the project budget after a
  fresh low-util audit, and Program S-Stage 1 may use network + disk under
  `/data/cyx/1030/dataset` for dataset download/augmentation. Provenance-track all
  downloads.
- Still forbidden regardless of program: held-out Track-C query/canonical-multi;
  reopening the closed Track-C support-only or UCE/species-latent branches;
  promoting a default model without a strict no-harm/promotion gate; overwriting
  `*_pert_means.npz` or any prior artifact.
- Long-run programs use detached execution with a RUN_STATUS; do not stop merely
  because the audit or a stage completes.

Forbidden actions:

- do not edit `local_goal.md`, `local_audit.md`, or `local_suggestion.md`;
- do not reopen Track-C support-only GPU work, UCE/species-latent, or the narrow
  ZSCAPE regularizer route;
- do not fabricate per-condition geometry from dataset-level vectors;
- do not claim a scaling law from collapsed or underpowered data;
- do not keep re-tuning endpoint / MMD weight as a blind route.

Stop rules:

- soft-block a single program/stage and continue the others if its inputs are
  missing; record `SOFT_BLOCK` / `ROUTE_PIVOT` in `remote_decision.md`;
- mark hard `BLOCKED` only for genuine user-owned decisions: changing the final
  goal / threshold / resource boundary / permission / data source, or a
  destructive operation. A failing gate or a negative result is NOT `BLOCKED`.

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
