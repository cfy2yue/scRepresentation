# local_audit.md

Updated: 2026-07-02.

This file is part of the local-authored remote execution packet. Remote Codex
must read it before executing `local_goal.md`; it provides the evidence,
negative results, risks, and context behind the next task.

Remote Codex must not edit this file during execution. If execution reveals new
evidence, blockers, or corrections, remote Codex should write them in
RUN_STATUS/reports/final output and suggest updates for the next local audit.
Local CC/Codex then updates this file and pushes it.

## Audit Scope

Local audit owns strategy, document consistency, risk review, and authoring the
next bounded remote task package. Remote Codex owns execution of that package.
Local audit may read Git-tracked docs/source and report indexes and may run
small grep/static checks. It should not run large experiments, launch GPU work,
edit production code, or treat archived handoffs as active instructions unless
the user explicitly asks for such work.

## Data Flow To Understand

- Canonical project root: `/data/cyx/1030/scLatent`.
- Shared data root: `/data/cyx/1030/dataset`.
- Runtime entry: `source /data/cyx/1030/scLatent/init-scdfm.sh`.
- LatentFM training-ready data live under dataset-side `latentfm_full` and
  related latent/scFM artifacts; old docs may mention pre-migration paths.
- scFMBench outputs and figures live under scLatent-owned output roots; large
  generated outputs stay server-local and out of Git.
- Run provenance is in `runs/<run>/RUN_STATUS.md`; human-readable evidence is
  in `reports/`; Git-tracked docs should only summarize high-signal current
  conclusions.

## Model And Evaluation Flow To Understand

- Maintained LatentFM model: `ControlMLPVelocityField` in
  `CoupledFM/model/latent/models/mlp.py`.
- Main latent trainer: `CoupledFM/model/latent/train.py`.
- Latent flow: precomputed latent source/control to GT perturbation, CondOT
  linear interpolation, velocity target `x1 - x0`, and Euler ODE evaluation.
- Perturbation conditioning is additive through gene/drug condition embeddings;
  this is a known capacity bottleneck for unseen/combinatorial cases.
- OT pairing is used in training; evaluation headline ODE-MMD/Pearson metrics
  are pairing-free, while eval velocity-MSE has known pairing caveats.
- Relevant eval files include `eval_split_groups.py`,
  `eval_condition_families.py`, `eval_background_groups.py`, and
  `eval_condition_residuals.py`.
- Raw-expression trainer `CoupledFM/model/train.py` matters if a future
  expression-space regularizer is proposed, because the latent trainer has no
  latent-to-expression decoder.

## Current Evidence State

- Default model: `xverse_8k_anchor` remains current until superseded by a strict
  no-harm/promotion gate.
- scFMBench: benchmark infrastructure and figure/metric layer are usable for
  the current phase; NicheFormer/TranscriptFormer are limited chempert-only
  evidence unless broader count-compatible embeddings exist.
- Track-C support-only: CLOSED. The 2026-07-01 pair-type support-only gate had
  seed43 and seed44 passing but seed45 hard-failed
  `support_pp_delta_below_0p04`, violating the no-hard-fail rule.
- Track-C query/multi-condition: not solved and not authorized by support-only
  evidence. Any future query route needs a fresh split/no-harm protocol.
- CPU-only manuscript package: verified at
  `reports/trackc_support_only_closed_scaling_manuscript_manifest_20260701/`;
  claim scope is scaling-axis/failure-map plus negative evidence, not model
  promotion.
- Scaling-unit: the per-arm geometry prerequisite is now REPAIRED (17/17 arms,
  11737 train-condition vectors, dim 384), but the condition-mean geometry
  regression came back `none_no_scaling_law_claim`. Decisive finding: even
  condition count (the strongest full-table axis) has negative LODO OOS R2, so
  the 17-row table is an underpowered/confounded negative, not an axis-selection
  problem. The plain condition-mean geometry sub-route is CLOSED as preserved
  negative. The user's OT pair-mode diversity axis remains UNTESTED (no explicit
  train-only pair-mode assignments materialized). See the 2026-07-02 second-round
  update below.
- Zebrafish/ZSCAPE: current regularizer-mining coverage was negative. No
  generalized dynamic-response regularity survived wrong-time/wrong-lineage or
  permutation nulls, so no differentiable flow regularizer is validated from
  that run.
- Architecture audit: known issues include train/eval estimator mismatch,
  eval velocity-MSE random pairing, additive-only conditioning, gradient
  conflict, batch-mean heterogeneity collapse, and linear-path/Euler sensitivity.
  These are audit facts, not permission to edit code.

## User Research Intent - 2026-07-01

The user wants computation plus biological insight to guide and constrain
LatentFM, not simply endpoint metric chasing. Scaling and zebrafish/ZSCAPE are
therefore higher-priority research lines than direct flow-matching optimization
unless a method change is motivated by those insights.

Scaling intent:

- raw cell count is probably too weak as the scaling `x` unit for single-cell
  perturbation data;
- the target is a quantitative notion of effective perturbation-training
  information, information density, or mode coverage;
- candidate views include cluster/centroid coverage, condition+OT pair modes,
  transition diversity, statistical summaries, information-theoretic axes,
  gene-token information, HVG contribution, `G_eff`, and combinations such as
  `N_eff x G_eff`;
- external review of scFM data selection motivates the concern that low
  heterogeneity or repeated cells can make a large raw cell count look like a
  large scale while adding little useful training information;
- the main data regime is perturb-seq, and any scaling claim must separate
  information from abundance/source/dataset confounds.

Zebrafish/ZSCAPE intent:

- zebrafish is valuable because it provides perturbation dynamic transition
  ground truth rather than just endpoint perturbation labels;
- the goal is scientific discovery of generalizable dynamic response laws in
  both expression and latent space, not immediate regularizer launch;
- macro views include distribution statistics across timepoints, e-distance or
  mean-shift trajectories, and time/lineage/perturbation controls;
- pseudo-individual views may use OT across multiple timepoints to construct
  synthetic single-cell tracks, sampling one cell or prototype per stage;
- expression-space views include target genes, marker genes, possible
  CellOracle/GEARS-style GRN propagation, pathway enrichment, and pathway
  cascade relationships;
- latent-space views include geometry, direction, branching, delay, and
  trajectory curvature. These laws may later inform regularization, but only
  after controls pass.

Architecture intent:

- LatentFM architecture should be audited in parallel for bottlenecks and
  optimization space, especially if scaling/ZSCAPE findings suggest constraints
  that the current additive-conditioning or estimator design cannot express;
- code changes, new training, or default-model promotion still require a fresh
  explicit gate.

## Closed Or Restricted Branches

- CLOSED: Track-C support-only GPU branch.
- CLOSED: UCE/species-latent zebrafish route.
- NOT ACTIVE: flow-matching endpoint tuning as the main project direction.
- NOT ACTIVE: scaling replay / scaling-derived checkpoint promotion.
- NOT ACTIVE: chemical V2 or any GPU branch unless exact ACK, resource audit,
  split boundary, hypothesis, stop rule, and `RUN_STATUS.md` are present.
- NOT ALLOWED: claiming first-in-field monotonic scaling law, validated
  zebrafish regularizer, or promoted default model from current evidence.

## Risk Checklist

Local audit must explicitly check these risks before writing `Exact Next Task`:

- Leakage or split mismatch, especially Track-C support/query/canonical-multi.
- Query use before a frozen support-val/no-harm protocol.
- Overclaiming negative or blocked scaling evidence as a solved scaling law.
- Treating archived legacy handoffs as active instructions.
- Old path drift: `/data/cyx/1030/...` versus
  `/data/cyx/1030/scLatent/...` after workspace consolidation.
- Generated outputs entering Git or source docs pointing agents at ignored
  server-local archives as first-read material.
- Missing or collapsed per-arm geometry in scaling-unit work.
- Zebrafish regularizer launch from rejected/narrow coverage.
- Metric caveats from the architecture audit, especially eval MSE pairing and
  aux one-step versus eval multi-step mismatch.
- Output overwrite risk for report builders and status scripts.
- Resource creep from CPU audit/report work into GPU training or dataset-wide
  processing.

## Must-Check Files

Always inspect:

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
- `docs/LATENTFM_ARCHITECTURE_AUDIT_20260701.md`
- `docs/RESEARCH_VISION_20260701.md`
- `docs/GIT_AND_COLLABORATION.md`
- `docs/GITHUB_FILE_MAP.md`
- `docs/WORKSPACE_ORGANIZATION.md`

Check source paths only as needed for the next task:

- `CoupledFM/model/latent/train.py`
- `CoupledFM/model/latent/config.py`
- `CoupledFM/model/latent/models/mlp.py`
- `CoupledFM/model/latent/fm_ot.py`
- `CoupledFM/model/latent/dataset.py`
- `CoupledFM/model/latent/eval_split_groups.py`
- `CoupledFM/model/latent/eval_condition_families.py`
- `CoupledFM/model/latent/eval_background_groups.py`
- `CoupledFM/model/train.py`
- `scFMBench/README.md`
- `scFMBench/STATUS.md`
- `scFMBench/benchmark/docs/metrics_protocol.md`
- `scFMBench/fm/docs/encoder_overview.md`
- relevant `ops/audit_*`, `ops/synthesize_*`, and `ops/validate_*` files
  named by the task or report index.

## Local Checks To Prefer

- `git status --short`
- `rg` for task-specific terms, run names, report names, and claim phrases.
- Static reads of report manifests and JSON/Markdown indexes.
- Small syntax or schema checks only if directly relevant and cheap.

Do not run remote-only scripts locally on Windows unless they are clearly
platform-independent and tiny.

## First-Round Local Audit Update - 2026-07-01

User context: first manual local-audit round for all three projects; no new
remote `LOCAL_AUDIT_REQUEST` was provided. Local audit used SSH read-only
checks on `/data/cyx/1030/scLatent` for server-local scaling and ZSCAPE
summaries.

Remote sync/state:

- Historical snapshot HEAD when this first-round evidence was pulled:
  `0fde3f6`. Current remote execution-packet commit after catchup is
  `e3c79be`.
- Remote dirty state: 5 untracked entries: `docs/literature/`,
  `ops/analyze_scaling_perarm_regression_20260701.py`,
  `ops/analyze_scaling_unit_regression_20260701.py`,
  `ops/mine_zscape_dynamic_regularizer_laws_20260701.py`, and `ref/`.

Remote evidence verified:

- `runs/scaling_unit_cpu_regression_20260701/RUN_STATUS.md` and
  `reports/scaling_unit_regression_20260701/scaling_unit_decision.md`:
  all 17 scaling rows lack usable per-train-condition vectors. Referenced
  `*_pert_means.npz` artifacts contain dataset-label vectors, with
  `condition_name_hits=0` and `condition_key_hits=0`; regression was correctly
  stopped.
- `reports/scaling_unit_regression_20260701/missing_per_arm_artifacts.csv`
  has 17 data rows. Examples: expected 586 or 1582 train-condition vectors for
  early arms but observed only 22 dataset-label keys.
- `runs/scaling_perarm_regression_20260701/RUN_STATUS.md` and
  `reports/scaling_perarm_regression_20260701/scaling_perarm_decision.md`:
  condition-level NPZ rows `0/17`; dataset-mean NPZ geometry rows `17/17`;
  winning scaling-x `none`; pair-mode diversity skipped because condition/OT
  inputs are absent.
- `runs/zscape_regularizer_mining_20260701_123950/RUN_STATUS.md` and report:
  strict rerun decision `none_generalized_close_route`; no law generalized past
  nulls across expression PCA and encoder-agnostic latent proxy spaces; no GPU
  escalation is justified.

Local subagent review:

- scLatent independent subagent agreed the first remote goal should repair the
  per-train-condition mean materialization prerequisite rather than wait for
  user decision.
- Optional tightening from subagent has been applied to `local_goal.md`: new
  NPZ outputs must be written to a fresh run-scoped artifact directory and must
  never overwrite old `*_pert_means.npz`; preflight must record split train
  keys, observed condition/name fields, source matrix path, train-only mask, and
  mapping to materialized NPZ keys.

## Second-Round Local Audit Update - 2026-07-02

Trigger: user ran a multi-project `审计工作开始` round with two (near-identical)
remote `LOCAL_AUDIT_REQUEST` outputs for scLatent (both reporting the
`scaling_condition_mean_materialization_20260701` run). Local audit ran in an
isolated subagent that stalled on large remote `cat` output; the main CC thread
completed the write from verified evidence plus local checks.

Remote sync/state:

- Local clone clean and in sync with GitHub at HEAD `54e9520`. Remote reported the
  same HEAD plus untracked `docs/literature/`, `ref/`, and four untracked
  `ops/*_20260701.py` scripts (server-local, not in Git).
- HEAD anomaly RESOLVED: the materialization RUN_STATUS recorded start HEAD
  `90fd97a`, but `git merge-base --is-ancestor 90fd97a 54e9520` = YES. `90fd97a`
  ("integrate user research priorities") is an ancestor of current `54e9520`; two
  docs commits (`c57831b`, `54e9520`) landed after the run started. Benign forward
  progress, not an integrity problem.
- `remote_decision.md` still holds only the init line; the materialization decision
  was NOT appended (same protocol gap seen across all three projects this round).
  The next task requires back-filling it.

Materialization result (per remote report; inventory re-checkable via SSH):

- prerequisite REPAIRED: preflight 17/17 source-ready, materialized 17/17, 11737
  train-condition vectors, dim 384, NPZ keys `dataset::condition` (not
  dataset-label keys). This closes the prior `0/17` condition-level blocker.
- CPU rerun over the materialized condition means: `none_no_scaling_law_claim`,
  winning axis `none`.

Decisive negative (why the route is CLOSED, not just "no winner yet"):

- Condition count is the STRONGEST full-table correlate (`n_train_conditions`
  spearman ~0.601, p~0.011 vs `family_mmd_delta`; ~0.594, p~0.012 vs `tail_score`)
  yet its OWN leave-one-dataset-out OOS R2 is negative (~-0.176 / +0.025).
- Every geometry/information candidate sits at partial rho ~|0.4| with NEGATIVE
  LODO OOS R2 (e.g. `condition_mean_pairwise_l2_mass` LODO -0.0007;
  `condition_mean_vendi_rbf_effective_count` LODO -0.0174).
- Interpretation: on 17 dataset-level rows even the baseline cannot LODO
  generalize. The binding constraint is statistical POWER + dataset/source
  CONFOUND, not axis choice. Testing a new axis on the same 17 rows repeats the
  error. The condition-mean geometry sub-route is closed as PRESERVED
  underpowered-negative evidence (failure-map).

Provenance anomalies to persist (next task C0):

- `split_information_metrics.csv` mapped two `budget64` true-cell rows to a
  `budget128` artifact path; remote used an `AUTONOMOUS_DECISION` to select the
  capped-H5 source by exact manifest split-name match. Not yet in any tracked doc.
- Old-root provenance `/data/cyx/1030/runs/...` no longer exists; the live path is
  `/data/cyx/1030/scLatent/runs/...`.

User three-direction evaluation:

- Scaling (new information axis): ADOPT direction, but the plain condition-mean
  geometry rerun is DEMOTED/closed. The user's "pair-mode diversity" hypothesis is
  still UNTESTED because explicit train-only OT pair-mode assignments were never
  materialized; the next task first checks whether they are recoverable AND whether
  a power-adequate design exists, rather than grinding 17 rows again.
- Zebrafish (dynamic-response discovery): ADOPT, raised toward primary pivot given
  the scaling power wall. Must BROADEN beyond the rejected narrow regularizer-mining
  coverage: macro distribution dynamics + multi-timepoint OT pseudo-tracking +
  GRN/pathway cascade + latent geometry, with wrong-time/wrong-lineage/permutation
  nulls. Discovery only; no regularizer launch until controls pass.
- Architecture hygiene: ADOPT as a parallel low-risk track, but PLANNING ONLY this
  round (localize P4 `train.py:3500-3501` and P1 `train.py:2941,2958,2969,3003`,
  write the R1/R2 fix plan + no-harm gate). Applying even the near-zero-risk R1
  metric-only fix needs an explicit user greenlight because code edits are gated;
  flagged to the user for decision. Structural constraint recorded: the latent
  trainer has no latent->expression decoder, so an expression-space prior must
  attach in `CoupledFM/model/train.py` (~1643-1644) or add a frozen decoder.

Local checks run this round:

- `git status/rev-parse/fetch` on all three local clones (all clean, in sync);
- `merge-base --is-ancestor 90fd97a 54e9520` = YES (HEAD anomaly resolved);
- read `goal.md`, three `local_*.md`, `docs/LATENTFM_ARCHITECTURE_AUDIT_20260701.md`
  for architecture file:line targets and the user research vision.

## Open Questions For Next Audit

- Are recoverable train-only OT pair-mode / cluster-centroid assignment artifacts
  present, or only collapsed/dataset-level ones? (`next_route_audit` Track A2.)
- Is any power-adequate scaling design reachable (arm count beyond 17, or
  within-dataset budget sweeps) without new dataset-wide processing? (Track A3.)
- Which zebrafish/ZSCAPE assets are CPU-loadable, and does a broader distribution-
  dynamics / OT-pseudo-tracking lens avoid the rejected narrow coverage? (Track B.)
- Will the user greenlight applying the R1 metric-only eval-pairing fix, or keep
  architecture at planning-only? (Track C1 / user decision.)
- REMOTE-ONLY, NOT LOCALLY VERIFIED THIS ROUND: `docs/literature/` notes and
  `ref/zebrafish_dataset.pdf` (untracked on remote); materialization inventory
  numbers are per the remote report, not re-read locally (SSH large-`cat` stalled
  in the isolated subagent this round).
