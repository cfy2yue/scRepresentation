# local_audit.md

Updated: 2026-07-01.

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
- Scaling-unit: blocked at the fair-regression prerequisite. Existing joined
  artifacts collapse true-cell arms to one parent geometry; next bounded CPU
  requirement is per-arm geometry materialization before testing Vendi
  `N_eff`, effective rank, participation ratio, pair-mode diversity, `G_eff`,
  or `N_eff x G_eff` against performance.
- Zebrafish/ZSCAPE: current regularizer-mining coverage was negative. No
  generalized dynamic-response regularity survived wrong-time/wrong-lineage or
  permutation nulls, so no differentiable flow regularizer is validated from
  that run.
- Architecture audit: known issues include train/eval estimator mismatch,
  eval velocity-MSE random pairing, additive-only conditioning, gradient
  conflict, batch-mean heterogeneity collapse, and linear-path/Euler sensitivity.
  These are audit facts, not permission to edit code.

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

## Open Questions For Next Audit

- Is the next remote cycle manuscript polish, scaling per-arm geometry, broader
  zebrafish discovery, or architecture-hygiene planning?
- If scaling is chosen, which exact report/run artifacts contain the per-arm
  `*_pert_means.npz` inputs and what overwrite guards are required?
- If zebrafish is chosen, what new coverage or broader analysis lens avoids
  repeating the rejected regularizer-mining coverage?
- If architecture hygiene is chosen, which defect is metric-only enough to fix
  first, and what no-harm gate prevents a model-claim overreach?
