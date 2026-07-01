# CC Audit & Active Codex Handoff — 2026-07-01 (scaling unit)

Author: CC. `goal.md` remains top steering authority. CC owns this doc; Codex
executes it. Analysis-first, CPU-only, NO training. Higher-priority insight goal
(replaces flow-matching endpoint tuning).

## Goal
Answer decisively, using ONLY existing run results (no GPU, no training): **is
there an information / effective-complexity scaling unit that predicts LatentFM
performance better than raw cell count (and raw condition count)?** Specifically
test an effective-number-of-states axis (Vendi `N_eff`) and an
abundance/response-energy-weighted effective-gene-count `G_eff` (and the product
`N_eff(cells) x G_eff`) as replacement x-axes in a regression over the existing
scaling runs. Decide: upgrade the scaling claim to an information-parameterized law,
or record a leakage-safe negative.

## Why now
The multi-axis scaling audit already closed every RAW-COUNT axis
(`reports/scaling_law_ready_evidence_table_20260626/axis_law_readiness.csv`: all
`not_law_ready`; true-cell fails no-harm all 3 seeds; condition-exposure
non-monotonic). The user's thesis: cell count is the wrong minimal unit; the axis
should be information / cluster / pair / statistic based, gene-token-information
weighted (HVGs count more). Half-validated already: HVG *concentration* is real
(top-2k genes carry ~0.84 of primary response energy) BUT the HVG-specific
intervention collapses to abundance/detectability
(`reports/observable_gene_budget_scaling_law_gate_20260630/`). So the correct
weighted unit is abundance/response-energy-weighted, not a bespoke HVG score. The
`information_axis_v2` file is only a planning table, not an estimator — the real
quantities are already computed and on disk. This is decidable with zero GPU.

## Read first (exact server paths)
- reports/scaling_law_ready_evidence_table_20260626/axis_law_readiness.csv
- reports/multiaxis_information_scaling_incremental_gate_20260629/multiaxis_information_scaling_join_rows.csv  (the 17-run x 52-col decisive dataset) + ..._association_rows.csv + ..._lodo_rows.csv  (regression+LODO harness to REUSE)
- reports/downstream_information_scaling_preflight_20260628/split_information_metrics.csv  (entropy/effective_count/effective_rank/pairwise_l2 per split)
- reports/zscape_hvg_fullgene_information_axis_20260628/zscape_hvg_fullgene_information_curve.csv  (HVG response-energy concentration)
- reports/observable_gene_budget_scaling_law_gate_20260630/  (abundance-confound result — the bar to clear)
- ops/build_latentfm_scaling_information_axis_v2_matrix_20260628.py (note: synthesis table, NOT an estimator)
- goal.md, docs/EXPERIMENT_INDEX.md, docs/PROJECT_REVIEW.md, docs/DECISIONS.md, docs/CC_CODEX_COOPERATION_PROTOCOL.md (anti-spin, goal-doc execution)

## Method (CPU, no training)
1. PRE-REQUISITE — materialize PER-ARM geometry. The current join collapses the 5
   truecell-budget-curve arms to ONE parent geometry (all share
   residual_effective_rank=40.23, vendi=7.04) — a real power defect. Recompute
   per-arm Vendi effective-count, effective rank, participation ratio, pairwise-L2,
   and the abundance/response-energy-weighted `G_eff` from the per-arm condition-mean
   artifacts (`*_pert_means.npz` referenced in split_information_metrics.csv). If a
   per-arm artifact is missing, record exactly which and STOP for that arm (do not
   fabricate; do not train).
2. Assemble one tidy table: Y = {cross_pp_delta, family_pp_delta, family_mmd_delta,
   tail_score} (join cols 7-10); X = {cell_count, condition_count, Vendi N_eff,
   effective_rank, participation_ratio, Kish N_eff, state_entropy, G_eff,
   N_eff x G_eff, exact_condition_fraction}.
3. For each X: fit y~x (linear + log-x); report Spearman monotonicity, R^2, and
   leave-one-run-out (LODO) out-of-sample R^2 (reuse the existing LODO harness).
4. Rank axes by: sign-stability x out-of-sample R^2 x survives partial-correlation
   control for cell count AND dataset identity (reuse residual_spearman machinery).

## Codex owns
tools/ (small CPU analysis scripts), runs/<run>/, RUN_STATUS.md, new analysis
outputs under reports/<new scaling-unit dir>. May APPEND dated sections to
docs/EXPERIMENT_INDEX.md as text proposals inside RUN_STATUS.md.

## CC owns
goal.md, docs/PROJECT_REVIEW.md, docs/DECISIONS.md, docs/EXPERIMENT_INDEX.md, this
handoff doc, and ALL git operations.

## Permissions
sandbox = workspace-write; CPU-only; model gpt-5.5; effort high. NO GPU, NO training.

## Forbidden
- Any GPU job, model training, or checkpoint selection.
- Fabricating per-arm geometry when the artifact is missing (report instead).
- Claiming a law on the collapsed-geometry rows (see stop rules).
- Destructive writes; secret print/commit; git commit/push/pull (CC manages git).
- Claim overreach: do NOT assert a deployable scaling law unless the pre-declared
  bar is cleared; default model stays xverse_8k_anchor.

## Success criteria (pre-declared)
An information/weighted axis "wins" iff, vs cell count on the SAME runs: (a) higher
|Spearman|, (b) higher leave-one-run-out R^2, AND (c) same-sign LODO fraction >= 0.6
AFTER partialling out cell count AND dataset identity (the exact bar where, so far,
only exact_condition_fraction cleared unadjusted significance and NOTHING cleared
the residual control). Report the winning axis (or "none"), with the regression
table, and propose a one-paragraph DECISIONS entry.

## Stop rules (incl. anti-spin DECISION NEEDED)
- POWER FLOOR: if the winning axis's advantage rests on the collapsed
  parent-geometry rows (distinct geometry values < ~12), declare UNDERPOWERED, do
  not claim a law; recommend materializing more distinct-geometry arms (CPU) as the
  next step.
- MISSING-INPUT: if per-arm geometry cannot be materialized from existing
  artifacts, the deliverable becomes "list exactly what per-arm artifact is missing"
  — NOT launching training. STOP + DECISION NEEDED.
- CONFOUND: any winning axis MUST beat cell count after the abundance control that
  sank the HVG-specific claim; if it only correlates with dataset identity, it's a
  confound, not a unit.
- ANTI-SPIN: two non-converging attempts or a repeated failure class -> STOP and
  append a DECISION NEEDED block for CC.

## Expected output paths
- runs/<run>/RUN_STATUS.md (plan, per-arm materialization log, regression table,
  ranked axes, win/none decision vs the pre-declared bar).
- reports/<scaling_unit_regression_dir>/ with the tidy table + fits.

## Progress reporting format
Append dated lines to RUN_STATUS.md at: plan, per-arm geometry materialized,
regression complete, decision recorded. Final summary via --output-last-message.
No git operations; propose DECISIONS/EXPERIMENT_INDEX text inside RUN_STATUS.md.
