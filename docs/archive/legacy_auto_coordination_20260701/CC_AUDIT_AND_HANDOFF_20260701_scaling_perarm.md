# CC Audit & Active Codex Handoff — 2026-07-01 (scaling per-arm geometry + info-unit regression)

Author: CC (planned from prior scaling audit + user's scaling vision). CC owns
audit/direction/git; Codex EXECUTES this goal only. CPU-only, NO training. This unblocks
the scaling-unit test that previously STOPPED at the collapsed-parent-geometry prerequisite.

## Direction basis
- Cell count is the wrong scaling unit; want a **quantitative scaling-x / single-cell
  "information content"** (perturb-seq), from clustering / pair / statistical / info-theoretic
  angles. HVG concentration is real (top-2k ~84% response energy) but the HVG-specific
  signal collapses to **abundance** — so weight by abundance/response-energy, not "HVG-ness".
- Prior scaling-unit goal STOPPED honestly: the 17-run join collapses the 5 truecell arms to
  ONE parent geometry (all share effective_rank=40.23, vendi=7.04) → the regression can't be
  fair until **per-arm geometry** is materialized.

## GOAL (Codex executes — CPU, no training)
1. **Materialize PER-ARM geometry** from the existing per-arm condition-mean artifacts
   (`*_pert_means.npz` referenced in `split_information_metrics.csv`): for each scaling run
   arm compute Vendi effective-count, effective rank, participation ratio, pairwise-L2/cosine
   mass, state entropy, Kish effective-sample-size, and the **abundance/response-energy-
   weighted effective-gene-count G_eff**; ALSO compute a **pair-mode diversity** axis
   (cluster + OT → count distinct pair-modes / condition-pair patterns) as a candidate
   scaling-x. If a per-arm artifact is missing, record exactly which and SKIP that arm (do
   not fabricate).
2. **Regression**: with Y = {cross_pp_delta, family_pp_delta, family_mmd_delta, tail_score}
   (from `multiaxis_information_scaling_join_rows.csv`), fit y~x (linear + log-x) for each
   candidate x AND for cell count / condition count; report Spearman monotonicity, R², and
   leave-one-run-out (LODO) out-of-sample R² (reuse the existing LODO/association harness);
   rank axes by sign-stability × out-of-sample R² × survives partial-correlation control for
   **cell count AND dataset identity**.

## Why now
This is the prerequisite the prior goal flagged; with per-arm geometry, we can decisively
test whether an information/effective-state/pair-mode/G_eff axis predicts performance better
than cell count — the core of the quantitative single-cell scaling law.

## Read first (server paths)
- reports/multiaxis_information_scaling_incremental_gate_20260629/multiaxis_information_scaling_join_rows.csv
  (+ ..._association_rows.csv, ..._lodo_rows.csv — REUSE the regression+LODO harness)
- reports/downstream_information_scaling_preflight_20260628/split_information_metrics.csv
  (entropy/effective_count/effective_rank/pairwise_l2 per split; points to the per-arm *_pert_means.npz)
- reports/zscape_hvg_fullgene_information_axis_20260628/zscape_hvg_fullgene_information_curve.csv
  (response-energy concentration → G_eff basis)
- reports/observable_gene_budget_scaling_law_gate_20260630/  (abundance-confound = the bar any weighted axis must clear)
- reports/scaling_unit_regression_20260701/  (prior missing_per_arm_artifacts.csv + per_arm_geometry_rows.csv)
- docs/RESEARCH_VISION_20260701.md, docs/DECISIONS.md, docs/CC_CODEX_COOPERATION_PROTOCOL.md (Standard Flow, Anti-Spin)

## Codex owns
tools/ CPU analysis scripts, runs/, new outputs under
reports/scaling_perarm_regression_20260701/, RUN_STATUS.md.

## CC owns
goal.md, docs/PROJECT_REVIEW.md, docs/EXPERIMENT_INDEX.md, docs/DECISIONS.md, this doc, ALL git.

## Permissions
sandbox = workspace-write; CPU-only; model gpt-5.5; effort high. NO GPU, NO training.

## Forbidden
- Any GPU/training/checkpoint selection; fabricating per-arm geometry when missing (report instead);
  claiming a law on collapsed-geometry rows; secret print/commit; **git commit/push (CC finalizes).**
- Claim overreach: no deployable-scaling-law claim unless the pre-declared bar is cleared; default stays xverse_8k_anchor.

## Success criteria (pre-declared)
An information/weighted/pair-mode axis "wins" iff, vs cell count on the same runs: (a) higher
|Spearman|, (b) higher LODO out-of-sample R², AND (c) same-sign LODO fraction >= 0.6 AFTER
partialling out cell count AND dataset identity (the bar where, so far, nothing cleared the
residual control). Report the winning axis (or "none") + the regression table + a proposed
one-paragraph DECISIONS entry.

## Stop rules / anti-spin
- POWER FLOOR: if a win rests on rows with < ~12 distinct geometry values, declare UNDERPOWERED,
  recommend materializing more distinct-geometry arms (CPU) — do not claim a law.
- CONFOUND: any winning axis MUST beat cell count after the abundance control; if it only
  correlates with dataset identity, it's a confound.
- If per-arm geometry cannot be materialized from existing artifacts → list exactly what's
  missing + STOP (no training). Two non-converging attempts / repeated failure → DECISION NEEDED.

## Expected outputs
runs/<run>/RUN_STATUS.md (per-arm materialization log + regression table + ranked axes +
win/none decision vs the bar) + reports/scaling_perarm_regression_20260701/.

## Progress format
One line per step `[step] status | key metric`; final summary via --output-last-message:
the winning scaling-x (or "none" / underpowered) + its margin over cell count.
