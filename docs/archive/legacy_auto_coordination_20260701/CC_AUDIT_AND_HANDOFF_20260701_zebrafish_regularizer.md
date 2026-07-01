# CC Audit & Active Codex Handoff — 2026-07-01 (zebrafish dynamic-law regularizer)

Author: CC. `goal.md` remains top steering authority. CC owns this doc; Codex
executes it. Analysis-first, CPU-only (at most 1 GPU only if later justified — NOT
in this goal). Higher-priority insight goal (replaces flow-matching endpoint tuning).

## Goal
Mine the zebrafish developmental ground truth for GENERALIZABLE, systematic
perturbation dynamic-response REGULARITIES, and spec each surviving one as a
DIFFERENTIABLE regularizer on the LatentFM flow-matching vector field. Deliverable
= a shortlist (target 2-4) of candidate dynamic-response laws, each with (i)
held-out generalization evidence and (ii) a concrete differentiable-loss spec
(which term, on velocity vs trajectory, in expression vs latent space). If nothing
generalizes past the nulls -> a decisive, publishable NEGATIVE that protects the
model from a false constraint.

## Why now
Zebrafish (GSE202639) is a rare observed GT for perturbation DYNAMIC transitions.
The UCE/species-latent route is CLOSED with calibrated negative evidence
(`reports/zscape_uce_danio_latent_continuity_gate_*`, `..._failure_calibration_*`) —
do NOT reopen it; mine geometry in expression + encoder-agnostic latent proxy
instead. The unexploited enabling asset is the WILD-TYPE developmental reference
atlas (dense timepoints 18-96 hpf, 98 cell types, `mean_nn_time`, shared coords
with the perturbation atlas). Two laws are already partially evidenced (periderm
noto/smo) and are geometric + differentiable: L2 developmental-tangent split and L1
state-preservation. Scaling tells us to constrain STRUCTURE, not capacity (more
data != better under no-harm), so the regularizer must be
support/abundance-residualized and no-harm-gated.

## Read first (exact server paths)
- dataset/external/zscape_20260628/GSE202639_reference_cell_metadata.csv.gz  (WILD-TYPE GT: timepoint, cell_type_broad, germ_layer, umap3d_*/subumap3d_*, mean_nn_time)
- dataset/external/zscape_20260628/GSE202639_zperturb_full_*  (perturbation snapshots)
- runs/zscape_raw_counts_cell_manifest_extraction_20260628/.../outputs/zscape_manifest_selected_counts_csc.npz  (extracted matrix)
- ops/audit_zscape_ot_dynamic_response_gate_20260628.py  (OT displacement engine to EXTEND from 2 rows to many lineages)
- ops/audit_zscape_expression_latent_biology_preflight_20260628.py  (temporal-tangent-alignment metric)
- ops/plan_zscape_continuity_ot_gate_from_coverage_20260628.py  (the un-run reference-atlas trajectory hook)
- ops/audit_latentfm_zscape_to_trainset_translation_gate_20260629.py  (strict controls harness: residual Spearman, bootstrap CI, LODO same-sign, permutation p)
- reports/LATENTFM_ABINITIO_RESPONSE_LAWS_SCALING_ZSCAPE_TRANSLATION_20260628.md  (L1-L7 laws scaffold — update with generalization results)
- CoupledFM/model/train.py (raw-expression trainer: gene-space x1_hat/x_gt at ~1643-1644 — the attach point for an EXPRESSION-space prior), CoupledFM/model/latent/train.py + latent/models/mlp.py (velocity field; velocity/x_t/ode_integrate_diff for a LATENT/velocity prior)
- goal.md, docs/DECISIONS.md, docs/LATENTFM_ARCHITECTURE_AUDIT_20260701.md, docs/CC_CODEX_COOPERATION_PROTOCOL.md (anti-spin, goal-doc execution)

## Method (CPU, analysis-first)
1. Build the WILD-TYPE developmental tangent field from the reference atlas
   (per lineage, adjacent-stage centroid tangents over timepoint/mean_nn_time in
   umap3d/subumap3d) — the un-run half of the continuity/OT planner.
2. EXTEND the OT dynamic-response gate from the 2 periderm rows to the full
   primary-row set and multiple lineages; for each perturbation compute candidate
   GEOMETRIC regularities: (a) tangent-alignment cosine (velocity vs developmental
   tangent) and orthogonal residual magnitude (L2); (b) within-state vs
   cross-composition flux at small displacement (L1); (c) displacement
   straightness/curvature; (d) intrinsic dimensionality of the displacement set
   (participation ratio / effective rank); (e) flow divergence/convergence.
   Optionally (secondary) BIOLOGICAL regularities: g:Profiler program kinetics over
   pseudotime — only if specificity gates can be satisfied.
3. Compute each regularity TWICE: expression-space (log1p-HVG-SVD; the validated
   stable substrate) AND an encoder-agnostic latent proxy (control-only SVD /
   PCA / random projection — NOT UCE). Report expr-vs-latent consistency.
4. Test GENERALIZATION with the strict harness: held-out-condition and
   held-out-cell-type effect vs wrong-time / wrong-lineage / label-shuffle /
   embryo-heldout nulls; LODO same-sign >= 0.80; stratified-permutation p.

## Codex owns
tools/ + ops/ analysis scripts (CPU), runs/<run>/, RUN_STATUS.md, new outputs under
reports/<zscape regularity dir>. Do NOT modify CoupledFM training code in this goal
(this goal only READS it to write the regularizer spec).

## CC owns
goal.md, docs/DECISIONS.md, docs/PROJECT_REVIEW.md, the L1-L7 laws doc, this handoff
doc, and ALL git operations.

## Permissions
sandbox = workspace-write; CPU-only; model gpt-5.5; effort high. NO GPU, NO LatentFM
training, NO checkpoint selection in this goal.

## Forbidden
- Reopening UCE / species-latent extraction (closed with calibrated negatives).
- Using canonical-multi / Track-C-query data.
- Treating g:Profiler programs as validated without specificity gates.
- Modifying CoupledFM training code (spec the regularizer as a written design, do
  not implement it here).
- Destructive writes; secret print/commit; git commit/push/pull (CC manages git).

## Success criteria
>= 1 candidate regularity with effect surviving ALL wrong-controls AND source-LODO
same-sign >= 0.80 AND stratified-permutation p <= 0.10 AND consistent sign across
>= 2 lineages/germ layers on HELD-OUT cell types, PLUS a written,
dimensionally-consistent differentiable-loss spec (term, velocity vs trajectory,
expr vs latent, with the mandatory abundance/support residualization + no-harm
guard). Target 2-4 shortlist candidates. Note explicitly, per the architecture
audit, that an EXPRESSION-space prior must attach in CoupledFM/model/train.py
(raw-expr trainer), since the latent trainer has no latent->gene decoder.

## Stop rules (incl. anti-spin DECISION NEEDED)
- If no geometric statistic generalizes past wrong-time/wrong-lineage nulls across
  >= 2 held-out lineages -> STOP and CLOSE the regularizer route as
  "single-lineage-only" (record the negative; do NOT escalate to GPU).
- If the only surviving regularity is the known periderm noto/smo effect -> downgrade
  to "diagnostic, not regularizer".
- ANTI-SPIN: two non-converging attempts or a repeated failure class -> STOP and
  append a DECISION NEEDED block for CC.

## Expected output paths
- runs/<run>/RUN_STATUS.md (plan, tangent-field build, per-lineage regularity table,
  generalization/null results, shortlist + differentiable-loss specs, decision).
- reports/<zscape regularity mining dir>/ with the shortlist table + specs.

## Progress reporting format
Append dated lines to RUN_STATUS.md at: plan, tangent field built, OT gate extended,
generalization tested, shortlist + specs written, decision. Final summary via
--output-last-message. No git operations; propose DECISIONS text inside RUN_STATUS.md.
