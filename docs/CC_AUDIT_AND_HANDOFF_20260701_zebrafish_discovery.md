# CC Audit & Active Codex Handoff — 2026-07-01 (zebrafish perturbation-dynamics DISCOVERY)

Author: CC (from `docs/RESEARCH_VISION_20260701.md` + user direction). Scientific
discovery track: from the zebrafish time-series ground truth, discover how cells respond to
perturbation and the dynamic process, via a multi-lens program. Analysis-first (CPU; <=1
GPU only if a specific step needs it, after a resource note). CC owns direction/git; Codex
executes. Supersedes the narrow 2026-07-01 zscape mining (which found no generalization on
tiny coverage) — this BROADENS the search substantially.

## Understand first
- ref/zebrafish_dataset.pdf (skim) and, importantly, Codex's synthesis
  docs/literature/SCALING_ZSCAPE_SQUIDIFF_NOTES_20260701.md — read+understand, return to
  the PDF key spots as needed.
- dataset/external/zscape_20260628/: GSE202639 zperturb (CRISPR crispant knockouts) +
  GSE202639_reference_cell_metadata.csv.gz (WILD-TYPE developmental atlas: dense timepoints
  18-96 hpf, 98 cell types, mean_nn_time, umap3d/subumap3d — the dynamic-transition GT).

## GOAL (discover generalizable perturbation-dynamics regularities; do NOT stop after a shallow pass)
Run the multi-lens program below across MANY lineages/conditions (not the 2 periderm rows).
Deliverable = a SHORTLIST of dynamic-response regularities that GENERALIZE, each with a
stats table + interpretation; OR a documented honest negative that nothing generalizes on
the available coverage.

Lenses (from the user's vision):
- **Distribution (macro) view**: across and between time points, distribution statistics and
  their regularities — e-distance, means, higher moments, and how they evolve with
  developmental time / dose.
- **Individual view**: build single-cell-dimension time-series via OT. NOT just two
  timepoints — **multi-timepoint OT pairing**, sampling one cell per stage to form a
  **pseudo single-cell tracking** trajectory. Then analyze per-trajectory:
  - **Expression space**: target-gene and marker-gene change trajectories; introduce a
    **GRN (CellOracle / GEARS-style)** to test whether the signal propagates FROM the target
    gene outward; pathway enrichment to test cross-pathway association and single-pathway
    upstream->downstream cascade.
  - **Latent space**: geometry, direction, curvature, speed of the transition.

## Success criteria (DONE)
>= 2-4 candidate regularities (geometric OR biological) that survive: held-out
conditions AND held-out cell-types/lineages, wrong-time / wrong-lineage / label-permutation
nulls, LODO same-sign >= 0.80, stratified-permutation p <= 0.10, consistent sign across >= 2
germ layers. Each with a stats table + biological interpretation + (if geometric) a note on
how it could regularize the LatentFM flow (attach at the raw-expr trainer for expression
space; velocity/ode_integrate_diff for latent). OR: a documented honest negative (which
lenses were run, coverage, why nothing generalized).

## Read first (server paths)
- docs/literature/SCALING_ZSCAPE_SQUIDIFF_NOTES_20260701.md; docs/RESEARCH_VISION_20260701.md
- dataset/external/zscape_20260628/GSE202639_reference_cell_metadata.csv.gz + zperturb_full_*
- runs/zscape_raw_counts_cell_manifest_extraction_20260628/.../outputs/zscape_manifest_selected_counts_csc.npz
- ops/audit_zscape_ot_dynamic_response_gate_20260628.py (OT displacement engine — EXTEND across lineages)
- ops/audit_zscape_expression_latent_biology_preflight_20260628.py (temporal-tangent metric)
- ops/plan_zscape_continuity_ot_gate_from_coverage_20260628.py (the un-run reference-atlas trajectory hook)
- ops/audit_latentfm_zscape_to_trainset_translation_gate_20260629.py (strict controls harness to reuse)
- reports/LATENTFM_ABINITIO_RESPONSE_LAWS_SCALING_ZSCAPE_TRANSLATION_20260628.md (L1-L7 scaffold — update)

## Codex owns
tools/ + ops/ analysis scripts, runs/, reports/<zscape discovery dir>, RUN_STATUS.
Do NOT modify CoupledFM training code (only note attach points for a future regularizer).
## CC owns
goal.md, docs/DECISIONS.md, docs/PROJECT_REVIEW.md, the L1-L7 doc, ALL git.

## Permissions
sandbox = workspace-write; CPU-first (<=1 GPU only if a specific encoder/step needs it, with
a resource note); model gpt-5.5; effort high.

## Forbidden
- Reopening the UCE / species-latent route (closed with calibrated negatives).
- Using canonical-multi / Track-C-query data; treating g:Profiler programs as validated
  without specificity gates. Modifying CoupledFM training code. Secret print/commit;
  git commit/push (CC finalizes).

## Do NOT stop until (goal-mode intent)
Keep running lenses across lineages until the shortlist (>=2-4 generalizing regularities) is
found, OR the program is exhausted and a documented honest negative is written (lenses run,
coverage, null results). Do not stop after one lens / one lineage. Anti-spin: a repeated
failure class -> STOP + DECISION NEEDED with what's blocking.

## Expected outputs
runs/<run>/RUN_STATUS.md + reports/<zscape discovery dir>/ with the per-lens/per-lineage
tables, the generalization+null results, and the shortlist (or documented negative).

## Progress format
One line per lens/lineage: `[lens] lineage: statistic / generalization verdict`; final
summary via --output-last-message: the shortlist of regularities (or "none generalized").
