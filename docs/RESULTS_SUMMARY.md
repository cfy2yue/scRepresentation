# Results Summary

## Status

The full project goal is not complete.

The benchmark/data/artifact layer is strong enough for the current phase. The
main open requirement is still LatentFM reaching the biological-insight bar:
stable MMD/pp/pc, split/family robustness, and interpretable perturbation
structure, especially for unseen multi-perturbation generalization.

## scFMBench

Current artifact layer:

- figure manifest:
  `/data/cyx/1030/scFM_output/figures/manifest.json`
- figures: 10
- failed figures: 0
- skipped figures: 0
- aggregate metric rows: 522
- output formats: PDF, SVG, PNG, per-figure metadata

NicheFormer and TranscriptFormer:

- implemented and resource-validated;
- included as chempert-only evidence across 4 SciPlex chempert datasets;
- raw and PCA-128 metric rows are present;
- count-source handling is explicit and avoids duplicate `log1p`.

Full-coverage LatentFM encoder set remains:

| Rank | Model | Performance score | Median cells/s | Datasets |
|---:|---|---:|---:|---:|
| 1 | `stack` | 0.6509 | 1602.88 | 23 |
| 2 | `scldm` | 0.5852 | 106.27 | 23 |
| 3 | `scfoundation` | 0.6741 | 15.24 | 23 |

NicheFormer and TranscriptFormer should be shown as limited-coverage
new-model evidence, not as failed full benchmark models.

## LatentFM Reference State

Primary scFoundation remains the current formal reference:

| Metric | Value |
|---|---:|
| test MMD | 0.027124 |
| test pp | 0.0338 |
| family gene pp | 0.0437 |
| family drug pp | -0.0082 |
| multi seen pp | 0.2112 |
| unseen1 pp | -0.0032 |
| unseen2 pp | -0.1386 |

Strong-composition scFoundation improved selected multi-unseen behavior but was
not promotable because it lost aggregate pp, family-gene pp, and drug-family
behavior.

## Current LatentFM Operating State

Current deployable/default model remains `xverse_8k_anchor`.

Recent Track C support/routed-distill routes are closed before any further
held-out query use. The older 2026-06-23 frozen anchor-gated support-teacher
blend is historical diagnostic context, not the current deployable/default
model and not a route for more query tuning.

Current authority for whether a GPU run is legal:

```text
/data/cyx/1030/reports/LATENTFM_CURRENT_GPU_CANDIDATE_INVENTORY_20260625.md
```

Latest inventory status:

- immediate non-ACK LatentFM GPU candidate count: `0`;
- scaling/model-promotion routes: no GPU authorized;
- training-set metadata/QC/Jiang routes: supplement/failure-map only;
- Track C support/routed-distill: closed before query;
- nearest GPU route: chemical V2 fixed-step controls after exact protocol ACK
  and fresh resource audit.

Scaling is now report-ready as a mechanism/failure-map package, not a model
promotion:

```text
/data/cyx/1030/reports/LATENTFM_SCALING_NM_CLAIM_FAILURE_PACKAGE_20260625.md
/data/cyx/1030/reports/LATENTFM_SCALING_NM_PROVENANCE_MANIFEST_20260625.md
/data/cyx/1030/reports/LATENTFM_SCALING_FIGURE_READINESS_20260625.md
/data/cyx/1030/reports/LATENTFM_SCALING_NARRATIVE_SKELETON_20260625.md
/data/cyx/1030/reports/LATENTFM_SCALING_REPRODUCTION_MANIFEST_20260625.md
```

Allowed scaling claim: leakage-safe cross-dataset scaling-axis audit with
no-harm vetoes, bootstrap/CI, provenance, and negative evidence maps.

Disallowed scaling claim: first-in-field scaling law, monotonic more-data law,
checkpoint improvement, deployed scaling model, or chemical scaling success.

## Figure Readiness

Current benchmark figures are reproducible artifacts, and a manuscript-specific
figure export now exists:

```text
/data/cyx/1030/scFM_output/figures_manuscript
/data/cyx/1030/scFM_output/figures_manuscript/CAPTIONS_DRAFT.md
```

This export contains 10 figures with PDF, SVG, PNG, and per-figure metadata.
The figure layer is ready for benchmark-side interpretation. LatentFM scaling
figures are also QA-passed and provenance-consistent; use the scaling narrative
skeleton for wording boundaries.

Main figure architecture is recorded in:

```text
/data/cyx/1030/reports/MANUSCRIPT_FIGURE_ARCHITECTURE_20260619.md
```

## Next Decision

No active LatentFM GPU job is currently known. The next useful action is final
report/narrative polish from the scaling packages, chemical V2 after exact ACK,
or a materially new CPU-only gate for a genuinely new mechanism.

Do not launch query follow-ups, alpha sweeps, QC weighting, Jiang-specialized
training, or scaling replay GPU jobs from current evidence. If compute is to be
used, first write a new CPU gate with explicit hypothesis, forbidden inputs,
promotion criteria, and fail-close rule, then refresh the GPU candidate
inventory.
