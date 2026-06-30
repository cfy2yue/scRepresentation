# Project Overview

## Current Project Goal

Build a Nature Methods-level single-cell foundation model representation
benchmark and latent perturbation modeling workflow under `/data/cyx/1030`.

The scientific thesis is that single-cell foundation model latent
representations are systematically under-used and under-benchmarked, while the
field over-focuses on direct perturbation prediction. The first-stage paper
therefore needs two strong pillars:

- a reproducible, broad scFMBench representation benchmark;
- a LatentFM modeling demonstration showing what high-quality latent spaces can
  support, with strict zero-shot and split/family evaluation.

## Workspace Layout

| Path | Role |
|---|---|
| `/data/cyx/1030/scLatent` | Current scLatent project root and intended GitHub target `cfy2yue/scLatent`. |
| `/data/cyx/1030/scLatent/scFMBench` | Benchmark repo checkout, GitHub `cfy2yue/scFMBench`, currently nested under scLatent. |
| `/data/cyx/1030/scLatent/CoupledFM` | Modeling repo checkout, GitHub `cfy2yue/CoupledFM`, currently nested under scLatent. |
| `/data/cyx/1030/dataset` | Canonical local dataset root. |
| `/data/cyx/1030/scLatent/scFM_output` | Benchmark metrics, embeddings, figures, manifests. |
| `/data/cyx/1030/scLatent/runs` | Long-run status, launch scripts, scheduled one-shots. |
| `/data/cyx/1030/scLatent/reports` | Human-readable reports, audits, decision gates. |
| `/data/cyx/1030/scLatent/docs` | Project-level overview, review, decisions, experiment index. |

Historical reports and old run statuses may still mention pre-migration root
paths such as `/data/cyx/1030/runs/...` or `/data/cyx/1030/reports/...`.
Treat those as provenance unless a current entrypoint repeats them as live
instructions.

## Dataset Summary

The canonical dataset root is `/data/cyx/1030/dataset`.

Current disk usage is about 850G because it contains both training-ready
artifacts and LiLab/source rebuild layers. The current training-ready package is
about 444G and consists of:

- `latentfm_full`
- `biFlow_data`
- `scFM_data`
- `drug_cache`
- `cellgene_census`
- `dataset/README.md`

The largest optional future cleanup candidate is `Training_data` at about 351G,
but no data should be deleted until active LatentFM probes finish and package
validation plus backup/staging checks are repeated.

## Benchmark Status

scFMBench currently has a reproducible artifact layer:

- 10 figures in `/data/cyx/1030/scLatent/scFM_output/figures`;
- PDF, SVG, PNG, and `.meta.json` for each figure;
- 0 failed and 0 skipped figures in the manifest;
- 522 rows in the aggregate metric table;
- NicheFormer and TranscriptFormer included as chempert-only new-model evidence;
- explicit count-source handling for NicheFormer/TranscriptFormer with no
  duplicate `log1p`.

The full-coverage LatentFM encoder set remains:

- `stack`
- `scldm`
- `scfoundation`

NicheFormer and TranscriptFormer should be shown in the benchmark as
chempert-only new-model evidence until broad count-compatible atlas/genepert
embeddings are available.

## LatentFM Status

The current best historical formal reference remains scFoundation primary for
canonical single/family context:

| Metric | Value |
|---|---:|
| test MMD | 0.027124 |
| test pp | 0.0338 |
| family gene pp | 0.0437 |
| family drug pp | -0.0082 |
| multi seen pp | 0.2112 |
| unseen1 pp | -0.0032 |
| unseen2 pp | -0.1386 |

The current deployable/default LatentFM model is `xverse_8k_anchor`.
No scaling-derived or Track C checkpoint is currently promoted. The 2026-06-23
frozen anchor-gated support-teacher blend is retained only as historical Track C
diagnostic context, not as the current default model and not as a route for
additional query tuning.

Current scaling status:

- true-cell/per-condition support is the strongest internal mechanism signal;
- condition-count/exposure and background/type/source breadth remain
  non-monotonic, confounded, or tail-unsafe;
- scaling is report-ready as an axis-specific mechanism/failure-map package;
- no scaling route currently passes frozen canonical no-harm and promotion
  gates.

## Main Blockers

- LatentFM has not yet met the biological-insight bar across MMD, pp, pc,
  split/family metrics, and condition-level interpretability.
- Formal Track C multi capability remains unsolved: support/routed-distill
  routes were closed before held-out query use, and canonical multi must remain
  diagnostic only unless a future support-val route freezes before final query.
- Scaling is scientifically valuable but currently supports mechanism and
  failure-map claims, not a promoted monotonic scaling law or default model.
- Drug/chemical perturbation conditioning is auditable but not yet a formal
  molecular zero-shot route; chemical V2 fixed-step controls require exact ACK.
- Manuscript-specific LatentFM claims must include no-harm boundaries,
  provenance, bootstrap/CI, and representative negative tails.

As of the latest 2026-06-25 review, the deployable/default LatentFM model is
`xverse_8k_anchor`. The 2026-06-23 anchor-gated support-teacher blend remains
historical Track C diagnostic context, not the current default model and not a
route for further query tuning.

Current active status:

- immediate non-ACK LatentFM GPU candidate count: `0`;
- Track C routed-distill/support routes: closed before query;
- scaling: report-ready mechanism/failure-map, no model promotion;
- training-set metadata/QC/Jiang: supplement/failure-map only;
- nearest GPU route: chemical V2 fixed-step controls after exact protocol ACK
  and fresh resource audit.

Primary current entry points:

```text
/data/cyx/1030/scLatent/reports/LATENTFM_CURRENT_GPU_CANDIDATE_INVENTORY_20260625.md
/data/cyx/1030/scLatent/reports/LATENTFM_SCALING_NM_CLAIM_FAILURE_PACKAGE_20260625.md
/data/cyx/1030/scLatent/reports/LATENTFM_SCALING_NM_PROVENANCE_MANIFEST_20260625.md
/data/cyx/1030/scLatent/reports/LATENTFM_SCALING_NARRATIVE_SKELETON_20260625.md
/data/cyx/1030/scLatent/reports/LATENTFM_SCALING_REPRODUCTION_MANIFEST_20260625.md
```

## Current Best Next Action

Use the current GPU candidate inventory plus scaling package as the project
entry points. Do not run another held-out query, alpha sweep, scaling replay,
QC-weighting, or Jiang-specialized GPU job from current evidence.

The nearest GPU action is chemical V2 only after exact ACK. Otherwise, the next
useful action is final report/narrative polish or a materially new CPU-only
gate with explicit hypothesis, forbidden inputs, promotion criteria, and
fail-close rule.
