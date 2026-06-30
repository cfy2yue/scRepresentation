# Model Notes

## Main Workflows

This workspace currently contains several related but distinct workflows.

| Workflow | Code root | Current role |
|---|---|---|
| scFMBench embedding benchmark | `/data/cyx/1030/scLatent/scFMBench` | Compare foundation-model latent spaces across atlas, geometry, chempert, and genepert tasks. |
| Raw FM / perturb-aware pretraining | `/data/cyx/1030/scLatent/CoupledFM` | Pretraining and raw-space perturbation-facing model components. |
| CoupledFM / biFlow-style modeling | `/data/cyx/1030/scLatent/CoupledFM` | Coupled flow modeling using control/GT resources under `dataset/biFlow_data`. |
| LatentFM | `/data/cyx/1030/scLatent/CoupledFM/model/latent` | Flow matching in foundation-model latent spaces using `dataset/latentfm_full`. |
| Benchmark inference/reporting | `/data/cyx/1030/scLatent/scFM_output` and `/data/cyx/1030/scLatent/reports` | Metrics, figures, manifests, and manuscript-facing evidence. |

## Current Encoder Decision

Full-coverage LatentFM encoder set:

```text
stack
scldm
scfoundation
```

Reason:

- all three cover 23 datasets across the current benchmark categories;
- NicheFormer and TranscriptFormer currently cover only 4 chempert datasets;
- NicheFormer/TranscriptFormer should appear as limited-coverage new-model
  benchmark evidence, not as full cross-dataset LatentFM encoders yet.

## Conditioning Policy

Genetic perturbations:

- current primary source: `scgpt_embed_gene`;
- CellNavi remains useful for sensitivity/reproduction but should not be mixed
  silently into the same formal comparison matrix;
- formal RawFM, CoupledFM, and LatentFM comparisons should record the same
  genetic perturbation embedding source when compared directly.

Drug perturbations:

- current short-term cache: SciPlex label identity cache under
  `dataset/drug_cache/sciplex_label_identity_561`;
- this is useful for in-distribution diagnostic conditioning;
- it is not a molecular zero-shot drug encoder and should not be described as
  State/molecular generalization.

## LatentFM Training Constraints

- OT unit is the microbatch. Do not enlarge microbatch blindly, because many
  conditions have only around 64 usable control/GT cells.
- Larger velocity MLP capacity was tested and did not solve weak perturbation
  direction metrics by itself.
- Low GPU utilization is expected for small LatentFM branches; improve
  throughput by safe scheduling, batching, and colocation only when CPU/RAM/GPU
  headroom is verified.
- Full-data LatentFM runs should save `latest.pt` periodically; runs predating
  the long-epoch checkpoint fix should not be treated as resumable if
  interrupted before epoch-end.

## Objective Families Tested

| Objective / branch family | Main finding |
|---|---|
| Baseline flow matching | Can learn endpoint/distribution structure, but perturbation-specific pp is weak. |
| Larger velocity MLP | Does not solve pp; capacity alone is not the main bottleneck. |
| Endpoint mean-delta loss | Improves MMD/pc, but pp and unseen multi-composition remain weak. |
| Gene-only filtering | Drug fallback is not the only bottleneck. |
| Explicit SciPlex label-drug cache | Improves traceability, not formal unseen-drug biology. |
| Multi-pool condition encoder | Improves MMD/pc, not enough for perturbation-specific direction. |
| Perturbation-residual direction loss | Residual frame alone is insufficient. |
| Condition-contrastive residual loss | Did not solve unseen multi-composition. |
| Composition delta and strong-composition branches | Improve selected multi-unseen slices but can hurt aggregate pp, gene pp, drug pp, or MMD. |
| Relational residual branch | Rejected as mainline: MMD improved but aggregate perturbation, gene-family, and multi-split gates failed. |
| 12-run endpoint/composition/condition-delta strategy gate | Completed with no strict repeat candidate; do not scale any branch directly. |
| Condition-prior teacher dose table | Completed with diagnostic candidates only; train-single priors are useful but not yet internalized as a promotable velocity objective. |
| Explicit head injection | Diagnostic only; slightly improves MMD/overall pp/unseen2 but no repeat candidate. |
| Active condition-prior additive-head smoke | Tests whether directly supervising additive condition-delta atoms fixes the injected-head decomposition failure; pending. |

## Historical Active LatentFM Branch

The following branch record is kept as historical provenance from 2026-06-19.
It is not the current project-state entrypoint. Use `goal.md`,
`docs/PROJECT_REVIEW.md`, and `docs/EXPERIMENT_INDEX.md` for the latest gate
state.

Run root:

```text
/data/cyx/1030/scLatent/runs/latentfm_condition_prior_additive_head_20260619
```

Branch:

```text
scf_prioradd005_prior010_inject_e2_4k
```

Purpose:

Test one narrow mechanism: whether train-single prior supervision of
`predict_additive_condition_delta` can make the condition-delta head's additive
atom surface useful on top of the prior010 injected-head diagnostic branch.

Configuration:

```text
condition_prior_delta_loss_weight=0.10
condition_delta_head_use_in_model=True
condition_prior_additive_delta_loss_weight=0.05
```

Automation:

```text
/data/cyx/1030/scLatent/runs/latentfm_condition_prior_additive_head_posthoc_20260619/RUN_STATUS.md
/data/cyx/1030/scLatent/runs/latentfm_condition_prior_additive_head_summary_20260619/RUN_STATUS.md
/data/cyx/1030/scLatent/runs/latentfm_condition_prior_additive_head_one_shot_1656_20260619/RUN_STATUS.md
/data/cyx/1030/scLatent/runs/latentfm_condition_prior_additive_head_one_shot_1730_20260619/RUN_STATUS.md
```

Strict gate:

- improve over `scf_prior010_inject_e2_4k` on aggregate pp and unseen2 pp;
- keep MMD within 15 percent of primary scFoundation;
- preserve family-gene pp and avoid worsening drug-family behavior;
- report Wessels unseen2 decomposition metrics, not only scalar pp;
- treat one-metric wins as diagnostics, not mainline model success.

## Model Success Bar

LatentFM is not considered paper-ready until a branch shows:

- stable MMD/pp/pc;
- split/family robustness, especially unseen multi-perturbation behavior;
- condition-level top improved/failed perturbation tables;
- interpretable perturbation structure rather than only a global scalar win;
- repeat/deeper validation if a candidate passes the first strict gate.
