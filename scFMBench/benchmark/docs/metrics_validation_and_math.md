# Metrics validation and math notes

Date: 2026-04-27

This note records the current validation state for the newly added metric code, and explains the mathematical meaning of each implemented method. It is meant as a catch-up document before continuing the benchmark build.

## Current status

Implemented metric modules:

- `tools/metrics/atlas_scib.py`: atlas integration and preservation metrics A1-A6.
- `tools/metrics/tx_eval_ported/`: Tx-Evaluation code paths ported without Lightning or WandB.
- `tools/metrics/perturb_geom.py`: latent perturbation geometry summaries.
- `tools/metrics/perturb_xcellline.py`: per-cell-line perturbation summaries for multi-cell-line chemical screens.
- `tools/metrics/geometry.py`: Tier-2 geometry metrics G1-G6 and an LDM-readiness proxy.
- `tools/metrics/post_process.py`: raw / center / center-scale / TVN latent post-processing.
- `tools/aggregate_report.py`: metric JSON aggregation to JSONL/CSV/stdout.
- `smoke/test_metrics_pipeline.py`: synthetic end-to-end smoke for post-processing, geometry, perturbation summaries, xCellLine summaries, and report aggregation.
- `tools/metrics/__init__.py`: package exports for the modules above.

Validation run:

```bash
cd <delivery_root>/scFM
$SCFM_ENVS_ROOT/scdfm/bin/python - <<'PY'
# Synthetic smoke covering atlas_scib, perturb_geom, linear probe, kNN,
# accuracy@k top-5 cap, and reconstruction metrics.
PY
```

Result: passed.

Additional P8/P9/P10 validation run:

```bash
cd <delivery_root>/scFM
$SCFM_ENVS_ROOT/scdfm/bin/python smoke/test_metrics_pipeline.py
```

Result: passed (`metrics pipeline smoke OK`).

Observed warnings:

- JAX reported no CUDA-enabled `jaxlib` and fell back to CPU. This affects performance only for the synthetic smoke; it did not block metric execution.
- `torchmetrics.SpearmanCorrCoef` warns that it buffers predictions/targets. For large reconstruction tests, compute in chunks or avoid Spearman if memory becomes a problem.

## Important fixes made during validation

1. `accuracy_at_k` now caps `k` by the number of classes. This avoids failure when asking for top-5 accuracy on datasets with fewer than 5 labels.
2. `perturb_geom.py` now names the OT output `emd_*` instead of `emd2_*`. With Euclidean ground cost, POT `emd2` returns the optimal transport objective for EMD/W1, not W2 squared.
3. `StructuralTranscriptomeDistance` now builds boolean indices on the same device as the tensors, so it is safer for CPU and GPU tensors.
4. `scib_metrics.graph_connectivity` has a local compatibility implementation because the installed pandas version no longer has `pd.value_counts`.
5. `bmdb.py` treats `geomloss` as optional. Importing the package works without it; only energy-distance magnitude metrics require it.
6. `post_process.py` TVN order now matches the Tx-Evaluation source: global control center-scale, PCA fit on controls, global control center-scale, then optional per-batch whitening.

## Inputs expected by the metrics

For atlas metrics:

- `latent.npy`: shape `(n_cells, d)`, row-aligned to obs.
- `obs.parquet`: must contain `batch` and `cell_type` by default.
- Optional `adata-ref`: same cell order, used only for expression-reference UMAP panels.

For perturbation geometry:

- `latent`: shape `(n_cells, d)`.
- `obs`: contains a perturbation column such as `gene`, `perturbation`, or `condition`.
- `obs['is_control']`: boolean control flag. If present, all controls are pooled into one control centroid.

For Tx-Evaluation port:

- `WeightedKNNClassifier`: train/test embeddings and labels, either pre-encoded tensors or string labels.
- `fit_linear_probe`: train/validation embeddings and labels.
- `fit_decoder_mlp`: embedding matrix and expression target matrix.
- `bmdb` functions: perturbation metadata and optional external BMDB relationship files, depending on the benchmark.

For geometry metrics:

- `latent`: shape `(n_cells, d)`.
- Optional `obs`: row-aligned metadata.
- Optional `label_col`: enables G2 local label consistency and G4 silhouette.
- Optional `batch_col`: currently records number of batches in output; batch-specific geometry is not yet computed.

For post-processing:

- `embeddings`: shape `(n_cells, d)`.
- `metadata`: row-aligned table.
- `pert_col` and `control_key`: define controls.
- Optional `batch_col`: used by centering/center-scale per batch and by final TVN whitening.

## Atlas metrics A1-A6

Let each cell have latent vector \(z_i \in \mathbb{R}^d\), biological label \(y_i\), and batch label \(b_i\). Build a kNN graph \(G_k(Z)\) in latent space.

### A1: NMI / ARI after Leiden

Leiden clustering assigns cluster labels \(\hat{c}_i\) on the latent kNN graph. A1 compares \(\hat{c}\) with biological labels \(y\).

Normalized Mutual Information:

\[
\mathrm{NMI}(Y, C) =
\frac{2 I(Y; C)}{H(Y) + H(C)}
\]

Adjusted Rand Index:

\[
\mathrm{ARI} =
\frac{\mathrm{RI} - \mathbb{E}[\mathrm{RI}]}
{1 - \mathbb{E}[\mathrm{RI}]}
\]

Interpretation:

- Higher NMI/ARI means latent neighborhoods preserve biological cell-type structure.
- These are not batch-mixing metrics.

### A2: cLISI

cLISI is Local Inverse Simpson's Index over cell-type labels in local neighborhoods. For each cell \(i\), let \(p_{i,c}\) be the fraction or smoothed probability of neighbors with cell type \(c\).

\[
\mathrm{LISI}_i =
\frac{1}{\sum_c p_{i,c}^2}
\]

For cell-type conservation, the desired behavior is low local label diversity within a biological cell type. `scib_metrics.clisi_knn` reports a normalized conservation score where higher is better.

Interpretation:

- Higher cLISI score means local neighborhoods are biologically coherent.

### A3: iLISI

iLISI uses the same inverse Simpson index, but over batch labels:

\[
\mathrm{iLISI}_i =
\frac{1}{\sum_b p_{i,b}^2}
\]

Interpretation:

- Higher iLISI means better batch mixing in local neighborhoods.
- If only one batch exists, the implementation returns `None` for iLISI.

### A4: Graph connectivity

For each biological label \(c\), restrict the latent kNN graph to cells with \(y_i=c\). Let the largest connected component have size \(L_c\), and the full subgraph have \(N_c\) cells.

\[
\mathrm{GC}
= \frac{1}{|\mathcal{C}|}
\sum_{c \in \mathcal{C}}
\frac{L_c}{N_c}
\]

Interpretation:

- Higher graph connectivity means cells of the same biological label remain connected after embedding.
- This is useful for detecting fragmented cell types.

### A5: Trustworthiness of latent UMAP

The code computes a 2D UMAP of latent vectors and then uses sklearn trustworthiness between original latent space and the 2D projection.

For cell \(i\), let \(U_k(i)\) be points that appear among the k nearest neighbors in low-dimensional space but not in high-dimensional space. Let \(r(i,j)\) be the rank of \(j\) in high-dimensional space. Trustworthiness is:

\[
T(k) =
1 -
\frac{2}{n k (2n - 3k - 1)}
\sum_i \sum_{j \in U_k(i)}
(r(i,j) - k)
\]

Interpretation:

- Higher trustworthiness means the UMAP panel is less visually misleading relative to latent kNN structure.
- This is a visualization-faithfulness metric, not a biological metric.

### A6: UMAP panels

The figure code creates:

- expression-reference UMAP from HVG + PCA + neighbors + UMAP;
- latent UMAPs from one or more `latent.npy` arrays;
- panels colored by `cell_type`, `batch`, and optionally `compartment`.

Important protocol point:

- The atlas staging files are already prepared. Do not normalize atlas again inside the metric runner. `adata-ref` is only used for figure reference.

## Tx-Evaluation port

The port keeps core evaluation logic but removes Lightning/WandB runtime requirements.

### Weighted kNN

Given train embeddings \(x_j\), train labels \(y_j\), and test embedding \(x\), retrieve the top \(k\) neighbors.

For cosine mode:

\[
s_j = \frac{x^\top x_j}{\|x\|\|x_j\|}
\]

Neighbor votes are temperature-weighted:

\[
w_j = \exp(s_j / T)
\]

Class score:

\[
\mathrm{score}(c \mid x)
= \sum_{j \in \mathcal{N}_k(x)}
w_j \mathbf{1}[y_j=c]
\]

Prediction ranks classes by score. Reported values are top-1 and top-5 accuracy in percent.

For Euclidean mode, the similarity used by the code is:

\[
s_j = \frac{1}{\|x - x_j\|_2 + \epsilon}
\]

### Linear probe

`fit_linear_probe` trains a single linear classifier:

\[
\hat{p}(y=c \mid z)
= \mathrm{softmax}(Wz + a)_c
\]

Loss:

\[
\mathcal{L}
= -\frac{1}{n}
\sum_i \log \hat{p}(y_i \mid z_i)
\]

Interpretation:

- High accuracy means cell labels are linearly decodable from the latent.
- This measures linear separability, not necessarily neighborhood quality.

### Decoder reconstruction

`fit_decoder_mlp` trains an MLP \(f_\theta\) from latent embedding \(z_i\) to expression target \(x_i\):

\[
\hat{x}_i = f_\theta(z_i)
\]

Supported losses:

\[
\mathrm{MSE}
= \frac{1}{ng}
\sum_{i,j}(\hat{x}_{ij} - x_{ij})^2
\]

\[
\mathrm{MAE}
= \frac{1}{ng}
\sum_{i,j}|\hat{x}_{ij} - x_{ij}|
\]

Reconstruction report includes MSE, MAE, average Pearson, average Spearman, and R-squared:

\[
R^2 = 1 -
\frac{\sum_{i,j}(x_{ij} - \hat{x}_{ij})^2}
{\sum_{i,j}(x_{ij} - \bar{x})^2}
\]

Interpretation:

- This tests whether transcriptome information is recoverable from the latent.
- Training-set metrics are easy to overfit; use a held-out split for real comparison.

### Structural transcriptome distance

For each batch \(m\), the metric centers predictions and targets by the mean control transcriptome in that batch:

\[
\tilde{x}_{i} = x_i - \mu^{\mathrm{ctrl}}_{m(i)}
\]

\[
\tilde{\hat{x}}_{i} = \hat{x}_i - \hat{\mu}^{\mathrm{ctrl}}_{m(i)}
\]

Then it computes a Frobenius-style normalized integrity:

\[
D_m =
\frac{\|\tilde{\hat{X}}_m - \tilde{X}_m\|_F}{n_m}
\]

\[
M_m =
\frac{\|\tilde{X}_m\|_F}{n_m}
\]

\[
\mathrm{Integrity}
= 1 - \frac{\frac{1}{M}\sum_m D_m}{2 \cdot \frac{1}{M}\sum_m M_m}
\]

Interpretation:

- Higher integrity means reconstructed perturbation structure is closer to target after batch/control centering.
- This is sensitive to the definition of controls and batches.

### BMDB / perturbation signal metrics

The port includes the upstream BMDB-style helpers:

Perturbation signal consistency:

For perturbation \(p\), collect all replicate embeddings \(Z_p = \{z_1,\dots,z_r\}\). The score is average pairwise cosine similarity:

\[
\mathrm{Consistency}(p)
=
\frac{2}{r(r-1)}
\sum_{a<b}
\cos(z_a, z_b)
\]

A p-value can be computed by comparing to a sorted null distribution from negative-control perturbations.

Perturbation signal magnitude:

For perturbation cloud \(Z_p\) and control cloud \(Z_0\), the upstream code uses `geomloss.SamplesLoss("energy")` and multiplies by 2. Conceptually this measures distributional separation between perturbed cells and controls. This path requires `geomloss`.

Known-relationship retrieval:

For aggregated perturbation embeddings, the code computes pairwise similarities and asks whether known related perturbations are retrieved among top-ranked neighbors. This depends on external BMDB relationship files such as CORUM, Reactome, HuMAP, SIGNOR, and StringDB.

## Perturbation geometry

### Pooled control centroid

For control cells:

\[
\mu_0 =
\frac{1}{|C_0|}
\sum_{i \in C_0} z_i
\]

For perturbation \(p\):

\[
\mu_p =
\frac{1}{|C_p|}
\sum_{i \in C_p} z_i
\]

### Centroid shift

Per perturbation:

\[
\Delta_p =
\|\mu_p - \mu_0\|_2
\]

The module reports each \(\Delta_p\), plus mean and median over perturbations.

Interpretation:

- Larger values mean the perturbation induces a stronger shift from controls in latent space.
- This does not by itself say whether the shift is biologically correct.

### EMD / OT delta

For control empirical distribution:

\[
P_0 = \frac{1}{n_0}\sum_i \delta_{z_i^{(0)}}
\]

For perturbation distribution:

\[
P_p = \frac{1}{n_p}\sum_j \delta_{z_j^{(p)}}
\]

With Euclidean ground cost:

\[
c(i,j) = \|z_i^{(0)} - z_j^{(p)}\|_2
\]

The optimal transport objective is:

\[
\mathrm{EMD}(P_0, P_p)
=
\min_{\pi \in \Pi(P_0, P_p)}
\sum_{i,j}\pi_{ij} c(i,j)
\]

The implementation subsamples both clouds before building the pairwise cost matrix to control memory. If POT is unavailable, the function returns `None` for EMD fields.

Interpretation:

- EMD captures whole-cloud distributional movement, not just centroid movement.
- It can detect spread/shape changes that centroid distance misses.

## xCellLine perturbation summaries

`perturb_xcellline.py` applies the perturbation geometry metrics within each cell line and then aggregates across cell lines.

Let \(l\) index a cell line. Within cell line \(l\), define controls \(C_{0,l}\) and perturbation cells \(C_{p,l}\):

\[
\mu_{0,l}
=
\frac{1}{|C_{0,l}|}
\sum_{i \in C_{0,l}} z_i
\]

\[
\mu_{p,l}
=
\frac{1}{|C_{p,l}|}
\sum_{i \in C_{p,l}} z_i
\]

Per-cell-line centroid shift:

\[
\Delta_{p,l}
=
\|\mu_{p,l} - \mu_{0,l}\|_2
\]

The cross-line summary is the mean over cell lines with valid outputs:

\[
\overline{\Delta}_{\mathrm{xCellLine}}
=
\frac{1}{|\mathcal{L}_{valid}|}
\sum_{l \in \mathcal{L}_{valid}}
\mathrm{mean}_{p}(\Delta_{p,l})
\]

The EMD summary is analogous, but uses the within-line control and perturbation empirical distributions.

Interpretation:

- This avoids pooling cell lines before measuring perturbation displacement.
- It is most useful for `sciplex3_xCellLine`, where the same chemical should be evaluated inside each cell-line background.

## Geometry metrics G1-G6

Let \(Z \in \mathbb{R}^{n \times d}\) be the latent matrix after optional post-processing. Most geometry metrics are subsampled for speed on large datasets.

### G1: Spectrum / participation ratio / effective rank

After centering \(Z\), compute PCA explained variance ratios \(\rho_1,\dots,\rho_m\).

The number of PCs required to explain 90% variance is:

\[
k_{90}
=
\min \left\{ k :
\sum_{r=1}^{k} \rho_r \ge 0.9
\right\}
\]

Participation ratio:

\[
\mathrm{PR}
=
\frac{\left(\sum_r \rho_r\right)^2}
{\sum_r \rho_r^2}
\]

Let covariance eigenvalues be \(\lambda_1,\dots,\lambda_d\), and \(p_j=\lambda_j / \sum_k \lambda_k\). Shannon effective rank:

\[
\mathrm{erank}
=
\exp\left(
-\sum_j p_j \log p_j
\right)
\]

Interpretation:

- Larger PR/effective rank means the latent uses more dimensions.
- Very low rank suggests representation collapse; extremely diffuse rank may indicate noise.

### G2: Local label consistency

For each cell \(i\), let \(\mathcal{N}_k(i)\) be its kNN set in latent space and \(y_i\) a label such as cell type.

\[
\mathrm{LC}
=
\frac{1}{n}
\sum_i
\frac{1}{k}
\sum_{j \in \mathcal{N}_k(i)}
\mathbf{1}[y_j = y_i]
\]

Interpretation:

- Higher local consistency means neighbors tend to share labels.
- This complements cLISI but uses a simple explicit kNN majority-style statistic.

### G3: Isotropy / anisotropy

Let \(C\) be the covariance of centered latent vectors and \(\lambda_{\max}\) the largest eigenvalue.

\[
\mathrm{anisotropy}
=
\frac{\lambda_{\max}}
{\mathrm{tr}(C)}
\]

The code also reports an eigenvalue condition proxy:

\[
\kappa
=
\frac{\lambda_{\max}}
{\lambda_{\min} + 10^{-12}}
\]

Interpretation:

- Lower \(\lambda_{\max}/\mathrm{tr}(C)\) means variance is less concentrated in one dominant direction.
- Very high condition number may indicate a poorly conditioned latent distribution.

### G4: Label silhouette

For cell \(i\), let \(a(i)\) be mean distance to cells with the same label and \(b(i)\) the smallest mean distance to a different-label group.

\[
s(i)
=
\frac{b(i)-a(i)}
{\max(a(i), b(i))}
\]

The metric reports:

\[
\mathrm{Silhouette}
=
\frac{1}{n}\sum_i s(i)
\]

Interpretation:

- Higher silhouette means labels are more separated in Euclidean latent space.
- It can penalize continuous trajectories where sharp clusters are not expected.

### G5: Noise stability

Subsample cells and compute pairwise distance vector \(D\). Add Gaussian noise:

\[
z'_i
=
z_i + \epsilon_i,
\quad
\epsilon_i \sim \mathcal{N}(0, \sigma^2 \cdot \mathrm{std}(Z)^2)
\]

Compute the post-noise pairwise distance vector \(D'\). The metric is Spearman rank correlation:

\[
\mathrm{Stability}
=
\rho_{\mathrm{Spearman}}(D, D')
\]

Interpretation:

- Higher value means local/global distance ordering is stable to small perturbations.
- This is a proxy for smooth geometry, useful for downstream generative modeling.

### G6: kNN graph Dirichlet energy

Build a kNN graph. For edge \((i,j)\), use Gaussian weight:

\[
w_{ij}
=
\exp\left(
-\frac{\|z_i-z_j\|_2^2}{2\sigma^2}
\right)
\]

where \(\sigma\) is the median k-th-neighbor distance. The reported energy is:

\[
E
=
\frac{1}{n}
\sum_i
\sum_{j \in \mathcal{N}_k(i)}
\frac{1}{2} w_{ij}\|z_i-z_j\|_2^2
\]

Interpretation:

- Lower energy means neighboring points change smoothly over the graph.
- Because the graph is built from \(Z\) itself, this is mainly a scale/smoothness diagnostic rather than a supervised quality score.

### LDM-readiness proxy

The current proxy is a heuristic composite in \([0,1]\). It combines:

\[
S_5
=
\mathrm{clip}\left(
\frac{\mathrm{G5}+1}{2},
0,1
\right)
\]

\[
S_1
=
\tanh(\mathrm{PR}/10)
\]

\[
S_3
=
\frac{1}{1+\exp(25(\mathrm{anisotropy}-0.35))}
\]

Then:

\[
\mathrm{LDM\_proxy}
=
\frac{S_5 + S_1 + S_3}{3}
\]

Interpretation:

- This is not a validated scientific metric.
- It is a triage score: stable distances, non-collapsed spectrum, and not-too-anisotropic covariance are treated as favorable for latent diffusion / flow-model training.

## Post-processing methods

Let \(z_i\) be a cell embedding. Let \(c(i)\) indicate whether cell \(i\) is a control, and \(m(i)\) be a batch if provided.

### Center

Global control centering:

\[
z'_i = z_i - \mu_0,
\quad
\mu_0 =
\frac{1}{|C_0|}
\sum_{j \in C_0} z_j
\]

Batch-wise control centering:

\[
z'_i = z_i - \mu_{0,m(i)}
\]

### Center-scale

Fit `StandardScaler` on controls and transform all cells:

\[
z'_i
=
\frac{z_i-\mu_0}{s_0}
\]

where \(\mu_0\) and \(s_0\) are control mean and control standard deviation. If `batch_col` is provided, this fit/transform is done per batch.

### TVN

The implemented TVN follows the upstream Tx-Evaluation order:

1. global center-scale using controls;
2. PCA fit on controls, transform all cells;
3. global center-scale again using controls;
4. optional per-batch whitening using control covariance.

For the final per-batch whitening:

\[
\Sigma_{0,m}
=
\mathrm{Cov}(Z_{0,m}) + 0.5I
\]

\[
z'_i
=
z_i \Sigma_{0,m(i)}^{-1/2}
\]

Interpretation:

- Centering removes control baseline.
- Center-scaling standardizes dimensions using controls.
- TVN additionally normalizes typical control variation, reducing directions dominated by control covariance.

## Aggregation and smoke test

`tools/aggregate_report.py` loads metric JSON files and returns a row per JSON. Nested fields are flattened only when CSV output uses `pandas.json_normalize`.

The current synthetic smoke test:

```bash
cd <delivery_root>/scFM
$SCFM_ENVS_ROOT/scdfm/bin/python smoke/test_metrics_pipeline.py
```

It validates:

- center / center-scale / TVN shape preservation;
- G1-G6 + LDM proxy output;
- perturbation centroid summaries;
- xCellLine summaries;
- `aggregate_report.aggregate`.

## Known limitations before continuing

- No real atlas/perturbation export has been run through these metric modules yet in this validation pass; only synthetic smoke tests were executed.
- `atlas_scib.py` figure generation can be expensive for large atlas files because it recomputes UMAP panels.
- `fit_decoder_mlp` currently reports training-set reconstruction metrics. For a real benchmark, wrap it with an explicit train/test split.
- `bmdb.py` remains close to upstream and still assumes external BMDB database files for known-relationship retrieval.
- `perturb_geom.py` and `perturb_xcellline.py` are importable Python helpers; only `geometry.py` and `aggregate_report.py` currently have CLI entry points.
- The LDM-readiness proxy is heuristic and should be treated as a triage score, not as a paper-grade metric without calibration.

## Recommended next checks

1. Run `atlas_scib.py` on one small exported atlas latent and its `obs.parquet`.
2. Run `perturb_geom.py` functions on staged Adamson and `sciplex3_K562` latent exports once a model export exists.
3. Decide whether EMD should remain W1/Euclidean, or whether the protocol wants W2 squared with squared Euclidean ground cost.
4. Add explicit CLI wrappers for `perturb_geom.py`, `perturb_xcellline.py`, and `post_process.py` if Composer should invoke them from shell rather than Python imports.
5. Run the real-export smoke once `data/staging` and encoder `exports/**/{latent.npy,obs.parquet}` are available.
