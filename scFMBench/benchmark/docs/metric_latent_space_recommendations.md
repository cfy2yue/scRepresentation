# Metric interpretation: raw vs pca128 latent space

Benchmarks write metrics under `output/metrics/<model>/<dataset>/{raw,pca128}/`. The **raw** branch uses exported model latents as stored in `latent.npy`. The **pca128** branch applies a **per-task** pipeline: `StandardScaler` fit on a chosen cell subset, then `sklearn.decomposition.PCA` with `K = min(128, latent_dim)`, transform on all cells. For chemical-perturbation datasets with enough control cells, PCs are fit on **control cells only** (`control` / `is_control`), then all cells are projected—mirroring a no–ground-truth-perturbation view at fit time. Atlas datasets use **all cells** for the fit.

## Indicator × latent_space applicability

| 指标 | raw 跨模型可比 | pca128 跨模型可比 | 备注 |
|------|----------------|-------------------|------|
| A1 NMI/ARI (Leiden) | OK | OK | kNN 图，对单调缩放不变 |
| A2 cLISI | OK | OK | kNN 图统计 |
| A3 iLISI | OK | OK | kNN 图统计 |
| A4 graph_connectivity | OK | OK | 纯图结构 |
| A5 trustworthiness | 受 d 弱影响 | 更公平 | 高维距离集中 |
| G1 PR / erank / k90 | 不公平（上限与 d 相关） | 公平 | 高维系统性偏大 |
| G2 knn_label_consistency | OK | OK | kNN |
| G3 anisotropy (λ_max/tr) | 不公平 | 公平 | d 大时比值偏小 |
| G4 silhouette | 弱不公平 | 公平 | 欧氏量纲 |
| G5 noise stability | 弱不公平 | 公平 | 距离集中 |
| G6 Laplacian energy | 弱不公平 | 公平 | |
| LDM_proxy | 不公平 | 公平 | 依赖 G1/G3/G5 |
| centroid L2 | 高维不可比 | 公平（在 PC 子空间） | 见下文 |
| OT EMD | 不可比 | 公平 | 欧氏地面度量 |
| xCellLine 相关 | 不可比 | 公平 | |

**展示原则**

- 排名/批效应/标签聚类类（A1–A4、G2、A5）：主图可用 **raw**。
- 几何/距离类（G1、G3–G6、LDM_proxy、centroid L2、OT、xCellLine）：主图用 **pca128**。
- raw 下的 L2/EMD 类指标仅建议作 **同一模型内** 的趋势阅读。

## Semantic notes after PCA

- kNN 与 Leiden/LISI 等 **纯图**指标在缩放/线性投影下通常仍具可比性，但与 raw 分支数值不必一致。
- **centroid L2（pca128）**：质心位移是在 **投影后的 128 维（或 K 维）空间**度量的，表示在“按控制（或全细胞）拟合的 PC 基”下的偏移，**不等于**原始高维 latent 里的欧氏位移。
- **Trustworthiness** 等基于距离的 trust 指标在 pca128 上更可跨模型比较，但仍受局部邻域结构影响。

## Artifacts

- 汇总表：`output/metrics/summary_all_raw.csv`, `summary_all_pca128.csv`, 以及含 `latent_space` 列的长表 `summary_all.csv`（由 `benchmark/cli/aggregate_report.py --write-scfm-benchmark-csvs` 生成）。
- 单任务 `summary.json` 中 pca128 运行会记录 `pca128_fit_scope`, `pca128_n_fit_cells`, `pca128_explained_variance`, `pca128_k` 等字段。
