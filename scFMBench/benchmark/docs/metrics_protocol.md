# 指标协议（Metrics protocol）

本文定义 **潜空间评估** 与导出产物之间的契约，取代此前已移除的零散「benchmark 指标协议」文档。

## 输入

- **嵌入目录**：`output/embeddings/<model>/<dataset_id>/raw/`
  - `latent.npy`：`float` 数组，形状 `(n_cells, d)`，与 `obs` **按行对齐**。
  - `obs.parquet` 或 `obs.csv.gz`（或 `meta.json` 的 `obs_artifact` 所指文件）：与导出时 `AnnData.obs` 一致。
  - `meta.json`：至少包含 `model`、`n_obs`、`latent_dim`；其余字段见 [`../schema/meta.schema.json`](../schema/meta.schema.json)。

## Atlas（A1–A6）必选列

- **`batch`**：批次 / covariate，用于 iLISI 等（单一批次时部分指标退化）。
- **`cell_type`**：生物学标签，用于 NMI / Leiden、ASW 相关项。

可通过 CLI `--batch-col` / `--label-col` 覆盖列名。

## Geometry（G1–G6）

- 仅依赖 `latent.npy`；若提供 `label_col` / `batch_col` 且列存在，则计算标签与批次相关子指标。

## Perturbation

- **`pert`**：扰动标识；**`is_control`**：`bool`，存在 **pooled control** 时 centroid-shift / OT 才有定义。
- **xCellLine**：若存在 **`cell_line`**，可计算跨细胞系的摘要（见 `perturb_xcellline.summarize_xcellline_by_line`）。
- **OT（EMD）**：可选依赖 **POT**；未安装时 OT 摘要中 `emd_*` 可为 `null`。

## 随机性

- 默认种子由 `run_metrics_one.py --seed` 统一传给 atlas、geometry、perturb 和 PCA 分支；子采样类指标对种子敏感，对比时需固定。

## Fit 范围

- **Zero-shot 嵌入**：流形 / 邻居图均在 **全细胞** latent 上估计（除非脚本显式子采样）。
- **Post-process**（centering / TVN 等）：若标注为 “on controls only”，仅在 control 子集上估计统计量后再应用到全体（见 `post_process.py` 各函数文档）。

## 一键评估

- [`../cli/run_metrics_one.py`](../cli/run_metrics_one.py) → `output/metrics/<model>/<dataset_id>/{atlas,geometry,perturb,summary}.json`
