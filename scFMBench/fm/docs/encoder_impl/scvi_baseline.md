# scVI（dataset-fitted baseline）

## 角色

- **Dataset-level fit**：对 **单个** `AnnData`（**control + gt 全细胞**）训练一套 `scvi.model.SCVI`，仅调用 **`get_latent_representation()`** 导出 cell latent。
- **非 zeroshot**：每 dataset 独立训练；与预训练 foundation 模型不对等比较。
- **无扰动条件**：不把扰动元数据作为 SCVI 的条件输入；默认仅 `batch_key`（可显式传入 `obs` 列，否则注入单批次 dummy 列）。

## 数据口径（重要）

scVI 训练应使用 **raw counts**：

- 默认从 **`adata.layers['counts']`** 读入（`SCVI.setup_anndata(..., layer='counts')`）。
- adapter 在写入 `setup_anndata` 之前对 **命名 counts layer 与 `counts_layer=None` 时的 `X`** 做同一套轻量 `_validate_count_matrix`（非负、有限、**仅对正数项**看是否类整数；若正数 `max <= 20` 且正数中「接近整数」比例 `< 0.88` 则拒收——拦截典型 log1p 小数矩阵；正数 `max > 25` 时再对**全体**做整数占比 `< 0.9` 的二次检查）。明显非 count 的矩阵会 **直接报错**。
- **禁止**在仅有 log1p `X` 时静默训练；若主 h5ad 无 counts，请：
  - 自行写入 `layers['counts']`，或
  - 使用 runner：`tools/run_dataset_fitted_baseline.py --scvi-counts-from-h5ad <raw.h5ad>`（与 `adapters.dataset_fitted_io.attach_counts_from_h5ad` 对齐 obs/var 交集后写入目标 layer）。

测试场景下可将 **`counts_layer=None`** 并保证 `adata.X` 为类整数计数，adapter 会做简易校验；**正式 benchmark 不推荐**。

## log1p 输入（方案 C：`gene_likelihood='normal'`）

当 benchmark 侧只有 `log1p(normalize_total(...))` 的 **`adata.X`**（或某一 layer 中同样是该口径的矩阵）时，可显式传入 **`encode(..., input_is_log1p=True)`**：

- **不**调用 `_validate_count_matrix`，**不**构造伪计数、**不**对表达做 `expm1` / 取整；数据原样进入 `SCVI.setup_anndata`（`layer=None` 用 `X`，否则 `layer=<counts_layer 实参>`）。
- 此时 **`log1p_gene_likelihood`**（默认 **`"normal"`**）会作为传给 `SCVI(..., gene_likelihood=...)` 的解码头；与 scvi-tools 官方的 Gaussian / continuous 表达路径一致。
- 若同时传入 **`gene_likelihood='nb'` 或 `'zinb'`**，adapter 会 **`ValueError`**：NB/ZINB 假设的是计数生成过程，而 log1p 归一化后是连续值，混用会歪曲生成语义。应改为默认的 `log1p_gene_likelihood='normal'`，或改回真 counts 并设 **`input_is_log1p=False`**。
- **何时应回退到真 counts**：需要与标准 scRNA 计数建模（NB/ZINB）、或与仅支持整数输入的下游严格对齐时，应提供 `layers['counts']`（或 `counts_layer=None` 的整数 `X`）并 **`input_is_log1p=False`**。

CLI：`tools/run_dataset_fitted_baseline.py --baseline scvi --scvi-input-is-log1p [--scvi-log1p-gene-likelihood normal]`；若表达在 **`X`** 而非默认的 `layers['counts']`，需将 **`--scvi-counts-layer`** 设为 **`""`**（空字符串，与 `counts_layer=None` 等价）或保证 layer 名指向 log1p 矩阵。

## 环境与依赖

- 推荐：`$SCFM_ENVS_ROOT/scldm/bin/python`（**`scvi-tools==1.2.0`** + `anndata==0.10.9`，见 [`../env_map.md`](../env_map.md)）。
- 上游源码镜像：`third_party/dataset_fitted_baseline/scvi`（可选查阅；运行时使用已安装 `scvi-tools`）。

## API

- 模块：`adapters.scvi_baseline.encoder.encode`
- 关键参数：`n_latent`、`max_epochs`、`counts_layer`、`input_is_log1p`、`log1p_gene_likelihood`（log1p 路径）、`batch_key`、`train_kwargs`（如 `accelerator`、`enable_progress_bar`）、`model_save_dir`（可选保存模型目录）。
- `meta` 含：`fit_method="scvi"`、`fit_scope="dataset"`、`encoder_role="ExpressionOnlyEncoder"`、`force_pert_effective=False`、`pert_source=None`、`gene_likelihood`（实际传入 SCVI 的值）、`input_is_log1p`、`log1p_strategy`、`data_source`（`counts_layer` / `X_counts` / `X_log1p`）、`counts_layer`、`counts_from_X`、`batch_key` 等；若经 `attach_counts_from_h5ad` 对齐，runner 会将 `dataset_fitted_align` 写入 `meta.json`（obs/var 交集裁剪统计）。

## 导出

```bash
cd <delivery_root>
python data/scFM/fm/tools/run_dataset_fitted_baseline.py \
  --baseline scvi --biflow-dir /path/to/biFlow_data --dataset-stem Adamson \
  --scvi-max-epochs 400 --scvi-n-latent 10 \
  --scvi-counts-from-h5ad /path/to/raw/Adamson.h5ad
```

输出：`exports/dataset_fitted/scvi/<stem>/latent.npy`、`meta.json`；`--scvi-save-model-dir` 可另存训练好的模型。

## Smoke

```bash
$SCFM_ENVS_ROOT/scldm/bin/python data/scFM/fm/smoke/test_scvi_baseline.py
```

使用 **合成** control+gt 合并数据 + **1 epoch CPU**，验证 setup / train / latent shape。
