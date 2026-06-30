# PCA（dataset-fitted baseline）

## 角色

- **Dataset-level fit**：对传入的 **单个** `AnnData` 在 **整张 `adata.X` 上直接拟合** `PCA`，再对同一矩阵做 `transform`。
- **非 zeroshot**：与预训练 foundation encoder 不同，每个 dataset 一套 PCA 参数；**不得**与 zeroshot 模型做“同一起点”公平对比。
- **无扰动条件**：不使用 `obs['perturbation']` 等作为模型输入；`obsm['pert_var_idx']` 若存在，本 baseline **不读取**。
- **实现位置**：实际 PCA 逻辑在 `third_party/dataset_fitted_baseline/PCA/pca.py`，`adapters/pca_baseline/encoder.py` 仅负责薄封装与 `meta`。

## 数据口径

与 benchmark 主协议一致：默认 **`adata.X` 已是 log1p(normalize_total(...))** 的表达矩阵，PCA 直接在其上计算。

对于 `DE5000_bench`，一个数据集本身就是一个 h5ad（例如 `data/raw/DE5000_bench/Adamson.h5ad`）。若该 bench 文件的 `X` 为空，则先用 `adapters.dataset_fitted_io.attach_expression_from_h5ad(...)` 或 runner 的 `--pca-expression-from-h5ad data/raw/DE5000/<dataset>.h5ad` 挂载表达矩阵，再对这**整个输入文件的全部细胞**一次性降维。

## API

- 模块：`adapters.pca_baseline.encoder.encode`
- 返回：`(latent, meta)`，`latent` 形状 `(n_obs, n_components_actual)`；`n_components_actual = min(n_obs, n_vars, n_components_requested)`。
- `meta` 含：`encoder_role="ExpressionOnlyEncoder"`、`fit_scope="dataset"`、`fit_method="pca"`、`force_pert_effective=False`、`pert_source=None`、`explained_variance_ratio` 等。

## 数据加载

- biFlow 成对 h5ad：`adapters.dataset_fitted_io.load_biflow_merged_anndata(biflow_dir, dataset_stem)`。
- 单文件数据集：`tools/run_dataset_fitted_baseline.py --baseline pca --adata data/raw/DE5000_bench/<dataset>.h5ad --pca-expression-from-h5ad data/raw/DE5000/<dataset>.h5ad`。
- biFlow 成对输入也仍支持；runner 会先合并成一个 `AnnData`，再对合并后的整集 PCA。

## Smoke

```bash
cd <delivery_root>
$SCFM_ENVS_ROOT/scdfm/bin/python data/scFM/fm/smoke/test_pca_baseline.py
```

只要环境含 `sklearn` 即可（示例用 `scdfm`；`scfoundation`、`scldm` 等亦可）。

默认读取 `data/raw/DE5000_bench/Adamson.h5ad`，再从 `data/raw/DE5000/Adamson.h5ad` 对齐恢复表达矩阵，并对整张 h5ad 做 PCA。
