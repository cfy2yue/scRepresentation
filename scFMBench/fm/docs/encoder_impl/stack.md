# arc-stack (Stack-Large) — 实现反馈

## 修改文件清单

- [`adapters/stack/encoder.py`](../../adapters/stack/encoder.py)
- [`adapters/stack/__init__.py`](../../adapters/stack/__init__.py)（新增）

第三方代码（`third_party/stack/`）未做任何修改。

## 关键实现

- **Checkpoint 加载**：直接调用官方 `stack.model_loading.load_model_from_checkpoint(ckpt, device)`。
  `.ckpt` 路径下的 `hyper_parameters.model_config` 会重建 `StateICLModel`，然后 strip
  `"model."` 前缀加载 `state_dict`。设备默认跟随 `CUDA_VISIBLE_DEVICES` / `cuda:0`。
- **推理入口**：`InferenceMixin.get_latent_representation(adata_path=<AnnData>, genelist_path=...)`。
  Stack 的 `TestSamplerDataset` 已原生支持 in-memory `AnnData` 对象（见
  `stack.data.training.datasets.TestSamplerDataset._load_adata_metadata`），adapter 直接传入
  `adata`，避免额外磁盘往返，也不触碰原 `obsm['pert_var_idx']`。
- **基因对齐**：`genelist = basecount_1000per_15000max.pkl`（15012 个 UPPER gene symbols）。
  smoke 数据 `adata.var_names` 本身就是 gene symbol（例如 `TBC1D5`），上游已自动 `upper()`，
  因此无需指定 `gene_name_col`。32 细胞 × 3086 基因中 `found_genes/len(target_genes)` ≈
  几百到上千即可——smoke 只关心端到端打通。
- **`force_pert` 语义**：Stack 用全基因线性归约（`gene_reduction: Linear(n_genes, ...)`），
  没有 token 采样路径，扰动基因天然出现在 `X` 中参与编码。因此 `force_pert` 对 Stack 无
  副作用（adapter 置 `meta["force_pert_effective"] = False`）。为了 manifest 统一，仍然
  通过 `histogram_pert_kept` 在 `meta["pert_kept_histogram"]` 中记录 per-cell 扰动命中分布。
- **有机体过滤**：smoke adata 无 `obs['organism']`，默认 `filter_organism=True` 会落入
  “assuming all cells are valid” 的分支；adapter 把默认值改成 `False` 以避免依赖这个外部
  字段，逻辑上等价（所有细胞都保留）。
- **log1p 往返**：默认 `input_is_log1p=True` 时，adapter 对 `X` 做 `clip` + **`expm1`** 再交给 Stack，
  Stack 内部再 `torch.log1p`，与 benchmark「`adata.X` 已是 log1p」约定对齐（见 [`README.md`](README.md) 风险小结与 log1p 审计表）。
  跨 adapter 对比时须统一 `input_is_log1p`。

## Monkey-patch 列表

无。Adapter 只通过 `sys.path` 暴露 `third_party/stack/src`，并调用公共 API。

## Smoke 验证（历史记录；仓库当前无 `smoke/test_stack.py`）

此前可用命令形如：

```bash
cd <delivery_root>
CUDA_VISIBLE_DEVICES=0 $SCFM_ENVS_ROOT/stack/bin/python \
    data/scFM/fm/smoke/test_stack.py
```

最后 ~15 行输出：

```
adata=(32, 3086), perts={'ASCC3': 8, 'ASCC3+SCYL1': 8, 'SCYL1': 8, 'control': 8}
pert_var_idx shape=(32, 2)
emb shape=(32, 1600), dtype=float32, runtime=6.07s
meta = {
  "pert_kept_histogram": {
    "0": 8,
    "1": 16,
    "2": 8,
    "3+": 0
  },
  "force_pert": true,
  "force_pert_effective": false
}
force_pert is a no-op for Stack (full gene-vector reduction); skipping diff call.
stack smoke test PASSED
```

- 嵌入维度 `1600 = n_hidden (200) × token_dim (8)`，检查 `np.isfinite` 全部为真。
- 运行时 ~6 s（checkpoint load + 1 个 batch_size=8 的上采样 batch；n_cells=128 由模型配置决定）。
- `pert_kept_histogram` 与 UCE / State / scGPT 三个 adapter 一致（`{0:8, 1:16, 2:8, 3+:0}`），
  证明 `pert_var_idx` 在 adapter 层被正确消费（尽管编码器本身不使用该信息）。

## 环境变量

| 变量 | 用途 |
|------|------|
| `LATENT_BENCH_STACK_CKPT` | Stack `.ckpt`（默认 `pretrained/stack/bc_large.ckpt`） |
| `LATENT_BENCH_STACK_GENELIST` | 基因列表 pickle（默认 `pretrained/stack/basecount_1000per_15000max.pkl`） |

## log1p audit（2026-04-20）

已核对的事实（代码级别，未跑 smoke）：

1. **Stack 内部确实再做一次 `log1p`**：
   `third_party/stack/src/stack/models/core/base.py:153`：
   `features = torch.log1p(features)`（在 `forward` 入口，紧接 `observed_lib_size = features.sum(...)` 之后）。
   `inference.py::get_latent_representation` 亦在 L477 再走一遍 `torch.log1p(features)`，逻辑等价。
2. **`adata.X` 到 forward 之间没有任何额外预处理**：
   `TestSamplerDataset._load_adata_metadata` 只读 `obs['organism']`、`var.index` / `gene_name_col`、
   构造 `gene_mapping`；`load_expression_data_from_adata`（datasets.py:937）直接切片
   `adata.X[absolute_indices, :]`，`.toarray()` 后按 `gene_mapping` 拷到
   `mapped_matrix` 并转 `float32`。**没有 row-sum / normalize_total / HVG**。
   `_generate_samples`（L864）只做按 `sample_size=128` 切块，末批上采样补齐；不动数值。
   `InferenceMixin.get_latent_representation`（inference.py:412）全程不出现
   `adata.X.copy()`、`normalize_total` 等调用。
3. **Adapter 的 expm1 拷贝满足要求**：
   - dtype：`np.asarray(..., dtype=np.float32)` + `np.expm1` 输出 float32。✓
   - 稀疏：只改 `.data`；`expm1(0)=0`，稀疏 pattern 保持不变。✓
   - `obs` / `var` / `obsm`（含 `pert_var_idx`） / `varm` / `uns` 全部随 `ad.AnnData(...)`
     重绑到 transient 上，**不**修改原对象的任何字段。✓
   - `var_names` 顺序即 `adata.var.index`，不做 reorder；Stack 的
     `gene_reduction: nn.Linear(n_genes, ...)` 所需的列序由 `genelist` + `gene_mapping`
     在数据端做 align，adapter 无需参与。✓
4. **不会改动调用方 `adata`**：
   - 稀疏路径：`X2 = X.copy()` 后修改 `X2.data`，原 `X` 不动。
   - 稠密路径：`np.asarray(X, dtype=float32)` 对已是 float32 的输入可能回原 buffer，
     因此再 `np.array(..., copy=True)` 一次，保证 `np.clip(..., out=...)` 在拷贝上原地运行。
   - Transient AnnData 不携带 `adata.raw`；否则 Stack 的 `_load_adata_metadata`（datasets.py:746）
     会优先用 `adata.raw.X`，绕过我们的 expm1。
5. **HVG / 基因顺序**：Stack 通过 `genelist_path` 加载 target genes，并把
   `adata.var.index`（或 `gene_name_col`）uppercase 后构造 `gene_to_idx` 做交集。Adapter 把
   `gene_name_col` 原样透传。所以调用方保持 `var_names` = gene symbol（或指定 `gene_name_col`）
   即可，adapter 无需做 HVG。
6. **边界防御**：log1p 数据理论上非负，但我们在 expm1 前加了
   `np.clip(X, 0.0, None)`（稀疏对 `.data`、稠密对拷贝数组均 in-place 执行），避免
   数值抖动导致的 `<0` → `expm1<0` → Stack 二次 `log1p` 得到 NaN。

结论：**FIXED**（在此次加 clip 之前其实亦可视为 CORRECT；加 clip 后更稳）。

## 已知风险 / 后续

- **输入分布**：Stack 假设 `X` 为 raw counts（内部 `log1p`）。若用户传入已 log-normalized 的数据，
  embedding 仍是 finite 的，但**跨 adapter 可比性**要求与其他模型一致。本 smoke 沿用现有子集
  （已 log-normalized），定性结论可信，定量指标需后续在 raw-counts 流上复测。
- **`filter_organism`**：默认 `False` 以兼容无 `organism` 列的数据；如果你的数据集包含人 / 其他
  物种混合细胞，请显式传 `filter_organism=True`。
- **batch / workers**：`num_workers=0` 是为了稳妥传递 in-memory adata（DataLoader fork 时对大
  对象有 pickle 开销）；大规模数据建议改用文件路径 + `num_workers>0`。
- **Checkpoint map_location**：`load_model_from_checkpoint` 默认把 checkpoint 直接加载到 GPU，
  大 checkpoint（217M 参数）暂占 ~1 GB 显存；共享 GPU 时注意。


---

## 审计摘要（历史 encoder_audit）


## 入口

- `stack/models/core/base.py`：`_reduce_and_tokenize` 对全基因维度线性投影，无 per-gene token 采样。
- `InferenceMixin.get_latent_representation`：`TestSamplerDataset` + `DataLoader`。

## 策略

- 扰动已包含在 `X` 中即参与编码；无需 reorder token（无 token 序列）。

## 状态

- Adapter 已实现：[`adapters/stack/encoder.py`](../../adapters/stack/encoder.py)；实现与审计见本文档全文。
