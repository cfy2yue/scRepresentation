# scFoundation adapter

## 角色

- **`encoder_role`**: `ExpressionOnlyEncoder`
- **Latent**: 官方 **`get_embedding.py`** 在 `output_type=cell` 下的向量：对 encoder 输出做 `pool_type` 池化（默认 `all` → 四路 concat，本机 **3072** 维）。
- **条件输入**: 无；不使用 GEARS / `pert_idx` / GNN 扰动分支。

## 代码入口

| 组件 | 路径 |
|------|------|
| Adapter | [`adapters/scfoundation/encoder.py`](../../adapters/scfoundation/encoder.py) |
| 上游参考 | [`third_party/scFoundation/model/get_embedding.py`](../../third_party/scFoundation/model/get_embedding.py) |
| `gatherData` | [`third_party/scFoundation/model/load.py`](../../third_party/scFoundation/model/load.py) |
| 基因表 | [`third_party/scFoundation/model/OS_scRNA_gene_index.19264.tsv`](../../third_party/scFoundation/model/OS_scRNA_gene_index.19264.tsv) |
| Smoke | [`smoke/test_scfoundation.py`](../../smoke/test_scfoundation.py) |

## 权重与路径

- 默认 checkpoint：`CoupledFM/pretrained/scFoundation/models.ckpt`（`LATENT_BENCH_SCFOUNDATION_CKPT` 可覆盖）。
- 基因 TSV 默认 `third_party/scFoundation/model/OS_scRNA_gene_index.19264.tsv`（`LATENT_BENCH_SCFOUNDATION_GENE_TSV` 可覆盖，与 `adapters/scfoundation/encoder.py` 一致）。
- `load_model_frommmf` 在官方脚本里固定用 key `cell`（`version=ce`）或 `rde`（`version=rde`）；adapter 与之对齐。
- 加载实现为 `_load_model_to_device`：支持 **CPU / CUDA**，避免上游无条件 `.cuda()` 在无 GPU 环境失败。

## 数据流与 log1p

- Benchmark 默认：`input_is_log1p=True` → adapter 使用 **`pre_normalized='T'`**，把 `adata.X` 直接当作官方文档中的 **已 normalize + log1p** 的 19264 维向量，**不再**在 adapter 内套一层 `log1p`。
- 若 `input_is_log1p=False`：使用 **`pre_normalized='F'`**，按官方对单细胞逻辑做 `log1p(count / sum * 1e4)`（需原始计数型 `X`）。

基因对齐：`var_names` 与 19264 列表取交集后列不足时，用 adapter 内 `_main_gene_selection`（等价于上游 `main_gene_selection`；因上游 `load.py` 该函数引用 `pd` 但未 import，故在 adapter 内实现）。

## Protected-gene coverage

- 上游在 `gatherData` 前使用 `value_labels = pretrain_gene_x > 0`，**零表达基因不会进入 encoder**。
- `force_pert=True` 且存在 `obsm['pert_var_idx']` 时：将每细胞 protected 的 `var` 列索引映射到 19264 基因表中的列位置，对 `value_labels` 相应位 **OR True**（表达值仍为 0，仅保证该位参与 `gatherData`）。
- **`force_pert_effective`**: `force_pert and pert_var_idx_present`。
- 不在 19264 列表中的基因：跳过（与全局约定一致）。

## Smoke

```bash
cd <delivery_root>
CUDA_VISIBLE_DEVICES=0 $SCFM_ENVS_ROOT/scfoundation/bin/python \
  data/scFM/fm/smoke/test_scfoundation.py
```

## 残余风险

- 大 checkpoint 首次加载较慢；需足够 GPU 显存（与官方推理一致）。
- `tgthighres` / `pool_type` / `version` 与官方 demo 不一致时，数值不可与论文附录直接逐位对齐；benchmark 内应固定这些超参。
