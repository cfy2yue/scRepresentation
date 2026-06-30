# xVERSE — 实现反馈

## 修改文件清单

- [`adapters/xverse/encoder.py`](../../adapters/xverse/encoder.py)（重写占位实现）
- [`adapters/xverse/__init__.py`](../../adapters/xverse/__init__.py)（新建，导出 `encode`）
- 依赖：`third_party/xVERSE_code/main/{utils_model.py,ensg_keys_high_quality.txt,tissue_name_to_id_map.csv}`（未修改）

## 关键实现

- 使用 `importlib` 以私名加载 `third_party/xVERSE_code/main/utils_model.py`，仅拿
  `XVerseModel`。不修改 third_party。
- Checkpoint `xVERSE_384.pth` 是 `{epoch, model_state_dict, …}` 字典（DDP
  `module.` 前缀）。`_load_checkpoint` 同时处理：
  - dict + `model_state_dict` → 去 `module.` 前缀后加载；
  - dict 本身就是 state_dict；
  - 直接是 `torch.nn.Module`（走 `.state_dict()` 分支）。
- 构造推理模型：`XVerseModel(num_samples=None, hidden_dim=384, total_gene=17999,
  num_tissues=64)`。因为预训练时 `num_samples=10509`，checkpoint 里包含
  `sample_emb / film_gamma / film_beta / sample_classifier_bio` 这些
  sample-head 权重；推理不需要，用 `strict=False`，再显式检查关键 key
  （`gene_embedding.weight`、`tissue_gene_bias.weight`、
  `bio_encoder.gene_emb`、`bio_encoder.tissue_emb.weight`）未缺失。
- 基因对齐：以 `ensg_keys_high_quality.txt`（17999 条 ENSG）为模型空间；
  优先读 `adata.var['Ensembl_ID' / 'ensembl_id' / 'gene_ids' / …]`，否则看
  `var_names` 是否 `ENSG…`。未对齐到的基因在输入矩阵中填 `-1`（模型把
  `-1` 视为“未测量”）。
- 输入矩阵 `values [B, 17999]`：默认 `input_is_log1p=True` 时对已对齐到词表的位点做 `expm1(clip≥0)`，
  与 `bio_encoder` 内 `log1p` 形成与 benchmark 约定一致的往返。
- Per-cell：`obsm['pert_var_idx']` 仅用于 **槽位修补**（例如把无效/未测位从 NaN 或坏值改为 `0.0`），
  不是并行条件流。`meta['force_pert_effective']` **恒为 false**（全基因无采样，与 [`per_cell_pert_design.md`](../per_cell_pert_design.md) 一致）；
  `meta['pert_var_idx_slot_repair']` 表示是否启用了 `force_pert` + `pert_var_idx` 下的修补分支；
  `meta['pert_var_idx_slot_repair_count']` 为本次 run 中实际把未测/无效槽位写成 observed `0.0` 的次数。
  `pert_kept_histogram` 仍写入以便与其它 encoder 的 manifest 字段对齐。
- Tissue：默认 `blood`（K562 所属），`_tissue_name_to_id` 读
  `tissue_name_to_id_map.csv`；允许直接传 int。

## Smoke 验证（历史记录；仓库当前无 `smoke/test_xverse.py`）

此前可用命令形如：

```bash
cd <delivery_root>
CUDA_VISIBLE_DEVICES=2 $SCFM_ENVS_ROOT/scdfm/bin/python \
  data/scFM/fm/smoke/test_xverse.py
```

输出：

```
adata=(32, 3086), perts={'ASCC3': 8, 'ASCC3+SCYL1': 8, 'SCYL1': 8, 'control': 8}
pert_var_idx shape=(32, 2)
emb_force shape=(32, 384), dtype=float32
emb_off   shape=(32, 384), dtype=float32
mean L2 diff (force vs off) = 0.000000
pert_kept_histogram = {"0": 8, "1": 16, "2": 8, "3+": 0}
xVERSE smoke test PASSED
```

- `n_aligned_genes = 3063 / 17999`（3086 基因中 23 个 ENSG 不在 xVERSE 词表）。
- `force_pert` 对 xVERSE **no-op**：标准 AnnData 中扰动列本就是非负 float，
  无需改动；所以 `mean L2 diff = 0`。这是符合模型契约的预期行为（与 scGPT
  的 token 重排不同），在 smoke test 与 `docs` 里均显式说明。

## 预处理决策 / 风险

- `adata.X` 已是 `log1p(normalize)` 之后的浮点。xVERSE 训练时期望**原始
  count**，内部再做 `log1p`。本 adapter 默认**直接透传**（相当于做了两次
  log1p），因为：
  1. 数据管线上游无法可靠还原原始 count；
  2. Adamson 测试集里值范围很小（max≈6），两次 log1p 造成的形变温和；
  3. 保持和 scGPT/UCE adapter 一致的“不修改 `adata.X`”约定。
  如需更严格复现训练条件，调用方可显式传 `expm1_input=True` 或先重新
  预处理 AnnData（不在本 adapter 内做 in-place 修改）。
- `num_tissues=64` 与 checkpoint 对齐；调用方默认 `tissue='blood'`，可通过
  参数覆盖。若设置未在 CSV 中的组织名，立即抛 `KeyError`。
- Stochastic masking：`CellEmbeddingbyGene._apply_random_mask` 仅在 `training`
  模式触发，本 adapter 全程 `model.eval()` 保证确定性。
- `bio_encoder.gene_emb` 既是 `gene_embedding.weight`（共享 Parameter），
  state_dict 中同名同值。`strict=False` 加载后两者指向同一张量。

## 已知 follow-ups

- 没有验证多 worker DataLoader / 大 batch（smoke 为 batch=8，32 cells）。
- `force_pert=True` 在更极端数据（含 NaN 或 `-1` sentinel）下的自动填 0 行为
  仅在真实异常 AnnData 上再验证一次即可。
- 如需严格训练时分布，可加一条 `raw` / `layers['counts']` 的优先读路径。

## log1p audit（2026-04 更新）

> 以下为 code-only 审计结论，用于对齐 benchmark 约定 `adata.X =
> log1p(normalize_total)`。如上文 "预处理决策" 段中 "默认透传 / 两次
> log1p" 的描述为旧实现，已被本次审计的 `input_is_log1p` 默认开启替代。

### 参数语义

- 参数名：`input_is_log1p: bool = True`（原 `expm1_input` 已弃用，含义相反）。
- 默认 `True`：`adata.X` 被视为 `log1p(normalize_total)`，适配器先做
  `np.clip(vals, 0, None)` 再 `np.expm1(...)` 还原 raw-count magnitude，交给
  xVERSE 内部 `torch.log1p` 再次取对数。
- 若调用方已握有真实 raw counts，显式传 `input_is_log1p=False`，仅做
  `clip(≥0)`，不再 `expm1`。

### 关键 invariant（由代码保证，非运行验证）

1. **Clip before expm1**：`np.expm1(np.clip(vals, 0.0, None))` 确保浮点噪声
   引起的极小负值被截为 0，`expm1(0)=0`，不会污染 raw-count 量纲。
2. **`-1` 哨兵保留**：`values [B, 17999]` 初始化为 `-1`，`expm1` 仅作用于
   `m2a>=0` 的已对齐列子集；未对齐列始终为 `-1`，模型 `_build_mask` 按
   "未测量" 处理。
3. **未修改 `adata`**：`_to_dense_row` → fancy-index `row[cols]` →
   `np.clip`/`np.expm1` 均产生新数组；写入目标是独立的 `values` numpy
   缓冲区。
4. **`force_pert` 仍读 post-expm1 值**：判定 `v<0` / non-finite 时兜底填
   `0.0`（"observed zero counts"），语义与训练期 "raw count = 0" 一致。
5. **无双重计数**：`CellEmbeddingbyGene.forward` 里 `values_cleaned` 只进
   `value_net` 算 attention，`gene_emb` / `tissue_gene_bias` 不会与观测
   计数逐位相乘（见 `utils_model.py:201-246`）。

### `third_party/xVERSE_code/main/utils_model.py` 关键行

- `utils_model.py:196-199` `_build_mask`：`(value_tensor != -1).float()`，
  `-1` 对应 mask=0。
- `utils_model.py:209-213` `torch.where(value_modified==-1, zeros_like, ...)`
  把哨兵替换为 0，随后 `torch.log1p(values_cleaned)`。`log1p` 只施加于
  非负观测值（哨兵已清零）。
- `utils_model.py:225-226` `masked_fill(mask==0, -inf)` + `softmax`
  剔除未测量位。

### 适配器关键行

- `adapters/xverse/encoder.py:306-322` 构造 `values` 与 `expm1` 只作用于
  `m2a>=0` 的对齐子集。
- `adapters/xverse/encoder.py:324-343` `force_pert` 仅在 `model_pos>=0`
  时填写；`v<0 or !isfinite` → `0.0` 防御性兜底，并累计 `n_slot_repairs`。
- `adapters/xverse/encoder.py:365-381` `meta`（含 `encoder_role`、`pert_var_idx_present`、`pert_var_idx_slot_repair_count`）。

### 验证结论

Verdict: **CORRECT**。当前 `input_is_log1p=True` 默认配合 `clip→expm1`
与模型内部 `log1p` 组合成一次正确的 `log1p(normalize_total)` 语义，
未对 `-1` 哨兵、未对非观测列、未对 `adata.X` 造成副作用。无需补丁。

### 残余风险

1. `log1p(normalize_total(X, target_sum=T))` 的 `T` 未知；`expm1` 返回的
   幅度与预训练 raw-count 分布（`FastXVerseBatchDataset.__getitem__` 原始
   计数）可能不同量级，属 benchmark 约定问题而非 log1p 正确性问题。
2. `adata.X` 含 NaN 时：`expm1(clip(nan, 0, None))=nan`；`force_pert=True`
   对扰动列由 `not np.isfinite(v)` 兜底填 0，但非扰动列的 NaN 会被喂给
   `torch.log1p` 产生 NaN 下游。调用方需保证输入无 NaN。
3. 若上游已有可靠 raw counts（`layers['counts']` / `raw.X`），可显式传
   `input_is_log1p=False` 并从该层读取，避开 log1p↔expm1 往返的浮点损失。

### 二次复核（2026-04-20，code-only）

对 8 条 checklist 再过一遍，行号与代码均未漂移：

| # | 条目 | 结果 | 关键引用 |
|---|---|---|---|
| 1 | `utils_model.py` 中 `-1` 哨兵→0→`log1p` | PASS | `utils_model.py:196-199`, `209-213` |
| 2 | `expm1` 只作用于 `m2a>=0` 已对齐列 | PASS | `encoder.py:311-321` |
| 3 | `clip(≥0)` 严格先于 `expm1` | PASS | `encoder.py:318-320` |
| 4 | `force_pert` 对 post-expm1 值的兜底语义正确 | PASS | `encoder.py:336-343` |
| 5 | `_tissue_name_to_id` 与 value 流正交 | PASS | `encoder.py:77-91` |
| 6 | 不 mutate `adata`（`_to_dense_row` / fancy-index / np 全新数组） | PASS | `encoder.py:136-139`, `306-322` |
| 7 | `gene_emb` / `tissue_gene_bias` 无被观测值放大 | PASS | `utils_model.py:216-218`, `234-237`, `319-331`（decoder 路径推理不走） |
| 8 | 对齐失败列保持 `-1`，不被 `expm1` | PASS | `encoder.py:294`, `305` |

推理路径 `model.bio_encoder(...)`（`encoder.py:349-355`）只拉
`CellEmbeddingbyGene`，**完全不经过** `compute_mu` / `gene_decoder` /
`tissue_gene_bias`（外层）；后者只在训练期重建基因 logits 时使用。因此
观测值与 `gene_embedding.weight` 的耦合仅通过 attention 权重（`utils_model.py:236`
`gene_emb_expanded * attn_weights`），不存在 value-by-embedding 的逐位放大。

Verdict 保持 **CORRECT**，未触发任何 patch。


---

## 审计摘要（历史 encoder_audit）


## 入口

- `xVERSE_code/main/utils_model.py`：`CellEmbeddingbyGene` 对 `num_genes` 维向量做 attention。

## 策略

- 扰动基因列需为「观测到」的值（非 `-1`），以便进入 softmax。

## 状态

- Adapter 已实现：[`adapters/xverse/encoder.py`](../../adapters/xverse/encoder.py)；实现与审计见本文档全文。
