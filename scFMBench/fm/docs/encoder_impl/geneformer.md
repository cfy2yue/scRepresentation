# Geneformer V2-316M — 实现反馈

## 修改文件清单

- [`adapters/geneformer/encoder.py`](../../adapters/geneformer/encoder.py)（替换 stub）
- [`docs/encoder_impl/geneformer.md`](./geneformer.md)（本文件）

## Tokenizer 资源

均位于 `data/scFM/fm/third_party/Geneformer/geneformer/`（gc104M = V2）：

- `token_dictionary_gc104M.pkl`：`{Ensembl_ID or <special>: token_id}`，含 `<pad>=0, <mask>=1, <cls>=2, <eos>=3`
- `gene_median_dictionary_gc104M.pkl`：gene-wise non-zero median（归一化因子）
- `ensembl_mapping_dict_gc104M.pkl`：Ensembl ID 折叠字典（deprecated → canonical），与 `TranscriptomeTokenizer.gene_mapping_dict` 一致

## 关键实现

- 权重：`pretrained/geneformer/Geneformer-V2-316M/`，通过
  `BertForMaskedLM.from_pretrained(model_dir, output_hidden_states=True, local_files_only=True)`
  加载。`model_input_size` 默认从 `config.max_position_embeddings = 4096` 读取。
- Per-cell tokenization（手动复刻 `tokenize_cell`，**不**调用 `TranscriptomeTokenizer`）：
  1. 将 `adata.var['Ensembl_ID']` 通过 `ensembl_mapping_dict` 映射到规范 Ensembl，
     过滤词表外/缺失 median 的列，得到 `coding_loc / coding_tokens / norm_factor`。
  2. 对每个细胞：`norm = x[coding_loc] / sum(x[coding_loc]) * 10000 / norm_factor`；
     非零位按 `-norm` 降序得到 `rank_tokens`。
  3. 通过 `obsm['pert_var_idx']` 取出该细胞的列索引 → 映射到 Geneformer token id（词表外跳过），
     仅作为 **protected set**：与表达秩排序得到的 `rank_tokens` 在
     `_merge_rank_tokens_with_protected` 中合并，保证这些 token 出现在 `[<cls>]` 后的序列里，
     **不**再作为 `<cls>` 后的独立“条件前缀”输入流。
- 序列在 mini-batch 内按最大长度 re-padding（`<pad>`），长度由真实 token 数决定。
- Pooling 遵循 `emb_extractor.get_embs` 中 `emb_mode="cell"` 且 `<cls>` 存在时的约定：
  `hidden_states[-1][:, 1:, :]` 对非 `<cls>`、非 pad 位取均值。未附加 `<eos>`。

**当前协议说明**（与历史 smoke 数值兼容）：上述 protected merge 仍会产生可观的
`mean L2 diff(force_pert on vs off)`，因为 on 时强制保留的 token 会挤出部分纯秩尾部；
差异反映 **coverage 约束**，不是第二条扰动条件分支。
- 仅用 token id，不消费表达值；也不使用模型的 expr head（和 V2 标准推理一致）。

## Smoke 验证（历史记录；仓库当前无 `smoke/test_geneformer.py`）

此前可用命令形如：

```bash
cd <delivery_root>
CUDA_VISIBLE_DEVICES=3 $SCFM_ENVS_ROOT/geneformer/bin/python \
    data/scFM/fm/smoke/test_geneformer.py
```

输出：

```
adata=(32, 3086), perts={'ASCC3': 8, 'ASCC3+SCYL1': 8, 'SCYL1': 8, 'control': 8}
pert_var_idx shape=(32, 2)
emb_force shape=(32, 1152), dtype=float32
emb_off   shape=(32, 1152),   dtype=float32
mean L2 diff (force vs off) = 0.460487
meta_force = { pert_kept_histogram: {0:8, 1:16, 2:8, 3+:0},
               max_len: 4096, hidden_size: 1152,
               n_genes_mapped_to_vocab: 3065, n_genes_total: 3086 }
Geneformer smoke test PASSED
```

- 扰动保留直方图与构造分布一致（8 个 control / 16 个单扰 / 8 个双扰）。
- 3086 个基因中 3065 映射到 V2 词表（21 个缺失：`Ensembl_ID` 缺失或不在 Geneformer `gene_median_dictionary` 中）。

## 已知风险 / 后续

- Adamson smoke 的 `X` 已 log-normalized；Geneformer 原文期望 raw counts。Rank-based tokenization 对单调变换鲁棒，
  但 `norm_factor` 除法（median-scaling）在非 counts 数据上会发生轻微 rank 扰动。下游使用 raw counts 数据集时无影响。
- 未对空 token 序列做额外保护：coding mapping 过后全零的细胞会得到 `[<cls>]` 一个 token，
  pooled mean 可能退化为 0 向量；当前 smoke 数据无此情况。
- 对 `<eos>` 的处理：当前不附加 `<eos>`。与 `emb_extractor` 的 `cls_present` +
  `eos_not_present` 分支一致，pooling 使用 `original_len - 1`（相对 `<cls>` 后的基因 token）。
- 未启用 mixed precision；batch_size=8 × 4096 seqlen × 1152 hidden 下 316M 模型在 A100/3090 上约 2-3 GB 峰值显存。

## NeedsDecision（下一轮口径裁定）

- **重复 Ensembl 的官方对齐策略**：`ensembl_mapping_dict_gc104M.pkl` 将 deprecated → canonical 的折叠规则，是否与官方 Geneformer 管线逐位一致、benchmark 是否需显式规定「按哪个 Ensembl 版本 / 多转录本去重」——**尚未裁定**，本 adapter 仅复刻当前词表 + median 资源下的行为。落地前需单独产品/方法口径会签。

## log1p audit（2026-04-20）

**背景**：Benchmark 约定 `adata.X` 已是 `log1p(normalize_total(...))`。Geneformer 参考
`tokenizer.tokenize_anndata`（见 `third_party/.../tokenizer.py:570-584`）对每个细胞做
`X / n_counts * 10000 / norm_factor`，其中 `norm_factor` = gene-wise non-zero
median，**在 raw counts 上估计**。因此 `norm_factor` 除法在非 raw 输入上会产生秩扰动。

**适配器保护**（`adapters/geneformer/encoder.py`）：

- 增加 `input_is_log1p: bool = True` 参数（默认 True，匹配 benchmark 约定）。
- 在 per-cell ranking **之前**（`encoder.py:151-155`）：
  ```
  if input_is_log1p:
      X_dense = np.expm1(np.clip(X_dense, 0.0, None)).astype(np.float32, copy=False)
  ```
  使 `X_dense` 恢复到 `normalize_total` 尺度（相当于每个细胞 `sum ≈ 10_000`），
  这依然不是 raw counts，但 rank 只对 per-cell 标量不敏感，median 除法得到的相对
  次序与 raw counts 上一致（因为 `normalize_total` 仅乘一个 per-cell 标量）。

**核对要点**（全部通过）：

1. Rank 数学与参考一致：`sub/row_sum * 10000 / norm_factor` → nonzero → argsort 降序。
   适配器用 `sub.sum()`（仅编码基因）代替 `n_counts`，但二者仅差一个 per-cell
   标量，argsort 不变。
2. `expm1` 位置正确：紧跟 dense 化后、进入 per-cell 循环之前，落在 `_rank_tokens_for_cell`
   的 row-sum 归一化与 median 除法之前。
3. Pooling 与 `emb_extractor.get_embs(emb_mode="cell")` 的 `cls_present &
   !eos_present` 分支一致（`hidden[:, 1:, :]` + mean over `L-1` 非 pad 位置）。
4. Forward 只消费 `input_ids` 与 `attention_mask`（与 V2 标准 cell embedding 一致）。
5. `adata.X` 不会被原地修改：稀疏分支走 `toarray()`；dense 分支虽然 `astype(copy=False)`
   可能与 `adata.X` 共享内存，但随后的 `np.clip(...)` 与 `np.expm1(...)` 都返回新
   ndarray 并重新绑定 `X_dense`，循环内也只读。
6. `input_is_log1p=False` 分支跳过 `expm1`，直接按原值排序（若下游以 raw counts 喂入，
   行为与 reference 完全一致）。
7. 边界：`expm1+clip` 后全零细胞得到 `[<cls>]` 单 token，前向不会 NaN；pooling 对
   `non_cls` 的被 mask 位置求 `mean/1`，输出是退化 embedding 而非崩溃。当前 benchmark
   数据不会触发。

**Verdict**: CORRECT / FIXED — 无需追加补丁。


---

## 审计摘要（历史 encoder_audit）


## 入口

- `geneformer/tokenizer.py`：`tokenize_cell` / `rank_genes`（非零基因 + 排序）。
- `geneformer/emb_extractor.py`：`get_embs`（CLS / cell pooling）。

## 策略

- 需在 rank 前将扰动基因 token **置于 `<cls>` 之后**；若扰动基因为 0 表达，需显式纳入 token 序列。

## 状态

- Adapter 已实现：[`adapters/geneformer/encoder.py`](../../adapters/geneformer/encoder.py)；实现与审计见本文档全文。
