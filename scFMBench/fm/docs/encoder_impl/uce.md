# UCE — 实现反馈

## 修改文件清单

- [`adapters/uce/encoder.py`](../../adapters/uce/encoder.py)（新建）
- 依赖：`third_party/uce/exp_emb/uce_inference.py`（`COUPLEDFM_UCE_SRC` → `third_party/uce`）

## 关键实现

- 直接复用 `UCEInference`（包含 `UCETokenizer.build_cell_sentence`）但**绕过** `encode_adata`：
  adapter 自己构造 `_CellDataset`，每个 `__getitem__` 把 **per-cell** 的 protected 列索引集合作为
  `pert_gene_set` 传入 `build_cell_sentence`，从而在本细胞的 top-K（SAMPLE_SIZE=1024）内优先保留这些基因。
- **仅**在 `force_pert=True` 且存在 `obsm['pert_var_idx']` 时使用 protected 集合；**不**读取 `obs['perturbation']`
  字符串（与第三方 `encode_adata` 在缺 `pert_var_idx` 时 fallback `obs['perturbation']` 的行为不同）。
- `vocab_set` 使用大写匹配；不在 UCE PE 词表里的扰动基因会自动跳过（meta 里通过 histogram 反馈）。

## Smoke 验证（历史记录；仓库当前无 `smoke/test_uce.py`）

此前可用命令形如：

```bash
$SCFM_ENVS_ROOT/uce/bin/python data/scFM/fm/smoke/test_uce.py
```

- 输出：`emb shape (32, 1280)`，`mean L2 diff (force vs off) = 0.0058`
  （差异较小是因为 SAMPLE_SIZE=1024 而数据集仅 3086 valid genes，多数基因本身就被纳入；
  差异主要发生在“低表达 perturbation 基因被强制放进序列”的情形）
- `pert_kept_histogram = {0:8, 1:16, 2:8, 3+:0}` ✅
- 默认权重：`pretrained/uce/model_files/{33layer_model.torch, all_tokens.torch, ...}`

## 已知风险

- `inf.encode_adata` 自带 batch-level union 行为；adapter 没有调用它，避免误用。
- 多 worker DataLoader 路径未在 smoke 中验证（`n_collate_workers=0`）；上游已知支持 4~8 workers。

## log1p audit（2026-04-20，code-only，无 smoke）

**Benchmark 约定**：`adata.X` 已是 `log1p(normalize_total)`。UCE adapter 对此**严格不变（invariant）**——
`input_is_log1p=True` 是带文档的 no-op，无需 `expm1`。

**证据链**（`adata.X` → tokens → embedding，全部只用 rank/不动原数据）：

1. Adapter 不 mutate `adata`：[`adapters/uce/encoder.py`](../../adapters/uce/encoder.py) 全文无 `adata.X = ...`/
   `adata.obsm[...] = ...` 等写入；第 128–131 行在非稀疏分支用 `csr_matrix(adata.X)` 新建矩阵，
   稀疏分支 `adata.X.tocsr()` 仅做视图转换且之后只读取。
2. 每个 cell 的 counts 只经过 `np.asarray(row.todense()).ravel()` 传入 tokenizer
   （`adapters/uce/encoder.py` 第 148 行）。
3. **关键不变性点**：`third_party/uce/exp_emb/uce_inference.py` `build_cell_sentence` 中
   `counts` 的**全部**用法是：
   - 第 186 行：`sort_order = np.argsort(-counts[valid_idx])` —— 纯 rank，任何
     正的单调变换（`log1p`、`expm1`、`normalize_total`）都保持该排序。
   - 第 192 行：`pert_valid = [g for g in pert_gene_set if g < len(counts) and valid[g]]`
     —— 只用 `len(counts)`（形状），不碰 magnitude。
   - 无 `sum` / `mean` / `log` / `norm` 等依赖绝对表达量的操作（已用 ripgrep
     在 `third_party/uce/exp_emb/` 全局搜索确认；`step1_ctrl_embedding.py`、
     `step5_gt_embedding.py` 中的 `.sum()` 都作用在 `obs['perturbation']` 布尔掩码上，
     与 `counts` 无关）。
4. **Token 侧 L2 normalize**：`third_party/uce/model.py` 与 `exp_emb/uce_inference.py`
   在进入 attention 之前对 token embedding 做 `F.normalize`：
   - `exp_emb/uce_inference.py` 第 329 行：`token_embs = F.normalize(token_embs, dim=2)`
     （embedding lookup 之后、调 `model.forward` 之前）。
   - `model.py` 第 106 行：`embedding = nn.functional.normalize(embedding, dim=1)`
     （CLS token 输出再做一次 L2 normalize）。
   - 这使 cell embedding 的大小完全由 token **身份 + 顺序**决定，不受 counts 幅值影响。

**结论：CORRECT (invariant)。** 不需要在 adapter 里 `expm1`；`input_is_log1p` 仅作为
meta 透传记录。代码路径下 `log1p(normalize_total)` 与 raw counts 产出**完全相同**的
token 序列和 embedding（假设没有 NaN/负值干扰 argsort，此处 log1p 保号所以安全）。


---

## 审计摘要（历史 encoder_audit）


## 入口

- `third_party/uce/exp_emb/uce_inference.py`：`UCETokenizer.build_cell_sentence` 已支持 per-cell `pert_gene_set`；原 `encode_adata` 在 batch 内对 `pert_var_idx` 做 union。

## 长度策略

- `PAD_LENGTH=1536`，`SAMPLE_SIZE=1024` 基因进入染色体块。

## 改造（已实现）

- **Adapter**：[`adapters/uce/encoder.py`](../../adapters/uce/encoder.py) 中 `batch_pert_sets[k] = set(row)`，**不**做 batch union。

## 验收 checklist

- [ ] `pretrained/uce/model_files/*.torch` 存在；`COUPLEDFM_ROOT` 正确。
- [ ] 与 `data/scFM/uce/exp_emb/uce_inference.py` **不** import（仅用 `third_party/uce/exp_emb`）。
