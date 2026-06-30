# Encoder audit & readiness（expression-only 协议）

本页汇总 expression-only 协议下的验收结论：编码器输入只有单细胞表达谱；`pert_var_idx`
仅在部分模型中作为 sampling / truncation 的 protected-gene coverage 约束。

## 角色分类（`encoder_role`）

| 取值 | 含义 |
|------|------|
| `ExpressionOnlyEncoder` | 编码只依赖表达谱；可选地使用 `pert_var_idx` 做 protected-gene coverage，但不作为单独条件输入。 |
| `OutOfScope` | 当前 latent bench（逐细胞 scRNA）不纳入推理。 |

## 逐模型结论

| 模型 | encoder_role | force_pert_effective | 无扰动路径 | Readiness | 说明 |
|------|----------------|----------------------|------------|-----------|------|
| **scGPT** | ExpressionOnlyEncoder | `force_pert and pert_var_idx_present` | 始终可走纯表达编码 | **Ready** | 不再从 `obs['perturbation']` 读条件；`obsm['pert_var_idx']` 仅作为 truncation/sampling 的 protected set。 |
| **UCE** | ExpressionOnlyEncoder | `force_pert and pert_var_idx_present` | 始终可走纯表达编码 | **ReadyWithKnownLimits** | 仍通过 tokenizer 的 protected-gene 覆盖保证 top-K 保留；`build_cell_sentence` 语义应理解为 coverage 而非条件输入。 |
| **State** | ExpressionOnlyEncoder | `force_pert and pert_var_idx_present` | `force_pert=False` 时会写出去除 pert metadata 的临时 h5ad | **Ready** | adapter 已屏蔽 `obs` 条件列；coverage 仅来自 `obsm['pert_var_idx']` + runtime patch。 |
| **Geneformer** | ExpressionOnlyEncoder | `force_pert and pert_var_idx_present` | 始终可走纯表达 rank token 编码 | **Ready** | protected genes 只用于保证被编码 token 集覆盖，不再插入 `<cls>` 后前缀。 |
| **stack** | ExpressionOnlyEncoder | 恒 `False` | 不依赖 `pert_var_idx` 即可编码 | **Ready** | 浅拷贝 `expm1(X)`；勿传 `adata.raw` 以免绕过 adapter。 |
| **scldm** | ExpressionOnlyEncoder | `force_pert and pert_var_idx_present`（subset coverage） | 始终可走纯表达编码 | **ReadyWithKnownLimits** | 实际 forward 以 **expressed subset** 为主；现已补 protected coverage，但仍依赖 subset 打包路径。 |
| **xVERSE** | ExpressionOnlyEncoder | 恒 `False` | 无 `pert_var_idx` 亦可编码 | **Ready** | `pert_var_idx` 仅作可观测性/覆盖校验；不构成单独条件输入。 |
| **CellNavi** | ExpressionOnlyEncoder | `force_pert and pert_var_idx_present` | 无矩阵亦可编码（仅表达 >0 且在 vocab 的基因进图） | **ReadyWithKnownLimits** | 稀疏子图编码；上游默认丢弃表达为 0 的基因 → `force_pert` 时用伪计数强制纳入 protected set。需要 `Nichenet/graph.pkl` + `pretrain_weights.pth` 等（默认优先 `pretrained/cellnavi/data/`，见 `encoder_impl/cellnavi.md`）。 |
| **scFoundation** | ExpressionOnlyEncoder | `force_pert and pert_var_idx_present` | 无矩阵可走官方 cell 路径（仅 >0 基因进 ``gatherData``） | **ReadyWithKnownLimits** | 复刻 ``get_embedding.py`` cell 分支；protected 仅 OR 进 ``value_labels``，零表达位仍编码为 0。权重默认 ``pretrained/scFoundation/models.ckpt``；基因表在 ``third_party/scFoundation/model/OS_scRNA_gene_index.19264.tsv``。 |
| **NicheFormer** | ExpressionOnlyEncoder | 恒 `False` | 需要 explicit raw counts + Ensembl gene IDs 与官方 model mean 对齐 | **ReadyWithRawCountsRequired** | 支持 legacy Lightning `.ckpt` 与 HuggingFace `model.safetensors`；当前本地 HF 权重可用。benchmark `X` 已是 log1p 时只读取 `layers['counts']`/`raw_counts` 等显式 count layer；找不到就 fail-fast，不做二次 log1p，也不隐式 `expm1`。 |
| **TranscriptFormer** | ExpressionOnlyEncoder | 恒 `False` | 官方 CLI 可直接从 h5ad 输出 `obsm['embeddings']` | **ReadyWithRawCountsRequired** | 默认 `tf_sapiens`，要求 `var['ensembl_id']`/`var['ensemblid']`/`var['Ensembl_ID']`/`var['ENSEMBL']` 或 Ensembl `var_names`；官方要求 raw counts。benchmark `X` 已是 log1p 时优先 `adata.raw.X`，其次 `layers['counts']`/`raw_counts`；找不到就 fail-fast，不做二次 log1p，也不隐式 `expm1`。 |

### Dataset-fitted baselines（PCA / scVI）

与上表 **预训练 zeroshot encoder** 并列但语义不同：每个 **dataset** 单独 **fit** 再 **transform**，`meta` 含 `fit_scope="dataset"`、`fit_method="pca"|"scvi"`；**不参与**与 foundation 模型的“同一起点”公平对比。

| Baseline | encoder_role | force_pert_effective | 说明 |
|----------|----------------|----------------------|------|
| **PCA（dataset-fitted）** | ExpressionOnlyEncoder | 恒 `False` | 在单 dataset 合并 `AnnData` 的 **`adata.X`（log1p 口径）** 上联合拟合 PCA；不使用扰动元数据作为条件。 |
| **scVI（dataset-fitted）** | ExpressionOnlyEncoder | 恒 `False` | 在单 dataset **control+gt** 上训练 SCVI，只导出 **`get_latent_representation()`**。**默认**用真 **counts**（默认 `layers['counts']`）。若仅有 log1p 归一化表达，可 **`encode(..., input_is_log1p=True)`**（CLI：`--scvi-input-is-log1p`），以 **`gene_likelihood='normal'`** 接连续表达；不得在无显式 opt-in 时对 log1p `X` 静默走 NB/ZINB。 |

详见 [`encoder_impl/pca_baseline.md`](encoder_impl/pca_baseline.md)、[`encoder_impl/scvi_baseline.md`](encoder_impl/scvi_baseline.md)。

## Manifest 建议字段

导出时在 manifest 中写入（若适用）：

- `force_pert`、`pert_kept_per_cell`（已有 schema）
- `input_is_log1p`、`force_pert_effective`（schema 已扩展；此处 `force_pert_effective` 表示 protected-gene coverage 是否启用）
- `encoder_role`（建议由 runner 根据上表写入）

详见 [`benchmark/manifest.schema.json`](../benchmark/manifest.schema.json)。

## 当前协议说明

当前 benchmark 协议已经切换为：

- **所有模型都只编码 sc 表达谱**；
- `obsm['pert_var_idx']` 若存在，只能用于 **protected-gene coverage**
  （sampling / truncation / subset packing 时保证这些基因不会被丢掉）；
- 不再允许把 perturbation metadata 作为单独 token 前缀、条件集合、或并行条件输入传给模型。

因此这里的 `force_pert_effective` 应解释为：
**是否启用了 protected-gene coverage 策略**，而不是“是否输入了 perturbation 条件”。

### 谁需要 `obsm['pert_var_idx']`（protected coverage）

- **需要**（存在采样 / 截断 / expressed-subset 且可能丢基因时）：**scGPT、UCE、State、Geneformer、scldm** — 无矩阵时仍可编码，但无法保证扰动相关基因在截断后仍出现在 token/subset 中。
- **不依赖 coverage 也能完整吃到全基因表达**：**stack、xVERSE** — 全基因（或等价全列）路径下扰动已含在 `X`；`pert_var_idx` 若存在仅用于 manifest 直方图等元数据对齐，`force_pert_effective` 恒为 false。
- **稀疏表达子图（非 top-k 截断，但会丢零表达基因）**：**CellNavi** — 若需保证扰动基因出现在子图节点集合中，即使该基因在本细胞计数为 0，也应提供 `obsm['pert_var_idx']` 且 `force_pert=True`（adapter 注入最小伪计数）；否则仅非零表达基因进图。
- **按非零位打包的序列 encoder（scFoundation）**：官方 ``gatherData`` 用 ``x > 0`` 选位；零表达 protected 基因需 `obsm['pert_var_idx']` + `force_pert=True`，adapter 对 mask 做 OR（表达值仍为 0，非第二条条件流）。

## 子代理审计来源

本结论基于对 [`adapters/`](../adapters/) 与 `third_party/` 入口的只读子代理报告 + 主代理代码复核；未在本轮重新跑全量 smoke。

## 包导入

[`adapters/__init__.py`](../adapters/__init__.py) 对子模块使用 **惰性导入**：`import ...adapters` 不会立刻加载 Geneformer（需 `transformers`）等重依赖；仅在访问 `adapters.geneformer` 时加载。
