# Encoder 实现反馈（总表）

审计与 **无扰动 / 角色分类** 验收结论见 **[encoder_readiness.md](../encoder_readiness.md)**。

| 模型 | 状态 | smoke | 文档 |
|------|------|--------|------|
| UCE | 已实现 | 仓库内暂无 `test_uce.py`；历史结果见 [uce.md](uce.md) | [uce.md](uce.md) |
| State (SE-600M) | 已实现 | 仓库内暂无 `test_state.py`；历史结果见 [state.md](state.md) | [state.md](state.md) |
| scGPT | 已实现 | 仓库内暂无 `test_scgpt.py`；历史结果见 [scgpt.md](scgpt.md) | [scgpt.md](scgpt.md) |
| Geneformer V2-316M | 已实现 | 仓库内暂无 `test_geneformer.py`；历史结果见 [geneformer.md](geneformer.md) | [geneformer.md](geneformer.md) |
| stack (Stack-Large) | 已实现 | 仓库内暂无 `test_stack.py`；历史结果见 [stack.md](stack.md) | [stack.md](stack.md) |
| scldm (70M) | 已实现 | 仓库内暂无 `test_scldm.py`；历史结果见 [scldm.md](scldm.md) | [scldm.md](scldm.md) |
| xVERSE-384 | 已实现 | 仓库内暂无 `test_xverse.py`；历史结果见 [xverse.md](xverse.md) | [xverse.md](xverse.md) |
| CellNavi | 已实现 | [`smoke/test_cellnavi.py`](../../smoke/test_cellnavi.py)（缺资产时 SKIP） | [cellnavi.md](cellnavi.md) |
| scFoundation | 已实现 | [`smoke/test_scfoundation.py`](../../smoke/test_scfoundation.py)（缺 ckpt 时 SKIP） | [scfoundation.md](scfoundation.md) |
| NicheFormer | adapter 已实现，缺 ckpt | 待官方 Mendeley ckpt 落盘后 smoke | [nicheformer.md](nicheformer.md) |
| TranscriptFormer | 已实现，chempert full embedding ready | `sciplex3_A549` 128-cell smoke 已通过 | [transcriptformer.md](transcriptformer.md) |
| PCA（dataset-fitted） | 已实现 | [`smoke/test_pca_baseline.py`](../../smoke/test_pca_baseline.py) | [pca_baseline.md](pca_baseline.md) |
| scVI（dataset-fitted） | 已实现 | [`smoke/test_scvi_baseline.py`](../../smoke/test_scvi_baseline.py) | [scvi_baseline.md](scvi_baseline.md) |

**Smoke 数据**: `fm/smoke/build_subset.py`（若存在）或各 smoke 脚本内说明；历史示例取 `data/raw/DE5000_bench/Adamson.h5ad`
（指标元数据）+ `data/raw/DE5000/Adamson.h5ad`（X counts），合成 8 control / 8 ASCC3 / 8 SCYL1 /
8 ASCC3+SCYL1 = 32 cells。`obsm['pert_var_idx']` 形状 `(32, 2)`，各 adapter 的
`pert_kept_histogram` 均为 `{0:8, 1:16, 2:8, 3+:0}`，与构造的 per-cell 扰动占位一致
（用于 **protected-gene coverage** 元数据对齐，而非单独条件输入）。**当前协议说明**见
[`per_cell_pert_design.md`](../per_cell_pert_design.md) 与 [`encoder_readiness.md`](../encoder_readiness.md)。

**当前仓库内可执行的 smoke**（其余模型的 `test_*.py` 可按各 `encoder_impl/*.md` 历史记录自行补回）：

```bash
cd <delivery_root>
# 若存在：
# $SCFM_ENVS_ROOT/scdfm/bin/python data/scFM/fm/smoke/build_subset.py
$SCFM_ENVS_ROOT/cellnavi/bin/python data/scFM/fm/smoke/test_cellnavi.py
CUDA_VISIBLE_DEVICES=0 $SCFM_ENVS_ROOT/scfoundation/bin/python data/scFM/fm/smoke/test_scfoundation.py
$SCFM_ENVS_ROOT/scdfm/bin/python data/scFM/fm/smoke/test_pca_baseline.py
$SCFM_ENVS_ROOT/scldm/bin/python data/scFM/fm/smoke/test_scvi_baseline.py
```

**环境变量（常用）**

| 变量 | 用途 |
|------|------|
| `COUPLEDFM_ROOT` | CoupledFM 根（UCE 默认权重路径） |
| `LATENT_BENCH_STATE_CKPT` | State `.ckpt`：adapter **无代码内默认路径**，必须 `checkpoint=...` 或设置本变量；**推荐** `CoupledFM/pretrained/state/SE-600M/*.ckpt`（例如 `se600m_epoch16.ckpt`） |
| `LATENT_BENCH_STATE_PE` | State protein embeddings（默认: 同 ckpt 目录下 `protein_embeddings.pt`） |
| `LATENT_BENCH_SCGPT_MODEL_DIR` | scGPT 目录（含 `vocab.json` / `args.json` / `best_model.pt`） |
| `LATENT_BENCH_STACK_CKPT` | Stack `.ckpt`（默认 `pretrained/stack/bc_large.ckpt`） |
| `LATENT_BENCH_STACK_GENELIST` | Stack 基因列表 pickle（默认 `pretrained/stack/basecount_1000per_15000max.pkl`） |
| `LATENT_BENCH_SCLDM_CKPT` | scldm Lightning ckpt（默认 `pretrained/scdlm/vae_census/70M.ckpt`） |
| `LATENT_BENCH_SCLDM_CONFIG` | scldm Hydra yaml（默认 `pretrained/scdlm/vae_census/70M.yaml`） |
| `LATENT_BENCH_XVERSE_CKPT` | xVERSE `.pth`（默认 `pretrained/xVerse/xVERSE_384.pth`） |
| `LATENT_BENCH_GENEFORMER_DIR` | Geneformer V2 HF 目录（默认 `pretrained/geneformer/Geneformer-V2-316M`） |
| `LATENT_BENCH_CELLNAVI_CKPT` | CellNavi `pretrain_weights.pth`（默认：若存在则 `pretrained/cellnavi/data/pretrain/`，否则 `third_party/CellNavi/data/pretrain/`） |
| `LATENT_BENCH_CELLNAVI_GRAPH_PKL` | NicheNet `graph.pkl`（默认：若存在则 `pretrained/cellnavi/data/Nichenet/`，否则 `third_party/CellNavi/Nichenet/`；须与同目录 `node2idx.json` 一致） |
| `LATENT_BENCH_CELLNAVI_GENE_NAME` / `LATENT_BENCH_CELLNAVI_NODE2IDX` | 可选覆盖；默认同样优先 `pretrained/cellnavi/data/` 再回退第三方。**import 仍只来自 `third_party/CellNavi`。** |
| `LATENT_BENCH_SCFOUNDATION_CKPT` | scFoundation `models.ckpt`（默认 `pretrained/scFoundation/models.ckpt`） |
| `LATENT_BENCH_SCFOUNDATION_GENE_TSV` | `OS_scRNA_gene_index.19264.tsv`（默认 `third_party/scFoundation/model/...`） |

**已知风险 / 适配补丁**

- scGPT: 第三方 `FlashTransformerEncoderLayer` 仍按旧 flash-attn API 调用 `FlashMHA(batch_first=…, attention_dropout=…)`；adapter 在 `_patch_flash_mha()` 里对 `flash_attn.flash_attention.FlashMHA` 做了无侵入的兼容子类，删除 `batch_first`、把 `attention_dropout` 重命名成 `dropout`，并补回 `batch_first=True` 属性。
- State: `Inference.load_model` 默认从配置中的绝对路径读 `Homo_sapiens.GRCh38...ESM2.pt`（指向 ARC 内部 `/large_storage/...` 不存在）。adapter 在调用前显式 `torch.load(protein_embeddings.pt)` 并通过 `Inference(protein_embeds=...)` 注入，无需修改第三方源。
- UCE: 仍走 `third_party/uce/exp_emb/uce_inference.py`，但 adapter 内 `_CellDataset` 在 `getitem` 中改用 **per-cell** 的 `pert_gene_set`，未触发原 `encode_adata` 的 batch-union 路径。
- Geneformer: 手动复刻 `tokenize_cell`（median-scaling + rank argsort），避免调用 `TranscriptomeTokenizer` 的 loom / h5ad 磁盘管线；每个细胞在表达排序得到的 `rank_tokens` 上与 `obsm['pert_var_idx']` 映射的 token 做 **protected merge**（`_merge_rank_tokens_with_protected`：保证 protected 进入序列、仍仅一份表达驱动），再按 batch-max padding；pool 遵循 `emb_extractor.get_embs` 的 `cls_present/eos_absent`（`hidden[:, 1:, :]` mean over 非 pad 位）。
- stack: 全基因路径，无 token 采样；`force_pert_effective` 恒为 false；`pert_var_idx` 仅用于 `pert_kept_histogram` 等元数据。
- xVERSE: 全基因 attention，无 token 采样；`force_pert_effective` 恒为 false（与 [`per_cell_pert_design.md`](../per_cell_pert_design.md) 一致）。`force_pert` + `obsm['pert_var_idx']` 时可能对对齐矩阵做 **槽位修补**（无效/未测位改为 0），`meta['pert_var_idx_slot_repair']` 记录是否启用；仍非单独条件流。
- scldm: adapter 构造全基因张量与 **expressed subset**；上游 `TransformerVAE.encode` 在传入 subset 时 **仅对 subset 走 encoder**（全基因张量当前分支不参与编码）。`force_pert` 仅影响 subset 打包时的 protected coverage，不是第二条扰动条件流。加载路径：`OmegaConf.load` → `scldm._utils.remap_config` → `OmegaConf.resolve` → 强制 `compile=False` → `hydra.utils.instantiate`；`torch.load` 时传 `pickle_module=remap_pickle`，并过滤 compile 遗留重复 key。
- xVERSE: checkpoint 是 DDP `module.` 前缀 state_dict；`_load_checkpoint` 统一剥 `module.` 并 `strict=False` 加载，显式断言 4 个核心 key（`gene_embedding` / `tissue_gene_bias` / `bio_encoder.gene_emb` / `bio_encoder.tissue_emb`）都存在。
- CellNavi: 薄封装 `SparseCellNaviEncoder`，主输出为 forward 返回的 **CLS** 向量（`cls_out`）。`input_is_log1p=True` 时对 `X` 做 `expm1` 再四舍五入为整数计数，再走与上游一致的 `log1p(count/sum*10000)` 归一化分支，避免把 log 空间误当 raw。零表达基因默认不进子图；`force_pert` + `obsm['pert_var_idx']` 时用伪计数强制纳入 **protected** 基因（仍要求基因符号在 CellNavi∩NicheNet 词表内）。**权重与 NicheNet 图文件**请放在 `CoupledFM/pretrained/cellnavi/data/`（见 [cellnavi.md](cellnavi.md)）；精简 `third_party/CellNavi` 镜像可能仍不全。
- scFoundation: 复刻 `third_party/scFoundation/model/get_embedding.py` 的 **cell** 分支（`gatherData` → `token_emb`/`pos_emb`/`encoder` → 官方 `pool_type` 池化）。默认 `input_is_log1p=True` → `pre_normalized='T'`（与官方「已 normalize+log1p」一致），**不**在 adapter 内再 `log1p`。上游 `gatherData` 用 `x>0` 选基因位；`force_pert` 时对 `value_labels` 与 `obsm['pert_var_idx']` 映射的列做 OR，零表达 protected 仍进入序列（值为 0）。**不**走 GEARS perturbation 分支。因上游 `load.main_gene_selection` 缺少 `import pandas`，adapter 内实现了等价的 `_main_gene_selection`。

## log1p 数据流审计（2026-04-20，code-only）

Benchmark 约定：`adata.X` 上游已是 `log1p(normalize_total(...))`。所有 adapter 的
`encode(...)` 都接受 `input_is_log1p: bool = True`（默认 True），并按各模型实际需求选择
**no-op** 或 **显式 `expm1`**，避免双重 log。

| 模型 | Verdict | 处理策略 | 关键证据 |
|------|---------|----------|----------|
| scGPT | CORRECT (invariant) | no-op；`do_binning` 51-bin rank 对单调变换不变，`<cls>` 与 protected（pert）前缀位用 `kfn` 保护不参与分箱（coverage，非条件流） | `third_party/scGPT-main/scgpt/preprocess.py:274-303` |
| UCE | CORRECT (invariant) | no-op；`build_cell_sentence` 走 `argsort(-counts)` 纯 rank，token embedding L2-normalize | `third_party/uce/exp_emb/uce_inference.py:186,329` |
| State | FIXED (deterministic) | `install_input_mode(is_log1p=True)` 类级 monkey-patch `VCIDatasetSentenceCollator.is_raw_integer_counts`，**锁死**走 log1p 分支，不再依赖启发式 | `third_party/state/src/state/emb/data/loader.py:579-624` + `adapters/state/per_cell_collator_patch.py:150-188` |
| Geneformer | FIXED | 在 per-cell ranking 前对 dense `X` 做 `expm1(clip(X, 0, None))`，恢复 `normalize_total` 尺度，让 median 除法得到 raw-counts 一致的 rank | `adapters/geneformer/encoder.py:151-155` |
| stack | FIXED | shallow-copy AnnData，对 `X.data`（稀疏）/ 拷贝数组（稠密）做 `expm1(clip(X, 0, None))`，不 mutate 原 adata；Stack `forward` 内部再 `torch.log1p` 一次，两者对齐 | `third_party/stack/.../base.py:153` + `adapters/stack/encoder.py:109-124` |
| scldm | FIXED | 在 `_align_expression_to_vocab` 之后对 `X_full` 做 `expm1(clip(X, 0, None))`；`counts_subset` 由已 expm1 的 `X_full` 派生（两路同源）；VAE `ProjectionConcat` 再 `log1p` | `third_party/scldm/.../layers.py:62` + `adapters/scldm/encoder.py:212-216` |
| xVERSE | CORRECT (FIXED in prev round) | 对已对齐到 vocab 的 `vals` 做 `expm1(clip(vals, 0, None))`；`bio_encoder` 内部 `torch.log1p` 完成往返 | `third_party/xVERSE_code/main/utils_model.py:209-213` + `adapters/xverse/encoder.py:298-305` |
| CellNavi | CORRECT (adapter) | `expm1(clip)` → 整数化 rawcount；`normalize=True` 时 `log1p(raw/sum*1e4)` 与上游 `prepare_cell_input` 一致 | `adapters/cellnavi/encoder.py` + `third_party/CellNavi/cellnavi/data_provider/data_utils.py` |
| scFoundation | CORRECT (adapter) | 默认 `pre_normalized='T'`：直接把 `adata.X` 当作官方 normalize+log1p 输入；`input_is_log1p=False` 时走官方 `F` 分支（`log1p(count/sum*1e4)`） | `adapters/scfoundation/encoder.py` + `third_party/scFoundation/model/get_embedding.py` |
| scVI (dataset-fitted) | FIXED | `input_is_log1p=True` 时走 `gene_likelihood='normal'`（Gaussian 解码头），直接使用 log1p `X`/layer，不造伪计数 | `adapters/scvi_baseline/encoder.py:164-215` |
| TranscriptFormer | CORRECT (explicit counts only) | 官方路径需要 raw counts；benchmark `X` 已是 log1p 时只使用 `adata.raw.X` 或显式 count layer（默认 `layers['counts']`/`raw_counts`/`count`），不对 `X` 做第二次 `log1p`，也不 `expm1` 造伪 counts | `adapters/transcriptformer/encoder.py` |
| NicheFormer | CORRECT (explicit counts only) | direct tokenization 需要 count-like 输入；benchmark `X` 已是 log1p 时只使用显式 count layer（默认 `layers['counts']`/`raw_counts`/`count`），缺失即报错，不 `expm1` 造伪 counts | `adapters/nicheformer/encoder.py` |

- **Zeroshot foundation 表内**：`input_is_log1p=True` 是**默认且唯一经过审计的**口径。若调用方有真 raw counts，必须显式
  传 `input_is_log1p=False`（State 同时对应把 monkey-patch 切到 raw 分支）。TranscriptFormer/NicheFormer
  是例外中的严格路径：默认 log1p 口径下会寻找 `raw.X`/`layers['counts']` 这类显式 count source，
  找不到就失败，避免把 benchmark log1p `X` 当 counts。
- 所有 adapter 都**不 mutate** 原 `adata`（稀疏用 `.copy()` + `.data`，稠密用
  `np.array(..., copy=True)`/新分配 ndarray，State 仅 `write_h5ad` 到临时路径）。
- 每个模型的详细推导、行号与残余风险见各自 `encoder_impl/<model>.md` 末尾的
  `## log1p audit` 章节。
- `third_party/` 未修改。

**Adamson 现象学说明**

- `data/raw/DE5000_bench/Adamson.h5ad` 仅保留指标 / obs / var；`X` 字段以 `null` 占位。smoke 脚本会自动从 `data/raw/DE5000/Adamson.h5ad` 抽出对应行的 X 重建（cell barcode 100% overlap）。
- 数据集只有单扰动条件；多扰动通过取一对单扰动细胞的均值合成 `ASCC3+SCYL1`，让 collator 真实经历 `nperts=2` 路径。
