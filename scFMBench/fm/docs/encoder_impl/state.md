# State (SE-600M) — 实现反馈

## 修改文件清单

- [`adapters/state/encoder.py`](../../adapters/state/encoder.py)
- [`adapters/state/per_cell_collator_patch.py`](../../adapters/state/per_cell_collator_patch.py)

## 关键实现

- 加载 third_party `state.emb.inference.Inference`，但在调用 `Inference()` 之前先：
  1. `install_per_cell_collator()`：把 `VCIDatasetSentenceCollator.__call__` 替换为
     **per-cell** 版（不再 batch-union），每个细胞用自己 `pert_var_matrix[idx]` 行的非负条目
     构造 `cell_cond` 集合并传给 `_collate_one_cell`；
  2. 显式 `torch.load(protein_embeddings.pt)` 并通过 `Inference(protein_embeds=...)` 注入，
     绕过 config 中指向 ARC 内部 `/large_storage/...` 的不存在路径。
- adapter 把 in-memory adata 写到临时 h5ad 再调用 `inf.encode_adata`，确保 `obsm['pert_var_idx']`
  原样传入。

## Smoke 验证（历史记录；仓库当前无 `smoke/test_state.py`）

此前可用命令形如：

```bash
<delivery_root>/scFM/fm/third_party/state/.venv/bin/python \
  data/scFM/fm/smoke/test_state.py
```

- 输出：`emb shape (32, 2058)`（`emsize=2048` + `dataset_correction=10`），
  `mean L2 diff (force vs off) = 0.0367`
- `pert_kept_histogram = {0:8, 1:16, 2:8, 3+:0}` ✅
- 权重：adapter 内**无**硬编码默认；需 `checkpoint=` 或 `LATENT_BENCH_STATE_CKPT`。**推荐**落盘为 `CoupledFM/pretrained/state/SE-600M/se600m_epoch16.ckpt`，`protein_embeddings.pt` 同目录或 `LATENT_BENCH_STATE_PE`。
- 提示：`!!! 3078 genes mapped to embedding file (out of 3086)` 是上游正常输出，
  少量基因符号未在 ESM2 词表中。

## 已知风险

- 必须保证 `obsm['pert_var_idx']` 已构建。写入临时 h5ad **之前**，adapter 会删除 `obs` 中的 `perturbation` / `condition` / `gene` 等列，**不会**回退到解析 `obs['perturbation']`（与第三方 `encode_adata` 在缺 `pert_var_idx` 时的行为不同）。
- patch 是 monkey-patch；如果上游 `loader.py` 重构 `_collate_one_cell` 签名需要同步更新
  per_cell_collator_patch.py。
- `dataloader_num_workers` 默认 0（避免 fork 多次加载 protein embeddings）；如需提速请显式传入。

## log1p audit（输入口径与启发式短路）

**基准约定**：latent_bench 的 `adata.X` 已经过 `log1p(normalize_total)` 预处理
（Adamson smoke 上 `X.max() ≈ 6.28`，单细胞 `row_sum ≈ 1300`）。

**上游启发式**（未改动）：
`third_party/state/src/state/emb/data/loader.py`

- 常量：`RAW_COUNT_HEURISTIC_THRESHOLD = 35`（行 101），
  `EXPONENTIATED_UMIS_LIMIT = 5_000_000`（行 99）。
- `VCIDatasetSentenceCollator.is_raw_integer_counts`（行 579-601）：
  1. `max(counts) > 35` → 判为原始计数（raw）；
  2. 否则 `int(expm1(counts).sum()) > 5_000_000` → 判为 raw；
  3. 否则判为 log1p。
- `VCIDatasetSentenceCollator.sample_cell_sentences`（行 605-624）：
  - `is_raw_integer_counts=True` → 内部先做 `count_expr_dist = counts / counts.sum()`
    再 `counts_raw = torch.log1p(counts_raw)`；
  - `is_raw_integer_counts=False` → 通过 `expm1` 恢复原始分布得到 `count_expr_dist`，
    `counts_raw` 保持 log1p 形态。

**对我们的 smoke 数据的推演**：`max≈6.28 < 35`，规则 1 通过；
单细胞 `expm1(row).sum() ≈ 10_000 ≪ 5_000_000`（因 `normalize_total` 目标和为 1e4），
规则 2 通过；最终落到 log1p 分支。启发式在本场景下**会做出正确判断**。

**但我们仍做显式/确定性处理**：
- `adapters/state/encoder.py::encode` 新增 `input_is_log1p: bool = True` 参数，
  默认与基准约定一致，并写入返回的 `meta["input_is_log1p"]`。
- `adapters/state/per_cell_collator_patch.py::install_input_mode(is_log1p)`
  通过类级 monkey-patch 覆写 `VCIDatasetSentenceCollator.is_raw_integer_counts`：
  `True` 强制返回 `False`（走 log1p 分支），`False` 强制返回 `True`
  （走 raw 分支），`None` 恢复原始启发式。`encode()` 每次调用都会应用最新值。

**权衡（trade-off）**：
- 优点：消除启发式在异常输入（例如 max≤35 且行和≤5e6 的低深度 raw 样本）上误判的风险；
  审计可重复，行为由显式参数决定。
- 代价：若调用方传错 `input_is_log1p`，将静默进入错误分支；因此默认值与 latent_bench
  约定对齐（`True`），并要求传入 raw 数据的场景必须显式传 `False`。

**未变动**：
- 不修改 `third_party/state/...` 任何源文件。
- adapter 不在 `adata` 上做原地修改（仅 `adata.write_h5ad(tmp)` 与只读访问
  `obsm['pert_var_idx']`）。


---

## 审计摘要（历史 encoder_audit）


## 入口

- `state/emb/inference.py`：`Inference.encode_adata` → `create_dataloader`。
- `state/emb/data/loader.py`：`VCIDatasetSentenceCollator.__call__` 将 `pert_var_matrix` 行合并为 **batch union** 后传入 `sample_cell_sentences`。

## 长度策略

- `pad_length`（配置，约 4096）。

## 改造（已实现）

- **Runtime patch**：[`adapters/state/per_cell_collator_patch.py`](../../adapters/state/per_cell_collator_patch.py) 替换为每细胞 `cell_cond`。
- **Adapter**：[`adapters/state/encoder.py`](../../adapters/state/encoder.py)（需 `LATENT_BENCH_STATE_CKPT`）。

## 验收 checklist

- [ ] 提供 `.ckpt` 路径；`state` 包在 `third_party/state/src` 上。
- [ ] `force_pert=True` 时 patch 已安装。
