# scGPT — 实现反馈

## 当前协议说明（latent_bench）

Benchmark 仅向模型馈送 **单细胞表达**（`adata.X` 导出的 token + 连续 value）。`obsm['pert_var_idx']`
若存在，只用于在 **采样/截断** 时锁定对应基因 token（含其表达值），避免被 `DataCollator`
随机丢掉——语义是 **protected-gene coverage**，不是并行 perturbation 条件输入。实现上 pert 基因
与第三方 `scgpt/tasks/cell_emb.py` 的 **非零基因列顺序（`np.nonzero`）** 不同：本 adapter 对候选基因
（protected ∪ 非零）按 **表达量降序、列索引升序** 排序后再序列化，属于 **latent_bench 口径化** 行为，
与官方 cell_emb **数值不必逐位一致**；对比官方脚本时需单独说明。

## 修改文件清单

- [`adapters/scgpt/encoder.py`](../../adapters/scgpt/encoder.py)（新建）

## 关键实现

- 复用 third_party `scgpt.data_collator.DataCollator`，在外层使用 **`ProtectedGeneDataCollator`**：
  在 `keep_first_n_tokens=1`（仅 `<cls>` 固定）前提下，对 `expressions[1:]` 做分箱与带 protected 集合的
  截断/子采样，再 `model._encode` 取 CLS；**不是**「pert 块整块锁定在 `<cls>` 后」的旧文档模型。
- `_Dataset.__getitem__`：`<cls>` + `pad_value`；候选索引为 **protected（来自 `obsm['pert_var_idx']`）∪ 非零表达列**，
  按 `(-expr, col_idx)` 排序后依次追加 token 与表达值。
- `_patch_flash_mha`：`flash_attn>=2.x` API 兼容子类。
- 模型加载时同时兼容 `torch.load(..., weights_only=False)`，缺失该 kwarg 时回落到旧签名。

## Smoke 验证（历史记录；仓库当前无 `smoke/test_scgpt.py`）

此前可用命令形如：

```bash
$SCFM_ENVS_ROOT/scdfm/bin/python data/scFM/fm/smoke/test_scgpt.py
```

- 输入：8 control / 8 ASCC3 / 8 SCYL1 / 8 ASCC3+SCYL1（synthetic 多扰）
- 输出：`emb shape (32, 512)`，`mean L2 diff (force vs off) = 0.4090`
- `pert_kept_histogram = {0:8, 1:16, 2:8, 3+:0}`（与构造分布一致）
- 默认 `LATENT_BENCH_SCGPT_MODEL_DIR=<delivery_root>/pretrained/scgpt`

## 已知风险

- 当 GPU 缺 flash-attn 时需手动指定 `use_fast_transformer=False`（当前依赖 flash 路径，与
  third_party `cell_emb` 一致）。
- `ProtectedGeneDataCollator` 面向 cell-emb 推理路径，不覆盖上游 MLM 训练 collator 全部分支。

## log1p 审计（Code-only audit）

**结论：CORRECT（在基准约定下无需修改数值流）。** 基准 `adata.X` 为
`log1p(normalize_total)` 后的浮点矩阵；该 adapter 的数值路径在此约定下与 scGPT
预训练分布一致，且绝大多数位置对单调变换不敏感。

### 路径追踪（`adata.X` → `_encode`）

1. `_Dataset.__getitem__`（`adapters/scgpt/encoder.py:278-314`）：
   - 取 `row = self.count_matrix[idx]`（即 `adata.X[idx]` 在 vocab 过滤后的子矩阵上；未做额外
     `log1p` / `normalize_total` / 行归一化）。
   - 以 `<cls>` 开头，值为 `model_configs["pad_value"]`。
   - 当 `force_pert=True` 且存在 `obsm['pert_var_idx']` 时，**protected 基因集合**与所有非零表达位并集后，按 `(-row[j], j)` 排序，形成基因 token 序列与 `protected_mask`（**不是** `obs` 上的 perturbation 条件流）。
2. `ProtectedGeneDataCollator.__call__`（`adapters/scgpt/encoder.py:67-103`）+ 底层
   `DataCollator(..., keep_first_n_tokens=1, do_binning=True, sampling=True)`（`adapters/scgpt/encoder.py:316-325`）：
   - 对 `expressions[1:]` 调 `scgpt.preprocess.binning`（`n_bins=51`），`<cls>` 位不参与分箱。
   - `_sample_or_truncate_plus_pad_keep_protected` 在超长序列时**优先保留** protected 位，再对剩余位随机下采样，最后 padding 到 `max_length`。
3. `TransformerModel._encode`（`third_party/scGPT-main/scgpt/model/model.py:167-197`）：
   - `values = self.value_encoder(values)`；`input_emb_style="continuous"`
     走 `ContinuousValueEncoder`（`model.py:765-792`），对标量做 `Linear(1,d) → ReLU
     → Linear(d,d) → LayerNorm`，并在 `model.py:788` 将 x clamp 到 `max_value=512`。
   - `total_embs = src + values`；再进入 Transformer。

### 分箱的单调不变性证明

`scgpt.preprocess.binning`（`third_party/scGPT-main/scgpt/preprocess.py:274-303`）：

- `model.py:293-299`：当 `row.min() <= 0` 时，仅对非零子集计算
  `bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))`。
- `model.py:300-302`：否则对整行计算同样的 quantile 切点。
- 随后 `_digitize(row, bins)`（`preprocess.py:239-270`）做 `np.digitize`。

关键事实：`log1p` 在 `x ≥ 0` 上严格单调递增，且 `log1p(0) = 0`。因此：

- 零元素集合不变（仍为零），非零元素集合不变。
- 非零元素的**秩序**被保持，`np.quantile` 的切点随值域同向单调变换，
  `np.digitize` 返回的 bin index 与用 raw counts 得到的一致。
- 结论：**凡是进入 `binning` 的位置（`expressions[1:]`，即除 `<cls>` 外的全部基因
  token；见 `ProtectedGeneDataCollator`），其最终输入到 `value_encoder` 的整数分箱值对
  log1p 与 raw 之间的单调变换完全不变。**

### 例外与残余风险

- **`<cls>` 位**：`pad_value` 是模型常量，不受输入尺度影响（invariant by
  construction）。
- **基因 token 位**（`expressions[1:]`）：统一经 `binning` 后送 `value_encoder`；
  protected 基因与纯表达位**同一分箱规则**，无历史上的「pert 前缀免分箱」旁路。
- **下游重复归一化**：在 `adapters/scgpt/encoder.py` 中没有任何
  `log1p` / `normalize_total` / 行求和归一化——审计通过。
- **`force_pert_head_tokens`**：本 adapter 使用
  `ProtectedGeneDataCollator` + 有序基因列表，**未** import 或使用
  `_common.force_pert_head_tokens` ——符合预期。
- **`adata` 原地修改**：原实现会将 `id_in_vocab`（以及 `gene_col=="index"` 分支
  下的 `index`）写回调用方传入的 `adata.var`。本次审计将列计算移至本地数组
  `id_in_vocab`，仅通过 `adata[:, keep_mask].copy()` 产出新对象，**不再污染调用方
  adata**。

### 决策

- 保留 `input_is_log1p: bool = True` 仅作为 API 一致性标志（与
  stack/scldm/xVERSE/Geneformer 对齐），**不做运行时变换**；
- 文档与 docstring 明确了 pert 位对尺度敏感这一细节，以及基准约定即预训练分布；
- 本次未修改 `third_party/`；未跑 smoke 测试。

### 本次改动

- `adapters/scgpt/encoder.py`：细化 `encode()` docstring；将 `adata.var` 的
  `id_in_vocab` 计算改为本地数组，避免原地修改调用方 AnnData。
- `docs/encoder_impl/scgpt.md`：新增本节。

## log1p 复核（Re-audit, code-only）

二次 code-only 复核确认上节结论，仅列出关键定位行号，避免重复叙述。

### 逐步追踪

- 入口：`adapters/scgpt/encoder.py:222-223` `count_matrix = adata.X`（直接读
  `adata.X`，不做 `log1p` / `normalize_total` / 行归一化）。
- `_Dataset.__getitem__`（`adapters/scgpt/encoder.py:225-267`）：
  - 首位 `<cls>` 值 = `model_configs["pad_value"]`（常量）；
  - 其后为 pert prefix + 非零基因 token，全部填 `float(row[j])` 原始值；
  - `keep_first_n_tokens = min(1 + len(pert_ordered), max_length)`。
- `PerCellPertDataCollator.__call__`（`adapters/scgpt/encoder.py:68-101`）：
  - `expressions.clone()` 防止共享 storage 的就地改写；
  - 分箱仅作用于 `expressions[kfn:]`，即 `<cls>` 与所有 pert 位被
    `keep_first_n_tokens` 完整保护。
- `DataCollator._sample_or_truncate_plus_pad` + `_sample`
  （`third_party/scGPT-main/scgpt/data_collator.py:134-171`）：当
  `keep_first_n_tokens>0` 时，前 `_n` 位固定保留、仅对尾部随机打乱，语义正确。
- 模型路径 `TransformerModel._encode`
  （`third_party/scGPT-main/scgpt/model/model.py:167-197`）：
  `ContinuousValueEncoder` 在 `model.py:779-792` 做 `unsqueeze → clamp(max=512)
  → Linear → ReLU → Linear → LayerNorm`，直接以分箱整数 / pert 原值作为标量输入。

### 分箱对单调变换的不变性

`scgpt.preprocess.binning`（`third_party/scGPT-main/scgpt/preprocess.py:274-303`）
严格基于经验分位数：

- `preprocess.py:293-299` 对 `row.min() <= 0` 的情况取非零子集的
  `np.quantile(non_zero_row, np.linspace(0, 1, n_bins-1))` 作为切点；
- `preprocess.py:300-302` 对全正情况对整行取切点；
- `_digitize`（`preprocess.py:239-271`）等价于 `np.digitize` 的 rank 分桶。

`log1p` 在 `[0, +∞)` 严格单调递增且 `log1p(0)=0`，因此：

1. 零元素映射不变；
2. 非零元素的相对秩、`np.quantile` 切点位置（在变换后域上）与原域一一对应，
   桶 id 完全一致。

→ 凡进入分箱的位置对 log1p ↔ raw 完全不变。

### `<cls>` / pert 位的处理

- `<cls>`（索引 0）：`values_list.append(pad_val)`（`encoder.py:240`），并由
  `kfn ≥ 1` 始终排除在分箱之外；`pad_val` 来自 `args.json`（常量），与输入
  尺度无关，`force_pert` 头的语义未被污染。
- Pert prefix（索引 `1..1+len(pert)`）：同样由
  `kfn = 1 + len(pert_ordered)` 排除在分箱之外，保留原始（log1p）值。
  严格讲对 log1p ↔ raw **并非不变**，但基准约定 `input_is_log1p=True` 恰好与
  scGPT 预训练分布对齐，属于正确路径。

### 其它核验

1. 下游重复归一化：**无**。`encode()` 路径除 L2 norm
   （`encoder.py:308`，作用于 cell embedding 而非输入表达）外无任何 log1p /
   row-sum / normalize_total 调用。
2. `force_pert_head_tokens`（`_common`）：**未被使用**，scGPT 通过
   `PerCellPertDataCollator` 走自己的 per-cell prefix 路径。
3. 调用方 `adata` 不被污染：当前
   `encoder.py:176-187` 已把 `id_in_vocab` 计算放在局部 numpy 数组中，
   `adata = adata[:, keep_mask].copy()` 仅绑定局部名；不存在
   `adata.var[...] = ...` 的赋值语句，原始 AnnData 保持不变。

### 结论

- **Verdict: CORRECT (invariant)**，当前代码无需改动。
- 证据：`preprocess.py:274-303`（rank 分箱）、`data_collator.py:87-91`（分箱
  入口）、`encoder.py:68-101`（per-cell `keep_first_n_tokens`
  保护 pert/`<cls>` 位）、`encoder.py:176-187`（不污染 `adata`）。
- 残余风险：
  1. 若调用方违反基准约定传入 raw counts，pert prefix 位会 OOD
     （`ContinuousValueEncoder.max_value=512` 截断），embedding 质量下降但不
     报错；已在 docstring 与 meta 中提示。
  2. `_digitize` 中对切点处的并列使用 `np.random.rand` 做 tie-break
     （`preprocess.py:267-270`），每次分箱可能有 ≤1 bin 的随机漂移；这与
     log1p 无关，是 scGPT 上游固有行为。
  3. `DataCollator._sample`（`data_collator.py:151-171`）会对非 prefix 位
     做随机采样/打乱，跨 run 稳定性依赖 PyTorch RNG；同样与 log1p 无关。


---

## 审计摘要（历史 encoder_audit）


## 入口

- `scgpt/tasks/cell_emb.py`：`get_batch_cell_embeddings` → `DataCollator` + `Dataset`（非零基因 + 前缀 `<cls>`）。
- `scgpt/data_collator.py`：`keep_first_n_tokens=1` 时随机采样保留首 token。

## 长度策略

- `max_length` 默认 1200；padding + 随机子采样。

## 改造（已实现）

- **Adapter**：[`adapters/scgpt/encoder.py`](../../adapters/scgpt/encoder.py) 中 `PerCellPertDataCollator` + 每细胞 `keep_first_n_tokens = 1 + |P_i|`。

## 验收 checklist

- [ ] 本地 `best_model.pt` 存在；`LATENT_BENCH_SCGPT_MODEL_DIR` 指向。
- [ ] 单扰扰动基因出现在 `<cls>` 之后。
