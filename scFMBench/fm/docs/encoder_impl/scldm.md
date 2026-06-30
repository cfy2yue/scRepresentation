# scldm — 实现反馈

## 修改文件清单

- [`adapters/scldm/encoder.py`](../../adapters/scldm/encoder.py)（替换原占位实现）
- [`adapters/scldm/__init__.py`](../../adapters/scldm/__init__.py)（新建，导出 `encode`）
- 依赖：`third_party/scldm/src/scldm/`（未改动）

**环境**：与 dataset-fitted scVI 共用 `scldm` venv（`scvi-tools` 等 pin 见 [`../env_map.md`](../env_map.md)），详见 [`scvi_baseline.md`](scvi_baseline.md)。

## 关键实现

- **Hydra 配置 + ckpt 加载**：直接读 `70M.yaml` (`OmegaConf.load`)，
  注册 `eval` resolver，然后：
  1. `scldm._utils.remap_config` 把所有 `scg_vae.*` target 重命名为 `scldm.*`；
  2. `OmegaConf.resolve(cfg)` 展开 `${...}` 插值（包括 `${datamodule.label_encoder.n_genes}=36130`、
     `${model.decoder_head.${model.decoder_name}}` 等嵌套引用）；
  3. 强制 `module_cfg.compile=False`（`torch.compile` 只在训练 `on_fit_start` 触发，推理时无需）；
  4. `hydra.utils.instantiate(module_cfg)` 直接得到 `scldm.models.VAE` LightningModule。
- **checkpoint**：`torch.load(ckpt, pickle_module=remap_pickle, weights_only=False)`，取
  `state_dict`；ckpt 同时包含 `vae_model.*` 与 `vae_model_compiled._orig_mod.*`（训练时
  `torch.compile` 遗留的副本），仅保留 module 本身识别的 178 个 key，
  `load_state_dict(filtered, strict=True)` —— missing=0，extra 的 compile 副本忽略。
- **基因对齐**：`concatenated_unique_genes.parquet`（36130 genes，ENSG `feature_id` + symbol
  `feature_name`）。adata 先按 `var['Ensembl_ID']` 匹配 ENSG；没有 `Ensembl_ID`/ENSG 前缀时再
  fallback 按 symbol（大写）匹配。缺失的 vocab 基因填 0 列，多余的 adata 基因直接丢弃。Adamson
  smoke：3065/3086 命中 ENSG 匹配。
- **输入构造**（对齐训练 `sample_genes="expressed"`, `genes_seq_len=8000`）：
  - `counts`: 全 36130 列，`genes = [1..n_vocab]`（mask token idx=0 保留给 padding）。
  - `counts_subset`/`genes_subset`: 把每个 cell 的非零表达装入 `(N, 8000)`（token id = 列索引+1，
    末尾用 0=mask pad）。如果单 cell 非零基因 > 8000 会自动扩容 seq_len 以避免越界。
- **编码**：`module.vae_model.encode(counts, genes, counts_subset, genes_subset)` 返回
  `(B, n_inducing_points=256, n_embed_latent=16)`，flatten 得 `(B, 4096)` float32。
  `TransformerVAE` 是确定性 encoder（非 reparameterization），输出**本身就是 posterior mean**，
  满足“use mu, not a sample”约束。
- **subset 与全基因**：adapter 会构造全 36130 维 `counts`/`genes` 与 per-cell `counts_subset`/`genes_subset`。
  上游 [`TransformerVAE.encode`](../../third_party/scldm/src/scldm/vae.py) 在 **`counts_subset` 非空时只把 subset 送入 `input_layer`**，全基因张量在该分支下**不参与** encoder 前向；信息量以 **expressed（+protected）子集** 为准。
- **force_pert 与 subset**：当 `force_pert=True` 且存在 `obsm['pert_var_idx']` 时，映射后的 vocab 索引作为 **protected set** 写入 `_build_expressed_subset`，避免截断时丢掉这些列——属 coverage 而非扰动条件流。
  `force_pert_effective` 在 manifest 中为 `force_pert && pert_var_idx 存在`（见 adapter meta）。
- **精度**：forward 在 `torch.amp.autocast("cuda", bfloat16)` 下跑（ckpt 按 bf16-mixed 训练），
  输出 `float32`。

## Smoke 验证（历史记录；仓库当前无 `smoke/test_scldm.py`）

此前可用命令形如：

```bash
cd <delivery_root>
CUDA_VISIBLE_DEVICES=1 $SCFM_ENVS_ROOT/scldm/bin/python \
  data/scFM/fm/smoke/test_scldm.py
```

结果（最后约 15 行）：

```
adata=(32, 3086), perts={'ASCC3': 8, 'ASCC3+SCYL1': 8, 'SCYL1': 8, 'control': 8}
pert_var_idx shape=(32, 2)
emb shape=(32, 4096), dtype=float32
meta = {
  "latent_dim": 4096,
  "vocab_hits": 3065,
  "vocab_size": 36130,
  "force_pert": true,
  "force_pert_effective": true,
  "pert_kept_histogram": {
    "0": 8,
    "1": 16,
    "2": 8,
    "3+": 0
  }
}
scldm smoke test PASSED
```

- `pert_kept_histogram = {0:8, 1:16, 2:8, 3+:0}` 与 Adamson 构造一致。
- 默认路径：`<delivery_root>/pretrained/scdlm/vae_census/{70M.ckpt,70M.yaml,concatenated_unique_genes.parquet}`，
  可用 `LATENT_BENCH_SCLDM_CKPT/_CFG/_GENES` 覆盖。

## 已知风险 / 后续

- 展平 `(256, 16) → 4096` 的 latent 维度较大；benchmark 下游如果有 per-token linear probe，
  需要注意不要再做一次 flatten。
- 没有调用 `VAE.inference`/`predict_step`（那条路径依赖 `trainer.datamodule.vocabulary_encoder`
  来 decode genes，在纯 inference 场景多余）。
- ckpt 中的 `vae_model_compiled._orig_mod.*` 副本被显式忽略；若未来 scldm 改成不再保存
  compile 副本，本 adapter 仍可正常工作（`filtered` 由 module keys 驱动）。
- `sample_genes="expressed"` 对齐训练；如果个别 cell 非零基因超过 8000，adapter 会自动把
  `seq_len` 扩到实际最大值（防越界），Adamson smoke 最大 2033 << 8000。
- 依赖 `$SCFM_ENVS_ROOT/scldm/bin/python`，该环境已装 `hydra-core`,
  `pytorch_lightning==2.6.1`, `flex_attention` (via `torch>=2.5`), `cellarium-ml`（datamodule
  imports 会触发，但我们不实例化 datamodule，故不会真正用到）。

## log1p audit（2026-04-20，code-only）

**Verdict: CORRECT**（一次性 log1p，无双重 log）。

- 训练/推理路径：`TransformerVAE.encode` → `InputTransformerVAE` → `ProjectionConcat`
  （由 `70M.yaml` 中 `agg_func=projconcat` 选择），在 `scldm/layers.py` ~L62 对
  `counts` 调用 `torch.log1p(counts)`。
- 不走的路径（**不**在 70M 配置上）：`nnets.EncoderScvi` 的 `torch.log1p`
  (`scldm/nnets.py` ~L43) 与 `log1p_transform` helper (`scldm/layers.py` ~L28)。
- Adapter 约定：`adapters/scldm/encoder.py:157` `input_is_log1p: bool = True`。
  - 在 `_align_expression_to_vocab` 之后（L206），若 `input_is_log1p=True`：
    - `X_full = np.expm1(np.clip(X_full, 0.0, None)).astype(np.float32)`（L212-214）。
    - `counts_subset` 由 `_build_expressed_subset(X_full, ...)` 从已 `expm1` 的
      `X_full` 派生（L216），所以两路入口（全向量 + expressed subset）**同源**。
  - 之后模型内部再做一次 `log1p`，等价于对原始 `log1p` 输入做恒等变换（至 float 精度）。
- `meta["input_is_log1p"]` 记录口径，便于下游审计（L243）。
- 残余风险：
  1. 如果上游 `normalize_total` 的 `target_sum` 与 scldm 预训练所用不一致，
     `expm1→log1p` 的数值仍然准确，但量级可能与预训练分布不完全对齐（属约定问题）。
  2. `np.clip(X, 0, None)` 防止 log1p 输入存在极小负值（浮点噪声）造成 `expm1<0`；
     若上游保证非负则是 no-op。
- Adamson smoke（`/tmp/adamson_smoke.h5ad`）`max(X)≈6.28`，`expm1(6.28)≈531`，
  与 `normalize_total(target_sum=1e4)` 的量级吻合。
- `third_party/scldm/` **未改动**。


---

## 审计摘要（历史 encoder_audit）


## 入口

- `scldm/encoder.py`：VAE 编码器；`experiments/scripts/inference.py`：Hydra 推理。

## 策略

- 全表达向量或配置维度；扰动在 `X` 中体现。

## 状态

- Adapter 已实现：[`adapters/scldm/encoder.py`](../../adapters/scldm/encoder.py)；实现与审计见本文档全文。
