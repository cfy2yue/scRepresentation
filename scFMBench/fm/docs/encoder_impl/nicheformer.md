# NicheFormer adapter

## 角色

- `encoder_role`: `ExpressionOnlyEncoder`
- 输出：`Nicheformer.get_embeddings(..., layer=-1)` 的 cell embedding。
- 条件输入：无；不把 perturbation metadata 作为条件。

## 官方要求

官方仓库：`https://github.com/theislab/nicheformer`

官方 README 说明：

- Python >= 3.9；
- 建议独立环境安装；
- 通过源码 `pip install -e .`；
- 预训练权重由作者通过 Mendeley Data 提供；新版本也在 HuggingFace 提供
  `theislab/Nicheformer` 的 `model.safetensors` 导出；
- 仓库主要提供 Lightning model 与 tokenization/data-loading 示例，而不是稳定的 h5ad inference CLI。

本 benchmark 当前默认：

- source: `<delivery_root>/scFM_third_party/nicheformer/src`
- model mean: `<delivery_root>/scFM_third_party/nicheformer/data/model_means/model.h5ad`
- checkpoint 二选一：
  - legacy Lightning: `<delivery_root>/scFM_pretrained/nicheformer/nicheformer.ckpt`
  - HuggingFace: `<delivery_root>/scFM_pretrained/nicheformer/theislab_Nicheformer/{config.json,model.safetensors,...}`

## 输入口径

NicheFormer adapter 同样要求 count-like expression。Benchmark staging `X` 通常已是
`log1p(normalize_total)`，所以 adapter **不会**：

- 对 log1p `X` 再 `log1p`；
- 静默 `expm1(X)` 伪造 counts。

当 `input_is_log1p=True` 时，只允许显式 count layer：

1. `layers["counts"]`
2. `layers["raw_counts"]`
3. `layers["count"]`

找不到则 fail-fast。

基因 ID：

- 优先要求输入 `var_names` 与 official model mean genes 对齐；
- 若 overlap 不足，会尝试 `ENSEMBL`、`ensembl_id`、`ensemblid` 等 `var` 列；
- overlap <1000 时拒绝编码。

## 当前状态

本地已有：

```text
/data/cyx/1030/scLatent/scFM_third_party/nicheformer/src
/data/cyx/1030/scLatent/scFM_third_party/nicheformer/data/model_means/model.h5ad
```

权重可以使用 legacy `.ckpt`，也可以使用 HuggingFace snapshot。默认环境变量：

```text
LATENT_BENCH_NICHEFORMER_CKPT=/data/cyx/1030/scLatent/scFM_pretrained/nicheformer/nicheformer.ckpt
LATENT_BENCH_NICHEFORMER_HF_DIR=/data/cyx/1030/scLatent/scFM_pretrained/nicheformer/theislab_Nicheformer
```

adapter 会优先使用 `.ckpt`；若缺失则使用 HuggingFace 目录。权重就绪后，应先跑
chempert smoke，因为当前只有 chempert 文件有显式 `layers["counts"]`。

## 当前限制

- 官方推理 API 不如 TranscriptFormer CLI 稳定；adapter 直接调用内部 tokenization 和
  `model.get_embeddings()`，需要跟随上游版本变动审计。
- Atlas/genepert staging 文件缺显式 raw/count source，不能合法用于 formal benchmark。
- NicheFormer 是 spatial + single-cell corpus 预训练模型，meta 会记录
  `specie/assay/modality` context token。默认按 human / 10x 3' v3 / dissociated
  设置；可用 `LATENT_BENCH_NICHEFORMER_SPECIE_TOKEN`、
  `LATENT_BENCH_NICHEFORMER_ASSAY_TOKEN`、`LATENT_BENCH_NICHEFORMER_MODALITY_TOKEN`
  覆盖。
