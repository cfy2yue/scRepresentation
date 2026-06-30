# TranscriptFormer adapter

## 角色

- `encoder_role`: `ExpressionOnlyEncoder`
- 输出：官方 CLI 写入的 `obsm["embeddings"]`，当前 `tf_sapiens` 为 2048 维 cell embedding。
- 条件输入：无；不读取 perturbation metadata 作为条件。

## 官方要求

官方仓库：`https://github.com/czi-ai/transcriptformer`

官方 README 说明：

- Python >= 3.11；
- 推荐通过 `transcriptformer download tf-sapiens|tf-exemplar|tf-metazoa` 下载 checkpoint；
- `tf_sapiens/` checkpoint 目录包含 `config.json`、`model_weights.pt`、`vocabs/`；
- GPU 推理推荐 A100，16GB GPU 可用 batch size 1-4；
- CLI 支持 `inference`，输出 AnnData `obsm["embeddings"]`。

本 benchmark 当前默认：

- checkpoint: `<delivery_root>/scFM_pretrained/transcriptformer/tf_sapiens`
- third-party source: `<delivery_root>/scFM_third_party/transcriptformer/src`
- model env: `$SCFM_ENVS_ROOT/transcriptformer/bin/python`

## 输入口径

TranscriptFormer 需要 raw/count-like expression。Benchmark staging `X` 通常已是
`log1p(normalize_total)`，因此 adapter **不会**：

- 对 `X` 再做 `log1p`；
- 静默 `expm1(X)` 伪造 counts。

当 `input_is_log1p=True` 时，只允许显式 count 来源：

1. `adata.raw.X`
2. `layers["counts"]`
3. `layers["raw_counts"]`
4. `layers["count"]`

找不到则 fail-fast。当前可正式运行的 benchmark 输入为 SciPlex/chemical perturbation
文件，因为它们有 `layers["counts"]`。

基因 ID：

- 默认传给官方 CLI 的列名是 `ensembl_id`；
- 如果输入没有该列，adapter 会从 `ENSEMBL`、`ensemblid`、`Ensembl_ID` 等列复制；
- meta 记录 `gene_col_name` 和 `gene_col_source`。

## 已验证输出

已存在并通过 validator 的 full chempert embedding：

| dataset | cells | dim | counts source |
| --- | ---: | ---: | --- |
| `sciplex3_A549` | 55,173 | 2048 | `layers['counts']` |
| `sciplex3_K562` | 56,657 | 2048 | `layers['counts']` |
| `sciplex3_MCF7` | 56,846 | 2048 | `layers['counts']` |
| `sciplex3_xCellLine` | 168,676 | 2048 | `layers['counts']` |

Smoke:

```bash
CUDA_VISIBLE_DEVICES=2 $SCFM_ENVS_ROOT/transcriptformer/bin/python \
  fm/tools/export_embedding_one.py \
  --model transcriptformer \
  --adata /data/cyx/1030/dataset/scFM_data/staging/chempert/sciplex3_A549.h5ad \
  --out-dir /data/cyx/1030/scLatent/scFM_output/embeddings/transcriptformer/sciplex3_A549/smoke128_bs4 \
  --device cuda --batch-size 4 --max-cells 128
```

Observed: `(128, 2048)` float32, finite, wall time ~36s including model load.

## 当前限制

- Atlas/genepert staging 文件缺显式 raw/count source，不能合法用于 TranscriptFormer formal benchmark。
- 旧 full outputs 的 meta 里 `gene_col_name="ensembl_id"`，这是 adapter 写给官方 CLI 的临时列名；实际来源为 input `var["ENSEMBL"]`。新代码会写 `gene_col_source`。
- Full output 目前覆盖 chempert，可先纳入 drug perturbation benchmark；gene/atlas 需要补 counts layer 或 raw input。
