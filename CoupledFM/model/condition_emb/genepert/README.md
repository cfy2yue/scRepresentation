# genepert — shared perturbation condition encoding

Small library used by **latent**, **coupled**, and (when vendored) **raw_independent** for parsing biFlow / AnnData perturbation metadata, loading disk-backed gene and chem embedding caches, and feeding `PerturbationConditionEncoder`.

This package only handles **condition encoding**. Dataset manifests such as `metainfo.json` remain under `data/raw/` (not moved here).

## Modules

| File | Role |
|------|------|
| `perturbation.py` | Gene string parsing, type IDs, `ConditionMetadata` / `PerturbationBatch`, batch ↔ device helpers |
| `perturbation_encoder.py` | `PerturbationConditionEncoder` (learned / pretrained gene + type + optional chem branch) |
| `h5ad_obs.py` | `obs` column discovery (`pick_obs_columns`), `condition_metadata_from_obs_row` |
| `metainfo.py` | `load_dataset_metainfo`, `apply_pert_metainfo_fallback` from genepert-style JSON |
| `chem_embedding_hook.py` | `ChemEmbeddingCache`, `resolve_chem_embedding` for precomputed molecule vectors |
| `gene_embedding_source.py` | `GeneEmbeddingLoaderConfig`, `load_gene_embedding_cache` |
| `gene_cache.py` | `GeneEmbeddingCache`, `GeneEmbeddingTable` (numpy + index + manifest on disk) |
| `tools/export_gene_embedding_cache.py` | Build `gene_embeddings.npy` + `gene_index.tsv` + `manifest.json` from CellNavi / other pretrained checkpoints |
| `tools/export_chem_embedding_cache.py` | Build chem cache (`embeddings.npy` + `index.tsv` + `manifest.json`) |

## Bundled cache: `cache/cellnavi_embed_gene`

Derived from CellNavi `pretrain_weights.pth` `embed_gene.0.weight`, aligned to CellNavi `gene_name.txt`.

- **Hit rate:** 100% (**18801** / **18801** symbols)
- **`embed_dim`:** **256**
- **Layout:** `gene_embeddings.npy`, `gene_index.tsv`, `manifest.json`

## Optional comparison cache: `cache/scgpt_embed_gene`

Derived from scGPT `best_model.pt` `encoder.embedding.weight`, restricted to CellNavi `gene_name.txt`.

- **Hit rate:** ~98.3% (**18477** / **18801** symbols)
- **`embed_dim`:** **512**
- **Layout:** `gene_embeddings.npy`, `gene_index.tsv`, `manifest.json`

## CLI examples

From repo root with `PYTHONPATH=.`:

```bash
python model/condition_emb/genepert/tools/export_gene_embedding_cache.py \
  --format cellnavi_ckpt \
  --ckpt-path pretrainckpt/cellnavi/data/pretrain/pretrain_weights.pth \
  --gene-name-path pretrainckpt/cellnavi/data/gene_name.txt \
  --restrict-genes pretrainckpt/cellnavi/data/gene_name.txt \
  --out-dir pretrainckpt/genepert_cache/cellnavi_embed_gene

python model/condition_emb/genepert/tools/export_gene_embedding_cache.py \
  --format scgpt_ckpt \
  --ckpt-path pretrainckpt/scgpt/best_model.pt \
  --vocab-path pretrainckpt/scgpt/vocab.json \
  --restrict-genes pretrainckpt/cellnavi/data/gene_name.txt \
  --out-dir pretrainckpt/genepert_cache/scgpt_embed_gene

python condition_emb/genepert/tools/export_chem_embedding_cache.py \
  --format passthrough_dict --input path/to/vecs.json --out-dir path/to/chem_cache
```

## Importing in code

```python
from condition_emb.genepert import PerturbationBatch, ConditionMetadata, GeneEmbeddingCache, PerturbationConditionEncoder
```

Latent / coupled **config** defaults point at `<delivery_root>/pretrainckpt/genepert_cache/scgpt_embed_gene` for `pert_gene_emb_cache_dir`（可通过 `SCDFM_GENE_CACHE_ROOT` 覆盖）。CellNavi remains supported for sensitivity analysis or reproduction by explicitly setting the cache path / `PERT_EMBED_SOURCE`.

## raw_independent（vendored）

仓库内已同步一份：`raw_independent/condition_emb/`（与 `CoupledFM/condition_emb/` 同源）。`raw_independent/src` 各入口里的 `sys.path` 顺序保证 **`raw_independent` 根先于 `CoupledFM` 根**，从而 `import condition_emb.genepert` 解析到 vendored 拷贝。单端打包时仅需 `raw_independent/` + 顶层 `condition_emb/` 即可，不必再依赖主仓库里的 `utils/conditioning` shim。

维护：主仓 `condition_emb/` 变更后执行  
`rsync -a CoupledFM/condition_emb/ CoupledFM/raw_independent/condition_emb/`（或由 CI 校验 hash）。
