# `tools/` — repo hygiene & data helpers

## Gatekeeping（无 PyTorch import）

- **`validate_repo.py`** — AST-parse `model/` 下 `.py`，检查必需文件、`pert_split.py` legacy 字样，以及对历史路径/token 的黑名单扫描（黑名单定义见脚本顶部 `BLACKLIST`）。
  ```bash
  python tools/validate_repo.py
  ```

## Split / 数据契约

- **`build_split.py`** — 写入与 `utils.data.split` 一致的 canonical `split_seed*.json`（需在具备 `biflow_dir` 数据的环境运行；参数见 `--help`）。

## Embedding export（缓存gene / chem 向量给 conditioning）

- **`export_gene_embedding_cache.py`** — 导出基因级嵌入缓存（供 `GeneEmbeddingCache` / perturbation conditioning）。
- **`export_chem_embedding_cache.py`** — 化学 perturb 嵌入占位导出路径（配合 `chem_emb_source_dir`、`pert_chem_enabled`）。
- **`export_state_perturbation_cache.py`** — 将 State tx 训练目录中的 `pert_onehot_map.pt`
  导出为 `DrugEmbeddingCache` 布局（`drug_embeddings.npy`、`drug_index.tsv`、
  `manifest.json`），可作为 LatentFM 药物条件的短期 label-based cache。
- **`diagnose_latent_conditioning.py`** — 只读取 `condition_metadata.json`，汇总
  LatentFM 数据中的 perturbation type、空 gene 条件、药物条件数，并可检查
  `DrugEmbeddingCache` 命中率：
  ```bash
  python -m model.tools.diagnose_latent_conditioning \
    --data-dir /path/to/latentfm_full/stack \
    --gene-cache-dir /path/to/gene_cache \
    --drug-cache-dir /path/to/drug_cache
  ```
- LatentFM prior-correction diagnostics live under `model/latent/`:
  `python -m model.latent.evaluate_prior_correction --help`.

## Backbone 流水线子目录

- **`stack_embedding/`**、 **`scldm_embedding/`** — 若干 backbone 的单向 step 脚本与 inspect 辅助。
- 名称含 **`scfoundation_embedding/`** 的目录为可选工具脚本；请勿在 **`utils/`/`latent/`/`model/`** 等新代码中字面引用已禁止的历史标识（见 `validate_repo.py` 黑名单）。
