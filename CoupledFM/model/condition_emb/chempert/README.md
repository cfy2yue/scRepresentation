# ChemPert (chemical perturbation) — integration contract

This directory holds **interfaces and docs** only. Gene-level conditioning stays in
[`../genepert`](../genepert); chemical vectors flow through the same
`PerturbationBatch` / `resolve_chem_embedding` path.

## Enable in training

1. **Caches**: export embeddings with [`../genepert/tools/export_chem_embedding_cache.py`](../genepert/tools/export_chem_embedding_cache.py)
   → `ChemEmbeddingCache` layout (`embeddings.npy`, `manifest.json`).  
     *Naming:* gene-level caches use **`gene_embeddings.npy`** under [`genepert`](../genepert/README.md); chem caches use **`embeddings.npy`** here.
2. **Data config**:
   - `pert_chem_enabled=True`
   - `chem_emb_source_dir=<cache_root>`
   - Optional: `chem_obs_column`, `chemical_metainfo_path`
   - **Model**: set `pert_chem_emb_dim` / `pert_chem_projector_hidden` > 0 on the velocity field /
     `ControlMLP` so `PerturbationConditionEncoder` builds the chem branch.

## Implement a new backbone

Subclass `ChemPertEncoderProvider` in a new module under this package (or vendor a model),
then extend `resolve_chem_embedding` (or a parallel hook) to return a 1-D `float32` vector per cell/condition row.

Shapes must match `pert_chem_emb_dim` in the encoder config.

## Relation to GenePert

Runtime code intentionally lives under `genepert` for historical reasons (`ChemEmbeddingCache`,
`resolve_chem_embedding`). This `chempert/` package is the **semantic home** for future chem-specific
implementations so the repo layout stays clear.
