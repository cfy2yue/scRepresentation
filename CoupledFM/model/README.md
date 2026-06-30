# CoupledFM `model/`

This is the lightweight code package for the scdfm handoff. It is designed to run without a sibling `scFM/` checkout.

## Delivery Layout

Place this package under a delivery root with two sibling resource directories:

```text
<delivery_root>/
  model/          # code, scripts, docs, lightweight configs
  dataset/        # local data, not committed or uploaded
  pretrainckpt/   # pretrained resources and gene caches, not committed or uploaded
```

Default paths can be overridden with:

```bash
export SCDFM_DATASET_ROOT=/path/to/dataset
export SCDFM_PRETRAIN_ROOT=/path/to/pretrainckpt
export SCDFM_GENE_CACHE_ROOT=/path/to/pretrainckpt/genepert_cache
```

`COUPLEDFM_SCFM_ROOT` 仅作为显式兼容覆盖，**只影响 cellnavi 资源解析路径**（即 `gene_name.txt` / `Nichenet/` / `pretrain_weights.pth`），不影响 dataset / pretrain / gene cache 根目录。未设置时一律走 `SCDFM_PRETRAIN_ROOT`（默认 `<delivery_root>/pretrainckpt`）。

## Required Resources

```text
dataset/
  biFlow_data/
    control_stack/{dataset}.h5ad
    gt_stack/{dataset}.h5ad
    control_center_stack/{dataset}.h5ad
    split_seed42.json
  cellgene_census/processed/
    tissue_metainfo.csv
    celltype_metainfo.csv
    kidney/kidney_top6000var.h5ad
    ...

pretrainckpt/
  cellnavi/data/
    gene_name.txt
    Nichenet/node2idx.json
    Nichenet/graph.pkl
    pretrain/pretrain_weights.pth
  genepert_cache/
    cellnavi_embed_gene/{gene_embeddings.npy,gene_index.tsv,manifest.json}
    scgpt_embed_gene/{gene_embeddings.npy,gene_index.tsv,manifest.json}
```

Check the resolved paths and missing files:

```bash
cd <delivery_root>
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m model.tools.validate_resources
```

## Main Entrypoints

```text
model/scripts/submit_cellgene_pretrain.sh        # raw flow pretrain
model/scripts/submit_pert_embed_compare_8gpu.sh  # CellNavi vs scGPT condition comparison
model/scripts/sweep_gene_pert_32.sh              # CoupledFM parameter sweep
model/tests/local_single_gpu_smoke.sh            # local GPU smoke
```

Quick smoke:

```bash
cd <delivery_root>
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
GPU=0 RUN_SCGPT=0 PRETRAIN_STEPS=1 PRETRAIN_BATCH=2 PRETRAIN_MICRO=1 \
COUPLED_EPOCHS=1 COUPLED_BATCH=2 COUPLED_MICRO=1 \
bash model/tests/local_single_gpu_smoke.sh
```

See [TEST_TUTORIAL.md](TEST_TUTORIAL.md) and [RUN_TUTORIAL.md](RUN_TUTORIAL.md) for the full test and H20/H40 run commands.
