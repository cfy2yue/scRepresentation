# CoupledFM

CoupledFM is a coupled flow-matching model for single-cell perturbation prediction.

## Repository Layout

```text
CoupledFM/              ← git repo root (this repo)
  model/                ← all source code, scripts, configs
  dataset/              ← NOT in git; transferred separately
  pretrainckpt/         ← NOT in git; transferred separately
```

`dataset/` and `pretrainckpt/` must be placed **as siblings of `model/`** in the same directory, or their paths can be overridden with environment variables (see below).

## Quick Start

### 1. Clone the code

```bash
git clone <REPO_URL>
cd CoupledFM
```

### 2. Place data

Ensure the following sibling directories exist (already transferred to your machine):

```text
dataset/
  biFlow_data/
    control_stack/   gt_stack/   control_center_stack/   split_seed42.json
  cellgene_census/processed/   (tissue_metainfo.csv, celltype_metainfo.csv, *.h5ad …)

pretrainckpt/
  cellnavi/data/
    gene_name.txt
    Nichenet/node2idx.json   Nichenet/graph.pkl
    pretrain/pretrain_weights.pth
  genepert_cache/
    cellnavi_embed_gene/   scgpt_embed_gene/
```

If the data lives elsewhere, export overrides **before** running any script:

```bash
export SCDFM_DATASET_ROOT=/path/to/dataset
export SCDFM_PRETRAIN_ROOT=/path/to/pretrainckpt
export SCDFM_GENE_CACHE_ROOT=/path/to/pretrainckpt/genepert_cache
```

### 3. Install the environment

```bash
cd model/env
conda env create -f environment.yml      # or: bash bootstrap_h20.sh
conda activate scdfm
cd ../..
pip install -e model/                    # installs coupledfm_model in editable mode
```

### 4. Validate resources

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m model.tools.validate_resources
```

### 5. Quick smoke test (single GPU)

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
GPU=0 RUN_SCGPT=0 PRETRAIN_STEPS=1 PRETRAIN_BATCH=2 PRETRAIN_MICRO=1 \
COUPLED_EPOCHS=1 COUPLED_BATCH=2 COUPLED_MICRO=1 \
bash model/tests/local_single_gpu_smoke.sh
```

### 6. Full training runs

See **[model/RUN_TUTORIAL.md](model/RUN_TUTORIAL.md)** for H20/H40 multi-GPU launch commands.

## Main Scripts

| Script | Purpose |
|---|---|
| `model/scripts/submit_cellgene_pretrain.sh` | Raw flow pretrain |
| `model/scripts/submit_pert_embed_compare_8gpu.sh` | CellNavi vs scGPT condition comparison |
| `model/scripts/sweep_gene_pert_32.sh` | CoupledFM parameter sweep |
| `model/tests/local_single_gpu_smoke.sh` | Local single-GPU smoke test |

## Summarizing Results

After a run completes, generate a `summary/` folder with tables, curves, and report:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# 预训练 run
python -m model.tools.summarize_run output/cellgene_pretrain_YYYYMMDD_HHMMSS

# 参数搜索 sweep run
python -m model.tools.summarize_run output/sweep_gene_pert_32_YYYYMMDD_HHMMSS

# 也支持 glob（自动取最新）
python -m model.tools.summarize_run "output/sweep_gene_pert_32_*"
```

Output written to `<run_dir>/summary/`:

| 文件 | 内容 |
|---|---|
| `report.txt` | 纯文本汇总，直接发给协作者 |
| `report.md` | Markdown 格式汇总（sweep 专用） |
| `ranked_runs.csv` | 所有超参组合排名表（sweep 专用） |
| `curves_top10.png` | Top-10 run 的 val pearson_delta_ctrl 曲线（sweep 专用） |
| `curves_all.png` | 所有 run 曲线（sweep 专用） |
| `training_curve.png` | Loss 训练曲线（pretrain 专用） |
| `metrics_log.csv` | 完整 metrics 表（pretrain 专用） |

## Further Docs

- [model/README.md](model/README.md) — detailed package overview
- [model/RUN_TUTORIAL.md](model/RUN_TUTORIAL.md) — full run instructions
- [model/TEST_TUTORIAL.md](model/TEST_TUTORIAL.md) — test instructions
