# 正式运行教程

本教程只覆盖本轮交付范围：

- Cellgene Census raw flow pretrain
- CoupledFM gene perturbation 参数搜索
- CellNavi vs scGPT 条件 embedding 对比

默认交付结构：

```text
<delivery_root>/
  model/
  dataset/
  pretrainckpt/
```

从交付根目录运行：

```bash
cd <delivery_root>
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

可选路径覆盖：

```bash
export SCDFM_DATASET_ROOT=/path/to/dataset
export SCDFM_PRETRAIN_ROOT=/path/to/pretrainckpt
export SCDFM_GENE_CACHE_ROOT=/path/to/pretrainckpt/genepert_cache
```

## 1. 运行前检查

```bash
python -m model.tools.validate_resources
```

必要资源：

```text
dataset/biFlow_data/{control_stack,gt_stack,control_center_stack}
dataset/cellgene_census/processed
pretrainckpt/cellnavi/data
pretrainckpt/genepert_cache/{cellnavi_embed_gene,scgpt_embed_gene}
```

## 2. Raw Flow Pretrain

8 卡例子：

```bash
PRETRAIN_EPOCHS=20 \
PRETRAIN_BATCH=128 \
PRETRAIN_MICRO_BATCH=16 \
PRETRAIN_STEPS_PER_EPOCH=1000 \
bash model/scripts/submit_cellgene_pretrain.sh \
  --gpus 0,1,2,3,4,5,6,7 \
  --out-root /root/myproject/runs/pretrain_run_001 \
  --log-dir  /root/myproject/logs/pretrain_run_001
```

自定义 Census processed 根目录：

```bash
bash model/scripts/submit_cellgene_pretrain.sh \
  --processed-dir /path/to/cellgene_census/processed \
  --out-root /root/myproject/runs/pretrain_custom \
  --log-dir  /root/myproject/logs/pretrain_custom
```

## 3. CellNavi vs scGPT 条件 Embedding 对比

dry-run：

```bash
bash model/scripts/submit_pert_embed_compare_8gpu.sh \
  --dry-run \
  --gpus 0,1,2,3,4,5,6,7 \
  --out-root /root/myproject/runs/pert_embed_compare_dry \
  --log-dir  /root/myproject/logs/pert_embed_compare_dry
```

顺序运行：

```bash
bash model/scripts/submit_pert_embed_compare_8gpu.sh \
  --gpus 0,1,2,3,4,5,6,7 \
  --out-root /root/myproject/runs/pert_embed_compare_001 \
  --log-dir  /root/myproject/logs/pert_embed_compare_001
```

32 卡节点并发：

```bash
PARALLEL=1 \
CELLNAVI_GPUS=16,17,18,19,20,21,22,23 \
SCGPT_GPUS=24,25,26,27,28,29,30,31 \
bash model/scripts/submit_pert_embed_compare_8gpu.sh \
  --out-root /root/myproject/runs/pert_embed_compare_32gpu \
  --log-dir  /root/myproject/logs/pert_embed_compare_32gpu
```

只跑 Adamson 快速对比：

```bash
DATASETS_OVERRIDE=Adamson \
EPOCHS=5 RAW_BATCH=32 RAW_MICRO=8 \
bash model/scripts/submit_pert_embed_compare_8gpu.sh \
  --gpus 0,1,2,3,4,5,6,7 \
  --out-root /root/myproject/runs/pert_embed_compare_adamson \
  --log-dir  /root/myproject/logs/pert_embed_compare_adamson
```

## 4. CoupledFM 多参数搜索

dry-run：

```bash
bash model/scripts/sweep_gene_pert_32.sh \
  --dry-run \
  --out-base /root/myproject/runs/sweep_gene_pert_dry \
  --log-dir  /root/myproject/logs/sweep_gene_pert_dry
```

正式运行：

```bash
bash model/scripts/sweep_gene_pert_32.sh \
  --out-base /root/myproject/runs/sweep_gene_pert_001 \
  --log-dir  /root/myproject/logs/sweep_gene_pert_001
```

默认调度：

- GPU 0-15
- 8 个 slot
- 每个 slot 2 卡 DDP
- 40 个 run

常用覆盖：

```bash
DATASETS_OVERRIDE=Adamson
EPOCHS_SWEEP=30
RAW_BATCH=32
RAW_MICRO=8
PERT_EMBED_CACHE_DIR=/path/to/pretrainckpt/genepert_cache/scgpt_embed_gene
PERT_EMBED_SOURCE=scgpt_embed_gene
```

## 5. 输出约定

正式运行建议显式指定：

```text
/<root>/runs/<experiment_name>/
/<root>/logs/<experiment_name>/
```

`dataset/` 和 `pretrainckpt/` 不进 git，不随代码上传；同事只需按目录树准备资源即可。
