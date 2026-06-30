# 测试教程

所有命令默认从交付根目录运行：

```bash
cd <delivery_root>
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

目录约定：

```text
<delivery_root>/
  model/
  dataset/
  pretrainckpt/
```

## 1. 资源检查

```bash
python -m model.tools.validate_resources
```

只检查本地单卡 smoke 需要的最小资源：

```bash
python -m model.tools.validate_resources --mode local-smoke --datasets Adamson
```

dry-run 打印解析到的路径和期望目录树，不强制检查文件：

```bash
python -m model.tools.validate_resources --print-only
```

## 2. 关键文件

```bash
ls dataset/biFlow_data/control_stack/Adamson.h5ad
ls dataset/biFlow_data/gt_stack/Adamson.h5ad
ls dataset/biFlow_data/split_seed42.json
ls dataset/cellgene_census/processed/kidney/kidney_top6000var.h5ad
ls pretrainckpt/cellnavi/data/gene_name.txt
ls pretrainckpt/cellnavi/data/Nichenet/node2idx.json
ls pretrainckpt/cellnavi/data/pretrain/pretrain_weights.pth
ls pretrainckpt/genepert_cache/cellnavi_embed_gene/gene_embeddings.npy
ls pretrainckpt/genepert_cache/scgpt_embed_gene/gene_embeddings.npy
```

## 3. CPU 回归

```bash
python -m pytest \
  model/tests/test_multi_pool_aggregation.py \
  model/tests/test_unified_condition_embedding.py \
  -q

python model/tools/smoke_test.py
```

## 4. Launcher Dry-run

```bash
DRY_RUN=1 bash model/scripts/sweep_gene_pert_32.sh \
  --out-base /tmp/handover_sweep_dry \
  --log-dir /tmp/handover_sweep_dry/logs

bash model/scripts/submit_pert_embed_compare_8gpu.sh --dry-run \
  --gpus 0,1,2,3,4,5,6,7 \
  --out-root /tmp/handover_compare_dry \
  --log-dir /tmp/handover_compare_dry/logs
```

## 5. 本地单卡 GPU Smoke

极短验收，只跑目标流程：

```bash
GPU=0 RUN_SCGPT=0 \
PRETRAIN_EPOCHS=1 PRETRAIN_STEPS=1 PRETRAIN_BATCH=2 PRETRAIN_MICRO=1 \
COUPLED_EPOCHS=1 COUPLED_BATCH=2 COUPLED_MICRO=1 \
COUPLED_VAL_EVERY=2 COUPLED_TEST_EVERY_EPOCH=99 \
OUT_ROOT=/tmp/scdfm_local_smoke \
bash model/tests/local_single_gpu_smoke.sh
```

完整本地 smoke：

```bash
GPU=0 bash model/tests/local_single_gpu_smoke.sh
```

成功标志：

- raw pretrain log 无 `Traceback`，至少完成一次训练 step。
- coupledFM Adamson log 无 `Traceback`，至少完成一次 forward/backward 或 epoch 输出。
- 运行时即使设置 `COUPLEDFM_SCFM_ROOT=/tmp/definitely_no_scfm` 也不会访问 sibling `scFM/`。
