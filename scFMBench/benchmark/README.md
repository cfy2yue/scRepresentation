# Benchmark（`benchmark/`）

在 **已导出 latent** 上计算 **Atlas（A1–A6）**、**Geometry（G1–G6）**、**Perturbation**（centroid / OT / xCellLine）等指标；`tx_eval_ported/` 保留为移植的 treatment 评估代码。

## 指标概览

| 族 | 脚本 / 入口 | 输出 JSON 字段前缀 |
|----|-------------|-------------------|
| Atlas | `metrics/atlas_scib.py` | `A1_`…`A6_` |
| Geometry | `metrics/geometry.py` | `G1_`…`LDM_proxy_*` |
| Perturb | `metrics/perturb_geom.py`、`perturb_xcellline.py` | `centroid` / `ot` / `xcellline` |
| Post-process | `metrics/post_process.py` | 居中 / TVN 等（供下游对比） |

**契约**（列名、形状、随机种子）：[`docs/metrics_protocol.md`](docs/metrics_protocol.md)。

## Schema

- [`schema/meta.schema.json`](schema/meta.schema.json)：**导出** `meta.json`（embedding 侧 car）
- [`schema/manifest.schema.json`](schema/manifest.schema.json)：**`SCFM_OUTPUT_ROOT/embedding_runs/manifest*.jsonl`** 每行

## 一键评估

```bash
cd <delivery_root>/scFM
export PYTHONPATH="$PWD/benchmark:$PWD/fm:${PYTHONPATH:-}"
python3 benchmark/cli/run_metrics_one.py \
  --emb-dir ../scFM_output/embeddings/stack/Blood/raw
# 写入 ../scFM_output/metrics/stack/Blood/raw/{atlas,geometry,perturb,summary}.json
```

`--skip atlas geometry perturb` 可禁用子集；`--out-dir` 可覆盖默认输出目录。

## 聚合多跑次

```bash
PYTHONPATH=benchmark python3 -m cli.aggregate_report \
  --inputs '../scFM_output/metrics/**/*.json' \
  --out-csv ../scFM_output/metrics/summary_all.csv
```

（在 `benchmark/` 目录下执行时，`PYTHONPATH` 需包含当前目录。）

## 数学与验证笔记

- [`docs/metrics_validation_and_math.md`](docs/metrics_validation_and_math.md)

## Smoke

- `smoke/test_metrics_pipeline.py`：合成数据管道冒烟
