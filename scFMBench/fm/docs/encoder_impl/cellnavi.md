# CellNavi（SparseCellNaviEncoder）adapter

## 代码包 vs 数据资产（必读）

- **Python 包 / import**：仍只从 [`third_party/CellNavi`](../../third_party/CellNavi) 加入 `sys.path` 并 `import cellnavi`（与上游实现一致）。
- **默认数据资产**（`gene_name.txt`、`pretrain_weights.pth`、`Nichenet/node2idx.json`、`Nichenet/graph.pkl`）：**优先**解析本仓库下的  
  **`<delivery_root>/pretrained/cellnavi/data/`**（若文件存在）；否则回退到 `third_party/CellNavi` 内同名相对路径。  
  **NicheNet 子图与边信息**（`NicheNetGraph` 读取的 `graph.pkl`，决定稀疏子图的邻接 / 过滤）应与 `node2idx.json` **成对**放在同一目录：  
  **`<delivery_root>/pretrained/cellnavi/data/Nichenet/`**（不要把图资产与词表拆到两套根目录）。

通过环境变量或 `encode(..., checkpoint=..., graph_pkl=..., ...)` 可覆盖上述默认（兼容旧布局或 CI 只挂载部分文件的场景）。

## 角色

- **`encoder_role`**: `ExpressionOnlyEncoder`
- **Latent**: `SparseCellNaviEncoder.forward` 返回的 **`cls_out`**（CLS 节点向量，`d_model=256`），不是 node 上的临时 mean pool。
- **条件输入**: 无；不把 `obs['perturbation']` 等传入模型。

## 代码入口

| 组件 | 路径 |
|------|------|
| Adapter | [`adapters/cellnavi/encoder.py`](../../adapters/cellnavi/encoder.py) |
| 上游 encoder | [`third_party/CellNavi/cellnavi/model/pretrain_model.py`](../../third_party/CellNavi/cellnavi/model/pretrain_model.py) |
| 输入构造（参考） | [`third_party/CellNavi/cellnavi/data_provider/data_utils.py`](../../third_party/CellNavi/cellnavi/data_provider/data_utils.py) |
| Smoke | [`smoke/test_cellnavi.py`](../../smoke/test_cellnavi.py) |

## 数据流与 log1p

1. 从 `adata.X` 取向量；若 `input_is_log1p=True`（默认），`counts = max(expm1(x), 0)`，再 `round → int64` 作为 raw count 代理。
2. 若 `normalize=True`（默认），对每个细胞的选中基因计算  
   `expression = log1p(raw / sum(raw) * 10000)`，与上游 `prepare_cell_input(..., normalize=True)` 一致。
3. `rawcount` 张量与 `expression` 同源（protected 零表达基因在 coverage 模式下用 **伪计数 1** 注入，以便进入子图）。

**真 raw counts**：设 `input_is_log1p=False`，此时跳过 `expm1`，直接 `max(x,0)` 后四舍五入。

## Protected-gene coverage（是否需要？）

- **无 top-k token 截断**；子图节点来自「在 vocab 中且（表达 > 0 或 protected）」的基因集合。
- 上游 `prepare_cell_input` 会 **丢弃表达为 0 的基因**，因此若扰动基因在某细胞中为 0，默认不会出现在图中 → **与 benchmark「扰动基因须被编码」冲突**。
- Adapter 在 `force_pert=True` 且存在 `obsm['pert_var_idx']` 时，将对应 `var` 列上的基因并入集合；若 rounded count 为 0，置为伪计数 1。  
  **`force_pert_effective`**: `force_pert and pert_var_idx_present`（与 scGPT 等一致，表示 coverage 策略启用）。
- 基因不在 CellNavi∩NicheNet 词表时无法加入（与全局约定一致：跳过）。

## 运行依赖（镜像外资产）

精简 git 镜像里的 `third_party/CellNavi` **可能缺少**大文件（尤其 `data/pretrain/pretrain_weights.pth`、`Nichenet/graph.pkl`）。**项目内推荐落盘位置**（与 adapter 默认探测顺序一致）：

| 资产 | 推荐路径 |
|------|----------|
| 预训练权重 | `<delivery_root>/pretrained/cellnavi/data/pretrain/pretrain_weights.pth` |
| 基因表 | `<delivery_root>/pretrained/cellnavi/data/gene_name.txt` |
| NicheNet 节点表 | `<delivery_root>/pretrained/cellnavi/data/Nichenet/node2idx.json` |
| NicheNet 整图（边 / 拓扑，`NicheNetGraph`） | `<delivery_root>/pretrained/cellnavi/data/Nichenet/graph.pkl` |

覆盖默认（可选，与显式 `encode(...)` 参数等价）：

- `LATENT_BENCH_CELLNAVI_CKPT`
- `LATENT_BENCH_CELLNAVI_GRAPH_PKL`
- `LATENT_BENCH_CELLNAVI_GENE_NAME`、`LATENT_BENCH_CELLNAVI_NODE2IDX`

## Smoke

若已将 ckpt 与 `graph.pkl` 按上表放入 `pretrained/cellnavi/data/`，**可不设** `LATENT_BENCH_CELLNAVI_*`（`test_cellnavi.py` 与 `encode` 会从该树自动探测）。否则显式 export：

```bash
cd <delivery_root>
export LATENT_BENCH_CELLNAVI_CKPT=<delivery_root>/pretrained/cellnavi/data/pretrain/pretrain_weights.pth
export LATENT_BENCH_CELLNAVI_GRAPH_PKL=<delivery_root>/pretrained/cellnavi/data/Nichenet/graph.pkl
# 可选：export LATENT_BENCH_SMOKE_H5AD=/tmp/adamson_smoke.h5ad
$SCFM_ENVS_ROOT/cellnavi/bin/python data/scFM/fm/smoke/test_cellnavi.py
```

缺少 ckpt / `graph.pkl` / h5ad 时脚本 **SKIP**（退出码 0），便于 CI 占位。

## 残余风险

- `expm1(log1p(normalize_total))` 仅为近似恢复计数，与真实 raw UMI 有偏差；可比性依赖所有细胞使用同一 `input_is_log1p` 口径。
- NicheNet 子图仅连接 vocab 内基因；极高表达但孤立节点依赖 CLS 全连接边，与上游设计一致。
