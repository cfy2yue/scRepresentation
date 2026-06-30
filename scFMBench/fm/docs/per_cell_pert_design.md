# Protected perturbation-gene coverage（latent_bench）

## 目标

- 编码器输入**只有单细胞表达谱 `adata.X`**，不再把 perturbation metadata 当作单独条件输入。
- 若某模型存在 gene sampling / truncation / subset packing，则每个细胞使用自己的扰动基因集合 \(P_i\) 作为**protected set**，保证这些基因不会在编码时被采样掉。
- 表达值始终来自细胞**真实**表达（含 0）；扰动基因不会获得单独的“条件 token / 条件向量”语义。
- `vocab`/词表缺失的基因：跳过（不强制）。

## 公共约定

| 组件 | 路径 |
|------|------|
| `pert_var_idx` 构建 | [`utils/data/pert_var_idx.py`](../../../../utils/data/pert_var_idx.py) |
| 共享辅助 | [`adapters/_common.py`](../adapters/_common.py) |
| Adapter 根 | [`adapters/`](../adapters/) |

## 各模型策略摘要

| 模型 | 策略 |
|------|------|
| **UCE** | `build_cell_sentence(..., protected_genes per cell)`；protected genes 只保证进入 top-K，不作为单独条件流 |
| **State** | `VCIDatasetSentenceCollator` runtime patch 用 per-cell protected set 覆盖采样，不再从 `obs` 推断条件 |
| **scGPT** | 纯表达序列；`obsm['pert_var_idx']` 仅作为 truncation/sampling 的 protected set |
| **Geneformer** | 纯表达 rank token 序列；`obsm['pert_var_idx']` 仅保证 protected genes 保留在 token 集内 |
| **stack** | 全基因线性归约，无 token 采样；`X` 已含扰动 |
| **scldm** | expressed-subset + full-vector VAE；`obsm['pert_var_idx']` 仅用于 subset packing 的 protected coverage |
| **xVERSE** | 全基因 attention；`obsm['pert_var_idx']` 仅作可观测性/覆盖校验 |
| **CellNavi** | 稀疏子图节点来自「表达 >0 ∩ vocab」；`obsm['pert_var_idx']` 在 `force_pert=True` 时把 protected 基因并入节点集（零表达用伪计数），不是单独条件输入 |
| **scFoundation** | 官方 ``gatherData`` 默认 ``value_labels = x > 0``，零表达基因不进 encoder；`obsm['pert_var_idx']` 在 `force_pert=True` 时对 mask 做 OR，仅覆盖基因选择（表达值仍为 0），不是扰动条件输入 |

## Manifest 元数据

导出时在 manifest 中记录：

- `force_pert`: bool — **调用方意图**：当为 true 且 `obsm['pert_var_idx']` 存在时，凡支持 coverage 的 adapter 应把这些基因当作 protected set（见上）；**不**表示向模型额外输入了扰动条件流。
- `pert_kept_per_cell`: `{ "0", "1", "2", "3+" }` 计数（histogram），描述每个细胞在 `pert_var_idx` 里有多少个索引被视为有效扰动位点（与词表映射无关的计数口径，便于跨 run 对齐）。
- `force_pert_effective`: bool — **本 run 是否实际执行了 encoding-time protected coverage**（例如 token 采样/截断/expressed-subset 打包时强制保留上述基因）。全基因输入、无采样路径的模型通常为 false（扰动已体现在 `X` 中，无需再靠 coverage 约束）。

见 [`benchmark/manifest.schema.json`](../benchmark/manifest.schema.json)。

### `force_pert` 与 `force_pert_effective` 对照

| 场景 | `force_pert` | `pert_var_idx` | `force_pert_effective`（典型） |
|------|--------------|----------------|-------------------------------|
| 希望启用 coverage 且矩阵存在 | true | 有 | true（scGPT / UCE / State / Geneformer / scldm subset 等） |
| 希望启用但无矩阵 | true | 无 | false（无 protected 源） |
| 关闭 coverage 请求 | false | 任意 | false |
| 全基因、无采样（stack / xVERSE） | 任意 | 可有 | **false**（仅记录 histogram 等元数据时仍可有 `pert_kept_per_cell`）；xVERSE 若对对齐矩阵做了 protected 槽位修补，另见 `meta['pert_var_idx_slot_repair']`（是否启用修补分支）与 `meta['pert_var_idx_slot_repair_count']`（实际修补槽位数），**不**改变本条对 `force_pert_effective` 的定义 |
| 稀疏子图、丢零表达基因（CellNavi） | true | 有 | **true**（protected 强制入图）；`false` / 无矩阵时仅非零表达基因 |
| ``gatherData``、按非零打包（scFoundation） | true | 有 | **true**（mask OR protected 列）；`false` / 无矩阵时与上游一致仅 >0 |

说明：个别模型在序列里会把 protected 基因排在靠前 token（如 scGPT），这是 **同一表达输入下的顺序/截断策略**，不是第二条“条件分支”输入。
