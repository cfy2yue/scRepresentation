# latent encoder 候选汇总

这份文档合并了此前的 `latent_encoder_1.md` 到 `latent_encoder_5.md`，只保留当前项目真正需要的信息：

1. **哪些模型目前被认为有潜力作为 pretrained encoder**
2. **它们更适合哪种角色**：主候选、补充候选、结构参考
3. **是否需要额外下载预训练参数 / checkpoint**
4. **与当前 `CoupledFM` 路线的兼容性**

本文默认面向当前目标：为后续 **latent-space conditional flow matching（cFM）** 寻找可复用的 encoder / decoder-capable backbone。

**环境与本地权重是否就绪**：见 [`pretrained_index.md`](pretrained_index.md) 与 [`encoder_impl/README.md`](encoder_impl/README.md)（按模型逐项）。

---

## 一句话结论

如果只从“**现在就值得继续推进为 pretrained encoder 候选**”这个角度看，当前最值得保留在主候选池里的模型是：

1. **`xVERSE_code`**
2. **`scldm`**
3. **`stack`**

其中：

- **最适合先接现有 `CoupledFM` 向量 latent 接口**：`xVERSE_code`
- **最强 decoder-first / latent-first 候选**：`scldm`（通用 scLDM + 本地 `pretrained/scdlm` 权重）
- **最适合作为 trunk-first 对照线**：`stack`

当前**不建议优先投入实现适配**的有：

- `CellFM`（MindSpore 栈，迁移成本高）
- `X-Cell`（当前仓库仍是占位 API）

---

## 当前接口约束

当前 [`<delivery_root>/coupled/models/velocity_field.py`](<delivery_root>/coupled/models/velocity_field.py) 更偏向以下假设：

```python
z_t: Tensor  # [B, d_latent]
```

也就是：

- 每个样本一个固定维向量 latent
- 通过 `d_latent` 注入速度场
- 更适合直接消费 **`[B, D]` 连续向量**

因此：

- **向量 latent** 候选最容易先接入
- **token / set latent** 候选不是不能用，但需要 pooling / flatten / projection

---

## 主候选池

### 1. `xVERSE_code`

**定位**  
转录组 foundation model，既能输出 `z_bio` embedding，也能通过 `mu_bio` 做生成。

**为什么有潜力**
- 明确的 per-cell 连续向量 latent
- 有生成路径，不只是黑盒 embedding
- 和当前 `CoupledFM` 的 `aux_emb -> d_latent` 形式最接近

**当前判断**
- 适合作为**第一优先 adapter 候选**
- 很适合回答“强 per-cell pretrained vector encoder 是否能提升 latent-cFM”

**latent 形态**
- `z_bio: [B, 384]`

**额外预训练参数**
- **需要**
- 需要下载官方权重：`xVERSE_384.pth`

**当前建议**
- 先做 `XVerseAdapter`
- 最小策略是 `384 -> d_latent` 的线性投影，再接现有 `velocity_field`

---

### 2. `scldm`

**定位**  
更通用的 Transformer-VAE + latent diffusion / flow matching 主干。

**为什么有潜力**
- 理论上最符合“可重构 latent + latent FM”
- 结构上是最像长期主 backbone 的一类

**当前判断**
- 是长期最重要参考对象之一；benchmark 以通用 `scldm` + 本地 `pretrained/scdlm`（如 `vae_census`）为准

**latent 形态**
- `z: [B, M, D]` 的 set/token latent

**额外预训练参数**
- **需要**
- 需要通过 `scldm-download-artifacts` 或对应 checkpoint / artifacts 下载（与 [`pretrained_index.md`](pretrained_index.md) 一致）

**当前建议**
- 先做 `ScldmAdapter`：token latent 需 `pooling / flatten + MLP` 接到向量型 `d_latent`
- 若后续把 `CoupledFM` 升级成 token-latent cFM，`scldm` 仍是主参考对象

---

### 3. `stack`

**定位**  
tabular attention 基础模型，能导出 cell embedding，也有 NB 解码头。

**为什么有潜力**
- PyTorch 实现
- backbone 清楚
- 比较适合作为 trunk-first 强基线

**当前判断**
- 适合作为**主候选池中的对照线**
- 不是原生 VAE latent，但可作为强 encoder trunk

**latent / embedding 形态**
- `cell_embeddings: [B, n_hidden * token_dim]`
- 常见配置量级如 `1600`

**额外预训练参数**
- **需要**
- 需要官方 checkpoint；通常还需要与 checkpoint 配套的 `genelist`

**当前建议**
- 作为 `StackAdapter` 对照线
- 如果 decoder-first 路线太重，可以先用它建立 baseline

---

## 暂缓候选

### `CellFM`

**结论**
- 模型思路本身有价值
- 但核心工程栈是 **MindSpore**
- 对当前 `CoupledFM` 短期不友好

**额外预训练参数**
- 即使有权重，也不建议当前优先投入迁移成本

---

### `X-Cell`

**结论**
- README 很强
- 但当前本地仓库仍主要是占位 API
- 暂时还不能进入实现适配阶段

**额外预训练参数**
- 当前不是“要不要下载权重”的问题，而是**实现还不完整**

---

## 是否需要额外下载预训练参数

### 需要下载的

| 模型 | 是否需要额外权重 | 备注 |
|------|------------------|------|
| `xVERSE_code` | **是** | 需要 `xVERSE_384.pth` |
| `scldm` | **是** | 需要 `scldm-download-artifacts` 或对应 checkpoint / 本地 `pretrained/scdlm` |
| `stack` | **是** | 需要官方 checkpoint，通常还要配套 `genelist` |
| `CellFM` | 视情况 | 即使有权重也不建议当前优先处理 |

### 当前没有稳定公开 pretrained checkpoint 或实现不完整的

| 模型 | 情况 |
|------|------|
| `X-Cell` | 仓库实现不完整，暂不进入参数下载阶段 |

---

## 建议的后续优先级

### 若目标是“尽快接到现有 CoupledFM”

1. `xVERSE_code`
2. `stack`

原因：二者都更容易收敛到当前的向量型 `d_latent` 接口（`scldm` 需先做 pooling / 投影）。

### 若目标是“优先保留 decoder-first / latent-first 正统路线”

1. `scldm`
2. `xVERSE_code`（亦有生成路径）

原因：更接近“先学 latent，再做 latent-space flow”的主线。

### 当前最推荐的实际组合

如果只保留最值得推进的几条：

- **第一梯队**
  - `xVERSE_code`
  - `scldm`

- **第二梯队**
  - `stack`

---

## 建议的下一步

后续如果继续推进实现，不建议再从“看文档”开始，而是直接进入：

1. 定义统一 `LatentEncoderAdapter`
2. 先做：
   - `XVerseAdapter`
   - `ScldmAdapter`
3. 再做：
   - `StackAdapter`
4. 再决定是否扩展到 token-latent cFM

---

## 本地状态说明

当前本地已完成的是：

- 第三方源码仓库已克隆到 `fm/third_party/`
- 已完成 README 级与源码级扫描

当前**还没有统一下载所有预训练权重**。如后续开始适配，建议按以下最小顺序下载：

1. `xVERSE_384.pth`
2. `scldm` 对应 artifacts / `pretrained/scdlm`
3. 视需要再补 `stack` checkpoint

---

## Latent 表征 benchmark

与 **CoupledFM 训练链解耦** 的表征 benchmark 入口：

- [`../../benchmark/README.md`](../../benchmark/README.md) — 指标与命令
- [`../../benchmark/docs/metrics_protocol.md`](../../benchmark/docs/metrics_protocol.md) — 指标契约
- [`encoder_impl/README.md`](encoder_impl/README.md) — 各模型实现说明
- [`pretrained_index.md`](pretrained_index.md) — 权重路径

---

## 文档说明

本文件已替代此前拆开的：

- `latent_encoder_1.md`
- `latent_encoder_2.md`
- `latent_encoder_3.md`
- `latent_encoder_4_impl_scan.md`
- `latent_encoder_5_interface_contracts.md`

后续统一维护本文件即可。
