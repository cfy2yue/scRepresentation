# scdfm 环境包（恢复指南）

本目录包含 `coupledFM` 主依赖（`scdfm` conda env）的两套交付物：

| 文件 | 大小 | 用途 |
|------|------|------|
| `scdfm.tar.gz` | ~3.8 GB | `conda-pack` 离线打包，**最快恢复**，无需联网 |
| `environment.yml` | 10 KB | conda 全量声明（`--no-builds`，跨平台/跨架构友好） |
| `requirements.txt` | 9 KB | `pip freeze` 输出，纯 pip 依赖参考 |

> 来源：source env `scdfm`（约 8.5 GB 已安装），导出于 `2026-04-17`，`conda-pack 0.0.0`、`conda 23.x` 系列。

---

## 方案 A：`conda-pack` 极速恢复（推荐，仅 Linux x86_64）

适用于「目标机器 OS 与源机器一致（Linux x86_64 + glibc 2.28+）」。

```bash
# 任选一个 conda 安装目录
DST=/data2/<your-name>/miniconda/envs/scdfm
mkdir -p "$DST"
tar -xzf scdfm.tar.gz -C "$DST"

# 必须执行：把硬编码的 prefix 替换成新的实际路径
"$DST"/bin/conda-unpack

# 激活
source "$DST"/bin/activate
python -c "import torch, anndata, omegaconf; print(torch.__version__, anndata.__version__)"
```

注意事项：
- conda-pack 不会重新解析依赖，所有 `.so` / `.pyc` 都是 freezed 的。
- 不要把目标解压目录放到包含空格或非 ASCII 的路径下（部分包的 RPATH 会出问题）。
- 如果目标机器 CUDA 驱动比源机器旧，PyTorch 可能 `runtime error: CUDA driver too old` —— 这种情况请用方案 B。

## 方案 B：`environment.yml` 重建（推荐，跨架构 / 离线源不可用时）

```bash
conda env create -n scdfm -f environment.yml
conda activate scdfm
# 如果 yml 里 pip 段失败，可以补一遍：
pip install -r requirements.txt
```

`environment.yml` 是 `conda env export -n scdfm --no-builds` 的产物，因此：
- 只锁定了 `name=version`，**不锁 build**，便于跨平台。
- pip 部分被嵌在 `dependencies.pip` 段，会在 conda 装完后由 conda 调用 pip 安装。

## 方案 C：`pip` 纯净重建（仅当你已有自管 Python 3.11 + CUDA 12.x 时）

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
这种方案不会拿到 `mamba`、`hdf5` 等 conda-only 二进制依赖，仅作为最后手段或 CI 用。

---

## 包含的关键依赖（亮点）

- `python=3.11`
- `pytorch=2.x` + `cudatoolkit=12.x` + `torchvision`
- `anndata`、`scanpy`、`scvi-tools`、`scib-metrics`
- `pyg`（torch-geometric）、`torch-scatter`、`torch-sparse`
- `pot`（Optimal Transport）
- `omegaconf`、`hydra-core`、`lightning`
- `state`（local editable，**已通过 `--ignore-editable-packages` 排除**，需要时按下文「editable 重装」补上）
- `uv`（用于 state subproject 自身的 lock 管理）

### Editable 包重装（State / UCE / coupledFM 自身）

`scdfm.tar.gz` 因 `--ignore-editable-packages` 跳过了所有 `pip install -e .` 安装的本地源码包。恢复后请执行：

```bash
cd /path/to/CoupledFM
pip install -e .                        # coupledFM utils / coupled / latent / raw
pip install -e data/scFM/state          # State backbone（含 src/state/）
# UCE 没有 setup.py；直接 sys.path 注入即可，无需 pip install
```

---

## 校验

```bash
python - <<'PY'
import importlib, sys
mods = ['torch','anndata','scanpy','ot','omegaconf','lightning']
for m in mods:
    try:
        v = importlib.import_module(m).__version__
        print(f'{m:20s} {v}')
    except Exception as e:
        print(f'{m:20s} FAIL  {e}')
import torch
print('CUDA available:', torch.cuda.is_available(), 'device cnt:', torch.cuda.device_count())
PY
```

预期输出：所有模块均有版本号，`CUDA available: True`。

---

## 故障排查

| 症状 | 处理 |
|------|------|
| `conda-unpack` 报 `Could not find conda-meta` | 解压时 `--strip-components` 不要传，按默认即可 |
| `ImportError: libcudart.so.12: cannot open` | 目标机器 CUDA driver < 525；改用方案 B 或升级驱动 |
| `pip` 阶段失败：`No matching distribution for state==X` | state 是 editable 本地包，请按上方「Editable 重装」执行 |
| 体积不够：tarball 4G 但目录解开后 8.5G | 正常，`conda-pack` 是 gz 压缩；验证 `du -sh $DST` ≈ 8.5G 即可 |
