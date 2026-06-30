# 环境与分模型 Runbook（合并版）

本文件合并原 `env_setup_run.md`、`env_setup_scldm.md`、`env_setup_stack.md`、`env_setup_geneformer.md`、`env_setup_xverse.md`，作为**单一运维入口**。总则仍以根目录 [`env_setup.md`](../env_setup.md)、[`env_map.md`](../env_map.md) 为准。

---

## 通用：cache / 临时目录（不写 `/home`）

```bash
export UV_CACHE_DIR=$SCFM_CACHE_ROOT/uv
export PIP_CACHE_DIR=$SCFM_CACHE_ROOT/pip
export HF_HOME=$SCFM_CACHE_ROOT/huggingface
export HUGGINGFACE_HUB_CACHE=$SCFM_CACHE_ROOT/huggingface/hub
export TORCH_HOME=$SCFM_CACHE_ROOT/torch
export TMPDIR=$SCFM_CACHE_ROOT/tmp
```

---

<a id="runbook-scgpt"></a>

## 1. scGPT 装入 `scdfm`（运行记录）

### 1.1 时间 & 环境

- python: `$SCFM_ENVS_ROOT/scdfm/bin/python` — Python 3.11.13
- torch（全程未改版本）: **`2.7.1+cu126`**
- uv: `uv 0.9.26`（`$SCFM_ENVS_ROOT/../miniconda/bin/uv`）

### 1.2 关键事实（实际改动）

- **numpy**：由 conda 锁定的 `2.4.3` 恢复为 **`numpy==2.2.6`**（与 `environment.yml` / Numba 要求一致，消除 `numba` 与 NumPy 2.4 冲突）。
- **torchtext**：原 **`torchtext==0.18.0`**（与 torch 2.7.1 ABI 不兼容）已卸载；改为 site-packages 下 **纯 Python shim**。`pip freeze` 中不再出现 `torchtext==` 行属预期。
- **新增 pip 可感知包**：`cell-gears==0.0.1`，`scgpt==0.2.4`，`flash_attn`（预编译 wheel），`ninja==1.13.0`。
- **torch**：全程保持 **`2.7.1+cu126`**。
- **import 验收**：`torch`、`gears`、`scgpt`、`torchtext`（shim）、`flash_attn`、`flash_attn.flash_attention.FlashMHA`、scgpt `GeneVocab` 均通过。

### 1.3 shim 文件位置（`SP` = scdfm site-packages）

- `SP` = `$SCFM_ENVS_ROOT/scdfm/lib/python3.11/site-packages`
- torchtext shim：`SP/torchtext/__init__.py`、`SP/torchtext/vocab.py`
- FlashMHA 兼容：`SP/flash_attn/flash_attention.py`（`from flash_attn.modules.mha import MHA as FlashMHA`）

### 1.4 各 step 摘要

| Step | 内容 |
|------|------|
| S1 | `uv pip install --python .../scdfm/bin/python --no-deps "cell-gears==0.0.1"` |
| S1.5 | `numpy==2.2.6` 恢复 |
| S2 | `uv pip install --python ... --no-deps "scgpt==0.2.4"` |
| S3 | flash-attn wheel：`flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl` |
| S4 | 写 `flash_attention.py` FlashMHA shim |
| S6 | `python -m pip uninstall -y torchtext` 后装 shim（`uv pip uninstall` 可能挂起） |

### 1.5 与本任务无关的 scdfm 预存问题（仅记录）

`import scvi` 可能报 `SparseDataset` 与 anndata 0.12 不兼容；与 scgpt 链路无关。修复需改 scdfm 核心版本，超出 scgpt 安装边界。

---

<a id="runbook-scldm"></a>

## 2. scLDM（`scldm` uv venv）

- **Venv**：`$SCFM_ENVS_ROOT/scldm`
- **创建**：`$SCFM_ENVS_ROOT/../miniconda/bin/uv venv --python 3.11 $SCFM_ENVS_ROOT/scldm`
- **安装**：`cd third_party/scldm` → `uv pip install --python .../scldm/bin/python -e .`
- **必须 pin**：`uv pip install --python ... "scvi-tools==1.2.0"`（否则 scvi 1.4.x 与 `anndata<=0.10.9` 冲突）。
- **关键版本**：torch 2.7.0+cu126、lightning 2.4.0、scvi-tools 1.2.0、anndata 0.10.9、numpy 1.26.4。
- **未装**：`cellarium-ml`（需 git）。
- **磁盘**：约 6.7G。

---

<a id="runbook-stack"></a>

## 3. arc-stack（`stack` uv venv）

- **Venv**：`$SCFM_ENVS_ROOT/stack`
- **源码**：`third_party/stack`（包名 `arc-stack`）
- **安装**：`uv pip install --python .../stack/bin/python -e .`
- **关键版本**：torch 2.11.0+cu130、pytorch-lightning 2.6.1、scvi-tools 1.4.2、anndata 0.12.10。
- **本地预训练**：`<delivery_root>/pretrained/stack/bc_large.ckpt`（`load_model_from_checkpoint` 已验证）。
- **磁盘**：约 5.4G。

---

<a id="runbook-geneformer"></a>

## 4. Geneformer（`geneformer` uv venv）

- **Venv**：`$SCFM_ENVS_ROOT/geneformer`
- **安装**：`cd third_party/Geneformer` → `uv pip install -r requirements.txt` + `uv pip install -e .`
- **关键版本**：torch 2.10.0+cu128、transformers 4.46.0、bitsandbytes 0.49.2。
- **本地权重（事实路径）**：`<delivery_root>/pretrained/geneformer/Geneformer-V2-316M/`（含 `model.safetensors`；**不在** `data/scFM/` 下）。
- **磁盘**：约 8.1G。

---

<a id="runbook-xverse"></a>

## 5. xVERSE（复用 `scdfm`）

- **无需**单独 venv；设置 `PYTHONPATH` 指向 `third_party/xVERSE_code`。
- **本地权重**：`<delivery_root>/pretrained/xVerse/xVERSE_384.pth`（注意目录名 `xVerse` 大小写）。
- **入口**：`main/utils_model.py`（`XVerseModel`、`CellEmbeddingbyGene`）。

```bash
export PYTHONPATH=<delivery_root>/scFM/fm/third_party/xVERSE_code:${PYTHONPATH:-}
$SCFM_ENVS_ROOT/scdfm/bin/python -c "import main.utils_model as um; print(hasattr(um,'XVerseModel'))"
```
