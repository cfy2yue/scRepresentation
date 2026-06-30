"""
Unified training for CoupledFM: raw expression flow matching with 3 modes.

Mode 1 (baseline): random control-GT pairing, no latent guidance
Mode 2 (ot):       OT pairing in latent space, no latent guidance
Mode 3 (coupled):  OT pairing + latent z_t → dual-path guidance
                   (adaLN conditioning + gated CLS injection)

Latent z_t modes (coupled only):
  - interp: z_t = (1-t)*z_ctrl + t*z_gt  (default, fast)
  - ode:    z_t = FrozenLatentFM.ode_at_t(z_ctrl, t)  (precise, GPU)

Evaluation schedule:
  - explicit_pert_split: val and epoch test share the **same** test conditions
  - Every val_every_steps:    fast val on full test set (default 500 steps)
  - Every test_every_epoch:   full test for early stopping (default each epoch)
  - Early stopping:           patience × test_every_epoch epochs without
                              improvement → stop

Learning rate:
  - Cosine decay with linear warmup
  - Peak 5e-5, warmup 2 epochs (suitable for fine-tuning pretrained CellNavi)

Supports:
  - Mixed precision (AMP) via --use_amp / --amp_dtype
  - Multi-GPU DDP via torchrun (auto-detected from LOCAL_RANK env var)

Loss: MSE on predicted velocity v_θ(x_t, t, x_ctrl; z_t) vs target dx;
optional MMD² regularization (latent FM style) after epoch 3 + γ warmup.
"""

import contextlib
import gc
import hashlib
import json
import math
import os
import random
import resource
import sys
import subprocess
import time
import warnings
from typing import Optional
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 仅提醒一次：TrainConfig.min_lr 已弃用，实际 floor 为 base_lr * min_lr_ratio
_MIN_LR_DEP_WARNED = False

from model.config import Config
from model.data.vocab import GeneVocab
from model.data.dataset import CoupledFMDataset
from model.utils.train.ema import ModelEMA
from model.utils.train.schedulers import (
    get_ode_prob_curriculum,
    lr_warmup_cosine_ratio_floor_absolute,
)
from model.utils.train.loss_weights import compute_loss_weight
# NOTE: canonical split 已迁至 utils.data.split；data.pert_split 仅做兼容 re-export。
# 保留此 import 仅用于任何仍依赖 old 名字的下游调用点。
from model.data.pert_split import (  # noqa: F401
    build_explicit_pert_split,
    load_explicit_json,
)
from model.models.velocity_field import RawExprVelocityField, load_velocity_pretrained_bundle
from model.evaluate import (
    evaluate,
    print_eval_results,
    _collect_eval_tasks,
    build_monitor_val_tasks,
    _safe_train_mb,
)
from model.mmd_utils import median_sigmas, mmd2_unbiased
from model.pert_batch_utils import (
    latent_fm_wants_perturbation,
    null_perturbation_batch,
    perturbation_batch_to_device,
    slice_perturbation_batch,
    unpack_training_batch,
)


def _gamma_schedule_mmd(step: int, tc) -> float:
    """Exponential γ ramp after warmup (same shape as FM/latent/train.py)."""
    if step < tc.mmd_warmup_start:
        return 0.0
    if step >= tc.mmd_warmup_end:
        return tc.mmd_gamma_max
    progress = (step - tc.mmd_warmup_start) / (
        tc.mmd_warmup_end - tc.mmd_warmup_start
    )
    return tc.mmd_gamma_max * (1.0 - math.exp(-5.0 * progress))


# ── helpers ──────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _elapsed_str(seconds: float) -> str:
    """Human-readable elapsed time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h{m:02d}m"


def _eta_str(seconds: float) -> str:
    if seconds <= 0 or not math.isfinite(seconds):
        return "?"
    return _elapsed_str(seconds)


def _gpu_mem_mb() -> str:
    if not torch.cuda.is_available():
        return "N/A"
    alloc = torch.cuda.memory_allocated() / 1024**2
    resrv = torch.cuda.memory_reserved() / 1024**2
    return f"{alloc:.0f}/{resrv:.0f}MB"


def _rss_gb() -> str:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return f"{rss / 1024**2:.1f}GB"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_has_key(split: dict, key: str) -> bool:
    return any(bool((sp or {}).get(key)) for sp in split.values())


def _validate_split_disjoint(split: dict, keys: tuple[str, ...]) -> None:
    for ds_name, sp in split.items():
        seen: dict[str, str] = {}
        for key in keys:
            for cond in (sp or {}).get(key, []):
                s = str(cond)
                if s in seen:
                    raise ValueError(
                        f"Split overlap in dataset {ds_name!r}: condition {s!r} "
                        f"appears in both {seen[s]!r} and {key!r}."
                    )
                seen[s] = key


def _set_lr(optimizer, lr):
    """Uniformly set all param groups to the same LR (legacy single-group path)."""
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def _set_group_lrs(optimizer, lrs):
    """Set each group's LR independently. ``lrs`` may be a scalar or a list."""
    if not isinstance(lrs, (list, tuple)):
        lrs = [lrs] * len(optimizer.param_groups)
    for pg, lr in zip(optimizer.param_groups, lrs):
        pg["lr"] = lr


# ── Backbone / new-module parameter groups ───────────────────────
# 识别 CellNavi backbone 预训层：embed_gene.* / encoder.layers.N.attn.* / encoder.layers.N.ffn.*
# 其余（t_embed / ctrl_proj / value_encoder / latent_proj / cls_gate / z_gene_proj /
#      adaln_mod / gene_adaln / out_proj / cond_fuse …）均视为"新模块"。
_BACKBONE_SUBSTR = (
    "embed_gene",
    ".attn.",
    ".ffn.",
)


def _is_backbone_name(name: str) -> bool:
    return any(s in name for s in _BACKBONE_SUBSTR)


def _split_params(model: nn.Module, include_frozen: bool = False):
    """Return (backbone_params, new_params) with names preserved for logging.

    When ``include_frozen`` is True, also include parameters whose
    ``requires_grad`` is currently False. This is used in two-stage FT so that
    currently-frozen backbone params still end up in the optimizer and can be
    trained after stage2 unfreezes them.
    """
    raw = model.module if hasattr(model, "module") else model
    bb, nw = [], []
    bb_names, nw_names = [], []
    for n, p in raw.named_parameters():
        if not p.requires_grad and not include_frozen:
            continue
        if "lora_A" in n or "lora_B" in n:
            nw.append(p)
            nw_names.append(n)
            continue
        if _is_backbone_name(n):
            bb.append(p)
            bb_names.append(n)
        else:
            nw.append(p)
            nw_names.append(n)
    return bb, nw, bb_names, nw_names


def _stage1_freeze_backbone(model: nn.Module, verbose: bool = False) -> int:
    raw = model.module if hasattr(model, "module") else model
    n_frozen = 0
    for n, p in raw.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            continue
        if _is_backbone_name(n):
            if p.requires_grad:
                p.requires_grad = False
                n_frozen += 1
    if verbose:
        print(f"[stage1] froze backbone: {n_frozen} params", flush=True)
    return n_frozen


def _stage2_unfreeze_backbone(model: nn.Module, verbose: bool = False) -> int:
    raw = model.module if hasattr(model, "module") else model
    n_unfrozen = 0
    for n, p in raw.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            continue
        if _is_backbone_name(n) and not p.requires_grad:
            p.requires_grad = True
            n_unfrozen += 1
    if verbose:
        print(f"[stage2] unfroze backbone: {n_unfrozen} params", flush=True)
    return n_unfrozen


def _is_main(rank):
    return rank == 0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_repo_root()), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode().strip()
    except Exception:
        return ""


def _model_state_keys_digest(model: nn.Module) -> str:
    keys = "|".join(sorted(model.state_dict().keys()))
    return hashlib.sha256(keys.encode("utf-8")).hexdigest()[:24]


def _iter_with_last_flag(iterable):
    """Yield (item, is_last). Empty iterator yields nothing."""
    it = iter(iterable)
    try:
        prev = next(it)
    except StopIteration:
        return
    for item in it:
        yield prev, False
        prev = item
    yield prev, True


def _amp_dtype(name: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _save(model, optimizer, epoch, global_step, best_test_loss, path,
          best_pd_ctrl=None, best_corr_pert=None, best_selection_score=None,
          no_improve_count=0,
          optimizer_step=None, accum_batch_idx=None, ema=None,
          selection_metric=None, scored_with_ema=None, ema_update_after=None,
          latent_fm_ckpt_path: str = "",
          model_keys_digest: str = "",
          git_commit: str = "",
          torch_version: str = ""):
    d = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_test_loss": best_test_loss,
        "no_improve_count": no_improve_count,
    }
    if best_pd_ctrl is not None:
        d["best_pd_ctrl"] = best_pd_ctrl
    if best_corr_pert is not None:
        d["best_corr_pert"] = best_corr_pert
    if best_selection_score is not None:
        d["best_selection_score"] = best_selection_score
    if optimizer_step is not None:
        d["optimizer_step"] = optimizer_step
    if accum_batch_idx is not None:
        d["accum_batch_idx"] = accum_batch_idx
    if ema is not None:
        d["ema"] = ema.state_dict()
    if selection_metric is not None:
        d["selection_metric"] = selection_metric
    if scored_with_ema is not None:
        d["scored_with_ema"] = bool(scored_with_ema)
    if ema_update_after is not None:
        d["ema_update_after"] = int(ema_update_after)
    if latent_fm_ckpt_path:
        d["latent_fm_ckpt"] = str(latent_fm_ckpt_path)
    if model_keys_digest:
        d["model_keys_digest"] = model_keys_digest
    if git_commit:
        d["git_commit"] = git_commit
    if torch_version:
        d["torch_version"] = torch_version
    torch.save(d, path)
    print(f"  [{_now()}] saved → {path}", flush=True)


def _run_eval(model, test_ds, device, tc, latent_fm, use_amp, amp_dt,
              rank, world_size, tasks=None, tag="eval", step=0,
              per_dataset=True, max_cells=0, ode_steps=None,
              ctrl_means=None, pert_means=None,
              cfg_w: float = 1.0,
              use_residual_flow: bool = False,
              max_pert_genes: int = 16):
    """Shared helper for val / full-test evaluation."""
    t_eval_start = time.time()
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    if ode_steps is None:
        ode_steps = tc.eval_ode_steps
    n_tasks = len(tasks) if tasks is not None else "all"
    mc_str = f", max_cells={max_cells}" if max_cells > 0 else ""
    if _is_main(rank):
        print(f"  [{_now()}] {tag} started ({n_tasks} conds, "
              f"ode_steps={ode_steps}{mc_str}) ...", flush=True)
    results = evaluate(
        raw_model, test_ds, device,
        n_ode_steps=ode_steps,
        coupling_mode=tc.coupling_mode,
        latent_fm=latent_fm,
        use_amp=use_amp, amp_dtype=amp_dt,
        rank=rank, world_size=world_size,
        tasks=tasks,
        max_cells=max_cells,
        ctrl_means=ctrl_means,
        pert_means=pert_means,
        cfg_w=cfg_w,
        use_residual_flow=use_residual_flow,
        ode_method=getattr(tc, "val_ode_method", "euler"),
        max_pert_genes=max_pert_genes,
    )
    torch.cuda.empty_cache()
    t_eval = time.time() - t_eval_start
    if _is_main(rank) and results is not None:
        print_eval_results(results, step, tag=tag, per_dataset=per_dataset)
        print(f"  [{_now()}] {tag} done in {_elapsed_str(t_eval)}", flush=True)
    return results


def _select_metric_value(results: dict, metric_name: str, mmd_lambda: float = 0.5) -> float:
    g = results["global"]

    def _scalar_metric(key: str, default: float) -> float:
        v = g.get(key, default)
        try:
            x = float(v)
        except (TypeError, ValueError):
            return float("-inf")
        if math.isnan(x):
            return float("-inf")
        return x

    if metric_name == "corr_pert_mean":
        return _scalar_metric("corr_pert_mean", float("nan"))
    if metric_name == "corr_minus_mmd":
        c = _scalar_metric("corr_pert_mean", float("nan"))
        m = _scalar_metric("mmd", 0.0)
        return c - mmd_lambda * m
    if metric_name == "pearson_delta_ctrl":
        return _scalar_metric("pearson_delta_ctrl", float("-inf"))
    if metric_name == "mmd":
        return -_scalar_metric("mmd", float("inf"))
    raise ValueError(f"Unknown selection_metric: {metric_name}")


def _project_mmd_to_visible_genes(
    x_hat: torch.Tensor,
    x_gt: torch.Tensor,
    gene_mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Drop masked genes before train-time MMD bandwidth and kernel evaluation.

    ``gene_mask`` is 1 for hidden genes and 0 for visible genes. For a
    deterministic gene-budget manifest the mask is identical across rows, so
    this becomes exactly the budget keep-index projection used by final eval.
    If stochastic row-wise masks are active, use genes visible in every row to
    avoid letting masked-out zeros affect the median heuristic.
    """
    if gene_mask is None:
        return x_hat, x_gt, int(x_hat.shape[1])
    if gene_mask.shape != x_hat.shape:
        raise ValueError(
            f"gene_mask shape {tuple(gene_mask.shape)} must match MMD tensors "
            f"{tuple(x_hat.shape)}"
        )
    keep = (gene_mask.detach() < 0.5).all(dim=0)
    keep_count = int(keep.sum().item())
    if keep_count <= 0:
        raise ValueError("train-time MMD has no commonly visible genes after mask projection")
    return x_hat[:, keep], x_gt[:, keep], keep_count


def _validate_model_train_config(tc, mc, handles: dict) -> None:
    """启动期校验：非法注意力 / 图组合在训练前失败，不静默开跑。"""
    gbm = str(getattr(mc, "graph_bias_mode", "none") or "none")
    if str(getattr(mc, "attn_backend", "sdpa")) == "sparse" and gbm == "sdpa_bias":
        raise ValueError(
            "attn_backend='sparse' 与 graph_bias_mode='sdpa_bias' 不兼容；"
            "请改为 graph_bias_mode='none' 或 'sparse'。"
        )
    if str(getattr(mc, "attn_backend", "sdpa")) == "sparse":
        ok = False
        for h in handles.values():
            ei = getattr(h, "edge_index", None)
            if ei is not None and int(getattr(ei, "numel", lambda: 0)() or 0) > 0:
                ok = True
                break
        if not ok:
            raise ValueError(
                "attn_backend='sparse' 需要数据集句柄上存在非空的 edge_index；"
                "请确认 model.use_graph=True 且 nichenet 图 pkl 可用。"
            )
    vode = str(getattr(tc, "val_ode_method", "euler") or "euler")
    if vode not in ("euler", "midpoint", "rk4"):
        raise ValueError(
            f"val_ode_method must be one of euler|midpoint|rk4, got {vode!r}"
        )


# ── training ─────────────────────────────────────────────────────

def train(cfg: Config, _amp_explicit: bool = False):
    global _MIN_LR_DEP_WARNED
    tc = cfg.train
    mc = cfg.model
    dc = cfg.data

    # linear 注意力在 fp16/bf16 下易 NaN；未显式指定精度时自动切双精度。
    # sdpa / flash 保持默认混合精度（use_amp=True）。
    if not _amp_explicit and mc.attn_backend == "linear":
        tc.use_amp = False
        tc.fp64_training = True

    train_start_wall = time.time()

    # ── DDP setup ────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    use_ddp = local_rank >= 0
    if local_rank <= 0:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")
        print(
            f"[{_now()}] train() entered "
            f"(local_rank={local_rank}, use_ddp={use_ddp}, CUDA_VISIBLE_DEVICES={cvd})",
            flush=True,
        )
        if not _MIN_LR_DEP_WARNED:
            warnings.warn(
                "TrainConfig.min_lr is deprecated and ignored; "
                "LR floor is base_lr * min_lr_ratio only.",
                UserWarning,
                stacklevel=1,
            )
            _MIN_LR_DEP_WARNED = True
    if use_ddp:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        if local_rank == 0:
            print(f"[{_now()}] DDP: calling init_process_group (nccl) ...", flush=True)
        try:
            dist.init_process_group(
                "nccl",
                timeout=timedelta(minutes=180),
                device_id=device,
            )
        except TypeError:
            dist.init_process_group("nccl", timeout=timedelta(minutes=180))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if local_rank == 0:
            print(
                f"[{_now()}] DDP: init_process_group ok "
                f"(rank={rank}, world_size={world_size})",
                flush=True,
            )
    else:
        rank = 0
        world_size = 1
        device = torch.device(tc.device if torch.cuda.is_available() else "cpu")

    # DDP：固定「全局」每步 OT 配对总数（与「每卡各跑 batch_size」二选一语义：设了则覆盖 batch_size）
    if use_ddp and tc.global_ot_batch is not None:
        gob = int(tc.global_ot_batch)
        if gob <= 0:
            raise ValueError("global_ot_batch must be positive")
        if gob % world_size != 0:
            raise ValueError(
                f"global_ot_batch={gob} 必须能被 world_size={world_size} 整除 "
                f"（例如 60 配 3 卡→每卡 20；60 配 6 卡→每卡 10；120 配 6 卡→每卡 20）"
            )
        tc.batch_size = gob // world_size
        if _is_main(rank):
            print(
                f"[{_now()}] DDP: global_ot_batch={gob} → per-rank batch_size={tc.batch_size} "
                f"(全局每步 OT 条数 = {tc.batch_size * world_size})",
                flush=True,
            )

    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)

    if tc.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        if _is_main(rank):
            print(f"[{_now()}] detect_anomaly=True (slow backward)", flush=True)

    run_name = tc.coupling_mode
    out_dir = Path(tc.output_dir) / run_name
    if _is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "config.json", "w") as f:
            import dataclasses
            json.dump(dataclasses.asdict(cfg), f, indent=2)

    # ── data ─────────────────────────────────────────────────────
    # Canonical split: biflow_dir/split_seed{seed}.json（latent / raw / coupled 统一入口）
    from model.utils.data.split import (
        canonical_split_path,
        build_canonical_split,
        load_split_json,
        save_split as save_canonical_split,
    )
    split_override_s = str(getattr(dc, "split_file", "") or "").strip()
    split_is_override = bool(split_override_s)
    if split_is_override:
        split_path = Path(split_override_s).expanduser()
        if not split_path.is_file():
            raise FileNotFoundError(f"Configured split_file does not exist: {split_path}")
    else:
        split_path = canonical_split_path(dc.biflow_dir, dc.split_seed)
    # 为了可追溯，run 目录再留一份副本
    run_split_copy = out_dir / (
        "split_override.json" if split_is_override else f"split_seed{dc.split_seed}.json"
    )

    if _is_main(rank):
        print(f"[{_now()}] loading GeneVocab ...", flush=True)
    vocab = GeneVocab(dc.gene_name_path, dc.nichenet_node2idx_path)
    ds_subset = dc.datasets if dc.datasets else None

    use_ds_pert = bool(
        getattr(mc, "use_pert_condition", False)
        or getattr(dc, "use_raw_cond", False)
    )
    if getattr(dc, "use_raw_cond", False) and not getattr(mc, "use_pert_condition", False):
        raise ValueError("data.use_raw_cond requires model.use_pert_condition=True")

    if not dc.explicit_pert_split:
        warnings.warn(
            "data.explicit_pert_split=False is deprecated: canonical split JSON is "
            "always biflow_dir/split_seed{seed}.json (latent_data_dir split is no longer used).",
            UserWarning,
            stacklevel=2,
        )

    # DDP：仅 rank0 扫 h5ad 建 split，避免多进程同时打开整套 biFlow 把磁盘/NFS 拖死。
    if use_ddp:
        dist.barrier()
    if _is_main(rank):
        if split_is_override:
            print(f"[{_now()}] loading explicit split override → {split_path}", flush=True)
            split = load_split_json(split_path)
        elif split_path.exists():
            print(f"[{_now()}] reusing canonical split → {split_path}", flush=True)
            split = load_split_json(split_path)
        else:
            print(
                f"[{_now()}] building canonical split (rank0 only) → {split_path}",
                flush=True,
            )
            split = build_canonical_split(
                dc.biflow_dir,
                vocab,
                seed=dc.split_seed,
                min_cells=dc.min_cells_per_cond,
                coupling_mode=tc.coupling_mode,
                dataset_names=ds_subset,
                ot_feature=tc.ot_feature,
                de_dir=tc.de_dir,
                latent_backbone=str(
                    getattr(dc, "latent_backbone", "state") or "state",
                ),
            )
            save_canonical_split(split_path, split)
            print(f"[{_now()}] canonical split saved → {split_path}", flush=True)
        save_canonical_split(run_split_copy, split)
        split_meta = {
            "source": "override" if split_is_override else "canonical",
            "source_path": str(split_path),
            "copied_path": str(run_split_copy),
            "sha256": _sha256_file(run_split_copy),
        }
        with open(out_dir / "split_provenance.json", "w") as f:
            json.dump(split_meta, f, indent=2)
    if use_ddp:
        dist.barrier()
        if not _is_main(rank):
            split = load_split_json(split_path)
        dist.barrier()
    val_key_cfg = str(getattr(tc, "val_split_key", "auto") or "auto").strip()
    test_key = str(getattr(tc, "test_split_key", "test") or "test").strip()
    protocol = str(getattr(tc, "selection_protocol", "metric") or "metric").strip().lower()
    protocol = protocol.replace("-", "_")
    if protocol not in {"metric", "fixed_steps_no_selection"}:
        raise ValueError(
            "train.selection_protocol must be 'metric' or "
            f"'fixed_steps_no_selection', got {protocol!r}"
        )
    if protocol == "fixed_steps_no_selection":
        tc.val_every_steps = 0
        tc.test_every_epoch = 0
        tc.early_stop_patience = 0
        tc.run_initial_val = False
    budget_mmd_mask_aware = bool(
        str(getattr(tc, "gene_budget_manifest_path", "") or "").strip()
        and bool(getattr(tc, "use_mmd", False))
    )
    if budget_mmd_mask_aware and _is_main(rank):
        print(
            f"[{_now()}] gene-budget MMD: enabled with mask-aware visible-gene projection",
            flush=True,
        )

    if val_key_cfg == "auto":
        val_key = "val" if _split_has_key(split, "val") else (
            "test" if not split_is_override else ""
        )
    else:
        val_key = val_key_cfg
    check_keys = tuple(k for k in ("train", val_key, test_key) if k)
    _validate_split_disjoint(split, check_keys)
    has_val_split = bool(val_key and _split_has_key(split, val_key))
    if (
        split_is_override
        and protocol == "metric"
        and not has_val_split
        and (
            int(getattr(tc, "val_every_steps", 0) or 0) > 0
            or int(getattr(tc, "test_every_epoch", 0) or 0) > 0
            or bool(getattr(tc, "run_initial_val", True))
        )
    ):
        raise ValueError(
            "Explicit split-file metric selection requires a non-empty val split "
            "or disabled training-time eval. Use --fixed-step-no-selection for "
            "fixed-step RawFM smokes."
        )

    t_load = time.time()
    print(f"[{_now()}] rank {rank}/{world_size} loading datasets ...", flush=True)
    train_ds = CoupledFMDataset(
        dc.biflow_dir, vocab, split,
        mode="train", coupling_mode=tc.coupling_mode,
        latent_z_mode=tc.latent_z_mode,
        batch_size=tc.batch_size, ds_alpha=tc.ds_alpha,
        ot_method=tc.ot_method, ot_threads=tc.ot_threads,
        ot_sinkhorn_reg=getattr(tc, "ot_sinkhorn_reg", 0.05),
        ot_sinkhorn_iter=getattr(tc, "ot_sinkhorn_iter", 50),
        ot_emb_cap_src=tc.ot_emb_cap_src,
        ot_emb_cap_gt=tc.ot_emb_cap_gt,
        ot_feature=tc.ot_feature,
        de_dir=tc.de_dir,
        time_sampling=tc.time_sampling,
        ot_cost=tc.ot_cost,
        ot_sample_mode=tc.ot_sample_mode,
        gene_mask_prob=getattr(tc, "gene_mask_prob", 0.0),
        gene_mask_all_prob=getattr(tc, "gene_mask_all_prob", 0.0),
        gene_budget_manifest_path=getattr(tc, "gene_budget_manifest_path", ""),
        gene_budget_label=getattr(tc, "gene_budget_label", ""),
        use_graph=getattr(mc, "use_graph", False),
        nichenet_graph_pkl=dc.nichenet_graph_pkl_path,
        pert_idx_mode=getattr(tc, "pert_idx_mode", "zero"),
        num_pert_ids=getattr(mc, "num_pert_ids", 10000),
        use_residual_flow=getattr(tc, "use_residual_flow", False),
        use_raw_pert_condition=use_ds_pert,
        max_pert_genes=getattr(mc, "max_pert_genes", 16),
        pert_gene_emb_cache_dir=(
            getattr(dc, "pert_gene_emb_cache_dir", "") if use_ds_pert else ""
        ),
        use_h5ad_pert_metadata=getattr(dc, "use_h5ad_pert_metadata", False),
        pert_metainfo_path=(
            str(getattr(dc, "pert_metainfo_path", "") or "") if use_ds_pert else ""
        ),
        chemical_metainfo_path=(
            str(getattr(dc, "chemical_metainfo_path", "") or "") if use_ds_pert else ""
        ),
        latent_backbone=str(getattr(dc, "latent_backbone", "state") or "state"),
        chem_emb_source_dir=str(getattr(dc, "chem_emb_source_dir", "") or ""),
        chem_obs_column=str(getattr(dc, "chem_obs_column", "") or ""),
        drug_emb_cache_dir=str(getattr(dc, "drug_emb_cache_dir", "") or ""),
        max_chem_keys=int(getattr(dc, "max_chem_keys", 4)),
        chem_fallback_embed_dim=max(
            8, int(getattr(mc, "pert_chem_emb_dim", getattr(dc, "chem_fallback_embed_dim", 512)) or 512),
        ),
        pert_chem_enabled=bool(getattr(dc, "pert_chem_enabled", False)),
        seed=tc.seed, rank=rank,
        dataset_names=ds_subset,
        min_cells=dc.min_cells_per_cond,
    )
    def _make_eval_dataset(mode_key: str, seed_offset: int) -> CoupledFMDataset:
        return CoupledFMDataset(
            dc.biflow_dir, vocab, split,
            mode=mode_key, coupling_mode=tc.coupling_mode,
            latent_z_mode=tc.latent_z_mode,
            batch_size=tc.batch_size, ds_alpha=1.0,
            ot_method=tc.ot_method, ot_threads=tc.ot_threads,
            ot_sinkhorn_reg=getattr(tc, "ot_sinkhorn_reg", 0.05),
            ot_sinkhorn_iter=getattr(tc, "ot_sinkhorn_iter", 50),
            ot_emb_cap_src=tc.ot_emb_cap_src,
            ot_emb_cap_gt=tc.ot_emb_cap_gt,
            ot_feature=tc.ot_feature,
            de_dir=tc.de_dir,
            time_sampling=tc.time_sampling,
            ot_cost=tc.ot_cost,
            ot_sample_mode=tc.ot_sample_mode,
            gene_mask_prob=0.0,
            gene_mask_all_prob=0.0,
            gene_budget_manifest_path=getattr(tc, "gene_budget_manifest_path", ""),
            gene_budget_label=getattr(tc, "gene_budget_label", ""),
            use_graph=getattr(mc, "use_graph", False),
            nichenet_graph_pkl=dc.nichenet_graph_pkl_path,
            pert_idx_mode="zero",
            num_pert_ids=getattr(mc, "num_pert_ids", 10000),
            use_residual_flow=getattr(tc, "use_residual_flow", False),
            use_raw_pert_condition=use_ds_pert,
            max_pert_genes=getattr(mc, "max_pert_genes", 16),
            pert_gene_emb_cache_dir=(
                getattr(dc, "pert_gene_emb_cache_dir", "") if use_ds_pert else ""
            ),
            use_h5ad_pert_metadata=getattr(dc, "use_h5ad_pert_metadata", False),
            pert_metainfo_path=(
                str(getattr(dc, "pert_metainfo_path", "") or "") if use_ds_pert else ""
            ),
            chemical_metainfo_path=(
                str(getattr(dc, "chemical_metainfo_path", "") or "") if use_ds_pert else ""
            ),
            latent_backbone=str(getattr(dc, "latent_backbone", "state") or "state"),
            chem_emb_source_dir=str(getattr(dc, "chem_emb_source_dir", "") or ""),
            chem_obs_column=str(getattr(dc, "chem_obs_column", "") or ""),
            drug_emb_cache_dir=str(getattr(dc, "drug_emb_cache_dir", "") or ""),
            max_chem_keys=int(getattr(dc, "max_chem_keys", 4)),
            chem_fallback_embed_dim=max(
                8, int(getattr(mc, "pert_chem_emb_dim", getattr(dc, "chem_fallback_embed_dim", 512)) or 512),
            ),
            pert_chem_enabled=bool(getattr(dc, "pert_chem_enabled", False)),
            seed=tc.seed + seed_offset, rank=rank,
            dataset_names=ds_subset,
            shared_handles=train_ds.handles,
            min_cells=dc.min_cells_per_cond,
        )

    test_ds = _make_eval_dataset(test_key, 1000)
    val_ds = None
    val_source = "none"
    if has_val_split:
        if val_key == test_key:
            val_ds = test_ds
        else:
            val_ds = _make_eval_dataset(val_key, 2000)
        val_source = val_key
    gc.collect()
    print(f"[{_now()}] rank {rank} data loaded in "
          f"{_elapsed_str(time.time() - t_load)}, RSS={_rss_gb()}",
          flush=True)
    if _is_main(rank):
        eff_ot = train_ds.ot_feature_effective_label()
        run_meta = {
            "effective_ot_feature": eff_ot,
            "train_ot_feature_requested": tc.ot_feature,
            "split_source": "override" if split_is_override else "canonical",
            "split_path": str(split_path),
            "split_copy": str(run_split_copy),
            "split_sha256": _sha256_file(run_split_copy),
            "selection_protocol": protocol,
            "val_split_key_effective": val_key,
            "val_source": val_source,
            "test_split_key": test_key,
            "gene_budget_manifest_path": str(getattr(tc, "gene_budget_manifest_path", "") or ""),
            "gene_budget_label": str(getattr(tc, "gene_budget_label", "") or ""),
            "budget_mmd_mask_aware": bool(budget_mmd_mask_aware),
            "pytorch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "git_commit": _git_commit_short(),
            "environment_filtered": {
                k: v
                for k, v in os.environ.items()
                if k.startswith(
                    ("RAW_", "COUPLEDFM_", "PRETRAIN_", "TORCH_", "CUDA", "LOCAL_RANK", "RANK", "WORLD_SIZE"),
                )
            },
        }
        with open(out_dir / "run_meta.json", "w") as f:
            json.dump(run_meta, f, indent=2)
        cfg_dump_path = out_dir / "config.json"
        if cfg_dump_path.is_file():
            d_load = json.loads(cfg_dump_path.read_text(encoding="utf-8"))
            d_load["effective_ot_feature"] = eff_ot
            d_load["_runtime"] = run_meta
            cfg_dump_path.write_text(
                json.dumps(d_load, indent=2, default=str),
                encoding="utf-8",
            )
    if use_ddp:
        dist.barrier()

    # Compute gene-space means on-the-fly from the already-loaded dataset handles.
    # pert_means.npz / ctrl_means.npz were in latent space (2058-dim) which mismatches
    # gene-space pred/gt vectors; we now derive the correct gene-space vectors directly.
    if _is_main(rank):
        t_means = time.time()
        ctrl_means: dict = {}
        pert_means: dict = {}
        mean_handles = dict(train_ds.handles)
        if val_ds is not None:
            mean_handles.update(val_ds.handles)
        mean_handles.update(test_ds.handles)
        for ds_name, h in mean_handles.items():
            ctrl_means[ds_name] = h.ctrl_mean_gene()
            pert_means[ds_name] = h.compute_gt_mean_gene()
        print(
            f"[{_now()}] computed gene-space ctrl/pert means for "
            f"{len(pert_means)} datasets ({_elapsed_str(time.time() - t_means)})",
            flush=True,
        )
    else:
        ctrl_means = None
        pert_means = None
    if use_ddp:
        shared = [ctrl_means, pert_means]
        dist.broadcast_object_list(shared, src=0)
        ctrl_means, pert_means = shared

    # 非法 sparse / bias / ODE 配置在加载数据后立即失败
    _validate_model_train_config(tc, mc, test_ds.handles)

    ga = max(1, tc.grad_accum_steps)
    steps_per_epoch = train_ds.epoch_steps
    full_steps_per_epoch = steps_per_epoch
    max_steps_per_epoch = int(getattr(tc, "max_train_steps_per_epoch", 0) or 0)
    if max_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, max_steps_per_epoch)
    total_steps = steps_per_epoch * tc.epochs
    warmup_steps = steps_per_epoch * tc.warmup_epochs
    total_opt_steps = max(1, (total_steps + ga - 1) // ga)
    warmup_opt_steps = max(1, (warmup_steps + ga - 1) // ga)
    if getattr(tc, "warmup_steps", 0) > 0:
        warmup_opt_steps = max(1, int(tc.warmup_steps))

    nf_s = getattr(tc, "mmd_warmup_start_frac", None)
    nf_e = getattr(tc, "mmd_warmup_end_frac", None)
    if nf_s is not None and nf_e is not None:
        ws = int(total_opt_steps * float(nf_s))
        we = int(total_opt_steps * float(nf_e))
        if we <= ws:
            we = ws + 1
        tc.mmd_warmup_start = max(0, ws)
        tc.mmd_warmup_end = min(
            max(we, tc.mmd_warmup_start + 1),
            max(total_opt_steps, tc.mmd_warmup_start + 1),
        )

    # step-level val / epoch-level selection use ``val`` when available.  The
    # final test split is kept separate for explicit split-file RawFM smokes.
    final_eval_tasks = _collect_eval_tasks(test_ds)
    selection_ds = val_ds
    selection_all_tasks = _collect_eval_tasks(selection_ds) if selection_ds is not None else []
    if selection_ds is not None and int(getattr(tc, "val_every_steps", 0) or 0) > 0:
        val_tasks = build_monitor_val_tasks(
            selection_ds,
            fraction=tc.val_sample_ratio,
            seed=tc.seed,
            max_per_ds=max(0, int(tc.val_max_per_ds or 0)),
            min_per_ds=max(0, int(tc.val_min_per_ds or 0)),
            per_ds_target_range=True,
        )
    else:
        val_tasks = []
    if (
        selection_ds is not None
        and int(getattr(tc, "test_every_epoch", 0) or 0) > 0
        and getattr(tc, "test_max_per_ds", 0)
        and int(tc.test_max_per_ds) > 0
    ):
        test_tasks = build_monitor_val_tasks(
            selection_ds,
            fraction=1.0,
            seed=tc.seed,
            max_per_ds=int(tc.test_max_per_ds),
            min_per_ds=0,
            per_ds_target_range=False,
        )
    elif selection_ds is not None and int(getattr(tc, "test_every_epoch", 0) or 0) > 0:
        test_tasks = selection_all_tasks
    else:
        test_tasks = []
    selection_tag = "TEST" if selection_ds is test_ds else "SELECTION"

    if _is_main(rank):
        print(f"\n{'='*70}", flush=True)
        print(f"  [{_now()}] CoupledFM Training Configuration", flush=True)
        print(
            f"  mode={tc.coupling_mode}  ot_feature={tc.ot_feature}  "
            f"z_mode={tc.latent_z_mode}",
            flush=True,
        )
        steps_msg = f"  steps/epoch={steps_per_epoch}"
        if steps_per_epoch != full_steps_per_epoch:
            steps_msg += f" (limited from {full_steps_per_epoch})"
        print(f"{steps_msg}  total_steps={total_steps}  epochs={tc.epochs}", flush=True)
        print(f"  batch_size={tc.batch_size}  micro_batch={tc.micro_batch}  "
              f"world_size={world_size}", flush=True)
        if tc.coupling_mode in ("ot", "coupled"):
            csrc = tc.ot_emb_cap_src
            cgt = tc.ot_emb_cap_gt
            csrc_s = "∞" if csrc is None else str(csrc)
            cgt_s = "∞" if cgt is None else str(cgt)
            print(
                f"  OT ({tc.ot_feature}) Sinkhorn: cap_src={csrc_s}  cap_gt={cgt_s}  "
                f"(then sample {tc.batch_size} pairs / step / rank)",
                flush=True,
            )
        _lr_floor = float(tc.lr) * float(getattr(tc, "min_lr_ratio", 0.1))
        print(
            f"  lr={tc.lr:.1e}  warmup={tc.warmup_epochs}ep "
            f"({warmup_steps} steps)  cosine_lr_floor=lr*min_lr_ratio={_lr_floor:.1e}",
            flush=True,
        )
        print(
            f"  val: {len(val_tasks)}/{len(selection_all_tasks)} conds  |  "
            f"source={val_source}  "
            f"val every {tc.val_every_steps} steps, "
            f"ode_steps={tc.val_ode_steps} method={getattr(tc, 'val_ode_method', 'euler')}, "
            f"max_cells={tc.val_max_cells}",
            flush=True,
        )
        print(
            f"  selection: {len(test_tasks)}/{len(selection_all_tasks)} conds  |  "
            f"protocol={protocol} every {tc.test_every_epoch} ep, "
            f"ode_steps={tc.eval_ode_steps} method={getattr(tc, 'val_ode_method', 'euler')}, "
            f"max_cells={tc.test_max_cells}, early_stop_patience={tc.early_stop_patience}",
            flush=True,
        )
        print(f"  final_test: {len(final_eval_tasks)} conds from split key {test_key!r}", flush=True)
        print(
            f"  selection_metric={tc.selection_metric}  "
            f"loss_guard_epochs={tc.loss_guard_epochs}  "
            f"min_epochs_before_stop={tc.min_epochs_before_stop}",
            flush=True,
        )
        print(
            f"  MMD: enabled={tc.use_mmd}  gamma_max={tc.mmd_gamma_max}  "
            f"warmup=[{tc.mmd_warmup_start},{tc.mmd_warmup_end}]  "
            f"every={tc.mmd_every}  epoch_start={tc.mmd_epoch_start}",
            flush=True,
        )
        print(
            f"  split: {'override' if split_is_override else 'canonical'} {split_path} "
            f"(explicit_pert_split legacy flag={'set' if dc.explicit_pert_split else 'clear'})",
            flush=True,
        )
        print(f"  attn_backend={mc.attn_backend}  "
              f"amp={tc.use_amp} ({tc.amp_dtype})  fp64={tc.fp64_training}  "
              f"debug_nan={tc.debug_nan}  grad_clip={tc.grad_clip}",
              flush=True)
        print(
            f"  grad_accum_steps={ga}  "
            f"(micro_batch chunks still within each batch; "
            f"LR cosine on opt_step: {total_opt_steps} opt steps, "
            f"warmup {warmup_opt_steps} opt steps)",
            flush=True,
        )
        print(f"{'='*70}\n", flush=True)

    # ── model ────────────────────────────────────────────────────
    pert_cache_model = None
    if getattr(mc, "use_pert_condition", False):
        pm = str(getattr(mc, "pert_embed_mode", "random_learned")).lower().strip()
        dcache = str(getattr(dc, "pert_gene_emb_cache_dir", "") or "").strip()
        if pm.startswith("pretrained"):
            if not dcache:
                raise ValueError(
                    "model.use_pert_condition with pretrained* pert_embed_mode requires "
                    "data.pert_gene_emb_cache_dir"
                )
            from model.condition_emb.genepert.gene_cache import GeneEmbeddingCache
            pert_cache_model = GeneEmbeddingCache(Path(dcache).expanduser())
        elif dcache:
            from model.condition_emb.genepert.gene_cache import GeneEmbeddingCache
            pert_cache_model = GeneEmbeddingCache(Path(dcache).expanduser())

    model = RawExprVelocityField(
        d_model=mc.d_model, n_layer=mc.n_layer, n_head=mc.n_head,
        d_ff=mc.d_ff, dropout=mc.dropout, attn_mode=mc.attn_mode,
        d_latent=mc.d_latent,
        attn_backend=mc.attn_backend,
        coupling_mode=tc.coupling_mode,
        use_pert_token=getattr(mc, "use_pert_token", False),
        num_pert_ids=getattr(mc, "num_pert_ids", 10000),
        graph_bias_mode=getattr(mc, "graph_bias_mode", "none"),
        use_latent_resampler=getattr(mc, "use_latent_resampler", False),
        latent_resampler_n_tokens=getattr(mc, "latent_resampler_n_tokens", 8),
        latent_resampler_n_head=getattr(mc, "latent_resampler_n_head", 4),
        cross_attn_independent_kv=getattr(mc, "cross_attn_independent_kv", False),
        value_encoder=getattr(mc, "value_encoder", "linear"),
        fourier_n_freqs=getattr(mc, "fourier_n_freqs", 32),
        use_residual_flow=getattr(tc, "use_residual_flow", False),
        use_pert_condition=getattr(mc, "use_pert_condition", False),
        pert_embed_mode=getattr(mc, "pert_embed_mode", "random_learned"),
        pert_cond_dim=getattr(mc, "pert_cond_dim", mc.d_model),
        pert_type_emb_dim=getattr(mc, "pert_type_emb_dim", 32),
        pert_encoder_num_embeddings=getattr(mc, "pert_encoder_num_embeddings", 8192),
        pert_gene_emb_dim=getattr(mc, "pert_gene_emb_dim", 256),
        pert_encoder_dropout=getattr(mc, "pert_encoder_dropout", 0.0),
        max_combo_id_exclusive=getattr(mc, "max_combo_id_exclusive", 4096),
        gene_embedding_cache=pert_cache_model,
        legacy_cond_vec_dim=int(getattr(mc, "legacy_cond_vec_dim", 0)),
        pert_chem_emb_dim=int(getattr(mc, "pert_chem_emb_dim", 0)),
        pert_chem_projector_hidden=int(getattr(mc, "pert_chem_projector_hidden", 0)),
        pert_gene_projector_hidden=int(getattr(mc, "pert_gene_projector_hidden", 0)),
        pert_type_scale_init=tuple(getattr(mc, "pert_type_scale_init", (0.0, -1.0, -1.0, -1.0, 1.0, 1.0))),
        pool_aggregations=tuple(getattr(mc, "pert_pool_aggregations", ("mean",))),
        pool_scale_init=tuple(float(x) for x in getattr(mc, "pert_pool_scale_init", (1.0,))),
        pool_fusion_mode=str(getattr(mc, "pert_pool_fusion_mode", "sum")),
        type_adapter_mode=str(getattr(mc, "pert_type_adapter_mode", "scalar")),
        condition_embedding_source=(
            str(getattr(mc, "pert_condition_embedding_source", "") or "").strip() or None
        ),
    ).to(device)

    if tc.pretrained_ckpt:
        if not os.path.isfile(tc.pretrained_ckpt):
            raise FileNotFoundError(
                f"train.pretrained_ckpt is set but missing: {tc.pretrained_ckpt}. "
                "Set RAW_PRETRAINED_CKPT='' only when intentionally training from scratch."
            )
        load_velocity_pretrained_bundle(
            model, tc.pretrained_ckpt, verbose=_is_main(rank),
        )

    if getattr(mc, "use_lora", False):
        from model.utils.models.lora import apply_lora_to_model
        n_lora = apply_lora_to_model(
            model,
            rank=mc.lora_rank,
            target_substrs=mc.lora_target,
        )
        if _is_main(rank):
            print(f"[{_now()}] LoRA: replaced {n_lora} Linear modules "
                  f"(rank={mc.lora_rank})", flush=True)

    if tc.fp64_training:
        model = model.double()
        if _is_main(rank):
            print(f"[{_now()}] fp64_training: model cast to float64", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if _is_main(rank):
        print(f"[{_now()}] model params: {n_params:,} total, "
              f"{n_train:,} trainable, GPU_mem={_gpu_mem_mb()}", flush=True)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # ── 两阶段 FT 阶段 1：先冻 backbone（必须在 optimizer 构造前） ──
    stage1_active = bool(getattr(tc, "two_stage_ft", False))
    if stage1_active:
        _stage1_freeze_backbone(model, verbose=_is_main(rank))

    # ── optimizer & AMP ──────────────────────────────────────────
    betas = (float(getattr(tc, "adam_beta1", 0.9)), float(getattr(tc, "adam_beta2", 0.95)))
    if getattr(tc, "use_param_groups", False):
        bb_params, nw_params, bb_names, nw_names = _split_params(
            model, include_frozen=stage1_active,
        )
        if _is_main(rank):
            n_bb = sum(p.numel() for p in bb_params)
            n_nw = sum(p.numel() for p in nw_params)
            print(f"[{_now()}] param groups: backbone={len(bb_params)} tensors "
                  f"({n_bb:,} params)  new_modules={len(nw_params)} tensors ({n_nw:,} params)",
                  flush=True)
        # stage1 时 backbone 一律 lr=0（即便有 grad，也被 scale 成 0；requires_grad=False 下 grad=None 则 skip）
        bb_base_lr = tc.lr * (tc.stage2_backbone_mult if stage1_active else 1.0)
        nw_base_lr = tc.lr * tc.lr_new_module_mult
        optimizer = torch.optim.AdamW([
            {"params": bb_params, "lr": bb_base_lr,
             "weight_decay": tc.weight_decay_backbone, "name": "backbone",
             "base_lr": tc.lr},  # base_lr = 原始峰值；stage2 mult 在 epoch 切换时重设
            {"params": nw_params, "lr": nw_base_lr,
             "weight_decay": tc.weight_decay_new, "name": "new_modules",
             "base_lr": nw_base_lr},
        ], betas=betas)
    else:
        # two-stage 也要把当前冻结参数放进 optimizer，后续解冻后才能继续沿用同一 optimizer 训练。
        single_group_params = list(model.parameters()) if stage1_active else [
            p for p in model.parameters() if p.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            single_group_params,
            lr=tc.lr, weight_decay=tc.weight_decay, betas=betas,
        )
    use_amp = (
        tc.use_amp and torch.cuda.is_available() and not tc.fp64_training
    )
    amp_dt = _amp_dtype(tc.amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dt == torch.float16))

    # ── EMA shadow weights ───────────────────────────────────────
    ema = None
    if getattr(tc, "use_ema", False):
        # DDP：EMA 跟踪 model.module（带 DDP wrapper 内部），ModelEMA 内部自动 _unwrap
        ema = ModelEMA(
            model,
            decay=tc.ema_decay,
            update_after=tc.ema_update_after,
            update_every=tc.ema_update_every,
            device=device,
            dynamic=getattr(tc, "ema_dynamic", False),
        )
        if _is_main(rank):
            print(
                f"[{_now()}] EMA enabled: decay={tc.ema_decay}  "
                f"update_after={tc.ema_update_after}  "
                f"dynamic={getattr(tc, 'ema_dynamic', False)}",
                flush=True,
            )

    start_epoch = 0
    global_step = 0
    optimizer_step = 0
    accum_batch_idx = 0
    best_pd_ctrl = -float("inf")
    best_corr_pert = -float("inf")
    best_selection_score = -float("inf")
    no_improve_count = 0
    patience_enabled = int(getattr(tc, "early_stop_patience", 0)) > 0
    lr = tc.lr

    if tc.resume_from and os.path.isfile(tc.resume_from):
        ckpt = torch.load(tc.resume_from, map_location=device, weights_only=False)
        raw_model = model.module if use_ddp else model
        resume_info = raw_model.load_state_dict(ckpt["model"], strict=False)
        if _is_main(rank):
            print(
                f"[{_now()}] resume load strict=False missing={len(resume_info.missing_keys)} "
                f"unexpected={len(resume_info.unexpected_keys)}",
                flush=True,
            )
        optimizer.load_state_dict(ckpt["optimizer"])
        if ema is not None and "ema" in ckpt:
            try:
                ema.load_state_dict(ckpt["ema"], strict=False)
            except Exception as e:  # pragma: no cover
                if _is_main(rank):
                    print(f"[{_now()}] WARNING: failed to load EMA state ({e}); "
                          f"starting from current model weights", flush=True)
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        optimizer_step = ckpt.get("optimizer_step", global_step // ga)
        accum_batch_idx = ckpt.get("accum_batch_idx", global_step % ga)
        best_pd_ctrl = ckpt.get("best_pd_ctrl", -float("inf"))
        best_corr_pert = ckpt.get("best_corr_pert", -float("inf"))
        best_selection_score = ckpt.get("best_selection_score", best_corr_pert)
        no_improve_count = ckpt.get("no_improve_count", 0)
        if _is_main(rank):
            print(f"[{_now()}] resumed from epoch {start_epoch}, "
                  f"global_step={global_step} opt_step={optimizer_step} "
                  f"accum_batch_idx={accum_batch_idx} "
                  f"best_pp={best_corr_pert:.5f} "
                  f"best_sel={best_selection_score:.5f}",
                  flush=True)

    # ── frozen latent FM ─────────────────────────────────────────
    latent_fm = None
    if tc.coupling_mode == "coupled" and tc.latent_z_mode in ("ode", "curriculum") and not tc.latent_fm_ckpt:
        raise ValueError(
            f"latent_z_mode={tc.latent_z_mode!r} requires cfg.train.latent_fm_ckpt to be set."
        )
    if tc.coupling_mode == "coupled" and tc.latent_fm_ckpt:
        from model.latent_utils import FrozenLatentFM
        latent_fm = FrozenLatentFM(tc.latent_fm_ckpt, device=str(device))

    use_ode_zt = (tc.latent_z_mode == "ode" and latent_fm is not None)
    use_curriculum = (tc.latent_z_mode == "curriculum" and latent_fm is not None)
    # d_latent 用于 curriculum 模式下切分 (interp, z_ctrl) 两半
    d_latent = mc.d_latent

    _save_git = _git_commit_short()
    _save_torch_v = torch.__version__
    _latent_fm_ckpt_s = str(getattr(tc, "latent_fm_ckpt", "") or "")

    log_file = out_dir / "train_log.jsonl"
    mb = tc.micro_batch

    # ── initial fast val (before any training) ───────────────────
    if bool(getattr(tc, "run_initial_val", True)) and selection_ds is not None and val_tasks:
        if _is_main(rank):
            print(f"\n[{_now()}] === Initial val (before training) ===", flush=True)
        val_res = _run_eval(
            model, selection_ds, device, tc, latent_fm, use_amp, amp_dt,
            rank, world_size, tasks=val_tasks, tag="val", step=0,
            per_dataset=True, max_cells=tc.val_max_cells,
            ode_steps=tc.val_ode_steps,
            ctrl_means=ctrl_means, pert_means=pert_means,
            cfg_w=getattr(cfg.inference, "cfg_w", 1.0),
            use_residual_flow=getattr(tc, "use_residual_flow", False),
            max_pert_genes=int(getattr(mc, "max_pert_genes", 16)),
        )
        if _is_main(rank) and val_res is not None:
            record = {
                "global_step": 0, "epoch": 0, "train_loss": None,
                "lr": lr, "eval_type": "val",
                **{f"eval_{k}": v for k, v in val_res["global"].items()},
            }
            with open(log_file, "a") as f:
                f.write(json.dumps(record) + "\n")
    elif _is_main(rank):
        print(f"\n[{_now()}] === Initial val skipped ===", flush=True)

    early_stopped = False
    step_times = []
    _debug_first_fwd = True

    if _is_main(rank):
        print(f"\n[{_now()}] === Training started ===\n", flush=True)

    stage2_backbone_warmup_start: Optional[int] = None

    for epoch in range(start_epoch, tc.epochs):
        # ── 两阶段 FT：epoch 切换时解冻 backbone + 重设其 base_lr ──
        if (stage1_active and epoch >= tc.two_stage_freeze_epochs):
            _stage2_unfreeze_backbone(model, verbose=_is_main(rank))
            if ema is not None:
                n_added = ema.add_missing_parameters(model)
                if _is_main(rank) and n_added > 0:
                    print(f"[{_now()}] [stage2] EMA registered {n_added} newly unfrozen params",
                          flush=True)
            stage1_active = False
            stage2_backbone_warmup_start = optimizer_step
            if getattr(tc, "use_param_groups", False):
                # 把 backbone group 的 base_lr 下调到 stage2 倍数（如 0.1×）
                for pg in optimizer.param_groups:
                    if pg.get("name") == "backbone":
                        pg["base_lr"] = tc.lr * tc.stage2_backbone_mult
                        if _is_main(rank):
                            print(f"[{_now()}] [stage2] backbone base_lr -> {pg['base_lr']:.2e}",
                                  flush=True)

        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t_epoch = time.time()

        for batch, is_last_batch in _iter_with_last_flag(train_ds):
            t_step = time.time()

            core_batch, perturbation_batch_cpu, latent_data = unpack_training_batch(batch)
            (x_t, x_ctrl, t_val, gene_ids,
             dx_target, gene_mask, pert_idx, edge_index, ds_name, cond,
             dx_prior) = core_batch

            B = x_t.shape[0]
            G = x_t.shape[1]
            raw_train_m = model.module if use_ddp else model
            raw_wants_pert = bool(getattr(raw_train_m, "use_pert_condition", False))
            latent_wants = latent_fm_wants_perturbation(latent_fm)
            max_pg_raw = int(getattr(mc, "max_pert_genes", 16))
            max_pg_lat = (
                int(getattr(latent_fm, "max_pert_genes", max_pg_raw))
                if latent_fm is not None
                else max_pg_raw
            )
            pb_dev = perturbation_batch_to_device(perturbation_batch_cpu, device)
            gene_ids_d = gene_ids.to(device)
            eff_mb = _safe_train_mb(G, mb, attn_backend=mc.attn_backend)
            n_chunks = max(1, (B + eff_mb - 1) // eff_mb)

            # ── LR schedule ──────────────────────────────────────
            # 分组 cosine_with_min_lr_ratio vs 单组 legacy cosine
            if getattr(tc, "use_param_groups", False):
                lrs = []
                for pg in optimizer.param_groups:
                    lr_i = lr_warmup_cosine_ratio_floor_absolute(
                        optimizer_step, warmup_opt_steps, total_opt_steps,
                        pg["base_lr"], getattr(tc, "min_lr_ratio", 0.1),
                    )
                    lrs.append(lr_i)
                # stage1：强制 backbone lr=0（若 stage2_backbone_mult>0，backbone 不动也可设 0）
                if stage1_active:
                    for i, pg in enumerate(optimizer.param_groups):
                        if pg.get("name") == "backbone":
                            lrs[i] = 0.0
                sw = getattr(tc, "stage2_backbone_warmup_steps", 0) or 0
                if (
                    not stage1_active
                    and sw > 0
                    and stage2_backbone_warmup_start is not None
                ):
                    prog = optimizer_step - stage2_backbone_warmup_start
                    if prog < sw:
                        alpha = min(1.0, max(0.0, float(prog + 1) / float(sw)))
                        for i, pg in enumerate(optimizer.param_groups):
                            if pg.get("name") == "backbone":
                                lrs[i] *= alpha
                _set_group_lrs(optimizer, lrs)
                lr = lrs[-1]  # 日志用 new_modules 的 lr
            else:
                lr = lr_warmup_cosine_ratio_floor_absolute(
                    optimizer_step, warmup_opt_steps, total_opt_steps,
                    tc.lr, getattr(tc, "min_lr_ratio", 0.1),
                )
                _set_lr(optimizer, lr)

            # ── Curriculum decision（per batch flip a coin） ─────
            curriculum_use_ode = False
            if use_curriculum:
                p_ode = get_ode_prob_curriculum(
                    global_step,
                    warmup_steps=tc.curriculum_warmup_steps,
                    anneal_steps=tc.curriculum_anneal_steps,
                    max_prob=tc.curriculum_max_prob,
                )
                curriculum_use_ode = (random.random() < p_ode)

            cur_gamma = _gamma_schedule_mmd(global_step, tc)
            if (not tc.use_mmd) or epoch < tc.mmd_epoch_start:
                cur_gamma = 0.0
            if tc.mmd_every > 1 and global_step % tc.mmd_every != 0:
                cur_gamma = 0.0
            mmd_will_run = cur_gamma > 0
            mmd_raw_val = None
            mmd_gene_count = None

            if accum_batch_idx == 0:
                optimizer.zero_grad(set_to_none=True)
            chunk_loss_vals = []
            chunk_sizes = [
                min((ci + 1) * eff_mb, B) - ci * eff_mb for ci in range(n_chunks)
            ]
            xm = getattr(tc, "xt_noise_sigma_max", 0.0) or 0.0
            xt_sigma = random.uniform(0.0, xm) if xm > 0 else 0.0
            noise_xt = (
                torch.randn(B, G, device=device, dtype=torch.float32)
                if xt_sigma > 0 else None
            )
            eidx_d = (
                edge_index.to(device) if edge_index is not None else None
            )
            cfg_drop = (
                (getattr(tc, "cfg_drop_prob", 0.0) or 0.0) > 0.0
                and random.random() < float(tc.cfg_drop_prob)
            )

            def _raw_pb_chunk(s_i: int, e_i: int):
                if not raw_wants_pert:
                    return None
                if cfg_drop:
                    return null_perturbation_batch(
                        e_i - s_i, max_pg_raw, device=device,
                    )
                if pb_dev is None:
                    return null_perturbation_batch(
                        e_i - s_i, max_pg_raw, device=device,
                    )
                return slice_perturbation_batch(pb_dev, s_i, e_i, device)

            def _lat_pb_chunk(s_i: int, e_i: int):
                if not latent_wants:
                    return None
                if cfg_drop:
                    return null_perturbation_batch(
                        e_i - s_i, max_pg_lat, device=device,
                    )
                if pb_dev is None:
                    return null_perturbation_batch(
                        e_i - s_i, max_pg_lat, device=device,
                    )
                return slice_perturbation_batch(pb_dev, s_i, e_i, device)

            def _raw_pb_rows(n_rows: int):
                if not raw_wants_pert:
                    return None
                if cfg_drop:
                    return null_perturbation_batch(n_rows, max_pg_raw, device=device)
                if pb_dev is None:
                    return null_perturbation_batch(n_rows, max_pg_raw, device=device)
                return pb_dev

            def _lat_pb_rows(n_rows: int):
                if not latent_wants:
                    return None
                if cfg_drop:
                    return null_perturbation_batch(n_rows, max_pg_lat, device=device)
                if pb_dev is None:
                    return null_perturbation_batch(n_rows, max_pg_lat, device=device)
                return pb_dev

            urf = getattr(tc, "use_residual_flow", False)
            dx_prior_dev = None
            if urf and dx_prior is not None:
                dx_prior_dev = dx_prior.to(device)
            for ci in range(n_chunks):
                s, e = ci * eff_mb, min((ci + 1) * eff_mb, B)
                x_t_c = x_t[s:e].to(device)
                if noise_xt is not None:
                    x_t_c = x_t_c + xt_sigma * noise_xt[s:e]
                x_ctrl_c = x_ctrl[s:e].to(device)
                t_c = t_val[s:e].to(device)
                dx_c = dx_target[s:e].to(device)
                if urf and dx_prior_dev is not None:
                    dp = dx_prior_dev.unsqueeze(0).expand_as(dx_c)
                    dx_c = dx_c - dp
                gm_c = gene_mask[s:e].to(device)
                pi_c = pert_idx[s:e].to(device)
                if tc.fp64_training:
                    x_t_c = x_t_c.double()
                    x_ctrl_c = x_ctrl_c.double()
                    t_c = t_c.double()
                    dx_c = dx_c.double()
                    gm_c = gm_c.double()

                aux_c = None
                if latent_data is not None:
                    if use_curriculum:
                        # latent_data.shape[-1] == 2 * d_latent；前半是 interp，后半是 z_ctrl
                        ld = latent_data[s:e].to(device)
                        if tc.fp64_training:
                            ld = ld.double()
                        z_t_interp_c, z_lat_src_c = torch.chunk(ld, 2, dim=-1)
                        if curriculum_use_ode:
                            aux_c = latent_fm.ode_at_t(
                                z_lat_src_c, t_c, n_steps=tc.latent_ode_steps,
                                perturbation_batch=_lat_pb_chunk(s, e),
                            )
                            if tc.fp64_training and aux_c is not None:
                                aux_c = aux_c.double()
                        else:
                            aux_c = z_t_interp_c
                    elif use_ode_zt:
                        z_src_c = latent_data[s:e].to(device)
                        if tc.fp64_training:
                            z_src_c = z_src_c.double()
                        aux_c = latent_fm.ode_at_t(
                            z_src_c, t_c, n_steps=tc.latent_ode_steps,
                            perturbation_batch=_lat_pb_chunk(s, e),
                        )
                        if tc.fp64_training and aux_c is not None:
                            aux_c = aux_c.double()
                    else:
                        aux_c = latent_data[s:e].to(device)
                        if tc.fp64_training:
                            aux_c = aux_c.double()

                x_ctrl_eff = x_ctrl_c
                pi_eff = pi_c
                if cfg_drop:
                    x_ctrl_eff = torch.zeros_like(x_ctrl_c)
                    pi_eff = torch.zeros_like(pi_c)
                    aux_c = None

                fwd_kw = {}
                if raw_wants_pert:
                    fwd_kw["perturbation_batch"] = _raw_pb_chunk(s, e)
                with torch.amp.autocast("cuda", dtype=amp_dt, enabled=use_amp):
                    v_pred = model(
                        x_t_c, x_ctrl_eff, t_c, gene_ids_d, aux_c, gm_c, pi_eff,
                        edge_index=eidx_d,
                        **fwd_kw,
                    )
                    per_elem = (v_pred - dx_c).pow(2)
                    w = compute_loss_weight(
                        t_c,
                        mode=getattr(tc, "loss_weighting", "none"),
                        snr_gamma=getattr(tc, "min_snr_gamma", 5.0),
                    )
                    vis = (1.0 - gm_c).to(per_elem.dtype)
                    d = vis.sum(dim=-1)
                    per_sample = (per_elem * vis).sum(dim=-1) / d.clamp(min=1e-6)
                    per_sample = torch.where(
                        d > 0, per_sample, torch.zeros_like(per_sample),
                    )
                    chunk_loss = (per_sample * w).mean()

                if tc.debug_nan and _is_main(rank) and _debug_first_fwd and ci == 0:
                    _debug_first_fwd = False
                    vp = v_pred.detach().float()
                    dx_dbg = dx_c.detach().float()
                    print(
                        f"[{_now()}] DEBUG first_fwd: v_pred "
                        f"finite={torch.isfinite(vp).all().item()} "
                        f"nan_cnt={torch.isnan(vp).sum().item()} "
                        f"min={vp.min().item():.6g} max={vp.max().item():.6g} | "
                        f"dx finite={torch.isfinite(dx_dbg).all().item()}",
                        flush=True,
                    )

                chunk_loss_vals.append(chunk_loss.detach())
                # 按 chunk 加权 / grad_accum：使 ga 个 batch 的梯度均值为一次 update
                w = chunk_sizes[ci] / B / float(ga)
                scaled = scaler.scale(chunk_loss * w)
                # DDP：除「整窗最后一步」外，epoch 最后一个 batch 的最后 microchunk 也必须同步，
                # 否则 epoch 末尾部分累积时各 rank 梯度未对齐。
                is_last_backward = (ci == n_chunks - 1) and (
                    (accum_batch_idx == ga - 1) or is_last_batch
                )
                if mmd_will_run:
                    is_last_backward = False
                sync_ctx = (
                    model.no_sync()
                    if use_ddp and not is_last_backward
                    else contextlib.nullcontext()
                )
                with sync_ctx:
                    scaled.backward()

            if mmd_will_run:
                x_t_f = x_t.to(device)
                if noise_xt is not None:
                    x_t_f = x_t_f + xt_sigma * noise_xt
                x_ctrl_f = x_ctrl.to(device)
                t_f = t_val.to(device)
                dx_f = dx_target.to(device)
                if urf and dx_prior is not None:
                    dp_f = dx_prior.to(device).to(dx_f.dtype).unsqueeze(0).expand_as(dx_f)
                    dx_f = dx_f - dp_f
                gm_f = gene_mask.to(device)
                pi_f = pert_idx.to(device)
                if tc.fp64_training:
                    x_t_f = x_t_f.double()
                    x_ctrl_f = x_ctrl_f.double()
                    t_f = t_f.double()
                    dx_f = dx_f.double()
                    gm_f = gm_f.double()

                aux_f = None
                if latent_data is not None:
                    if use_curriculum:
                        ld_f = latent_data.to(device)
                        if tc.fp64_training:
                            ld_f = ld_f.double()
                        z_t_interp_f, z_lat_src_f = torch.chunk(ld_f, 2, dim=-1)
                        if curriculum_use_ode:
                            aux_f = latent_fm.ode_at_t(
                                z_lat_src_f, t_f, n_steps=tc.latent_ode_steps,
                                perturbation_batch=_lat_pb_rows(B),
                            )
                            if tc.fp64_training and aux_f is not None:
                                aux_f = aux_f.double()
                        else:
                            aux_f = z_t_interp_f
                    elif use_ode_zt:
                        z_src_f = latent_data.to(device)
                        if tc.fp64_training:
                            z_src_f = z_src_f.double()
                        aux_f = latent_fm.ode_at_t(
                            z_src_f, t_f, n_steps=tc.latent_ode_steps,
                            perturbation_batch=_lat_pb_rows(B),
                        )
                        if tc.fp64_training and aux_f is not None:
                            aux_f = aux_f.double()
                    else:
                        aux_f = latent_data.to(device)
                        if tc.fp64_training:
                            aux_f = aux_f.double()

                x_ctrl_m = x_ctrl_f
                pi_m = pi_f
                if cfg_drop:
                    x_ctrl_m = torch.zeros_like(x_ctrl_f)
                    pi_m = torch.zeros_like(pi_f)
                    aux_f = None

                # ── OOM-safe: 按 cell 维切 micro-chunk 前向 ────────
                mmd_mc = getattr(tc, "mmd_micro_chunk", 0)
                if mmd_mc and mmd_mc > 0 and B > mmd_mc:
                    v_parts = []
                    with torch.amp.autocast("cuda", dtype=amp_dt, enabled=use_amp):
                        for s_m in range(0, B, mmd_mc):
                            e_m = min(s_m + mmd_mc, B)
                            aux_s = aux_f[s_m:e_m] if aux_f is not None else None
                            mmd_kw = {}
                            if raw_wants_pert:
                                mmd_kw["perturbation_batch"] = _raw_pb_chunk(s_m, e_m)
                            v_parts.append(
                                model(
                                    x_t_f[s_m:e_m], x_ctrl_m[s_m:e_m], t_f[s_m:e_m],
                                    gene_ids_d, aux_s, gm_f[s_m:e_m], pi_m[s_m:e_m],
                                    edge_index=eidx_d,
                                    **mmd_kw,
                                )
                            )
                    v_full = torch.cat(v_parts, dim=0)
                else:
                    mmd_kw_full = {}
                    if raw_wants_pert:
                        mmd_kw_full["perturbation_batch"] = _raw_pb_rows(B)
                    with torch.amp.autocast("cuda", dtype=amp_dt, enabled=use_amp):
                        v_full = model(
                            x_t_f, x_ctrl_m, t_f, gene_ids_d, aux_f, gm_f,
                            pi_m, edge_index=eidx_d,
                            **mmd_kw_full,
                        )

                t_1 = t_f.unsqueeze(-1)
                x1_hat = x_t_f + v_full * (1.0 - t_1)
                x_gt = x_t_f + (1.0 - t_1) * dx_f

                xh = x1_hat.float()
                yh = x_gt.float()
                xh_mmd, yh_mmd, mmd_gene_count = _project_mmd_to_visible_genes(
                    xh, yh, gm_f,
                )
                sigmas = median_sigmas(yh_mmd)
                mmd_raw = mmd2_unbiased(xh_mmd, yh_mmd, sigmas)
                mmd_t = torch.clamp(mmd_raw, min=0.0)
                loss_mmd = cur_gamma * mmd_t
                mmd_raw_val = float(mmd_raw.detach().item())

                is_last_backward_mmd = (accum_batch_idx == ga - 1) or is_last_batch
                sync_mmd_ctx = (
                    model.no_sync()
                    if use_ddp and not is_last_backward_mmd
                    else contextlib.nullcontext()
                )
                w_mmd = 1.0 / float(ga)
                scaled_mmd = scaler.scale(loss_mmd * w_mmd)
                with sync_mmd_ctx:
                    scaled_mmd.backward()

            if tc.debug_nan and _is_main(rank):
                raw_m = model.module if use_ddp else model
                bad_g = [
                    n for n, p in raw_m.named_parameters()
                    if p.grad is not None and not torch.isfinite(p.grad).all()
                ]
                bad_p = [
                    n for n, p in raw_m.named_parameters()
                    if not torch.isfinite(p).all()
                ]
                if bad_g or bad_p:
                    print(
                        f"[{_now()}] DEBUG non-finite after bwd: "
                        f"grads({len(bad_g)})={bad_g[:12]} | "
                        f"params({len(bad_p)})={bad_p[:12]}",
                        flush=True,
                    )

            accum_batch_idx += 1
            if accum_batch_idx == ga:
                if tc.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer_step += 1
                accum_batch_idx = 0
                # EMA 在真正 optimizer.step 之后更新（grad_accum 的跨 batch 累加也能对齐）
                if ema is not None:
                    ema.update(model, step=optimizer_step)

            batch_loss = sum(
                t.item() * (chunk_sizes[ci] / B)
                for ci, t in enumerate(chunk_loss_vals)
            )
            if tc.debug_nan and _is_main(rank) and not math.isfinite(batch_loss):
                print(
                    f"[{_now()}] DEBUG batch_loss non-finite: {batch_loss}",
                    flush=True,
                )

            step_dt = time.time() - t_step
            step_times.append(step_dt)
            if len(step_times) > 100:
                step_times = step_times[-100:]

            epoch_loss += batch_loss
            epoch_steps += 1
            global_step += 1

            if _is_main(rank) and (global_step == 1 or global_step % tc.log_every == 0):
                avg_step_t = sum(step_times) / len(step_times)
                ep_remaining = steps_per_epoch - epoch_steps
                ep_eta = ep_remaining * avg_step_t
                total_eta = (total_steps - global_step) * avg_step_t
                epoch_progress = epoch_steps / steps_per_epoch * 100
                ga_note = f" opt={optimizer_step}/{total_opt_steps}" if ga > 1 else ""
                mmd_note = ""
                if mmd_raw_val is not None:
                    mmd_note = f" mmd={mmd_raw_val:.6f} γ={cur_gamma:.5f}"
                    if mmd_gene_count is not None:
                        mmd_note += f" mmd_genes={mmd_gene_count}"
                print(
                    f"  [{_now()}] ep {epoch} step {epoch_steps}/{steps_per_epoch}"
                    f"({epoch_progress:.0f}%) "
                    f"[global {global_step}/{total_steps}{ga_note}] "
                    f"loss={batch_loss:.5f}{mmd_note} lr={lr:.2e} "
                    f"| {step_dt:.2f}s avg={avg_step_t:.2f}s "
                    f"ep_ETA={_eta_str(ep_eta)} total_ETA={_eta_str(total_eta)} "
                    f"| B={B} G={G} ds={ds_name} "
                    f"GPU={_gpu_mem_mb()}",
                    flush=True,
                )

            # ── step-level: fast val monitoring ───────────────────
            if (tc.val_every_steps > 0
                    and global_step % tc.val_every_steps == 0
                    and selection_ds is not None
                    and val_tasks):
                if ema is not None and optimizer_step >= tc.ema_update_after:
                    with ema.apply_to(model):
                        val_res = _run_eval(
                            model, selection_ds, device, tc, latent_fm, use_amp, amp_dt,
                            rank, world_size, tasks=val_tasks, tag="val",
                            step=global_step, per_dataset=True,
                            max_cells=tc.val_max_cells,
                            ode_steps=tc.val_ode_steps,
                            ctrl_means=ctrl_means, pert_means=pert_means,
                            cfg_w=getattr(cfg.inference, "cfg_w", 1.0),
                            use_residual_flow=getattr(tc, "use_residual_flow", False),
                            max_pert_genes=int(getattr(mc, "max_pert_genes", 16)),
                        )
                else:
                    val_res = _run_eval(
                        model, selection_ds, device, tc, latent_fm, use_amp, amp_dt,
                        rank, world_size, tasks=val_tasks, tag="val",
                        step=global_step, per_dataset=True,
                        max_cells=tc.val_max_cells,
                        ode_steps=tc.val_ode_steps,
                        ctrl_means=ctrl_means, pert_means=pert_means,
                        cfg_w=getattr(cfg.inference, "cfg_w", 1.0),
                        use_residual_flow=getattr(tc, "use_residual_flow", False),
                        max_pert_genes=int(getattr(mc, "max_pert_genes", 16)),
                    )
                if _is_main(rank) and val_res is not None:
                    record = {
                        "global_step": global_step, "epoch": epoch,
                        "train_loss": batch_loss, "lr": lr,
                        "eval_type": "val",
                        **{f"eval_{k}": v for k, v in val_res["global"].items()},
                    }
                    if mmd_raw_val is not None:
                        record["train_mmd_raw"] = mmd_raw_val
                        record["train_gamma"] = cur_gamma
                        if mmd_gene_count is not None:
                            record["train_mmd_visible_genes"] = int(mmd_gene_count)
                    with open(log_file, "a") as f:
                        f.write(json.dumps(record) + "\n")
                model.train()

            if epoch_steps >= steps_per_epoch:
                break

        # epoch 末尾不足 ga 个 batch 时仍做一次参数更新（梯度按实际累积数缩放）
        if accum_batch_idx > 0:
            scaler.unscale_(optimizer)
            scale_up = ga / float(accum_batch_idx)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale_up)
            if tc.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer_step += 1
            accum_batch_idx = 0
            if ema is not None:
                ema.update(model, step=optimizer_step)

        # ── epoch summary ────────────────────────────────────────
        avg_train = epoch_loss / max(epoch_steps, 1)
        epoch_elapsed = time.time() - t_epoch
        total_elapsed = time.time() - train_start_wall
        if _is_main(rank):
            remaining_epochs = tc.epochs - epoch - 1
            epoch_eta = remaining_epochs * epoch_elapsed
            print(
                f"\n[{_now()}] ── epoch {epoch}/{tc.epochs-1} done ──  "
                f"train_loss={avg_train:.5f}  "
                f"steps={epoch_steps}  "
                f"epoch_time={_elapsed_str(epoch_elapsed)}  "
                f"total={_elapsed_str(total_elapsed)}  "
                f"ETA≈{_eta_str(epoch_eta)}  "
                f"best_pp={best_corr_pert:.4f}  "
                f"GPU={_gpu_mem_mb()}\n",
                flush=True,
            )

        # ── epoch-level: full test + early stopping ──────────────
        if (
            tc.test_every_epoch > 0
            and (epoch + 1) % tc.test_every_epoch == 0
            and selection_ds is not None
            and test_tasks
        ):
            if _is_main(rank):
                print(f"[{_now()}] === {selection_tag} (epoch {epoch}) ===", flush=True)
            # 评估时 swap 到 EMA 权重（若启用且过了 update_after）
            if ema is not None and optimizer_step >= tc.ema_update_after:
                with ema.apply_to(model):
                    test_res = _run_eval(
                        model, selection_ds, device, tc, latent_fm, use_amp, amp_dt,
                        rank, world_size, tasks=test_tasks, tag=selection_tag,
                        step=global_step, per_dataset=True,
                        max_cells=tc.test_max_cells,
                        ode_steps=tc.eval_ode_steps,
                        ctrl_means=ctrl_means, pert_means=pert_means,
                        cfg_w=getattr(cfg.inference, "cfg_w", 1.0),
                        use_residual_flow=getattr(tc, "use_residual_flow", False),
                        max_pert_genes=int(getattr(mc, "max_pert_genes", 16)),
                    )
            else:
                test_res = _run_eval(
                    model, selection_ds, device, tc, latent_fm, use_amp, amp_dt,
                    rank, world_size, tasks=test_tasks, tag=selection_tag,
                    step=global_step, per_dataset=True,
                    max_cells=tc.test_max_cells,
                    ode_steps=tc.eval_ode_steps,
                    ctrl_means=ctrl_means, pert_means=pert_means,
                    cfg_w=getattr(cfg.inference, "cfg_w", 1.0),
                    use_residual_flow=getattr(tc, "use_residual_flow", False),
                    max_pert_genes=int(getattr(mc, "max_pert_genes", 16)),
                )

            improved = False
            if _is_main(rank) and test_res is not None:
                pd_ctrl = test_res["global"]["pearson_delta_ctrl"]
                corr_pert = test_res["global"].get("corr_pert_mean", float("-inf"))
                metric_value = _select_metric_value(
                    test_res, tc.selection_metric,
                    mmd_lambda=getattr(tc, "selection_mmd_lambda", 0.5),
                )
                if tc.selection_metric in ("corr_pert_mean", "corr_minus_mmd"):
                    try:
                        _cpv = float(test_res["global"].get("corr_pert_mean", float("nan")))
                    except (TypeError, ValueError):
                        _cpv = float("nan")
                    if math.isnan(_cpv):
                        print(
                            f"[{_now()}] TEST: corr_pert_mean is NaN (no gene-space pert mean); "
                            f"selection_metric={tc.selection_metric!r} cannot improve best on this signal",
                            flush=True,
                        )
                best_metric = best_selection_score
                record = {
                    "global_step": global_step, "epoch": epoch,
                    "train_loss": avg_train, "lr": lr,
                    "eval_type": "test" if selection_ds is test_ds else "selection",
                    **{f"eval_{k}": v for k, v in test_res["global"].items()},
                }
                with open(log_file, "a") as f:
                    f.write(json.dumps(record) + "\n")

                allow_patience = (
                    epoch + 1 >= tc.min_epochs_before_stop
                    and epoch >= tc.loss_guard_epochs
                )

                if metric_value > best_metric:
                    best_pd_ctrl = pd_ctrl
                    best_corr_pert = corr_pert
                    best_selection_score = metric_value
                    no_improve_count = 0
                    improved = True
                    scored_ema = ema is not None and optimizer_step >= tc.ema_update_after
                    if scored_ema:
                        with ema.apply_to(model):
                            m_save = model.module if use_ddp else model
                            _save(
                                m_save, optimizer, epoch, global_step,
                                float("inf"), out_dir / "best.pt",
                                best_pd_ctrl=best_pd_ctrl,
                                best_corr_pert=best_corr_pert,
                                best_selection_score=best_selection_score,
                                no_improve_count=no_improve_count,
                                optimizer_step=optimizer_step,
                                accum_batch_idx=accum_batch_idx,
                                ema=ema,
                                selection_metric=tc.selection_metric,
                                scored_with_ema=True,
                                ema_update_after=tc.ema_update_after,
                                latent_fm_ckpt_path=_latent_fm_ckpt_s,
                                model_keys_digest=_model_state_keys_digest(m_save),
                                git_commit=_save_git,
                                torch_version=_save_torch_v,
                            )
                    else:
                        raw_model = model.module if use_ddp else model
                        _save(
                            raw_model, optimizer, epoch, global_step,
                            float("inf"), out_dir / "best.pt",
                            best_pd_ctrl=best_pd_ctrl,
                            best_corr_pert=best_corr_pert,
                            best_selection_score=best_selection_score,
                            no_improve_count=no_improve_count,
                            optimizer_step=optimizer_step,
                            accum_batch_idx=accum_batch_idx,
                            ema=ema,
                            selection_metric=tc.selection_metric,
                            scored_with_ema=False,
                            ema_update_after=tc.ema_update_after,
                            latent_fm_ckpt_path=_latent_fm_ckpt_s,
                            model_keys_digest=_model_state_keys_digest(raw_model),
                            git_commit=_save_git,
                            torch_version=_save_torch_v,
                        )
                    print(
                        f"  [{_now()}] ★ new best {tc.selection_metric}="
                        f"{metric_value:.5f} | pp={best_corr_pert:.5f}",
                        flush=True,
                    )
                else:
                    if allow_patience and patience_enabled:
                        no_improve_count += 1
                        print(
                            f"  [{_now()}] no improvement "
                            f"({no_improve_count}/{tc.early_stop_patience}) | "
                            f"best pp={best_corr_pert:.5f}",
                            flush=True,
                        )
                    elif not patience_enabled:
                        print(
                            f"  [{_now()}] no improvement; early stopping disabled "
                            f"(early_stop_patience={tc.early_stop_patience}) | "
                            f"pp={corr_pert:.5f}",
                            flush=True,
                        )
                    else:
                        print(
                            f"  [{_now()}] no improvement, but patience is guarded "
                            f"until epoch>={tc.loss_guard_epochs} and "
                            f"epoch+1>={tc.min_epochs_before_stop}; "
                            f"pp={corr_pert:.5f}",
                            flush=True,
                        )

            if use_ddp:
                stop_tensor = torch.tensor(
                    [1 if (
                        patience_enabled
                        and not improved
                        and no_improve_count >= tc.early_stop_patience
                    )
                     else 0],
                    device=device,
                )
                if not _is_main(rank):
                    stop_tensor.zero_()
                dist.broadcast(stop_tensor, src=0)
                if stop_tensor.item() == 1:
                    early_stopped = True
            else:
                if patience_enabled and no_improve_count >= tc.early_stop_patience:
                    early_stopped = True

            if use_ddp:
                sync = torch.tensor(
                    [best_pd_ctrl, best_corr_pert, best_selection_score, float(no_improve_count)], device=device,
                )
                dist.broadcast(sync, src=0)
                best_pd_ctrl = sync[0].item()
                best_corr_pert = sync[1].item()
                best_selection_score = sync[2].item()
                no_improve_count = int(sync[3].item())

            model.train()

        # ── save last checkpoint every epoch ─────────────────────
        if _is_main(rank):
            raw_model = model.module if use_ddp else model
            _save(raw_model, optimizer, epoch, global_step,
                  float("inf"), out_dir / "last.pt",
                  best_pd_ctrl=best_pd_ctrl,
                  best_corr_pert=best_corr_pert,
                  best_selection_score=best_selection_score,
                  no_improve_count=no_improve_count,
                  optimizer_step=optimizer_step,
                  accum_batch_idx=accum_batch_idx,
                  ema=ema,
                  selection_metric=tc.selection_metric,
                  scored_with_ema=bool(ema is not None and optimizer_step >= tc.ema_update_after),
                  ema_update_after=tc.ema_update_after,
                  latent_fm_ckpt_path=_latent_fm_ckpt_s,
                  model_keys_digest=_model_state_keys_digest(raw_model),
                  git_commit=_save_git,
                  torch_version=_save_torch_v)

        if early_stopped:
            if _is_main(rank):
                total_time = _elapsed_str(time.time() - train_start_wall)
                print(f"\n[{_now()}] EARLY STOPPING at epoch {epoch} "
                      f"(no improvement for {no_improve_count} full-test evals "
                      f"= {no_improve_count * tc.test_every_epoch} epochs). "
                      f"Total training time: {total_time}",
                      flush=True)
            break

    run_final_eval = (
        bool(getattr(tc, "run_final_test", True))
        and protocol == "fixed_steps_no_selection"
        and bool(final_eval_tasks)
    )
    if run_final_eval:
        if _is_main(rank):
            print(f"\n[{_now()}] === FINAL_TEST (fixed-step, no checkpoint selection) ===", flush=True)
        if ema is not None and optimizer_step >= tc.ema_update_after:
            with ema.apply_to(model):
                final_res = _run_eval(
                    model, test_ds, device, tc, latent_fm, use_amp, amp_dt,
                    rank, world_size, tasks=final_eval_tasks, tag="FINAL_TEST",
                    step=global_step, per_dataset=True,
                    max_cells=tc.test_max_cells,
                    ode_steps=tc.eval_ode_steps,
                    ctrl_means=ctrl_means, pert_means=pert_means,
                    cfg_w=getattr(cfg.inference, "cfg_w", 1.0),
                    use_residual_flow=getattr(tc, "use_residual_flow", False),
                    max_pert_genes=int(getattr(mc, "max_pert_genes", 16)),
                )
        else:
            final_res = _run_eval(
                model, test_ds, device, tc, latent_fm, use_amp, amp_dt,
                rank, world_size, tasks=final_eval_tasks, tag="FINAL_TEST",
                step=global_step, per_dataset=True,
                max_cells=tc.test_max_cells,
                ode_steps=tc.eval_ode_steps,
                ctrl_means=ctrl_means, pert_means=pert_means,
                cfg_w=getattr(cfg.inference, "cfg_w", 1.0),
                use_residual_flow=getattr(tc, "use_residual_flow", False),
                max_pert_genes=int(getattr(mc, "max_pert_genes", 16)),
            )
        if _is_main(rank) and final_res is not None:
            record = {
                "global_step": global_step,
                "epoch": epoch if "epoch" in locals() else None,
                "train_loss": None,
                "lr": lr,
                "eval_type": "final_test",
                **{f"eval_{k}": v for k, v in final_res["global"].items()},
            }
            with open(log_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        model.train()

    train_ds.close()
    if val_ds is not None and val_ds is not test_ds:
        val_ds.close()
    test_ds.close()

    if use_ddp:
        dist.destroy_process_group()

    if _is_main(rank):
        total_time = _elapsed_str(time.time() - train_start_wall)
        if protocol == "fixed_steps_no_selection":
            print(
                f"\n[{_now()}] Training complete (fixed-step/no-selection). "
                f"last checkpoint saved at {out_dir / 'last.pt'}, "
                f"total_time={total_time}",
                flush=True,
            )
        else:
            print(f"\n[{_now()}] Training complete. "
                  f"best_pp={best_corr_pert:.5f}, "
                  f"best_{tc.selection_metric}={best_selection_score:.5f}, "
                  f"total_time={total_time}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CoupledFM training")
    parser.add_argument("--mode", choices=["baseline", "ot", "coupled"],
                        default=None, help="Coupling mode (default: config, usually ot)")
    parser.add_argument(
        "--ot_feature", choices=["latent", "de", "raw"], default=None,
        help="Feature space for minibatch OT (default: config, usually de)",
    )
    parser.add_argument(
        "--de_dir", type=str, default=None,
        help="Directory with {dataset}.json DE gene lists (default: config RAW_DE_DIR)",
    )
    parser.add_argument(
        "--ot_method", type=str, default=None,
        help="OT backend: torch_sinkhorn | sinkhorn | exact (default: config)",
    )
    parser.add_argument("--ot_sinkhorn_reg", type=float, default=None)
    parser.add_argument("--ot_sinkhorn_iter", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None,
                        help="Linear LR warmup length in epochs (optimizer schedule)")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--global_ot_batch", type=int, default=None,
        help="DDP only: total OT pairs per step over all ranks; sets batch_size=global_ot_batch/world_size",
    )
    parser.add_argument("--attn_mode", choices=["diff", "self_only"], default=None)
    parser.add_argument(
        "--attn_backend",
        choices=["sdpa", "flash", "linear", "sparse"],
        default=None,
        help=(
            "Attention backend: sdpa (PyTorch SDPA, supports attn_bias), "
            "flash (flash-attn, falls back to sdpa if bias given), "
            "linear (ELU+1 kernel), "
            "sparse (CellNavi-style scatter on edge_index)."
        ),
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--micro_batch", type=int, default=None,
                        help="GPU micro-batch size (chunks per optimizer step)")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Subset of dataset names to load")
    parser.add_argument("--split_file", "--split-file", dest="split_file", type=str, default=None,
                        help="Explicit train/val/test split JSON; loaded read-only with provenance copy")
    parser.add_argument("--use_amp", action="store_true",
                        help="Enable mixed precision (Config default is already True)")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable AMP: full float32 matmul on GPU (no autocast)")
    parser.add_argument("--debug_nan", action="store_true",
                        help="Print first forward v_pred/dx stats and list params with NaN grads")
    parser.add_argument("--detect_anomaly", action="store_true",
                        help="torch.autograd.set_detect_anomaly(True); very slow")
    parser.add_argument("--fp64", action="store_true",
                        help="Train in float64 (implies --no_amp); slow and memory-heavy")
    parser.add_argument("--amp_dtype", choices=["float16", "bfloat16"], default=None,
                        help="AMP dtype")
    parser.add_argument("--val_every_steps", type=int, default=None,
                        help="Run fast val eval every N steps")
    parser.add_argument("--gene_budget_manifest", "--gene-budget-manifest",
                        dest="gene_budget_manifest", type=str, default=None,
                        help="Deterministic per-dataset raw gene keep-index manifest")
    parser.add_argument("--gene_budget_label", "--gene-budget-label",
                        dest="gene_budget_label", type=str, default=None,
                        help="Human-readable gene-budget/control label for provenance")
    parser.add_argument("--val_split_key", "--val-split-key", dest="val_split_key",
                        type=str, default=None,
                        help="Split key for training-time validation; 'auto' uses val when present")
    parser.add_argument("--test_split_key", "--test-split-key", dest="test_split_key",
                        type=str, default=None,
                        help="Split key reserved for final test evaluation")
    parser.add_argument("--max_train_steps_per_epoch", type=int, default=None,
                        help="Limit train steps per epoch for smoke tests (0 disables)")
    parser.add_argument("--val_ode_steps", type=int, default=None,
                        help="ODE steps used for fast val monitoring")
    parser.add_argument("--val_sample_ratio", type=float, default=None,
                        help="Fraction of test conditions sampled for val monitoring (default 0.2)")
    parser.add_argument("--val_max_per_ds", type=int, default=None,
                        help="Cap per-dataset conditions in val monitoring")
    parser.add_argument("--test_every_epoch", type=int, default=None,
                        help="Run full test eval every N epochs")
    parser.add_argument("--eval_ode_steps", type=int, default=None,
                        help="ODE steps used for epoch-end full test")
    parser.add_argument(
        "--val_ode_method", choices=["euler", "midpoint", "rk4"], default=None,
        help="ODE integrator for val/test (inference.integrate; default euler)",
    )
    parser.add_argument("--early_stop_patience", type=int, default=None,
                        help="Early stop after N full-test evals without improvement")
    parser.add_argument("--selection_protocol", "--selection-protocol",
                        dest="selection_protocol",
                        choices=["metric", "fixed_steps_no_selection", "fixed-steps-no-selection"],
                        default=None,
                        help="Checkpoint-selection protocol")
    parser.add_argument("--fixed_step_no_selection", "--fixed-step-no-selection",
                        dest="fixed_step_no_selection", action="store_true",
                        help="Convenience alias for --selection-protocol fixed_steps_no_selection")
    parser.add_argument("--no_initial_val", "--no-initial-val",
                        dest="no_initial_val", action="store_true",
                        help="Skip initial validation before training")
    parser.add_argument("--no_final_test", "--no-final-test",
                        dest="no_final_test", action="store_true",
                        help="Skip final-only test in fixed-step/no-selection mode")
    parser.add_argument("--selection_metric",
                        choices=["corr_pert_mean", "corr_minus_mmd", "pearson_delta_ctrl", "mmd"],
                        default=None,
                        help="Metric used for best checkpoint selection")
    parser.add_argument("--loss_guard_epochs", type=int, default=None,
                        help="Do not consume early-stop patience before this epoch")
    parser.add_argument("--min_epochs_before_stop", type=int, default=None,
                        help="Hard minimum epoch count before early stopping is allowed")
    parser.add_argument("--latent_z_mode", choices=["interp", "ode", "curriculum"], default=None,
                        help="Latent z_t computation: interp | ode | curriculum "
                             "(curriculum: early=interp, late=ode; 需要 --latent_fm_ckpt)")
    parser.add_argument("--latent_fm_ckpt", type=str, default=None,
                        help="Frozen Latent FM checkpoint path (required for ode / curriculum)")
    parser.add_argument("--curriculum_warmup_steps", type=int, default=None)
    parser.add_argument("--curriculum_anneal_steps", type=int, default=None)
    parser.add_argument("--curriculum_max_prob", type=float, default=None)
    # two-stage FT + param groups + EMA
    parser.add_argument("--two_stage_ft", action="store_true",
                        help="Stage 1: freeze CellNavi backbone, only train new modules; "
                             "Stage 2: unfreeze at two_stage_freeze_epochs with small backbone LR.")
    parser.add_argument("--no_two_stage_ft", action="store_true",
                        help="Explicitly disable two-stage FT (overrides config default).")
    parser.add_argument("--two_stage_freeze_epochs", type=int, default=None,
                        help="Epochs to keep backbone frozen (stage 1 length).")
    parser.add_argument("--stage2_backbone_mult", type=float, default=None,
                        help="Stage 2 backbone lr = lr * this mult (default 0.1).")
    parser.add_argument("--use_param_groups", action="store_true",
                        help="AdamW with separate (lr, wd) for backbone vs new modules.")
    parser.add_argument("--no_param_groups", action="store_true",
                        help="Disable param-groups (overrides config default).")
    parser.add_argument("--lr_new_module_mult", type=float, default=None,
                        help="new_modules lr = lr * this mult (default 3.0).")
    parser.add_argument("--weight_decay_backbone", type=float, default=None)
    parser.add_argument("--weight_decay_new", type=float, default=None)
    parser.add_argument("--min_lr_ratio", type=float, default=None,
                        help="cosine-with-min-lr floor: lr never drops below lr*ratio (default 0.1).")
    parser.add_argument("--adam_beta2", type=float, default=None,
                        help="AdamW beta2; FM/diffusion 常用 0.95。")
    parser.add_argument("--use_ema", action="store_true",
                        help="Enable EMA shadow weights (evaluations swap to EMA).")
    parser.add_argument("--no_ema", action="store_true",
                        help="Disable EMA (overrides config default).")
    parser.add_argument("--ema_decay", type=float, default=None)
    parser.add_argument("--ema_update_after", type=int, default=None)
    parser.add_argument("--mmd_micro_chunk", type=int, default=None,
                        help="OOM-safe MMD: split batch into chunks of this size for the 2nd forward (0=off).")
    pert_chem_g = parser.add_mutually_exclusive_group()
    pert_chem_g.add_argument(
        "--pert_chem_enabled",
        action="store_true",
        help="Enable chemical embedding slots (DataConfig gate; aligns with upstream coupled).",
    )
    pert_chem_g.add_argument(
        "--no_pert_chem_enabled",
        action="store_true",
        help="Disable chemical perturbation embeddings.",
    )
    parser.add_argument(
        "--legacy_json_split",
        action="store_true",
        help="Obsolete: canonical split JSON is always biflow_dir/split_seed{seed}.json.",
    )
    parser.add_argument(
        "--ot_emb_cap_src", type=int, default=None,
        help="Max source-pool cells for OT EMD. ≤0 = no cap (None → batch_size).",
    )
    parser.add_argument(
        "--ot_emb_cap_ir", type=int, default=None,
        help="Deprecated: use --ot_emb_cap_src.",
    )
    parser.add_argument(
        "--ot_emb_cap_gt", type=int, default=None,
        help="Max GT cells for latent OT EMD (default config 128). 0 = no cap.",
    )
    parser.add_argument(
        "--grad_accum_steps", type=int, default=None,
        help="Optimizer step every N data batches (default 1). LR cosine uses opt steps.",
    )
    parser.add_argument("--use_mmd", action="store_true",
                        help="Enable MMD regularization explicitly")
    parser.add_argument("--no_mmd", action="store_true",
                        help="Disable MMD regularization")
    parser.add_argument("--mmd_gamma_max", type=float, default=None)
    parser.add_argument("--mmd_warmup_start", type=int, default=None)
    parser.add_argument("--mmd_warmup_end", type=int, default=None)
    parser.add_argument("--mmd_warmup_start_frac", type=float, default=None,
                        help="若与 end_frac 同时设置则覆盖绝对 warmup（相对 total_opt_steps）")
    parser.add_argument("--mmd_warmup_end_frac", type=float, default=None)
    parser.add_argument("--mmd_no_rel_frac", action="store_true",
                        help="禁用相对 MMD warmup（不根据 total_opt_steps 覆写 start/end）")
    parser.add_argument("--no_ema_dynamic", action="store_true",
                        help="关闭 EMA 动态 decay")
    parser.add_argument("--mmd_every", type=int, default=None)
    parser.add_argument("--mmd_epoch_start", type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.mode is not None:
        cfg.train.coupling_mode = args.mode
    if args.ot_feature is not None:
        cfg.train.ot_feature = args.ot_feature
    if args.de_dir is not None:
        cfg.train.de_dir = args.de_dir
    if args.ot_method is not None:
        cfg.train.ot_method = args.ot_method
    if args.ot_sinkhorn_reg is not None:
        cfg.train.ot_sinkhorn_reg = args.ot_sinkhorn_reg
    if args.ot_sinkhorn_iter is not None:
        cfg.train.ot_sinkhorn_iter = args.ot_sinkhorn_iter
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.warmup_epochs is not None:
        cfg.train.warmup_epochs = args.warmup_epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.global_ot_batch is not None:
        cfg.train.global_ot_batch = args.global_ot_batch
    if args.attn_mode is not None:
        cfg.model.attn_mode = args.attn_mode
    if args.attn_backend is not None:
        cfg.model.attn_backend = args.attn_backend
    if args.resume is not None:
        cfg.train.resume_from = args.resume
    if args.device is not None:
        cfg.train.device = args.device
    if args.output_dir is not None:
        cfg.train.output_dir = args.output_dir
    if args.micro_batch is not None:
        cfg.train.micro_batch = args.micro_batch
    if args.datasets is not None:
        cfg.data.datasets = args.datasets
    if args.split_file is not None:
        cfg.data.split_file = args.split_file
    _amp_explicit = False   # 用户是否显式指定了精度
    if args.use_amp:
        cfg.train.use_amp = True
        _amp_explicit = True
    if args.no_amp:
        cfg.train.use_amp = False
        _amp_explicit = True
    if args.fp64:
        cfg.train.fp64_training = True
        cfg.train.use_amp = False
        _amp_explicit = True
    if args.debug_nan:
        cfg.train.debug_nan = True
    if args.detect_anomaly:
        cfg.train.detect_anomaly = True
    if args.amp_dtype is not None:
        cfg.train.amp_dtype = args.amp_dtype
    if args.val_every_steps is not None:
        cfg.train.val_every_steps = args.val_every_steps
    if args.gene_budget_manifest is not None:
        cfg.train.gene_budget_manifest_path = args.gene_budget_manifest
    if args.gene_budget_label is not None:
        cfg.train.gene_budget_label = args.gene_budget_label
    if args.val_split_key is not None:
        cfg.train.val_split_key = args.val_split_key
    if args.test_split_key is not None:
        cfg.train.test_split_key = args.test_split_key
    if args.max_train_steps_per_epoch is not None:
        cfg.train.max_train_steps_per_epoch = args.max_train_steps_per_epoch
    if args.val_ode_steps is not None:
        cfg.train.val_ode_steps = args.val_ode_steps
    if args.val_sample_ratio is not None:
        cfg.train.val_sample_ratio = args.val_sample_ratio
    if args.val_max_per_ds is not None:
        cfg.train.val_max_per_ds = args.val_max_per_ds
    if args.test_every_epoch is not None:
        cfg.train.test_every_epoch = args.test_every_epoch
    if args.eval_ode_steps is not None:
        cfg.train.eval_ode_steps = args.eval_ode_steps
    if args.val_ode_method is not None:
        cfg.train.val_ode_method = args.val_ode_method
    if args.early_stop_patience is not None:
        cfg.train.early_stop_patience = args.early_stop_patience
    if args.selection_protocol is not None:
        cfg.train.selection_protocol = args.selection_protocol.replace("-", "_")
    if args.fixed_step_no_selection:
        cfg.train.selection_protocol = "fixed_steps_no_selection"
    if args.no_initial_val:
        cfg.train.run_initial_val = False
    if args.no_final_test:
        cfg.train.run_final_test = False
    if args.selection_metric is not None:
        cfg.train.selection_metric = args.selection_metric
    if args.loss_guard_epochs is not None:
        cfg.train.loss_guard_epochs = args.loss_guard_epochs
    if args.min_epochs_before_stop is not None:
        cfg.train.min_epochs_before_stop = args.min_epochs_before_stop
    if args.latent_z_mode is not None:
        cfg.train.latent_z_mode = args.latent_z_mode
    if args.latent_fm_ckpt is not None:
        cfg.train.latent_fm_ckpt = args.latent_fm_ckpt
    if args.curriculum_warmup_steps is not None:
        cfg.train.curriculum_warmup_steps = args.curriculum_warmup_steps
    if args.curriculum_anneal_steps is not None:
        cfg.train.curriculum_anneal_steps = args.curriculum_anneal_steps
    if args.curriculum_max_prob is not None:
        cfg.train.curriculum_max_prob = args.curriculum_max_prob
    if args.two_stage_ft:
        cfg.train.two_stage_ft = True
    if args.no_two_stage_ft:
        cfg.train.two_stage_ft = False
    if args.two_stage_freeze_epochs is not None:
        cfg.train.two_stage_freeze_epochs = args.two_stage_freeze_epochs
    if args.stage2_backbone_mult is not None:
        cfg.train.stage2_backbone_mult = args.stage2_backbone_mult
    if args.use_param_groups:
        cfg.train.use_param_groups = True
    if args.no_param_groups:
        cfg.train.use_param_groups = False
    if args.lr_new_module_mult is not None:
        cfg.train.lr_new_module_mult = args.lr_new_module_mult
    if args.weight_decay_backbone is not None:
        cfg.train.weight_decay_backbone = args.weight_decay_backbone
    if args.weight_decay_new is not None:
        cfg.train.weight_decay_new = args.weight_decay_new
    if args.min_lr_ratio is not None:
        cfg.train.min_lr_ratio = args.min_lr_ratio
    if args.adam_beta2 is not None:
        cfg.train.adam_beta2 = args.adam_beta2
    if args.use_ema:
        cfg.train.use_ema = True
    if args.no_ema:
        cfg.train.use_ema = False
    if args.ema_decay is not None:
        cfg.train.ema_decay = args.ema_decay
    if args.ema_update_after is not None:
        cfg.train.ema_update_after = args.ema_update_after
    if args.mmd_micro_chunk is not None:
        cfg.train.mmd_micro_chunk = args.mmd_micro_chunk
    if getattr(args, "pert_chem_enabled", False):
        cfg.data.pert_chem_enabled = True
    if getattr(args, "no_pert_chem_enabled", False):
        cfg.data.pert_chem_enabled = False
    if args.legacy_json_split:
        warnings.warn(
            "--legacy_json_split is obsolete; biflow canonical split_json is always used.",
            UserWarning,
            stacklevel=1,
        )
    _ot_cap_src = args.ot_emb_cap_src
    if args.ot_emb_cap_ir is not None:
        warnings.warn(
            "--ot_emb_cap_ir is deprecated; use --ot_emb_cap_src",
            DeprecationWarning,
            stacklevel=1,
        )
        if _ot_cap_src is not None and _ot_cap_src != args.ot_emb_cap_ir:
            parser.error("Conflicting --ot_emb_cap_src and --ot_emb_cap_ir")
        _ot_cap_src = args.ot_emb_cap_ir
    if _ot_cap_src is not None:
        cfg.train.ot_emb_cap_src = None if _ot_cap_src <= 0 else _ot_cap_src
    if args.ot_emb_cap_gt is not None:
        cfg.train.ot_emb_cap_gt = None if args.ot_emb_cap_gt <= 0 else args.ot_emb_cap_gt
    if args.grad_accum_steps is not None:
        cfg.train.grad_accum_steps = max(1, args.grad_accum_steps)
    if args.use_mmd:
        cfg.train.use_mmd = True
    if args.no_mmd:
        cfg.train.use_mmd = False
    if args.mmd_gamma_max is not None:
        cfg.train.mmd_gamma_max = args.mmd_gamma_max
    if args.mmd_warmup_start is not None:
        cfg.train.mmd_warmup_start = args.mmd_warmup_start
    if args.mmd_warmup_end is not None:
        cfg.train.mmd_warmup_end = args.mmd_warmup_end
    if args.mmd_no_rel_frac:
        cfg.train.mmd_warmup_start_frac = None
        cfg.train.mmd_warmup_end_frac = None
    if args.mmd_warmup_start_frac is not None:
        cfg.train.mmd_warmup_start_frac = args.mmd_warmup_start_frac
    if args.mmd_warmup_end_frac is not None:
        cfg.train.mmd_warmup_end_frac = args.mmd_warmup_end_frac
    if args.no_ema_dynamic:
        cfg.train.ema_dynamic = False
    if args.mmd_every is not None:
        cfg.train.mmd_every = args.mmd_every
    if args.mmd_epoch_start is not None:
        cfg.train.mmd_epoch_start = args.mmd_epoch_start

    train(cfg, _amp_explicit=_amp_explicit)
