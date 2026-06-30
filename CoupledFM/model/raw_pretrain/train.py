#!/usr/bin/env python3
"""cellgene_census pairwise raw FM pretraining — standalone DDP loop (plan v2)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from model.models.velocity_field import RawExprVelocityField
from model.raw_pretrain.config import RawPretrainConfig
from model.raw_pretrain.data_source import discover_h5ad_shards, write_shard_summaries
from model.raw_pretrain.dataset import PairwisePretrainDataset
from model.raw_pretrain.direction_adapter import PretrainDirectionAdapter
from model.raw_pretrain.losses import compute_loss_weight, endpoint_loss, velocity_loss
from model.train import _elapsed_str, _eta_str, _gpu_mem_mb, _is_main, _now
from model.utils.data.vocab import GeneVocab
from model.utils.train.ema import ModelEMA
from model.utils.train.schedulers import lr_warmup_cosine_ratio_floor_absolute
from model.utils.train.time_sampling import sample_t_torch


class PretrainBundle(torch.nn.Module):
    """Single ``nn.Module`` for DDP (velocity + adapter)."""

    def __init__(self, velocity: RawExprVelocityField, adapter: PretrainDirectionAdapter):
        super().__init__()
        self.velocity = velocity
        self.adapter = adapter

    def forward(
        self,
        x_t: torch.Tensor,
        x_ctrl: torch.Tensor,
        t: torch.Tensor,
        gene_ids_1d: torch.Tensor,
        pert_gene_ids: torch.Tensor,
        pert_signs: torch.Tensor,
        pert_mags: torch.Tensor,
        pert_mask: torch.Tensor,
    ) -> torch.Tensor:
        cond_vec = self.adapter(pert_gene_ids, pert_signs, pert_mags, pert_mask)
        return self.velocity(
            x_t,
            x_ctrl,
            t,
            gene_ids=gene_ids_1d,
            aux_emb=None,
            gene_mask=None,
            pert_idx=None,
            cond_vec=cond_vec,
            edge_index=None,
        )


def _unwrap(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if isinstance(m, DDP) else m


def _unique_params(*modules: torch.nn.Module) -> List[torch.nn.Parameter]:
    seen: set[int] = set()
    out: List[torch.nn.Parameter] = []
    for m in modules:
        for p in m.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
    return out


def _prepare_loader(cfg: RawPretrainConfig, vocab: GeneVocab, rank: int, ws: int):
    shards = discover_h5ad_shards(
        cfg.processed_dir,
        vocab,
        num_bins=cfg.num_bins,
        strict_same_genes=cfg.strict_same_genes,
        tissue_metainfo_path=cfg.tissue_metainfo_path,
        gene_symbol_column=cfg.gene_symbol_column,
        min_gene_hit_rate=cfg.min_gene_hit_rate,
    )
    ds = PairwisePretrainDataset(
        shards,
        rank=rank,
        world_size=ws,
        max_pert_genes=cfg.max_pert_genes,
        cond_tau=cfg.cond_tau,
        cond_alpha=cfg.cond_alpha,
        pseudo_delta_min=cfg.pseudo_delta_min,
        batch_size=cfg.batch_size,
        seed=cfg.seed + rank,
    )
    nw = int(os.environ.get("PRETRAIN_NUM_WORKERS", "0"))
    return ds, DataLoader(
        ds,
        batch_size=None,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
    )


def _autocast(use_amp: bool, amp_dt: torch.dtype):
    if not use_amp or not torch.cuda.is_available():
        return torch.amp.autocast(device_type="cuda", enabled=False)
    return torch.amp.autocast(device_type="cuda", dtype=amp_dt, enabled=True)


def train_once(cfg: RawPretrainConfig) -> None:
    use_dist = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if use_dist:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = int(os.environ["RANK"])
        ws = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    else:
        rank, ws, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)

    vocab = GeneVocab(str(cfg.gene_name_path), str(cfg.nichenet_node2idx_path))
    dataset, loader = _prepare_loader(cfg, vocab, rank, ws)

    velocity = RawExprVelocityField(
        d_model=cfg.d_model,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        attn_backend=cfg.attn_backend,
        coupling_mode="baseline",
        use_pert_condition=False,
        use_pert_token=False,
        gene_embedding_cache=None,
        legacy_cond_vec_dim=int(cfg.adapter_cond_dim),
        graph_bias_mode="none",
        value_encoder="linear",
    ).to(device)
    if velocity.cond_vec_proj is not None:
        torch.nn.init.xavier_uniform_(velocity.cond_vec_proj.weight)
        torch.nn.init.zeros_(velocity.cond_vec_proj.bias)

    if cfg.pretrained_ckpt.is_file():
        velocity.load_pretrained_weights(str(cfg.pretrained_ckpt), verbose=_is_main(rank))
    elif _is_main(rank):
        print(f"[{_now()}] WARNING: missing pretrained_ckpt: {cfg.pretrained_ckpt}", flush=True)

    adapter = PretrainDirectionAdapter(
        velocity.embed_gene,
        cfg.d_model,
        d_cond=cfg.adapter_cond_dim,
        n_heads=cfg.adapter_n_heads,
    ).to(device)

    bundle = PretrainBundle(velocity, adapter).to(device)
    if use_dist:
        bundle = DDP(
            bundle,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    u_vel = velocity
    u_adapt = adapter
    params = _unique_params(u_vel, u_adapt)
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    amp_dt = torch.bfloat16 if str(cfg.amp_dtype).lower() in ("bfloat16", "bf16") else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(cfg.use_amp and amp_dt == torch.float16 and device.type == "cuda"),
    )

    ema: ModelEMA | None = None
    if cfg.use_ema:
        ema = ModelEMA(
            bundle,
            decay=cfg.ema_decay,
            update_after=cfg.ema_update_after,
            dynamic=cfg.ema_dynamic,
        )

    out_dir = Path(cfg.output_dir).expanduser().resolve()
    if _is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(
            json.dumps(
                {k: str(v) if isinstance(v, Path) else v for k, v in vars(cfg).items()},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        write_shard_summaries(dataset.sources, out_dir / "data_summary.json")
        for rec in (s.schema_summary() for s in dataset.sources):
            print(
                f"[{_now()}] data shard {rec['name']}: "
                f"shape=({rec['n_obs']}, {rec['n_vars']}) "
                f"gene_hit={rec['gene_hit_count']}/{rec['n_vars']} "
                f"source={rec['gene_symbol_source']}",
                flush=True,
            )

    metrics_path = out_dir / "metrics_log.jsonl"
    ga = max(1, int(cfg.grad_accum_steps))
    steps_per_epoch = int(cfg.steps_per_epoch)
    total_micro_steps = steps_per_epoch * int(cfg.epochs)
    total_opt_steps = max(1, (total_micro_steps + ga - 1) // ga)
    warmup_opt_steps = max(1, int(cfg.warmup_steps))

    global_step = 0
    optimizer_step = 0
    train_start_wall = time.time()
    best_train = float("inf")
    accum_slot = 0

    for epoch in range(int(cfg.epochs)):
        bundle.train()
        epoch_loss = 0.0
        n_batches = 0
        t_epoch = time.time()

        it = iter(loader)
        for _step_in_epoch in range(steps_per_epoch):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                try:
                    batch = next(it)
                except StopIteration:
                    if _is_main(rank):
                        print(f"[{_now()}] dataset exhausted early at epoch {epoch}", flush=True)
                    break

            lr_now = lr_warmup_cosine_ratio_floor_absolute(
                optimizer_step,
                warmup_opt_steps,
                total_opt_steps,
                cfg.lr,
                cfg.min_lr_ratio,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            x_ctrl = batch["x_ctrl"].to(device, non_blocking=True)
            x_gt = batch["x_gt"].to(device, non_blocking=True)
            gene_ids_1d = batch["gene_ids"].to(device, non_blocking=True)
            pg = batch["pert_gene_ids"].to(device, non_blocking=True)
            signs = batch["pert_signs"].to(device, non_blocking=True)
            mags = batch["pert_mags"].to(device, non_blocking=True)
            pmask = batch["pert_mask"].to(device, non_blocking=True)

            B, G = x_ctrl.shape
            gm = torch.zeros((B, G), device=device, dtype=x_ctrl.dtype)
            t = sample_t_torch(B, device, mode=cfg.time_sampling)
            t_col = t.view(-1, 1)
            x_t = (1.0 - t_col) * x_ctrl + t_col * x_gt
            if cfg.xt_noise_sigma_max > 0:
                sig = torch.rand(B, 1, device=device, dtype=x_t.dtype) * float(cfg.xt_noise_sigma_max)
                x_t = x_t + sig * torch.randn_like(x_t)

            dx = x_gt - x_ctrl
            tw = compute_loss_weight(t, mode=cfg.loss_weighting, snr_gamma=cfg.min_snr_gamma)

            if accum_slot == 0:
                optimizer.zero_grad(set_to_none=True)

            mb = max(1, int(cfg.micro_batch))
            chunk_losses: list[float] = []

            for s in range(0, B, mb):
                e = min(B, s + mb)
                w_chunk = (e - s) / float(B) / float(ga)
                ctx = _autocast(cfg.use_amp, amp_dt)
                with ctx:
                    v = bundle(
                        x_t[s:e],
                        x_ctrl[s:e],
                        t[s:e],
                        gene_ids_1d,
                        pg[s:e],
                        signs[s:e],
                        mags[s:e],
                        pmask[s:e],
                    )
                    lv = velocity_loss(
                        v,
                        dx[s:e],
                        gm[s:e],
                        loss_type=str(cfg.loss_type),
                        smooth_beta=float(cfg.smooth_l1_beta),
                        time_w=tw[s:e],
                    )
                    if cfg.loss_endpoint_weight > 0:
                        le = endpoint_loss(
                            x_t[s:e],
                            v,
                            t[s:e],
                            x_gt[s:e],
                            gm[s:e],
                            loss_type=str(cfg.loss_type),
                            smooth_beta=float(cfg.smooth_l1_beta),
                            time_w=tw[s:e],
                        )
                    else:
                        le = torch.zeros((), device=device, dtype=lv.dtype)
                    loss_mb = cfg.loss_velocity_weight * lv + cfg.loss_endpoint_weight * le

                if scaler.is_enabled():
                    scaler.scale(loss_mb * w_chunk).backward()
                else:
                    (loss_mb * w_chunk).backward()
                chunk_losses.append(float(loss_mb.detach().item()))

            accum_slot += 1
            flush_optimizer = accum_slot >= ga or (_step_in_epoch == steps_per_epoch - 1)
            if flush_optimizer:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                if accum_slot < ga:
                    scale_up = ga / float(accum_slot)
                    for p in params:
                        if p.grad is not None:
                            p.grad.mul_(scale_up)
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer_step += 1
                if ema is not None:
                    ema.update(bundle, step=optimizer_step)
                accum_slot = 0

            batch_loss = float(np.mean(chunk_losses))
            epoch_loss += batch_loss
            n_batches += 1
            global_step += 1

            if _is_main(rank) and cfg.log_every_steps > 0 and global_step % cfg.log_every_steps == 0:
                rec = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "train_loss": batch_loss,
                    "lr": lr_now,
                    "task": "raw_pretrain_pairwise",
                }
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                print(
                    f"[{_now()}] step={global_step} epoch={epoch} loss={batch_loss:.5f} lr={lr_now:.2e}",
                    flush=True,
                )

            if _is_main(rank) and cfg.ckpt_every_steps > 0 and global_step % cfg.ckpt_every_steps == 0:
                uv = _unwrap(bundle).velocity
                ua = _unwrap(bundle).adapter
                torch.save(uv.state_dict(), out_dir / f"backbone_step{global_step}.pt")
                torch.save(ua.state_dict(), out_dir / f"pretrain_adapter_step{global_step}.pt")
                if ema is not None:
                    torch.save(ema.state_dict(), out_dir / f"ema_step{global_step}.pt")

        avg_train = epoch_loss / max(n_batches, 1)
        epoch_elapsed = time.time() - t_epoch
        total_elapsed = time.time() - train_start_wall
        if avg_train < best_train:
            best_train = avg_train
        if _is_main(rank):
            rem = int(cfg.epochs) - epoch - 1
            print(
                f"\n[{_now()}] ── epoch {epoch}/{cfg.epochs - 1} done ──  "
                f"train_loss={avg_train:.5f}  "
                f"steps={n_batches}  "
                f"epoch_time={_elapsed_str(epoch_elapsed)}  "
                f"total={_elapsed_str(total_elapsed)}  "
                f"ETA≈{_eta_str(rem * epoch_elapsed)}  "
                f"best={best_train:.5f}  "
                f"GPU={_gpu_mem_mb()}\n",
                flush=True,
            )
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "global_step": global_step,
                            "epoch": epoch,
                            "train_loss": avg_train,
                            "eval_type": "epoch_summary",
                            "best_train_loss": best_train,
                        },
                    )
                    + "\n",
                )

    if _is_main(rank):
        uv = _unwrap(bundle).velocity
        ua = _unwrap(bundle).adapter
        torch.save(uv.state_dict(), out_dir / "backbone.pt")
        torch.save(ua.state_dict(), out_dir / "pretrain_adapter.pt")
        if ema is not None:
            torch.save(ema.state_dict(), out_dir / "ema.pt")
        print(f"[{_now()}] saved backbone.pt / pretrain_adapter.pt under {out_dir}", flush=True)

    if use_dist:
        dist.barrier()
        dist.destroy_process_group()

    for s in dataset.sources:
        ad = getattr(s, "_adata", None)
        if ad is not None:
            try:
                ad.file.close()
            except Exception:
                pass


def parse_cfg(argv: List[str] | None) -> RawPretrainConfig:
    ap = argparse.ArgumentParser(description="cellgene pairwise raw FM pretrain")
    ap.add_argument("--processed-dir", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--steps-per-epoch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ckpt-every-steps", type=int, default=None)
    ap.add_argument("--log-every-steps", type=int, default=None)
    ap.add_argument("--loss-type", choices=["mse", "smooth_l1"], default=None)
    ap.add_argument("--num-bins", type=int, default=None)
    ap.add_argument("--tissue-metainfo-path", type=str, default=None)
    ap.add_argument("--gene-symbol-column", type=str, default=None)
    ap.add_argument("--gene-name-path", type=str, default=None)
    ap.add_argument("--nichenet-node2idx-path", type=str, default=None)
    ap.add_argument("--pretrained-ckpt", type=str, default=None)
    ap.add_argument("--min-gene-hit-rate", type=float, default=None)
    ap.add_argument("--max-pert-genes", type=int, default=None)
    ap.add_argument("--pseudo-delta-min", type=float, default=None)
    ap.add_argument("--adapter-cond-dim", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = RawPretrainConfig()
    tissue_meta_explicit = False
    if args.processed_dir:
        cfg.processed_dir = Path(args.processed_dir)
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    elif os.environ.get("PRETRAIN_OUT_DIR"):
        cfg.output_dir = Path(os.environ["PRETRAIN_OUT_DIR"])

    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    elif os.environ.get("PRETRAIN_BATCH"):
        cfg.batch_size = int(os.environ["PRETRAIN_BATCH"])

    if args.micro_batch is not None:
        cfg.micro_batch = int(args.micro_batch)
    elif os.environ.get("PRETRAIN_MICRO_BATCH"):
        cfg.micro_batch = int(os.environ["PRETRAIN_MICRO_BATCH"])

    if args.epochs is not None:
        cfg.epochs = int(args.epochs)
    elif os.environ.get("PRETRAIN_EPOCHS"):
        cfg.epochs = int(os.environ["PRETRAIN_EPOCHS"])

    if args.steps_per_epoch is not None:
        cfg.steps_per_epoch = int(args.steps_per_epoch)
    elif os.environ.get("PRETRAIN_STEPS_PER_EPOCH"):
        cfg.steps_per_epoch = int(os.environ["PRETRAIN_STEPS_PER_EPOCH"])

    if args.lr is not None:
        cfg.lr = float(args.lr)
    elif os.environ.get("PRETRAIN_LR"):
        cfg.lr = float(os.environ["PRETRAIN_LR"])

    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.ckpt_every_steps is not None:
        cfg.ckpt_every_steps = int(args.ckpt_every_steps)
    elif os.environ.get("PRETRAIN_CKPT_EVERY"):
        cfg.ckpt_every_steps = int(os.environ["PRETRAIN_CKPT_EVERY"])

    if args.log_every_steps is not None:
        cfg.log_every_steps = int(args.log_every_steps)

    if args.loss_type is not None:
        cfg.loss_type = str(args.loss_type)
    elif os.environ.get("PRETRAIN_LOSS_TYPE"):
        cfg.loss_type = str(os.environ["PRETRAIN_LOSS_TYPE"])

    if args.num_bins is not None:
        cfg.num_bins = int(args.num_bins)
    elif os.environ.get("PRETRAIN_NUM_BINS"):
        cfg.num_bins = int(os.environ["PRETRAIN_NUM_BINS"])

    if args.tissue_metainfo_path:
        cfg.tissue_metainfo_path = Path(args.tissue_metainfo_path)
        tissue_meta_explicit = True
    elif os.environ.get("PRETRAIN_TISSUE_METAINFO"):
        cfg.tissue_metainfo_path = Path(os.environ["PRETRAIN_TISSUE_METAINFO"])
        tissue_meta_explicit = True
    elif not tissue_meta_explicit:
        cfg.tissue_metainfo_path = cfg.processed_dir / "tissue_metainfo.csv"

    if args.gene_symbol_column:
        cfg.gene_symbol_column = str(args.gene_symbol_column)
    elif os.environ.get("PRETRAIN_GENE_SYMBOL_COLUMN"):
        cfg.gene_symbol_column = str(os.environ["PRETRAIN_GENE_SYMBOL_COLUMN"])

    if args.gene_name_path:
        cfg.gene_name_path = Path(args.gene_name_path)
    elif os.environ.get("PRETRAIN_GENE_NAME_PATH"):
        cfg.gene_name_path = Path(os.environ["PRETRAIN_GENE_NAME_PATH"])

    if args.nichenet_node2idx_path:
        cfg.nichenet_node2idx_path = Path(args.nichenet_node2idx_path)
    elif os.environ.get("PRETRAIN_NICHENET_NODE2IDX_PATH"):
        cfg.nichenet_node2idx_path = Path(os.environ["PRETRAIN_NICHENET_NODE2IDX_PATH"])

    if args.pretrained_ckpt:
        cfg.pretrained_ckpt = Path(args.pretrained_ckpt)
    elif os.environ.get("PRETRAIN_CKPT"):
        cfg.pretrained_ckpt = Path(os.environ["PRETRAIN_CKPT"])

    if args.min_gene_hit_rate is not None:
        cfg.min_gene_hit_rate = float(args.min_gene_hit_rate)
    elif os.environ.get("PRETRAIN_MIN_GENE_HIT_RATE"):
        cfg.min_gene_hit_rate = float(os.environ["PRETRAIN_MIN_GENE_HIT_RATE"])

    if args.max_pert_genes is not None:
        cfg.max_pert_genes = int(args.max_pert_genes)
    elif os.environ.get("PRETRAIN_MAX_PERT_GENES"):
        cfg.max_pert_genes = int(os.environ["PRETRAIN_MAX_PERT_GENES"])

    if args.pseudo_delta_min is not None:
        cfg.pseudo_delta_min = float(args.pseudo_delta_min)
    elif os.environ.get("PRETRAIN_PSEUDO_DELTA_MIN"):
        cfg.pseudo_delta_min = float(os.environ["PRETRAIN_PSEUDO_DELTA_MIN"])

    if args.adapter_cond_dim is not None:
        cfg.adapter_cond_dim = int(args.adapter_cond_dim)
    elif os.environ.get("PRETRAIN_D_COND"):
        cfg.adapter_cond_dim = int(os.environ["PRETRAIN_D_COND"])

    if os.environ.get("PRETRAIN_COND_TAU"):
        cfg.cond_tau = float(os.environ["PRETRAIN_COND_TAU"])
    if os.environ.get("PRETRAIN_COND_ALPHA"):
        cfg.cond_alpha = float(os.environ["PRETRAIN_COND_ALPHA"])

    return cfg


def main(argv: List[str] | None = None) -> None:
    train_once(parse_cfg(argv))


if __name__ == "__main__":
    main()
