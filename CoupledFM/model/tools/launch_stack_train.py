#!/usr/bin/env python3
"""Launch stack-backbone training with explicit config overrides.

This wrapper is intentionally small: it only fills fields that the existing
train.py CLI does not expose consistently across ``coupled`` and
``coupled_independent``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from model import paths

ROOT = Path(__file__).resolve().parents[2]  # CoupledFM repo root (parent of ``model/``)


def _split_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value or "").replace(",", " ").split() if x.strip()]


def _parse_pool_ops(value: str) -> tuple[str, ...]:
    ops = tuple(_split_csv(value))
    return ops or ("mean",)


def _parse_pool_scales(value: str) -> tuple[float, ...]:
    vals = tuple(float(x) for x in _split_csv(value))
    return vals or (1.0,)


def _apply_common(cfg, args: argparse.Namespace) -> None:
    cfg.data.biflow_dir = str(Path(args.biflow_dir).expanduser())
    cfg.data.latent_backbone = args.latent_backbone
    cfg.data.split_seed = int(args.split_seed)
    if str(getattr(args, "split_file", "") or "").strip():
        cfg.data.split_file = str(Path(args.split_file).expanduser())
    cfg.data.datasets = list(args.datasets)
    cfg.data.use_h5ad_pert_metadata = True
    cfg.data.use_raw_cond = True
    cfg.data.pert_chem_enabled = args.data_kind in {"drug", "all"}
    cfg.data.max_chem_keys = int(args.max_chem_keys)

    cfg.model.use_pert_condition = True
    cfg.model.pert_embed_mode = args.pert_embed_mode
    cfg.model.max_pert_genes = int(args.max_pert_genes)
    cfg.model.pert_pool_aggregations = _parse_pool_ops(args.pert_pool_aggregations)
    cfg.model.pert_pool_scale_init = _parse_pool_scales(args.pert_pool_scale_init)
    cfg.model.pert_pool_fusion_mode = str(args.pert_pool_fusion_mode).lower().strip()
    cfg.model.pert_type_adapter_mode = str(args.pert_type_adapter_mode).lower().strip()
    ped = str(getattr(args, "pert_embed_cache_dir", "") or "").strip()
    if ped:
        cfg.data.pert_gene_emb_cache_dir = str(Path(ped).expanduser())
    pes = str(getattr(args, "pert_embed_source", "") or "").strip()
    if pes:
        cfg.model.pert_condition_embedding_source = pes
    cfg.model.pert_chem_emb_dim = int(args.pert_chem_emb_dim)
    if args.use_pert_token is not None:
        cfg.model.use_pert_token = bool(args.use_pert_token)

    cfg.train.coupling_mode = args.mode
    cfg.train.epochs = int(args.epochs)
    cfg.train.batch_size = int(args.batch_size)
    cfg.train.micro_batch = int(args.micro_batch)
    if args.ot_emb_cap_src is not None and hasattr(cfg.train, "ot_emb_cap_src"):
        cfg.train.ot_emb_cap_src = None if int(args.ot_emb_cap_src) <= 0 else int(args.ot_emb_cap_src)
    if args.ot_emb_cap_gt is not None and hasattr(cfg.train, "ot_emb_cap_gt"):
        cfg.train.ot_emb_cap_gt = None if int(args.ot_emb_cap_gt) <= 0 else int(args.ot_emb_cap_gt)
    if args.ot_sample_mode is not None and hasattr(cfg.train, "ot_sample_mode"):
        cfg.train.ot_sample_mode = str(args.ot_sample_mode)
    if args.ot_cost is not None and hasattr(cfg.train, "ot_cost"):
        cfg.train.ot_cost = str(args.ot_cost)
    if args.ot_sinkhorn_reg is not None and hasattr(cfg.train, "ot_sinkhorn_reg"):
        cfg.train.ot_sinkhorn_reg = float(args.ot_sinkhorn_reg)
    if args.ot_sinkhorn_iter is not None and hasattr(cfg.train, "ot_sinkhorn_iter"):
        cfg.train.ot_sinkhorn_iter = int(args.ot_sinkhorn_iter)
    cfg.train.lr = float(args.lr)
    if args.warmup_steps is not None and hasattr(cfg.train, "warmup_steps"):
        cfg.train.warmup_steps = int(args.warmup_steps)
    if args.min_lr_ratio is not None and hasattr(cfg.train, "min_lr_ratio"):
        cfg.train.min_lr_ratio = float(args.min_lr_ratio)
    cfg.train.output_dir = str(Path(args.output_dir).expanduser())
    cfg.train.latent_z_mode = str(args.latent_z_mode).strip().lower()
    if getattr(args, "latent_fm_ckpt", None) is not None:
        cfg.train.latent_fm_ckpt = str(args.latent_fm_ckpt).strip()
    cfg.train.val_every_steps = int(args.val_every_steps)
    if str(getattr(args, "gene_budget_manifest", "") or "").strip():
        cfg.train.gene_budget_manifest_path = str(Path(args.gene_budget_manifest).expanduser())
    cfg.train.gene_budget_label = str(getattr(args, "gene_budget_label", "") or "")
    cfg.train.val_split_key = str(args.val_split_key)
    cfg.train.max_train_steps_per_epoch = int(args.max_train_steps_per_epoch)
    cfg.train.selection_protocol = str(args.selection_protocol).replace("-", "_")
    if bool(getattr(args, "fixed_step_no_selection", False)):
        cfg.train.selection_protocol = "fixed_steps_no_selection"
    cfg.train.run_initial_val = not bool(args.no_initial_val)
    cfg.train.run_final_test = not bool(args.no_final_test)
    cfg.train.test_split_key = str(args.test_split_key)
    cfg.train.test_every_epoch = int(args.test_every_epoch)
    cfg.train.val_ode_steps = int(args.val_ode_steps)
    cfg.train.eval_ode_steps = int(args.eval_ode_steps)
    cfg.train.early_stop_patience = int(args.early_stop_patience)
    cfg.train.grad_accum_steps = int(args.grad_accum_steps)
    if args.gene_mask_prob is not None and hasattr(cfg.train, "gene_mask_prob"):
        cfg.train.gene_mask_prob = float(args.gene_mask_prob)
    if args.gene_mask_all_prob is not None and hasattr(cfg.train, "gene_mask_all_prob"):
        cfg.train.gene_mask_all_prob = float(args.gene_mask_all_prob)
    cfg.train.use_amp = bool(args.use_amp)
    cfg.train.amp_dtype = args.amp_dtype
    cfg.train.detect_anomaly = bool(args.detect_anomaly)
    cfg.train.debug_nan = bool(args.debug_nan)

    if args.selection_metric is not None and hasattr(cfg.train, "selection_metric"):
        cfg.train.selection_metric = str(args.selection_metric)
    if args.mmd_gamma_max is not None and hasattr(cfg.train, "mmd_gamma_max"):
        cfg.train.mmd_gamma_max = float(args.mmd_gamma_max)
    if getattr(args, "use_mmd", None) is not None and hasattr(cfg.train, "use_mmd"):
        cfg.train.use_mmd = bool(args.use_mmd)
    if args.mmd_every is not None and hasattr(cfg.train, "mmd_every"):
        cfg.train.mmd_every = int(args.mmd_every)
    if args.mmd_epoch_start is not None and hasattr(cfg.train, "mmd_epoch_start"):
        cfg.train.mmd_epoch_start = int(args.mmd_epoch_start)
    if args.mmd_micro_chunk is not None and hasattr(cfg.train, "mmd_micro_chunk"):
        cfg.train.mmd_micro_chunk = int(args.mmd_micro_chunk)
    if args.mmd_warmup_start_frac is not None and hasattr(cfg.train, "mmd_warmup_start_frac"):
        cfg.train.mmd_warmup_start_frac = float(args.mmd_warmup_start_frac)
    if args.mmd_warmup_end_frac is not None and hasattr(cfg.train, "mmd_warmup_end_frac"):
        cfg.train.mmd_warmup_end_frac = float(args.mmd_warmup_end_frac)

    # Optimizer / param-groups overrides
    if args.use_param_groups is not None and hasattr(cfg.train, "use_param_groups"):
        cfg.train.use_param_groups = bool(args.use_param_groups)
    if args.lr_new_module_mult is not None and hasattr(cfg.train, "lr_new_module_mult"):
        cfg.train.lr_new_module_mult = float(args.lr_new_module_mult)
    if args.weight_decay_backbone is not None and hasattr(cfg.train, "weight_decay_backbone"):
        cfg.train.weight_decay_backbone = float(args.weight_decay_backbone)
    if args.weight_decay_new is not None and hasattr(cfg.train, "weight_decay_new"):
        cfg.train.weight_decay_new = float(args.weight_decay_new)

    # Training-step recipe overrides
    if args.ds_alpha is not None and hasattr(cfg.train, "ds_alpha"):
        cfg.train.ds_alpha = float(args.ds_alpha)
    if args.cfg_drop_prob is not None and hasattr(cfg.train, "cfg_drop_prob"):
        cfg.train.cfg_drop_prob = float(args.cfg_drop_prob)
    if args.pert_idx_mode is not None and hasattr(cfg.train, "pert_idx_mode"):
        cfg.train.pert_idx_mode = str(args.pert_idx_mode)
    if args.time_sampling is not None and hasattr(cfg.train, "time_sampling"):
        cfg.train.time_sampling = str(args.time_sampling)
    if args.loss_weighting is not None and hasattr(cfg.train, "loss_weighting"):
        cfg.train.loss_weighting = str(args.loss_weighting)
    if args.min_snr_gamma is not None and hasattr(cfg.train, "min_snr_gamma"):
        cfg.train.min_snr_gamma = float(args.min_snr_gamma)


def _run_model(args: argparse.Namespace) -> None:
    from model.config import Config
    from model.train import train

    cfg = Config()
    _apply_common(cfg, args)
    cfg.train.ot_feature = str(args.ot_feature or "latent").lower()
    dd = str(args.de_dir or "").strip()
    cfg.train.de_dir = str(Path(dd).expanduser()) if dd else None
    train(cfg, _amp_explicit=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Launch stack-backbone CoupledFM training")
    p.add_argument(
        "--variant",
        choices=["model", "coupled", "coupled_independent"],
        default="model",
        help="model (default) or legacy aliases coupled / coupled_independent; all call model.train",
    )
    p.add_argument("--data-kind", choices=["gene", "drug", "all"], default="gene")
    p.add_argument(
        "--biflow-dir",
        default=str(paths.biflow_dir()),
        help="biFlow 根目录（含 control_{backbone}/gt_{backbone}）",
    )
    p.add_argument("--latent-backbone", default="stack")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--split-file", default="",
                   help="explicit train/val/test split JSON; loaded read-only")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--mode", choices=["baseline", "ot", "coupled"], default="ot")
    p.add_argument("--ot-feature", choices=["latent", "de", "raw"], default="raw")
    p.add_argument("--ot-emb-cap-src", type=int, default=None,
                   help="override train.ot_emb_cap_src; <=0 means full available pool")
    p.add_argument("--ot-emb-cap-gt", type=int, default=None,
                   help="override train.ot_emb_cap_gt; <=0 means full available pool")
    p.add_argument("--ot-sample-mode", choices=["assignment", "multinomial"], default=None,
                   help="override train.ot_sample_mode")
    p.add_argument("--ot-cost", choices=["l2", "cosine", "zscore_l2", "rank_l2"], default=None,
                   help="override train.ot_cost")
    p.add_argument("--ot-sinkhorn-reg", type=float, default=None,
                   help="override train.ot_sinkhorn_reg")
    p.add_argument("--ot-sinkhorn-iter", type=int, default=None,
                   help="override train.ot_sinkhorn_iter")
    p.add_argument(
        "--de-dir",
        default=str(paths.de_dir()),
        help="DE JSON 目录（ot-feature=de）",
    )
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--gene-mask-prob", type=float, default=None,
                   help="override train.gene_mask_prob; use 0 for deterministic budget-MMD smokes")
    p.add_argument("--gene-mask-all-prob", type=float, default=None,
                   help="override train.gene_mask_all_prob")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup-steps", type=int, default=None,
                   help="override train.warmup_steps; useful for short fixed-step smokes")
    p.add_argument("--min-lr-ratio", type=float, default=None,
                   help="override train.min_lr_ratio cosine LR floor")
    p.add_argument("--val-every-steps", type=int, default=500)
    p.add_argument("--gene-budget-manifest", default="",
                   help="deterministic per-dataset raw gene keep-index manifest")
    p.add_argument("--gene-budget-label", default="",
                   help="human-readable gene-budget/control label for provenance")
    p.add_argument("--val-split-key", default="auto",
                   help="split key for training-time validation; 'auto' uses val when present")
    p.add_argument("--max-train-steps-per-epoch", type=int, default=0)
    p.add_argument("--selection-protocol",
                   choices=["metric", "fixed_steps_no_selection", "fixed-steps-no-selection"],
                   default="metric")
    p.add_argument("--fixed-step-no-selection", action="store_true",
                   help="disable training-time selection/eval and save last.pt only")
    p.add_argument("--no-initial-val", action="store_true")
    p.add_argument("--no-final-test", action="store_true")
    p.add_argument("--test-split-key", default="test",
                   help="split key reserved for final test evaluation")
    p.add_argument("--test-every-epoch", type=int, default=1)
    p.add_argument("--val-ode-steps", type=int, default=10)
    p.add_argument("--eval-ode-steps", type=int, default=10)
    p.add_argument("--early-stop-patience", type=int, default=4)
    p.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--no-amp", dest="use_amp", action="store_false")
    p.set_defaults(use_amp=True)
    p.add_argument("--detect-anomaly", action="store_true")
    p.add_argument("--debug-nan", action="store_true")

    p.add_argument("--pert-embed-mode", default="pretrained_frozen")
    p.add_argument("--max-pert-genes", type=int, default=16)
    p.add_argument("--max-chem-keys", type=int, default=4)
    p.add_argument("--pert-chem-emb-dim", type=int, default=512)
    p.add_argument("--pert-pool-aggregations", default="mean,max,min")
    p.add_argument("--pert-pool-scale-init", default="1.0,0.5,0.5")
    p.add_argument(
        "--pert-pool-fusion-mode",
        choices=("sum", "concat_linear"),
        default="sum",
    )
    p.add_argument(
        "--pert-type-adapter-mode",
        choices=("scalar", "vector_scale", "vector_scale_gate"),
        default="scalar",
    )
    p.add_argument("--pert-embed-cache-dir", default="",
                   help="if set, overrides cfg.data.pert_gene_emb_cache_dir")
    p.add_argument("--pert-embed-source", default="",
                   help="metadata tag recorded in cfg.model.pert_condition_embedding_source")

    p.add_argument("--selection-metric", default=None,
                   help="overrides cfg.train.selection_metric (corr_pert_mean | corr_minus_mmd | pearson_delta_ctrl | mmd)")
    p.add_argument("--mmd-gamma-max", type=float, default=None,
                   help="overrides cfg.train.mmd_gamma_max")
    p.add_argument("--use-mmd", dest="use_mmd",
                   action="store_const", const=True, default=None,
                   help="enable MMD regularization")
    p.add_argument("--no-mmd", dest="use_mmd",
                   action="store_const", const=False,
                   help="disable MMD regularization")
    p.add_argument("--mmd-every", type=int, default=None,
                   help="overrides cfg.train.mmd_every (apply MMD every N steps)")
    p.add_argument("--mmd-epoch-start", type=int, default=None,
                   help="overrides cfg.train.mmd_epoch_start (epoch at which MMD starts)")
    p.add_argument("--mmd-micro-chunk", type=int, default=None,
                   help="overrides cfg.train.mmd_micro_chunk (sub-chunk for MMD backward, 0=off)")
    p.add_argument("--mmd-warmup-start-frac", type=float, default=None,
                   help="overrides cfg.train.mmd_warmup_start_frac")
    p.add_argument("--mmd-warmup-end-frac", type=float, default=None,
                   help="overrides cfg.train.mmd_warmup_end_frac")

    # P0 fixes — optimizer param groups (enable backbone vs new-module separation)
    p.add_argument("--use-param-groups", dest="use_param_groups",
                   action="store_const", const=True, default=None,
                   help="enable AdamW param groups (backbone vs new modules)")
    p.add_argument("--no-param-groups", dest="use_param_groups",
                   action="store_const", const=False,
                   help="explicitly disable AdamW param groups")
    p.add_argument("--lr-new-module-mult", type=float, default=None,
                   help="new-module lr = base_lr * mult (default 3.0; needs --use-param-groups)")
    p.add_argument("--weight-decay-backbone", type=float, default=None,
                   help="weight decay for pretrained backbone params (needs --use-param-groups)")
    p.add_argument("--weight-decay-new", type=float, default=None,
                   help="weight decay for newly initialized modules (needs --use-param-groups)")

    # P0 fixes — model conditioning channel
    p.add_argument("--no-pert-token", dest="use_pert_token",
                   action="store_const", const=False, default=None,
                   help="disable model.use_pert_token (pert_idx channel becomes a no-op)")
    p.add_argument("--use-pert-token", dest="use_pert_token",
                   action="store_const", const=True,
                   help="explicitly enable model.use_pert_token")

    # P0 fixes — dataset / sampling
    p.add_argument("--ds-alpha", type=float, default=None,
                   help="dataset subsampling exponent; 1.0 = visit all conds each epoch")
    p.add_argument("--cfg-drop-prob", type=float, default=None,
                   help="probability of dropping condition for CFG training (0=off)")
    p.add_argument("--pert-idx-mode", default=None,
                   choices=["zero", "random", "cond_hash"],
                   help="how to assign pert_idx for the pert_token channel")

    # P1 — flow-matching time / loss weighting recipe
    p.add_argument("--time-sampling", default=None,
                   choices=["uniform", "logit_normal", "lognormal"],
                   help="t sampler for flow-matching training")
    p.add_argument("--loss-weighting", default=None,
                   choices=["none", "uniform", "min_snr"],
                   help="per-sample loss weighting by flow time t")
    p.add_argument("--min-snr-gamma", type=float, default=None,
                   help="Min-SNR gamma (only used when --loss-weighting min_snr)")
    p.add_argument(
        "--latent-z-mode",
        choices=["interp", "ode", "curriculum"],
        default="interp",
        help="teacher-forcing / ODE / curriculum for coupled latent path (mainly "
        "applies when coupling_mode=coupled; preserved for sweep CLI consistency)",
    )
    p.add_argument(
        "--latent-fm-ckpt",
        default=None,
        help="FrozenLatentFM checkpoint path (sets cfg.train.latent_fm_ckpt; "
        "required for ode/curriculum when coupling_mode=coupled)",
    )
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if len(_parse_pool_ops(args.pert_pool_aggregations)) != len(
        _parse_pool_scales(args.pert_pool_scale_init)
    ):
        raise ValueError("pert pool op count must match scale count")
    _run_model(args)


if __name__ == "__main__":
    main()
