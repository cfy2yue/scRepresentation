"""
Inference for CoupledFM with optional latent guidance.

In coupled mode, the frozen latent FM model generates z_t at each ODE step,
which is injected into the CLS token of the raw expression model.
"""

from __future__ import annotations

import gc
import hashlib
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import torch

from model.config import Config
from model.data.vocab import GeneVocab
from model.data.dataset import _LazyH5
from model.utils.io.lazy_loader import read_obs_meta
from model.utils.data.biflow_paths import resolve_biflow_control_gt_h5ad
from model.models.velocity_field import RawExprVelocityField
from model.pert_batch_utils import (
    build_perturbation_batch_from_cond,
    latent_fm_wants_perturbation,
    null_perturbation_batch,
    slice_perturbation_batch,
    try_load_gene_cache_for_inference,
)

from model.condition_emb.chempert.chem_resolver import load_chemical_embed_backend


def _apply_saved_config(cfg: Config, saved_run: Optional[dict]) -> None:
    """Best-effort restore of dataclass config fields saved next to checkpoints."""
    if not isinstance(saved_run, dict):
        return
    for section in ("model", "data", "train", "inference"):
        src = saved_run.get(section)
        dst = getattr(cfg, section, None)
        if not isinstance(src, dict) or dst is None:
            continue
        for key, value in src.items():
            if hasattr(dst, key):
                setattr(dst, key, value)


def _resolve_inference_control_gt_h5ad(cfg: Config, dataset_name: str) -> Tuple[Path, Path]:
    """Resolve the control-center / GT h5ad pair using training layout rules."""
    dc = cfg.data
    biflow_dir = Path(dc.biflow_dir)
    pair = resolve_biflow_control_gt_h5ad(
        biflow_dir,
        dataset_name,
        latent_backbone=getattr(dc, "latent_backbone", "state"),
    )
    if pair is not None:
        return pair

    raise FileNotFoundError(
        "Could not resolve inference control h5ad for "
        f"dataset={dataset_name!r}, biflow_dir={biflow_dir}, "
        f"latent_backbone={getattr(dc, 'latent_backbone', 'state')!r}. "
        "Expected a training-compatible control/gt pair."
    )


def _resolve_inference_control_h5ad(cfg: Config, dataset_name: str) -> Path:
    """Resolve the control h5ad using the same layout rules as training."""
    return _resolve_inference_control_gt_h5ad(cfg, dataset_name)[0]


def _stable_condition_seed(cond: str, base_seed: int = 42) -> int:
    payload = f"{int(base_seed)}::{cond}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0xFFFFFFFF


def _sample_control_indices(n_ctrl: int, max_cells_per_cond: int, cond: str, *, seed: int = 42) -> np.ndarray:
    """Deterministically sample control-center rows used as flow starts."""
    n = int(n_ctrl)
    if n <= 0:
        raise ValueError("control-center pool is empty")
    if max_cells_per_cond and max_cells_per_cond > 0:
        k = min(int(max_cells_per_cond), n)
    else:
        k = n
    rng = np.random.default_rng(_stable_condition_seed(cond, seed))
    if k >= n:
        return np.arange(n, dtype=np.int64)
    return np.asarray(rng.choice(n, size=k, replace=False), dtype=np.int64)


def _target_conditions_from_gt(gt_path: Path, requested: Optional[List[str]], max_conditions: int = 0) -> List[str]:
    labels, _ = read_obs_meta(str(gt_path), read_index=False)
    available = sorted(c for c in np.unique(labels.astype(str)) if c != "control")
    if requested is None:
        out = available
    else:
        aset = set(available)
        missing = [c for c in requested if c not in aset]
        if missing:
            warnings.warn(
                f"requested conditions absent from GT h5ad and will be skipped: {missing[:8]}",
                UserWarning,
                stacklevel=2,
            )
        out = [c for c in requested if c in aset]
    if max_conditions and max_conditions > 0:
        out = out[:max_conditions]
    return out


@torch.no_grad()
def integrate(
    model: RawExprVelocityField,
    x_0: torch.Tensor,
    x_ctrl: torch.Tensor,
    gene_ids: torch.Tensor,
    n_steps: int = 100,
    method: str = "euler",
    latent_fm=None,
    z_src: Optional[torch.Tensor] = None,
    cfg_w: float = 1.0,
    edge_index: Optional[torch.Tensor] = None,
    dx_prior: Optional[torch.Tensor] = None,
    gene_mask: Optional[torch.Tensor] = None,
    perturbation_batch: Optional[Tuple[torch.Tensor, ...]] = None,
    max_pert_genes: int = 16,
) -> torch.Tensor:
    """Integrate from x_0 (t=0) to x_1 (t=1).

    If latent_fm and z_src are provided, generates z_t at each step
    for CLS token injection (coupled mode).

    ``cfg_w``: classifier-free guidance scale (1.0 disables).
    ``dx_prior``: optional per-gene residual prior (G,) added to velocity each step
    when ``use_residual_flow`` training target was debiased.
    ``perturbation_batch``: optional tuple when ``model.use_pert_condition``.
    ``max_pert_genes``: slot width for null / latent FM fallback (default 16).
    """
    dt = 1.0 / n_steps
    x = x_0.clone()
    B = x.shape[0]
    device = x.device
    gm = None
    keep = None
    if gene_mask is not None:
        gm = gene_mask.to(device=device, dtype=x.dtype)
        if gm.shape != x.shape:
            raise ValueError(
                f"integrate: gene_mask shape {tuple(gm.shape)} must match x_0 {tuple(x.shape)}"
            )
        keep = 1.0 - gm
        x = x * keep
        x_ctrl = x_ctrl * keep

    raw_inner = model.module if hasattr(model, "module") else model
    raw_wants = bool(getattr(raw_inner, "use_pert_condition", False))
    latent_wants = latent_fm_wants_perturbation(latent_fm)
    max_pg_lat = (
        int(getattr(latent_fm, "max_pert_genes", max_pert_genes))
        if latent_fm is not None
        else max_pert_genes
    )

    def _pb_latent_integrator() -> Optional[Tuple[torch.Tensor, ...]]:
        if not latent_wants:
            return None
        if perturbation_batch is not None:
            return perturbation_batch
        return null_perturbation_batch(B, max_pg_lat, device=device)

    pb_lat = _pb_latent_integrator()

    z = z_src.clone() if z_src is not None else None
    dpx = None
    if dx_prior is not None:
        if not getattr(model, "use_residual_flow", False):
            warnings.warn(
                "integrate: dx_prior ignored because model.use_residual_flow is False",
                UserWarning,
                stacklevel=2,
            )
        else:
            dpx = dx_prior.to(device=device, dtype=x.dtype).unsqueeze(0).expand(B, -1)
            if keep is not None:
                dpx = dpx * keep

    def vel(
        xx: torch.Tensor,
        xcc: torch.Tensor,
        tt: torch.Tensor,
        auxb: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bb = xx.shape[0]
        pi_b = torch.zeros((bb,), device=device, dtype=torch.long)
        gm_b = gm[:bb] if gm is not None else None

        def _pb_cond_rows() -> Optional[Tuple[torch.Tensor, ...]]:
            if not raw_wants:
                return None
            if perturbation_batch is not None:
                if perturbation_batch[0].shape[0] == bb:
                    return perturbation_batch
                raise ValueError(
                    "integrate: perturbation_batch batch dim must match velocity batch size "
                    f"(got {perturbation_batch[0].shape[0]} vs bb={bb})"
                )
            return null_perturbation_batch(bb, max_pert_genes, device=device)

        def _pb_uncond_rows() -> Optional[Tuple[torch.Tensor, ...]]:
            if not raw_wants:
                return None
            return null_perturbation_batch(bb, max_pert_genes, device=device)

        kw_c: Dict[str, object] = {}
        kw_u: Dict[str, object] = {}
        if raw_wants:
            kw_c["perturbation_batch"] = _pb_cond_rows()
            kw_u["perturbation_batch"] = _pb_uncond_rows()

        if cfg_w != 1.0:
            v_c = model(
                xx, xcc, tt, gene_ids, auxb, gm_b, pi_b,
                edge_index=edge_index,
                **kw_c,
            )
            v_u = model(
                xx, torch.zeros_like(xcc), tt, gene_ids, None, gm_b, pi_b,
                edge_index=edge_index,
                **kw_u,
            )
            v = v_u + cfg_w * (v_c - v_u)
        else:
            v = model(
                xx, xcc, tt, gene_ids, auxb, gm_b, pi_b,
                edge_index=edge_index,
                **kw_c,
            )
        if dpx is not None:
            v = v + dpx
        return v

    for i in range(n_steps):
        t_val = i * dt
        t_vec = torch.full((B,), t_val, device=device)

        aux_emb = None
        if latent_fm is not None and z is not None:
            aux_emb = z

        if method == "euler":
            v = vel(x, x_ctrl, t_vec, aux_emb)
            x = x + dt * v
            if keep is not None:
                x = x * keep
        elif method == "midpoint":
            v1 = vel(x, x_ctrl, t_vec, aux_emb)
            x_mid = x + 0.5 * dt * v1
            if keep is not None:
                x_mid = x_mid * keep
            t_mid = torch.full((B,), t_val + 0.5 * dt, device=device)
            if latent_fm is not None and z is not None:
                z_mid = latent_fm.ode_step(
                    z, z_src, t_val, 0.5 * dt, perturbation_batch=pb_lat,
                )
                aux_mid = z_mid
            else:
                aux_mid = aux_emb
            v2 = vel(x_mid, x_ctrl, t_mid, aux_mid)
            x = x + dt * v2
            if keep is not None:
                x = x * keep
        elif method == "rk4":
            k1 = vel(x, x_ctrl, t_vec, aux_emb)

            t2 = torch.full((B,), t_val + 0.5 * dt, device=device)
            if latent_fm is not None and z is not None:
                z_half = latent_fm.ode_step(
                    z, z_src, t_val, 0.5 * dt, perturbation_batch=pb_lat,
                )
                aux_half = z_half
            else:
                aux_half = aux_emb
            x2 = x + 0.5 * dt * k1
            if keep is not None:
                x2 = x2 * keep
            k2 = vel(x2, x_ctrl, t2, aux_half)
            x3 = x + 0.5 * dt * k2
            if keep is not None:
                x3 = x3 * keep
            k3 = vel(x3, x_ctrl, t2, aux_half)

            t4 = torch.full((B,), t_val + dt, device=device)
            if latent_fm is not None and z is not None:
                z_full = latent_fm.ode_step(
                    z, z_src, t_val, dt, perturbation_batch=pb_lat,
                )
                aux_full = z_full
            else:
                aux_full = aux_emb
            x4 = x + dt * k3
            if keep is not None:
                x4 = x4 * keep
            k4 = vel(x4, x_ctrl, t4, aux_full)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if keep is not None:
                x = x * keep

        if latent_fm is not None and z is not None:
            z = latent_fm.ode_step(
                z, z_src, t_val, dt, perturbation_batch=pb_lat,
            )

    return x


def predict_dataset(
    cfg: Config,
    dataset_name: str,
    conditions: Optional[List[str]] = None,
    max_cells_per_cond: int = 256,
    micro_batch: int = 4,
    max_conditions: int = 0,
) -> Dict[str, np.ndarray]:
    ic = cfg.inference
    mc = cfg.model
    dc = cfg.data
    use_residual_flow_m = bool(getattr(ic, "use_residual_flow", False))
    if use_residual_flow_m:
        raise ValueError(
            "Standalone inference does not support use_residual_flow=True. "
            "Residual-flow training subtracts a condition-specific GT-control prior, "
            "which is unavailable for target-condition prediction without GT. "
            "Use the default RAW_USE_RESIDUAL_FLOW=0 / cfg.inference.use_residual_flow=False route."
        )

    device = torch.device(ic.device if torch.cuda.is_available() else "cpu")

    vocab = GeneVocab(dc.gene_name_path, dc.nichenet_node2idx_path)

    pert_cache_model = try_load_gene_cache_for_inference(mc, dc)

    model = RawExprVelocityField(
        d_model=mc.d_model, n_layer=mc.n_layer, n_head=mc.n_head,
        d_ff=mc.d_ff, dropout=0.0, attn_mode=mc.attn_mode,
        d_latent=mc.d_latent,
        attn_backend=mc.attn_backend,
        coupling_mode=ic.coupling_mode,
        use_pert_token=getattr(mc, "use_pert_token", False),
        num_pert_ids=getattr(mc, "num_pert_ids", 10000),
        graph_bias_mode=getattr(mc, "graph_bias_mode", "none"),
        use_latent_resampler=getattr(mc, "use_latent_resampler", False),
        latent_resampler_n_tokens=getattr(mc, "latent_resampler_n_tokens", 8),
        latent_resampler_n_head=getattr(mc, "latent_resampler_n_head", 4),
        cross_attn_independent_kv=getattr(mc, "cross_attn_independent_kv", False),
        value_encoder=getattr(mc, "value_encoder", "linear"),
        fourier_n_freqs=getattr(mc, "fourier_n_freqs", 32),
        use_residual_flow=False,
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
    model.eval()
    ckpt = torch.load(ic.ckpt_path, map_location=device, weights_only=False)
    load_result = model.load_state_dict(ckpt["model"], strict=False)
    missing = list(getattr(load_result, "missing_keys", []) or [])
    unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
    if missing or unexpected:
        warnings.warn(
            "checkpoint/model key mismatch during inference: "
            f"missing={missing[:32]} unexpected={unexpected[:32]} "
            f"(counts: missing={len(missing)}, unexpected={len(unexpected)})",
            UserWarning,
            stacklevel=2,
        )
    critical_missing = [k for k in missing if k.endswith(("pool_scale", "type_scale"))]
    if critical_missing:
        warnings.warn(
            f"checkpoint missing critical perturbation params: {critical_missing}; "
            f"these are kept at constructor init values, which can cause silent "
            f"behavior drift between train and inference",
            UserWarning,
            stacklevel=2,
        )

    latent_fm = None
    if ic.coupling_mode == "coupled" and not ic.latent_fm_ckpt:
        raise ValueError(
            "Coupled inference requires cfg.inference.latent_fm_ckpt to be set."
        )
    if ic.coupling_mode == "coupled" and ic.latent_fm_ckpt:
        from model.latent_utils import FrozenLatentFM
        latent_fm = FrozenLatentFM(ic.latent_fm_ckpt, device=str(device))

    max_pg = int(getattr(mc, "max_pert_genes", 16))

    chem_backend = None
    if bool(getattr(dc, "pert_chem_enabled", False)):
        chem_backend = load_chemical_embed_backend(
            dc,
            fallback_dim=int(getattr(dc, "chem_fallback_embed_dim", 512)),
        )

    # ── load data (lazy — safe for gwps) ────────────────────────
    control_path, gt_path = _resolve_inference_control_gt_h5ad(cfg, dataset_name)
    ctrl = ad.read_h5ad(str(control_path))

    var_names = list(ctrl.var_names)
    n_cols = len(var_names)
    gene_ids_full = np.array(
        [vocab.gene2token.get(g, -1) for g in var_names], dtype=np.int64
    )
    in_vocab = gene_ids_full >= 0
    full_mask = np.ones(n_cols, dtype=bool)
    gene_ids = torch.from_numpy(gene_ids_full[in_vocab]).to(device)

    from scipy.sparse import issparse as _issparse
    X_ctrl = np.asarray(ctrl.X.toarray() if _issparse(ctrl.X) else ctrl.X,
                        dtype=np.float32)
    del ctrl
    gc.collect()

    # Flow start pool: same control-center cells as training (source / x_0 pool).
    pert_path = str(control_path)
    pert_h5 = _LazyH5(pert_path, load_latent=(latent_fm is not None))

    if latent_fm is not None and not pert_h5.has_latent:
        raise ValueError(
            "coupled inference requires latent embeddings in the control h5ad (obsm['emb'] or "
            "obsm['exp_emb1']), but this file provides none. Cannot run coupled ODE without z."
        )

    conditions = _target_conditions_from_gt(gt_path, conditions, max_conditions=max_conditions)

    results: Dict[str, np.ndarray] = {}

    for cond in conditions:
        ctrl_idx = _sample_control_indices(pert_h5.n_rows, max_cells_per_cond, cond)
        x_src_np = pert_h5.read_X_rows(ctrl_idx, in_vocab)
        x_ctrl_np = X_ctrl[ctrl_idx][:, in_vocab]

        z_src_np = None
        if latent_fm is not None and pert_h5.has_latent:
            z_src_np = pert_h5.read_z_rows(ctrl_idx)

        n_cells = len(ctrl_idx)
        _cbd = str(getattr(dc, "chem_emb_source_dir", "") or "").strip()
        pb_full = build_perturbation_batch_from_cond(
            cond,
            n_cells,
            cache=pert_cache_model,
            max_genes=max_pg,
            device=device,
            max_chem_slots=int(getattr(dc, "max_chem_keys", 4)),
            chem_backend=chem_backend,
            chem_metainfo=None,
            chem_max_keys=int(getattr(dc, "max_chem_keys", 4)),
            chem_legacy_dirs=[_cbd] if _cbd else None,
            pert_chem_enabled=bool(getattr(dc, "pert_chem_enabled", False)),
        )

        pred_chunks = []
        for ci in range(0, n_cells, micro_batch):
            s, e = ci, min(ci + micro_batch, n_cells)
            x_src_c = torch.from_numpy(x_src_np[s:e]).to(device)
            x_ctrl_c = torch.from_numpy(x_ctrl_np[s:e]).to(device)
            z_src_c = None
            if z_src_np is not None:
                z_src_c = torch.from_numpy(z_src_np[s:e]).to(device)

            pb_chunk = (
                slice_perturbation_batch(pb_full, s, e, device)
                if pb_full is not None
                else None
            )
            pred_c = integrate(
                model, x_src_c, x_ctrl_c, gene_ids,
                n_steps=ic.n_steps, method=ic.method,
                latent_fm=latent_fm, z_src=z_src_c,
                cfg_w=float(getattr(ic, "cfg_w", 1.0)),
                perturbation_batch=pb_chunk,
                max_pert_genes=max_pg,
            )
            pred_chunks.append(pred_c.cpu().numpy())

        x_pred_vocab = np.concatenate(pred_chunks, axis=0)
        x_pred_full = pert_h5.read_X_rows(ctrl_idx, full_mask)
        x_pred_full[:, in_vocab] = x_pred_vocab
        x_pred_mean = x_pred_full.mean(axis=0, keepdims=True)
        results[cond] = x_pred_mean
        print(
            f"  [{dataset_name}] {cond}: sampled {x_pred_full.shape[0]} control centers -> mean prediction",
            flush=True,
        )

    pert_h5.close()
    return results


def save_predictions(
    results: Dict[str, np.ndarray],
    dataset_name: str,
    var_names: List[str],
    output_dir: str,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_X = []
    obs_conds = []
    for cond, X in results.items():
        all_X.append(X)
        obs_conds.extend([cond] * X.shape[0])

    X_concat = np.concatenate(all_X, axis=0)
    obs = pd.DataFrame({"perturbation": obs_conds})
    adata = ad.AnnData(X=X_concat, obs=obs)
    adata.var_names = var_names

    out_path = output_dir / f"{dataset_name}_pred.h5ad"
    adata.write_h5ad(str(out_path))
    print(f"[inference] saved {out_path} ({adata.n_obs} cells)")
    return out_path


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="CoupledFM inference")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--mode", choices=["baseline", "ot", "coupled"], default=None)
    parser.add_argument("--method", choices=["euler", "midpoint", "rk4"],
                        default="euler")
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--conditions", type=str, nargs="+", default=None)
    parser.add_argument("--max-cells-per-cond", type=int, default=256)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--max-conditions", type=int, default=0)
    args = parser.parse_args()

    cfg = Config()
    ckpt_side = Path(args.ckpt).expanduser().resolve().parent / "config.json"
    saved_run = None
    if ckpt_side.is_file():
        try:
            saved_run = json.loads(ckpt_side.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover
            saved_run = None
    _apply_saved_config(cfg, saved_run)
    cfg.inference.ckpt_path = args.ckpt
    if args.mode is not None:
        cfg.inference.coupling_mode = args.mode
    elif saved_run:
        tconf = saved_run.get("train") or {}
        if tconf.get("coupling_mode"):
            cfg.inference.coupling_mode = str(tconf["coupling_mode"])
    if saved_run:
        tconf = saved_run.get("train") or {}
        iconf = saved_run.get("inference") or {}
        if iconf.get("use_residual_flow") is not None:
            cfg.inference.use_residual_flow = bool(iconf["use_residual_flow"])
        elif tconf.get("use_residual_flow"):
            raise ValueError(
                "Checkpoint config was trained with use_residual_flow=True, but standalone "
                "inference cannot reconstruct the required GT-control residual prior. "
                "Use a non-residual-flow checkpoint for prediction."
            )
    cfg.inference.method = args.method
    cfg.inference.n_steps = args.n_steps
    cfg.inference.device = args.device
    if args.output_dir:
        cfg.inference.output_dir = args.output_dir

    results = predict_dataset(
        cfg,
        args.dataset,
        conditions=args.conditions,
        max_cells_per_cond=args.max_cells_per_cond,
        micro_batch=args.micro_batch,
        max_conditions=args.max_conditions,
    )

    ctrl_path = _resolve_inference_control_h5ad(cfg, args.dataset)
    ctrl = ad.read_h5ad(str(ctrl_path))
    save_predictions(results, args.dataset, list(ctrl.var_names), cfg.inference.output_dir)
