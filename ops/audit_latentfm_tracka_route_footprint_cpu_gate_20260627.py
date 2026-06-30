#!/usr/bin/env python3
"""CPU route-footprint gate for a default-off Track A perturbation-token adapter.

This is a model-path dry run, not a performance experiment. It checks whether
the existing default-off ``condition_delta_head -> condition_delta_to_c`` route
can preserve the xverse anchor exactly at initialization while still exposing a
nonzero gradient and one-step, condition-specific prediction footprint.

No training loop, inference over datasets, checkpoint selection, canonical
multi, Track C query, or GPU is used. Passing this gate would only justify a
separate train-only proxy/no-harm gate; it does not authorize a GPU smoke.
"""

from __future__ import annotations

import dataclasses
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path("/data/cyx/1030/scLatent")
REPO = ROOT / "CoupledFM"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from model.condition_emb.genepert import PERT_TYPE_CRISPRI  # noqa: E402
from model.latent.config import Config  # noqa: E402
from model.latent.train import apply_finetune_freeze, build_model, load_model_weights_only  # noqa: E402


ANCHOR_CKPT = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/"
    / "xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt"
)
EXACT_CROSS_ROWS = ROOT / "reports/tracka_cross_background_seen_gene_exact_20260627/cross_background_seen_gene_rows.csv"
OUT_JSON = ROOT / "reports/latentfm_tracka_route_footprint_cpu_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_ROUTE_FOOTPRINT_CPU_GATE_20260627.md"


def _cfg_from_checkpoint(path: Path) -> Config:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    raw = ckpt.get("config") or {}
    if not isinstance(raw, dict):
        raise ValueError(f"checkpoint config is not a dict: {path}")
    fields = {f.name for f in dataclasses.fields(Config)}
    data = {k: v for k, v in raw.items() if k in fields}
    return Config(**data)


def _clone_cfg(cfg: Config, **updates: Any) -> Config:
    data = dataclasses.asdict(cfg)
    data.update(updates)
    fields = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in data.items() if k in fields})


def _condition_batch(cfg: Config, *, device: torch.device, genes: list[str], batch_size: int) -> dict[str, torch.Tensor]:
    from model.utils.embeddings.gene_cache import GeneEmbeddingCache

    cache = GeneEmbeddingCache(Path(cfg.pert_gene_emb_cache_dir))
    max_genes = int(getattr(cfg, "max_pert_genes", 16) or 16)
    gene_ids = torch.zeros((batch_size, max_genes), dtype=torch.long, device=device)
    mask = torch.zeros((batch_size, max_genes), dtype=torch.bool, device=device)
    for i in range(batch_size):
        gene = genes[i % len(genes)]
        gid = int(cache.lookup(gene))
        if gid in {int(cache.pad_index), int(cache.unk_index)}:
            raise ValueError(f"gene {gene!r} not usable in cache {cfg.pert_gene_emb_cache_dir}")
        gene_ids[i, 0] = gid
        mask[i, 0] = True
    return {
        "pert_gene_ids": gene_ids,
        "pert_mask": mask,
        "pert_type_id": torch.full((batch_size,), int(PERT_TYPE_CRISPRI), dtype=torch.long, device=device),
        "nperts": torch.ones((batch_size,), dtype=torch.long, device=device),
        "combo_id": torch.zeros((batch_size,), dtype=torch.long, device=device),
        "chem_emb": None,
        "chem_mask": torch.zeros((batch_size, 1), dtype=torch.bool, device=device),
    }


def _select_cache_genes(cfg: Config, *, n: int) -> list[str]:
    from model.utils.embeddings.gene_cache import GeneEmbeddingCache

    cache = GeneEmbeddingCache(Path(cfg.pert_gene_emb_cache_dir))
    picked: list[str] = []
    if EXACT_CROSS_ROWS.is_file():
        with EXACT_CROSS_ROWS.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                gene = str(row.get("gene") or row.get("condition") or "").strip().upper()
                if not gene or gene in picked:
                    continue
                gid = int(cache.lookup(gene))
                if gid in {int(cache.pad_index), int(cache.unk_index)}:
                    continue
                picked.append(gene)
                if len(picked) >= n:
                    return picked
    for gene in ("CCT4", "CHCHD10", "AKT1", "FASN", "TTK", "USP9X", "CAD", "CCND3"):
        if gene in picked:
            continue
        gid = int(cache.lookup(gene))
        if gid not in {int(cache.pad_index), int(cache.unk_index)}:
            picked.append(gene)
        if len(picked) >= n:
            return picked
    if len(picked) < n:
        raise ValueError(f"could not find {n} usable genes in {cfg.pert_gene_emb_cache_dir}")
    return picked


def _forward(model: torch.nn.Module, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor, pb: dict[str, Any]) -> torch.Tensor:
    return model(
        x_t,
        t,
        x0,
        pert_gene_ids=pb["pert_gene_ids"],
        pert_mask=pb["pert_mask"],
        pert_type_id=pb["pert_type_id"],
        nperts=pb["nperts"],
        combo_id=pb["combo_id"],
        chem_emb=pb["chem_emb"],
        chem_mask=pb["chem_mask"],
    )


def _norm(x: torch.Tensor) -> float:
    return float(x.detach().float().norm().cpu())


def main() -> None:
    torch.manual_seed(20260627)
    device = torch.device("cpu")
    base_cfg = _cfg_from_checkpoint(ANCHOR_CKPT)
    base_cfg = _clone_cfg(base_cfg, gpu=0)
    adapter_cfg = _clone_cfg(
        base_cfg,
        condition_delta_head_use_in_model=True,
        condition_delta_head_hidden=int(getattr(base_cfg, "condition_delta_head_hidden", 1024) or 1024),
        condition_prior_delta_loss_weight=0.01,
        finetune_trainable_scope="condition_prior_adapter",
    )

    base = build_model(base_cfg, device)
    adapter = build_model(adapter_cfg, device)
    base_missing, base_unexpected, base_skipped = load_model_weights_only(
        ANCHOR_CKPT, base, device, strict=False, prefer_ema=True
    )
    ad_missing, ad_unexpected, ad_skipped = load_model_weights_only(
        ANCHOR_CKPT, adapter, device, strict=False, prefer_ema=True
    )
    apply_finetune_freeze(adapter, adapter_cfg)
    base.eval()
    adapter.eval()

    trainable = [name for name, param in adapter.named_parameters() if param.requires_grad]
    bridge_weight = dict(adapter.named_parameters()).get("condition_delta_to_c.weight")
    bridge_bias = dict(adapter.named_parameters()).get("condition_delta_to_c.bias")
    if bridge_weight is None or bridge_bias is None:
        raise RuntimeError("adapter model lacks condition_delta_to_c bridge")

    batch = 8
    genes = _select_cache_genes(adapter_cfg, n=batch)
    pb = _condition_batch(adapter_cfg, device=device, genes=genes, batch_size=batch)
    emb_dim = int(adapter_cfg.emb_dim)
    x0 = torch.randn(batch, emb_dim, device=device) * 0.2
    x1 = x0 + torch.randn(batch, emb_dim, device=device) * 0.05
    t = torch.linspace(0.1, 0.9, steps=batch, device=device)
    x_t = (1.0 - t[:, None]) * x0 + t[:, None] * x1
    target_v = x1 - x0

    with torch.no_grad():
        base_out = _forward(base, x_t, t, x0, pb)
        adapter_out0 = _forward(adapter, x_t, t, x0, pb)
    initial_diff = (adapter_out0 - base_out).detach()
    no_op_max_abs = float(initial_diff.abs().max().cpu())
    no_op_l2 = _norm(initial_diff)

    adapter.zero_grad(set_to_none=True)
    pred = _forward(adapter, x_t, t, x0, pb)
    loss = torch.nn.functional.mse_loss(pred, target_v)
    loss.backward()
    grad_by_name = {
        name: _norm(param.grad)
        for name, param in adapter.named_parameters()
        if param.requires_grad and param.grad is not None
    }
    total_grad_norm = math.sqrt(sum(v * v for v in grad_by_name.values()))
    bridge_grad_norm = math.sqrt(
        sum(
            grad_by_name.get(name, 0.0) ** 2
            for name in ("condition_delta_to_c.weight", "condition_delta_to_c.bias")
        )
    )

    with torch.no_grad():
        for param in adapter.parameters():
            if param.requires_grad and param.grad is not None:
                param.add_(param.grad, alpha=-1e-2)
        adapter_out1 = _forward(adapter, x_t, t, x0, pb)
        one_step_delta = adapter_out1 - adapter_out0
        # Same x/input, different perturbation token. This isolates whether the
        # route can express condition-token-specific changes after one update.
        pb_alt = _condition_batch(
            adapter_cfg,
            device=device,
            genes=list(reversed(genes)),
            batch_size=batch,
        )
        adapter_alt = _forward(adapter, x_t, t, x0, pb_alt)
        base_alt = _forward(base, x_t, t, x0, pb_alt)
        condition_specific_delta = adapter_alt - adapter_out1
        adapter_only_delta = adapter_out1 - base_out
        adapter_only_alt_delta = adapter_alt - base_alt
        adapter_only_condition_specific_delta = adapter_only_alt_delta - adapter_only_delta

    one_step_l2 = _norm(one_step_delta)
    one_step_max_abs = float(one_step_delta.abs().max().cpu())
    condition_specific_l2 = _norm(condition_specific_delta)
    condition_specific_max_abs = float(condition_specific_delta.abs().max().cpu())
    adapter_only_condition_specific_l2 = _norm(adapter_only_condition_specific_delta)
    adapter_only_condition_specific_max_abs = float(adapter_only_condition_specific_delta.abs().max().cpu())
    row_l2 = one_step_delta.detach().float().norm(dim=1).cpu().numpy()
    nonzero_row_fraction = float((row_l2 > 1e-8).mean())

    reasons = []
    if no_op_max_abs > 1e-7:
        reasons.append("initial_noop_drift_gt_1e_minus_7")
    if bridge_grad_norm <= 1e-10 or total_grad_norm <= 1e-10:
        reasons.append("route_gradient_inactive")
    if one_step_l2 <= 1e-8 or nonzero_row_fraction < 0.15:
        reasons.append("one_step_nonnoop_footprint_too_small")
    if adapter_only_condition_specific_l2 <= 1e-8:
        reasons.append("condition_specific_footprint_absent")
    # This gate is intentionally not enough for GPU: it does not evaluate
    # train-only validation metrics or no-harm against real held-out rows.
    reasons.append("performance_noharm_gate_not_run_no_gpu")

    status = "tracka_route_footprint_cpu_gate_pass_needs_trainonly_metric_gate_no_gpu"
    if any(r != "performance_noharm_gate_not_run_no_gpu" for r in reasons):
        status = "tracka_route_footprint_cpu_gate_fail_no_gpu"

    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_model_path_dry_run": True,
            "training_loop": False,
            "dataset_inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection_used": False,
            "trackc_query_used": False,
            "performance_noharm_gate_run": False,
        },
        "inputs": {
            "anchor_checkpoint": str(ANCHOR_CKPT),
            "exact_cross_rows_context": str(EXACT_CROSS_ROWS),
        },
        "config": {
            "latent_backbone": adapter_cfg.latent_backbone,
            "emb_dim": adapter_cfg.emb_dim,
            "mlp_d_model": adapter_cfg.mlp_d_model,
            "condition_delta_head_use_in_model": adapter_cfg.condition_delta_head_use_in_model,
            "finetune_trainable_scope": adapter_cfg.finetune_trainable_scope,
            "prefer_ema": True,
        },
        "load_state": {
            "base_missing_keys": base_missing,
            "base_unexpected_keys": base_unexpected,
            "base_skipped_shape_mismatch": base_skipped,
            "adapter_missing_keys": ad_missing,
            "adapter_unexpected_keys": ad_unexpected,
            "adapter_skipped_shape_mismatch": ad_skipped,
        },
        "trainable_names": trainable,
        "metrics": {
            "initial_noop_max_abs": no_op_max_abs,
            "initial_noop_l2": no_op_l2,
            "loss": float(loss.detach().cpu()),
            "total_grad_norm": total_grad_norm,
            "bridge_grad_norm": bridge_grad_norm,
            "one_step_l2": one_step_l2,
            "one_step_max_abs": one_step_max_abs,
            "one_step_nonzero_row_fraction_gt_1e_minus_8": nonzero_row_fraction,
            "condition_specific_l2": condition_specific_l2,
            "condition_specific_max_abs": condition_specific_max_abs,
            "adapter_only_condition_specific_l2": adapter_only_condition_specific_l2,
            "adapter_only_condition_specific_max_abs": adapter_only_condition_specific_max_abs,
        },
        "grad_by_name": grad_by_name,
        "decision_reasons": reasons,
        "next_action": (
            "implement train-only metric/no-harm proxy gate before any GPU"
            if status.endswith("needs_trainonly_metric_gate_no_gpu")
            else "close this route-footprint path or fix default-off/gradient issue before more work"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Track A Route-Footprint CPU Gate",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "CPU model-path dry run only. It loads the xverse 8k anchor EMA, enables the default-off condition-delta bridge, checks no-op preservation, then applies one synthetic gradient step to test route footprint. No dataset inference, canonical multi selection, Track C query, or GPU is used.",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, val in payload["metrics"].items():
        lines.append(f"| `{key}` | {float(val):.6e} |")
    lines.extend(["", "## Trainable Tensors", ""])
    lines.extend(f"- `{name}`" for name in trainable)
    lines.extend(["", "## Decision Reasons", ""])
    lines.extend(f"- `{reason}`" for reason in reasons)
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This gate can only authorize the next CPU train-only metric/no-harm gate. It cannot authorize a GPU smoke because no real validation metric or no-harm evaluation is run here.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "md": str(OUT_MD), "json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
