#!/usr/bin/env python3
"""No-harm PCGrad adapter unit gate.

CPU/report-only. This checks a proposed PCGrad-style no-harm adapter update on
frozen xverse internal condition means before any training-loop or GPU work.

Key question: does a default-off residual adapter have a usable first-order
anchor/no-harm gradient at initialization? For a standard anchor-replay loss
||adapter_output||^2, exact no-op initialization makes the no-harm loss and its
gradient zero, so vanilla PCGrad may be first-order blind.

No checkpoint selection, canonical multi selection, Track C query, training,
inference, or GPU is used.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path("/data/cyx/1030/scLatent")))

from ops.audit_latentfm_control_radius_residual_clip_preflight_20260627 import (  # noqa: E402
    ROOT,
    load_conditions,
    norm,
)
from ops.audit_latentfm_perturbation_identity_residual_adapter_unit_gate_20260627 import (  # noqa: E402
    FORENSICS_CSV,
    read_csv,
)


REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "noharm_pcgrad_adapter_unit_gate_20260627"
OUT_ROWS = OUT_DIR / "noharm_pcgrad_adapter_step_rows.csv"
OUT_JSON = REPORTS / "latentfm_noharm_pcgrad_adapter_unit_gate_20260627.json"
OUT_MD = REPORTS / "LATENTFM_NOHARM_PCGRAD_ADAPTER_UNIT_GATE_20260627.md"
RNG_SEED = 20260627
STEP_SIZES = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0]
EPS = 1e-12


class ZeroInitGeneResidualAdapter(nn.Module):
    def __init__(self, n_genes: int, emb_dim: int, rank: int = 16, hidden: int = 64):
        super().__init__()
        self.gene_emb = nn.Embedding(n_genes, rank)
        self.net = nn.Sequential(
            nn.LayerNorm(rank),
            nn.Linear(rank, hidden),
            nn.SiLU(),
            nn.Linear(hidden, emb_dim, bias=False),
        )
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, gene_ids: torch.Tensor) -> torch.Tensor:
        return self.net(self.gene_emb(gene_ids))


@dataclass
class Batch:
    gene_ids: torch.Tensor
    effect: torch.Tensor
    gt_effect: torch.Tensor
    task_mask: torch.Tensor
    anchor_mask: torch.Tensor


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_gene_map() -> dict[tuple[str, str, str], str]:
    gene_map: dict[tuple[str, str, str], str] = {}
    for row in read_csv(FORENSICS_CSV):
        group = norm(row.get("group"))
        dataset = norm(row.get("dataset"))
        condition = norm(row.get("condition"))
        gene = norm(row.get("gene")) or condition
        if group and dataset and condition and gene:
            gene_map[(group, dataset, condition)] = gene
    return gene_map


def build_rows() -> list[dict[str, Any]]:
    gene_map = load_gene_map()
    rows: list[dict[str, Any]] = []
    for row in load_conditions():
        gene = gene_map.get((row["group"], row["dataset"], row["condition"]), row["condition"])
        new = dict(row)
        new["gene"] = gene
        new["endpoint_mse"] = float(np.mean((row["effect"] - row["gt_effect"]) ** 2))
        rows.append(new)
    return rows


def build_batch(rows: list[dict[str, Any]]) -> tuple[ZeroInitGeneResidualAdapter, Batch, dict[str, Any]]:
    torch.manual_seed(RNG_SEED)
    genes = sorted({row["gene"] for row in rows})
    gene_to_id = {gene: idx for idx, gene in enumerate(genes)}
    emb_dim = int(rows[0]["effect"].shape[0])
    endpoint_mses = np.asarray([row["endpoint_mse"] for row in rows], dtype=np.float64)
    q75 = float(np.quantile(endpoint_mses, 0.75))
    # Task rows emphasize hard/tail-like rows where a correction would matter.
    task_flags = [bool(row["hard_tail"]) or float(row["endpoint_mse"]) >= q75 for row in rows]
    if sum(task_flags) < 16:
        task_flags = [float(row["endpoint_mse"]) >= float(np.quantile(endpoint_mses, 0.50)) for row in rows]
    batch = Batch(
        gene_ids=torch.tensor([gene_to_id[row["gene"]] for row in rows], dtype=torch.long),
        effect=torch.tensor(np.stack([row["effect"] for row in rows]), dtype=torch.float32),
        gt_effect=torch.tensor(np.stack([row["gt_effect"] for row in rows]), dtype=torch.float32),
        task_mask=torch.tensor(task_flags, dtype=torch.bool),
        anchor_mask=torch.ones(len(rows), dtype=torch.bool),
    )
    model = ZeroInitGeneResidualAdapter(len(genes), emb_dim)
    meta = {
        "n_rows": len(rows),
        "n_genes": len(genes),
        "emb_dim": emb_dim,
        "task_rows": int(sum(task_flags)),
        "endpoint_mse_q75": q75,
    }
    return model, batch, meta


def losses(model: nn.Module, batch: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta = model(batch.gene_ids)
    task_delta = delta[batch.task_mask]
    task_effect = batch.effect[batch.task_mask] + task_delta
    task_target = batch.gt_effect[batch.task_mask]
    task_loss = torch.mean((task_effect - task_target) ** 2)
    anchor_loss = torch.mean(delta[batch.anchor_mask] ** 2)
    footprint = torch.linalg.norm(delta, dim=1).mean()
    return task_loss, anchor_loss, footprint


def params(model: nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in params(model)])


def set_flat_params(model: nn.Module, vec: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for p in params(model):
            n = p.numel()
            p.copy_(vec[offset : offset + n].reshape_as(p))
            offset += n


def grad_vector(loss: torch.Tensor, model: nn.Module) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params(model), retain_graph=False, allow_unused=True)
    chunks = []
    for p, g in zip(params(model), grads):
        chunks.append(torch.zeros_like(p).reshape(-1) if g is None else g.detach().reshape(-1))
    return torch.cat(chunks)


def pcgrad(task_grad: torch.Tensor, anchor_grad: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    anchor_norm_sq = float(torch.dot(anchor_grad, anchor_grad).item())
    dot = float(torch.dot(task_grad, anchor_grad).item())
    if anchor_norm_sq <= EPS:
        return task_grad.clone(), {"dot_before": dot, "dot_after": dot, "anchor_norm": math.sqrt(anchor_norm_sq), "projection_coeff": 0.0}
    coeff = min(0.0, dot / anchor_norm_sq)
    projected = task_grad - coeff * anchor_grad
    return projected, {
        "dot_before": dot,
        "dot_after": float(torch.dot(projected, anchor_grad).item()),
        "anchor_norm": math.sqrt(anchor_norm_sq),
        "projection_coeff": coeff,
    }


def evaluate_vector(model: nn.Module, batch: Batch, vec: torch.Tensor) -> dict[str, float]:
    set_flat_params(model, vec)
    with torch.no_grad():
        task_loss, anchor_loss, footprint = losses(model, batch)
        delta = model(batch.gene_ids)
        unique_rows = torch.unique(delta.round(decimals=10), dim=0).shape[0]
        row_l2 = torch.linalg.norm(delta, dim=1)
    return {
        "task_loss": float(task_loss.item()),
        "anchor_loss": float(anchor_loss.item()),
        "footprint_mean_l2": float(footprint.item()),
        "material_row_frac": float((row_l2 > 1e-6).float().mean().item()),
        "condition_specific_unique_frac": float(unique_rows / max(1, delta.shape[0])),
    }


def main() -> None:
    rows = build_rows()
    model, batch, meta = build_batch(rows)
    p0 = flat_params(model).clone()
    with torch.no_grad():
        initial_delta = model(batch.gene_ids)
    initial_max_abs = float(initial_delta.abs().max().item())

    task_loss0, anchor_loss0, footprint0 = losses(model, batch)
    task_grad = grad_vector(task_loss0, model)
    task_grad_norm = float(torch.linalg.norm(task_grad).item())
    # Recompute because the previous autograd graph was consumed.
    task_loss0b, anchor_loss0b, footprint0b = losses(model, batch)
    anchor_grad0 = grad_vector(anchor_loss0b, model)
    projected0, proj0 = pcgrad(task_grad, anchor_grad0)
    projection_changed0 = float(torch.linalg.norm(projected0 - task_grad).item())
    anchor_grad0_norm = float(torch.linalg.norm(anchor_grad0).item())

    base_metrics = evaluate_vector(model, batch, p0)
    step_rows: list[dict[str, Any]] = []
    best_projected: dict[str, Any] | None = None
    for step in STEP_SIZES:
        unproj_vec = p0 - float(step) * task_grad
        unproj_metrics = evaluate_vector(model, batch, unproj_vec)
        # Lookahead variant: compute anchor gradient at the unprojected probe.
        task_l, anchor_l, _ = losses(model, batch)
        anchor_grad_probe = grad_vector(anchor_l, model)
        set_flat_params(model, p0)
        proj_grad, proj_probe = pcgrad(task_grad, anchor_grad_probe)
        proj_vec = p0 - float(step) * proj_grad
        proj_metrics = evaluate_vector(model, batch, proj_vec)
        set_flat_params(model, p0)
        row = {
            "step": step,
            "unproj_task_delta": unproj_metrics["task_loss"] - base_metrics["task_loss"],
            "unproj_anchor_delta": unproj_metrics["anchor_loss"] - base_metrics["anchor_loss"],
            "unproj_footprint_mean_l2": unproj_metrics["footprint_mean_l2"],
            "proj_task_delta": proj_metrics["task_loss"] - base_metrics["task_loss"],
            "proj_anchor_delta": proj_metrics["anchor_loss"] - base_metrics["anchor_loss"],
            "proj_footprint_mean_l2": proj_metrics["footprint_mean_l2"],
            "proj_material_row_frac": proj_metrics["material_row_frac"],
            "proj_condition_specific_unique_frac": proj_metrics["condition_specific_unique_frac"],
            "probe_anchor_grad_norm": proj_probe["anchor_norm"],
            "probe_dot_before": proj_probe["dot_before"],
            "probe_dot_after": proj_probe["dot_after"],
            "probe_projection_coeff": proj_probe["projection_coeff"],
            "projection_reduced_anchor_delta": proj_metrics["anchor_loss"] <= unproj_metrics["anchor_loss"] + 1e-12,
        }
        step_rows.append(row)
        candidate_ok = (
            row["proj_task_delta"] < -1e-10
            and row["proj_anchor_delta"] <= 1e-6
            and row["proj_footprint_mean_l2"] > 1e-6
            and row["proj_material_row_frac"] >= 0.15
            and row["proj_condition_specific_unique_frac"] >= 0.15
        )
        if candidate_ok and (
            best_projected is None
            or row["proj_task_delta"] < best_projected["proj_task_delta"]
        ):
            best_projected = row

    reasons: list[str] = []
    if initial_max_abs > 1e-7:
        reasons.append("initial_adapter_not_noop")
    if task_grad_norm <= 1e-8:
        reasons.append("task_gradient_not_live")
    if anchor_grad0_norm <= 1e-10:
        reasons.append("anchor_gradient_zero_at_default_off_noop")
    if projection_changed0 <= 1e-12:
        reasons.append("vanilla_pcgrad_projection_noop_at_initialization")
    if best_projected is None:
        reasons.append("no_lookahead_step_satisfies_task_anchor_footprint_gate")
    else:
        # Lookahead is a different algorithm from vanilla PCGrad; record it as
        # insight, not as authorization.
        reasons.append("lookahead_projection_required_not_vanilla_pcgrad")
    status = "noharm_pcgrad_adapter_unit_gate_fail_no_gpu"
    if (
        initial_max_abs <= 1e-7
        and task_grad_norm > 1e-8
        and anchor_grad0_norm > 1e-10
        and projection_changed0 > 1e-12
        and best_projected is not None
    ):
        status = "noharm_pcgrad_adapter_unit_gate_pass_external_audit_only_no_gpu"
        reasons = []

    write_csv(
        OUT_ROWS,
        step_rows,
        [
            "step",
            "unproj_task_delta",
            "unproj_anchor_delta",
            "unproj_footprint_mean_l2",
            "proj_task_delta",
            "proj_anchor_delta",
            "proj_footprint_mean_l2",
            "proj_material_row_frac",
            "proj_condition_specific_unique_frac",
            "probe_anchor_grad_norm",
            "probe_dot_before",
            "probe_dot_after",
            "probe_projection_coeff",
            "projection_reduced_anchor_delta",
        ],
    )
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
        "status": status,
        "gpu_authorized": False,
        "reasons": reasons,
        "meta": meta,
        "base_metrics": base_metrics,
        "initial": {
            "initial_max_abs": initial_max_abs,
            "task_loss": float(task_loss0.detach().item()),
            "anchor_loss": float(anchor_loss0.detach().item()),
            "footprint": float(footprint0.detach().item()),
            "task_grad_norm": task_grad_norm,
            "anchor_grad_norm": anchor_grad0_norm,
            "initial_pcgrad": proj0,
            "projection_changed_norm": projection_changed0,
        },
        "best_projected_lookahead_step": best_projected,
        "outputs": {
            "rows": str(OUT_ROWS),
            "report": str(OUT_MD),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    best_line = "None."
    if best_projected is not None:
        best_line = (
            f"step `{best_projected['step']}`; projected task delta "
            f"`{best_projected['proj_task_delta']:.6g}`; projected anchor delta "
            f"`{best_projected['proj_anchor_delta']:.6g}`; footprint "
            f"`{best_projected['proj_footprint_mean_l2']:.6g}`."
        )
    report = [
        "# LatentFM No-Harm PCGrad Adapter Unit Gate",
        "",
        f"Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M %Z')}`",
        "",
        f"Status: `{status}`",
        "",
        "## Scope",
        "",
        "CPU/report-only frozen-means unit gate. No training, inference, GPU, "
        "canonical multi selection, Track C query, or checkpoint selection.",
        "",
        "## Initial State",
        "",
        f"- rows/genes/task rows: `{meta['n_rows']}` / `{meta['n_genes']}` / `{meta['task_rows']}`",
        f"- initial max abs: `{initial_max_abs:.6g}`",
        f"- task grad norm: `{task_grad_norm:.6g}`",
        f"- anchor/no-harm grad norm at no-op: `{anchor_grad0_norm:.6g}`",
        f"- vanilla PCGrad projection changed norm: `{projection_changed0:.6g}`",
        "",
        "## Lookahead Diagnostic",
        "",
        best_line,
        "",
        "## Decision",
        "",
        f"Fail/pass reasons: `{reasons}`",
        "",
        "Vanilla PCGrad is not GPU-authorized if the default-off no-harm "
        "gradient is zero at initialization. A lookahead/trust-region variant "
        "would be a distinct method and needs a separate gate.",
        "",
        "## Outputs",
        "",
        f"- Rows: `{OUT_ROWS}`",
        f"- JSON: `{OUT_JSON}`",
        "",
    ]
    OUT_MD.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": status, "reasons": reasons, "initial": payload["initial"], "best_projected_lookahead_step": best_projected}, indent=2))


if __name__ == "__main__":
    main()
