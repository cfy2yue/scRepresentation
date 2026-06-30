#!/usr/bin/env python3
"""CPU gate for a Track A solver-state residual adapter idea.

This gate has two parts:

1. A model-path dry run: wrap the xverse 8k anchor EMA with a default-off,
   low-rank residual adapter that consumes solver state (x_t, x_0, t, anchor
   velocity). It checks exact no-op at initialization and one synthetic
   gradient step for adapter-only footprint.
2. A train/internal proxy gate: on residual-forensics rows, test whether
   non-target proxy features can learn a leave-one-dataset-out policy for
   internal residual repair. This is still only a CPU proxy; it cannot
   authorize GPU by itself.

No training loop, dataset inference, checkpoint selection, canonical multi,
Track C query, or GPU is used.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


ROOT = Path("/data/cyx/1030/scLatent")
REPO = ROOT / "CoupledFM"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from model.condition_emb.genepert import PERT_TYPE_CRISPRI  # noqa: E402
from model.latent.config import Config  # noqa: E402
from model.latent.train import build_model, load_model_weights_only  # noqa: E402
from model.utils.embeddings.gene_cache import GeneEmbeddingCache  # noqa: E402


ANCHOR_CKPT = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/"
    / "xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt"
)
FORENSICS = ROOT / "reports/latentfm_xverse_tracka_residual_forensics_conditions_20260622.csv"
FAILURES = ROOT / "reports/tracka_deployable_benchmark_failure_taxonomy_20260627/failure_cases.csv"
CROSSBG_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
OUT_JSON = ROOT / "reports/latentfm_tracka_solver_state_residual_footprint_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_SOLVER_STATE_RESIDUAL_FOOTPRINT_GATE_20260627.md"


class LowRankSolverResidual(nn.Module):
    def __init__(self, emb_dim: int, rank: int = 8):
        super().__init__()
        self.down = nn.Linear(3 * emb_dim + 1, rank)
        self.up = nn.Linear(rank, emb_dim)
        nn.init.xavier_uniform_(self.down.weight, gain=0.1)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x_t: torch.Tensor, x0: torch.Tensor, t: torch.Tensor, anchor_v: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([x_t, x0, anchor_v, t.reshape(-1, 1)], dim=1)
        return self.up(torch.tanh(self.down(feat)))


def _cfg_from_checkpoint(path: Path) -> Config:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    raw = ckpt.get("config") or {}
    fields = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in raw.items() if k in fields})


def _select_cache_genes(cfg: Config, n: int) -> list[str]:
    cache = GeneEmbeddingCache(Path(cfg.pert_gene_emb_cache_dir))
    genes: list[str] = []
    with FORENSICS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            gene = str(row.get("gene", "")).strip().upper()
            if not gene or gene in genes:
                continue
            gid = int(cache.lookup(gene))
            if gid in {int(cache.pad_index), int(cache.unk_index)}:
                continue
            genes.append(gene)
            if len(genes) >= n:
                return genes
    raise ValueError(f"could not select {n} cache genes")


def _condition_batch(cfg: Config, genes: list[str], batch_size: int, device: torch.device) -> dict[str, Any]:
    cache = GeneEmbeddingCache(Path(cfg.pert_gene_emb_cache_dir))
    max_genes = int(getattr(cfg, "max_pert_genes", 16) or 16)
    gid = torch.zeros((batch_size, max_genes), dtype=torch.long, device=device)
    mask = torch.zeros((batch_size, max_genes), dtype=torch.bool, device=device)
    for i in range(batch_size):
        gid[i, 0] = int(cache.lookup(genes[i % len(genes)]))
        mask[i, 0] = True
    return {
        "pert_gene_ids": gid,
        "pert_mask": mask,
        "pert_type_id": torch.full((batch_size,), int(PERT_TYPE_CRISPRI), dtype=torch.long, device=device),
        "nperts": torch.ones((batch_size,), dtype=torch.long, device=device),
        "combo_id": torch.zeros((batch_size,), dtype=torch.long, device=device),
        "chem_emb": None,
        "chem_mask": torch.zeros((batch_size, 1), dtype=torch.bool, device=device),
    }


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


def model_footprint_gate() -> dict[str, Any]:
    torch.manual_seed(20260627)
    device = torch.device("cpu")
    cfg = _cfg_from_checkpoint(ANCHOR_CKPT)
    model = build_model(cfg, device)
    missing, unexpected, skipped = load_model_weights_only(ANCHOR_CKPT, model, device, strict=False, prefer_ema=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    emb_dim = int(cfg.emb_dim)
    adapter = LowRankSolverResidual(emb_dim=emb_dim, rank=8).to(device)
    batch = 8
    genes = _select_cache_genes(cfg, batch)
    pb = _condition_batch(cfg, genes, batch, device)
    x0 = torch.randn(batch, emb_dim, device=device) * 0.2
    x1 = x0 + torch.randn(batch, emb_dim, device=device) * 0.05
    t = torch.linspace(0.1, 0.9, steps=batch, device=device)
    x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
    target_v = x1 - x0

    with torch.no_grad():
        anchor_v = _forward(model, x_t, t, x0, pb)
        init_resid = adapter(x_t, x0, t, anchor_v)
    initial_noop_max_abs = float(init_resid.abs().max().cpu())
    initial_noop_l2 = float(init_resid.float().norm().cpu())

    adapter.zero_grad(set_to_none=True)
    anchor_v_train = _forward(model, x_t, t, x0, pb).detach()
    pred = anchor_v_train + adapter(x_t, x0, t, anchor_v_train)
    loss = torch.nn.functional.mse_loss(pred, target_v)
    loss.backward()
    grad_norm = math.sqrt(sum(float(p.grad.detach().float().norm().cpu()) ** 2 for p in adapter.parameters() if p.grad is not None))
    with torch.no_grad():
        for p in adapter.parameters():
            if p.grad is not None:
                p.add_(p.grad, alpha=-0.1)
        out_resid = adapter(x_t, x0, t, anchor_v)
        genes_alt = list(reversed(genes))
        pb_alt = _condition_batch(cfg, genes_alt, batch, device)
        anchor_alt = _forward(model, x_t, t, x0, pb_alt)
        out_resid_alt = adapter(x_t, x0, t, anchor_alt)
        condition_specific = out_resid_alt - out_resid
    row_norms = out_resid.detach().float().norm(dim=1).cpu().numpy()
    return {
        "genes": genes,
        "load_state": {
            "missing": missing,
            "unexpected": unexpected,
            "skipped_shape_mismatch": skipped,
        },
        "initial_noop_max_abs": initial_noop_max_abs,
        "initial_noop_l2": initial_noop_l2,
        "loss": float(loss.detach().cpu()),
        "adapter_grad_norm": grad_norm,
        "one_step_l2": float(out_resid.detach().float().norm().cpu()),
        "one_step_max_abs": float(out_resid.detach().abs().max().cpu()),
        "one_step_nonzero_row_fraction_gt_1e_minus_8": float((row_norms > 1e-8).mean()),
        "condition_specific_l2": float(condition_specific.detach().float().norm().cpu()),
        "condition_specific_max_abs": float(condition_specific.detach().abs().max().cpu()),
    }


def fnum(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def load_forensics() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with FORENSICS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("group") != "internal_val_cross_background_seen_gene_proxy":
                continue
            anchor = fnum(row.get("anchor_pearson_pert"))
            if anchor is None:
                continue
            candidates = [
                fnum(row.get("gene_raw_mean")),
                fnum(row.get("dataset_mean")),
                fnum(row.get("global_mean")),
                fnum(row.get("shrink_k8")),
            ]
            if any(v is None for v in candidates):
                continue
            # Optimistic internal repair target from existing train-only proxy
            # baselines. This is not solver-state performance, only a proxy for
            # whether row-level residual direction is learnable.
            oracle_delta = max([anchor] + [float(v) for v in candidates]) - anchor
            safe_features = [
                math.log1p(float(row.get("gene_train_count") or 0.0)),
                float(row.get("gene_pred_norm") or 0.0),
                float(row.get("dataset_pred_norm") or 0.0),
                float(row.get("global_pred_norm") or 0.0),
                float(row.get("gene_dataset_cosine") or 0.0),
            ]
            rows.append(
                {
                    "dataset": str(row.get("dataset")),
                    "condition": str(row.get("condition")),
                    "anchor": anchor,
                    "anchor_mmd": float(row.get("anchor_mmd_clamped") or 0.0),
                    "oracle_delta": float(oracle_delta),
                    "features": safe_features,
                }
            )
    return rows


def ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    mu = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    xt = (x_train - mu) / sd
    xv = (x_test - mu) / sd
    xt = np.concatenate([np.ones((xt.shape[0], 1)), xt], axis=1)
    xv = np.concatenate([np.ones((xv.shape[0], 1)), xv], axis=1)
    reg = np.eye(xt.shape[1]) * float(alpha)
    reg[0, 0] = 0.0
    beta = np.linalg.solve(xt.T @ xt + reg, xt.T @ y_train)
    return xv @ beta


def bootstrap(vals: list[float], seed: int, n_boot: int = 5000) -> dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return {"ci_low": float("nan"), "ci_high": float("nan"), "p_gt0": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boots = arr[idx].mean(axis=1)
    return {
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
        "p_gt0": float(np.mean(boots > 0)),
    }


def proxy_probe_gate() -> dict[str, Any]:
    rows = load_forensics()
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_ds[row["dataset"]].append(row)
    all_ds = sorted(by_ds)
    pred_rows: list[dict[str, Any]] = []
    for heldout in all_ds:
        train = [r for ds, part in by_ds.items() if ds != heldout for r in part]
        test = by_ds[heldout]
        if not train or not test:
            continue
        x_train = np.asarray([r["features"] for r in train], dtype=float)
        y_train = np.asarray([r["oracle_delta"] for r in train], dtype=float)
        x_test = np.asarray([r["features"] for r in test], dtype=float)
        pred = ridge_fit_predict(x_train, y_train, x_test, alpha=1.0)
        for r, p in zip(test, pred):
            enable = float(p) > 0.01
            # The realized delta is still based on the internal oracle proxy,
            # not a real solver adapter. Penalize false positives with zero
            # lower bound rather than hallucinating model gains.
            delta = float(r["oracle_delta"]) if enable else 0.0
            pred_rows.append({**r, "predicted_delta": float(p), "enabled": enable, "policy_delta": delta})
    deltas = [float(r["policy_delta"]) for r in pred_rows]
    ds_means = {
        ds: float(np.mean([r["policy_delta"] for r in part]))
        for ds, part in defaultdict(list, {ds: [r for r in pred_rows if r["dataset"] == ds] for ds in all_ds}).items()
        if part
    }
    bs = bootstrap(deltas, seed=20260627)
    rng = np.random.default_rng(20260627)
    y = np.asarray([r["oracle_delta"] for r in rows], dtype=float)
    shuffle_means = []
    for _ in range(2000):
        y_perm = rng.permutation(y)
        # simple optimistic shuffle control: same enable fraction, permuted
        # realized deltas.
        enabled = np.asarray([r["enabled"] for r in pred_rows], dtype=bool)
        vals = np.zeros(len(pred_rows), dtype=float)
        vals[enabled] = y_perm[: len(pred_rows)][enabled]
        shuffle_means.append(float(vals.mean()))
    shuffle_arr = np.asarray(shuffle_means, dtype=float)
    return {
        "n_rows": len(rows),
        "n_pred_rows": len(pred_rows),
        "n_datasets": len(all_ds),
        "enabled_fraction": float(np.mean([r["enabled"] for r in pred_rows])) if pred_rows else 0.0,
        "mean_policy_delta": float(np.mean(deltas)) if deltas else 0.0,
        "dataset_min": min(ds_means.values()) if ds_means else 0.0,
        "bootstrap": bs,
        "shuffle_mean": float(shuffle_arr.mean()) if shuffle_means else 0.0,
        "shuffle_p_ge_actual": float(np.mean(shuffle_arr >= (np.mean(deltas) if deltas else 0.0))) if shuffle_means else 1.0,
        "dataset_means": ds_means,
    }


def main() -> None:
    model_gate = model_footprint_gate()
    proxy_gate = proxy_probe_gate()
    reasons: list[str] = []
    if model_gate["initial_noop_max_abs"] > 1e-7:
        reasons.append("initial_noop_drift_gt_1e_minus_7")
    if model_gate["one_step_l2"] <= 1e-5:
        reasons.append("one_step_adapter_footprint_l2_le_1e_minus_5")
    if model_gate["condition_specific_l2"] <= 1e-8:
        reasons.append("condition_specific_footprint_absent")
    if proxy_gate["mean_policy_delta"] < 0.01:
        reasons.append("proxy_mean_delta_lt_0p01")
    if proxy_gate["bootstrap"]["ci_low"] <= 0:
        reasons.append("proxy_bootstrap_ci_low_not_above_0")
    if proxy_gate["dataset_min"] < -0.02:
        reasons.append("proxy_dataset_min_below_minus_0p02")
    if proxy_gate["shuffle_p_ge_actual"] > 0.01:
        reasons.append("proxy_shuffle_not_beaten")
    reasons.append("real_trainonly_mmd_noharm_not_run_no_gpu")
    status = "tracka_solver_state_residual_footprint_gate_fail_no_gpu"
    if not any(r != "real_trainonly_mmd_noharm_not_run_no_gpu" for r in reasons):
        status = "tracka_solver_state_residual_footprint_gate_pass_needs_mmd_noharm_no_gpu"

    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "training_loop": False,
            "dataset_inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection_used": False,
            "trackc_query_used": False,
            "heldout_exact_rows_used_for_selection": False,
        },
        "inputs": {
            "anchor_checkpoint": str(ANCHOR_CKPT),
            "residual_forensics": str(FORENSICS),
            "failure_cases_context": str(FAILURES),
            "crossbg_split": str(CROSSBG_SPLIT),
        },
        "model_footprint_gate": model_gate,
        "proxy_probe_gate": proxy_gate,
        "decision_reasons": reasons,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Track A Solver-State Residual Footprint Gate",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "CPU-only model-path and train/internal proxy gate. No training loop, dataset inference, checkpoint selection, canonical multi selection, Track C query, held-out exact-row selection, or GPU.",
        "",
        "## Model Footprint",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in (
        "initial_noop_max_abs",
        "initial_noop_l2",
        "adapter_grad_norm",
        "one_step_l2",
        "one_step_max_abs",
        "one_step_nonzero_row_fraction_gt_1e_minus_8",
        "condition_specific_l2",
        "condition_specific_max_abs",
    ):
        lines.append(f"| `{key}` | {float(model_gate[key]):.6e} |")
    lines.extend(["", "## Train/Internal Proxy Probe", "", "| metric | value |", "|---|---:|"])
    for key in ("n_rows", "n_datasets", "enabled_fraction", "mean_policy_delta", "dataset_min", "shuffle_p_ge_actual"):
        lines.append(f"| `{key}` | {float(proxy_gate[key]):.6e} |")
    lines.append(f"| `bootstrap_ci_low` | {float(proxy_gate['bootstrap']['ci_low']):.6e} |")
    lines.append(f"| `bootstrap_ci_high` | {float(proxy_gate['bootstrap']['ci_high']):.6e} |")
    lines.extend(["", "## Decision Reasons", ""])
    lines.extend(f"- `{reason}`" for reason in reasons)
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This gate does not authorize GPU. Passing would only allow a stricter train-only MMD/no-harm gate; failing closes the current solver-state residual adapter as an immediate route.",
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
