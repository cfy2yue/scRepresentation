#!/usr/bin/env python3
"""Perturbation-identity residual adapter unit/proxy gate.

CPU/report-only. This gate tests whether a small perturbation-identity residual
adapter is even worth turning into a GPU smoke. It has two parts:

1. A zero-init PyTorch adapter unit check: initial no-op, live gradient path,
   one-step nonzero residual, and gene-id swap footprint.
2. A leave-one-dataset-out frozen-means proxy: estimate same-gene residual
   corrections from other datasets in the same seed/group slice, then apply
   them to the held-out dataset.

No training, inference, checkpoint selection, canonical multi selection, Track C
query, or GPU is used. Passing this gate would still require external audit and
a real implementation smoke before any promotion claim.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path("/data/cyx/1030/scLatent")))

from ops.audit_latentfm_control_radius_residual_clip_preflight_20260627 import (  # noqa: E402
    EPS,
    ROOT,
    dataset_bootstrap_ci_low,
    load_conditions,
    norm,
    pearson_np,
)


REPORTS = ROOT / "reports"
FORENSICS_CSV = REPORTS / "latentfm_xverse_tracka_residual_forensics_conditions_20260622.csv"
OUT_DIR = REPORTS / "perturbation_identity_residual_adapter_unit_gate_20260627"
OUT_ROWS = OUT_DIR / "perturbation_identity_residual_adapter_rows.csv"
OUT_SUMMARY = OUT_DIR / "perturbation_identity_residual_adapter_summary.csv"
OUT_JSON = REPORTS / "latentfm_perturbation_identity_residual_adapter_unit_gate_20260627.json"
OUT_MD = REPORTS / "LATENTFM_PERTURBATION_IDENTITY_RESIDUAL_ADAPTER_UNIT_GATE_20260627.md"

ALPHAS = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
RNG_SEED = 20260627


class ZeroInitGeneResidualAdapter(nn.Module):
    """Tiny gene-conditioned residual head with exactly zero initial output."""

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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def attach_genes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gene_map = load_gene_map()
    out: list[dict[str, Any]] = []
    for row in rows:
        gene = gene_map.get((row["group"], row["dataset"], row["condition"]), row["condition"])
        new = dict(row)
        new["gene"] = gene
        new["target_residual"] = row["gt_effect"] - row["effect"]
        out.append(new)
    return out


def unit_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    torch.manual_seed(RNG_SEED)
    usable = [row for row in rows if norm(row.get("gene"))]
    genes = sorted({row["gene"] for row in usable})
    gene_to_id = {gene: idx for idx, gene in enumerate(genes)}
    batch = usable[: min(64, len(usable))]
    emb_dim = int(batch[0]["effect"].shape[0])
    gene_ids = torch.tensor([gene_to_id[row["gene"]] for row in batch], dtype=torch.long)
    base = torch.tensor(np.stack([row["effect"] for row in batch]), dtype=torch.float32)
    target = torch.tensor(np.stack([row["gt_effect"] for row in batch]), dtype=torch.float32)

    model = ZeroInitGeneResidualAdapter(len(genes), emb_dim)
    with torch.no_grad():
        initial = model(gene_ids)
    initial_max_abs = float(initial.abs().max().item())

    loss = torch.mean((base + model(gene_ids) - target) ** 2)
    loss.backward()
    grad_sq = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_sq += float(torch.sum(param.grad.detach() ** 2).item())
    grad_norm = math.sqrt(grad_sq)

    opt = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        after = model(gene_ids)
        one_step_residual_l2 = float(torch.linalg.norm(after).item())
        swap_gene_ids = gene_ids.roll(1)
        swap_l2 = float(torch.linalg.norm(model(gene_ids) - model(swap_gene_ids)).item())
        residual_mean_l2 = float(torch.linalg.norm(after, dim=1).mean().item())
        swap_fraction = swap_l2 / (float(torch.linalg.norm(after).item()) + EPS)

    checks = {
        "initial_noop": initial_max_abs <= 1e-7,
        "gradient_live": grad_norm > 1e-6,
        "one_step_nonzero": one_step_residual_l2 > 1e-5,
        "swap_footprint": swap_l2 > 1e-5 and swap_fraction >= 0.10,
    }
    return {
        "n_batch": len(batch),
        "n_genes": len(genes),
        "emb_dim": emb_dim,
        "initial_max_abs": initial_max_abs,
        "grad_norm": grad_norm,
        "one_step_residual_l2": one_step_residual_l2,
        "residual_mean_l2": residual_mean_l2,
        "swap_l2": swap_l2,
        "swap_fraction": swap_fraction,
        "checks": checks,
        "passed": all(checks.values()),
    }


def correction_map(train_rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    by_gene: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in train_rows:
        by_gene[row["gene"]].append(row["target_residual"])
    return {gene: np.mean(np.stack(vals, axis=0), axis=0).astype(np.float32) for gene, vals in by_gene.items()}


def proxy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_slice: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_slice[(row["seed"], row["group"])].append(row)

    for (seed, group), slice_rows in sorted(by_slice.items()):
        datasets = sorted({row["dataset"] for row in slice_rows})
        for heldout_dataset in datasets:
            eval_rows = [row for row in slice_rows if row["dataset"] == heldout_dataset]
            train_rows = [row for row in slice_rows if row["dataset"] != heldout_dataset]
            cmap = correction_map(train_rows)
            for row in eval_rows:
                raw_corr = cmap.get(row["gene"])
                has_source = raw_corr is not None
                if raw_corr is None:
                    raw_corr = np.zeros_like(row["effect"])
                for alpha in ALPHAS:
                    corr = float(alpha) * raw_corr
                    candidate_effect = row["effect"] + corr
                    candidate_pred = row["pert"] + candidate_effect
                    candidate_pp = pearson_np(candidate_effect, row["gt_effect"])
                    if candidate_pp is None:
                        continue
                    out.append(
                        {
                            "seed": seed,
                            "group": group,
                            "heldout_dataset": heldout_dataset,
                            "dataset": row["dataset"],
                            "condition": row["condition"],
                            "gene": row["gene"],
                            "alpha": alpha,
                            "has_same_gene_source": has_source,
                            "source_correction_l2": float(np.linalg.norm(raw_corr)),
                            "base_pp": row["base_pp"],
                            "candidate_pp": candidate_pp,
                            "delta_pp": candidate_pp - float(row["base_pp"]),
                            "base_endpoint_mse": row["base_endpoint_mse"],
                            "candidate_endpoint_mse": float(np.mean((candidate_pred - row["gt"]) ** 2)),
                            "endpoint_mse_delta": float(np.mean((candidate_pred - row["gt"]) ** 2)) - float(row["base_endpoint_mse"]),
                            "mmd_original": row["mmd_original"],
                            "hard_tail": row["hard_tail"],
                        }
                    )
    return out


def summarize(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for seed in sorted({row["seed"] for row in candidates}):
        for group in sorted({row["group"] for row in candidates if row["seed"] == seed}):
            for alpha in ALPHAS:
                sub = [row for row in candidates if row["seed"] == seed and row["group"] == group and row["alpha"] == alpha]
                if not sub:
                    continue
                covered = [row for row in sub if row["has_same_gene_source"]]
                hard = [row for row in sub if row["hard_tail"]]
                covered_hard = [row for row in covered if row["hard_tail"]]
                per_dataset: dict[str, float] = {}
                for dataset in sorted({row["dataset"] for row in sub}):
                    vals = [float(row["delta_pp"]) for row in sub if row["dataset"] == dataset]
                    if vals:
                        per_dataset[dataset] = mean(vals)
                summaries.append(
                    {
                        "seed": seed,
                        "group": group,
                        "alpha": alpha,
                        "n": len(sub),
                        "datasets": len(per_dataset),
                        "covered_n": len(covered),
                        "covered_gene_count": len({row["gene"] for row in covered}),
                        "covered_dataset_count": len({row["dataset"] for row in covered}),
                        "changed_condition_frac": len(covered) / len(sub),
                        "mean_delta_pp": mean(float(row["delta_pp"]) for row in sub),
                        "covered_mean_delta_pp": mean(float(row["delta_pp"]) for row in covered) if covered else None,
                        "hard_tail_delta_pp": mean(float(row["delta_pp"]) for row in hard) if hard else None,
                        "covered_hard_tail_delta_pp": mean(float(row["delta_pp"]) for row in covered_hard) if covered_hard else None,
                        "endpoint_mse_delta_mean": mean(float(row["endpoint_mse_delta"]) for row in sub),
                        "covered_endpoint_mse_delta_mean": mean(float(row["endpoint_mse_delta"]) for row in covered) if covered else None,
                        "dataset_min_delta_pp": min(per_dataset.values()) if per_dataset else None,
                        "dataset_bootstrap_ci_low": dataset_bootstrap_ci_low(sub, "delta_pp"),
                    }
                )
    return summaries


def choose(summaries: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str, list[str]]:
    by_alpha: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        by_alpha[float(row["alpha"])].append(row)
    candidates: list[dict[str, Any]] = []
    for alpha, rows in sorted(by_alpha.items()):
        if len(rows) < 4:
            continue
        candidates.append(
            {
                "alpha": alpha,
                "slices": rows,
                "min_changed_condition_frac": min(float(row["changed_condition_frac"]) for row in rows),
                "max_changed_condition_frac": max(float(row["changed_condition_frac"]) for row in rows),
                "worst_mean_delta_pp": min(float(row["mean_delta_pp"]) for row in rows),
                "best_mean_delta_pp": max(float(row["mean_delta_pp"]) for row in rows),
                "worst_covered_mean_delta_pp": min(float(row["covered_mean_delta_pp"] if row["covered_mean_delta_pp"] is not None else -999.0) for row in rows),
                "worst_hard_tail_delta_pp": min(float(row["hard_tail_delta_pp"] if row["hard_tail_delta_pp"] is not None else -999.0) for row in rows),
                "worst_dataset_min_delta_pp": min(float(row["dataset_min_delta_pp"] if row["dataset_min_delta_pp"] is not None else -999.0) for row in rows),
                "worst_ci_low": min(float(row["dataset_bootstrap_ci_low"] if row["dataset_bootstrap_ci_low"] is not None else -999.0) for row in rows),
                "max_endpoint_mse_delta": max(float(row["endpoint_mse_delta_mean"]) for row in rows),
                "worst_covered_endpoint_mse_delta": max(float(row["covered_endpoint_mse_delta_mean"] if row["covered_endpoint_mse_delta_mean"] is not None else 999.0) for row in rows),
            }
        )
    if not candidates:
        return None, "perturbation_identity_residual_adapter_unit_proxy_fail_no_gpu", ["no_complete_candidate"]
    best = max(
        candidates,
        key=lambda row: (
            row["worst_covered_mean_delta_pp"],
            row["worst_dataset_min_delta_pp"],
            row["worst_mean_delta_pp"],
            -row["max_endpoint_mse_delta"],
        ),
    )
    reasons = []
    if best["min_changed_condition_frac"] < 0.05:
        reasons.append("same_gene_lodo_coverage_below_0p05")
    if best["worst_mean_delta_pp"] < -0.002:
        reasons.append("worst_internal_mean_delta_pp_below_minus_0p002")
    if best["worst_covered_mean_delta_pp"] < 0.01:
        reasons.append("covered_same_gene_mean_delta_pp_below_0p01")
    if best["worst_hard_tail_delta_pp"] < 0.0:
        reasons.append("worst_hard_tail_delta_pp_negative")
    if best["worst_dataset_min_delta_pp"] < -0.01:
        reasons.append("dataset_min_delta_pp_below_minus_0p01")
    if best["worst_ci_low"] < -0.002:
        reasons.append("dataset_bootstrap_ci_low_below_minus_0p002")
    if best["max_endpoint_mse_delta"] > 0:
        reasons.append("endpoint_mse_surrogate_harm")
    status = (
        "perturbation_identity_residual_adapter_unit_proxy_pass_external_audit_only_no_gpu"
        if not reasons
        else "perturbation_identity_residual_adapter_unit_proxy_fail_no_gpu"
    )
    return best, status, reasons


def permutation_pvalue(rows: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    rng = random.Random(RNG_SEED)
    alpha_rows = [row for row in rows if abs(float(row["alpha"]) - float(alpha)) < 1e-12]
    observed_all = mean(float(row["delta_pp"]) for row in alpha_rows) if alpha_rows else 0.0
    observed_covered_rows = [row for row in alpha_rows if row["has_same_gene_source"]]
    observed_covered = mean(float(row["delta_pp"]) for row in observed_covered_rows) if observed_covered_rows else 0.0
    null_all: list[float] = []
    null_covered: list[float] = []

    # Rebuild a row-level null by shuffling candidate deltas among covered genes
    # inside each seed/group slice. This tests whether the signed covered signal
    # is stronger than arbitrary same-slice residual corrections.
    by_slice: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in alpha_rows:
        by_slice[(row["seed"], row["group"])].append(row)
    for _ in range(500):
        vals_all: list[float] = []
        vals_cov: list[float] = []
        for sub in by_slice.values():
            covered_deltas = [float(row["delta_pp"]) for row in sub if row["has_same_gene_source"]]
            if not covered_deltas:
                vals_all.extend(0.0 for _ in sub)
                continue
            shuffled = covered_deltas[:]
            rng.shuffle(shuffled)
            idx = 0
            for row in sub:
                if row["has_same_gene_source"]:
                    val = shuffled[idx % len(shuffled)]
                    idx += 1
                    vals_cov.append(val)
                    vals_all.append(val)
                else:
                    vals_all.append(0.0)
        null_all.append(mean(vals_all) if vals_all else 0.0)
        null_covered.append(mean(vals_cov) if vals_cov else 0.0)
    p_all = (1 + sum(v >= observed_all for v in null_all)) / (len(null_all) + 1)
    p_cov = (1 + sum(v >= observed_covered for v in null_covered)) / (len(null_covered) + 1)
    return {
        "n_perm": len(null_all),
        "observed_mean_delta_pp": observed_all,
        "observed_covered_mean_delta_pp": observed_covered,
        "p_greater_all": p_all,
        "p_greater_covered": p_cov,
        "null_all_mean": mean(null_all) if null_all else None,
        "null_covered_mean": mean(null_covered) if null_covered else None,
    }


def write_report(
    *,
    status: str,
    reasons: list[str],
    unit: dict[str, Any],
    best: dict[str, Any] | None,
    perm: dict[str, Any] | None,
    summaries: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    best_line = "No complete alpha candidate."
    if best is not None:
        best_line = (
            f"alpha `{best['alpha']}`; changed-condition frac "
            f"`{best['min_changed_condition_frac']:.4f}-{best['max_changed_condition_frac']:.4f}`; "
            f"worst mean delta `{best['worst_mean_delta_pp']:.6f}`; "
            f"worst covered mean delta `{best['worst_covered_mean_delta_pp']:.6f}`; "
            f"worst dataset min delta `{best['worst_dataset_min_delta_pp']:.6f}`; "
            f"worst CI low `{best['worst_ci_low']:.6f}`; "
            f"max endpoint-MSE delta `{best['max_endpoint_mse_delta']:.6g}`."
        )
    covered_rows = [row for row in rows if row["has_same_gene_source"] and abs(float(row["alpha"]) - (float(best["alpha"]) if best else -1.0)) < 1e-12]
    covered_genes = sorted({row["gene"] for row in covered_rows})
    lines = [
        "# LatentFM Perturbation-Identity Residual Adapter Unit/Proxy Gate",
        "",
        f"Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M %Z')}`",
        "",
        f"Status: `{status}`",
        "",
        "## Scope",
        "",
        "CPU/report-only. Uses frozen xverse 8k internal condition means and "
        "leave-one-dataset-out same-gene residual corrections. It does not use "
        "canonical multi for Track A selection, Track C query, GPU, training, "
        "inference, or checkpoint selection.",
        "",
        "## Unit Gate",
        "",
        f"- initial max abs: `{unit['initial_max_abs']:.6g}`",
        f"- gradient norm: `{unit['grad_norm']:.6g}`",
        f"- one-step residual L2: `{unit['one_step_residual_l2']:.6g}`",
        f"- swap L2 / fraction: `{unit['swap_l2']:.6g}` / `{unit['swap_fraction']:.6g}`",
        f"- checks: `{unit['checks']}`",
        "",
        "## LODO Proxy",
        "",
        best_line,
        "",
        f"Covered rows at best alpha: `{len(covered_rows)}`; covered genes: `{len(covered_genes)}`.",
        "",
        "Top covered genes:",
        "",
        ", ".join(covered_genes[:20]) if covered_genes else "None.",
        "",
        "Permutation control:",
        "",
        f"`{perm}`" if perm is not None else "Not available.",
        "",
        "## Decision",
        "",
        f"Fail/pass reasons: `{reasons}`",
        "",
    ]
    if status.endswith("fail_no_gpu"):
        lines.extend(
            [
                "No GPU is authorized. If this failed on coverage, pure gene/condition-id "
                "residual adapters are unlikely to solve cross-background generalization "
                "without a richer prior or explicit support signal.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "This is not a promotion claim. Passing only authorizes external audit "
                "and design of a bounded real adapter smoke with frozen split boundaries.",
                "",
            ]
        )
    lines.extend(
        [
            "## Outputs",
            "",
            f"- Rows: `{OUT_ROWS}`",
            f"- Summary: `{OUT_SUMMARY}`",
            f"- JSON: `{OUT_JSON}`",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    base_rows = attach_genes(load_conditions())
    unit = unit_check(base_rows)
    rows = proxy_rows(base_rows)
    summaries = summarize(rows)
    best, status, reasons = choose(summaries)
    if not unit["passed"]:
        status = "perturbation_identity_residual_adapter_unit_proxy_fail_no_gpu"
        reasons = ["unit_gate_failed", *reasons]
    perm = permutation_pvalue(rows, float(best["alpha"])) if best is not None else None
    if perm is not None and perm["p_greater_covered"] > 0.05:
        if "permutation_p_greater_covered_above_0p05" not in reasons:
            reasons.append("permutation_p_greater_covered_above_0p05")
        status = "perturbation_identity_residual_adapter_unit_proxy_fail_no_gpu"

    write_csv(
        OUT_ROWS,
        rows,
        [
            "seed",
            "group",
            "heldout_dataset",
            "dataset",
            "condition",
            "gene",
            "alpha",
            "has_same_gene_source",
            "source_correction_l2",
            "base_pp",
            "candidate_pp",
            "delta_pp",
            "base_endpoint_mse",
            "candidate_endpoint_mse",
            "endpoint_mse_delta",
            "mmd_original",
            "hard_tail",
        ],
    )
    write_csv(
        OUT_SUMMARY,
        summaries,
        [
            "seed",
            "group",
            "alpha",
            "n",
            "datasets",
            "covered_n",
            "covered_gene_count",
            "covered_dataset_count",
            "changed_condition_frac",
            "mean_delta_pp",
            "covered_mean_delta_pp",
            "hard_tail_delta_pp",
            "covered_hard_tail_delta_pp",
            "endpoint_mse_delta_mean",
            "covered_endpoint_mse_delta_mean",
            "dataset_min_delta_pp",
            "dataset_bootstrap_ci_low",
        ],
    )
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
        "status": status,
        "gpu_authorized": False,
        "reasons": reasons,
        "unit": unit,
        "best": best,
        "permutation_control": perm,
        "n_rows": len(rows),
        "n_summaries": len(summaries),
        "inputs": {
            "internal_means": [
                str(REPORTS / "latentfm_xverse_8k_seed_ensemble_internal_means_20260627/seed42_internal_split_group_means_evalseed42.json"),
                str(REPORTS / "latentfm_xverse_8k_seed_ensemble_internal_means_20260627/seed43_internal_split_group_means_evalseed42.json"),
            ],
            "forensics_csv": str(FORENSICS_CSV),
        },
        "outputs": {
            "rows": str(OUT_ROWS),
            "summary": str(OUT_SUMMARY),
            "report": str(OUT_MD),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(
        status=status,
        reasons=reasons,
        unit=unit,
        best=best,
        perm=perm,
        summaries=summaries,
        rows=rows,
    )
    print(json.dumps({"status": status, "reasons": reasons, "best": best, "unit_passed": unit["passed"]}, indent=2))


if __name__ == "__main__":
    main()
