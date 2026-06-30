#!/usr/bin/env python3
"""CPU-only gate for HVG/full-gene downstream scaling design.

This report answers a narrow orchestration question: after ZSCAPE shows a
biologically meaningful HVG/full-gene response-energy curve, can we translate
that into an immediate LatentFM/xVERSE GPU experiment, or must it first become
a raw-expression/rawFM design gate?
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "reports/hvg_fullgene_downstream_design_gate_20260628"
OUT_MD = ROOT / "reports/LATENTFM_HVG_FULLGENE_DOWNSTREAM_DESIGN_GATE_20260628.md"
OUT_JSON = ROOT / "reports/latentfm_hvg_fullgene_downstream_design_gate_20260628.json"
ZSCAPE_CURVE = ROOT / "reports/zscape_hvg_fullgene_information_axis_20260628/zscape_hvg_fullgene_information_curve.csv"
ZSCAPE_SUMMARY = ROOT / "reports/zscape_hvg_fullgene_information_axis_20260628/zscape_hvg_response_energy_summary.csv"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt_float(x: Any, digits: int = 4) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def latent_h5_schema(path: Path, model: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model,
        "dataset": path.stem,
        "path": str(path),
        "exists": path.is_file(),
        "has_gene_matrix": False,
        "has_var_names": False,
        "has_ctrl_emb": False,
        "has_gt_emb": False,
        "ctrl_emb_dim": "",
        "gt_emb_dim": "",
        "condition_count": "",
    }
    if not path.is_file():
        return row
    with h5py.File(path, "r") as h5:
        row["condition_count"] = int(h5["conditions"].shape[0]) if "conditions" in h5 else ""
        if "ctrl/emb" in h5:
            row["has_ctrl_emb"] = True
            row["ctrl_emb_dim"] = int(h5["ctrl/emb"].shape[1]) if len(h5["ctrl/emb"].shape) > 1 else ""
        if "gt/emb" in h5:
            row["has_gt_emb"] = True
            row["gt_emb_dim"] = int(h5["gt/emb"].shape[1]) if len(h5["gt/emb"].shape) > 1 else ""
        for key in ("X", "raw/X", "counts", "var_names", "genes", "gene_names"):
            if key in h5:
                if key in ("X", "raw/X", "counts"):
                    row["has_gene_matrix"] = True
                else:
                    row["has_var_names"] = True
    return row


def h5ad_schema(path: Path, group: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "group": group,
        "dataset": path.stem.replace("__single", "").replace("__multiple", ""),
        "path": str(path),
        "exists": path.is_file(),
        "n_obs": "",
        "n_vars": "",
        "has_emb": False,
        "has_counts_layer": False,
        "var_name_sample": "",
        "obs_cols_sample": "",
    }
    if not path.is_file():
        return row
    import anndata as ad

    adata = ad.read_h5ad(path, backed="r")
    try:
        row["n_obs"] = int(adata.n_obs)
        row["n_vars"] = int(adata.n_vars)
        row["has_emb"] = "emb" in adata.obsm
        row["has_counts_layer"] = "counts" in adata.layers
        row["var_name_sample"] = ";".join(map(str, list(adata.var_names[:5])))
        row["obs_cols_sample"] = ";".join(map(str, list(adata.obs.columns[:8])))
    finally:
        adata.file.close()
    return row


def collect_latent_schema() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in ("xverse", "scfoundation", "scldm", "stack"):
        root = ROOT / "dataset/latentfm_full" / model
        for path in sorted(root.glob("*.h5")):
            if path.name in {"ctrl_means.npz", "pert_means.npz"}:
                continue
            rows.append(latent_h5_schema(path, model))
    return rows


def collect_h5ad_schema() -> list[dict[str, Any]]:
    specs = [
        ("raw_genepert_DE5000", ROOT / "dataset/raw/genepert_DE5000"),
        ("raw_chemicalpert_DE5000", ROOT / "dataset/raw/chemicalpert_DE5000"),
        ("raw_genepert_bench", ROOT / "dataset/raw/genepert_bench"),
        ("raw_chemicalpert_bench", ROOT / "dataset/raw/chemicalpert_bench"),
        ("training_scfoundation_gt", ROOT / "dataset/Training_data/scfoundation/gt_scfoundation"),
        ("training_scfoundation_control", ROOT / "dataset/Training_data/scfoundation/control_scfoundation"),
        ("training_scldm_gt", ROOT / "dataset/Training_data/scldm/gt_scldm"),
        ("training_scldm_control", ROOT / "dataset/Training_data/scldm/control_scldm"),
    ]
    rows: list[dict[str, Any]] = []
    for group, root in specs:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.h5ad")):
            rows.append(h5ad_schema(path, group))
    return rows


def summarize_h5ad(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group"])].append(row)
    out = []
    for group, part in sorted(grouped.items()):
        n_vars = [int(r["n_vars"]) for r in part if str(r.get("n_vars", "")).isdigit()]
        n_obs = [int(r["n_obs"]) for r in part if str(r.get("n_obs", "")).isdigit()]
        out.append(
            {
                "group": group,
                "files": len(part),
                "n_obs_min": min(n_obs) if n_obs else "",
                "n_obs_max": max(n_obs) if n_obs else "",
                "n_vars_min": min(n_vars) if n_vars else "",
                "n_vars_median": sorted(n_vars)[len(n_vars) // 2] if n_vars else "",
                "n_vars_max": max(n_vars) if n_vars else "",
                "files_with_emb": sum(1 for r in part if r.get("has_emb") is True),
                "files_with_counts_layer": sum(1 for r in part if r.get("has_counts_layer") is True),
            }
        )
    return out


def summarize_latent(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    out = []
    for model, part in sorted(grouped.items()):
        dims = Counter(str(r.get("ctrl_emb_dim", "")) for r in part if r.get("ctrl_emb_dim") != "")
        out.append(
            {
                "model": model,
                "files": len(part),
                "emb_dims": ",".join(f"{k}:{v}" for k, v in sorted(dims.items())),
                "files_with_ctrl_emb": sum(1 for r in part if r.get("has_ctrl_emb") is True),
                "files_with_gt_emb": sum(1 for r in part if r.get("has_gt_emb") is True),
                "files_with_gene_matrix": sum(1 for r in part if r.get("has_gene_matrix") is True),
                "files_with_var_names": sum(1 for r in part if r.get("has_var_names") is True),
            }
        )
    return out


def find_zscape_values(curve: list[dict[str, str]]) -> dict[str, str]:
    by_k = {int(float(r["top_genes"])): r for r in curve}
    out = {}
    for k in (500, 1000, 2000, 4000, 8000, 16000, 32031):
        row = by_k.get(k)
        if not row:
            continue
        out[f"hvg{k}_primary_mean"] = row.get("primary_rows_response_energy_share_mean", "")
        out[f"hvg{k}_primary_min"] = row.get("primary_rows_response_energy_share_min", "")
        out[f"hvg{k}_gene_fraction"] = row.get("gene_fraction", "")
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zscape_curve = read_csv(ZSCAPE_CURVE)
    zscape_summary = read_csv(ZSCAPE_SUMMARY)
    latent_rows = collect_latent_schema()
    h5ad_rows = collect_h5ad_schema()
    latent_summary = summarize_latent(latent_rows)
    h5ad_summary = summarize_h5ad(h5ad_rows)
    zscape_vals = find_zscape_values(zscape_curve)

    current_latent_supports_gene_budget = any(r["files_with_gene_matrix"] for r in latent_summary)
    h5ad_gene_counts = [
        int(r["n_vars"])
        for r in h5ad_rows
        if str(r.get("n_vars", "")).isdigit() and ("DE5000" in str(r.get("group")) or "bench" in str(r.get("group")))
    ]
    expression_matrices_available = bool(h5ad_gene_counts)
    full_gene_like_available = bool(h5ad_gene_counts and max(h5ad_gene_counts) > 8000)

    reasons = []
    gpu_authorized = False
    status = "hvg_fullgene_design_cpu_gate_no_gpu"
    if not current_latent_supports_gene_budget:
        reasons.append("current_latentfm_h5_files_are_embedding_only")
    if expression_matrices_available and not full_gene_like_available:
        reasons.append("local_downstream_h5ad_panels_are_gene_limited_not_true_fullgene")
    if not expression_matrices_available:
        reasons.append("no_local_expression_matrix_route_found")

    next_action = (
        "build a CPU raw-expression condition-mean/HVG-budget predictability gate "
        "before any GPU; do not launch current LatentFM/xVERSE HVG-budget training"
    )

    payload = {
        "timestamp": now_cst(),
        "status": status,
        "gpu_authorized": gpu_authorized,
        "decision_reasons": reasons,
        "next_action": next_action,
        "zscape_values": zscape_vals,
        "latent_summary": latent_summary,
        "h5ad_summary": h5ad_summary,
        "inputs": {
            "zscape_curve": str(ZSCAPE_CURVE),
            "zscape_summary": str(ZSCAPE_SUMMARY),
        },
    }

    write_csv(
        OUT_DIR / "latent_embedding_schema_rows.csv",
        latent_rows,
        [
            "model",
            "dataset",
            "path",
            "has_ctrl_emb",
            "has_gt_emb",
            "ctrl_emb_dim",
            "gt_emb_dim",
            "condition_count",
            "has_gene_matrix",
            "has_var_names",
        ],
    )
    write_csv(
        OUT_DIR / "latent_embedding_schema_summary.csv",
        latent_summary,
        [
            "model",
            "files",
            "emb_dims",
            "files_with_ctrl_emb",
            "files_with_gt_emb",
            "files_with_gene_matrix",
            "files_with_var_names",
        ],
    )
    write_csv(
        OUT_DIR / "expression_h5ad_schema_rows.csv",
        h5ad_rows,
        [
            "group",
            "dataset",
            "path",
            "n_obs",
            "n_vars",
            "has_emb",
            "has_counts_layer",
            "var_name_sample",
            "obs_cols_sample",
        ],
    )
    write_csv(
        OUT_DIR / "expression_h5ad_schema_summary.csv",
        h5ad_summary,
        [
            "group",
            "files",
            "n_obs_min",
            "n_obs_max",
            "n_vars_min",
            "n_vars_median",
            "n_vars_max",
            "files_with_emb",
            "files_with_counts_layer",
        ],
    )
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM HVG/Full-Gene Downstream Design Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only design gate.",
        "- Does not train, infer, read canonical multi for selection, read Track C query, or use GPU.",
        "- Asks whether ZSCAPE HVG/full-gene biology can be translated into an immediate LatentFM/xVERSE GPU experiment.",
        "",
        "## ZSCAPE Signal",
        "",
        "| gene budget | gene fraction | primary response mean | primary response min |",
        "|---:|---:|---:|---:|",
    ]
    for k in (500, 1000, 2000, 4000, 8000, 16000, 32031):
        if f"hvg{k}_primary_mean" not in zscape_vals:
            continue
        lines.append(
            f"| {k} | {fmt_float(zscape_vals[f'hvg{k}_gene_fraction'])} | "
            f"{fmt_float(zscape_vals[f'hvg{k}_primary_mean'])} | "
            f"{fmt_float(zscape_vals[f'hvg{k}_primary_min'])} |"
        )
    lines += [
        "",
        "Interpretation: ZSCAPE supports a compact-but-not-zero residual gene-information axis.",
        "Top 2k genes capture most primary response energy, while top 8k is close to saturation.",
        "",
        "## Current LatentFM Data Schema",
        "",
        "| model | files | emb dims | files with ctrl/gt emb | files with gene matrix | files with var names |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in latent_summary:
        lines.append(
            f"| `{row['model']}` | {row['files']} | `{row['emb_dims']}` | "
            f"{row['files_with_ctrl_emb']}/{row['files_with_gt_emb']} | "
            f"{row['files_with_gene_matrix']} | {row['files_with_var_names']} |"
        )
    lines += [
        "",
        "Current LatentFM train artifacts are embedding-only. They do not expose a gene/token budget knob.",
        "",
        "## Local Expression Matrices",
        "",
        "| group | files | n_obs range | n_vars min/median/max | emb files | counts-layer files |",
        "|---|---:|---|---|---:|---:|",
    ]
    for row in h5ad_summary:
        lines.append(
            f"| `{row['group']}` | {row['files']} | {row['n_obs_min']}-{row['n_obs_max']} | "
            f"{row['n_vars_min']}/{row['n_vars_median']}/{row['n_vars_max']} | "
            f"{row['files_with_emb']} | {row['files_with_counts_layer']} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "- Do not launch an immediate LatentFM/xVERSE HVG-budget GPU smoke.",
        "- Reason: current `dataset/latentfm_full/*/*.h5` files contain only fixed scFM embeddings, not gene matrices or var names.",
        "- Local downstream h5ad matrices are useful for expression/rawFM preflights, but most available benchmark panels are already gene-limited rather than true full-gene.",
        "- ZSCAPE Danio full-gene evidence remains biological/information-axis support, not a direct human LatentFM train input.",
        "",
        "## Next Gate",
        "",
        next_action,
        "",
        "A valid next CPU gate should materialize condition-level expression means for matched local h5ad panels, compute HVG500/1k/2k/all response-energy and predictability curves under train-only splits, and require shuffle/source/background controls before any model smoke.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- latent schema rows: `{OUT_DIR / 'latent_embedding_schema_rows.csv'}`",
        f"- latent schema summary: `{OUT_DIR / 'latent_embedding_schema_summary.csv'}`",
        f"- expression h5ad rows: `{OUT_DIR / 'expression_h5ad_schema_rows.csv'}`",
        f"- expression h5ad summary: `{OUT_DIR / 'expression_h5ad_schema_summary.csv'}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
