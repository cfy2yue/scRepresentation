#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from audit_zscape_expression_ot_strict_controls_20260628 import (
    assignment_ot,
    control_only_embed,
    embryo_balanced_positions,
    greedy_match,
    make_cell_level_manifest,
    quantile_or_nan,
    read_cell_index,
    stable_seed,
)


def utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def load_manifest(counts: sp.spmatrix, cell_index_path: Path, matched_manifest_path: Path, args: argparse.Namespace):
    cell_index = read_cell_index(cell_index_path)
    manifest = pd.read_csv(matched_manifest_path)
    manifest["expression_col_index"] = manifest["cell"].map(cell_index)
    manifest = manifest.dropna(subset=["expression_col_index"]).copy()
    manifest["expression_col_index"] = manifest["expression_col_index"].astype(int)
    manifest = manifest.set_index("expression_col_index", drop=False)
    cell_manifest = make_cell_level_manifest(manifest, counts.shape[1])
    emb, libraries, embed_meta = control_only_embed(counts, cell_manifest, args.n_hvg, args.n_pca, args.seed)
    manifest["expression_library"] = libraries[manifest.index.to_numpy(dtype=int)]
    manifest["log_library"] = np.log1p(pd.to_numeric(manifest["expression_library"], errors="coerce").fillna(0))
    for col in ["manifest_timepoint", "timepoint", "n_umi", "num_genes_expressed"]:
        if col in manifest.columns:
            manifest[col] = pd.to_numeric(manifest[col], errors="coerce")
    return manifest, emb, embed_meta


def real_row_metrics(row_df: pd.DataFrame) -> dict[str, Any]:
    gate = row_df.get("strict_row_gate", pd.Series(dtype=bool)).fillna(False).astype(bool)
    ratio = pd.to_numeric(row_df.get("effect_ratio_vs_max_null_p95", pd.Series(dtype=float)), errors="coerce")
    return {
        "real_n_rows": int(len(row_df)),
        "real_n_pass_rows": int(gate.sum()) if len(gate) else 0,
        "real_pass_fraction": float(gate.mean()) if len(gate) else float("nan"),
        "real_mean_ratio": float(ratio.mean()) if ratio.notna().any() else float("nan"),
    }


def build_placebo_pool(row_group: pd.DataFrame, manifest: pd.DataFrame, mode: str) -> pd.DataFrame:
    first = row_group.iloc[0]
    lineage = str(first["manifest_cell_type_broad"])
    target = str(first["manifest_gene_target"])
    timepoint = float(first["manifest_timepoint"])
    pool = manifest[
        (manifest["selection_role"] == "control")
        & (manifest["manifest_cell_type_broad"].astype(str) == lineage)
    ].copy()
    if mode == "wrong_target_same_time":
        pool = pool[(pool["manifest_timepoint"] == timepoint) & (pool["manifest_gene_target"].astype(str) != target)]
    elif mode == "wrong_time_any_target":
        pool = pool[pool["manifest_timepoint"] != timepoint]
    elif mode == "wrong_target_or_time":
        pool = pool[
            (pool["manifest_timepoint"] != timepoint)
            | (pool["manifest_gene_target"].astype(str) != target)
        ]
    else:
        raise ValueError(f"Unknown placebo mode: {mode}")
    return pool.drop_duplicates("cell")


def placebo_for_row(
    row_id: str,
    row_group: pd.DataFrame,
    manifest: pd.DataFrame,
    emb: np.ndarray,
    seed: int,
    ot_cells: int,
    repeats: int,
    mode: str,
) -> dict[str, Any]:
    p_all = row_group[row_group["selection_role"] == "perturb"].drop_duplicates("cell")
    c_real = row_group[row_group["selection_role"] == "control"].drop_duplicates("cell")
    placebo_pool = build_placebo_pool(row_group, manifest, mode)
    n = min(ot_cells, len(p_all), len(c_real), len(placebo_pool))
    if n < 40:
        return {
            "row_id": row_id,
            "placebo_mode": mode,
            "status": "too_few_cells",
            "n_perturb": int(len(p_all)),
            "n_real_control": int(len(c_real)),
            "n_placebo_pool": int(len(placebo_pool)),
        }
    placebo_values: list[float] = []
    real_values: list[float] = []
    ratios: list[float] = []
    for rep in range(repeats):
        p_pos = embryo_balanced_positions(p_all, n, seed, f"{row_id}|{mode}|p|{rep}")
        p_sample = p_all.loc[p_pos]
        real_pos = greedy_match(p_sample, c_real, n, seed, f"{row_id}|{mode}|real|{rep}", prefer_subtype=True)
        placebo_pos = greedy_match(p_sample, placebo_pool, n, seed, f"{row_id}|{mode}|placebo|{rep}", prefer_subtype=True)
        m = min(len(p_pos), len(real_pos), len(placebo_pos), n)
        if m < 40:
            continue
        real_ot = assignment_ot(emb[p_pos[:m]], emb[real_pos[:m]])
        placebo_ot = assignment_ot(emb[p_pos[:m]], emb[placebo_pos[:m]])
        real_values.append(real_ot)
        placebo_values.append(placebo_ot)
        ratios.append(real_ot / max(placebo_ot, 1e-8))
    return {
        "row_id": row_id,
        "placebo_mode": mode,
        "status": "ok" if placebo_values else "no_valid_repeats",
        "n": int(n),
        "repeats": int(len(placebo_values)),
        "real_ot_median": float(np.median(real_values)) if real_values else float("nan"),
        "placebo_ot_median": float(np.median(placebo_values)) if placebo_values else float("nan"),
        "placebo_ot_p95": quantile_or_nan(placebo_values, 0.95),
        "real_vs_placebo_ratio_median": float(np.median(ratios)) if ratios else float("nan"),
        "real_beats_placebo_fraction": float(np.mean(np.array(real_values) > np.array(placebo_values))) if placebo_values else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts-npz", type=Path, required=True)
    parser.add_argument("--cell-index", type=Path, required=True)
    parser.add_argument("--matched-manifest", type=Path, required=True)
    parser.add_argument("--fixedcell-row-results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--n-pca", type=int, default=32)
    parser.add_argument("--ot-cells", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--min-real-pass-rows", type=int, default=3)
    parser.add_argument("--min-pass-margin", type=float, default=0.40)
    parser.add_argument("--min-ratio-margin", type=float, default=0.05)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fixed_rows = pd.read_csv(args.fixedcell_row_results)
    periderm_real = fixed_rows[fixed_rows["cell_type_broad"].astype(str) == "periderm"].copy()
    real = real_row_metrics(periderm_real)

    counts = sp.load_npz(args.counts_npz)
    manifest, emb, embed_meta = load_manifest(counts, args.cell_index, args.matched_manifest, args)
    periderm_row_ids = sorted(periderm_real["row_id"].astype(str).unique().tolist())
    placebo_rows: list[dict[str, Any]] = []
    for row_id in periderm_row_ids:
        group = manifest[manifest["row_id"].astype(str) == row_id]
        for mode in ["wrong_target_same_time", "wrong_time_any_target", "wrong_target_or_time"]:
            placebo_rows.append(
                placebo_for_row(row_id, group, manifest, emb, args.seed, args.ot_cells, args.repeats, mode)
            )
    placebo_df = pd.DataFrame(placebo_rows)
    ok = placebo_df[placebo_df["status"] == "ok"].copy()
    placebo_pass_fraction_proxy = float((ok["real_beats_placebo_fraction"] >= 0.95).mean()) if len(ok) else float("nan")
    placebo_ratio_p95 = float(ok["real_vs_placebo_ratio_median"].quantile(0.95)) if len(ok) else float("nan")
    real_minus_placebo_ratio = real["real_mean_ratio"] - placebo_ratio_p95
    real_minus_placebo_pass = real["real_pass_fraction"] - placebo_pass_fraction_proxy
    reasons: list[str] = []
    if real["real_n_pass_rows"] < args.min_real_pass_rows:
        reasons.append("real_periderm_fixedcell_pass_rows_below_min")
    if real_minus_placebo_pass < args.min_pass_margin:
        reasons.append("real_pass_fraction_not_above_placebo_proxy_margin")
    if real_minus_placebo_ratio < args.min_ratio_margin:
        reasons.append("real_mean_ratio_not_above_placebo_p95_margin")
    status = (
        "zscape_periderm_placebo_control_pass_no_gpu"
        if not reasons
        else "zscape_periderm_placebo_control_fail_or_partial_no_gpu"
    )

    placebo_csv = args.out_dir / "zscape_periderm_placebo_rows.csv"
    placebo_df.to_csv(placebo_csv, index=False)
    json_path = args.out_dir / "zscape_periderm_placebo_control_20260628.json"
    summary = {
        "timestamp_utc": utc_now(),
        "status": status,
        "gpu_authorized": False,
        "reasons": reasons,
        "real_metrics": real,
        "placebo_pass_fraction_proxy": placebo_pass_fraction_proxy,
        "placebo_ratio_p95": placebo_ratio_p95,
        "real_minus_placebo_pass": real_minus_placebo_pass,
        "real_minus_placebo_ratio": real_minus_placebo_ratio,
        "embed_meta": embed_meta,
        "placebo_csv": str(placebo_csv),
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    md_path = args.out_dir / "LATENTFM_ZSCAPE_PERIDERM_PLACEBO_CONTROL_20260628.md"
    lines = [
        "# LatentFM ZSCAPE Periderm Placebo Control",
        "",
        f"Timestamp: `{utc_now()}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only placebo follow-up for a positive fixed-cell periderm gate.",
        "- Uses the same selected ZSCAPE raw counts and control-only HVG/SVD.",
        "- Does not train, infer, run scFM embeddings, read canonical multi, or read Track C query.",
        "",
        "## Real Fixed-Cell Metrics",
        "",
        f"- real periderm pass rows: `{real['real_n_pass_rows']}/{real['real_n_rows']}`",
        f"- real periderm pass fraction: `{real['real_pass_fraction']:.4f}`",
        f"- real periderm mean ratio: `{real['real_mean_ratio']:.4f}`",
        "",
        "## Placebo Summary",
        "",
        f"- placebo pass-fraction proxy: `{placebo_pass_fraction_proxy:.4f}`",
        f"- placebo ratio p95: `{placebo_ratio_p95:.4f}`",
        f"- real-minus-placebo pass margin: `{real_minus_placebo_pass:.4f}`",
        f"- real-minus-placebo ratio margin: `{real_minus_placebo_ratio:.4f}`",
        "",
        "## Decision Reasons",
        "",
        *(f"- {reason}" for reason in reasons),
        "",
        "## Output Files",
        "",
        f"- placebo rows: `{placebo_csv}`",
        f"- JSON: `{json_path}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)
    print(status)
    return 0 if status.endswith("pass_no_gpu") else 2


if __name__ == "__main__":
    raise SystemExit(main())
