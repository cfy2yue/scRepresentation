#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.decomposition import TruncatedSVD


def utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def read_cell_index(path: Path) -> dict[str, int]:
    out = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            out[row["cell"]] = int(row["expression_col_index"])
    return out


def stable_seed(seed: int, *parts: str) -> int:
    h = hashlib.sha256(("|".join([str(seed), *parts])).encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2**32)


def normalize_and_embed(counts_genes_by_cells: sp.spmatrix, n_hvg: int, n_pca: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    x = counts_genes_by_cells.T.tocsr().astype(np.float32)
    lib = np.asarray(x.sum(axis=1)).ravel()
    scale = 1e4 / np.maximum(lib, 1.0)
    x = x.multiply(scale[:, None]).tocsr()
    x.data = np.log1p(x.data)

    mean = np.asarray(x.mean(axis=0)).ravel()
    mean_sq = np.asarray(x.power(2).mean(axis=0)).ravel()
    var = np.maximum(mean_sq - mean**2, 0)
    nonzero = np.flatnonzero(np.asarray(x.getnnz(axis=0)).ravel() > 0)
    if nonzero.size == 0:
        raise ValueError("No nonzero genes in selected expression matrix")
    hvg_n = min(n_hvg, nonzero.size)
    hvg_idx = nonzero[np.argsort(var[nonzero])[-hvg_n:]]
    x_hvg = x[:, hvg_idx]
    pca_n = max(2, min(n_pca, hvg_n - 1, x_hvg.shape[0] - 1))
    emb = TruncatedSVD(n_components=pca_n, random_state=seed).fit_transform(x_hvg)
    meta = {
        "n_cells": int(x_hvg.shape[0]),
        "n_genes": int(x.shape[1]),
        "n_hvg": int(hvg_n),
        "n_pca": int(pca_n),
        "library_min": float(lib.min()) if lib.size else 0.0,
        "library_median": float(np.median(lib)) if lib.size else 0.0,
        "library_max": float(lib.max()) if lib.size else 0.0,
    }
    return np.asarray(emb, dtype=np.float32), meta


def embryo_balanced_positions(df: pd.DataFrame, n: int, seed: int, key: str) -> np.ndarray:
    if len(df) <= n:
        return df.index.to_numpy()
    rng = np.random.default_rng(stable_seed(seed, key))
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, embryo in zip(df.index, df["embryo"].astype(str)):
        groups[embryo].append(int(idx))
    for group in groups.values():
        rng.shuffle(group)
    embryos = sorted(groups, key=lambda e: (len(groups[e]), e))
    selected: list[int] = []
    cursor = 0
    while len(selected) < n and embryos:
        embryo = embryos[cursor % len(embryos)]
        if groups[embryo]:
            selected.append(groups[embryo].pop())
        embryos = [e for e in embryos if groups[e]]
        cursor += 1
    return np.array(selected[:n], dtype=np.int64)


def random_positions(values: np.ndarray, n: int, rng: np.random.Generator, replace: bool = False) -> np.ndarray:
    if len(values) <= n and not replace:
        return values.copy()
    return rng.choice(values, size=n, replace=replace)


def assignment_ot(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    aa = a[:n]
    bb = b[:n]
    dist = cdist(aa, bb, metric="euclidean")
    rows, cols = linear_sum_assignment(dist)
    return float(dist[rows, cols].mean())


def js_divergence(a: list[str], b: list[str]) -> float:
    ca = Counter(a)
    cb = Counter(b)
    keys = sorted(set(ca) | set(cb))
    pa = np.array([ca[k] for k in keys], dtype=np.float64)
    pb = np.array([cb[k] for k in keys], dtype=np.float64)
    pa = pa / max(pa.sum(), 1)
    pb = pb / max(pb.sum(), 1)
    m = 0.5 * (pa + pb)

    def kl(p: np.ndarray, q: np.ndarray) -> float:
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / np.maximum(q[mask], 1e-12))))

    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


def summarize_row(
    row_id: str,
    row_df: pd.DataFrame,
    emb: np.ndarray,
    seed: int,
    ot_cells: int,
    null_repeats: int,
) -> dict[str, Any]:
    p_df = row_df[row_df["selection_role"] == "perturb"].drop_duplicates("cell")
    c_df = row_df[row_df["selection_role"] == "control"].drop_duplicates("cell")
    n = min(ot_cells, len(p_df), len(c_df))
    if n < 20:
        return {
            "row_id": row_id,
            "status": "too_few_cells",
            "n_perturb": int(len(p_df)),
            "n_control": int(len(c_df)),
        }
    p_pos = embryo_balanced_positions(p_df, n, seed, f"{row_id}|obs|perturb")
    c_pos = embryo_balanced_positions(c_df, n, seed, f"{row_id}|obs|control")
    observed = assignment_ot(emb[p_pos], emb[c_pos])

    c_positions = c_df.index.to_numpy()
    combined = pd.concat([p_df, c_df], axis=0)
    combined_positions = combined.index.to_numpy()
    cc_null = []
    label_null = []
    for rep in range(null_repeats):
        rng = np.random.default_rng(stable_seed(seed, row_id, str(rep)))
        if len(c_positions) >= 2 * n:
            chosen = random_positions(c_positions, 2 * n, rng, replace=False)
            cc_null.append(assignment_ot(emb[chosen[:n]], emb[chosen[n:]]))
        else:
            cc_a = random_positions(c_positions, n, rng, replace=True)
            cc_b = random_positions(c_positions, n, rng, replace=True)
            cc_null.append(assignment_ot(emb[cc_a], emb[cc_b]))
        shuffled = random_positions(combined_positions, 2 * n, rng, replace=False)
        label_null.append(assignment_ot(emb[shuffled[:n]], emb[shuffled[n:]]))

    cc_arr = np.array(cc_null, dtype=np.float64)
    label_arr = np.array(label_null, dtype=np.float64)
    p_cc = float((np.sum(cc_arr >= observed) + 1) / (len(cc_arr) + 1))
    p_label = float((np.sum(label_arr >= observed) + 1) / (len(label_arr) + 1))
    row_gate = p_cc <= 0.05 and p_label <= 0.05
    subtype_jsd = js_divergence(
        p_df.get("cell_type_sub", pd.Series(dtype=str)).astype(str).tolist(),
        c_df.get("cell_type_sub", pd.Series(dtype=str)).astype(str).tolist(),
    )

    first = row_df.iloc[0].to_dict()
    return {
        "row_id": row_id,
        "status": "ok",
        "audit_role": first.get("audit_role", ""),
        "trajectory_anchor": str(first.get("trajectory_anchor", "")),
        "cell_type_broad": first.get("manifest_cell_type_broad", ""),
        "gene_target": first.get("manifest_gene_target", ""),
        "timepoint": first.get("manifest_timepoint", ""),
        "n_perturb": int(len(p_df)),
        "n_control": int(len(c_df)),
        "ot_n": int(n),
        "observed_assignment_ot": observed,
        "cc_null_median": float(np.median(cc_arr)),
        "cc_null_p95": float(np.quantile(cc_arr, 0.95)),
        "label_null_median": float(np.median(label_arr)),
        "label_null_p95": float(np.quantile(label_arr, 0.95)),
        "p_observed_le_cc_null": p_cc,
        "p_observed_le_label_null": p_label,
        "row_expression_ot_gate": bool(row_gate),
        "subtype_jsd": float(subtype_jsd),
    }


def temporal_control_checks(manifest: pd.DataFrame, emb: np.ndarray, seed: int, null_repeats: int) -> list[dict[str, Any]]:
    out = []
    controls = manifest[manifest["selection_role"] == "control"].drop_duplicates("cell")
    for lineage, lineage_df in controls.groupby("manifest_cell_type_broad"):
        times = sorted(lineage_df["manifest_timepoint"].astype(float).unique())
        if len(times) < 2:
            continue
        for t0, t1 in zip(times[:-1], times[1:]):
            a = lineage_df[lineage_df["manifest_timepoint"].astype(float) == t0]
            b = lineage_df[lineage_df["manifest_timepoint"].astype(float) == t1]
            n = min(128, len(a), len(b))
            if n < 20:
                continue
            a_pos = embryo_balanced_positions(a, n, seed, f"{lineage}|{t0}|temporal_a")
            b_pos = embryo_balanced_positions(b, n, seed, f"{lineage}|{t1}|temporal_b")
            observed = assignment_ot(emb[a_pos], emb[b_pos])
            same_null = []
            for rep in range(null_repeats):
                rng = np.random.default_rng(stable_seed(seed, lineage, str(t0), str(t1), str(rep)))
                source = a.index.to_numpy() if len(a) >= 2 * n else lineage_df.index.to_numpy()
                replace = len(source) < 2 * n
                chosen = random_positions(source, 2 * n, rng, replace=replace)
                same_null.append(assignment_ot(emb[chosen[:n]], emb[chosen[n:]]))
            null = np.array(same_null)
            out.append(
                {
                    "cell_type_broad": lineage,
                    "timepoint_a": float(t0),
                    "timepoint_b": float(t1),
                    "n_per_time": int(n),
                    "observed_control_temporal_ot": float(observed),
                    "same_time_null_median": float(np.median(null)),
                    "same_time_null_p95": float(np.quantile(null, 0.95)),
                    "p_temporal_le_same_null": float((np.sum(null >= observed) + 1) / (len(null) + 1)),
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts-npz", type=Path, required=True)
    parser.add_argument("--cell-index", type=Path, required=True)
    parser.add_argument("--matched-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--n-pca", type=int, default=32)
    parser.add_argument("--ot-cells", type=int, default=128)
    parser.add_argument("--null-repeats", type=int, default=100)
    parser.add_argument("--primary-pass-min", type=int, default=7)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = sp.load_npz(args.counts_npz)
    cell_index = read_cell_index(args.cell_index)
    manifest = pd.read_csv(args.matched_manifest)
    manifest["expression_col_index"] = manifest["cell"].map(cell_index)
    manifest = manifest.dropna(subset=["expression_col_index"]).copy()
    manifest["expression_col_index"] = manifest["expression_col_index"].astype(int)
    # Reindex embeddings by expression column. Counts columns already match cell_index order.
    emb, embed_meta = normalize_and_embed(counts, args.n_hvg, args.n_pca, args.seed)
    manifest = manifest.set_index("expression_col_index", drop=False)

    row_results = [
        summarize_row(row_id, group, emb, args.seed, args.ot_cells, args.null_repeats)
        for row_id, group in manifest.groupby("row_id", sort=True)
    ]
    temporal_results = temporal_control_checks(manifest, emb, args.seed, args.null_repeats)

    row_csv = args.out_dir / "zscape_expression_ot_row_results.csv"
    temporal_csv = args.out_dir / "zscape_expression_temporal_control_results.csv"
    row_df = pd.DataFrame(row_results)
    row_df.to_csv(row_csv, index=False)
    pd.DataFrame(temporal_results).to_csv(temporal_csv, index=False)

    primary = row_df[row_df.get("audit_role", "") == "primary_mechanism_test"]
    primary_pass = int(primary.get("row_expression_ot_gate", pd.Series(dtype=bool)).fillna(False).sum())
    lineage_pass = {}
    if not primary.empty:
        for lineage, group in primary.groupby("cell_type_broad"):
            lineage_pass[lineage] = int(group["row_expression_ot_gate"].fillna(False).sum())
    status = (
        "zscape_expression_ot_continuity_gate_pass_no_gpu"
        if primary_pass >= args.primary_pass_min and all(v >= 3 for v in lineage_pass.values())
        else "zscape_expression_ot_continuity_gate_fail_or_partial_no_gpu"
    )

    json_path = args.out_dir / "zscape_expression_ot_continuity_gate_20260628.json"
    md_path = args.out_dir / "LATENTFM_ZSCAPE_EXPRESSION_OT_CONTINUITY_GATE_20260628.md"
    payload = {
        "timestamp_utc": utc_now(),
        "status": status,
        "gpu_authorized": False,
        "counts_npz": str(args.counts_npz),
        "matched_manifest": str(args.matched_manifest),
        "embedding": embed_meta,
        "filters": {
            "ot_cells": args.ot_cells,
            "null_repeats": args.null_repeats,
            "primary_pass_min": args.primary_pass_min,
        },
        "summary": {
            "rows": int(len(row_df)),
            "primary_rows": int(len(primary)),
            "primary_rows_passing_ot_gate": primary_pass,
            "primary_lineage_pass_counts": lineage_pass,
            "temporal_checks": int(len(temporal_results)),
        },
        "row_results_csv": str(row_csv),
        "temporal_results_csv": str(temporal_csv),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM ZSCAPE Expression OT / Continuity Gate",
        "",
        f"Timestamp: `{payload['timestamp_utc']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only expression-space validation over the manifest-selected ZSCAPE cells.",
        "- Does not train, infer, run scFM embeddings, read canonical multi, or read Track C query.",
        "- Metadata UMAP is not used as expression evidence in this gate.",
        "",
        "## Gate Summary",
        "",
        f"- rows evaluated: `{len(row_df)}`",
        f"- primary rows passing OT/null gate: `{primary_pass}/{len(primary)}`",
        f"- primary lineage pass counts: `{lineage_pass}`",
        f"- temporal control checks: `{len(temporal_results)}`",
        f"- HVGs/PCA: `{embed_meta['n_hvg']}/{embed_meta['n_pca']}`",
        "",
        "## Primary Row Results",
        "",
        "| row_id | target | time | obs OT | cc p95 | label p95 | p_cc | p_label | subtype JSD | gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in row_results:
        if row.get("audit_role") != "primary_mechanism_test":
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("row_id", "")),
                    str(row.get("gene_target", "")),
                    str(row.get("timepoint", "")),
                    f"{float(row.get('observed_assignment_ot', float('nan'))):.4f}",
                    f"{float(row.get('cc_null_p95', float('nan'))):.4f}",
                    f"{float(row.get('label_null_p95', float('nan'))):.4f}",
                    f"{float(row.get('p_observed_le_cc_null', float('nan'))):.4f}",
                    f"{float(row.get('p_observed_le_label_null', float('nan'))):.4f}",
                    f"{float(row.get('subtype_jsd', float('nan'))):.4f}",
                    str(row.get("row_expression_ot_gate", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "This CPU gate supports proceeding to a bounded latent/trajectory modeling design review."
                if status.endswith("pass_no_gpu")
                else "Do not launch GPU from this branch; inspect failure modes, subtype composition, and null controls first."
            ),
            "Even a pass here does not by itself authorize model promotion.",
            "",
            "## Output Files",
            "",
            f"- row results: `{row_csv}`",
            f"- temporal controls: `{temporal_csv}`",
            f"- JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)
    print(json_path)
    print(status)
    return 0 if status.endswith("pass_no_gpu") else 2


if __name__ == "__main__":
    raise SystemExit(main())
