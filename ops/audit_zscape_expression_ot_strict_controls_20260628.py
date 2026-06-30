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


def stable_seed(seed: int, *parts: str) -> int:
    h = hashlib.sha256("|".join([str(seed), *parts]).encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2**32)


def read_cell_index(path: Path) -> dict[str, int]:
    with path.open(newline="") as handle:
        return {row["cell"]: int(row["expression_col_index"]) for row in csv.DictReader(handle)}


def make_cell_level_manifest(manifest: pd.DataFrame, n_cells: int) -> pd.DataFrame:
    manifest_for_cells = manifest.reset_index(drop=True)
    cell_manifest = (
        manifest_for_cells.sort_values(["expression_col_index", "selection_role", "row_id"])
        .drop_duplicates("expression_col_index")
        .set_index("expression_col_index", drop=False)
        .sort_index()
    )
    expected = np.arange(n_cells, dtype=np.int64)
    observed = cell_manifest.index.to_numpy(dtype=np.int64)
    if len(cell_manifest) != n_cells or not np.array_equal(observed, expected):
        missing = sorted(set(expected.tolist()) - set(observed.tolist()))[:10]
        extra = sorted(set(observed.tolist()) - set(expected.tolist()))[:10]
        raise ValueError(
            "Cell-level manifest does not exactly match expression matrix columns: "
            f"n_manifest={len(cell_manifest)} n_cells={n_cells} missing_head={missing} extra_head={extra}"
        )
    return cell_manifest


def normalize_counts(counts_genes_by_cells: sp.spmatrix) -> tuple[sp.csr_matrix, np.ndarray]:
    x = counts_genes_by_cells.T.tocsr().astype(np.float32)
    lib = np.asarray(x.sum(axis=1)).ravel()
    x = x.multiply((1e4 / np.maximum(lib, 1.0))[:, None]).tocsr()
    x.data = np.log1p(x.data)
    return x, lib


def control_only_embed(
    counts_genes_by_cells: sp.spmatrix,
    manifest: pd.DataFrame,
    n_hvg: int,
    n_pca: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x, lib = normalize_counts(counts_genes_by_cells)
    control_mask = (manifest["selection_role"].to_numpy() == "control")
    fit = x[control_mask]
    mean = np.asarray(fit.mean(axis=0)).ravel()
    mean_sq = np.asarray(fit.power(2).mean(axis=0)).ravel()
    var = np.maximum(mean_sq - mean**2, 0)
    nonzero = np.flatnonzero(np.asarray(fit.getnnz(axis=0)).ravel() > 0)
    if nonzero.size == 0:
        raise ValueError("No nonzero control genes for control-only HVG/SVD")
    hvg_n = min(n_hvg, nonzero.size)
    hvg_idx = nonzero[np.argsort(var[nonzero])[-hvg_n:]]
    x_hvg = x[:, hvg_idx]
    pca_n = max(2, min(n_pca, hvg_n - 1, fit.shape[0] - 1))
    svd = TruncatedSVD(n_components=pca_n, random_state=seed)
    svd.fit(fit[:, hvg_idx])
    emb = svd.transform(x_hvg).astype(np.float32)
    meta = {
        "n_cells": int(x.shape[0]),
        "n_control_fit_cells": int(fit.shape[0]),
        "n_genes": int(x.shape[1]),
        "n_hvg": int(hvg_n),
        "n_pca": int(pca_n),
        "hvg_svd_fit": "control_only",
        "library_min": float(lib.min()) if lib.size else 0.0,
        "library_median": float(np.median(lib)) if lib.size else 0.0,
        "library_max": float(lib.max()) if lib.size else 0.0,
    }
    return emb, lib, meta


def assignment_ot(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    dist = cdist(a[:n], b[:n], metric="euclidean")
    rows, cols = linear_sum_assignment(dist)
    return float(dist[rows, cols].mean())


def js_divergence(a: list[str], b: list[str]) -> float:
    ca = Counter(a)
    cb = Counter(b)
    keys = sorted(set(ca) | set(cb))
    pa = np.array([ca[k] for k in keys], dtype=np.float64)
    pb = np.array([cb[k] for k in keys], dtype=np.float64)
    pa = pa / max(pa.sum(), 1.0)
    pb = pb / max(pb.sum(), 1.0)
    mix = 0.5 * (pa + pb)

    def kl(p: np.ndarray, q: np.ndarray) -> float:
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / np.maximum(q[mask], 1e-12))))

    return 0.5 * kl(pa, mix) + 0.5 * kl(pb, mix)


def smd(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    denom = np.sqrt(0.5 * (np.var(a) + np.var(b)))
    return float((np.mean(a) - np.mean(b)) / max(denom, 1e-8))


def embryo_balanced_positions(df: pd.DataFrame, n: int, seed: int, key: str) -> np.ndarray:
    if len(df) <= n:
        return df.index.to_numpy(dtype=np.int64)
    rng = np.random.default_rng(stable_seed(seed, key))
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, embryo in zip(df.index, df["embryo"].astype(str)):
        groups[embryo].append(int(idx))
    for values in groups.values():
        rng.shuffle(values)
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


def greedy_match(
    reference: pd.DataFrame,
    candidates: pd.DataFrame,
    n: int,
    seed: int,
    key: str,
    *,
    prefer_subtype: bool = True,
) -> np.ndarray:
    if len(reference) == 0 or len(candidates) == 0:
        return np.array([], dtype=np.int64)
    rng = np.random.default_rng(stable_seed(seed, key))
    ref = reference.copy()
    ref = ref.iloc[rng.permutation(len(ref))].head(n)
    cand_by_subtype: dict[str, list[int]] = defaultdict(list)
    all_candidates = list(candidates.index.astype(int))
    for idx, subtype in zip(candidates.index.astype(int), candidates["cell_type_sub"].astype(str)):
        cand_by_subtype[subtype].append(int(idx))
    for subtype in cand_by_subtype:
        cand_by_subtype[subtype].sort(key=lambda idx: float(candidates.at[idx, "log_library"]))
    used: set[int] = set()
    chosen: list[int] = []
    for _, row in ref.iterrows():
        subtype = str(row.get("cell_type_sub", ""))
        pool = [idx for idx in cand_by_subtype.get(subtype, []) if idx not in used] if prefer_subtype else []
        if not pool:
            pool = [idx for idx in all_candidates if idx not in used]
        if not pool:
            break
        target = float(row["log_library"])
        best = min(
            pool,
            key=lambda idx: (
                abs(float(candidates.at[idx, "log_library"]) - target),
                str(candidates.at[idx, "embryo"]),
                idx,
            ),
        )
        used.add(best)
        chosen.append(best)
    return np.array(chosen[:n], dtype=np.int64)


def quantile_or_nan(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.array(values, dtype=np.float64), q))


def row_qc_summary(p_df: pd.DataFrame, c_df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in ["expression_library", "n_umi", "num_genes_expressed"]:
        a = pd.to_numeric(p_df[col], errors="coerce").dropna().to_numpy(dtype=float)
        b = pd.to_numeric(c_df[col], errors="coerce").dropna().to_numpy(dtype=float)
        out[f"{col}_perturb_median"] = float(np.median(a)) if len(a) else float("nan")
        out[f"{col}_control_median"] = float(np.median(b)) if len(b) else float("nan")
        out[f"{col}_smd"] = smd(np.log1p(a), np.log1p(b))
    return out


def summarize_primary_row(
    row_id: str,
    row_df: pd.DataFrame,
    manifest_all: pd.DataFrame,
    emb: np.ndarray,
    seed: int,
    ot_cells: int,
    null_repeats: int,
    min_effect_ratio: float,
    max_subtype_jsd: float,
    max_library_abs_smd: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    p_all = row_df[row_df["selection_role"] == "perturb"].drop_duplicates("cell")
    c_all = row_df[row_df["selection_role"] == "control"].drop_duplicates("cell")
    n = min(ot_cells, len(p_all), len(c_all))
    if n < 40:
        return ({"row_id": row_id, "status": "too_few_cells", "n_perturb": len(p_all), "n_control": len(c_all)}, [])
    p_pos = embryo_balanced_positions(p_all, n, seed, f"{row_id}|strict|perturb")
    p_sample = p_all.loc[p_pos]
    c_pos = greedy_match(p_sample, c_all, n, seed, f"{row_id}|strict|obs_control")
    c_sample = c_all.loc[c_pos]
    n = min(len(p_sample), len(c_sample))
    p_sample = p_sample.head(n)
    c_sample = c_sample.head(n)
    p_pos = p_sample.index.to_numpy(dtype=np.int64)
    c_pos = c_sample.index.to_numpy(dtype=np.int64)
    observed = assignment_ot(emb[p_pos], emb[c_pos])

    cc_null: list[float] = []
    label_null: list[float] = []
    control_positions = set(c_all.index.astype(int))
    combined_all = pd.concat([p_all, c_all], axis=0).drop_duplicates("cell")
    for rep in range(null_repeats):
        c1_pos = greedy_match(p_sample, c_all, n, seed, f"{row_id}|cc1|{rep}")
        remaining = c_all.loc[[idx for idx in c_all.index.astype(int) if idx not in set(c1_pos.astype(int))]]
        c1 = c_all.loc[c1_pos]
        c2_pos = greedy_match(p_sample, remaining, n, seed, f"{row_id}|cc2|{rep}")
        if len(c1_pos) >= 40 and len(c2_pos) >= 40:
            cc_null.append(assignment_ot(emb[c1_pos[:n]], emb[c2_pos[:n]]))

        g1_pos = greedy_match(p_sample, combined_all, n, seed, f"{row_id}|label1|{rep}")
        remaining_combined = combined_all.loc[
            [idx for idx in combined_all.index.astype(int) if idx not in set(g1_pos.astype(int))]
        ]
        g2_pos = greedy_match(p_sample, remaining_combined, n, seed, f"{row_id}|label2|{rep}")
        if len(g1_pos) >= 40 and len(g2_pos) >= 40:
            label_null.append(assignment_ot(emb[g1_pos[:n]], emb[g2_pos[:n]]))

    cc_p95 = quantile_or_nan(cc_null, 0.95)
    label_p95 = quantile_or_nan(label_null, 0.95)
    max_p95 = max(cc_p95, label_p95)
    p_cc = float((np.sum(np.array(cc_null) >= observed) + 1) / (len(cc_null) + 1)) if cc_null else float("nan")
    p_label = (
        float((np.sum(np.array(label_null) >= observed) + 1) / (len(label_null) + 1)) if label_null else float("nan")
    )
    effect_ratio = float(observed / max(max_p95, 1e-8))
    subtype_jsd = js_divergence(
        p_sample["cell_type_sub"].astype(str).tolist(),
        c_sample["cell_type_sub"].astype(str).tolist(),
    )
    qc = row_qc_summary(p_sample, c_sample)
    first = row_df.iloc[0].to_dict()
    strict_gate = (
        p_cc <= 0.02
        and p_label <= 0.02
        and effect_ratio >= min_effect_ratio
        and subtype_jsd <= max_subtype_jsd
        and abs(qc["expression_library_smd"]) <= max_library_abs_smd
    )

    diagnostics: list[dict[str, Any]] = []
    p_lineage = str(first.get("manifest_cell_type_broad", ""))
    p_time = float(first.get("manifest_timepoint", 0.0))
    wrong_time = manifest_all[
        (manifest_all["selection_role"] == "control")
        & (manifest_all["manifest_cell_type_broad"].astype(str) == p_lineage)
        & (manifest_all["manifest_timepoint"].astype(float) != p_time)
    ].drop_duplicates("cell")
    wrong_lineage = manifest_all[
        (manifest_all["selection_role"] == "control")
        & (manifest_all["manifest_cell_type_broad"].astype(str) != p_lineage)
        & (manifest_all["manifest_timepoint"].astype(float) == p_time)
    ].drop_duplicates("cell")
    for diag_name, pool in [("wrong_time_control", wrong_time), ("wrong_lineage_control", wrong_lineage)]:
        d_pos = greedy_match(p_sample, pool, n, seed, f"{row_id}|{diag_name}")
        diagnostics.append(
            {
                "row_id": row_id,
                "diagnostic": diag_name,
                "n": int(min(n, len(d_pos))),
                "ot": assignment_ot(emb[p_pos[: len(d_pos)]], emb[d_pos]) if len(d_pos) >= 40 else float("nan"),
            }
        )

    result = {
        "row_id": row_id,
        "status": "ok",
        "audit_role": first.get("audit_role", ""),
        "cell_type_broad": p_lineage,
        "gene_target": first.get("manifest_gene_target", ""),
        "timepoint": first.get("manifest_timepoint", ""),
        "n_perturb": int(len(p_all)),
        "n_control": int(len(c_all)),
        "ot_n": int(n),
        "observed_strict_ot": observed,
        "cc_null_repeats": int(len(cc_null)),
        "cc_null_p95": cc_p95,
        "label_null_repeats": int(len(label_null)),
        "label_null_p95": label_p95,
        "p_observed_le_matched_cc_null": p_cc,
        "p_observed_le_matched_label_null": p_label,
        "effect_ratio_vs_max_null_p95": effect_ratio,
        "matched_subtype_jsd": subtype_jsd,
        "strict_row_gate": bool(strict_gate),
        **qc,
    }
    return result, diagnostics


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
    parser.add_argument("--null-repeats", type=int, default=500)
    parser.add_argument("--min-effect-ratio", type=float, default=1.05)
    parser.add_argument("--max-subtype-jsd", type=float, default=0.10)
    parser.add_argument("--max-library-abs-smd", type=float, default=0.35)
    parser.add_argument("--primary-pass-min", type=int, default=7)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = sp.load_npz(args.counts_npz)
    cell_index = read_cell_index(args.cell_index)
    manifest = pd.read_csv(args.matched_manifest)
    manifest["expression_col_index"] = manifest["cell"].map(cell_index)
    manifest = manifest.dropna(subset=["expression_col_index"]).copy()
    manifest["expression_col_index"] = manifest["expression_col_index"].astype(int)
    manifest = manifest.set_index("expression_col_index", drop=False)
    cell_manifest = make_cell_level_manifest(manifest, counts.shape[1])
    emb, libraries, embed_meta = control_only_embed(counts, cell_manifest, args.n_hvg, args.n_pca, args.seed)
    manifest["expression_library"] = libraries[manifest.index.to_numpy(dtype=int)]
    manifest["log_library"] = np.log1p(pd.to_numeric(manifest["expression_library"], errors="coerce").fillna(0))
    for col in ["n_umi", "num_genes_expressed"]:
        manifest[col] = pd.to_numeric(manifest[col], errors="coerce")

    primary_manifest = manifest[manifest["audit_role"] == "primary_mechanism_test"]
    row_results: list[dict[str, Any]] = []
    diag_results: list[dict[str, Any]] = []
    for row_id, group in primary_manifest.groupby("row_id", sort=True):
        result, diagnostics = summarize_primary_row(
            row_id,
            group,
            manifest,
            emb,
            args.seed,
            args.ot_cells,
            args.null_repeats,
            args.min_effect_ratio,
            args.max_subtype_jsd,
            args.max_library_abs_smd,
        )
        row_results.append(result)
        diag_results.extend(diagnostics)

    row_df = pd.DataFrame(row_results)
    diag_df = pd.DataFrame(diag_results)
    row_csv = args.out_dir / "zscape_expression_ot_strict_primary_rows.csv"
    diag_csv = args.out_dir / "zscape_expression_ot_strict_diagnostics.csv"
    row_df.to_csv(row_csv, index=False)
    diag_df.to_csv(diag_csv, index=False)

    row_pass = int(row_df.get("strict_row_gate", pd.Series(dtype=bool)).fillna(False).sum())
    lineage_pass: dict[str, int] = {}
    for lineage, group in row_df.groupby("cell_type_broad"):
        lineage_pass[str(lineage)] = int(group.get("strict_row_gate", pd.Series(dtype=bool)).fillna(False).sum())
    status = (
        "zscape_expression_ot_strict_controls_gate_pass_no_gpu"
        if row_pass >= args.primary_pass_min and all(v >= 3 for v in lineage_pass.values())
        else "zscape_expression_ot_strict_controls_gate_fail_or_partial_no_gpu"
    )

    json_path = args.out_dir / "zscape_expression_ot_strict_controls_gate_20260628.json"
    md_path = args.out_dir / "LATENTFM_ZSCAPE_EXPRESSION_OT_STRICT_CONTROLS_GATE_20260628.md"
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
            "min_effect_ratio": args.min_effect_ratio,
            "max_subtype_jsd": args.max_subtype_jsd,
            "max_library_abs_smd": args.max_library_abs_smd,
            "primary_pass_min": args.primary_pass_min,
        },
        "summary": {
            "primary_rows": int(len(row_df)),
            "strict_primary_rows_passing": row_pass,
            "strict_primary_lineage_pass_counts": lineage_pass,
        },
        "row_results_csv": str(row_csv),
        "diagnostics_csv": str(diag_csv),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM ZSCAPE Expression OT Strict Controls Gate",
        "",
        f"Timestamp: `{payload['timestamp_utc']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only strict-control follow-up to the exploratory expression OT gate.",
        "- Uses control-only HVG/SVD; no perturbation cells are used to fit the feature space.",
        "- Primary rows only: mature fast muscle and periderm.",
        "- No training, scFM embedding, canonical multi, or Track C query.",
        "",
        "## Gate Summary",
        "",
        f"- primary rows: `{len(row_df)}`",
        f"- strict primary rows passing: `{row_pass}/{len(row_df)}`",
        f"- lineage pass counts: `{lineage_pass}`",
        f"- null repeats: `{args.null_repeats}`",
        f"- min effect ratio vs max null p95: `{args.min_effect_ratio}`",
        f"- max matched subtype JSD: `{args.max_subtype_jsd}`",
        f"- max library abs SMD: `{args.max_library_abs_smd}`",
        "",
        "## Primary Row Results",
        "",
        "| row_id | target | time | obs OT | cc p95 | label p95 | ratio | p_cc | p_label | subtype JSD | lib SMD | gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in row_results:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("row_id", "")),
                    str(row.get("gene_target", "")),
                    str(row.get("timepoint", "")),
                    f"{float(row.get('observed_strict_ot', float('nan'))):.4f}",
                    f"{float(row.get('cc_null_p95', float('nan'))):.4f}",
                    f"{float(row.get('label_null_p95', float('nan'))):.4f}",
                    f"{float(row.get('effect_ratio_vs_max_null_p95', float('nan'))):.4f}",
                    f"{float(row.get('p_observed_le_matched_cc_null', float('nan'))):.4f}",
                    f"{float(row.get('p_observed_le_matched_label_null', float('nan'))):.4f}",
                    f"{float(row.get('matched_subtype_jsd', float('nan'))):.4f}",
                    f"{float(row.get('expression_library_smd', float('nan'))):.4f}",
                    str(row.get("strict_row_gate", "")),
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
                "This strict CPU gate supports a bounded latent/trajectory modeling design review."
                if status.endswith("pass_no_gpu")
                else "Do not launch GPU from this branch; strict confound controls weakened or failed the expression signal."
            ),
            "Even a pass here is not model promotion.",
            "",
            "## Output Files",
            "",
            f"- row results: `{row_csv}`",
            f"- diagnostics: `{diag_csv}`",
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
