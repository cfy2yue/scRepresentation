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


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def prepare_manifest(counts: sp.spmatrix, cell_index_path: Path, matched_manifest_path: Path, args: argparse.Namespace):
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


def matched_temporal_samples(a_pool: pd.DataFrame, b_pool: pd.DataFrame, n: int, seed: int, key: str):
    a_pos = embryo_balanced_positions(a_pool, n, seed, f"{key}|a")
    a_sample = a_pool.loc[a_pos]
    b_pos = greedy_match(a_sample, b_pool, n, seed, f"{key}|b", prefer_subtype=True)
    b_sample = b_pool.loc[b_pos]
    n2 = min(len(a_sample), len(b_sample))
    return a_sample.head(n2), b_sample.head(n2)


def same_time_null(pool: pd.DataFrame, emb: np.ndarray, n: int, seed: int, key: str, repeats: int) -> list[float]:
    values: list[float] = []
    if len(pool) < max(40, n * 2):
        return values
    for rep in range(repeats):
        a_pos = embryo_balanced_positions(pool, n, seed, f"{key}|same1|{rep}")
        a_sample = pool.loc[a_pos]
        remaining = pool.loc[[idx for idx in pool.index.astype(int) if idx not in set(a_pos.astype(int))]]
        b_pos = greedy_match(a_sample, remaining, n, seed, f"{key}|same2|{rep}", prefer_subtype=True)
        if len(a_pos) >= 40 and len(b_pos) >= 40:
            values.append(assignment_ot(emb[a_pos[:n]], emb[b_pos[:n]]))
    return values


def temporal_gate_for_lineage(
    lineage: str,
    controls: pd.DataFrame,
    manifest_all: pd.DataFrame,
    emb: np.ndarray,
    seed: int,
    ot_cells: int,
    null_repeats: int,
    min_ratio: float,
) -> tuple[list[dict[str, Any]], dict[tuple[float, float], np.ndarray]]:
    lineage_controls = controls[controls["manifest_cell_type_broad"].astype(str) == lineage]
    timepoints = sorted(t for t in lineage_controls["manifest_timepoint"].dropna().unique().tolist())
    rows: list[dict[str, Any]] = []
    vectors: dict[tuple[float, float], np.ndarray] = {}
    for t_a, t_b in zip(timepoints[:-1], timepoints[1:]):
        a_pool = lineage_controls[lineage_controls["manifest_timepoint"] == t_a].drop_duplicates("cell")
        b_pool = lineage_controls[lineage_controls["manifest_timepoint"] == t_b].drop_duplicates("cell")
        n = min(ot_cells, len(a_pool), len(b_pool))
        if n < 40:
            rows.append(
                {
                    "lineage": lineage,
                    "timepoint_a": t_a,
                    "timepoint_b": t_b,
                    "status": "too_few_control_cells",
                    "n_a": int(len(a_pool)),
                    "n_b": int(len(b_pool)),
                    "temporal_gate": False,
                }
            )
            continue
        a_sample, b_sample = matched_temporal_samples(a_pool, b_pool, n, seed, f"{lineage}|{t_a}|{t_b}|obs")
        a_idx = a_sample.index.to_numpy(dtype=np.int64)
        b_idx = b_sample.index.to_numpy(dtype=np.int64)
        observed = assignment_ot(emb[a_idx], emb[b_idx])
        vec = emb[b_idx].mean(axis=0) - emb[a_idx].mean(axis=0)
        vectors[(float(t_a), float(t_b))] = vec

        null = []
        null.extend(same_time_null(a_pool, emb, n, seed, f"{lineage}|{t_a}", null_repeats))
        null.extend(same_time_null(b_pool, emb, n, seed, f"{lineage}|{t_b}", null_repeats))
        null_p95 = quantile_or_nan(null, 0.95)
        p_temporal = float((np.sum(np.array(null) >= observed) + 1) / (len(null) + 1)) if null else float("nan")
        ratio = float(observed / max(null_p95, 1e-8))

        wrong_pool = manifest_all[
            (manifest_all["selection_role"] == "control")
            & (manifest_all["manifest_timepoint"] == t_b)
            & (manifest_all["manifest_cell_type_broad"].astype(str) != lineage)
        ].drop_duplicates("cell")
        wrong_pos = greedy_match(a_sample, wrong_pool, n, seed, f"{lineage}|{t_a}|{t_b}|wrong", prefer_subtype=True)
        wrong_ot = assignment_ot(emb[a_idx[: len(wrong_pos)]], emb[wrong_pos]) if len(wrong_pos) >= 40 else float("nan")
        wrong_ratio = float(wrong_ot / max(observed, 1e-8)) if np.isfinite(wrong_ot) else float("nan")
        gate = p_temporal <= 0.02 and ratio >= min_ratio and (not np.isfinite(wrong_ratio) or wrong_ratio >= 1.0)
        rows.append(
            {
                "lineage": lineage,
                "timepoint_a": float(t_a),
                "timepoint_b": float(t_b),
                "status": "ok",
                "n_per_time": int(n),
                "observed_temporal_ot": observed,
                "same_time_null_repeats": int(len(null)),
                "same_time_null_p95": null_p95,
                "p_temporal_le_same_time_null": p_temporal,
                "temporal_ratio_vs_null_p95": ratio,
                "wrong_lineage_same_time_ot": wrong_ot,
                "wrong_lineage_to_observed_ratio": wrong_ratio,
                "temporal_gate": bool(gate),
            }
        )
    return rows, vectors


def perturb_alignment_rows(
    primary_manifest: pd.DataFrame,
    controls: pd.DataFrame,
    emb: np.ndarray,
    temporal_vectors: dict[str, dict[tuple[float, float], np.ndarray]],
    seed: int,
    ot_cells: int,
    min_cosine_margin: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_id, group in primary_manifest.groupby("row_id", sort=True):
        first = group.iloc[0].to_dict()
        lineage = str(first.get("manifest_cell_type_broad", ""))
        timepoint = float(first.get("manifest_timepoint", 0.0))
        p_pool = group[group["selection_role"] == "perturb"].drop_duplicates("cell")
        c_pool = group[group["selection_role"] == "control"].drop_duplicates("cell")
        n = min(ot_cells, len(p_pool), len(c_pool))
        if n < 40:
            rows.append({"row_id": row_id, "status": "too_few_cells", "alignment_gate": False})
            continue
        p_pos = embryo_balanced_positions(p_pool, n, seed, f"{row_id}|perturb_align")
        p_sample = p_pool.loc[p_pos]
        c_pos = greedy_match(p_sample, c_pool, n, seed, f"{row_id}|control_align", prefer_subtype=True)
        if len(c_pos) < 40:
            rows.append({"row_id": row_id, "status": "too_few_matched_controls", "alignment_gate": False})
            continue
        n = min(len(p_pos), len(c_pos))
        displacement = emb[p_pos[:n]].mean(axis=0) - emb[c_pos[:n]].mean(axis=0)

        lineage_vectors = temporal_vectors.get(lineage, {})
        selected_pair = None
        if lineage_vectors:
            pairs = sorted(lineage_vectors)
            containing = [pair for pair in pairs if pair[0] <= timepoint <= pair[1]]
            selected_pair = containing[0] if containing else min(pairs, key=lambda pair: min(abs(pair[0] - timepoint), abs(pair[1] - timepoint)))
        true_cos = cosine(displacement, lineage_vectors[selected_pair]) if selected_pair else float("nan")

        wrong_cosines = []
        for wrong_lineage, pair_vectors in temporal_vectors.items():
            if wrong_lineage == lineage:
                continue
            if selected_pair in pair_vectors:
                wrong_cosines.append(cosine(displacement, pair_vectors[selected_pair]))
        max_wrong = float(np.nanmax(wrong_cosines)) if wrong_cosines else float("nan")
        margin = true_cos - max_wrong if np.isfinite(true_cos) and np.isfinite(max_wrong) else float("nan")
        gate = np.isfinite(true_cos) and true_cos > 0.0 and (not np.isfinite(max_wrong) or margin >= min_cosine_margin)
        rows.append(
            {
                "row_id": row_id,
                "status": "ok",
                "lineage": lineage,
                "gene_target": first.get("manifest_gene_target", ""),
                "timepoint": timepoint,
                "n": int(n),
                "temporal_pair": f"{selected_pair[0]}->{selected_pair[1]}" if selected_pair else "",
                "cosine_to_lineage_time_vector": true_cos,
                "max_cosine_to_wrong_lineage_time_vector": max_wrong,
                "cosine_margin_vs_wrong_lineage": margin,
                "alignment_gate": bool(gate),
            }
        )
    return rows


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
    parser.add_argument("--null-repeats", type=int, default=200)
    parser.add_argument("--min-temporal-ratio", type=float, default=1.05)
    parser.add_argument("--min-cosine-margin", type=float, default=0.02)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = sp.load_npz(args.counts_npz)
    manifest, emb, embed_meta = prepare_manifest(counts, args.cell_index, args.matched_manifest, args)
    controls = manifest[manifest["selection_role"] == "control"].drop_duplicates("cell")
    lineages = sorted(controls["manifest_cell_type_broad"].astype(str).unique().tolist())

    temporal_rows: list[dict[str, Any]] = []
    temporal_vectors: dict[str, dict[tuple[float, float], np.ndarray]] = {}
    for lineage in lineages:
        rows, vectors = temporal_gate_for_lineage(
            lineage,
            controls,
            manifest,
            emb,
            args.seed,
            args.ot_cells,
            args.null_repeats,
            args.min_temporal_ratio,
        )
        temporal_rows.extend(rows)
        temporal_vectors[lineage] = vectors

    primary = manifest[manifest["audit_role"] == "primary_mechanism_test"]
    alignment_rows = perturb_alignment_rows(
        primary,
        controls,
        emb,
        temporal_vectors,
        args.seed,
        args.ot_cells,
        args.min_cosine_margin,
    )

    temporal_df = pd.DataFrame(temporal_rows)
    align_df = pd.DataFrame(alignment_rows)
    temporal_csv = args.out_dir / "zscape_expression_trajectory_time_temporal_controls.csv"
    align_csv = args.out_dir / "zscape_expression_trajectory_time_perturb_alignment.csv"
    temporal_df.to_csv(temporal_csv, index=False)
    align_df.to_csv(align_csv, index=False)

    primary_temporal = temporal_df[
        temporal_df["lineage"].isin(["mature fast muscle", "periderm"])
        & temporal_df["timepoint_a"].eq(24.0)
        & temporal_df["timepoint_b"].eq(36.0)
    ]
    temporal_pass = int(primary_temporal.get("temporal_gate", pd.Series(dtype=bool)).fillna(False).sum())
    alignment_pass = int(align_df.get("alignment_gate", pd.Series(dtype=bool)).fillna(False).sum())
    status = (
        "zscape_expression_trajectory_time_gate_pass_no_gpu"
        if temporal_pass >= 2 and alignment_pass >= 4
        else "zscape_expression_trajectory_time_gate_partial_or_fail_no_gpu"
    )
    summary = {
        "timestamp_utc": utc_now(),
        "status": status,
        "gpu_authorized": False,
        "counts_npz": str(args.counts_npz),
        "matched_manifest": str(args.matched_manifest),
        "embed_meta": embed_meta,
        "primary_temporal_pass": temporal_pass,
        "primary_alignment_pass": alignment_pass,
        "temporal_csv": str(temporal_csv),
        "alignment_csv": str(align_csv),
    }
    json_path = args.out_dir / "zscape_expression_trajectory_time_gate_20260628.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    md_path = args.out_dir / "LATENTFM_ZSCAPE_EXPRESSION_TRAJECTORY_TIME_GATE_20260628.md"
    lines = [
        "# LatentFM ZSCAPE Expression Trajectory-Time Gate",
        "",
        f"Timestamp: `{utc_now()}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only dynamic biology gate over manifest-selected ZSCAPE raw counts.",
        "- Uses control-only HVG/SVD expression space.",
        "- Does not train, infer, run scFM embeddings, read canonical multi, or read Track C query.",
        "- Passing this gate authorizes only a bounded latent/trajectory design review, not model promotion.",
        "",
        "## Gate Summary",
        "",
        f"- primary temporal passes: `{temporal_pass}/2`",
        f"- primary perturb-alignment passes: `{alignment_pass}/{len(align_df)}`",
        f"- null repeats per same-time pool: `{args.null_repeats}`",
        "",
        "## Primary Temporal Controls",
        "",
        "| lineage | pair | obs OT | null p95 | ratio | p | wrong/obs | gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in primary_temporal.iterrows():
        lines.append(
            f"| {row.get('lineage', '')} | {row.get('timepoint_a', '')}->{row.get('timepoint_b', '')} | "
            f"{float(row.get('observed_temporal_ot', float('nan'))):.4f} | "
            f"{float(row.get('same_time_null_p95', float('nan'))):.4f} | "
            f"{float(row.get('temporal_ratio_vs_null_p95', float('nan'))):.4f} | "
            f"{float(row.get('p_temporal_le_same_time_null', float('nan'))):.4f} | "
            f"{float(row.get('wrong_lineage_to_observed_ratio', float('nan'))):.4f} | "
            f"{bool(row.get('temporal_gate', False))} |"
        )
    lines.extend(
        [
            "",
            "## Primary Perturbation Alignment",
            "",
            "| row_id | target | time | pair | true cosine | max wrong cosine | margin | gate |",
            "|---|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in align_df.iterrows():
        lines.append(
            f"| {row.get('row_id', '')} | {row.get('gene_target', '')} | {row.get('timepoint', '')} | "
            f"{row.get('temporal_pair', '')} | "
            f"{float(row.get('cosine_to_lineage_time_vector', float('nan'))):.4f} | "
            f"{float(row.get('max_cosine_to_wrong_lineage_time_vector', float('nan'))):.4f} | "
            f"{float(row.get('cosine_margin_vs_wrong_lineage', float('nan'))):.4f} | "
            f"{bool(row.get('alignment_gate', False))} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "This supports a dynamic trajectory design review after strict controls are integrated."
                if status.endswith("pass_no_gpu")
                else "Do not use this as a standalone GPU authorization; treat it as partial/negative dynamic evidence."
            ),
            "",
            "## Output Files",
            "",
            f"- temporal controls: `{temporal_csv}`",
            f"- perturb alignment: `{align_csv}`",
            f"- JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)
    print(status)
    return 0 if status.endswith("pass_no_gpu") else 2


if __name__ == "__main__":
    raise SystemExit(main())
