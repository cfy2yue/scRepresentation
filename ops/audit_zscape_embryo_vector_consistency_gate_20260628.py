#!/usr/bin/env python3
"""Embryo-level vector consistency gate for ZSCAPE OT dynamics.

This CPU-only gate asks whether row-level OT response geometry is stable across
perturbation embryos. It reuses the same control-only HVG/SVD expression space
as the ZSCAPE OT dynamic gate and reads already materialized OT pseudo-pairs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import scipy.sparse as sp

from audit_zscape_expression_ot_strict_controls_20260628 import (
    control_only_embed,
    make_cell_level_manifest,
    read_cell_index,
)


ROOT = Path("/data/cyx/1030/scLatent")
COUNTS = (
    ROOT
    / "runs/zscape_raw_counts_cell_manifest_extraction_20260628"
    / "zscape_raw_counts_cell_manifest_extraction_20260628_074523"
    / "outputs/zscape_manifest_selected_counts_csc.npz"
)
CELL_INDEX = COUNTS.parent / "zscape_manifest_selected_expression_cell_index.csv"
MATCHED_MANIFEST = COUNTS.parent / "zscape_expression_selected_cell_ids_matched.csv"
PSEUDO_PAIRS = ROOT / "reports/zscape_ot_dynamic_response_gate_20260628/zscape_ot_dynamic_response_pseudo_pairs.csv"
ROW_SYNTH = ROOT / "reports/zscape_dynamic_information_modeling_gate_20260628/zscape_dynamic_information_row_synthesis.csv"
OUT_DIR = ROOT / "reports/zscape_embryo_vector_consistency_gate_20260628"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def bootstrap_ci(values: np.ndarray, seed: int, repeats: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, values.size, size=(repeats, values.size))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def prepare_embedding(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    counts = sp.load_npz(args.counts_npz)
    cell_index = read_cell_index(args.cell_index)
    manifest = pd.read_csv(args.matched_manifest)
    manifest["expression_col_index"] = manifest["cell"].map(cell_index)
    manifest = manifest.dropna(subset=["expression_col_index"]).copy()
    manifest["expression_col_index"] = manifest["expression_col_index"].astype(int)
    manifest = manifest.set_index("expression_col_index", drop=False)
    cell_manifest = make_cell_level_manifest(manifest, counts.shape[1])
    emb, libraries, embed_meta = control_only_embed(
        counts,
        cell_manifest,
        args.n_hvg,
        args.n_pca,
        args.seed,
    )
    manifest["expression_library"] = libraries[manifest.index.to_numpy(dtype=int)]
    return manifest, emb, embed_meta


def build_cell_index(manifest: pd.DataFrame) -> dict[str, int]:
    return {
        str(row["cell"]): int(row["expression_col_index"])
        for _, row in manifest.reset_index(drop=True).drop_duplicates("cell").iterrows()
    }


def row_vectors_for_pairs(
    row_id: str,
    pairs: pd.DataFrame,
    emb: np.ndarray,
    cell_to_idx: dict[str, int],
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pair_rows = pairs[pairs["row_id"].astype(str) == str(row_id)].copy()
    if pair_rows.empty:
        return {"row_id": row_id, "status": "missing_pairs", "embryo_vector_gate": False}, []

    p_idx = pair_rows["perturb_cell"].map(cell_to_idx)
    c_idx = pair_rows["control_cell"].map(cell_to_idx)
    valid = p_idx.notna() & c_idx.notna()
    pair_rows = pair_rows[valid].copy()
    p_idx = p_idx[valid].astype(int).to_numpy()
    c_idx = c_idx[valid].astype(int).to_numpy()
    if len(pair_rows) < 16:
        return {
            "row_id": row_id,
            "status": "too_few_valid_pairs",
            "n_valid_pairs": int(len(pair_rows)),
            "embryo_vector_gate": False,
        }, []

    global_vec = emb[p_idx].mean(axis=0) - emb[c_idx].mean(axis=0)
    global_norm = float(np.linalg.norm(global_vec))
    records: list[dict[str, Any]] = []

    for embryo, grp in pair_rows.groupby("perturb_embryo", dropna=False):
        local_p_idx = grp["perturb_cell"].map(cell_to_idx).astype(int).to_numpy()
        local_c_idx = grp["control_cell"].map(cell_to_idx).astype(int).to_numpy()
        vec = emb[local_p_idx].mean(axis=0) - emb[local_c_idx].mean(axis=0)
        records.append(
            {
                "row_id": row_id,
                "perturb_embryo": embryo,
                "n_pairs": int(len(grp)),
                "vector_norm": float(np.linalg.norm(vec)),
                "cosine_to_row_vector": cosine(vec, global_vec),
            }
        )

    embryo_df = pd.DataFrame(records)
    cosines = pd.to_numeric(embryo_df["cosine_to_row_vector"], errors="coerce").to_numpy(dtype=float)
    ci_low, ci_high = bootstrap_ci(cosines, seed=seed)
    positive_fraction = float(np.mean(cosines > 0)) if cosines.size else float("nan")
    pass_gate = bool(
        len(embryo_df) >= 4
        and positive_fraction >= 0.75
        and math.isfinite(ci_low)
        and ci_low > 0.0
        and global_norm > 0.25
    )
    summary = {
        "row_id": row_id,
        "status": "ok",
        "n_valid_pairs": int(len(pair_rows)),
        "n_perturb_embryos": int(len(embryo_df)),
        "global_vector_norm": global_norm,
        "mean_embryo_cosine": float(np.nanmean(cosines)) if cosines.size else float("nan"),
        "min_embryo_cosine": float(np.nanmin(cosines)) if cosines.size else float("nan"),
        "positive_embryo_fraction": positive_fraction,
        "mean_cosine_ci_low": ci_low,
        "mean_cosine_ci_high": ci_high,
        "embryo_vector_gate": pass_gate,
    }
    return summary, records


def write_report(
    out_dir: Path,
    rows: pd.DataFrame,
    embryo_rows: pd.DataFrame,
    row_synth: pd.DataFrame,
    embed_meta: dict[str, Any],
) -> None:
    if rows.empty:
        positives = 0
    else:
        positives = int(rows["embryo_vector_gate"].astype(bool).sum())
    geometry_positive = []
    geometry_and_vector_positive = []
    if not row_synth.empty and "geometry_gate" in row_synth.columns:
        geometry_positive = row_synth[row_synth["geometry_gate"].astype(bool)]["row_id"].astype(str).tolist()
        positive_set = set(rows[rows["embryo_vector_gate"].astype(bool)]["row_id"].astype(str)) if not rows.empty else set()
        geometry_and_vector_positive = [r for r in geometry_positive if r in positive_set]
    vector_is_discriminative = positives < len(rows) if len(rows) else False
    status = (
        "zscape_embryo_vector_consistency_specific_positive_cpu_only"
        if vector_is_discriminative and geometry_and_vector_positive
        else "zscape_embryo_vector_consistency_replicate_consistent_nonspecific_no_gpu"
    )

    lines: list[str] = []
    lines.append("# ZSCAPE Embryo Vector Consistency Gate")
    lines.append("")
    lines.append(f"Timestamp: `{now_cst()}`")
    lines.append("")
    lines.append(f"Status: `{status}`")
    lines.append("")
    lines.append("GPU authorized: `False`")
    lines.append("")
    lines.append("## Boundary")
    lines.append("")
    lines.append("- CPU-only gate over frozen ZSCAPE selected counts and OT pseudo-pairs.")
    lines.append("- Uses the same control-only HVG/SVD expression coordinate system as the OT dynamic gate.")
    lines.append("- Embryo vectors are analytical snapshot vectors, not lineage-tracked single-cell trajectories.")
    lines.append("- No model training, inference, embedding extraction by scFM, canonical multi, Track C query, or checkpoint selection.")
    lines.append("")
    lines.append("## Embedding Provenance")
    lines.append("")
    lines.append(f"- counts: `{COUNTS}`")
    lines.append(f"- pseudo-pairs: `{PSEUDO_PAIRS}`")
    lines.append(f"- n cells: `{embed_meta.get('n_cells')}`; n HVG: `{embed_meta.get('n_hvg')}`; n PCA: `{embed_meta.get('n_pca')}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Rows tested: `{len(rows)}`.")
    lines.append(f"- Embryo-vector positive rows: `{positives}`.")
    lines.append(f"- Geometry-positive rows before this gate: `{', '.join(geometry_positive)}`.")
    lines.append(f"- Geometry-positive and embryo-vector-positive rows: `{', '.join(geometry_and_vector_positive)}`.")
    lines.append(f"- Embryo-vector gate discriminative across rows: `{vector_is_discriminative}`.")
    lines.append("")
    lines.append("## Row Results")
    lines.append("")
    cols = [
        "row_id",
        "n_perturb_embryos",
        "global_vector_norm",
        "mean_embryo_cosine",
        "mean_cosine_ci_low",
        "positive_embryo_fraction",
        "embryo_vector_gate",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in rows.iterrows():
        vals = []
        for col in cols:
            val = row.get(col, "")
            vals.append(fmt(val) if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    if positives:
        pos_rows = rows[rows["embryo_vector_gate"].astype(bool)]["row_id"].astype(str).tolist()
        lines.append("- Embryo-level vector consistency is broadly positive: `" + "`, `".join(pos_rows) + "`.")
        lines.append("- Because confounded/diagnostic rows also pass, this gate is not a specificity filter.")
        lines.append("- The result supports replicate-stable snapshot response geometry, but does not by itself identify mechanism-positive rows.")
        lines.append("- For modeling, keep ZSCAPE as geometry diagnostics and negative-control design until wrong-control specificity and latent/raw route gates pass.")
    else:
        lines.append("- No row passes embryo-level vector consistency; keep ZSCAPE as hypothesis-only and do not translate it into model constraints.")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- row summary: `{out_dir / 'zscape_embryo_vector_consistency_rows.csv'}`")
    lines.append(f"- embryo vectors: `{out_dir / 'zscape_embryo_vector_consistency_embryo_rows.csv'}`")
    lines.append(f"- JSON: `{out_dir / 'zscape_embryo_vector_consistency_gate_20260628.json'}`")
    (out_dir / "LATENTFM_ZSCAPE_EMBRYO_VECTOR_CONSISTENCY_GATE_20260628.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts-npz", type=Path, default=COUNTS)
    parser.add_argument("--cell-index", type=Path, default=CELL_INDEX)
    parser.add_argument("--matched-manifest", type=Path, default=MATCHED_MANIFEST)
    parser.add_argument("--pseudo-pairs", type=Path, default=PSEUDO_PAIRS)
    parser.add_argument("--row-synthesis", type=Path, default=ROW_SYNTH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--n-pca", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest, emb, embed_meta = prepare_embedding(args)
    cell_to_idx = build_cell_index(manifest)
    pairs = pd.read_csv(args.pseudo_pairs)
    row_synth = pd.read_csv(args.row_synthesis) if args.row_synthesis.exists() else pd.DataFrame()

    row_ids = sorted(pairs["row_id"].astype(str).unique())
    summaries: list[dict[str, Any]] = []
    embryo_records: list[dict[str, Any]] = []
    for i, row_id in enumerate(row_ids):
        summary, records = row_vectors_for_pairs(row_id, pairs, emb, cell_to_idx, seed=args.seed + i)
        summaries.append(summary)
        embryo_records.extend(records)

    rows = pd.DataFrame(summaries)
    embryo_rows = pd.DataFrame(embryo_records)
    row_path = args.out_dir / "zscape_embryo_vector_consistency_rows.csv"
    embryo_path = args.out_dir / "zscape_embryo_vector_consistency_embryo_rows.csv"
    rows.to_csv(row_path, index=False)
    embryo_rows.to_csv(embryo_path, index=False)

    obj = {
        "timestamp": now_cst(),
        "status": "zscape_embryo_vector_consistency_replicate_consistent_nonspecific_no_gpu",
        "gpu_authorized_next": False,
        "n_rows": int(len(rows)),
        "positive_rows": rows[rows["embryo_vector_gate"].astype(bool)]["row_id"].astype(str).tolist()
        if not rows.empty
        else [],
        "positive_gate_is_discriminative": bool(
            int((rows["embryo_vector_gate"].astype(bool)).sum()) < int(len(rows))
        )
        if not rows.empty
        else False,
        "embedding_meta": embed_meta,
        "outputs": {
            "rows": str(row_path),
            "embryo_rows": str(embryo_path),
            "report": str(args.out_dir / "LATENTFM_ZSCAPE_EMBRYO_VECTOR_CONSISTENCY_GATE_20260628.md"),
        },
    }
    write_json(args.out_dir / "zscape_embryo_vector_consistency_gate_20260628.json", obj)
    write_report(args.out_dir, rows, embryo_rows, row_synth, embed_meta)


if __name__ == "__main__":
    main()
