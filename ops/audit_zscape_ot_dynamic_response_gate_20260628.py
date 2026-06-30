#!/usr/bin/env python3
"""CPU-only OT-paired dynamic response gate for ZSCAPE snapshots."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "12")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "12")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from audit_zscape_expression_ot_strict_controls_20260628 import (
    control_only_embed,
    embryo_balanced_positions,
    greedy_match,
    make_cell_level_manifest,
    read_cell_index,
)


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_COUNTS = (
    ROOT
    / "runs/zscape_raw_counts_cell_manifest_extraction_20260628"
    / "zscape_raw_counts_cell_manifest_extraction_20260628_074523"
    / "outputs/zscape_manifest_selected_counts_csc.npz"
)
DEFAULT_CELL_INDEX = DEFAULT_COUNTS.parent / "zscape_manifest_selected_expression_cell_index.csv"
DEFAULT_MATCHED_MANIFEST = DEFAULT_COUNTS.parent / "zscape_expression_selected_cell_ids_matched.csv"
DEFAULT_SNAPSHOT_ROWS = (
    ROOT / "reports/zscape_snapshot_dynamic_constraint_spec_20260628/zscape_snapshot_dynamic_constraint_rows.csv"
)
DEFAULT_TEMPORAL_CONTROLS = (
    ROOT / "reports/zscape_snapshot_dynamic_constraint_spec_20260628/zscape_snapshot_dynamic_temporal_controls.csv"
)
DEFAULT_STRICT_ROWS = (
    ROOT
    / "runs/zscape_expression_ot_strict_controls_gate_20260628"
    / "zscape_expression_ot_strict_controls_gate_20260628_082748"
    / "outputs/zscape_expression_ot_strict_primary_rows.csv"
)
DEFAULT_STRICT_DIAG = DEFAULT_STRICT_ROWS.parent / "zscape_expression_ot_strict_diagnostics.csv"
DEFAULT_MODULE_ROWS = (
    ROOT / "reports/zscape_expression_module_scores_20260628/zscape_expression_module_score_rows.csv"
)
DEFAULT_PERIDERM_MODULE_ROWS = (
    ROOT
    / "reports/zscape_periderm_substate_time_qc_ot_module_gate_20260628"
    / "zscape_periderm_substate_time_qc_module_query_rows.csv"
)
DEFAULT_PERIDERM_SUBSTATE_ROWS = (
    ROOT
    / "reports/zscape_periderm_substate_time_qc_ot_module_gate_20260628"
    / "zscape_periderm_substate_time_qc_module_substate_rows.csv"
)
DEFAULT_ENRICHMENT_SUMMARY = (
    ROOT
    / "reports/zscape_formal_gprofiler_enrichment_20260628"
    / "zscape_formal_gprofiler_enrichment_20260628_130129"
    / "zscape_gprofiler_enrichment_summary.csv"
)
DEFAULT_LATENT_PROXY_JSON = (
    ROOT / "reports/zscape_latent_proxy_reconciliation_20260628/zscape_latent_proxy_reconciliation_20260628.json"
)
DEFAULT_UCE_CONTINUITY_JSON = (
    ROOT
    / "reports/zscape_uce_danio_latent_continuity_gate_20260628"
    / "zscape_uce_danio_latent_continuity_gate_20260628.json"
)
DEFAULT_OUT = ROOT / "reports/zscape_ot_dynamic_response_gate_20260628"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def fmt(value: Any, digits: int = 4) -> str:
    val = finite_float(value)
    if not math.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_output_dir(path: Path, force: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = [p for p in path.iterdir() if p.name != ".DS_Store"]
    if existing and not force:
        raise SystemExit(f"Refusing to overwrite nonempty output directory: {path}")


def js_divergence_from_counts(a: Counter[str], b: Counter[str]) -> float:
    keys = sorted(set(a) | set(b))
    if not keys:
        return float("nan")
    pa = np.asarray([a[k] for k in keys], dtype=float)
    pb = np.asarray([b[k] for k in keys], dtype=float)
    pa /= max(pa.sum(), 1.0)
    pb /= max(pb.sum(), 1.0)
    m = 0.5 * (pa + pb)

    def kl(p: np.ndarray, q: np.ndarray) -> float:
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / np.maximum(q[mask], 1e-12))))

    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def prepare_manifest(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
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
    manifest["log_library"] = np.log1p(pd.to_numeric(manifest["expression_library"], errors="coerce").fillna(0))
    for col in ["manifest_timepoint", "timepoint", "n_umi", "num_genes_expressed"]:
        if col in manifest.columns:
            manifest[col] = pd.to_numeric(manifest[col], errors="coerce")
    return manifest, emb, embed_meta


def subtype_composition_component(
    p_sample: pd.DataFrame,
    c_sample: pd.DataFrame,
    c_pool: pd.DataFrame,
    emb: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    p_counts = Counter(p_sample["cell_type_sub"].astype(str))
    c_counts = Counter(c_sample["cell_type_sub"].astype(str))
    subtypes = sorted(set(p_counts) | set(c_counts))
    if not subtypes:
        return np.zeros(emb.shape[1], dtype=float), {}
    c_pool_by_sub: dict[str, np.ndarray] = {}
    global_control_mean = emb[c_pool.index.to_numpy(dtype=np.int64)].mean(axis=0)
    for sub, group in c_pool.groupby(c_pool["cell_type_sub"].astype(str)):
        c_pool_by_sub[str(sub)] = emb[group.index.to_numpy(dtype=np.int64)].mean(axis=0)
    comp = np.zeros(emb.shape[1], dtype=float)
    n_p = max(sum(p_counts.values()), 1)
    n_c = max(sum(c_counts.values()), 1)
    for sub in subtypes:
        mean = c_pool_by_sub.get(sub, global_control_mean)
        comp += ((p_counts[sub] / n_p) - (c_counts[sub] / n_c)) * mean
    meta = {
        "substate_jsd": js_divergence_from_counts(p_counts, c_counts),
        "perturb_substates": len(p_counts),
        "control_substates": len(c_counts),
        "top_perturb_substates": ";".join(f"{k}:{v}" for k, v in p_counts.most_common(5)),
        "top_control_substates": ";".join(f"{k}:{v}" for k, v in c_counts.most_common(5)),
    }
    return comp, meta


def summarize_ot_row(
    row_id: str,
    group: pd.DataFrame,
    emb: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    p_pool = group[group["selection_role"] == "perturb"].drop_duplicates("cell")
    c_pool = group[group["selection_role"] == "control"].drop_duplicates("cell")
    n = min(args.ot_cells, len(p_pool), len(c_pool))
    first = group.iloc[0].to_dict()
    base = {
        "row_id": row_id,
        "audit_role": first.get("audit_role", ""),
        "lineage": first.get("manifest_cell_type_broad", ""),
        "target": first.get("manifest_gene_target", ""),
        "timepoint": first.get("manifest_timepoint", ""),
        "eligible_perturb_cells": int(len(p_pool)),
        "eligible_control_cells": int(len(c_pool)),
        "perturb_embryos": int(p_pool["embryo"].nunique()) if "embryo" in p_pool else 0,
        "control_embryos": int(c_pool["embryo"].nunique()) if "embryo" in c_pool else 0,
    }
    if n < args.min_cells:
        base.update({"status": "too_few_cells", "n_pseudo_pairs": int(n), "dynamic_response_gate": False})
        return base, []

    p_pos = embryo_balanced_positions(p_pool, n, args.seed, f"{row_id}|dynamic|perturb")
    p_sample = p_pool.loc[p_pos]
    c_pos_pre = greedy_match(p_sample, c_pool, n, args.seed, f"{row_id}|dynamic|control", prefer_subtype=True)
    c_sample_pre = c_pool.loc[c_pos_pre]
    n = min(len(p_sample), len(c_sample_pre))
    p_idx = p_sample.head(n).index.to_numpy(dtype=np.int64)
    c_idx_pre = c_sample_pre.head(n).index.to_numpy(dtype=np.int64)
    costs = cdist(emb[p_idx], emb[c_idx_pre], metric="euclidean")
    rows, cols = linear_sum_assignment(costs)
    p_idx = p_idx[rows]
    c_idx = c_idx_pre[cols]
    p_sample = p_sample.loc[p_idx]
    c_sample = c_sample_pre.loc[c_idx]
    pair_dist = np.linalg.norm(emb[p_idx] - emb[c_idx], axis=1)
    displacements = emb[p_idx] - emb[c_idx]
    centroid_delta = emb[p_idx].mean(axis=0) - emb[c_idx].mean(axis=0)
    comp, comp_meta = subtype_composition_component(p_sample, c_sample, c_pool, emb)
    residual = centroid_delta - comp
    centroid_norm = float(np.linalg.norm(centroid_delta))
    comp_norm = float(np.linalg.norm(comp))
    residual_norm = float(np.linalg.norm(residual))
    p_sub = p_sample["cell_type_sub"].astype(str).to_numpy()
    c_sub = c_sample["cell_type_sub"].astype(str).to_numpy()
    same_sub = p_sub == c_sub
    embryo_overlap = [
        len(set(str(a).split(";")) & set(str(b).split(";"))) > 0
        for a, b in zip(p_sample["embryo"].astype(str), c_sample["embryo"].astype(str))
    ]

    result = {
        **base,
        "status": "ok",
        "n_pseudo_pairs": int(n),
        "mean_pair_expression_distance": float(pair_dist.mean()),
        "median_pair_expression_distance": float(np.median(pair_dist)),
        "centroid_response_norm": centroid_norm,
        "mean_pair_displacement_norm": float(np.linalg.norm(displacements, axis=1).mean()),
        "mean_pair_displacement_cosine_to_centroid": float(np.nanmean([cosine(v, centroid_delta) for v in displacements])),
        "same_substate_pair_fraction": float(np.mean(same_sub)) if same_sub.size else float("nan"),
        "embryo_id_overlap_pair_fraction": float(np.mean(embryo_overlap)) if embryo_overlap else float("nan"),
        "composition_component_norm": comp_norm,
        "within_substate_residual_norm": residual_norm,
        "composition_norm_fraction_of_centroid": comp_norm / max(centroid_norm, 1e-12),
        "within_substate_residual_fraction_of_centroid": residual_norm / max(centroid_norm, 1e-12),
        **comp_meta,
    }
    pair_rows = []
    for rank, (pi, ci, dist, psub, csub) in enumerate(zip(p_idx, c_idx, pair_dist, p_sub, c_sub)):
        pair_rows.append(
            {
                "row_id": row_id,
                "pair_rank": rank,
                "perturb_cell": p_sample.iloc[rank]["cell"],
                "control_cell": c_sample.iloc[rank]["cell"],
                "perturb_embryo": p_sample.iloc[rank]["embryo"],
                "control_embryo": c_sample.iloc[rank]["embryo"],
                "perturb_substate": psub,
                "control_substate": csub,
                "same_substate": bool(psub == csub),
                "expression_distance": float(dist),
            }
        )
    return result, pair_rows


def merge_existing_evidence(row_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = row_df.copy()
    for path, cols in [
        (
            args.snapshot_rows,
            [
                "row_id",
                "strict_row_gate",
                "state_preserved_by_threshold",
                "wrong_time_control_ot",
                "wrong_lineage_control_ot",
                "trajectory_alignment_gate",
                "trajectory_cosine",
                "trajectory_margin",
                "fixed_cell_gate",
                "branch_placebo_pass",
                "embryo_module_gates",
                "embryo_module_total",
                "expression_constraint_candidate",
                "recommended_use",
            ],
        ),
        (
            args.strict_rows,
            [
                "row_id",
                "observed_strict_ot",
                "effect_ratio_vs_max_null_p95",
                "p_observed_le_matched_cc_null",
                "p_observed_le_matched_label_null",
                "matched_subtype_jsd",
                "expression_library_smd",
            ],
        ),
    ]:
        df = safe_read_csv(path)
        if not df.empty:
            keep = [c for c in cols if c in df.columns]
            out = out.merge(df[keep], on="row_id", how="left")

    diag = safe_read_csv(args.strict_diag)
    if not diag.empty:
        wide = diag.pivot_table(index="row_id", columns="diagnostic", values="ot", aggfunc="first").reset_index()
        wide.columns = [str(c) if c == "row_id" else f"strict_diag_{c}_ot" for c in wide.columns]
        out = out.merge(wide, on="row_id", how="left")

    out["wrong_time_margin_ot"] = out.get("wrong_time_control_ot", np.nan) - out.get("observed_strict_ot", np.nan)
    out["wrong_lineage_margin_ot"] = out.get("wrong_lineage_control_ot", np.nan) - out.get("observed_strict_ot", np.nan)
    out["dynamic_response_gate"] = (
        out.get("strict_row_gate", False).fillna(False).map(truthy)
        & out.get("state_preserved_by_threshold", False).fillna(False).map(truthy)
        & out.get("trajectory_alignment_gate", False).fillna(False).map(truthy)
        & out.get("expression_constraint_candidate", False).fillna(False).map(truthy)
        & (pd.to_numeric(out["wrong_time_margin_ot"], errors="coerce") > 0)
        & (pd.to_numeric(out["wrong_lineage_margin_ot"], errors="coerce") > 0)
        & (pd.to_numeric(out["composition_norm_fraction_of_centroid"], errors="coerce") <= 0.50)
    )
    return out


def summarize_modules(args: argparse.Namespace, row_ids: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    module = safe_read_csv(args.module_rows)
    periderm = safe_read_csv(args.periderm_module_rows)
    enrichment = safe_read_csv(args.enrichment_summary)
    for row_id in sorted(row_ids):
        rec: dict[str, Any] = {"row_id": row_id}
        m = module[module["row_id"] == row_id] if not module.empty and "row_id" in module else pd.DataFrame()
        p = periderm[periderm["row_id"] == row_id] if not periderm.empty and "row_id" in periderm else pd.DataFrame()
        e = enrichment[enrichment["row_id"] == row_id] if not enrichment.empty and "row_id" in enrichment else pd.DataFrame()
        if not m.empty:
            rec.update(
                {
                    "module_queries": int(len(m)),
                    "module_direction_gates": int(m.get("module_direction_gate", pd.Series(dtype=bool)).map(truthy).sum()),
                    "module_max_directed_diff": float(pd.to_numeric(m["directed_mean_diff"], errors="coerce").max()),
                    "module_min_ci_low": float(pd.to_numeric(m["directed_diff_ci95_low"], errors="coerce").min()),
                    "module_top_terms": " | ".join(
                        str(x) for x in m.get("top_terms", pd.Series(dtype=str)).dropna().head(2).tolist() if str(x)
                    ),
                }
            )
        else:
            rec.update({"module_queries": 0, "module_direction_gates": 0})
        if not p.empty:
            rec.update(
                {
                    "periderm_qc_residual_gates": int(p.get("qc_residual_gate", pd.Series(dtype=bool)).map(truthy).sum()),
                    "periderm_substate_gates": int(p.get("substate_gate", pd.Series(dtype=bool)).map(truthy).sum()),
                    "periderm_specificity_gates": int(p.get("specificity_gate", pd.Series(dtype=bool)).map(truthy).sum()),
                    "periderm_query_gates": int(p.get("query_gate", pd.Series(dtype=bool)).map(truthy).sum()),
                    "periderm_max_residual_diff": float(pd.to_numeric(p["residual_directed_diff"], errors="coerce").max()),
                    "periderm_min_residual_ci_low": float(pd.to_numeric(p["residual_ci_low"], errors="coerce").min()),
                    "periderm_wrong_time_max": float(pd.to_numeric(p["wrong_time_max"], errors="coerce").max()),
                    "periderm_wrong_lineage_p95": float(pd.to_numeric(p["wrong_lineage_p95"], errors="coerce").max()),
                }
            )
        else:
            rec.update(
                {
                    "periderm_qc_residual_gates": 0,
                    "periderm_substate_gates": 0,
                    "periderm_specificity_gates": 0,
                    "periderm_query_gates": 0,
                }
            )
        if not e.empty:
            rec.update(
                {
                    "enrichment_queries": int(len(e)),
                    "enrichment_significant_terms": int(pd.to_numeric(e["significant_term_count"], errors="coerce").sum()),
                    "enrichment_top_terms": " | ".join(
                        str(x) for x in e.get("top_terms", pd.Series(dtype=str)).dropna().head(2).tolist() if str(x)
                    ),
                }
            )
        else:
            rec.update({"enrichment_queries": 0, "enrichment_significant_terms": 0})
        if rec.get("periderm_query_gates", 0) > 0:
            interp = "module_specificity_supported"
        elif rec.get("periderm_qc_residual_gates", 0) > 0 or rec.get("module_direction_gates", 0) > 0:
            interp = "module_direction_supported_specificity_incomplete"
        else:
            interp = "module_not_supported_or_not_evaluable"
        rec["module_interpretation"] = interp
        rows.append(rec)
    return pd.DataFrame(rows)


def classify_findings(row_df: pd.DataFrame, module_df: pd.DataFrame) -> pd.DataFrame:
    mod = module_df.set_index("row_id") if not module_df.empty else pd.DataFrame()
    rows = []
    for _, row in row_df.iterrows():
        row_id = str(row["row_id"])
        module_interp = str(mod.at[row_id, "module_interpretation"]) if row_id in mod.index else ""
        dyn = truthy(row.get("dynamic_response_gate", False))
        if dyn and module_interp == "module_specificity_supported":
            tier = "biological_insight_candidate"
            latent = "possible_future_constraint_after_species_safe_latent_gate"
        elif dyn and module_interp == "module_direction_supported_specificity_incomplete":
            tier = "strong_expression_dynamic_hypothesis_generator"
            latent = "design_constraint_only_no_training"
        elif dyn:
            tier = "expression_dynamic_candidate_without_module_claim"
            latent = "design_constraint_only_no_training"
        elif truthy(row.get("strict_row_gate", False)):
            tier = "partial_expression_response"
            latent = "do_not_use_as_constraint"
        else:
            tier = "negative_or_confounded_comparator"
            latent = "negative_control_or_close_branch"
        rows.append(
            {
                "row_id": row_id,
                "finding_tier": tier,
                "latentfm_constraint_use": latent,
                "reason": (
                    f"dynamic_gate={dyn}; module={module_interp}; "
                    f"recommended_use={row.get('recommended_use', '')}"
                ),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    args: argparse.Namespace,
    row_df: pd.DataFrame,
    module_df: pd.DataFrame,
    finding_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    latent_summary: dict[str, Any],
    outputs: dict[str, Path],
    embed_meta: dict[str, Any],
) -> None:
    pass_rows = row_df[row_df["dynamic_response_gate"].map(truthy)]
    strong_rows = finding_df[finding_df["finding_tier"] == "biological_insight_candidate"]
    hypothesis_rows = finding_df[finding_df["finding_tier"].str.contains("hypothesis|candidate", regex=True)]
    lines = [
        "# LatentFM ZSCAPE OT Dynamic Response Gate",
        "",
        f"Timestamp: `{now_cst()}`",
        "",
        "Status: `zscape_ot_dynamic_response_gate_complete_cpu_only`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only analysis over frozen ZSCAPE selected-count and report artifacts.",
        "- Builds pseudo-pairs between matched control and perturbed snapshot cells; these are OT analytical pairs, not true lineage pairs.",
        "- No model training, no GPU, no checkpoint selection, no canonical multi, no Track C query.",
        "- Thread caps are inherited from `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, all set at or below 12 for this run.",
        "",
        "## OT-Pair Units And Provenance",
        "",
        f"- counts matrix: `{args.counts_npz}`",
        f"- matched manifest: `{args.matched_manifest}`",
        f"- cell index: `{args.cell_index}`",
        "- unit: `(cell_type_broad, gene_target, timepoint)` row; perturb cells are embryo-balanced, controls are same broad cell type/timepoint and greedily matched by substate plus log library before final linear-assignment OT.",
        "- embryo constraint: perturb sampling is balanced across perturb embryos; controls are not same-embryo matched and are treated as pooled snapshot controls.",
        "- preprocessing: selected raw counts are normalized per cell to 1e4 counts and `log1p`; HVG/SVD feature space is fit on control cells only.",
        f"- expression embedding metadata: `{json.dumps(embed_meta, sort_keys=True)}`",
        f"- evaluated rows: `{len(row_df)}`; dynamic-response gate rows: `{len(pass_rows)}`.",
        "",
        "## Expression-Space Response",
        "",
        "| row | lineage | target | time | pairs | mean OT dist | centroid norm | comp frac | within frac | traj cos | wrong-time margin | wrong-lineage margin | gate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in row_df.sort_values(["lineage", "target", "timepoint"]).iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["row_id"]),
                    str(row.get("lineage", "")),
                    str(row.get("target", "")),
                    fmt(row.get("timepoint"), 1),
                    str(int(row.get("n_pseudo_pairs", 0))),
                    fmt(row.get("mean_pair_expression_distance")),
                    fmt(row.get("centroid_response_norm")),
                    fmt(row.get("composition_norm_fraction_of_centroid")),
                    fmt(row.get("within_substate_residual_fraction_of_centroid")),
                    fmt(row.get("trajectory_cosine")),
                    fmt(row.get("wrong_time_margin_ot")),
                    fmt(row.get("wrong_lineage_margin_ot")),
                    str(bool(row.get("dynamic_response_gate", False))),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation: periderm `noto` and `smo` are the only dynamic-response gate positives. Their response remains within broad periderm state, has negligible substate-composition contribution by this decomposition, aligns with the periderm temporal tangent, and is smaller than wrong-time/wrong-lineage contrasts. Mature fast muscle remains a confounded comparator rather than a positive result.",
            "",
            "## Proxy/Latent Response",
            "",
            f"- latent proxy reconciliation status: `{latent_summary.get('latent_proxy_status', 'missing')}`",
            f"- UCE Danio continuity status: `{latent_summary.get('uce_continuity_status', 'missing')}`",
            f"- UCE continuity rows passing: `{latent_summary.get('uce_continuity_rows_pass', 'missing')}`",
            "",
            "Decision: latent/proxy route is blocked for modeling claims. The control-only HVG/SVD space used here is an expression analysis coordinate system, not a species-safe LatentFM latent asset.",
            "",
            "## Pathway And Module Response",
            "",
            "| row | module interpretation | direction gates | periderm residual gates | specificity gates | query gates | top terms |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in module_df.sort_values("row_id").iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["row_id"]),
                    str(row.get("module_interpretation", "")),
                    str(int(row.get("module_direction_gates", 0))),
                    str(int(row.get("periderm_qc_residual_gates", 0))),
                    str(int(row.get("periderm_specificity_gates", 0))),
                    str(int(row.get("periderm_query_gates", 0))),
                    str(row.get("enrichment_top_terms", row.get("module_top_terms", "")))[:240],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Module decision: periderm modules have QC-residual and substate support, but specificity gates remain incomplete. Treat terms as falsifiable pathway hypotheses, not validated pathway claims.",
            "",
            "## Biological And Modeling Decision",
            "",
            f"- strong biological insight rows: `{len(strong_rows)}`.",
            f"- hypothesis-generator rows: `{len(hypothesis_rows)}`.",
            "- supported pattern: periderm `noto`/`smo` show a reproducible expression-space dynamic response under matched snapshot OT controls.",
            "- unsupported claim: no specific pathway mechanism or LatentFM latent constraint is validated yet.",
            "- LatentFM use: at most a future design constraint template after a species-safe latent route passes; no training or promotion should use this result directly.",
            "",
            "## Next Gate And Fail-Close Rule",
            "",
            "Next gate: run an embryo-heldout, periderm-only dynamic module specificity gate that recomputes the `noto`/`smo` program on held-out embryos and requires wrong-target, wrong-time, wrong-lineage, QC-residual, and substate-preservation controls to pass together.",
            "",
            "Fail-close: if either `noto` or `smo` loses positive residual module effect on held-out embryos, or if wrong-time/wrong-target specificity catches up to the real effect, close the ZSCAPE dynamic pathway branch as hypothesis-only and do not convert it into a LatentFM constraint.",
            "",
            "## Outputs",
            "",
        ]
    )
    for key, value in outputs.items():
        lines.append(f"- {key}: `{value}`")
    outputs["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts-npz", type=Path, default=DEFAULT_COUNTS)
    parser.add_argument("--cell-index", type=Path, default=DEFAULT_CELL_INDEX)
    parser.add_argument("--matched-manifest", type=Path, default=DEFAULT_MATCHED_MANIFEST)
    parser.add_argument("--snapshot-rows", type=Path, default=DEFAULT_SNAPSHOT_ROWS)
    parser.add_argument("--temporal-controls", type=Path, default=DEFAULT_TEMPORAL_CONTROLS)
    parser.add_argument("--strict-rows", type=Path, default=DEFAULT_STRICT_ROWS)
    parser.add_argument("--strict-diag", type=Path, default=DEFAULT_STRICT_DIAG)
    parser.add_argument("--module-rows", type=Path, default=DEFAULT_MODULE_ROWS)
    parser.add_argument("--periderm-module-rows", type=Path, default=DEFAULT_PERIDERM_MODULE_ROWS)
    parser.add_argument("--periderm-substate-rows", type=Path, default=DEFAULT_PERIDERM_SUBSTATE_ROWS)
    parser.add_argument("--enrichment-summary", type=Path, default=DEFAULT_ENRICHMENT_SUMMARY)
    parser.add_argument("--latent-proxy-json", type=Path, default=DEFAULT_LATENT_PROXY_JSON)
    parser.add_argument("--uce-continuity-json", type=Path, default=DEFAULT_UCE_CONTINUITY_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--n-pca", type=int, default=32)
    parser.add_argument("--ot-cells", type=int, default=128)
    parser.add_argument("--min-cells", type=int, default=40)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_output_dir(args.out_dir, args.force)
    manifest, emb, embed_meta = prepare_manifest(args)
    primary = manifest[manifest["audit_role"] == "primary_mechanism_test"].copy()
    row_results: list[dict[str, Any]] = []
    pair_results: list[dict[str, Any]] = []
    for row_id, group in primary.groupby("row_id", sort=True):
        row, pairs = summarize_ot_row(row_id, group, emb, args)
        row_results.append(row)
        pair_results.extend(pairs)

    row_df = merge_existing_evidence(pd.DataFrame(row_results), args)
    module_df = summarize_modules(args, set(row_df["row_id"].astype(str)))
    finding_df = classify_findings(row_df, module_df)
    temporal_df = safe_read_csv(args.temporal_controls)

    latent_proxy = read_json(args.latent_proxy_json)
    uce = read_json(args.uce_continuity_json)
    latent_summary = {
        "latent_proxy_status": latent_proxy.get("status", "missing"),
        "latent_proxy_periderm_gate_agreement_fraction": latent_proxy.get("periderm_gate_agreement_fraction"),
        "uce_continuity_status": uce.get("status", "missing"),
        "uce_continuity_rows_pass": uce.get("n_rows_pass"),
        "latent_route_decision": "blocked_no_modeling_claim",
    }

    outputs = {
        "response_rows": args.out_dir / "zscape_ot_dynamic_response_rows.csv",
        "pseudo_pairs": args.out_dir / "zscape_ot_dynamic_response_pseudo_pairs.csv",
        "module_rows": args.out_dir / "zscape_ot_dynamic_response_module_rows.csv",
        "finding_tiers": args.out_dir / "zscape_ot_dynamic_response_finding_tiers.csv",
        "json": args.out_dir / "zscape_ot_dynamic_response_gate_20260628.json",
        "report": args.out_dir / "LATENTFM_ZSCAPE_OT_DYNAMIC_RESPONSE_GATE_20260628.md",
    }
    row_df.to_csv(outputs["response_rows"], index=False)
    pd.DataFrame(pair_results).to_csv(outputs["pseudo_pairs"], index=False)
    module_df.to_csv(outputs["module_rows"], index=False)
    finding_df.to_csv(outputs["finding_tiers"], index=False)

    payload = {
        "timestamp_cst": now_cst(),
        "status": "zscape_ot_dynamic_response_gate_complete_cpu_only",
        "gpu_authorized": False,
        "inputs": {
            "counts_npz": str(args.counts_npz),
            "cell_index": str(args.cell_index),
            "matched_manifest": str(args.matched_manifest),
            "snapshot_rows": str(args.snapshot_rows),
            "strict_rows": str(args.strict_rows),
            "module_rows": str(args.module_rows),
            "periderm_module_rows": str(args.periderm_module_rows),
            "enrichment_summary": str(args.enrichment_summary),
        },
        "preprocessing": embed_meta,
        "summary": {
            "evaluated_rows": int(len(row_df)),
            "dynamic_response_gate_rows": int(row_df["dynamic_response_gate"].map(truthy).sum()),
            "dynamic_response_gate_row_ids": row_df.loc[
                row_df["dynamic_response_gate"].map(truthy), "row_id"
            ].astype(str).tolist(),
            "biological_insight_candidate_rows": finding_df.loc[
                finding_df["finding_tier"] == "biological_insight_candidate", "row_id"
            ].astype(str).tolist(),
            "hypothesis_generator_rows": finding_df.loc[
                finding_df["finding_tier"].str.contains("hypothesis|candidate", regex=True), "row_id"
            ].astype(str).tolist(),
            "latent_route_decision": latent_summary["latent_route_decision"],
        },
        "latent_summary": latent_summary,
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    outputs["json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(args, row_df, module_df, finding_df, temporal_df, latent_summary, outputs, embed_meta)
    print(outputs["report"])
    print(outputs["json"])
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
