#!/usr/bin/env python3
"""Synthesize ZSCAPE dynamic-information evidence into modeling gates.

This is a CPU/report-only integration over already materialized ZSCAPE
expression-space reports. It does not train models, extract embeddings, or
authorize GPU work.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
OT_ROWS = ROOT / "reports/zscape_ot_dynamic_response_gate_20260628/zscape_ot_dynamic_response_rows.csv"
MODULE_ROWS = ROOT / "reports/zscape_ot_dynamic_response_gate_20260628/zscape_ot_dynamic_response_module_rows.csv"
PERIDERM_QUERY_ROWS = (
    ROOT
    / "reports/zscape_periderm_substate_time_qc_ot_module_gate_20260628"
    / "zscape_periderm_substate_time_qc_module_query_rows.csv"
)
HELDOUT_SUMMARY = (
    ROOT
    / "reports/zscape_embryo_heldout_periderm_module_specificity_20260628"
    / "zscape_embryo_heldout_periderm_module_specificity_summary.csv"
)
HELDOUT_EMBRYO_ROWS = (
    ROOT
    / "reports/zscape_embryo_heldout_periderm_module_specificity_20260628"
    / "zscape_embryo_heldout_periderm_module_specificity_heldout_rows.csv"
)
HVG_CURVE = ROOT / "reports/zscape_hvg_fullgene_information_axis_20260628/zscape_hvg_fullgene_information_curve.csv"
OUT_DIR = ROOT / "reports/zscape_dynamic_information_modeling_gate_20260628"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return df[col].map(truthy)


def summarize_queries(query_df: pd.DataFrame, heldout_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if query_df.empty and heldout_df.empty:
        return pd.DataFrame(rows)

    heldout_by_query = heldout_df.set_index("query_name", drop=False) if "query_name" in heldout_df.columns else pd.DataFrame()
    for _, q in query_df.iterrows():
        qname = str(q.get("query_name", ""))
        h = heldout_by_query.loc[qname] if not heldout_by_query.empty and qname in heldout_by_query.index else pd.Series(dtype=object)
        heldout_mean = h.get("heldout_mean_effect", q.get("residual_directed_diff", float("nan")))
        heldout_low = h.get("heldout_ci_low", q.get("residual_ci_low", float("nan")))
        wrong_qmax = h.get("wrong_control_qmax", q.get("periderm_placebo_p95", float("nan")))
        try:
            specificity_margin = float(heldout_low) - float(wrong_qmax)
        except (TypeError, ValueError):
            specificity_margin = float("nan")
        rows.append(
            {
                "query_name": qname,
                "row_id": q.get("row_id", ""),
                "direction": q.get("direction", ""),
                "heldout_mean_effect": heldout_mean,
                "heldout_ci_low": heldout_low,
                "wrong_control_qmax": wrong_qmax,
                "specificity_margin_low_minus_wrong": specificity_margin,
                "qc_residual_gate": truthy(q.get("qc_residual_gate", False)),
                "substate_gate": truthy(q.get("substate_gate", False)),
                "heldout_gate": truthy(h.get("heldout_gate", False)) if not h.empty else False,
                "specificity_gate": truthy(h.get("specificity_gate", q.get("specificity_gate", False))) if not h.empty else truthy(q.get("specificity_gate", False)),
                "query_gate": truthy(h.get("query_gate", q.get("query_gate", False))) if not h.empty else truthy(q.get("query_gate", False)),
                "top_terms": q.get("top_terms", ""),
            }
        )
    return pd.DataFrame(rows)


def embryo_consistency(embryo_df: pd.DataFrame) -> pd.DataFrame:
    if embryo_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (row_id, query_name), grp in embryo_df.groupby(["row_id", "query_name"], dropna=False):
        effects = pd.to_numeric(grp["heldout_directed_effect"], errors="coerce").dropna()
        rows.append(
            {
                "row_id": row_id,
                "query_name": query_name,
                "n_embryos": int(effects.shape[0]),
                "mean_effect": float(effects.mean()) if not effects.empty else float("nan"),
                "min_effect": float(effects.min()) if not effects.empty else float("nan"),
                "positive_fraction": float((effects > 0).mean()) if not effects.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_row_synthesis(ot: pd.DataFrame, modules: pd.DataFrame, query_summary: pd.DataFrame) -> pd.DataFrame:
    if ot.empty:
        return pd.DataFrame()

    module_by_row = modules.set_index("row_id", drop=False) if "row_id" in modules.columns else pd.DataFrame()
    query_group = query_summary.groupby("row_id", dropna=False) if not query_summary.empty else None
    rows: list[dict[str, Any]] = []

    for _, row in ot.iterrows():
        row_id = row.get("row_id", "")
        module = module_by_row.loc[row_id] if not module_by_row.empty and row_id in module_by_row.index else pd.Series(dtype=object)
        if query_group is not None and row_id in query_group.groups:
            qgrp = query_group.get_group(row_id)
            n_queries = int(qgrp.shape[0])
            heldout_pass = int(qgrp["heldout_gate"].sum())
            specificity_pass = int(qgrp["specificity_gate"].sum())
            query_pass = int(qgrp["query_gate"].sum())
            min_specificity_margin = float(pd.to_numeric(qgrp["specificity_margin_low_minus_wrong"], errors="coerce").min())
            min_heldout_ci_low = float(pd.to_numeric(qgrp["heldout_ci_low"], errors="coerce").min())
        else:
            n_queries = heldout_pass = specificity_pass = query_pass = 0
            min_specificity_margin = float("nan")
            min_heldout_ci_low = float("nan")

        geometry_gate = bool(
            truthy(row.get("dynamic_response_gate", False))
            and truthy(row.get("state_preserved_by_threshold", False))
            and float(row.get("wrong_time_margin_ot", float("nan"))) > 0
            and float(row.get("wrong_lineage_margin_ot", float("nan"))) > 0
            and float(row.get("composition_norm_fraction_of_centroid", float("inf"))) <= 0.5
        )
        module_direction = int(module.get("module_direction_gates", 0) or 0)
        module_total = int(module.get("module_queries", 0) or 0)
        module_direction_supported = module_total > 0 and module_direction == module_total
        heldout_all = n_queries > 0 and heldout_pass == n_queries
        specificity_all = n_queries > 0 and specificity_pass == n_queries

        if geometry_gate and heldout_all and not specificity_all:
            modeling_class = "geometry_positive_module_specificity_failed"
            modeling_action = "use_for_geometry_diagnostics_not_pathway_loss"
        elif geometry_gate and specificity_all:
            modeling_class = "geometry_and_specificity_positive"
            modeling_action = "candidate_for_future_constraint_after_latent_raw_route"
        elif "negative_confounded" in str(row.get("recommended_use", "")):
            modeling_class = "negative_confounded_comparator"
            modeling_action = "keep_as_negative_control_do_not_train_positive_weight"
        elif geometry_gate:
            modeling_class = "geometry_positive_module_unresolved"
            modeling_action = "run_module_or_embryo_specificity_gate"
        else:
            modeling_class = "diagnostic_or_unsupported"
            modeling_action = "diagnostic_only"

        rows.append(
            {
                "row_id": row_id,
                "lineage": row.get("lineage", ""),
                "target": row.get("target", ""),
                "timepoint": row.get("timepoint", ""),
                "n_pseudo_pairs": row.get("n_pseudo_pairs", ""),
                "centroid_response_norm": row.get("centroid_response_norm", ""),
                "composition_fraction": row.get("composition_norm_fraction_of_centroid", ""),
                "within_substate_fraction": row.get("within_substate_residual_fraction_of_centroid", ""),
                "trajectory_cosine": row.get("trajectory_cosine", ""),
                "wrong_time_margin_ot": row.get("wrong_time_margin_ot", ""),
                "wrong_lineage_margin_ot": row.get("wrong_lineage_margin_ot", ""),
                "dynamic_response_gate": truthy(row.get("dynamic_response_gate", False)),
                "geometry_gate": geometry_gate,
                "module_direction_supported": module_direction_supported,
                "heldout_queries": n_queries,
                "heldout_pass": heldout_pass,
                "specificity_pass": specificity_pass,
                "query_pass": query_pass,
                "min_heldout_ci_low": min_heldout_ci_low,
                "min_specificity_margin_low_minus_wrong": min_specificity_margin,
                "modeling_class": modeling_class,
                "modeling_action": modeling_action,
            }
        )
    return pd.DataFrame(rows)


def hvg_budget_summary(curve: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if curve.empty:
        return out
    top_col = "top_genes" if "top_genes" in curve.columns else "top genes"
    primary_col = (
        "primary_rows_response_energy_share_mean"
        if "primary_rows_response_energy_share_mean" in curve.columns
        else "primary response mean"
    )
    var_col = "lognorm_variance_share" if "lognorm_variance_share" in curve.columns else "log-var share"
    top_values = pd.to_numeric(curve[top_col], errors="coerce") if top_col in curve.columns else pd.Series(index=curve.index)
    for budget in [1000, 2000, 4000, 8000]:
        sub = curve[top_values == budget]
        if not sub.empty:
            row = sub.iloc[0]
            out[f"top{budget}_primary_response_mean"] = float(row.get(primary_col, float("nan")))
            out[f"top{budget}_log_var_share"] = float(row.get(var_col, float("nan")))
    return out


def write_report(
    out_dir: Path,
    row_syn: pd.DataFrame,
    query_syn: pd.DataFrame,
    embryo_syn: pd.DataFrame,
    hvg_summary: dict[str, Any],
) -> None:
    geometry_pos = int((row_syn["geometry_gate"] == True).sum()) if not row_syn.empty else 0
    specificity_pos = int((row_syn["specificity_pass"] > 0).sum()) if not row_syn.empty else 0
    geometry_specificity_failed = (
        row_syn[row_syn["modeling_class"] == "geometry_positive_module_specificity_failed"]
        if not row_syn.empty
        else pd.DataFrame()
    )
    gpu = False
    status = "zscape_dynamic_information_modeling_gate_geometry_only_no_gpu"

    lines: list[str] = []
    lines.append("# ZSCAPE Dynamic-Information Modeling Gate")
    lines.append("")
    lines.append(f"Timestamp: `{now_cst()}`")
    lines.append("")
    lines.append(f"Status: `{status}`")
    lines.append("")
    lines.append(f"GPU authorized: `{gpu}`")
    lines.append("")
    lines.append("## Boundary")
    lines.append("")
    lines.append("- CPU/report-only synthesis over frozen ZSCAPE OT, module, and heldout reports.")
    lines.append("- OT pairs are snapshot pseudo-pairs, not true lineage pairs.")
    lines.append("- No model training, embedding extraction, canonical multi, Track C query, or checkpoint selection.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Rows synthesized: `{len(row_syn)}`.")
    lines.append(f"- Geometry-positive rows: `{geometry_pos}`.")
    lines.append(f"- Rows with any specificity-positive query: `{specificity_pos}`.")
    lines.append(f"- Geometry-positive but specificity-failed rows: `{len(geometry_specificity_failed)}`.")
    if hvg_summary:
        lines.append(
            "- HVG response budget: "
            f"top2k primary response `{fmt(hvg_summary.get('top2000_primary_response_mean'))}`, "
            f"top8k primary response `{fmt(hvg_summary.get('top8000_primary_response_mean'))}`."
        )
    lines.append("")
    lines.append("## Row Decisions")
    lines.append("")
    show_cols = [
        "row_id",
        "geometry_gate",
        "composition_fraction",
        "trajectory_cosine",
        "heldout_pass",
        "specificity_pass",
        "min_specificity_margin_low_minus_wrong",
        "modeling_class",
        "modeling_action",
    ]
    lines.append("| " + " | ".join(show_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(show_cols)) + "|")
    for _, row in row_syn.iterrows():
        vals = []
        for col in show_cols:
            val = row.get(col, "")
            vals.append(fmt(val) if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## Query Specificity")
    lines.append("")
    if query_syn.empty:
        lines.append("No query-level module rows were available.")
    else:
        qcols = [
            "query_name",
            "heldout_mean_effect",
            "heldout_ci_low",
            "wrong_control_qmax",
            "specificity_margin_low_minus_wrong",
            "heldout_gate",
            "specificity_gate",
        ]
        lines.append("| " + " | ".join(qcols) + " |")
        lines.append("|" + "|".join(["---"] * len(qcols)) + "|")
        for _, row in query_syn.iterrows():
            vals = []
            for col in qcols:
                val = row.get(col, "")
                vals.append(fmt(val) if isinstance(val, float) else str(val))
            lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("- ZSCAPE currently supports state-preserving expression-space response geometry for periderm `noto/smo`.")
    lines.append("- It does not support a pathway-specific or latent-space model constraint because wrong controls remain competitive.")
    lines.append("- Modeling translation now: add diagnostics and negative controls; defer loss/architecture changes until species-safe latent/raw route and specificity gates pass.")
    lines.append("")
    lines.append("## Next Gates")
    lines.append("")
    lines.append("1. Embryo-level vector consistency for centroid/OT deltas, not only module scores.")
    lines.append("2. Common periderm response program with abundance-matched random gene sets.")
    lines.append("3. RawFM observable-budget readiness before any gene-budget GPU smoke.")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- row synthesis: `{out_dir / 'zscape_dynamic_information_row_synthesis.csv'}`")
    lines.append(f"- query synthesis: `{out_dir / 'zscape_dynamic_information_query_synthesis.csv'}`")
    lines.append(f"- embryo consistency: `{out_dir / 'zscape_dynamic_information_embryo_consistency.csv'}`")
    lines.append(f"- JSON: `{out_dir / 'zscape_dynamic_information_modeling_gate_20260628.json'}`")
    (out_dir / "LATENTFM_ZSCAPE_DYNAMIC_INFORMATION_MODELING_GATE_20260628.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ot = read_csv(OT_ROWS)
    modules = read_csv(MODULE_ROWS)
    query_rows = read_csv(PERIDERM_QUERY_ROWS)
    heldout = read_csv(HELDOUT_SUMMARY)
    embryo = read_csv(HELDOUT_EMBRYO_ROWS)
    curve = read_csv(HVG_CURVE)

    query_syn = summarize_queries(query_rows, heldout)
    embryo_syn = embryo_consistency(embryo)
    row_syn = build_row_synthesis(ot, modules, query_syn)
    hvg_summary = hvg_budget_summary(curve)

    row_path = args.out_dir / "zscape_dynamic_information_row_synthesis.csv"
    query_path = args.out_dir / "zscape_dynamic_information_query_synthesis.csv"
    embryo_path = args.out_dir / "zscape_dynamic_information_embryo_consistency.csv"
    row_syn.to_csv(row_path, index=False)
    query_syn.to_csv(query_path, index=False)
    embryo_syn.to_csv(embryo_path, index=False)

    status = "zscape_dynamic_information_modeling_gate_geometry_only_no_gpu"
    obj = {
        "timestamp": now_cst(),
        "status": status,
        "gpu_authorized_next": False,
        "rows": {
            "n_rows": int(len(row_syn)),
            "geometry_positive_rows": int((row_syn.get("geometry_gate", pd.Series(dtype=bool)) == True).sum()),
            "geometry_positive_specificity_failed_rows": int(
                (row_syn.get("modeling_class", pd.Series(dtype=str)) == "geometry_positive_module_specificity_failed").sum()
            ),
        },
        "hvg_summary": hvg_summary,
        "outputs": {
            "row_synthesis": str(row_path),
            "query_synthesis": str(query_path),
            "embryo_consistency": str(embryo_path),
            "report": str(args.out_dir / "LATENTFM_ZSCAPE_DYNAMIC_INFORMATION_MODELING_GATE_20260628.md"),
        },
    }
    write_json(args.out_dir / "zscape_dynamic_information_modeling_gate_20260628.json", obj)
    write_report(args.out_dir, row_syn, query_syn, embryo_syn, hvg_summary)


if __name__ == "__main__":
    main()
