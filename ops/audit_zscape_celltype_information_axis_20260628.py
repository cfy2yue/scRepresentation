#!/usr/bin/env python3
"""CPU-only ZSCAPE cell-type information-axis audit.

This report turns the current ZSCAPE expression-space branch into a
cell-type/subpopulation information-axis candidate for downstream scaling-law
work.  It consumes frozen reports only; it does not train, infer, or extract
latent embeddings.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "reports" / "zscape_celltype_information_axis_20260628"
MANIFEST = (
    ROOT
    / "runs"
    / "zscape_raw_counts_cell_manifest_extraction_20260628"
    / "zscape_raw_counts_cell_manifest_extraction_20260628_074523"
    / "outputs"
    / "zscape_expression_selected_cell_ids_matched.csv"
)
HVG_ROWS = (
    ROOT
    / "reports"
    / "zscape_hvg_fullgene_information_axis_20260628"
    / "zscape_hvg_response_energy_rows.csv"
)
DE_ROWS = (
    ROOT
    / "reports"
    / "zscape_expression_latent_biology_preflight_20260628"
    / "zscape_expression_de_row_summary.csv"
)
SNAPSHOT_ROWS = (
    ROOT
    / "reports"
    / "zscape_snapshot_dynamic_constraint_spec_20260628"
    / "zscape_snapshot_dynamic_constraint_rows.csv"
)
MODULE_ROWS = (
    ROOT
    / "reports"
    / "zscape_embryo_pseudobulk_module_gate_20260628"
    / "zscape_embryo_pseudobulk_module_rows.csv"
)


def safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def entropy_fraction(values: pd.Series) -> float:
    counts = values.dropna().astype(str).value_counts()
    if counts.empty:
        return 0.0
    probs = counts / counts.sum()
    entropy = float(-(probs * np.log(probs)).sum())
    denom = math.log(len(counts)) if len(counts) > 1 else 1.0
    return entropy / denom


def zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    mean = vals.mean()
    std = vals.std(ddof=0)
    if not math.isfinite(std) or std == 0:
        return pd.Series(np.zeros(len(vals)), index=series.index)
    return (vals - mean) / std


def bool_sum(series: pd.Series) -> int:
    return int(series.fillna(False).astype(bool).sum())


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(MANIFEST)
    hvg = pd.read_csv(HVG_ROWS)
    de = pd.read_csv(DE_ROWS)
    snapshot = pd.read_csv(SNAPSHOT_ROWS)
    modules = pd.read_csv(MODULE_ROWS)
    return manifest, hvg, de, snapshot, modules


def build_lineage_table(
    manifest: pd.DataFrame,
    hvg: pd.DataFrame,
    de: pd.DataFrame,
    snapshot: pd.DataFrame,
    modules: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for lineage, frame in manifest.groupby("cell_type_broad", dropna=False):
        lineage = str(lineage)
        perturb = frame[frame["selection_role"].astype(str) == "perturb"]
        control = frame[frame["selection_role"].astype(str) == "control"]
        hvg_l = hvg[hvg["lineage"].astype(str) == lineage]
        de_l = de[de["lineage"].astype(str) == lineage]
        snap_l = snapshot[snapshot["lineage"].astype(str) == lineage]
        mod_l = modules[modules["lineage"].astype(str) == lineage]
        module_gate_col = "embryo_module_gate" if "embryo_module_gate" in mod_l.columns else "module_direction_gate"
        rows.append(
            {
                "lineage": lineage,
                "selected_cells": int(len(frame)),
                "unique_cells": int(frame["cell"].astype(str).nunique()),
                "perturb_cells": int(len(perturb)),
                "control_cells": int(len(control)),
                "embryos": int(frame["embryo"].astype(str).nunique()),
                "perturb_embryos": int(perturb["embryo"].astype(str).nunique()),
                "control_embryos": int(control["embryo"].astype(str).nunique()),
                "subtypes": int(frame["cell_type_sub"].astype(str).nunique()),
                "subtype_entropy_norm": entropy_fraction(frame["cell_type_sub"]),
                "targets": int(perturb["gene_target"].astype(str).nunique()),
                "timepoints": int(frame["timepoint"].astype(str).nunique()),
                "rows": int(frame["row_id"].astype(str).nunique()),
                "median_n_umi": float(pd.to_numeric(frame["n_umi"], errors="coerce").median()),
                "median_num_genes": float(pd.to_numeric(frame["num_genes_expressed"], errors="coerce").median()),
                "low_qc_fraction": float(
                    (
                        (pd.to_numeric(frame["n_umi"], errors="coerce") < 100)
                        | (pd.to_numeric(frame["num_genes_expressed"], errors="coerce") < 100)
                    ).mean()
                ),
                "umap3d_spread": float(
                    np.nanmean(
                        [
                            pd.to_numeric(frame[col], errors="coerce").std(ddof=0)
                            for col in ["umap3d_1", "umap3d_2", "umap3d_3"]
                            if col in frame
                        ]
                    )
                ),
                "mean_response_energy_total": float(hvg_l["response_energy_total"].mean())
                if not hvg_l.empty
                else float("nan"),
                "mean_hvg1000_response_share": float(hvg_l["hvg1000_response_energy_share"].mean())
                if not hvg_l.empty
                else float("nan"),
                "mean_hvg2000_response_share": float(hvg_l["hvg2000_response_energy_share"].mean())
                if not hvg_l.empty
                else float("nan"),
                "mean_de_response_l2": float(de_l["response_energy_l2"].mean())
                if not de_l.empty
                else float("nan"),
                "snapshot_constraint_candidates": bool_sum(
                    snap_l.get("expression_constraint_candidate", pd.Series(dtype=bool))
                ),
                "strict_rows": bool_sum(snap_l.get("strict_row_gate", pd.Series(dtype=bool))),
                "fixed_cell_rows": bool_sum(snap_l.get("fixed_cell_gate", pd.Series(dtype=bool))),
                "state_preserved_rows": bool_sum(
                    snap_l.get("state_preserved_by_threshold", pd.Series(dtype=bool))
                ),
                "module_rows": int(len(mod_l)),
                "module_gate_rows": bool_sum(mod_l.get(module_gate_col, pd.Series(dtype=bool))),
                "module_min_ci_low": float(pd.to_numeric(mod_l.get("ci_low"), errors="coerce").min())
                if not mod_l.empty and "ci_low" in mod_l
                else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    score_terms = {
        "log_cells": np.log1p(out["selected_cells"]),
        "log_embryos": np.log1p(out["embryos"]),
        "log_targets": np.log1p(out["targets"]),
        "subtype_entropy_norm": out["subtype_entropy_norm"],
        "log_response_energy": np.log1p(out["mean_response_energy_total"].fillna(0.0)),
        "hvg2000_response_share": out["mean_hvg2000_response_share"].fillna(0.0),
        "module_gate_fraction": out["module_gate_rows"] / out["module_rows"].replace(0, np.nan),
        "strict_fraction": out["strict_rows"] / out["rows"].replace(0, np.nan),
    }
    score_df = pd.DataFrame(score_terms).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for col in score_df.columns:
        out[f"z_{col}"] = zscore(score_df[col])
    out["information_axis_score"] = out[[f"z_{col}" for col in score_df.columns]].mean(axis=1)
    return out.sort_values("information_axis_score", ascending=False).reset_index(drop=True)


def correlation_rows(lineage: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("subtypes", "mean_response_energy_total"),
        ("subtype_entropy_norm", "mean_response_energy_total"),
        ("embryos", "mean_response_energy_total"),
        ("targets", "mean_response_energy_total"),
        ("selected_cells", "mean_response_energy_total"),
        ("subtypes", "mean_hvg2000_response_share"),
        ("subtype_entropy_norm", "mean_hvg2000_response_share"),
        ("umap3d_spread", "mean_response_energy_total"),
        ("low_qc_fraction", "mean_response_energy_total"),
    ]
    rows = []
    for x, y in pairs:
        sub = lineage[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < 3 or sub[x].nunique() < 2 or sub[y].nunique() < 2:
            rho = p = float("nan")
        else:
            rho, p = spearmanr(sub[x], sub[y])
            rho = safe_float(rho)
            p = safe_float(p)
        rows.append({"x": x, "y": y, "n": int(len(sub)), "spearman_rho": rho, "p": p})
    return pd.DataFrame(rows)


def write_report(lineage: pd.DataFrame, corr: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lineage_csv = OUT_DIR / "zscape_celltype_information_axis_rows.csv"
    corr_csv = OUT_DIR / "zscape_celltype_information_axis_correlations.csv"
    json_path = OUT_DIR / "zscape_celltype_information_axis_20260628.json"
    md_path = OUT_DIR / "LATENTFM_ZSCAPE_CELLTYPE_INFORMATION_AXIS_20260628.md"

    lineage.to_csv(lineage_csv, index=False)
    corr.to_csv(corr_csv, index=False)

    primary = lineage[lineage["lineage"].isin(["periderm", "mature fast muscle"])]
    top = lineage.head(10)
    corr_focus = corr.sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False).head(6)

    payload = {
        "status": "zscape_celltype_information_axis_ready_no_gpu",
        "gpu_authorized": False,
        "inputs": {
            "manifest": str(MANIFEST),
            "hvg_rows": str(HVG_ROWS),
            "de_rows": str(DE_ROWS),
            "snapshot_rows": str(SNAPSHOT_ROWS),
            "module_rows": str(MODULE_ROWS),
        },
        "n_lineages": int(len(lineage)),
        "top_lineages": top[["lineage", "information_axis_score"]].to_dict(orient="records"),
        "primary_lineages": primary.to_dict(orient="records"),
        "outputs": {"lineage_csv": str(lineage_csv), "correlation_csv": str(corr_csv)},
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM ZSCAPE Cell-Type Information Axis",
        "",
        f"Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S CST')}`",
        "",
        "Status: `zscape_celltype_information_axis_ready_no_gpu`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only audit over frozen ZSCAPE expression-space artifacts.",
        "- No model training, inference, true scFM embedding extraction, canonical multi selection, or Track C query use.",
        "- The information score is a hypothesis-generating scaling x-axis, not a proof of model improvement.",
        "- Expression preprocessing for upstream inputs remains raw counts -> size-factor normalization -> exactly one `log1p`; this script does not re-normalize cells.",
        "",
        "## Summary",
        "",
        f"- lineages audited: `{len(lineage)}`.",
        f"- row-context cells: `{int(lineage['selected_cells'].sum())}`.",
        f"- unique selected cells across lineages: `{int(lineage['unique_cells'].sum())}`.",
        f"- total row contexts: `{int(lineage['rows'].sum())}`.",
        f"- snapshot constraint candidates: `{int(lineage['snapshot_constraint_candidates'].sum())}`.",
        "",
        "## Top Information-Axis Lineages",
        "",
        "| lineage | score | row-context cells | unique cells | embryos | subtypes | subtype entropy | targets | mean response L2 | HVG2000 share | strict/fixed/module |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in top.iterrows():
        lines.append(
            "| {lineage} | {score:.3f} | {cells:d} | {unique:d} | {embryos:d} | {subtypes:d} | "
            "{entropy:.3f} | {targets:d} | {resp:.3f} | {hvg:.3f} | {strict:d}/{fixed:d}/{module:d} |".format(
                lineage=row["lineage"],
                score=float(row["information_axis_score"]),
                cells=int(row["selected_cells"]),
                unique=int(row["unique_cells"]),
                embryos=int(row["embryos"]),
                subtypes=int(row["subtypes"]),
                entropy=float(row["subtype_entropy_norm"]),
                targets=int(row["targets"]),
                resp=float(row["mean_response_energy_total"]),
                hvg=float(row["mean_hvg2000_response_share"]),
                strict=int(row["strict_rows"]),
                fixed=int(row["fixed_cell_rows"]),
                module=int(row["module_gate_rows"]),
            )
        )
    lines.extend(
        [
            "",
            "## Primary Interpretation Rows",
            "",
            "| lineage | score | role | key interpretation |",
            "|---|---:|---|---|",
        ]
    )
    interp = {
        "periderm": "Narrow positive biology axis: lower response magnitude than muscle but strict/fixed-cell/placebo/module support and two expression snapshot dynamic candidates.",
        "mature fast muscle": "High response-energy and module-score comparator, but strict controls failed, so it remains strong-but-confounded rather than a positive mechanism.",
    }
    for _, row in primary.iterrows():
        lines.append(
            f"| {row['lineage']} | {float(row['information_axis_score']):.3f} | "
            f"{'positive_candidate' if row['lineage'] == 'periderm' else 'negative_confounded_comparator'} | "
            f"{interp.get(str(row['lineage']), '')} |"
        )
    lines.extend(
        [
            "",
            "## Scaling-Axis Correlations",
            "",
            "| x | y | n | Spearman rho | p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in corr_focus.iterrows():
        lines.append(
            f"| {row['x']} | {row['y']} | {int(row['n'])} | "
            f"{safe_float(row['spearman_rho']):.3f} | {safe_float(row['p']):.3g} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "ZSCAPE supports a cell-type/subpopulation information-axis hypothesis, but the axis should be used as biological/scaling interpretation first. Periderm is the clean positive branch because it combines strict response, fixed-cell/placebo support, embryo-level module support, and state preservation. Mature fast muscle is deliberately retained as a high-effect confounded comparator.",
            "",
            "This does not authorize LatentFM GPU training. A future model route still needs a source/train-only materialization, shuffled/count/dataset controls, dual baseline against anchor and source/control, and no-harm evaluation.",
            "",
            "## Outputs",
            "",
            f"- lineage rows: `{lineage_csv}`",
            f"- correlations: `{corr_csv}`",
            f"- JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    manifest, hvg, de, snapshot, modules = load_inputs()
    lineage = build_lineage_table(manifest, hvg, de, snapshot, modules)
    corr = correlation_rows(lineage)
    write_report(lineage, corr)
    print(
        json.dumps(
            {
                "status": "zscape_celltype_information_axis_ready_no_gpu",
                "n_lineages": int(len(lineage)),
                "out_dir": str(OUT_DIR),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
