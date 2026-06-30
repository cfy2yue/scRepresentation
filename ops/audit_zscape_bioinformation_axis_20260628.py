#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def entropy(values: pd.Series) -> float:
    counts = values.astype(str).value_counts()
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts.to_numpy(dtype=float) / total
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def effective_count(values: pd.Series) -> float:
    return float(2 ** entropy(values))


def rare_share(values: pd.Series, threshold: float = 0.05) -> float:
    counts = values.astype(str).value_counts()
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    rare = counts[counts / total <= threshold].sum()
    return float(rare / total)


def response_effective_count(ratios: pd.Series) -> float:
    weights = pd.to_numeric(ratios, errors="coerce").fillna(1.0).clip(lower=1.0) - 1.0
    weights = weights.to_numpy(dtype=float)
    if weights.sum() <= 0:
        return 0.0
    probs = weights / weights.sum()
    probs = probs[probs > 0]
    return float(2 ** (-(probs * np.log2(probs)).sum()))


def response_vendi_proxy(ratios: pd.Series) -> float:
    values = pd.to_numeric(ratios, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) <= 1:
        return float(len(values))
    # A tiny 1D effective-rank proxy for response diversity. This is not a
    # Vendi Score claim; it is a bounded preflight proxy for ranking subsets.
    dist = np.abs(values[:, None] - values[None, :])
    sigma = max(float(np.median(dist[dist > 0])) if np.any(dist > 0) else 0.0, 1e-6)
    kernel = np.exp(-(dist**2) / (2 * sigma**2))
    kernel = kernel / max(np.trace(kernel), 1e-12)
    eig = np.linalg.eigvalsh(kernel)
    eig = eig[eig > 1e-12]
    return float(2 ** (-(eig * np.log2(eig)).sum()))


def add_response_metrics(row_df: pd.DataFrame, ot_df: pd.DataFrame) -> pd.DataFrame:
    ot = ot_df.copy()
    if "effect_ratio_vs_max_null_p95" not in ot.columns:
        max_null = ot[["cc_null_p95", "label_null_p95"]].max(axis=1)
        ot["effect_ratio_vs_max_null_p95"] = ot["observed_assignment_ot"] / max_null.replace(0, np.nan)
    ot = ot[
        [
            "row_id",
            "observed_assignment_ot",
            "cc_null_p95",
            "label_null_p95",
            "p_observed_le_cc_null",
            "p_observed_le_label_null",
            "row_expression_ot_gate",
            "subtype_jsd",
            "effect_ratio_vs_max_null_p95",
        ]
    ].rename(
        columns={
            "subtype_jsd": "exploratory_subtype_jsd",
            "row_expression_ot_gate": "exploratory_row_gate",
        }
    )
    return row_df.merge(ot, on="row_id", how="left")


def build_row_metrics(manifest: pd.DataFrame, ot_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    lineage_time_counts = manifest.groupby("manifest_cell_type_broad")["manifest_timepoint"].nunique()
    lineage_target_counts = manifest.groupby("manifest_cell_type_broad")["manifest_gene_target"].nunique()
    for row_id, group in manifest.groupby("row_id", sort=True):
        first = group.iloc[0].to_dict()
        perturb = group[group["selection_role"] == "perturb"].drop_duplicates("cell")
        control = group[group["selection_role"] == "control"].drop_duplicates("cell")
        pooled = group.drop_duplicates("cell")
        lineage = str(first.get("manifest_cell_type_broad", ""))
        row = {
            "row_id": row_id,
            "audit_role": first.get("audit_role", ""),
            "lineage": lineage,
            "target": first.get("manifest_gene_target", ""),
            "timepoint": first.get("manifest_timepoint", ""),
            "n_perturb_cells": int(len(perturb)),
            "n_control_cells": int(len(control)),
            "n_unique_cells": int(pooled["cell"].nunique()),
            "n_perturb_embryos": int(perturb["embryo"].astype(str).nunique()),
            "n_control_embryos": int(control["embryo"].astype(str).nunique()),
            "perturb_subtype_effective_count": effective_count(perturb["cell_type_sub"]),
            "control_subtype_effective_count": effective_count(control["cell_type_sub"]),
            "pooled_subtype_effective_count": effective_count(pooled["cell_type_sub"]),
            "pooled_rare_subtype_share": rare_share(pooled["cell_type_sub"]),
            "lineage_time_coverage_count": int(lineage_time_counts.get(lineage, 0)),
            "lineage_target_coverage_count": int(lineage_target_counts.get(lineage, 0)),
        }
        rows.append(row)
    return add_response_metrics(pd.DataFrame(rows), ot_df)


def subset_metrics(name: str, rows: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, Any]:
    row_ids = set(rows["row_id"].astype(str))
    sub_manifest = manifest[manifest["row_id"].astype(str).isin(row_ids)].drop_duplicates("cell")
    ratios = pd.to_numeric(rows["effect_ratio_vs_max_null_p95"], errors="coerce")
    gates = rows["exploratory_row_gate"].fillna(False).astype(bool) if "exploratory_row_gate" in rows else pd.Series([])
    subtype_jsd = pd.to_numeric(rows["exploratory_subtype_jsd"], errors="coerce")
    n_units = int(rows[["lineage", "timepoint", "target"]].drop_duplicates().shape[0])
    eff_lineages = effective_count(rows["lineage"])
    eff_targets = effective_count(rows["target"])
    eff_timepoints = effective_count(rows["timepoint"])
    pooled_subtype_eff = effective_count(sub_manifest["cell_type_sub"]) if len(sub_manifest) else 0.0
    rare = rare_share(sub_manifest["cell_type_sub"]) if len(sub_manifest) else 0.0
    response_eff = response_effective_count(ratios)
    vendi_proxy = response_vendi_proxy(ratios)
    clean_signal_count = int(((gates) & (subtype_jsd <= 0.10)).sum()) if len(rows) else 0
    high_effect_count = int((ratios >= 1.50).sum())
    # Preflight ranking index. It intentionally mixes coverage and response
    # diversity, and is not used as a biological proof or model-selection gate.
    index = (
        n_units
        + eff_lineages
        + eff_targets
        + eff_timepoints
        + math.log2(1.0 + pooled_subtype_eff)
        + response_eff
        + clean_signal_count
    )
    return {
        "subset": name,
        "n_rows": int(len(rows)),
        "n_lineage_time_target_units": n_units,
        "effective_lineages": eff_lineages,
        "effective_targets": eff_targets,
        "effective_timepoints": eff_timepoints,
        "pooled_subtype_effective_count": pooled_subtype_eff,
        "pooled_rare_subtype_share": rare,
        "mean_response_ratio": float(ratios.mean()) if ratios.notna().any() else float("nan"),
        "max_response_ratio": float(ratios.max()) if ratios.notna().any() else float("nan"),
        "response_effective_count": response_eff,
        "response_vendi_proxy_1d": vendi_proxy,
        "exploratory_gate_fraction": float(gates.mean()) if len(gates) else float("nan"),
        "clean_signal_count_jsd_le_0p10": clean_signal_count,
        "high_effect_count_ratio_ge_1p50": high_effect_count,
        "conditional_bioinformation_index": float(index),
        "note": "preflight ranking only; not a scaling-law proof",
    }


def build_subsets(row_metrics: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    subsets: list[tuple[str, pd.DataFrame]] = []
    subsets.append(("all_selected_rows", row_metrics))
    primary = row_metrics[row_metrics["audit_role"] == "primary_mechanism_test"]
    subsets.append(("primary_mechanism_test", primary))
    subsets.append(("primary_mature_fast_muscle", primary[primary["lineage"] == "mature fast muscle"]))
    subsets.append(("primary_periderm", primary[primary["lineage"] == "periderm"]))
    clean_primary = primary[pd.to_numeric(primary["exploratory_subtype_jsd"], errors="coerce") <= 0.10]
    subsets.append(("clean_primary_jsd_le_0p10", clean_primary))
    high_effect_primary = primary[pd.to_numeric(primary["effect_ratio_vs_max_null_p95"], errors="coerce") >= 1.50]
    subsets.append(("high_effect_primary_ratio_ge_1p50", high_effect_primary))
    for role, group in row_metrics.groupby("audit_role", sort=True):
        subsets.append((f"audit_role__{role}", group))
    for lineage, group in row_metrics.groupby("lineage", sort=True):
        subsets.append((f"lineage__{lineage}", group))
    subset_rows = [subset_metrics(name, group, manifest) for name, group in subsets if len(group) > 0]
    return pd.DataFrame(subset_rows).sort_values("conditional_bioinformation_index", ascending=False)


def write_report(out_dir: Path, row_metrics: pd.DataFrame, subset_df: pd.DataFrame, args: argparse.Namespace) -> Path:
    md_path = out_dir / "LATENTFM_ZSCAPE_BIOLOGICAL_INFORMATION_AXIS_20260628.md"
    top = subset_df.head(12)
    primary = row_metrics[row_metrics["audit_role"] == "primary_mechanism_test"].copy()
    status = "zscape_biological_information_axis_ready_waiting_strict_controls_no_gpu"
    lines = [
        "# LatentFM ZSCAPE Biological Information Axis Preflight",
        "",
        f"Timestamp: `{utc_now()}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only preflight over the ZSCAPE selected-cell manifest and",
        "  completed exploratory expression OT output.",
        "- Does not train, infer, run scFM embeddings, read canonical multi, or read",
        "  Track C query.",
        "- The index below is a ranking/design aid for future scaling experiments,",
        "  not a scaling-law proof and not a model-selection criterion.",
        "",
        "## Why This Exists",
        "",
        "The working biological-scaling hypothesis is that useful downstream",
        "perturbation information is not proportional to raw cell count. The next",
        "x-axis should be conditional biological information: lineage/time/target",
        "coverage, subtype/state diversity, trajectory-time coverage, and response",
        "OT diversity after controls.",
        "",
        "## Inputs",
        "",
        f"- selected manifest: `{args.matched_manifest}`",
        f"- exploratory OT rows: `{args.expression_ot_rows}`",
        "",
        "## Top Subset Preflight Metrics",
        "",
        "| subset | rows | units | eff lineage | eff target | eff time | subtype eff | mean ratio | clean signals | high-effect rows | index |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in top.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["subset"]),
                    str(int(row["n_rows"])),
                    str(int(row["n_lineage_time_target_units"])),
                    f"{row['effective_lineages']:.2f}",
                    f"{row['effective_targets']:.2f}",
                    f"{row['effective_timepoints']:.2f}",
                    f"{row['pooled_subtype_effective_count']:.2f}",
                    f"{row['mean_response_ratio']:.3f}",
                    str(int(row["clean_signal_count_jsd_le_0p10"])),
                    str(int(row["high_effect_count_ratio_ge_1p50"])),
                    f"{row['conditional_bioinformation_index']:.2f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Primary Row Design Implications",
            "",
            "| lineage | target | time | ratio | subtype JSD | clean? | high-effect? |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in primary.sort_values(["lineage", "target", "timepoint"]).iterrows():
        ratio = float(row.get("effect_ratio_vs_max_null_p95", float("nan")))
        jsd = float(row.get("exploratory_subtype_jsd", float("nan")))
        lines.append(
            f"| {row['lineage']} | {row['target']} | {row['timepoint']} | "
            f"{ratio:.3f} | {jsd:.3f} | {jsd <= 0.10} | {ratio >= 1.50} |"
        )
    lines.extend(
        [
            "",
            "## Minimal Next Experiment Design",
            "",
            "1. Wait for the formal strict-controls gate.",
            "2. If it passes or gives lineage-specific partial support, build fixed-cell",
            "   high-information versus low-information subsets using the metrics here.",
            "3. Compare strict OT robustness first; only then consider a bounded",
            "   leakage-safe latent/trajectory or LatentFM GPU smoke.",
            "4. If strict controls fail, keep this report as a design map but do not use",
            "   it to authorize GPU.",
            "",
            "## Output Files",
            "",
            f"- row metrics: `{out_dir / 'zscape_bioinformation_row_metrics.csv'}`",
            f"- subset metrics: `{out_dir / 'zscape_bioinformation_subset_metrics.csv'}`",
            f"- JSON: `{out_dir / 'zscape_bioinformation_axis_20260628.json'}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-manifest", type=Path, required=True)
    parser.add_argument("--expression-ot-rows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.matched_manifest)
    ot_df = pd.read_csv(args.expression_ot_rows)
    row_metrics = build_row_metrics(manifest, ot_df)
    subset_df = build_subsets(row_metrics, manifest)

    row_csv = args.out_dir / "zscape_bioinformation_row_metrics.csv"
    subset_csv = args.out_dir / "zscape_bioinformation_subset_metrics.csv"
    row_metrics.to_csv(row_csv, index=False)
    subset_df.to_csv(subset_csv, index=False)

    summary = {
        "timestamp_utc": utc_now(),
        "status": "zscape_biological_information_axis_ready_waiting_strict_controls_no_gpu",
        "gpu_authorized": False,
        "matched_manifest": str(args.matched_manifest),
        "expression_ot_rows": str(args.expression_ot_rows),
        "n_rows": int(len(row_metrics)),
        "top_subsets": subset_df.head(10).to_dict(orient="records"),
    }
    json_path = args.out_dir / "zscape_bioinformation_axis_20260628.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    md_path = write_report(args.out_dir, row_metrics, subset_df, args)
    print(md_path)
    print(row_csv)
    print(subset_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
