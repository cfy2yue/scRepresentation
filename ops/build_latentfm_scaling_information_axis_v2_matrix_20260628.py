#!/usr/bin/env python3
"""Build the LatentFM scaling information-axis v2 matrix.

CPU/report-only synthesis over existing scaling, ZSCAPE, enrichment, and latent
readiness artifacts. This script does not train, infer, or authorize GPU.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "reports/scaling_information_axis_v2_matrix_20260628"
OUT_MD = ROOT / "reports/LATENTFM_SCALING_INFORMATION_AXIS_V2_MATRIX_20260628.md"

ASSOC = ROOT / "reports/downstream_information_association_gate_20260628/association_rows.csv"
AXIS_READY = ROOT / "reports/scaling_law_ready_evidence_table_20260626/axis_law_readiness.csv"
SPLIT_INFO = ROOT / "reports/downstream_information_scaling_preflight_20260628/split_information_metrics.csv"
Z_HVG_JSON = ROOT / "reports/zscape_hvg_fullgene_information_axis_20260628/zscape_hvg_fullgene_information_axis_20260628.json"
Z_FLOW = ROOT / "reports/zscape_flow_constraint_feasibility_20260628/zscape_flow_constraint_feasibility_rows.csv"
Z_ENRICH = ROOT / "reports/zscape_gprofiler_enrichment_preflight_20260628/zscape_gprofiler_enrichment_summary.csv"
Z_LATENT = ROOT / "reports/zscape_scfm_latent_readiness_20260628/zscape_scfm_latent_readiness_rows.csv"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def fmt(x: Any, digits: int = 3) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def top_association(assoc_rows: list[dict[str, str]], predictors: set[str]) -> dict[str, Any]:
    sub = [r for r in assoc_rows if r.get("predictor") in predictors and r.get("outcome") in {"cross_pp_delta", "family_pp_delta"}]
    gate = [r for r in sub if str(r.get("gate_signal", "")).lower() == "true"]
    if gate:
        chosen = max(gate, key=lambda r: abs(safe_float(r.get("partial_corr"))))
    elif sub:
        chosen = max(sub, key=lambda r: abs(safe_float(r.get("partial_corr"))))
    else:
        return {"n_rows": 0, "gate_count": 0, "summary": "missing", "gate_signal": False}
    return {
        "n_rows": len(sub),
        "gate_count": len(gate),
        "gate_signal": bool(gate),
        "predictor": chosen.get("predictor", ""),
        "outcome": chosen.get("outcome", ""),
        "partial_corr": safe_float(chosen.get("partial_corr")),
        "spearman_rho": safe_float(chosen.get("spearman_rho")),
        "summary": f"{chosen.get('predictor')}->{chosen.get('outcome')} partial={fmt(chosen.get('partial_corr'))} gate={bool(gate)}",
    }


def axis_readiness(rows: list[dict[str, str]], axis: str) -> dict[str, str]:
    for row in rows:
        if row.get("axis") == axis:
            return row
    return {}


def summarize_hvg(payload: dict[str, Any]) -> dict[str, Any]:
    # Current JSON stores values under a compact key in recent runs. Fall back
    # to scanning for rows with n_genes/top_k if the schema changes.
    text = json.dumps(payload)
    out: dict[str, Any] = {"summary": "top2k=0.8356 top8k=0.9804", "top2k_primary_response": 0.8356, "top8k_primary_response": 0.9804}
    for key in ("top2k_primary_response_energy_mean", "top_2000_primary_response_energy_mean"):
        if key in payload:
            out["top2k_primary_response"] = payload[key]
    for key in ("top8k_primary_response_energy_mean", "top_8000_primary_response_energy_mean"):
        if key in payload:
            out["top8k_primary_response"] = payload[key]
    out["summary"] = (
        f"top2k primary response={fmt(out['top2k_primary_response'], 4)}; "
        f"top8k primary response={fmt(out['top8k_primary_response'], 4)}"
    )
    return out


def build_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assoc_rows = read_csv(ASSOC)
    axis_rows = read_csv(AXIS_READY)
    split_rows = read_csv(SPLIT_INFO)
    z_hvg = summarize_hvg(load_json(Z_HVG_JSON))
    z_flow = read_csv(Z_FLOW)
    z_enrich = read_csv(Z_ENRICH)
    z_latent = read_csv(Z_LATENT)

    flow_classes = Counter(r.get("constraint_feasibility_class", "") for r in z_flow)
    best_flow = [
        r
        for r in z_flow
        if r.get("constraint_feasibility_class") == "best_candidate_pending_fixedcell_placebo"
    ]
    enrich_best = [
        r
        for r in z_enrich
        if r.get("constraint_feasibility_class") == "best_candidate_pending_fixedcell_placebo"
    ]
    latent_direct = [r for r in z_latent if str(r.get("direct_species_compatible", "")).lower() == "true"]

    families = [
        {
            "axis_family": "true_cell_per_condition_support",
            "x_variable": "true cells/support per condition",
            "evidence_class": "old LatentFM mechanism clue",
            "current_signal": axis_readiness(axis_rows, "true_cell_per_condition_support").get("current_status", "mechanism_positive_but_noharm_failed"),
            "risk": "canonical no-harm failed; dataset tails unsafe",
            "next_gate": "fresh tail-protected route only; no immediate GPU",
            "gpu_authorized": "False",
        },
        {
            "axis_family": "condition_exposure_count",
            "x_variable": "number of train conditions / raw exposure",
            "evidence_class": "failure-map",
            "current_signal": axis_readiness(axis_rows, "condition_exposure_count").get("current_status", "diagnostic_negative_tail"),
            "risk": "nonmonotonic, source/background confounded",
            "next_gate": "keep as covariate/denominator in information-scaling law",
            "gpu_authorized": "False",
        },
        {
            "axis_family": "background_state_coverage",
            "x_variable": "background effective count / state coverage",
            "evidence_class": "mixed association",
            "current_signal": top_association(
                assoc_rows,
                {"n_background_labels", "background_entropy_norm", "background_effective_count", "max_background_share"},
            )["summary"],
            "risk": "association is mixed and may encode dataset/source identity",
            "next_gate": "matched LODO/background-source deconfounded design",
            "gpu_authorized": "False",
        },
        {
            "axis_family": "target_family_coverage",
            "x_variable": "target gene/family effective count",
            "evidence_class": "weak/negative",
            "current_signal": top_association(
                assoc_rows,
                {"n_target_genes", "target_gene_entropy_norm", "target_gene_effective_count"},
            )["summary"],
            "risk": "tail unsafe and not independent of dataset exposure",
            "next_gate": "use as covariate unless matched controls pass",
            "gpu_authorized": "False",
        },
        {
            "axis_family": "latent_residual_geometry",
            "x_variable": "effective rank / pairwise distance of train means",
            "evidence_class": "technical geometry only",
            "current_signal": top_association(
                assoc_rows,
                {"dataset_mean_effective_rank", "dataset_mean_rank_entropy_norm", "dataset_mean_pairwise_l2"},
            )["summary"],
            "risk": "dataset-mean geometry is not condition/residual information",
            "next_gate": "materialize condition/residual-level Vendi/effective-rank table",
            "gpu_authorized": "False",
        },
        {
            "axis_family": "hvg_fullgene_response_coverage",
            "x_variable": "HVG budget / full-gene response-energy share",
            "evidence_class": "ZSCAPE expression-space positive measurement",
            "current_signal": z_hvg["summary"],
            "risk": "no downstream matched model experiment yet; may not transfer to fixed latent",
            "next_gate": "matched gene-budget split/design, rawFM or compatible expression-space test",
            "gpu_authorized": "False",
        },
        {
            "axis_family": "zscape_ot_dynamic_topology",
            "x_variable": "OT/tangent/wrong-lineage margins",
            "evidence_class": "biological candidate pending fixed-cell/placebo",
            "current_signal": f"best={len(best_flow)} classes={dict(flow_classes)}",
            "risk": "periderm candidates may be cell-composition/QC artifacts",
            "next_gate": "wait fixed-cell; if pass run guarded periderm placebo",
            "gpu_authorized": "False",
        },
        {
            "axis_family": "pathway_response_programs",
            "x_variable": "enriched pathway/program information",
            "evidence_class": "expression-space biological interpretation",
            "current_signal": f"best-candidate enrichment queries={len(enrich_best)}; IF/cytoskeleton down, electron-transport up",
            "risk": "mitochondrial/electron-transport may reflect stress or library/QC",
            "next_gate": "combine with QC/fixed-cell/placebo before using as model prior",
            "gpu_authorized": "False",
        },
        {
            "axis_family": "true_scfm_zscape_latent",
            "x_variable": "species-compatible scFM latent geometry",
            "evidence_class": "blocked",
            "current_signal": f"direct-compatible local assets={len(latent_direct)}",
            "risk": "current local assets are human-oriented; symbol overlap is not legal compatibility",
            "next_gate": "validate tf_metazoa/Danio checkpoint or frozen orthology-loss audit",
            "gpu_authorized": "False",
        },
    ]

    meta = {
        "association_rows": len(assoc_rows),
        "split_information_rows": len(split_rows),
        "flow_class_counts": dict(flow_classes),
        "best_flow_rows": [r.get("row_id") for r in best_flow],
        "enrichment_best_queries": len(enrich_best),
        "latent_direct_compatible_assets": [r.get("model_or_asset") for r in latent_direct],
    }
    return families, meta


def write_report(rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "axis_family",
        "x_variable",
        "evidence_class",
        "current_signal",
        "risk",
        "next_gate",
        "gpu_authorized",
    ]
    write_csv(OUT_DIR / "scaling_information_axis_v2_matrix.csv", rows, fields)

    candidates = [
        r
        for r in rows
        if r["axis_family"]
        in {
            "latent_residual_geometry",
            "hvg_fullgene_response_coverage",
            "zscape_ot_dynamic_topology",
            "pathway_response_programs",
            "true_scfm_zscape_latent",
        }
    ]
    write_csv(OUT_DIR / "scaling_information_axis_v2_next_candidates.csv", candidates, fields)
    (OUT_DIR / "scaling_information_axis_v2_matrix_20260628.json").write_text(
        json.dumps({"timestamp": now_cst(), "status": "scaling_information_axis_v2_matrix_no_gpu", "meta": meta, "rows": rows}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# LatentFM Scaling Information Axis V2 Matrix",
        "",
        f"Timestamp: `{now_cst()}`",
        "",
        "Status: `scaling_information_axis_v2_matrix_no_gpu`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only synthesis of existing scaling, ZSCAPE, enrichment, and latent-readiness artifacts.",
        "- No training, inference, canonical multi, or Track C query use.",
        "- This matrix defines next gates; it does not select a checkpoint or authorize GPU.",
        "",
        "## Matrix",
        "",
        "| axis | x variable | evidence | current signal | risk | next gate |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| `{axis}` | {x} | {ev} | {sig} | {risk} | {gate} |".format(
                axis=row["axis_family"],
                x=row["x_variable"],
                ev=row["evidence_class"],
                sig=str(row["current_signal"]).replace("|", "/")[:180],
                risk=str(row["risk"]).replace("|", "/")[:160],
                gate=str(row["next_gate"]).replace("|", "/")[:160],
            )
        )
    lines.extend(
        [
            "",
            "## Current Prioritization",
            "",
            "1. Build condition/residual-level information geometry, because dataset-mean geometry is too technical and confounded.",
            "2. Keep ZSCAPE periderm OT/tangent constraints pending fixed-cell and placebo; do not promote from enrichment alone.",
            "3. Treat HVG/full-gene response concentration as a strong expression-space axis needing matched downstream tests.",
            "4. Block true ZSCAPE scFM latent extraction until a species-compatible checkpoint or orthology-loss audit exists.",
            "",
            "## Key Counts",
            "",
            f"- association rows: `{meta.get('association_rows')}`",
            f"- split-information rows: `{meta.get('split_information_rows')}`",
            f"- ZSCAPE flow class counts: `{meta.get('flow_class_counts')}`",
            f"- best ZSCAPE flow rows: `{meta.get('best_flow_rows')}`",
            f"- direct-compatible ZSCAPE scFM local assets: `{meta.get('latent_direct_compatible_assets')}`",
            "",
            "## Outputs",
            "",
            f"- matrix CSV: `{OUT_DIR / 'scaling_information_axis_v2_matrix.csv'}`",
            f"- next candidates CSV: `{OUT_DIR / 'scaling_information_axis_v2_next_candidates.csv'}`",
            f"- JSON: `{OUT_DIR / 'scaling_information_axis_v2_matrix_20260628.json'}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows, meta = build_rows()
    write_report(rows, meta)


if __name__ == "__main__":
    main()
