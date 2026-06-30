#!/usr/bin/env python3
"""Gate whether ZSCAPE periderm variables can become LatentFM train-only features."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_MODULE_JSON = ROOT / "reports/zscape_expression_module_scores_20260628/zscape_expression_module_scores_20260628.json"
DEFAULT_DESIGN_REVIEW = ROOT / "reports/LATENTFM_ZSCAPE_PERIDERM_EXPRESSION_DESIGN_REVIEW_20260628.md"
DEFAULT_HVG_GATE_JSON = ROOT / "reports/latentfm_hvg_fullgene_downstream_design_gate_20260628.json"
DEFAULT_TRAINONLY_ROWS = ROOT / "reports/trainonly_condition_residual_information_20260628/trainonly_condition_residual_information_rows.csv"
DEFAULT_ASSOC_ROWS = ROOT / "reports/downstream_condition_residual_association_gate_20260628/condition_residual_information_outcome_join_rows.csv"
DEFAULT_COND_SLATE = ROOT / "reports/LATENTFM_CONDITION_RESIDUAL_SCALING_SLATE_DECISION_20260628.md"
DEFAULT_OUT = ROOT / "reports/zscape_to_latentfm_source_trainonly_gate_20260628"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bool_all_zero(rows: list[dict[str, Any]], key: str) -> bool:
    vals = []
    for row in rows:
        try:
            vals.append(float(row.get(key, 0)))
        except (TypeError, ValueError):
            vals.append(0.0)
    return bool(vals) and all(v == 0 for v in vals)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-json", type=Path, default=DEFAULT_MODULE_JSON)
    parser.add_argument("--design-review", type=Path, default=DEFAULT_DESIGN_REVIEW)
    parser.add_argument("--hvg-gate-json", type=Path, default=DEFAULT_HVG_GATE_JSON)
    parser.add_argument("--trainonly-rows", type=Path, default=DEFAULT_TRAINONLY_ROWS)
    parser.add_argument("--association-rows", type=Path, default=DEFAULT_ASSOC_ROWS)
    parser.add_argument("--condition-slate-report", type=Path, default=DEFAULT_COND_SLATE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    module = load_json(args.module_json)
    hvg = load_json(args.hvg_gate_json)
    train = pd.read_csv(args.trainonly_rows)
    assoc = pd.read_csv(args.association_rows)

    latent_summary = hvg.get("latent_summary", [])
    latent_embedding_only = bool_all_zero(latent_summary, "files_with_gene_matrix") and bool_all_zero(
        latent_summary, "files_with_var_names"
    )
    h5ad_summary = hvg.get("h5ad_summary", [])
    max_h5ad_vars = max(float(row.get("n_vars_max", 0) or 0) for row in h5ad_summary) if h5ad_summary else 0.0
    train_vectors_available = int(train.get("n_missing_vectors", pd.Series([1])).fillna(1).sum()) == 0
    n_train_splits = int(len(train))
    n_assoc_rows = int(len(assoc))
    robust_assoc = assoc[
        (assoc.get("status", pd.Series(dtype=str)).astype(str) == "done")
        & (
            assoc.get("cross_pp_delta", pd.Series(dtype=float)).fillna(0).abs()
            + assoc.get("family_pp_delta", pd.Series(dtype=float)).fillna(0).abs()
            > 0.01
        )
    ]

    rows = [
        {
            "candidate": "zscape_periderm_module_scores",
            "zscape_supported": True,
            "trainonly_materializable_now": False,
            "allowed_use": "scaling_interpretation_or_future_raw_expression_gate",
            "reason": "ZSCAPE modules are Danio expression-space variables; current LatentFM latent H5 files are embedding-only and expose no gene matrices/var names.",
        },
        {
            "candidate": "zscape_hvg_response_coverage",
            "zscape_supported": True,
            "trainonly_materializable_now": False,
            "allowed_use": "future_raw_expression_hvg_budget_gate",
            "reason": "ZSCAPE has a strong HVG response curve, but current xVERSE training path cannot change gene budget inside frozen embeddings.",
        },
        {
            "candidate": "trainonly_condition_residual_information",
            "zscape_supported": "indirect",
            "trainonly_materializable_now": bool(train_vectors_available),
            "allowed_use": "scaling_law_axis_and_covariate",
            "reason": "Condition residual vectors are train-only and complete, but the first GPU slate using related axes failed and these are not periderm module features.",
        },
        {
            "candidate": "dataset_background_effective_count",
            "zscape_supported": False,
            "trainonly_materializable_now": bool(train_vectors_available),
            "allowed_use": "confound_covariate_or_control",
            "reason": "Useful for scaling-law controls, not a biological periderm variable.",
        },
        {
            "candidate": "perturbation_type_or_target_breadth",
            "zscape_supported": "generic",
            "trainonly_materializable_now": bool(train_vectors_available),
            "allowed_use": "control_or_hypothesis_generator_only",
            "reason": "Perturbation-type breadth is measurable but the type-balanced GPU smoke regressed; do not promote without a new non-duplicate CPU gate.",
        },
    ]

    direct_model_route = False
    reasons = []
    if latent_embedding_only:
        reasons.append("current_latent_h5_embedding_only_no_gene_modules")
    if max_h5ad_vars < 10000:
        reasons.append("available_raw_expression_panels_gene_limited_not_true_fullgene")
    if not train_vectors_available:
        reasons.append("train_condition_residual_vectors_missing")
    if module.get("periderm_best_module_gate_pass") == module.get("periderm_best_module_gate_total"):
        reasons.append("zscape_module_signal_supported_but_species_expression_space_only")
    if len(robust_assoc) > 0:
        reasons.append("trainonly_scaling_axes_exist_but_prior_matched_gpu_slate_failed_direct_use")

    status = "zscape_to_latentfm_source_gate_no_direct_model_route_no_gpu"
    out_rows = args.out_dir / "zscape_to_latentfm_source_trainonly_gate_rows.csv"
    out_json = args.out_dir / "zscape_to_latentfm_source_trainonly_gate_20260628.json"
    out_md = args.out_dir / "LATENTFM_ZSCAPE_TO_LATENTFM_SOURCE_TRAINONLY_GATE_20260628.md"
    pd.DataFrame(rows).to_csv(out_rows, index=False)
    result = {
        "timestamp_cst": now_cst(),
        "status": status,
        "gpu_authorized": False,
        "direct_model_route_authorized": direct_model_route,
        "reasons": reasons,
        "latent_embedding_only": latent_embedding_only,
        "max_available_h5ad_vars": max_h5ad_vars,
        "n_trainonly_splits": n_train_splits,
        "train_vectors_available": train_vectors_available,
        "n_association_rows": n_assoc_rows,
        "n_nontrivial_done_association_rows": int(len(robust_assoc)),
        "rows": str(out_rows),
    }
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM ZSCAPE To Train-Only Source Gate",
        "",
        f"Timestamp: `{now_cst()}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only source/train-only design gate.",
        "- No model training, no inference, no scFM embedding extraction, no canonical multi, and no Track C query.",
        "",
        "## Summary",
        "",
        f"- ZSCAPE module signal supported: `{module.get('periderm_best_module_gate_pass')}/{module.get('periderm_best_module_gate_total')}` best-candidate periderm modules.",
        f"- current LatentFM latent H5 files embedding-only: `{latent_embedding_only}`.",
        f"- maximum local raw-expression panel genes: `{max_h5ad_vars:g}`.",
        f"- train-only condition residual splits available: `{n_train_splits}`; missing vectors: `{not train_vectors_available}`.",
        f"- nontrivial completed association rows: `{len(robust_assoc)}/{n_assoc_rows}`.",
        "",
        "## Decision",
        "",
        "Do not launch a ZSCAPE-derived LatentFM model route. The current training artifacts cannot materialize Danio periderm modules as train-only features, and the available train-only condition-residual axes are generic scaling variables rather than periderm biology.",
        "",
        "Use ZSCAPE as:",
        "",
        "- expression-space biological insight;",
        "- an information-scaling design axis;",
        "- a motivation for future raw-expression/HVG-budget gates;",
        "- a negative guard against overclaiming latent/flow constraints.",
        "",
        "## Candidate Translation Table",
        "",
        "| candidate | ZSCAPE supported | train-only materializable now | allowed use | reason |",
        "|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['zscape_supported']} | {row['trainonly_materializable_now']} | {row['allowed_use']} | {row['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Fail-Close",
            "",
            "- No ZSCAPE-derived GPU smoke without a new raw-expression or source-control CPU gate.",
            "- No latent/flow tangent route until true Danio-compatible latent embeddings or a reconciled proxy definition exists.",
            "- No training-set weighting from ZSCAPE modules while current LatentFM artifacts remain embedding-only.",
            "",
            "## Outputs",
            "",
            f"- rows: `{out_rows}`",
            f"- JSON: `{out_json}`",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_md)
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
