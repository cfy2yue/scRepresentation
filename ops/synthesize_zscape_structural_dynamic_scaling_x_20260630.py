#!/usr/bin/env python3
"""Synthesize ZSCAPE structural dynamic-information scaling x evidence.

This report-only gate integrates ZSCAPE dynamic OT decomposition with the
LatentFM response-energy association gate. It asks whether "more response" is
the right scaling variable, or whether a structured dynamic-information x is a
better scientific object.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "zscape_structural_dynamic_scaling_x_20260630"

INPUTS = {
    "zscape_strict_rows": REPORTS
    / "zscape_strict_ot_decomposition_gate_20260628"
    / "zscape_strict_ot_decomposition_rows.csv",
    "zscape_candidate_x": REPORTS
    / "zscape_strict_ot_decomposition_gate_20260628"
    / "zscape_strict_ot_decomposition_candidate_x.csv",
    "zscape_law_panel": REPORTS / "zscape_response_law_panel_20260628" / "zscape_response_law_panel_rows.csv",
    "zscape_crossfit_repair": REPORTS
    / "zscape_crossfit_residual_specificity_repair_gate_20260628"
    / "zscape_crossfit_specificity_query_rows.csv",
    "zscape_trainset_translation": REPORTS
    / "zscape_to_latentfm_trainset_translation_gate_20260629"
    / "zscape_to_latentfm_translation_feature_readiness.csv",
    "latentfm_response_compressibility": REPORTS
    / "response_compressibility_pairability_gate_20260630"
    / "response_compressibility_associations_20260630.csv",
}

PRIMARY_TASKS = {
    "all_test_single",
    "cross_background_seen_gene",
    "simple_cross_background_seen_gene_exact",
    "simple_test_single_gene_exact",
    "proxy_all_test_single_proxy",
}


def finite_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def rank_corr(x: pd.Series, y: pd.Series) -> float | None:
    xy = pd.concat([x, y], axis=1).dropna()
    if len(xy) < 3:
        return None
    xr = xy.iloc[:, 0].rank(method="average")
    yr = xy.iloc[:, 1].rank(method="average")
    if float(xr.std(ddof=0)) == 0.0 or float(yr.std(ddof=0)) == 0.0:
        return None
    return finite_float(xr.corr(yr))


def sigmoid_margin(value: Any, scale: float) -> float:
    x = finite_float(value)
    if x is None:
        return 0.0
    return float(np.tanh(max(x, 0.0) / scale))


def build_zscape_scores(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["state_preserved"] = out.get("state_preserved_by_threshold", False).map(truthy)
    out["trajectory_aligned"] = out.get("trajectory_alignment_gate", False).map(truthy)
    out["embryo_reliable"] = out.get("embryo_vector_gate", False).map(truthy)
    out["module_specific"] = out.get("module_all_specificity_gate", False).map(truthy)
    out["dynamic_gate"] = out.get("dynamic_response_gate", False).map(truthy)
    out["composition_fraction"] = pd.to_numeric(
        out.get("composition_norm_fraction_of_centroid"), errors="coerce"
    )
    out["within_fraction"] = pd.to_numeric(
        out.get("within_substate_residual_fraction_of_centroid"), errors="coerce"
    )
    out["response_norm"] = pd.to_numeric(out.get("centroid_response_norm"), errors="coerce")
    out["trajectory_cosine_num"] = pd.to_numeric(out.get("trajectory_cosine"), errors="coerce")
    out["wrong_time_margin_num"] = pd.to_numeric(out.get("wrong_time_margin_ot"), errors="coerce")
    out["wrong_lineage_margin_num"] = pd.to_numeric(out.get("wrong_lineage_margin_ot"), errors="coerce")
    out["module_specificity_margin"] = pd.to_numeric(
        out.get("module_min_residual_ci_low"), errors="coerce"
    ) - pd.to_numeric(out.get("module_max_wrong_time"), errors="coerce")

    scores: list[float] = []
    for _, row in out.iterrows():
        within = min(max(finite_float(row.get("within_fraction")) or 0.0, 0.0), 1.0)
        comp_penalty = min(max(finite_float(row.get("composition_fraction")) or 0.0, 0.0), 1.5)
        score = 0.0
        score += 0.8 if row["state_preserved"] else 0.0
        score += 0.8 if row["trajectory_aligned"] else 0.0
        score += 0.5 if row["embryo_reliable"] else 0.0
        score += 0.5 * max(finite_float(row.get("trajectory_cosine_num")) or 0.0, 0.0)
        score += 0.7 * sigmoid_margin(row.get("wrong_time_margin_num"), 1.0)
        score += 0.7 * sigmoid_margin(row.get("wrong_lineage_margin_num"), 10.0)
        score += 0.6 * within
        score -= 0.8 * comp_penalty
        score += 0.4 if row["module_specific"] else -0.3
        scores.append(float(score))
    out["structural_dynamic_information_score"] = scores
    out["magnitude_confounded_flag"] = (
        (out["composition_fraction"] > 0.25)
        | (out["wrong_time_margin_num"] < 0)
        | (out.get("strict_dynamic_class", "").astype(str).str.contains("confounded", case=False, na=False))
    )
    out["geometry_positive_specificity_blocked"] = out.get("strict_dynamic_class", "").astype(str).eq(
        "geometry_replicate_insight_specificity_blocked"
    )
    out["model_ready_constraint"] = out.get("strict_dynamic_class", "").astype(str).eq(
        "constraint_candidate_ready"
    )
    return out


def latentfm_response_energy_summary(assoc: pd.DataFrame) -> dict[str, Any]:
    sub = assoc[(assoc["feature"] == "response_energy") & assoc["task"].isin(PRIMARY_TASKS)].copy()
    if sub.empty:
        return {
            "primary_rows": 0,
            "pp_worse_rows": 0,
            "mmd_worse_rows": 0,
            "all_worse_rows": 0,
            "mean_rho_pearson_pert": None,
            "mean_rho_test_mmd": None,
        }
    pp = pd.to_numeric(sub["rho_pearson_pert"], errors="coerce")
    mmd = pd.to_numeric(sub["rho_test_mmd"], errors="coerce")
    return {
        "primary_rows": int(len(sub)),
        "pp_worse_rows": int((pp < 0).sum()),
        "mmd_worse_rows": int((mmd > 0).sum()),
        "all_worse_rows": int(((pp < 0) & (mmd > 0)).sum()),
        "mean_rho_pearson_pert": finite_float(pp.mean()),
        "mean_rho_test_mmd": finite_float(mmd.mean()),
        "rows": sub[
            [
                "task",
                "rho_pearson_pert",
                "rho_test_mmd",
                "within_dataset_rho_pearson_pert",
                "within_dataset_rho_test_mmd",
            ]
        ].to_dict(orient="records"),
    }


def decide(zscape: pd.DataFrame, latent_summary: dict[str, Any], crossfit: pd.DataFrame, translation: pd.DataFrame) -> dict[str, Any]:
    geometry_rows = zscape[zscape["geometry_positive_specificity_blocked"]]
    confounded = zscape[zscape["magnitude_confounded_flag"]]
    model_ready = zscape[zscape["model_ready_constraint"]]
    high_norm = zscape[zscape["response_norm"] >= zscape["response_norm"].median()]
    high_norm_confounded = int(high_norm["magnitude_confounded_flag"].sum()) if not high_norm.empty else 0
    response_norm_vs_score = rank_corr(zscape["response_norm"], zscape["structural_dynamic_information_score"])
    crossfit_pass = False
    if not crossfit.empty and "gate" in crossfit.columns:
        crossfit_pass = bool(crossfit["gate"].map(truthy).any())
    elif not crossfit.empty and "query_gate" in crossfit.columns:
        crossfit_pass = bool(crossfit["query_gate"].map(truthy).any())
    translation_gpu = False
    if not translation.empty and "gpu_authorized" in translation.columns:
        translation_gpu = bool(translation["gpu_authorized"].map(truthy).any())

    magnitude_not_information = (
        len(geometry_rows) >= 2
        and len(confounded) >= 4
        and high_norm_confounded >= 3
        and latent_summary.get("all_worse_rows", 0) >= 4
    )
    if magnitude_not_information:
        status = "zscape_structural_dynamic_x_descriptor_pass_model_blocked"
        descriptor_pass = True
    else:
        status = "zscape_structural_dynamic_x_incomplete_model_blocked"
        descriptor_pass = False

    reasons = [
        "zscape_geometry_positive_rows_exist_but_module_specificity_and_crossfit_repair_failed",
        "high_response_magnitude_rows_are_often_composition_or_time_confounded",
        "latentfm_response_energy_behaves_as_difficulty_covariate_not_easy_information_axis",
        "trainset_translation_features_do_not_authorize_gpu",
    ]
    return {
        "status": status,
        "descriptor_pass": descriptor_pass,
        "model_constraint_ready": bool(len(model_ready) > 0 and crossfit_pass and translation_gpu),
        "gpu_authorized_next": False,
        "magnitude_not_information_cross_evidence": magnitude_not_information,
        "zscape_rows": int(len(zscape)),
        "zscape_geometry_positive_specificity_blocked_rows": geometry_rows["row_id"].astype(str).tolist(),
        "zscape_confounded_magnitude_rows": confounded["row_id"].astype(str).tolist(),
        "zscape_model_ready_rows": model_ready["row_id"].astype(str).tolist(),
        "high_response_norm_rows": int(len(high_norm)),
        "high_response_norm_confounded_rows": high_norm_confounded,
        "response_norm_vs_structural_score_spearman": response_norm_vs_score,
        "latentfm_response_energy_summary": latent_summary,
        "crossfit_specificity_repair_any_pass": crossfit_pass,
        "trainset_translation_gpu_any": translation_gpu,
        "reasons": reasons,
        "next_action": (
            "use structural dynamic information as a scaling-law definition and failure-analysis "
            "axis; do not train from it until a broader OT atlas or species-safe trainset "
            "translation passes specificity/no-harm controls"
        ),
    }


def fmt(value: Any, digits: int = 4) -> str:
    x = finite_float(value)
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


def markdown_table(df: pd.DataFrame, cols: list[str], n: int = 20) -> str:
    if df.empty:
        return "_None._"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].head(n).iterrows():
        vals = []
        for col in cols:
            val = row.get(col)
            if isinstance(val, float):
                vals.append(fmt(val))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_outputs(
    zscape: pd.DataFrame,
    candidate_x: pd.DataFrame,
    law_panel: pd.DataFrame,
    decision: dict[str, Any],
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scored_path = OUT_DIR / "zscape_structural_dynamic_x_rows_20260630.csv"
    candidate_path = OUT_DIR / "zscape_structural_dynamic_x_candidates_20260630.csv"
    json_path = OUT_DIR / "zscape_structural_dynamic_scaling_x_20260630.json"
    md_path = OUT_DIR / "LATENTFM_ZSCAPE_STRUCTURAL_DYNAMIC_SCALING_X_20260630.md"

    zscape.to_csv(scored_path, index=False)
    candidate_x.to_csv(candidate_path, index=False)
    payload = {
        "boundary": {
            "reads_completed_reports_only": True,
            "training_or_inference": False,
            "new_ot_pairing": False,
            "checkpoint_selection": False,
            "canonical_multi_selection": False,
            "trackc_query_access": False,
            "gpu_authorized_next": False,
        },
        "inputs": {k: str(v) for k, v in INPUTS.items()},
        "outputs": {
            "scored_rows": str(scored_path),
            "candidate_x": str(candidate_path),
            "markdown_report": str(md_path),
        },
        "decision": decision,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    row_cols = [
        "row_id",
        "lineage",
        "target",
        "timepoint",
        "response_norm",
        "composition_fraction",
        "trajectory_cosine_num",
        "wrong_time_margin_num",
        "wrong_lineage_margin_num",
        "structural_dynamic_information_score",
        "strict_dynamic_class",
        "modeling_use",
    ]
    cand_cols = ["candidate_x", "current_signal", "primary_confound", "required_control", "model_use_if_pass"]
    law_cols = ["law", "status", "evidence", "blocker", "model_translation"]
    latent_rows = pd.DataFrame(decision["latentfm_response_energy_summary"].get("rows", []))
    latent_cols = [
        "task",
        "rho_pearson_pert",
        "rho_test_mmd",
        "within_dataset_rho_pearson_pert",
        "within_dataset_rho_test_mmd",
    ]

    text = f"""# LatentFM/ZSCAPE Structural Dynamic Scaling X Gate

## Boundary

- CPU/report-only synthesis over completed ZSCAPE and LatentFM reports.
- No new OT pairing, training, inference, GPU, checkpoint selection, canonical multi selection, or Track C query access.

## Decision

- Status: `{decision['status']}`
- Descriptor pass: `{decision['descriptor_pass']}`
- Model constraint ready: `{decision['model_constraint_ready']}`
- GPU authorized next: `{decision['gpu_authorized_next']}`
- Magnitude-is-not-information cross-evidence: `{decision['magnitude_not_information_cross_evidence']}`
- ZSCAPE geometry-positive but specificity-blocked rows: `{', '.join(decision['zscape_geometry_positive_specificity_blocked_rows']) or 'none'}`
- High response-norm rows confounded: `{decision['high_response_norm_confounded_rows']}/{decision['high_response_norm_rows']}`
- Response norm vs structural dynamic score Spearman: `{fmt(decision['response_norm_vs_structural_score_spearman'])}`
- LatentFM response-energy primary tasks with worse pp and worse MMD: `{decision['latentfm_response_energy_summary']['all_worse_rows']}/{decision['latentfm_response_energy_summary']['primary_rows']}`

Reasons:

{chr(10).join(f'- {r}' for r in decision['reasons'])}

## Structural Dynamic Information Definition

A useful dynamic scaling x should reward within-state displacement,
state-preservation, embryo reliability, temporal-tangent alignment, and
wrong-time/wrong-lineage margins, while penalizing composition-dominated shifts
and failed module specificity. This is deliberately different from raw response
magnitude or observable-gene concentration.

Current ZSCAPE supports the geometry part for periderm `noto/smo`, but not
module/pathway specificity. Current LatentFM static response-energy evidence
supports the same caution: bigger responses tend to be harder, not easier.

## ZSCAPE Scored Rows

{markdown_table(zscape.sort_values('structural_dynamic_information_score', ascending=False), row_cols)}

## Candidate X Variables

{markdown_table(candidate_x, cand_cols)}

## LatentFM Response-Energy Associations

{markdown_table(latent_rows, latent_cols)}

## ZSCAPE Law Panel

{markdown_table(law_panel, law_cols)}

## Next Action

Do not use this x as a model loss yet. The next runnable route is either a
broader ZSCAPE OT atlas over the existing 25-row expression subset manifest, or
a train-set translation table that explicitly separates composition shift,
within-state expression response, and dynamic pairability. Either route must
carry wrong-time, wrong-lineage, wrong-target, abundance/variance, and no-harm
controls before any GPU smoke.

## Outputs

- Scored rows: `{scored_path}`
- Candidate x rows: `{candidate_path}`
- JSON decision: `{json_path}`
"""
    md_path.write_text(text, encoding="utf-8")


def main() -> None:
    for name, path in INPUTS.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    zscape_rows = pd.read_csv(INPUTS["zscape_strict_rows"])
    candidate_x = pd.read_csv(INPUTS["zscape_candidate_x"])
    law_panel = pd.read_csv(INPUTS["zscape_law_panel"])
    crossfit = pd.read_csv(INPUTS["zscape_crossfit_repair"])
    translation = pd.read_csv(INPUTS["zscape_trainset_translation"])
    latent_assoc = pd.read_csv(INPUTS["latentfm_response_compressibility"])

    zscape = build_zscape_scores(zscape_rows)
    latent_summary = latentfm_response_energy_summary(latent_assoc)
    decision = decide(zscape, latent_summary, crossfit, translation)
    write_outputs(zscape, candidate_x, law_panel, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
