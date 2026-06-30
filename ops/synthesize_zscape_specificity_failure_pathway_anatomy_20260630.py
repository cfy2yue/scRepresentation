#!/usr/bin/env python3
"""ZSCAPE pathway anatomy for specificity failures.

CPU/report-only synthesis over existing expression-space module, enrichment,
heldout, wrong-control, and matched-random outputs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "zscape_specificity_failure_pathway_anatomy_20260630"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def compact_terms(part: pd.DataFrame, limit: int = 3) -> str:
    if part.empty:
        return "none"
    cols = [col for col in ["source", "native", "name", "p_value"] if col in part.columns]
    entries: list[str] = []
    for _, row in part.sort_values("p_value", ascending=True).head(limit).iterrows():
        label = str(row.get("name", row.get("native", "term")))
        source = str(row.get("source", ""))
        pval = fmt(row.get("p_value"), 2)
        entries.append(f"{source}:{label} p={pval}")
    return " | ".join(entries) if entries else "none"


def main() -> int:
    crossfit_dir = REPORTS / "zscape_crossfit_residual_specificity_repair_gate_20260628"
    query = pd.read_csv(crossfit_dir / "zscape_crossfit_specificity_query_rows.csv")
    wrong = pd.read_csv(crossfit_dir / "zscape_crossfit_specificity_wrong_control_rows.csv")
    random_rows = pd.read_csv(crossfit_dir / "zscape_crossfit_specificity_matched_random_rows.csv")
    module_rows = pd.read_csv(REPORTS / "zscape_expression_module_scores_20260628" / "zscape_expression_module_score_rows.csv")
    terms = pd.read_csv(REPORTS / "zscape_gprofiler_enrichment_preflight_20260628" / "zscape_gprofiler_enrichment_terms.csv")
    heldout = pd.read_csv(
        REPORTS
        / "zscape_embryo_heldout_dynamic_specificity_gate_20260628"
        / "zscape_embryo_heldout_dynamic_specificity_query_rows.csv"
    )
    substate = pd.read_csv(
        REPORTS
        / "zscape_periderm_substate_time_qc_ot_module_gate_20260628"
        / "zscape_periderm_substate_time_qc_module_query_rows.csv"
    )

    focus_rows = ["periderm__noto__24p0h", "periderm__smo__24p0h"]
    focus = query[query["row_id"].isin(focus_rows)].copy()
    out_rows: list[dict[str, Any]] = []
    for _, q in focus.iterrows():
        query_name = str(q["query_name"]).replace(".crossfit", "")
        row_id = str(q["row_id"])
        direction = str(q["direction"])
        wrong_part = wrong[wrong["query_name"].eq(q["query_name"])]
        random_part = random_rows[random_rows["query_name"].eq(q["query_name"])]
        mod_part = module_rows[module_rows["query_name"].eq(query_name)]
        term_part = terms[terms["query_name"].eq(query_name) & terms["significant"].astype(bool)]
        held_part = heldout[heldout["query_name"].eq(query_name)]
        sub_part = substate[substate["query_name"].eq(query_name)]
        out_rows.append(
            {
                "query_name": query_name,
                "row_id": row_id,
                "direction": direction,
                "heldout_effect_positive_fraction": float(q.get("effect_positive_fraction", np.nan)),
                "crossfit_specificity_positive_fraction": float(q.get("specificity_positive_fraction", np.nan)),
                "crossfit_specificity_margin_q05": float(q.get("specificity_margin_q05", np.nan)),
                "matched_random_margin_q05": float(q.get("random_margin_q05", np.nan)),
                "wrong_control_median_diff": float(pd.to_numeric(wrong_part["directed_diff"], errors="coerce").median()),
                "wrong_control_p95_diff": float(pd.to_numeric(wrong_part["directed_diff"], errors="coerce").quantile(0.95)),
                "matched_random_p95_median": float(pd.to_numeric(random_part["matched_random_p95"], errors="coerce").median()),
                "module_directed_diff": float(pd.to_numeric(mod_part["directed_mean_diff"], errors="coerce").mean())
                if not mod_part.empty
                else np.nan,
                "module_gate": bool(mod_part["module_direction_gate"].any()) if not mod_part.empty else False,
                "heldout_gate": bool(held_part["query_gate"].any()) if not held_part.empty else False,
                "substate_gate": bool(sub_part["gate"].any()) if not sub_part.empty and "gate" in sub_part.columns else False,
                "top_enrichment_terms": compact_terms(term_part),
                "failure_interpretation": (
                    "shared_wrong_control_or_random_program_catches_up"
                    if float(q.get("specificity_positive_fraction", 0.0)) <= 0.0
                    else "partial_specificity_review"
                ),
            }
        )

    out_df = pd.DataFrame(out_rows)
    any_specific = bool(out_df["crossfit_specificity_positive_fraction"].gt(0).any())
    all_module_effects = bool(out_df["heldout_effect_positive_fraction"].gt(0.9).any())
    status = "zscape_specificity_failure_pathway_shared_program_no_model_route"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUT_DIR / "zscape_specificity_failure_pathway_anatomy_rows_20260630.csv"
    json_path = OUT_DIR / "zscape_specificity_failure_pathway_anatomy_20260630.json"
    md_path = OUT_DIR / "LATENTFM_ZSCAPE_SPECIFICITY_FAILURE_PATHWAY_ANATOMY_20260630.md"
    out_df.to_csv(rows_path, index=False)

    payload = {
        "created_at": now_cst(),
        "status": status,
        "gpu_authorized_next": False,
        "model_route_authorized": False,
        "any_crossfit_specificity_positive": any_specific,
        "heldout_effects_exist": all_module_effects,
        "decision": (
            "periderm noto/smo modules show real expression effects but fail target/pathway specificity; "
            "use as pathway/failure anatomy and negative-control taxonomy only"
        ),
        "boundary": "cpu_report_only_expression_space_no_training_no_inference_no_gpu_no_canonical_multi_no_trackc_query",
        "outputs": {"markdown": str(md_path), "json": str(json_path), "rows": str(rows_path)},
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM ZSCAPE Specificity-Failure Pathway Anatomy 20260630",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only synthesis over existing ZSCAPE expression-space outputs.",
        "- No new OT, training, inference, checkpoint selection, canonical multi, Track C query, or GPU.",
        "- Focus is biological interpretation and negative-control taxonomy, not model promotion.",
        "",
        "## Decision",
        "",
        "- Periderm `noto/smo` have real heldout expression/module effects, but target/pathway specificity fails.",
        "- Wrong-target/time/lineage and matched-random controls catch up, so current modules look like shared epithelial/stress/cytoskeletal/oxidoreduction programs rather than target-specific constraints.",
        "- Do not convert these pathways into LatentFM/RawFM losses or sampling positives.",
        "",
        "## Query Anatomy",
        "",
        "| query | effect frac | specificity frac | spec q05 | wrong p95 | top terms |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for _, row in out_df.iterrows():
        lines.append(
            f"| `{row['query_name']}` | {fmt(row['heldout_effect_positive_fraction'])} | "
            f"{fmt(row['crossfit_specificity_positive_fraction'])} | "
            f"{fmt(row['crossfit_specificity_margin_q05'])} | "
            f"{fmt(row['wrong_control_p95_diff'])} | {row['top_enrichment_terms']} |"
        )
    lines += [
        "",
        "## Modeling Implication",
        "",
        "- ZSCAPE still has value as biological insight: state-preserved dynamic geometry exists for periderm `noto/smo`.",
        "- The supported modeling lesson is negative: magnitude or broad pathway activity is not enough; any future dynamic constraint must clear specificity and wrong-control gates first.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{json_path}`",
        f"- rows: `{rows_path}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "markdown": str(md_path), "json": str(json_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
