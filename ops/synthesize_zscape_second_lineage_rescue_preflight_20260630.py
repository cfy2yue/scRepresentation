#!/usr/bin/env python3
"""Preflight second-lineage rescue after ZSCAPE strict-control expansion."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "zscape_second_lineage_rescue_preflight_20260630"

INPUTS = {
    "atlas_rows": REPORTS
    / "zscape_dynamic_pairability_atlas_20260630"
    / "zscape_dynamic_pairability_atlas_rows_20260630.csv",
    "expansion_rows": REPORTS
    / "zscape_pairability_strict_control_expansion_20260630"
    / "zscape_pairability_strict_control_expansion_rows_20260630.csv",
    "expansion_candidates": REPORTS
    / "zscape_pairability_strict_control_expansion_20260630"
    / "zscape_pairability_strict_control_expansion_candidates_20260630.csv",
    "decision": REPORTS
    / "zscape_strict_control_decision_20260630"
    / "zscape_strict_control_decision_20260630.json",
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


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_lineages(expansion: pd.DataFrame, atlas: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    exp = expansion.copy()
    exp["strict_row_gate_bool"] = exp["strict_row_gate"].map(truthy)
    exp["effect_ratio_vs_max_null_p95"] = pd.to_numeric(exp["effect_ratio_vs_max_null_p95"], errors="coerce")
    exp["p_observed_le_matched_cc_null"] = pd.to_numeric(exp["p_observed_le_matched_cc_null"], errors="coerce")
    exp["p_observed_le_matched_label_null"] = pd.to_numeric(exp["p_observed_le_matched_label_null"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for lineage, sub in exp.groupby("cell_type_broad", sort=True):
        failed = sub[~sub["strict_row_gate_bool"]]
        label_pass = (sub["p_observed_le_matched_label_null"] <= 0.05).sum()
        cc_pass = (sub["p_observed_le_matched_cc_null"] <= 0.05).sum()
        ratio_pass = (sub["effect_ratio_vs_max_null_p95"] >= 1.05).sum()
        rows.append(
            {
                "lineage": lineage,
                "evaluated_rows": int(len(sub)),
                "strict_pass_rows": int(sub["strict_row_gate_bool"].sum()),
                "label_null_pass_rows": int(label_pass),
                "cc_null_pass_rows": int(cc_pass),
                "effect_ratio_pass_rows": int(ratio_pass),
                "max_effect_ratio": finite_float(sub["effect_ratio_vs_max_null_p95"].max()),
                "median_effect_ratio": finite_float(sub["effect_ratio_vs_max_null_p95"].median()),
                "failure_modes": ";".join(failure_modes(failed)),
            }
        )
    candidate_lineages = set(candidates.get("lineage", pd.Series(dtype=str)).astype(str))
    atlas_lineages = set(atlas.get("lineage", pd.Series(dtype=str)).astype(str))
    existing = {r["lineage"] for r in rows}
    for lineage in sorted((candidate_lineages | atlas_lineages) - existing):
        rows.append(
            {
                "lineage": lineage,
                "evaluated_rows": 0,
                "strict_pass_rows": 0,
                "label_null_pass_rows": 0,
                "cc_null_pass_rows": 0,
                "effect_ratio_pass_rows": 0,
                "max_effect_ratio": None,
                "median_effect_ratio": None,
                "failure_modes": "not_evaluated_in_strict_expansion",
            }
        )
    return pd.DataFrame(rows).sort_values(["strict_pass_rows", "max_effect_ratio"], ascending=[False, False])


def failure_modes(failed: pd.DataFrame) -> list[str]:
    modes: list[str] = []
    if failed.empty:
        return modes
    if (pd.to_numeric(failed["effect_ratio_vs_max_null_p95"], errors="coerce") < 1.05).any():
        modes.append("effect_ratio_below_1p05")
    if (pd.to_numeric(failed["p_observed_le_matched_cc_null"], errors="coerce") > 0.05).any():
        modes.append("matched_cc_null_not_exceeded")
    if (pd.to_numeric(failed["p_observed_le_matched_label_null"], errors="coerce") > 0.05).any():
        modes.append("label_null_not_exceeded")
    if (pd.to_numeric(failed.get("matched_subtype_jsd"), errors="coerce") > 0.1).any():
        modes.append("subtype_jsd_high")
    if (pd.to_numeric(failed.get("expression_library_smd"), errors="coerce").abs() > 0.35).any():
        modes.append("library_smd_high")
    return modes or ["gate_false_other"]


def decide(lineage_rows: pd.DataFrame, atlas: pd.DataFrame, expansion: pd.DataFrame) -> dict[str, Any]:
    pass_lineages = lineage_rows[lineage_rows["strict_pass_rows"] > 0]
    second_lineage_candidates = lineage_rows[
        (lineage_rows["lineage"] != "connective tissue-meninges-dermal FB")
        & (lineage_rows["evaluated_rows"] > 0)
    ].copy()
    near_second = second_lineage_candidates[
        (second_lineage_candidates["effect_ratio_pass_rows"] > 0)
        & (second_lineage_candidates["cc_null_pass_rows"] > 0)
    ]
    atlas_only_not_evaluated = atlas[
        ~atlas["row_id"].astype(str).isin(set(expansion["row_id"].astype(str)))
    ].copy()
    high_pair_not_evaluated = atlas_only_not_evaluated[
        pd.to_numeric(atlas_only_not_evaluated.get("within_state_pairability_score"), errors="coerce") >= 0.70
    ]
    reasons: list[str] = []
    if int(pass_lineages["lineage"].nunique()) < 2:
        reasons.append("only_one_passing_lineage")
    if near_second.empty:
        reasons.append("no_near_second_lineage_in_current_evaluated_candidates")
    if high_pair_not_evaluated.empty:
        reasons.append("no_high_pairability_untried_rows_left_in_current_25row_atlas")
    status = "zscape_second_lineage_rescue_current_manifest_blocked_no_gpu"
    next_action = (
        "do not rerun basal/retinal/current failed candidates blindly; second-lineage rescue requires "
        "a prospective expanded manifest or relaxed-but-predeclared candidate discovery followed by the same strict controls"
    )
    return {
        "status": status,
        "gpu_authorized_next": False,
        "reasons": reasons,
        "passing_lineages": pass_lineages["lineage"].astype(str).tolist(),
        "near_second_lineages": near_second["lineage"].astype(str).tolist(),
        "high_pairability_untried_rows_left": int(len(high_pair_not_evaluated)),
        "next_action": next_action,
    }


def fmt(value: Any, digits: int = 4) -> str:
    x = finite_float(value)
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


def md_table(df: pd.DataFrame) -> str:
    cols = [
        "lineage",
        "evaluated_rows",
        "strict_pass_rows",
        "cc_null_pass_rows",
        "label_null_pass_rows",
        "effect_ratio_pass_rows",
        "max_effect_ratio",
        "failure_modes",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        vals = [fmt(row.get(c)) if isinstance(row.get(c), float) else str(row.get(c)) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_outputs(lineage_rows: pd.DataFrame, decision: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUT_DIR / "zscape_second_lineage_rescue_lineage_rows_20260630.csv"
    json_path = OUT_DIR / "zscape_second_lineage_rescue_preflight_20260630.json"
    md_path = OUT_DIR / "LATENTFM_ZSCAPE_SECOND_LINEAGE_RESCUE_PREFLIGHT_20260630.md"
    lineage_rows.to_csv(rows_path, index=False)
    payload = {
        "boundary": {
            "cpu_report_only": True,
            "new_ot_pairing": False,
            "training_or_inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection": False,
            "trackc_query_access": False,
            "gpu_authorized_next": False,
        },
        "inputs": {k: str(v) for k, v in INPUTS.items()},
        "decision": decision,
        "outputs": {"lineage_rows": str(rows_path), "markdown_report": str(md_path)},
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    text = f"""# ZSCAPE Second-Lineage Rescue Preflight 20260630

## Boundary

- CPU/report-only synthesis over completed atlas and strict-control expansion rows.
- No new OT pairing, training, inference, GPU, checkpoint selection, canonical multi selection, or Track C query access.

## Decision

- status: `{decision['status']}`
- GPU authorized next: `{decision['gpu_authorized_next']}`
- reasons: `{', '.join(decision['reasons'])}`
- high-pairability untried rows left in current atlas: `{decision['high_pairability_untried_rows_left']}`
- next action: `{decision['next_action']}`

## Lineage Summary

{md_table(lineage_rows)}

## Interpretation

The current 25-row atlas and 14-row strict-control expansion do not contain an
immediate second-lineage rescue path. Basal and retinal candidates were already
tested and did not pass strict matched-null controls. A second-lineage claim
requires a prospective expanded candidate manifest or a predeclared relaxed
candidate-discovery stage followed by the same strict controls.

## Artifacts

- JSON: `{json_path}`
- lineage rows: `{rows_path}`
"""
    md_path.write_text(text, encoding="utf-8")


def main() -> None:
    atlas = pd.read_csv(INPUTS["atlas_rows"])
    expansion = pd.read_csv(INPUTS["expansion_rows"])
    candidates = pd.read_csv(INPUTS["expansion_candidates"])
    _ = load_json(INPUTS["decision"])
    lineage_rows = summarize_lineages(expansion, atlas, candidates)
    decision = decide(lineage_rows, atlas, expansion)
    write_outputs(lineage_rows, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
