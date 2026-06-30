#!/usr/bin/env python3
"""Synthesize decision for completed ZSCAPE strict-control expansion."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "zscape_strict_control_decision_20260630"

INPUTS = {
    "expansion_json": REPORTS
    / "zscape_pairability_strict_control_expansion_20260630"
    / "zscape_pairability_strict_control_expansion_20260630.json",
    "expansion_rows": REPORTS
    / "zscape_pairability_strict_control_expansion_20260630"
    / "zscape_pairability_strict_control_expansion_rows_20260630.csv",
    "atlas_json": REPORTS
    / "zscape_dynamic_pairability_atlas_20260630"
    / "zscape_dynamic_pairability_atlas_20260630.json",
    "structural_json": REPORTS
    / "zscape_structural_dynamic_scaling_x_20260630"
    / "zscape_structural_dynamic_scaling_x_20260630.json",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


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


def summarize_rows(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {
            "rows": 0,
            "pass_rows": 0,
            "pass_lineages": 0,
            "pass_targets": 0,
            "pass_timepoints": [],
            "passing_row_ids": [],
        }
    work = rows.copy()
    work["strict_row_gate_bool"] = work["strict_row_gate"].map(truthy)
    passing = work[work["strict_row_gate_bool"]].copy()
    return {
        "rows": int(len(work)),
        "pass_rows": int(len(passing)),
        "pass_lineages": int(passing["cell_type_broad"].nunique()),
        "pass_targets": int(passing["gene_target"].nunique()),
        "pass_timepoints": sorted({float(x) for x in passing["timepoint"].dropna().tolist()}),
        "passing_row_ids": passing["row_id"].astype(str).tolist(),
        "lineage_pass_counts": passing.groupby("cell_type_broad").size().astype(int).to_dict(),
        "target_pass_counts": passing.groupby("gene_target").size().astype(int).to_dict(),
        "failed_lineages": sorted(set(work["cell_type_broad"].astype(str)) - set(passing["cell_type_broad"].astype(str))),
        "median_pass_effect_ratio": finite_float(pd.to_numeric(passing["effect_ratio_vs_max_null_p95"], errors="coerce").median())
        if not passing.empty
        else None,
    }


def decide(summary: dict[str, Any], expansion: dict[str, Any]) -> dict[str, Any]:
    required_rows = 4
    required_lineages = 2
    pass_rows = int(summary["pass_rows"])
    pass_lineages = int(summary["pass_lineages"])
    if pass_rows >= required_rows and pass_lineages >= required_lineages:
        status = "zscape_strict_control_expansion_design_review_pass_no_gpu"
        interpretation = "cross-lineage strict-control design-review pass"
    elif pass_rows >= required_rows and pass_lineages == 1:
        status = "zscape_strict_control_expansion_partial_lineage_signal_design_gate_fail_no_gpu"
        interpretation = "lineage-local biological signal, cross-lineage design gate failed"
    elif pass_rows > 0:
        status = "zscape_strict_control_expansion_weak_partial_signal_design_gate_fail_no_gpu"
        interpretation = "weak partial signal, design gate failed"
    else:
        status = "zscape_strict_control_expansion_fail_close_no_gpu"
        interpretation = "no strict-control signal"
    return {
        "status": status,
        "interpretation": interpretation,
        "gpu_authorized_next": False,
        "design_gate": {
            "required_rows": required_rows,
            "required_lineages": required_lineages,
            "observed_rows": pass_rows,
            "observed_lineages": pass_lineages,
            "passed": bool(pass_rows >= required_rows and pass_lineages >= required_lineages),
        },
        "allowed_claims": [
            "A connective-tissue lineage-local dynamic pairability signal survived matched control-control and label-shuffle nulls.",
            "Magnitude alone is not the scaling variable; pairability depends on lineage/state/control context.",
            "The result can motivate a stricter biological atlas design or failure-analysis figure.",
        ],
        "forbidden_claims": [
            "Do not claim general cross-lineage ZSCAPE dynamic pairability.",
            "Do not use these atlas-only rows as LatentFM/RawFM model positives.",
            "Do not launch GPU training, losses, sampling weights, or checkpoint selection from this result.",
            "Do not treat OT snapshot pseudo-pairs as true lineage trajectories.",
        ],
        "next_actions": [
            {
                "name": "connective_tissue_specificity_repair_cpu_gate",
                "hypothesis": "the four passing connective-tissue rows reflect a real lineage-local perturbation trajectory rather than a generic lineage/time artifact",
                "gate": "pass only if wrong-target, wrong-time, and wrong-lineage controls plus module/pathway specificity all remain positive for >=3 targets and >=2 timepoints",
                "fail_close": "if specificity fails, keep the rows as pairability examples only, not biological mechanism or model constraints",
            },
            {
                "name": "cross_lineage_expansion_design_review",
                "hypothesis": "additional strict-control candidates may recover a second lineage if candidate selection is not restricted to current high-pairability atlas-only rows",
                "gate": "prospective candidate set must pass >=4 rows and >=2 lineages under the same matched-null controls before model translation",
                "fail_close": "if retinal/basal or new lineages remain null-dominated, close cross-lineage ZSCAPE model route",
            },
            {
                "name": "trainset_translation_blocker_record",
                "hypothesis": "human LatentFM train-set pairability needs its own train-only analogue rather than direct zebrafish labels",
                "gate": "only proceed if a train-only high/low design reaches >=300 matched pairs, >=2 perturbation types, SMD<=0.15/AUC<=0.60, null p95 below real, then no-harm",
                "fail_close": "do not use ZSCAPE labels, canonical multi, or Track C query to select train rows",
            },
        ],
        "source_status": (expansion.get("summary") or {}).get("status"),
    }


def fmt(value: Any, digits: int = 4) -> str:
    x = finite_float(value)
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


def write_outputs(rows: pd.DataFrame, summary: dict[str, Any], decision: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUT_DIR / "zscape_strict_control_decision_rows_20260630.csv"
    json_path = OUT_DIR / "zscape_strict_control_decision_20260630.json"
    md_path = OUT_DIR / "LATENTFM_ZSCAPE_STRICT_CONTROL_DECISION_20260630.md"
    rows.to_csv(rows_path, index=False)
    payload = {
        "boundary": {
            "cpu_report_only": True,
            "training_or_inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection": False,
            "trackc_query_access": False,
            "gpu_authorized_next": False,
        },
        "inputs": {k: str(v) for k, v in INPUTS.items()},
        "summary": summary,
        "decision": decision,
        "outputs": {"rows": str(rows_path), "markdown_report": str(md_path)},
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    pass_rows = rows[rows["strict_row_gate"].map(truthy)].copy()
    table_lines = [
        "| row_id | lineage | target | time | ratio | cc p | label p | gate |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in pass_rows.iterrows():
        table_lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("row_id")),
                    str(row.get("cell_type_broad")),
                    str(row.get("gene_target")),
                    fmt(row.get("timepoint"), 1),
                    fmt(row.get("effect_ratio_vs_max_null_p95")),
                    fmt(row.get("p_observed_le_matched_cc_null")),
                    fmt(row.get("p_observed_le_matched_label_null")),
                    str(row.get("strict_row_gate")),
                ]
            )
            + " |"
        )

    text = f"""# LatentFM ZSCAPE Strict-Control Decision 20260630

## Boundary

- CPU/report-only synthesis of the completed strict-control expansion.
- No new OT pairing, training, inference, GPU, checkpoint selection, canonical multi selection, or Track C query access.
- OT pairs are snapshot pseudo-pairs, not true lineage pairs.

## Decision

- status: `{decision['status']}`
- interpretation: `{decision['interpretation']}`
- GPU authorized next: `{decision['gpu_authorized_next']}`
- design gate passed: `{decision['design_gate']['passed']}`
- observed passing rows/lineages: `{decision['design_gate']['observed_rows']}` / `{decision['design_gate']['observed_lineages']}`
- required passing rows/lineages: `{decision['design_gate']['required_rows']}` / `{decision['design_gate']['required_lineages']}`

## Passing Rows

{chr(10).join(table_lines)}

## Allowed Claims

{chr(10).join(f'- {x}' for x in decision['allowed_claims'])}

## Forbidden Claims

{chr(10).join(f'- {x}' for x in decision['forbidden_claims'])}

## Next CPU/Design Steps

{chr(10).join(f'- `{x["name"]}`: {x["hypothesis"]} Gate: {x["gate"]} Fail-close: {x["fail_close"]}' for x in decision['next_actions'])}

## Interpretation

The expansion found a real-looking but lineage-local signal. Four rows passed
strict matched-null controls, but all are connective tissue-meninges-dermal FB.
That is biologically useful and supports the broader `magnitude != information`
story, yet it fails the predeclared cross-lineage design gate and cannot be
translated into a LatentFM/RawFM model route without new controls.

## Artifacts

- JSON: `{json_path}`
- rows: `{rows_path}`
"""
    md_path.write_text(text, encoding="utf-8")


def main() -> None:
    expansion = load_json(INPUTS["expansion_json"])
    rows = pd.read_csv(INPUTS["expansion_rows"])
    summary = summarize_rows(rows)
    decision = decide(summary, expansion)
    write_outputs(rows, summary, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
