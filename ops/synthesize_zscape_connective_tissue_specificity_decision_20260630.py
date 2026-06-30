#!/usr/bin/env python3
"""Decision wrapper for connective-tissue ZSCAPE crossfit specificity panel."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs"
OUT_DIR = REPORTS / "zscape_connective_tissue_specificity_decision_20260630"

PANEL_DIR = REPORTS / "zscape_connective_tissue_crossfit_specificity_20260630"
RUN_DIR = (
    RUNS
    / "zscape_connective_tissue_crossfit_specificity_20260630"
    / "zscape_connective_tissue_crossfit_specificity_20260630_0405"
)

INPUTS = {
    "panel_json": PANEL_DIR / "zscape_crossfit_specificity_gate_20260628.json",
    "query_rows": PANEL_DIR / "zscape_crossfit_specificity_query_rows.csv",
    "row_summary": PANEL_DIR / "zscape_crossfit_specificity_row_summary.csv",
    "wrong_control_rows": PANEL_DIR / "zscape_crossfit_specificity_wrong_control_rows.csv",
    "matched_random_rows": PANEL_DIR / "zscape_crossfit_specificity_matched_random_rows.csv",
    "run_exit_code": RUN_DIR / "EXIT_CODE",
    "run_status": RUN_DIR / "RUN_STATUS.md",
}

EXPECTED_ROWS = [
    "connective_tissue_meninges_dermal_fb__tbx16_tbx16l__24p0h",
    "connective_tissue_meninges_dermal_fb__tfap2a_foxd3__48p0h",
    "connective_tissue_meninges_dermal_fb__tbx16_tbx16l__36p0h",
    "connective_tissue_meninges_dermal_fb__wnt3a_wnt8__36p0h",
]


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_exit(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def summarize_queries(query: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    query = query.copy()
    query["query_gate_bool"] = query["query_gate"].map(truthy)
    for row_id in EXPECTED_ROWS:
        sub = query[query["row_id"].astype(str).eq(row_id)].copy()
        rows.append(
            {
                "row_id": row_id,
                "queries": int(len(sub)),
                "query_gates": int(sub["query_gate_bool"].sum()) if not sub.empty else 0,
                "any_query_gate": bool(sub["query_gate_bool"].any()) if not sub.empty else False,
                "best_direction": best_direction(sub),
                "best_margin_q05": finite_float(pd.to_numeric(sub.get("specificity_margin_q05"), errors="coerce").max())
                if not sub.empty
                else None,
                "best_random_margin_q05": finite_float(
                    pd.to_numeric(sub.get("random_margin_q05"), errors="coerce").max()
                )
                if not sub.empty
                else None,
                "best_heldout_ci_low_q05": finite_float(
                    pd.to_numeric(sub.get("heldout_ci_low_q05"), errors="coerce").max()
                )
                if not sub.empty
                else None,
                "failure_modes": ";".join(failure_modes(sub)),
            }
        )
    return pd.DataFrame(rows)


def best_direction(sub: pd.DataFrame) -> str:
    if sub.empty:
        return ""
    work = sub.copy()
    work["specificity_margin_q05_num"] = pd.to_numeric(work.get("specificity_margin_q05"), errors="coerce")
    work["query_gate_bool"] = work["query_gate"].map(truthy)
    work = work.sort_values(["query_gate_bool", "specificity_margin_q05_num"], ascending=[False, False])
    row = work.iloc[0]
    return str(row.get("direction", ""))


def failure_modes(sub: pd.DataFrame) -> list[str]:
    if sub.empty:
        return ["missing_query_rows"]
    modes: list[str] = []
    if not sub["query_gate"].map(truthy).any():
        for col, mode, threshold in [
            ("n_mapped_genes_min", "mapped_genes_lt_8", 8),
            ("heldout_ci_low_q05", "heldout_ci_low_not_positive", 0.0),
            ("effect_positive_fraction", "effect_fraction_lt_0p75", 0.75),
            ("specificity_positive_fraction", "specificity_fraction_lt_0p75", 0.75),
            ("specificity_margin_q05", "specificity_margin_q05_le_0p02", 0.02),
            ("random_margin_q05", "random_margin_q05_lt_0p01", 0.01),
        ]:
            values = pd.to_numeric(sub.get(col), errors="coerce")
            if values.empty or values.max() < threshold:
                modes.append(mode)
        signflip = pd.to_numeric(sub.get("signflip_margin_median"), errors="coerce")
        if signflip.empty or signflip.min() > 0.0:
            modes.append("signflip_margin_positive")
    return modes or ["query_gate_false_other"]


def decide(panel_json: dict[str, Any], row_summary: pd.DataFrame, exit_code: str | None) -> dict[str, Any]:
    if exit_code is None:
        return {
            "status": "zscape_connective_tissue_specificity_awaiting_panel",
            "gpu_authorized_next": False,
            "ready": False,
            "next_action": "wait for detached CPU specificity panel at long-job cadence",
        }
    if exit_code != "0":
        return {
            "status": "zscape_connective_tissue_specificity_panel_failed",
            "gpu_authorized_next": False,
            "ready": True,
            "exit_code": exit_code,
            "next_action": "read run log, fix minimal issue, and record bug before any rerun",
        }
    pass_rows = int(row_summary["any_query_gate"].sum())
    status = (
        "zscape_connective_tissue_specificity_biological_pass_no_gpu"
        if pass_rows >= 3
        else "zscape_connective_tissue_specificity_fail_keep_pairability_only_no_gpu"
    )
    return {
        "status": status,
        "gpu_authorized_next": False,
        "ready": True,
        "exit_code": exit_code,
        "panel_source_status": panel_json.get("status"),
        "rows_with_any_query_gate": pass_rows,
        "rows_required": 3,
        "row_gate_pass": bool(pass_rows >= 3),
        "allowed_claim_if_pass": (
            "lineage-local connective-tissue module/specificity biology candidate, still no model route"
        ),
        "forbidden_claims": [
            "no broad cross-lineage ZSCAPE pairability",
            "no LatentFM/RawFM loss, sampling, checkpoint, or model-positive use",
            "no canonical multi or Track C query use",
            "no true lineage trajectory claim from snapshot pseudo-pairs",
        ],
        "next_action": (
            "if pass, design stricter biological atlas expansion; if fail, close connective-tissue mechanism claim"
        ),
    }


def fmt(value: Any, digits: int = 4) -> str:
    x = finite_float(value)
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


def md_table(df: pd.DataFrame) -> str:
    cols = [
        "row_id",
        "queries",
        "query_gates",
        "any_query_gate",
        "best_direction",
        "best_margin_q05",
        "best_random_margin_q05",
        "best_heldout_ci_low_q05",
        "failure_modes",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        vals = [fmt(row.get(col)) if isinstance(row.get(col), float) else str(row.get(col)) for col in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_outputs(row_summary: pd.DataFrame, decision: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUT_DIR / "zscape_connective_tissue_specificity_decision_rows_20260630.csv"
    json_path = OUT_DIR / "zscape_connective_tissue_specificity_decision_20260630.json"
    md_path = OUT_DIR / "LATENTFM_ZSCAPE_CONNECTIVE_TISSUE_SPECIFICITY_DECISION_20260630.md"
    row_summary.to_csv(rows_path, index=False)
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
        "decision": decision,
        "outputs": {"rows": str(rows_path), "markdown_report": str(md_path)},
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    text = f"""# ZSCAPE Connective-Tissue Specificity Decision 20260630

## Boundary

- CPU/report-only decision wrapper over the detached connective-tissue crossfit specificity panel.
- No training, inference, GPU, checkpoint selection, canonical multi selection, or Track C query access.

## Decision

- status: `{decision['status']}`
- GPU authorized next: `{decision['gpu_authorized_next']}`
- ready: `{decision.get('ready')}`
- rows with any query gate: `{decision.get('rows_with_any_query_gate', 'NA')}`
- rows required: `{decision.get('rows_required', 'NA')}`
- next action: `{decision['next_action']}`

## Row Summary

{md_table(row_summary)}

## Artifacts

- JSON: `{json_path}`
- rows: `{rows_path}`
"""
    md_path.write_text(text, encoding="utf-8")


def main() -> None:
    exit_code = read_exit(INPUTS["run_exit_code"])
    panel_json = read_json(INPUTS["panel_json"])
    if INPUTS["query_rows"].exists():
        query = pd.read_csv(INPUTS["query_rows"])
        row_summary = summarize_queries(query)
    else:
        row_summary = pd.DataFrame(
            [
                {
                    "row_id": row_id,
                    "queries": 0,
                    "query_gates": 0,
                    "any_query_gate": False,
                    "best_direction": "",
                    "best_margin_q05": None,
                    "best_random_margin_q05": None,
                    "best_heldout_ci_low_q05": None,
                    "failure_modes": "awaiting_panel_output",
                }
                for row_id in EXPECTED_ROWS
            ]
        )
    decision = decide(panel_json, row_summary, exit_code)
    write_outputs(row_summary, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
