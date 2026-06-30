#!/usr/bin/env python3
"""Decision wrapper for prospective partial-row ZSCAPE specificity panel."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
DEFAULT_RUN = (
    ROOT
    / "runs/zscape_prospective_partial_crossfit_specificity_20260630"
    / "zscape_prospective_partial_crossfit_specificity_20260630_1148"
)
DEFAULT_PANEL_DIR = REPORTS / "zscape_prospective_partial_crossfit_specificity_20260630"
DEFAULT_STRICT_DECISION_JSON = (
    REPORTS
    / "zscape_prospective_strict_control_decision_20260630"
    / "zscape_prospective_strict_control_decision_20260630.json"
)
OUT_DIR = REPORTS / "zscape_prospective_partial_specificity_decision_20260630"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def load_exit_code(run_root: Path) -> str:
    path = run_root / "EXIT_CODE"
    if not path.exists():
        return "still_running"
    return path.read_text(encoding="utf-8").strip()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_output_dir(path: Path, force: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = [p for p in path.iterdir() if p.name != ".DS_Store"]
    if existing and not force:
        raise SystemExit(f"Refusing to overwrite nonempty output directory: {path}")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def expected_rows_from_strict(path: Path) -> list[str]:
    payload = load_json(path)
    return [str(x) for x in (payload.get("decision") or {}).get("passing_row_ids", [])]


def build_decision(
    row_summary: pd.DataFrame,
    query_rows: pd.DataFrame,
    panel: dict[str, Any],
    exit_code: str,
    expected_row_ids: list[str],
) -> dict[str, Any]:
    if exit_code == "still_running":
        return {
            "status": "zscape_prospective_partial_specificity_decision_awaiting_run",
            "ready": False,
            "gpu_authorized_next": False,
            "next_action": "wait for the focused specificity panel at normal long-job cadence",
        }
    if exit_code != "0":
        return {
            "status": "zscape_prospective_partial_specificity_runtime_fail",
            "ready": False,
            "exit_code": exit_code,
            "gpu_authorized_next": False,
            "next_action": "inspect focused specificity log; do not interpret biological signal",
        }
    if row_summary.empty:
        return {
            "status": "zscape_prospective_partial_specificity_no_rows_fail_close",
            "ready": True,
            "exit_code": exit_code,
            "gpu_authorized_next": False,
            "next_action": "close prospective second-lineage rescue",
        }

    work = row_summary.copy()
    work["any_query_gate_bool"] = work.get("any_query_gate", False).map(truthy)
    work["query_gates_num"] = pd.to_numeric(work.get("query_gates", 0), errors="coerce").fillna(0).astype(int)
    rows_total = int(len(work))
    observed_row_ids = work["row_id"].astype(str).tolist() if "row_id" in work.columns else []
    missing_expected = sorted(set(expected_row_ids) - set(observed_row_ids))
    unexpected_rows = sorted(set(observed_row_ids) - set(expected_row_ids)) if expected_row_ids else []
    provenance_validation = {
        "strict_decision_expected_rows": expected_row_ids,
        "observed_panel_rows": observed_row_ids,
        "missing_expected_rows": missing_expected,
        "unexpected_rows": unexpected_rows,
        "row_ids_match_strict_pass_rows": bool(expected_row_ids and not missing_expected and not unexpected_rows),
    }
    if expected_row_ids and (missing_expected or unexpected_rows):
        return {
            "status": "zscape_prospective_partial_specificity_row_id_mismatch_blocked",
            "ready": False,
            "exit_code": exit_code,
            "gpu_authorized_next": False,
            "provenance_validation": provenance_validation,
            "next_action": "fix row-id provenance before interpreting the focused specificity panel",
        }
    rows_any = int(work["any_query_gate_bool"].sum())
    total_query_gates = int(work["query_gates_num"].sum())
    passing_rows = work.loc[work["any_query_gate_bool"], "row_id"].astype(str).tolist()
    failing_rows = work.loc[~work["any_query_gate_bool"], "row_id"].astype(str).tolist()

    if rows_total >= 3 and rows_any == rows_total and total_query_gates >= rows_total:
        status = "zscape_prospective_partial_specificity_all_rows_pass_design_review_no_gpu"
        next_action = (
            "record focused biological specificity support; design only a CPU train-set translation/no-harm "
            "gate before any model route"
        )
    elif rows_any > 0:
        status = "zscape_prospective_partial_specificity_partial_signal_fail_close_no_gpu"
        next_action = (
            "record lineage-local partial signal; close broad prospective second-lineage rescue and do not "
            "launch model training"
        )
    else:
        status = "zscape_prospective_partial_specificity_fail_close_no_gpu"
        next_action = "close prospective second-lineage rescue and keep rows as descriptor/failure-analysis only"

    return {
        "status": status,
        "ready": True,
        "exit_code": exit_code,
        "gpu_authorized_next": False,
        "rows_total": rows_total,
        "rows_with_any_query_gate": rows_any,
        "total_query_gates": total_query_gates,
        "passing_rows": passing_rows,
        "failing_rows": failing_rows,
        "panel_status": panel.get("status"),
        "panel_biological_pass": panel.get("biological_pass"),
        "panel_model_constraint_precondition": panel.get("model_constraint_precondition"),
        "provenance_validation": provenance_validation,
        "forbidden_claims": [
            "no LatentFM/RawFM loss, sampling, checkpoint, model-positive, or constraint use",
            "no broad second-lineage ZSCAPE pairability claim because the prior strict gate failed the row-count criterion",
            "no true lineage trajectory claim from snapshot pseudo-pairs",
            "no canonical multi or Track C query use",
        ],
        "next_action": next_action,
    }


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_None._"
    cols = [c for c in ["row_id", "queries", "query_gates", "any_query_gate"] if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def write_outputs(row_summary: pd.DataFrame, query_rows: pd.DataFrame, decision: dict[str, Any], args: argparse.Namespace) -> None:
    ensure_output_dir(args.out_dir, args.force)
    rows_path = args.out_dir / "zscape_prospective_partial_specificity_decision_row_summary_20260630.csv"
    query_path = args.out_dir / "zscape_prospective_partial_specificity_decision_query_rows_20260630.csv"
    json_path = args.out_dir / "zscape_prospective_partial_specificity_decision_20260630.json"
    md_path = args.out_dir / "LATENTFM_ZSCAPE_PROSPECTIVE_PARTIAL_SPECIFICITY_DECISION_20260630.md"
    row_summary.to_csv(rows_path, index=False)
    query_rows.to_csv(query_path, index=False)
    payload = {
        "timestamp_cst": now_cst(),
        "boundary": {
            "cpu_report_only": True,
            "new_training_or_inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection": False,
            "trackc_query_access": False,
            "gpu_authorized_next": False,
        },
        "inputs": {
            "run_root": str(args.run_root),
            "panel_dir": str(args.panel_dir),
        },
        "decision": decision,
        "outputs": {
            "row_summary": str(rows_path),
            "query_rows": str(query_path),
            "markdown_report": str(md_path),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    text = f"""# ZSCAPE Prospective Partial Specificity Decision

## Boundary

- CPU/report-only synthesis after the focused 3-row specificity panel.
- No training, inference, checkpoint selection, canonical multi selection, Track C query access, or GPU.

## Decision

- Status: `{decision['status']}`
- Ready: `{decision.get('ready')}`
- GPU authorized next: `{decision['gpu_authorized_next']}`
- Rows with any query gate: `{decision.get('rows_with_any_query_gate', 'NA')}` / `{decision.get('rows_total', 'NA')}`
- Total query gates: `{decision.get('total_query_gates', 'NA')}`
- Passing rows: `{decision.get('passing_rows', [])}`
- Failing rows: `{decision.get('failing_rows', [])}`
- Row IDs match strict pass rows: `{(decision.get('provenance_validation') or {}).get('row_ids_match_strict_pass_rows', 'NA')}`
- Next action: {decision['next_action']}

## Row Summary

{md_table(row_summary)}

## Guardrails

{chr(10).join(f'- {x}' for x in decision.get('forbidden_claims', []))}

## Outputs

- Row summary: `{rows_path}`
- Query rows: `{query_path}`
- JSON: `{json_path}`
"""
    md_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--strict-decision-json", type=Path, default=DEFAULT_STRICT_DECISION_JSON)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    exit_code = load_exit_code(args.run_root)
    panel_json = load_json(args.panel_dir / "zscape_crossfit_specificity_gate_20260628.json")
    row_summary = read_csv(args.panel_dir / "zscape_crossfit_specificity_row_summary.csv") if exit_code == "0" else pd.DataFrame()
    query_rows = read_csv(args.panel_dir / "zscape_crossfit_specificity_query_rows.csv") if exit_code == "0" else pd.DataFrame()
    expected_row_ids = expected_rows_from_strict(args.strict_decision_json)
    decision = build_decision(row_summary, query_rows, panel_json, exit_code, expected_row_ids)
    write_outputs(row_summary, query_rows, decision, args)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
