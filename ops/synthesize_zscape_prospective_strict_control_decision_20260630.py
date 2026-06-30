#!/usr/bin/env python3
"""Decision wrapper for prospective ZSCAPE strict-control expansion."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
DEFAULT_RUN = (
    ROOT
    / "runs/zscape_prospective_strict_control_expansion_20260630"
    / "zscape_prospective_strict_control_expansion_20260630_0455"
)
DEFAULT_STRICT_DIR = REPORTS / "zscape_prospective_strict_control_expansion_20260630"
DEFAULT_CANDIDATES = (
    REPORTS
    / "zscape_prospective_atlas_strict_expansion_gate_20260630"
    / "zscape_prospective_atlas_strict_expansion_rows_20260630.csv"
)
OUT_DIR = REPORTS / "zscape_prospective_strict_control_decision_20260630"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


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


def load_exit_code(run_root: Path) -> str:
    path = run_root / "EXIT_CODE"
    if not path.exists():
        return "still_running"
    return path.read_text(encoding="utf-8").strip()


def ensure_output_dir(path: Path, force: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = [p for p in path.iterdir() if p.name != ".DS_Store"]
    if existing and not force:
        raise SystemExit(f"Refusing to overwrite nonempty output directory: {path}")


def counts_dict(df: pd.DataFrame, col: str) -> dict[str, int]:
    if df.empty or col not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[col].astype(str).value_counts(dropna=False).to_dict().items()}


def dominance_summary(df: pd.DataFrame, col: str) -> dict[str, Any]:
    counts = counts_dict(df, col)
    if not counts:
        return {"top": None, "top_count": 0, "top_fraction": None, "counts": {}}
    top, top_count = max(counts.items(), key=lambda item: item[1])
    total = sum(counts.values())
    return {
        "top": top,
        "top_count": int(top_count),
        "top_fraction": float(top_count / total) if total else None,
        "counts": counts,
    }


def build_risk_flags(pass_rows: pd.DataFrame) -> dict[str, Any]:
    target_dom = dominance_summary(pass_rows, "gene_target")
    time_dom = dominance_summary(pass_rows, "timepoint")
    lineage_dom = dominance_summary(pass_rows, "cell_type_broad")
    lineage_series = (
        pass_rows["cell_type_broad"].astype(str)
        if "cell_type_broad" in pass_rows.columns
        else pd.Series(dtype=str)
    )
    connective_mask = lineage_series.str.lower().str.contains("connective", regex=False)
    ambiguous_rows = (
        pass_rows.loc[connective_mask, "row_id"].astype(str).tolist()
        if "row_id" in pass_rows.columns and len(connective_mask) == len(pass_rows)
        else []
    )
    high_target_dominance = bool((target_dom.get("top_fraction") or 0.0) >= 0.75 and len(pass_rows) >= 4)
    high_time_dominance = bool((time_dom.get("top_fraction") or 0.0) >= 0.75 and len(pass_rows) >= 4)
    return {
        "target_dominance": target_dom,
        "timepoint_dominance": time_dom,
        "lineage_dominance": lineage_dom,
        "high_target_dominance": high_target_dominance,
        "high_timepoint_dominance": high_time_dominance,
        "connective_taxonomy_rows": ambiguous_rows,
        "manual_review_required": bool(high_target_dominance or high_time_dominance or ambiguous_rows),
        "interpretation_note": (
            "risk flags do not change the predeclared row/lineage design gate, "
            "but they block broad biological claims before independent specificity controls"
        ),
    }


def candidate_row_ids(candidates: pd.DataFrame) -> list[str]:
    if candidates.empty or "row_id" not in candidates.columns:
        return []
    work = candidates.copy()
    if "strict_expansion_candidate" in work.columns:
        work = work[work["strict_expansion_candidate"].map(truthy)]
    return work["row_id"].astype(str).tolist()


def build_decision(
    rows: pd.DataFrame,
    summary: dict[str, Any],
    exit_code: str,
    args: argparse.Namespace,
    candidates: pd.DataFrame,
) -> dict[str, Any]:
    if exit_code == "still_running":
        return {
            "status": "zscape_prospective_strict_control_decision_awaiting_run",
            "ready": False,
            "gpu_authorized_next": False,
            "next_action": "wait for the strict-control job at normal long-job cadence",
        }
    if exit_code != "0":
        return {
            "status": "zscape_prospective_strict_control_failed_runtime",
            "ready": False,
            "exit_code": exit_code,
            "gpu_authorized_next": False,
            "next_action": "inspect strict-control log; do not interpret candidates",
        }
    if rows.empty:
        return {
            "status": "zscape_prospective_strict_control_no_rows_fail_close",
            "ready": True,
            "exit_code": exit_code,
            "gpu_authorized_next": False,
            "next_action": "close prospective second-lineage rescue",
        }
    work = rows.copy()
    expected_rows = candidate_row_ids(candidates)
    observed_rows = work["row_id"].astype(str).tolist() if "row_id" in work.columns else []
    missing_candidate_rows = sorted(set(expected_rows) - set(observed_rows)) if expected_rows else []
    unexpected_rows = sorted(set(observed_rows) - set(expected_rows)) if expected_rows else []
    provenance_validation = {
        "candidate_rows": expected_rows,
        "observed_strict_rows": observed_rows,
        "missing_candidate_rows": missing_candidate_rows,
        "unexpected_rows": unexpected_rows,
        "strict_rows_match_candidates": bool(expected_rows and not missing_candidate_rows and not unexpected_rows),
    }
    if expected_rows and (missing_candidate_rows or unexpected_rows):
        return {
            "status": "zscape_prospective_strict_control_row_id_mismatch_blocked",
            "ready": False,
            "exit_code": exit_code,
            "gpu_authorized_next": False,
            "provenance_validation": provenance_validation,
            "next_action": "fix strict-row/candidate provenance before interpreting this branch",
        }
    work["strict_row_gate_bool"] = work.get("strict_row_gate", False).map(truthy)
    pass_rows = work[work["strict_row_gate_bool"]].copy()
    pass_lineages = pass_rows["cell_type_broad"].astype(str).nunique() if not pass_rows.empty else 0
    pass_targets = pass_rows["gene_target"].astype(str).nunique() if not pass_rows.empty else 0
    risk_flags = build_risk_flags(pass_rows)
    if len(pass_rows) >= args.min_pass_rows and pass_lineages >= args.min_pass_lineages:
        status = "zscape_prospective_strict_control_pass_design_review_no_gpu"
        next_action = (
            "design a stricter biological atlas expansion and independent specificity panel "
            "with wrong-target, wrong-time, wrong-lineage, and taxonomy review; still no "
            "LatentFM/RawFM model route"
        )
    elif len(pass_rows) > 0:
        status = "zscape_prospective_strict_control_partial_signal_fail_design_gate_no_gpu"
        next_action = (
            "record partial lineage-local signals; do not claim broad second-lineage pairability "
            "or launch model training"
        )
    else:
        status = "zscape_prospective_strict_control_fail_close_second_lineage_no_gpu"
        next_action = "close current prospective second-lineage rescue"
    return {
        "status": status,
        "ready": True,
        "exit_code": exit_code,
        "gpu_authorized_next": False,
        "strict_rows_evaluated": int(len(work)),
        "strict_rows_passing": int(len(pass_rows)),
        "passing_lineages": int(pass_lineages),
        "passing_targets": int(pass_targets),
        "passing_row_ids": pass_rows["row_id"].astype(str).tolist(),
        "passing_lineage_counts": counts_dict(pass_rows, "cell_type_broad"),
        "passing_target_counts": counts_dict(pass_rows, "gene_target"),
        "passing_timepoint_counts": counts_dict(pass_rows, "timepoint"),
        "risk_flags": risk_flags,
        "provenance_validation": provenance_validation,
        "audit_summary_status": summary.get("status"),
        "allowed_claims_if_pass": [
            "prospective non-connective ZSCAPE pairability/design-review signal",
            "candidate biological/scaling descriptor after additional specificity controls",
        ],
        "forbidden_claims": [
            "no LatentFM/RawFM loss, sampling, checkpoint, model-positive, or constraint use",
            "no true lineage trajectory claim from snapshot pseudo-pairs",
            "no canonical multi or Track C query use",
            "no broad mechanism/pathway claim before independent specificity controls",
            "no broad non-connective claim if passing rows are target/time dominated or taxonomy ambiguous",
        ],
        "next_action": next_action,
    }


def fmt(value: Any, digits: int = 4) -> str:
    x = finite_float(value)
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_None._"
    cols = [
        "row_id",
        "cell_type_broad",
        "gene_target",
        "timepoint",
        "observed_strict_ot",
        "cc_null_p95",
        "label_null_p95",
        "effect_ratio_vs_max_null_p95",
        "p_observed_le_matched_cc_null",
        "p_observed_le_matched_label_null",
        "matched_subtype_jsd",
        "expression_library_smd",
        "strict_row_gate",
    ]
    keep = [c for c in cols if c in df.columns]
    lines = ["| " + " | ".join(keep) + " |", "| " + " | ".join(["---"] * len(keep)) + " |"]
    for _, row in df[keep].iterrows():
        vals = []
        for col in keep:
            val = row.get(col)
            vals.append(fmt(val) if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_outputs(rows: pd.DataFrame, decision: dict[str, Any], args: argparse.Namespace) -> None:
    ensure_output_dir(args.out_dir, args.force)
    rows_path = args.out_dir / "zscape_prospective_strict_control_decision_rows_20260630.csv"
    json_path = args.out_dir / "zscape_prospective_strict_control_decision_20260630.json"
    md_path = args.out_dir / "LATENTFM_ZSCAPE_PROSPECTIVE_STRICT_CONTROL_DECISION_20260630.md"
    rows.to_csv(rows_path, index=False)
    payload = {
        "timestamp_cst": now_cst(),
        "boundary": {
            "cpu_report_only": True,
            "new_ot_pairing": False,
            "training_or_inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection": False,
            "trackc_query_access": False,
            "gpu_authorized_next": False,
        },
        "inputs": {
            "run_root": str(args.run_root),
            "strict_dir": str(args.strict_dir),
        },
        "decision": decision,
        "outputs": {
            "rows": str(rows_path),
            "markdown_report": str(md_path),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    text = f"""# ZSCAPE Prospective Strict-Control Decision

## Boundary

- CPU/report-only synthesis over the prospective strict-control expansion.
- No new OT, training, inference, checkpoint selection, canonical multi selection, Track C query access, or GPU.

## Decision

- Status: `{decision['status']}`
- Ready: `{decision.get('ready')}`
- GPU authorized next: `{decision['gpu_authorized_next']}`
- Strict rows evaluated: `{decision.get('strict_rows_evaluated', 'NA')}`
- Strict rows passing: `{decision.get('strict_rows_passing', 'NA')}`
- Passing lineages: `{decision.get('passing_lineages', 'NA')}`
- Passing targets: `{decision.get('passing_targets', 'NA')}`
- Strict rows match candidate rows: `{(decision.get('provenance_validation') or {}).get('strict_rows_match_candidates', 'NA')}`
- Next action: {decision['next_action']}

## Audit Risk Flags

- Passing lineage counts: `{decision.get('passing_lineage_counts', {})}`
- Passing target counts: `{decision.get('passing_target_counts', {})}`
- Passing timepoint counts: `{decision.get('passing_timepoint_counts', {})}`
- Manual review required: `{(decision.get('risk_flags') or {}).get('manual_review_required', 'NA')}`
- High target dominance: `{(decision.get('risk_flags') or {}).get('high_target_dominance', 'NA')}`
- High timepoint dominance: `{(decision.get('risk_flags') or {}).get('high_timepoint_dominance', 'NA')}`
- Connective/taxonomy-ambiguous rows: `{(decision.get('risk_flags') or {}).get('connective_taxonomy_rows', [])}`
- Note: {(decision.get('risk_flags') or {}).get('interpretation_note', 'NA')}

## Row Table

{md_table(rows)}

## Guardrails

Forbidden claims:

{chr(10).join(f'- {x}' for x in decision.get('forbidden_claims', []))}

## Outputs

- Rows: `{rows_path}`
- JSON: `{json_path}`
"""
    md_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--strict-dir", type=Path, default=DEFAULT_STRICT_DIR)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--min-pass-rows", type=int, default=4)
    parser.add_argument("--min-pass-lineages", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    exit_code = load_exit_code(args.run_root)
    strict_json = args.strict_dir / "zscape_pairability_strict_control_expansion_20260630.json"
    strict_rows = args.strict_dir / "zscape_pairability_strict_control_expansion_rows_20260630.csv"
    summary = load_json(strict_json).get("summary", {})
    rows = pd.read_csv(strict_rows) if strict_rows.exists() and exit_code == "0" else pd.DataFrame()
    candidates = pd.read_csv(args.candidates) if args.candidates.exists() and exit_code == "0" else pd.DataFrame()
    decision = build_decision(rows, summary, exit_code, args, candidates)
    write_outputs(rows, decision, args)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
