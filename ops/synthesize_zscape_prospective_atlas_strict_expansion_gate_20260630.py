#!/usr/bin/env python3
"""Gate prospective ZSCAPE atlas rows for later strict matched-null expansion."""

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
DEFAULT_ATLAS_ROWS = (
    REPORTS
    / "zscape_prospective_expansion_dynamic_pairability_atlas_20260630"
    / "zscape_dynamic_pairability_atlas_rows_20260630.csv"
)
DEFAULT_RUN = (
    ROOT
    / "runs/zscape_prospective_expansion_extract_atlas_20260630"
    / "zscape_prospective_expansion_extract_atlas_20260630_0435"
)
OUT_DIR = REPORTS / "zscape_prospective_atlas_strict_expansion_gate_20260630"


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


def load_exit_code(run_root: Path) -> str:
    path = run_root / "EXIT_CODE"
    if not path.exists():
        return "still_running"
    return path.read_text(encoding="utf-8").strip()


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def build_rows(atlas: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    work = atlas.copy()
    for col in [
        "within_state_pairability_score",
        "same_substate_pair_fraction",
        "composition_norm_fraction_of_centroid",
        "within_substate_residual_fraction_of_centroid",
        "n_pseudo_pairs",
        "centroid_response_norm",
        "magnitude_pairability_ratio",
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    status_ok = work.get("status", pd.Series("", index=work.index)).astype(str).eq("ok")
    row_candidate = (
        status_ok
        & (work["n_pseudo_pairs"] >= args.min_pseudo_pairs)
        & (work["within_state_pairability_score"] >= args.min_pairability)
        & (work["same_substate_pair_fraction"] >= args.min_same_substate)
        & (work["composition_norm_fraction_of_centroid"] <= args.max_composition_fraction)
    )
    work["strict_expansion_candidate"] = row_candidate
    work["candidate_reason"] = ""
    work.loc[~status_ok, "candidate_reason"] = "status_not_ok"
    work.loc[
        status_ok & (work["n_pseudo_pairs"] < args.min_pseudo_pairs),
        "candidate_reason",
    ] = "too_few_pseudo_pairs"
    work.loc[
        status_ok & (work["within_state_pairability_score"] < args.min_pairability),
        "candidate_reason",
    ] = "pairability_below_threshold"
    work.loc[
        status_ok & (work["same_substate_pair_fraction"] < args.min_same_substate),
        "candidate_reason",
    ] = "same_substate_below_threshold"
    work.loc[
        status_ok & (work["composition_norm_fraction_of_centroid"] > args.max_composition_fraction),
        "candidate_reason",
    ] = "composition_fraction_too_high"
    work.loc[work["strict_expansion_candidate"], "candidate_reason"] = (
        "high_within_state_pairability_low_composition_candidate"
    )
    return work.sort_values(
        ["strict_expansion_candidate", "within_state_pairability_score", "same_substate_pair_fraction"],
        ascending=[False, False, False],
    )


def decide(rows: pd.DataFrame, args: argparse.Namespace, exit_code: str) -> dict[str, Any]:
    if exit_code == "still_running":
        return {
            "status": "zscape_prospective_atlas_strict_expansion_awaiting_atlas",
            "ready": False,
            "gpu_authorized_next": False,
            "next_action": "wait for the detached atlas job at normal long-job cadence",
        }
    if exit_code != "0":
        return {
            "status": "zscape_prospective_atlas_strict_expansion_blocked_failed_atlas",
            "ready": False,
            "gpu_authorized_next": False,
            "exit_code": exit_code,
            "next_action": "inspect extraction/atlas log before any strict expansion",
        }
    cand = rows[rows["strict_expansion_candidate"]].copy()
    lineages = int(cand["lineage"].nunique()) if not cand.empty and "lineage" in cand else 0
    targets = int(cand["target"].nunique()) if not cand.empty and "target" in cand else 0
    if len(cand) >= args.min_candidate_rows and lineages >= args.min_candidate_lineages:
        status = "zscape_prospective_atlas_strict_expansion_ready_cpu_only"
        next_action = (
            "launch a separate CPU-only strict matched-null expansion over the "
            "top prospective atlas candidates; no GPU/model route"
        )
    elif len(cand) > 0:
        status = "zscape_prospective_atlas_strict_expansion_partial_design_review"
        next_action = (
            "review candidates manually or relax prospectively before strict controls; "
            "do not train"
        )
    else:
        status = "zscape_prospective_atlas_strict_expansion_no_candidate_close_rescue"
        next_action = "close current second-lineage rescue; retain ZSCAPE as descriptor/failure-map"
    return {
        "status": status,
        "ready": status.endswith("ready_cpu_only"),
        "gpu_authorized_next": False,
        "exit_code": exit_code,
        "candidate_rows": int(len(cand)),
        "candidate_lineages": lineages,
        "candidate_targets": targets,
        "candidate_row_ids": cand["row_id"].astype(str).head(args.max_candidates_out).tolist(),
        "thresholds": {
            "min_pseudo_pairs": args.min_pseudo_pairs,
            "min_pairability": args.min_pairability,
            "min_same_substate": args.min_same_substate,
            "max_composition_fraction": args.max_composition_fraction,
            "min_candidate_rows": args.min_candidate_rows,
            "min_candidate_lineages": args.min_candidate_lineages,
        },
        "next_action": next_action,
    }


def fmt(value: Any, digits: int = 4) -> str:
    x = finite_float(value)
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


def md_table(df: pd.DataFrame, n: int = 40) -> str:
    if df.empty:
        return "_None._"
    cols = [
        "row_id",
        "lineage",
        "target",
        "timepoint",
        "n_pseudo_pairs",
        "within_state_pairability_score",
        "same_substate_pair_fraction",
        "composition_norm_fraction_of_centroid",
        "strict_expansion_candidate",
        "candidate_reason",
    ]
    keep = [c for c in cols if c in df.columns]
    lines = ["| " + " | ".join(keep) + " |", "| " + " | ".join(["---"] * len(keep)) + " |"]
    for _, row in df[keep].head(n).iterrows():
        vals = []
        for col in keep:
            val = row.get(col)
            vals.append(fmt(val) if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_outputs(rows: pd.DataFrame, decision: dict[str, Any], args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "zscape_prospective_atlas_strict_expansion_rows_20260630.csv"
    cand_path = args.out_dir / "zscape_prospective_atlas_strict_expansion_candidates_20260630.csv"
    json_path = args.out_dir / "zscape_prospective_atlas_strict_expansion_gate_20260630.json"
    md_path = args.out_dir / "LATENTFM_ZSCAPE_PROSPECTIVE_ATLAS_STRICT_EXPANSION_GATE_20260630.md"
    rows.to_csv(rows_path, index=False)
    rows[rows.get("strict_expansion_candidate", pd.Series(False, index=rows.index)).map(truthy)].to_csv(
        cand_path, index=False
    )
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
            "atlas_rows": str(args.atlas_rows),
            "run_root": str(args.run_root),
        },
        "decision": decision,
        "outputs": {
            "rows": str(rows_path),
            "candidates": str(cand_path),
            "markdown_report": str(md_path),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    text = f"""# ZSCAPE Prospective Atlas Strict-Expansion Gate

## Boundary

- CPU/report-only decision over a completed prospective dynamic pairability atlas.
- Does not run new OT, train, infer, select checkpoints, use canonical multi, access Track C query, or use GPU.

## Decision

- Status: `{decision['status']}`
- Ready: `{decision.get('ready')}`
- GPU authorized next: `{decision['gpu_authorized_next']}`
- Candidate rows: `{decision.get('candidate_rows', 'NA')}`
- Candidate lineages: `{decision.get('candidate_lineages', 'NA')}`
- Candidate targets: `{decision.get('candidate_targets', 'NA')}`
- Next action: {decision['next_action']}

## Candidate Table

{md_table(rows)}

## Interpretation

A pass here only authorizes a later CPU-only strict matched-null expansion on
the frozen candidate rows. It does not authorize LatentFM/RawFM training or a
biological claim by itself.

## Outputs

- Rows: `{rows_path}`
- Candidates: `{cand_path}`
- JSON: `{json_path}`
"""
    md_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas-rows", type=Path, default=DEFAULT_ATLAS_ROWS)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--min-pseudo-pairs", type=int, default=64)
    parser.add_argument("--min-pairability", type=float, default=0.70)
    parser.add_argument("--min-same-substate", type=float, default=0.70)
    parser.add_argument("--max-composition-fraction", type=float, default=0.50)
    parser.add_argument("--min-candidate-rows", type=int, default=8)
    parser.add_argument("--min-candidate-lineages", type=int, default=3)
    parser.add_argument("--max-candidates-out", type=int, default=24)
    args = parser.parse_args()
    exit_code = load_exit_code(args.run_root)
    if args.atlas_rows.exists() and exit_code == "0":
        rows = build_rows(pd.read_csv(args.atlas_rows), args)
    else:
        rows = pd.DataFrame()
    decision = decide(rows, args, exit_code)
    write_outputs(rows, decision, args)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
