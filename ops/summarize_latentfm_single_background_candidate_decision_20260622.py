#!/usr/bin/env python3
"""Render a concise decision report from a single/background candidate gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUT_MD = Path("/data/cyx/1030/scLatent/reports/LATENTFM_SINGLE_BACKGROUND_CANDIDATE_DECISION_20260622.md")

KEY_ROWS = (
    ("cross_background_seen_gene", "pearson_pert", "primary"),
    ("all_test_single", "pearson_pert", "no_harm"),
    ("all_test_single", "test_mmd_clamped", "no_harm"),
    ("family_gene", "pearson_pert", "no_harm"),
    ("family_gene", "test_mmd_clamped", "no_harm"),
    ("globally_unseen_gene", "pearson_pert", "diagnostic"),
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def find_row(payload: dict[str, Any], stratum: str, metric: str) -> dict[str, Any] | None:
    for row in payload.get("paired_deltas", []):
        if row.get("stratum") == stratum and row.get("metric") == metric:
            return row
    return None


def as_float(row: dict[str, Any], key: str, default: float) -> float:
    value = row.get(key)
    if value is None:
        return default
    return float(value)


def near_miss(payload: dict[str, Any]) -> bool:
    if payload.get("gate", {}).get("status") == "candidate_gate_pass":
        return False
    primary = find_row(payload, "cross_background_seen_gene", "pearson_pert")
    if not primary or primary.get("status") != "ok":
        return False
    p_improve = as_float(primary, "p_improve", 0.0)
    delta = as_float(primary, "delta_mean", 0.0)
    no_harm_ok = True
    for stratum, metric, role in KEY_ROWS:
        if role != "no_harm":
            continue
        row = find_row(payload, stratum, metric)
        if not row or row.get("status") != "ok" or as_float(row, "p_harm", 1.0) > 0.35:
            no_harm_ok = False
    return delta > 0.0 and p_improve >= 0.75 and no_harm_ok


def render(payload: dict[str, Any], *, title: str, label: str, gate_json: Path) -> str:
    gate = payload.get("gate", {})
    status = gate.get("status", "unknown")
    if status == "candidate_gate_pass":
        action = "seed_robustness_confirm"
    elif near_miss(payload):
        action = "near_miss_one_targeted_followup_allowed"
    else:
        action = "close_or_wait_for_other_candidate"

    lines = [
        f"# {title}",
        "",
        f"Candidate: `{label}`",
        f"Gate JSON: `{gate_json}`",
        f"Status: `{status}`",
        f"Recommended action: `{action}`",
        "",
        "## Provenance",
        "",
        f"- anchor split JSON: `{payload.get('anchor_split_json')}`",
        f"- candidate split JSON: `{payload.get('candidate_split_json')}`",
        f"- anchor family JSON: `{payload.get('anchor_family_json')}`",
        f"- candidate family JSON: `{payload.get('candidate_family_json')}`",
        f"- bootstrap: `{payload.get('n_boot')}` resamples, seed `{payload.get('seed')}`",
        "- leakage note: this report reads held-out canonical posthoc only after training; it must not be used for checkpoint selection.",
        "",
        "## Gate Rows",
        "",
        "| role | stratum | metric | n cond | n datasets | delta | 95% CI | p improve | p harm | status |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for stratum, metric, role in KEY_ROWS:
        row = find_row(payload, stratum, metric)
        if not row:
            lines.append(f"| {role} | {stratum} | {metric} | 0 | 0 | NA | [NA, NA] | NA | NA | missing |")
            continue
        ci = row.get("ci95") or [None, None]
        lines.append(
            f"| {role} | {stratum} | {metric} | "
            f"{row.get('n_matched_conditions', 0)} | {row.get('n_matched_datasets', 0)} | "
            f"{fmt(row.get('delta_mean'))} | [{fmt(ci[0])}, {fmt(ci[1])}] | "
            f"{fmt(row.get('p_improve'))} | {fmt(row.get('p_harm'))} | {row.get('status', 'NA')} |"
        )
    lines += [
        "",
        "## Gate Reasons",
        "",
    ]
    reasons = gate.get("reasons") or []
    if reasons:
        for reason in reasons:
            lines.append(f"- `{reason}`")
    else:
        lines.append("- none")
    lines += [
        "",
        "## Decision Rules",
        "",
    ]
    for rule in gate.get("rules") or []:
        lines.append(f"- {rule}")
    lines += [
        "",
        "Interpretation:",
        "- `seed_robustness_confirm`: do one seed/anchor robustness run before a strong claim.",
        "- `near_miss_one_targeted_followup_allowed`: allow one predeclared correction, not a sweep.",
        "- `close_or_wait_for_other_candidate`: do not spend more GPU on this branch unless new CPU evidence appears.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-json", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--title", default="LatentFM Single/Background Candidate Decision")
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()
    payload = load_json(args.gate_json)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render(payload, title=args.title, label=args.label, gate_json=args.gate_json), encoding="utf-8")
    print(json.dumps({"out_md": str(args.out_md), "status": payload.get("gate", {}).get("status")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
