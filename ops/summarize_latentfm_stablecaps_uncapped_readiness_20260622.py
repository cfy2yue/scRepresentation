#!/usr/bin/env python3
"""Summarize capped/stablecaps bootstrap into an uncapped-readiness decision.

This is a triage report only. Stablecaps outputs do not include the formal
cross-background single-gene stratum, so this script can only decide whether a
candidate deserves condition-uncapped canonical posthoc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


KEY_ROWS = (
    ("split", "test", "pearson_pert", "support"),
    ("split", "test", "test_mmd_clamped", "no_harm"),
    ("split", "test_single", "pearson_pert", "single_support"),
    ("split", "test_single", "test_mmd_clamped", "single_no_harm"),
    ("family", "family_gene", "pearson_pert", "family_no_harm"),
    ("family", "family_gene", "test_mmd_clamped", "family_mmd_no_harm"),
    ("split", "test_multi_unseen2", "pearson_pert", "diagnostic"),
    ("split", "test_multi_unseen2", "test_mmd_clamped", "diagnostic_mmd"),
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def find_output(index: dict[str, Any], kind: str) -> Path | None:
    for row in index.get("outputs") or []:
        if row.get("kind") == kind and row.get("json"):
            p = Path(row["json"])
            if p.is_file():
                return p
    return None


def row_by(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for row in payload.get("rows") or []:
        out[(str(row.get("group")), str(row.get("metric")))] = row
    return out


def as_float(row: dict[str, Any], key: str, default: float) -> float:
    value = row.get(key)
    if value is None:
        return default
    return float(value)


def build_decision(rows: dict[tuple[str, str, str], dict[str, Any] | None]) -> dict[str, Any]:
    reasons: list[str] = []
    test_pp = rows.get(("split", "test", "pearson_pert")) or {}
    test_mmd = rows.get(("split", "test", "test_mmd_clamped")) or {}
    single_pp = rows.get(("split", "test_single", "pearson_pert")) or {}
    single_mmd = rows.get(("split", "test_single", "test_mmd_clamped")) or {}
    family_pp = rows.get(("family", "family_gene", "pearson_pert")) or {}
    family_mmd = rows.get(("family", "family_gene", "test_mmd_clamped")) or {}

    for label, row in (
        ("test_pp", test_pp),
        ("test_mmd", test_mmd),
        ("single_pp", single_pp),
        ("single_mmd", single_mmd),
        ("family_pp", family_pp),
        ("family_mmd", family_mmd),
    ):
        if not row or row.get("status") != "ok":
            reasons.append(f"missing_or_bad_{label}")

    if not reasons:
        if as_float(test_pp, "p_harm", 1.0) > 0.35:
            reasons.append("test_pp_harm_risk")
        if as_float(test_mmd, "p_harm", 1.0) > 0.80:
            reasons.append("test_mmd_hard_harm")
        if as_float(single_pp, "p_improvement", 0.0) < 0.75 and as_float(single_pp, "delta_mean", 0.0) <= 0:
            reasons.append("test_single_pp_not_positive")
        if as_float(single_mmd, "p_harm", 1.0) > 0.80:
            reasons.append("test_single_mmd_hard_harm")
        if as_float(family_pp, "p_harm", 1.0) > 0.35:
            reasons.append("family_gene_pp_harm_risk")
        if as_float(family_mmd, "p_harm", 1.0) > 0.80:
            reasons.append("family_gene_mmd_hard_harm")

    if not reasons:
        status = "stablecaps_ready_for_uncapped_posthoc"
        action = "run_condition_uncapped_single_background_gate"
    elif all(r.endswith("_risk") for r in reasons):
        status = "stablecaps_diagnostic_or_near_miss"
        action = "review_manually_before_uncapped"
    else:
        status = "stablecaps_fail_close_branch"
        action = "close_or_wait_for_other_candidate"
    return {"status": status, "action": action, "reasons": reasons}


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Stablecaps Uncapped-Readiness Decision",
        "",
        f"Candidate: `{payload['label']}`",
        f"Status: `{payload['decision']['status']}`",
        f"Recommended action: `{payload['decision']['action']}`",
        "",
        "## Provenance",
        "",
        f"- bootstrap index: `{payload['bootstrap_index']}`",
        f"- split bootstrap: `{payload.get('split_bootstrap_json')}`",
        f"- family bootstrap: `{payload.get('family_bootstrap_json')}`",
        "- scope: capped/stablecaps triage only; not a promotion claim.",
        "- limitation: formal cross-background stratum requires condition-uncapped posthoc and `audit_latentfm_xverse_single_background_candidate_20260622.py`.",
        "",
        "## Key Rows",
        "",
        "| role | source | group | metric | n cond | n ds | delta | 95% CI | p improve | p harm | status |",
        "|---|---|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for source, group, metric, role in KEY_ROWS:
        row = payload["rows"].get(f"{source}:{group}:{metric}") or {}
        ci = [row.get("ci95_low"), row.get("ci95_high")]
        lines.append(
            "| {role} | {source} | {group} | {metric} | {n} | {nds} | {delta} | [{lo}, {hi}] | {pi} | {ph} | {status} |".format(
                role=role,
                source=source,
                group=group,
                metric=metric,
                n=row.get("n_matched_conditions", 0),
                nds=row.get("n_matched_datasets", 0),
                delta=fmt(row.get("delta_mean")),
                lo=fmt(ci[0]),
                hi=fmt(ci[1]),
                pi=fmt(row.get("p_improvement")),
                ph=fmt(row.get("p_harm")),
                status=row.get("status", "missing"),
            )
        )
    lines += ["", "## Reasons", ""]
    reasons = payload["decision"].get("reasons") or []
    if reasons:
        lines.extend(f"- `{r}`" for r in reasons)
    else:
        lines.append("- none")
    lines += [
        "",
        "## Next Rule",
        "",
        "- If status is `stablecaps_ready_for_uncapped_posthoc`, run condition-uncapped canonical posthoc and then the single/background candidate gate.",
        "- If status is `stablecaps_fail_close_branch`, do not spend more GPU on this branch unless new train-only CPU evidence appears.",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap-index", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    args = ap.parse_args()

    index = load_json(args.bootstrap_index)
    split_path = find_output(index, "split")
    family_path = find_output(index, "family")
    split_rows = row_by(load_json(split_path)) if split_path else {}
    family_rows = row_by(load_json(family_path)) if family_path else {}
    selected: dict[tuple[str, str, str], dict[str, Any] | None] = {}
    flat_rows: dict[str, dict[str, Any] | None] = {}
    for source, group, metric, _role in KEY_ROWS:
        table = split_rows if source == "split" else family_rows
        row = table.get((group, metric))
        selected[(source, group, metric)] = row
        flat_rows[f"{source}:{group}:{metric}"] = row

    decision = build_decision(selected)
    payload = {
        "label": args.label,
        "bootstrap_index": str(args.bootstrap_index),
        "split_bootstrap_json": None if split_path is None else str(split_path),
        "family_bootstrap_json": None if family_path is None else str(family_path),
        "decision": decision,
        "rows": flat_rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
