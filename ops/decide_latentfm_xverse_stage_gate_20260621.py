#!/usr/bin/env python3
"""Gate decision for the xverse LatentFM stage-result candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_STAGE_JSON = ROOT / "reports/latentfm_xverse_stage_summary_20260621.json"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_stage_gate_decision_20260621.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_STAGE_GATE_DECISION_20260621.md"
LOWER_IS_BETTER = {"test_mmd_clamped", "test_mmd_biased", "test_mmd"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def row_lookup(rows: list[dict[str, Any]], group: str, metric: str) -> dict[str, Any] | None:
    for item in rows:
        row = item.get("row")
        if item.get("group") == group and item.get("metric") == metric and isinstance(row, dict):
            return row
    return None


def ci_positive(row: dict[str, Any] | None) -> bool | None:
    if row is None or row.get("ci95_low") is None:
        return None
    return float(row["ci95_low"]) > 0.0


def paired_improves(row: dict[str, Any] | None) -> bool | None:
    if row is None or row.get("ci95_low") is None or row.get("ci95_high") is None:
        return None
    metric = str(row.get("metric"))
    lo = float(row["ci95_low"])
    hi = float(row["ci95_high"])
    if metric in LOWER_IS_BETTER:
        return hi < 0.0
    return lo > 0.0


def paired_nonharm(row: dict[str, Any] | None) -> bool | None:
    if row is None or row.get("p_harm") is None:
        return None
    return float(row["p_harm"]) <= 0.20


def fmt(value: Any) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def status_from_checks(checks: list[bool | None]) -> str:
    if any(v is False for v in checks):
        return "fail"
    if any(v is None for v in checks):
        return "pending"
    return "pass"


def build_decision(stage: dict[str, Any]) -> dict[str, Any]:
    uncapped = stage.get("uncapped_rows", [])
    uncapped_paired = stage.get("uncapped_paired_rows", [])
    seed43 = stage.get("seed43_rows", [])

    seed42_test_pp = row_lookup(uncapped, "test", "pearson_pert")
    seed42_family_pp = row_lookup(uncapped, "family_gene", "pearson_pert")
    seed42_unseen2_pp = row_lookup(uncapped, "test_multi_unseen2", "pearson_pert")

    paired_test_pp = row_lookup(uncapped_paired, "test", "pearson_pert")
    paired_test_mmd = row_lookup(uncapped_paired, "test", "test_mmd_clamped")
    paired_family_pp = row_lookup(uncapped_paired, "family_gene", "pearson_pert")
    paired_family_mmd = row_lookup(uncapped_paired, "family_gene", "test_mmd_clamped")
    paired_structure_mmd = row_lookup(uncapped_paired, "structure_multi", "test_mmd_clamped")

    seed43_test_pp = row_lookup(seed43, "test", "pearson_pert")
    seed43_family_pp = row_lookup(seed43, "family_gene", "pearson_pert")
    seed43_test_mmd = row_lookup(seed43, "test", "test_mmd_clamped")
    seed43_iid_pp = row_lookup(seed43, "test_full_train_eval", "pearson_pert")
    seed43_iid_mmd = row_lookup(seed43, "test_full_train_eval", "test_mmd_clamped")

    checks = [
        {
            "name": "seed42_test_pp_ci_positive",
            "status": status_from_checks([ci_positive(seed42_test_pp)]),
            "evidence": seed42_test_pp,
        },
        {
            "name": "seed42_family_gene_pp_ci_positive",
            "status": status_from_checks([ci_positive(seed42_family_pp)]),
            "evidence": seed42_family_pp,
        },
        {
            "name": "uncapped_8k_vs_2k_test_pp_improves",
            "status": status_from_checks([paired_improves(paired_test_pp)]),
            "evidence": paired_test_pp,
        },
        {
            "name": "uncapped_8k_vs_2k_test_mmd_nonharm",
            "status": status_from_checks([paired_nonharm(paired_test_mmd)]),
            "evidence": paired_test_mmd,
        },
        {
            "name": "uncapped_8k_vs_2k_family_gene_pp_improves",
            "status": status_from_checks([paired_improves(paired_family_pp)]),
            "evidence": paired_family_pp,
        },
        {
            "name": "uncapped_8k_vs_2k_family_gene_mmd_nonharm",
            "status": status_from_checks([paired_nonharm(paired_family_mmd)]),
            "evidence": paired_family_mmd,
        },
        {
            "name": "uncapped_8k_vs_2k_structure_multi_mmd_nonharm",
            "status": status_from_checks([paired_nonharm(paired_structure_mmd)]),
            "evidence": paired_structure_mmd,
        },
        {
            "name": "seed43_iid_test_pp_ci_positive",
            "status": status_from_checks([ci_positive(seed43_iid_pp)]),
            "evidence": seed43_iid_pp,
        },
        {
            "name": "seed43_iid_test_mmd_available",
            "status": "pending" if seed43_iid_mmd is None else "pass",
            "evidence": seed43_iid_mmd,
        },
        {
            "name": "seed43_test_pp_ci_positive",
            "status": status_from_checks([ci_positive(seed43_test_pp)]),
            "evidence": seed43_test_pp,
        },
        {
            "name": "seed43_family_gene_pp_ci_positive",
            "status": status_from_checks([ci_positive(seed43_family_pp)]),
            "evidence": seed43_family_pp,
        },
        {
            "name": "seed43_test_mmd_available",
            "status": "pending" if seed43_test_mmd is None else "pass",
            "evidence": seed43_test_mmd,
        },
    ]
    claim_status = status_from_checks([c["status"] == "pass" if c["status"] != "pending" else None for c in checks])
    limitation = {
        "name": "unseen2_composition_limitation",
        "status": "limitation_expected",
        "evidence": seed42_unseen2_pp,
        "interpretation": "Do not require unseen2 pp to pass for a narrow xverse stage claim; report it as an unresolved failure mode.",
    }
    return {
        "stage_json": str(DEFAULT_STAGE_JSON),
        "overall_status": claim_status,
        "checks": checks,
        "limitations": [limitation],
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM xverse Stage Gate Decision 2026-06-21",
        "",
        f"Stage JSON: `{payload['stage_json']}`",
        f"Overall status: `{payload['overall_status']}`",
        "",
        "| check | status | metric | mean/delta | 95% CI | p harm |",
        "|---|---|---|---:|---|---:|",
    ]
    for check in payload["checks"]:
        row = check.get("evidence") or {}
        value = row.get("delta_mean", row.get("mean"))
        ci = "pending"
        if row.get("ci95_low") is not None and row.get("ci95_high") is not None:
            ci = f"[{fmt(row.get('ci95_low'))}, {fmt(row.get('ci95_high'))}]"
        lines.append(
            "| {name} | {status} | {metric} | {value} | {ci} | {pharm} |".format(
                name=check["name"],
                status=check["status"],
                metric=row.get("metric", "pending"),
                value=fmt(value),
                ci=ci,
                pharm=fmt(row.get("p_harm")),
            )
        )
    lines += [
        "",
        "## Limitations",
        "",
    ]
    for limitation in payload["limitations"]:
        row = limitation.get("evidence") or {}
        lines.append(
            "- {name}: {interpretation} Evidence metric={metric}, mean={mean}, CI=[{lo}, {hi}].".format(
                name=limitation["name"],
                interpretation=limitation["interpretation"],
                metric=row.get("metric", "pending"),
                mean=fmt(row.get("mean")),
                lo=fmt(row.get("ci95_low")),
                hi=fmt(row.get("ci95_high")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-json", type=Path, default=DEFAULT_STAGE_JSON)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    stage = load_json(args.stage_json)
    payload = build_decision(stage)
    payload["stage_json"] = str(args.stage_json)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_md": str(args.out_md), "overall_status": payload["overall_status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
