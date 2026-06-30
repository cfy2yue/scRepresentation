#!/usr/bin/env python3
"""CPU adjudication gate for Track C both_train_multi_gene conflict.

This gate resolves the conflict between target-stratum pass and whole-support
seed45 materiality failure using existing safe trainselect support-val reports.
It does not read held-out Track C query, canonical metrics, or canonical multi,
and it does not launch GPU work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_trackc_both_train_multi_gene_adjudication_gate_20260624.json"
OUT_MD = REPORTS / "LATENTFM_TRACKC_BOTH_TRAIN_MULTI_GENE_ADJUDICATION_GATE_20260624.md"

RUNS = {
    43: "latentfm_trackc_support_only_robustness_decision_xverse_trackc_support_pairtype_both_train_multi_gene_resfilm_ep050_replay2_2k_seed43.json",
    44: "latentfm_trackc_support_only_robustness_decision_xverse_trackc_support_pairtype_both_train_multi_gene_resfilm_ep050_replay2_2k_seed44.json",
    45: "latentfm_trackc_support_only_robustness_decision_xverse_trackc_support_pairtype_both_train_multi_gene_resfilm_ep050_replay2_2k_seed45.json",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def key(report: dict[str, Any], name: str) -> dict[str, Any]:
    return ((report.get("decision") or {}).get("key_rows") or {}).get(name) or {}


def whole_pass(report: dict[str, Any]) -> bool:
    dec = report.get("decision") or {}
    action = str(dec.get("action") or "")
    actual_pp = key(report, "actual_pp").get("delta_mean")
    if actual_pp is None:
        return False
    return "prepare external review" in action and float(actual_pp) >= 0.04


def hard_fail(report: dict[str, Any]) -> bool:
    dec = report.get("decision") or {}
    action = str(dec.get("action") or "")
    actual_pp = key(report, "actual_pp").get("delta_mean")
    return "close fixed support-only robustness branch" in action or (actual_pp is not None and float(actual_pp) < 0.04)


def main() -> int:
    reports = {seed: load_json(REPORTS / name) for seed, name in RUNS.items()}
    summary = load_json(REPORTS / "latentfm_trackc_support_only_pairtype_strata_summary_20260624_both_train_multi_gene.json")
    rows = []
    for seed, report in reports.items():
        rows.append(
            {
                "seed": seed,
                "whole_action": (report.get("decision") or {}).get("action"),
                "whole_pass": whole_pass(report),
                "whole_hard_or_materiality_fail": hard_fail(report),
                "actual_pp": key(report, "actual_pp").get("delta_mean"),
                "actual_mmd": key(report, "actual_mmd").get("delta_mean"),
                "zero_pp": key(report, "zero_pp").get("delta_mean"),
                "shuffle_pp": key(report, "shuffle_pp").get("delta_mean"),
                "absent_pp": key(report, "absent_pp").get("delta_mean"),
                "family_pp": key(report, "family_pp").get("delta_mean"),
                "family_mmd": key(report, "family_mmd").get("delta_mean"),
            }
        )

    n_whole_pass = sum(1 for row in rows if row["whole_pass"])
    n_whole_fail = sum(1 for row in rows if row["whole_hard_or_materiality_fail"])
    summary_status = summary.get("stability_status") or summary.get("status")
    target_next = summary.get("next_action")

    reasons = []
    if n_whole_pass < 3:
        reasons.append("whole_support_not_3_of_3")
    if n_whole_fail:
        reasons.append("at_least_one_seed_whole_support_materiality_fail")
    if str(summary_status) == "pass_2_of_3_no_hard_fail" and n_whole_fail:
        reasons.append("target_summary_conflicts_with_whole_support_fail_close")
    if target_next and "canonical" in str(target_next) and n_whole_fail:
        reasons.append("canonical_noharm_next_action_depends_on_target_stratum_override")

    status = "trackc_both_train_multi_gene_adjudication_fail_no_gpu"
    if not reasons:
        status = "trackc_both_train_multi_gene_adjudication_pass_external_review_next"

    payload = {
        "status": status,
        "gpu_authorized": status.endswith("_next"),
        "boundary": {
            "reads_safe_trainselect_support_reports": True,
            "reads_canonical_metrics": False,
            "reads_canonical_multi": False,
            "reads_heldout_trackc_query": False,
            "launches_gpu": False,
            "safe_split": "/data/cyx/1030/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json",
        },
        "summary": {
            "n_seeds": len(rows),
            "n_whole_support_pass": n_whole_pass,
            "n_whole_support_fail": n_whole_fail,
            "target_summary_status": summary_status,
            "target_summary_next_action": target_next,
        },
        "reasons": reasons,
        "rows": rows,
        "next_action": (
            "external review before frozen canonical single/family no-harm veto"
            if status.endswith("_next")
            else "downgrade to mechanism-only or design a predeclared target-population support gate; no canonical no-harm now"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Track C both_train_multi_gene Adjudication Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU/support-only adjudication of existing safe trainselect reports.",
        "- Does not read canonical metrics, canonical multi, held-out Track C query, active logs, train, infer, or use GPU.",
        "",
        "## Seed Matrix",
        "",
        "| seed | whole pass | materiality fail | actual pp | actual MMD | zero pp | shuffle pp | absent pp | action |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | `{row['whole_pass']}` | `{row['whole_hard_or_materiality_fail']}` | "
            f"{fmt(row['actual_pp'])} | {fmt(row['actual_mmd'])} | {fmt(row['zero_pp'])} | "
            f"{fmt(row['shuffle_pp'])} | {fmt(row['absent_pp'])} | {row['whole_action']} |"
        )
    lines.extend(
        [
            "",
            "## Target-Stratum Summary",
            "",
            f"- stability status: `{summary_status}`",
            f"- next action: `{target_next}`",
            "",
            "## Decision",
            "",
            f"- reasons: `{reasons}`",
            f"- GPU authorized: `{payload['gpu_authorized']}`",
            f"- next action: `{payload['next_action']}`",
            "",
            "## Interpretation",
            "",
            "- Target-stratum signal is retained as mechanism evidence.",
            "- Whole-support seed45 materiality failure blocks direct canonical no-harm under the current predeclared hierarchy.",
            "- A future target-population route would need its own predeclared CPU/support-only gate and external review before GPU.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "gpu_authorized": payload["gpu_authorized"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
