#!/usr/bin/env python3
"""CPU target-population support gate for Track C both_train_multi_gene.

This gate explicitly tests a narrower target-population interpretation after a
whole-support materiality failure. Passing this gate authorizes external review
only, not GPU or canonical no-harm by itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_trackc_target_population_support_gate_20260624.json"
OUT_MD = REPORTS / "LATENTFM_TRACKC_TARGET_POPULATION_SUPPORT_GATE_20260624.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def metric(block: dict[str, Any], key: str) -> float | None:
    value = (block.get(key) or {}).get("equal_dataset_mean_delta")
    return float(value) if value is not None else None


def min_dataset(block: dict[str, Any], key: str) -> float | None:
    value = (block.get(key) or {}).get("min_dataset_delta")
    return float(value) if value is not None else None


def max_dataset(block: dict[str, Any], key: str) -> float | None:
    value = (block.get(key) or {}).get("max_dataset_delta")
    return float(value) if value is not None else None


def n_datasets(block: dict[str, Any], key: str) -> int:
    return int((block.get(key) or {}).get("n_datasets") or 0)


def main() -> int:
    summary = load_json(REPORTS / "latentfm_trackc_support_only_pairtype_strata_summary_20260624_both_train_multi_gene.json")
    adjudication = load_json(REPORTS / "latentfm_trackc_both_train_multi_gene_adjudication_gate_20260624.json")
    rows = []
    for run in summary.get("runs") or []:
        target = run.get("target") or {}
        non_target = run.get("non_target") or {}
        actual_pp = metric(target, "actual_pp")
        actual_mmd = metric(target, "actual_mmd")
        zero_pp = metric(target, "zero_pp")
        shuffle_pp = metric(target, "shuffle_pp")
        absent_pp = metric(target, "absent_pp")
        control_max = max([v for v in [zero_pp, shuffle_pp, absent_pp] if v is not None], default=None)
        target_minus_control = actual_pp - control_max if actual_pp is not None and control_max is not None else None
        non_target_pp = metric(non_target, "actual_pp")
        reasons = []
        if n_datasets(target, "actual_pp") < 2:
            reasons.append("target_less_than_2_datasets")
        if actual_pp is None or actual_pp < 0.04:
            reasons.append("target_pp_lt_0p04")
        if min_dataset(target, "actual_pp") is None or min_dataset(target, "actual_pp") < 0.0:
            reasons.append("target_min_dataset_pp_lt_0")
        if actual_mmd is None or actual_mmd > 0.0:
            reasons.append("target_mmd_harm")
        if control_max is None or control_max > 0.02:
            reasons.append("target_control_pp_gt_0p02")
        if target_minus_control is None or target_minus_control < 0.02:
            reasons.append("target_minus_control_lt_0p02")
        if non_target_pp is not None and non_target_pp < -0.02:
            reasons.append("non_target_pp_harm")
        rows.append(
            {
                "run": run.get("run_name"),
                "status": run.get("status"),
                "target_pp": actual_pp,
                "target_mmd": actual_mmd,
                "target_min_dataset_pp": min_dataset(target, "actual_pp"),
                "target_n_datasets": n_datasets(target, "actual_pp"),
                "zero_pp": zero_pp,
                "shuffle_pp": shuffle_pp,
                "absent_pp": absent_pp,
                "control_max": control_max,
                "target_minus_control": target_minus_control,
                "non_target_pp": non_target_pp,
                "reasons": reasons,
                "pass": not reasons,
            }
        )

    failed = [row for row in rows if not row["pass"]]
    whole_fail = adjudication.get("summary", {}).get("n_whole_support_fail", 0)
    reasons = []
    if failed:
        reasons.append("one_or_more_target_population_seed_failed")
    if int(whole_fail or 0) == 0:
        reasons.append("no_whole_support_conflict_to_adjudicate")
    if summary.get("stability_status") != "pass_2_of_3_no_hard_fail":
        reasons.append("target_stratum_summary_not_stable_pass")

    status = "trackc_target_population_support_gate_fail_no_gpu"
    external_review_authorized = False
    if not reasons:
        status = "trackc_target_population_support_gate_pass_external_review_next"
        external_review_authorized = True

    payload = {
        "status": status,
        "gpu_authorized": False,
        "external_review_authorized": external_review_authorized,
        "boundary": {
            "reads_safe_trainselect_support_reports": True,
            "reads_canonical_metrics": False,
            "reads_canonical_multi": False,
            "reads_heldout_trackc_query": False,
            "training_or_inference": False,
            "active_logs_read": False,
            "gpu": False,
            "safe_split": "/data/cyx/1030/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json",
        },
        "gate_rule": {
            "target_pp_min": 0.04,
            "target_min_dataset_pp_min": 0.0,
            "target_mmd_max": 0.0,
            "target_control_pp_max": 0.02,
            "target_minus_control_min": 0.02,
            "non_target_pp_floor": -0.02,
            "requires_all_seeds": True,
            "pass_only_authorizes": "external_review_not_gpu",
        },
        "whole_support_adjudication_status": adjudication.get("status"),
        "whole_support_fail_count": whole_fail,
        "target_summary_status": summary.get("stability_status"),
        "reasons": reasons,
        "rows": rows,
        "next_action": (
            "external review of target-population gate; no GPU before review"
            if external_review_authorized
            else "close/downgrade target-population route; no GPU"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Track C Target-Population Support Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU/support-only target-population adjudication.",
        "- Reads existing safe trainselect support reports only.",
        "- Does not read canonical metrics, canonical multi, held-out Track C query, active logs, train, infer, or use GPU.",
        "- A pass authorizes external review only, not GPU.",
        "",
        "## Seed Rows",
        "",
        "| run | pass | target pp | target MMD | min-ds pp | zero | shuffle | absent | target-control | non-target pp | reasons |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['run']}` | `{row['pass']}` | {fmt(row['target_pp'])} | {fmt(row['target_mmd'])} | "
            f"{fmt(row['target_min_dataset_pp'])} | {fmt(row['zero_pp'])} | {fmt(row['shuffle_pp'])} | "
            f"{fmt(row['absent_pp'])} | {fmt(row['target_minus_control'])} | {fmt(row['non_target_pp'])} | `{row['reasons']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- whole-support adjudication: `{payload['whole_support_adjudication_status']}`",
            f"- whole-support fail count: `{whole_fail}`",
            f"- target summary status: `{payload['target_summary_status']}`",
            f"- reasons: `{reasons}`",
            f"- external review authorized: `{external_review_authorized}`",
            "- GPU authorized: `False`",
            f"- next action: `{payload['next_action']}`",
            "",
            "## Interpretation",
            "",
            "- This gate explicitly narrows the claim from whole-support robustness to target-population support.",
            "- Because whole-support already failed one seed, this gate cannot directly launch canonical no-harm.",
            "- External review must decide whether this narrowed target-population route is scientifically acceptable before any frozen no-harm veto.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "external_review_authorized": external_review_authorized, "gpu_authorized": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
