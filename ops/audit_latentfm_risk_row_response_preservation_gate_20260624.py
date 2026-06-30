#!/usr/bin/env python3
"""CPU-only risk-row response-preservation gate.

The exact risk-row CVaR recipe improved some train-only/internal MMD summaries
but failed frozen canonical Pearson no-harm. This gate checks whether the
completed train-only/internal condition metrics contain a reliable
response-preservation signal that could justify a materially new guarded
risk-row mechanism. It does not read canonical metrics for selection, canonical
multi, Track C query, active logs, or launch GPU work.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
RUN_ROOT = (
    ROOT
    / "runs/latentfm_risk_row_cvar_trainonly_20260624"
    / "xverse_risk_row_cvar_allrisk_w020_2k_seed42"
    / "posthoc_eval_internal"
)
OUT_JSON = REPORTS / "latentfm_risk_row_response_preservation_gate_20260624.json"
OUT_MD = REPORTS / "LATENTFM_RISK_ROW_RESPONSE_PRESERVATION_GATE_20260624.md"

RISK_DATASETS = {
    "Nadig_hepg2",
    "Nadig_jurket",
    "NormanWeissman2019_filtered",
    "ReplogleWeissman2022_K562_gwps",
    "Replogle_RPE1essential",
    "TianActivation",
}
GROUPS = ("internal_val_cross_background_seen_gene_proxy", "internal_val_family_gene_proxy")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(obj: dict[str, Any], group: str) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(r.get("dataset")), str(r.get("condition"))): r
        for r in ((obj.get("groups") or {}).get(group) or {}).get("condition_metrics") or []
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"n": 0}
    pp = [float(r["pp_delta"]) for r in records]
    mmd = [float(r["mmd_delta"]) for r in records]
    return {
        "n": len(records),
        "pp_delta_mean": mean(pp),
        "mmd_delta_mean": mean(mmd),
        "pp_delta_min": min(pp),
        "mmd_delta_max": max(mmd),
        "pp_harm_rows_lt_neg_0p02": sum(v < -0.02 for v in pp),
        "mmd_harm_rows_gt_0p002": sum(v > 0.002 for v in mmd),
        "joint_pp_mmd_harm_rows": sum(r["pp_delta"] < -0.02 and r["mmd_delta"] > 0.002 for r in records),
        "mmd_improve_pp_noharm_rows": sum(r["mmd_delta"] < -0.002 and r["pp_delta"] >= -0.005 for r in records),
    }


def main() -> None:
    required = {
        "anchor": RUN_ROOT / "split_group_eval_anchor_internal_ode20.json",
        "candidate": RUN_ROOT / "split_group_eval_candidate_internal_ode20.json",
        "internal_decision": REPORTS / "latentfm_risk_row_cvar_internal_posthoc_decision_20260624.json",
        "canonical_decision_context": REPORTS / "latentfm_risk_row_cvar_canonical_noharm_decision_20260624.json",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    anchor = load(required["anchor"]) if not missing else {}
    candidate = load(required["candidate"]) if not missing else {}
    by_group: dict[str, Any] = {}
    reasons: list[str] = []
    if missing:
        reasons.append("required_artifact_missing")

    for group in GROUPS:
        a = rows(anchor, group) if anchor else {}
        c = rows(candidate, group) if candidate else {}
        matched = []
        for key in sorted(a.keys() & c.keys()):
            ds, cond = key
            matched.append(
                {
                    "dataset": ds,
                    "condition": cond,
                    "is_risk": ds in RISK_DATASETS,
                    "pp_delta": float(c[key].get("pearson_pert", 0.0)) - float(a[key].get("pearson_pert", 0.0)),
                    "mmd_delta": float(c[key].get("test_mmd_clamped", 0.0))
                    - float(a[key].get("test_mmd_clamped", 0.0)),
                }
            )
        group_summary = {
            "all": summarize(matched),
            "risk": summarize([r for r in matched if r["is_risk"]]),
            "nonrisk": summarize([r for r in matched if not r["is_risk"]]),
            "top_joint_harm": sorted(
                [r for r in matched if r["pp_delta"] < -0.02 or r["mmd_delta"] > 0.002],
                key=lambda r: (r["pp_delta"], -r["mmd_delta"]),
            )[:12],
        }
        by_group[group] = group_summary

        risk = group_summary["risk"]
        nonrisk = group_summary["nonrisk"]
        if risk.get("pp_delta_min", 0.0) < -0.02:
            reasons.append(f"{group}_risk_pp_tail_harm")
        if risk.get("joint_pp_mmd_harm_rows", 0) > 0:
            reasons.append(f"{group}_risk_joint_pp_mmd_harm")
        if nonrisk.get("pp_delta_mean", 0.0) < -0.005:
            reasons.append(f"{group}_nonrisk_response_not_preserved")
        if risk.get("mmd_improve_pp_noharm_rows", 0) < max(5, int(0.10 * max(int(risk.get("n", 0)), 1))):
            reasons.append(f"{group}_insufficient_clean_mmd_improve_rows")

    reasons = sorted(set(reasons))
    status = (
        "risk_row_response_preservation_gate_pass_gpu_protocol_next"
        if not reasons
        else "risk_row_response_preservation_gate_fail_no_gpu"
    )
    payload = {
        "status": status,
        "gpu_authorized": not reasons,
        "canonical_metrics_used_for_selection": False,
        "canonical_multi_read": False,
        "trackc_query_read": False,
        "missing": missing,
        "risk_datasets": sorted(RISK_DATASETS),
        "groups": by_group,
        "decision": {
            "reasons": reasons,
            "next_action": (
                "close response-preservation continuation for this exact risk-row evidence"
                if reasons
                else "external review and one default-off guarded risk-row GPU protocol"
            ),
        },
        "boundary": {
            "inputs": {k: str(v) for k, v in required.items()},
            "selection_scope": "train-only/internal condition metrics only",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM Risk-Row Response-Preservation Gate",
        "",
        f"Status: `{status}`",
        f"GPU authorization: `{not reasons}`",
        "",
        "## Boundary",
        "",
        "- CPU-only train-only/internal condition-metric audit.",
        "- Does not use canonical metrics for parameter selection, canonical multi, Track C query, active logs, or GPU.",
        "",
        "## Summary",
        "",
    ]
    for group, summary in by_group.items():
        lines.append(f"### `{group}`")
        for label in ("all", "risk", "nonrisk"):
            s = summary[label]
            lines.append(
                f"- `{label}` n `{s.get('n')}`, pp mean `{s.get('pp_delta_mean')}`, "
                f"MMD mean `{s.get('mmd_delta_mean')}`, pp min `{s.get('pp_delta_min')}`, "
                f"MMD max `{s.get('mmd_delta_max')}`, joint harm rows `{s.get('joint_pp_mmd_harm_rows')}`"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            f"- reasons: `{reasons}`",
            f"- next action: `{payload['decision']['next_action']}`",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorized": not reasons, "out_md": str(OUT_MD)}, indent=2))


if __name__ == "__main__":
    main()
