#!/usr/bin/env python3
"""CPU-only Track C support negative-row abstention gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
IN_JSON = ROOT / "reports/latentfm_trackc_support_jackknife_reliability_gate_20260624.json"
OUT_JSON = ROOT / "reports/latentfm_trackc_support_negative_row_abstention_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_SUPPORT_NEGATIVE_ROW_ABSTENTION_GATE_20260624.md"


def nested(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur


def main() -> int:
    payload = json.loads(IN_JSON.read_text(encoding="utf-8"))
    rows = []
    for item in payload.get("cv_summaries", []):
        summary = item.get("summary") or {}
        pp = summary.get("paired_pp_delta") or {}
        by_ds = pp.get("by_dataset") or {}
        rows.append(
            {
                "spec": item.get("spec"),
                "pp_delta": pp.get("delta_mean"),
                "p_harm": pp.get("p_harm"),
                "norman_pp": by_ds.get("NormanWeissman2019_filtered"),
                "wessels_pp": by_ds.get("Wessels"),
                "enabled_rows": summary.get("enabled_rows"),
                "enabled_negative_rows": summary.get("enabled_negative_rows"),
                "enabled_min_pp_delta": summary.get("enabled_min_pp_delta"),
            }
        )
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            int(r.get("enabled_negative_rows") if r.get("enabled_negative_rows") is not None else 10**9),
            -float(r.get("pp_delta") if r.get("pp_delta") is not None else -10**9),
        ),
    )

    pass_specs = []
    for r in rows:
        reasons = []
        if float(r.get("pp_delta") or 0.0) < 0.02:
            reasons.append("train_cv_pp_lt_0p02")
        if float(r.get("p_harm") or 1.0) > 0.20:
            reasons.append("train_cv_pp_p_harm_gt_0p20")
        if float(r.get("norman_pp") or -1.0) < -0.01:
            reasons.append("train_cv_norman_pp_lt_minus_0p01")
        if float(r.get("wessels_pp") or -1.0) < 0.02:
            reasons.append("train_cv_wessels_pp_lt_0p02")
        if int(r.get("enabled_rows") or 0) < 6:
            reasons.append("train_cv_enabled_rows_lt_6")
        if int(r.get("enabled_negative_rows") or 0) > 2:
            reasons.append("train_cv_enabled_negative_rows_gt_2")
        if float(r.get("enabled_min_pp_delta") or -1.0) < -0.02:
            reasons.append("train_cv_enabled_min_pp_lt_minus_0p02")
        if not reasons:
            pass_specs.append(r)

    support_decision = payload.get("decision") or {}
    reasons = []
    if not pass_specs:
        reasons.append("no_train_cv_spec_satisfies_negative_row_abstention_gate")
    if int(support_decision.get("enabled_negative_rows") or 0) > 2:
        reasons.append("original_support_enabled_negative_rows_gt_2")
    if float(support_decision.get("enabled_min_pp_delta") or -1.0) < -0.02:
        reasons.append("original_support_enabled_min_pp_lt_minus_0p02")
    status = "trackc_support_negative_row_abstention_gate_fail_no_gpu"

    out = {
        "status": status,
        "boundary": {
            "cpu_only": True,
            "reads_safe_trainselect_support_reports": True,
            "reads_train_multi_cv_summaries": True,
            "uses_support_val_for_final_scoring_only": True,
            "reads_canonical_metrics": False,
            "reads_canonical_multi": False,
            "reads_heldout_trackc_query": False,
            "launches_gpu": False,
        },
        "source_json": str(IN_JSON),
        "gate_rule": {
            "train_cv_pp_delta_min": 0.02,
            "train_cv_p_harm_max": 0.20,
            "train_cv_norman_pp_min": -0.01,
            "train_cv_wessels_pp_min": 0.02,
            "train_cv_enabled_rows_min": 6,
            "train_cv_enabled_negative_rows_max": 2,
            "train_cv_enabled_min_pp_delta_min": -0.02,
            "support_val_is_not_allowed_for_threshold_selection": True,
        },
        "n_cv_specs": len(rows),
        "n_pass_specs": len(pass_specs),
        "best_by_negative_rows": rows_sorted[:12],
        "original_support_decision": support_decision,
        "reasons": reasons,
        "gpu_authorized": False,
        "next_action": "do not launch support-abstention GPU; new route needs train_multi row-level artifact or materially different reliability signal",
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Track C Support Negative-Row Abstention Gate",
        "",
        f"Status: `{status}`",
        "GPU authorization: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only gate over existing safe trainselect support-jackknife report.",
        "- Uses train_multi CV summaries for abstention eligibility.",
        "- support_val rows remain final scoring only; no support-val threshold rescue is allowed.",
        "- Does not read canonical metrics, canonical multi, held-out Track C query, active logs, train, infer, or use GPU.",
        "",
        "## Gate Rule",
        "",
        "- train-CV pp delta >= `+0.02` and p_harm <= `0.20`.",
        "- Norman train-CV pp >= `-0.01`; Wessels train-CV pp >= `+0.02`.",
        "- enabled rows >= `6`; enabled negative rows <= `2`.",
        "- enabled min pp delta >= `-0.02`.",
        "",
        "## Result",
        "",
        f"- CV specs checked: `{len(rows)}`",
        f"- passing CV specs: `{len(pass_specs)}`",
        f"- original support enabled negative rows: `{support_decision.get('enabled_negative_rows')}` / `{support_decision.get('enabled_rows')}`",
        f"- original support enabled min pp delta: `{support_decision.get('enabled_min_pp_delta')}`",
        "",
        "## Best CV Specs By Negative Rows",
        "",
        "| spec | pp | p_harm | Norman | Wessels | enabled | neg | min pp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows_sorted[:12]:
        lines.append(
            f"| `{r['spec']}` | {float(r.get('pp_delta') or 0.0):+.6f} | "
            f"{float(r.get('p_harm') or 0.0):+.6f} | {float(r.get('norman_pp') or 0.0):+.6f} | "
            f"{float(r.get('wessels_pp') or 0.0):+.6f} | {int(r.get('enabled_rows') or 0)} | "
            f"{int(r.get('enabled_negative_rows') or 0)} | {float(r.get('enabled_min_pp_delta') or 0.0):+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- reasons: `{reasons}`",
            "- next action: no support-abstention GPU launch; reopen only with a train_multi row-level reliability artifact or materially different support reliability signal.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
