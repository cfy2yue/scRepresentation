#!/usr/bin/env python3
"""CPU-only artifact sufficiency gate for Track C row-reliability V2.

The gate checks whether existing safe trainselect artifacts are sufficient to
test a materially new train_multi row-level reliability route, distinct from
the closed support-abstention sweep. It does not read held-out query,
canonical multi, or launch jobs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
JACKKNIFE = REPORTS / "latentfm_trackc_support_jackknife_reliability_gate_20260624.json"
ABSTENTION = REPORTS / "latentfm_trackc_support_negative_row_abstention_gate_20260624.json"
ADJUDICATION = REPORTS / "latentfm_trackc_both_train_multi_gene_adjudication_gate_20260624.json"

ROBUSTNESS = [
    REPORTS / "latentfm_trackc_support_only_robustness_decision_xverse_trackc_support_pairtype_both_train_multi_gene_resfilm_ep050_replay2_2k_seed43.json",
    REPORTS / "latentfm_trackc_support_only_robustness_decision_xverse_trackc_support_pairtype_both_train_multi_gene_resfilm_ep050_replay2_2k_seed44.json",
    REPORTS / "latentfm_trackc_support_only_robustness_decision_xverse_trackc_support_pairtype_both_train_multi_gene_resfilm_ep050_replay2_2k_seed45.json",
]

OUT_JSON = REPORTS / "latentfm_trackc_row_reliability_v2_artifact_gate_20260624.json"
OUT_MD = REPORTS / "LATENTFM_TRACKC_ROW_RELIABILITY_V2_ARTIFACT_GATE_20260624.md"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def main() -> int:
    jack = load_json(JACKKNIFE)
    abst = load_json(ABSTENTION)
    adjud = load_json(ADJUDICATION)
    robust = [load_json(path) for path in ROBUSTNESS]

    cv_summaries = list(jack.get("cv_summaries") or [])
    support_rows = list(jack.get("support_rows") or [])
    shuffled_rows = list(jack.get("shuffled_rows") or [])
    cv_has_row_tables = any(
        isinstance(item.get("rows"), list) or isinstance((item.get("summary") or {}).get("rows"), list)
        for item in cv_summaries
    )
    support_rows_have_reliability_features = bool(support_rows) and all(
        ("jackknife_cos_mean" in row and "jackknife_norm_cv" in row and "n_context_rows" in row)
        for row in support_rows
    )

    robustness_rows = []
    for obj in robust:
        decision = obj.get("decision") or {}
        robustness_rows.append(
            {
                "path": obj.get("inputs", {}).get("support_candidate_split") or obj.get("run_root") or obj.get("path"),
                "status": decision.get("status"),
                "seed": obj.get("seed"),
                "action": decision.get("action"),
            }
        )

    original = abst.get("original_support_decision") or {}
    reasons = []
    if jack.get("status") != "trackc_support_jackknife_reliability_gate_fail_no_gpu":
        reasons.append("unexpected_jackknife_status")
    if not cv_summaries:
        reasons.append("missing_train_multi_cv_summaries")
    if not cv_has_row_tables:
        reasons.append("missing_train_multi_cv_row_level_tables")
    if not support_rows_have_reliability_features:
        reasons.append("support_val_rows_missing_reliability_features")
    if int(abst.get("n_pass_specs") or 0) <= 0:
        reasons.append("closed_abstention_sweep_has_zero_pass_specs")
    if int(original.get("enabled_negative_rows") or 0) > 2:
        reasons.append("original_support_has_too_many_negative_rows")
    if float(original.get("enabled_min_pp_delta") or 0.0) < -0.02:
        reasons.append("original_support_enabled_min_pp_too_negative")
    if (adjud.get("status") or "").endswith("fail_no_gpu"):
        reasons.append("whole_support_adjudication_failed")
    if any((row.get("status") or "").endswith("fail_close") for row in robustness_rows):
        reasons.append("seed_robustness_has_fail_close_seed")

    # A pass would mean existing artifacts are enough to run the V2 gate now.
    status = "trackc_row_reliability_v2_artifact_gate_fail_no_gpu"
    if cv_summaries and cv_has_row_tables and support_rows_have_reliability_features and int(abst.get("n_pass_specs") or 0) > 0:
        status = "trackc_row_reliability_v2_artifact_gate_pass_cpu_gate_next"

    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "reads_safe_trainselect_reports": True,
            "reads_canonical_metrics": False,
            "reads_canonical_multi": False,
            "reads_heldout_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "artifact_summary": {
            "n_train_multi_rows": jack.get("n_train_multi_rows"),
            "n_support_val_rows": jack.get("n_support_val_rows"),
            "n_cv_summaries": len(cv_summaries),
            "cv_has_row_level_tables": cv_has_row_tables,
            "n_support_rows": len(support_rows),
            "n_shuffled_rows": len(shuffled_rows),
            "support_rows_have_reliability_features": support_rows_have_reliability_features,
            "abstention_n_specs": abst.get("n_cv_specs"),
            "abstention_n_pass_specs": abst.get("n_pass_specs"),
            "original_enabled_rows": original.get("enabled_rows"),
            "original_enabled_negative_rows": original.get("enabled_negative_rows"),
            "original_enabled_min_pp_delta": original.get("enabled_min_pp_delta"),
            "original_support_val_pp_delta": original.get("support_val_pp_delta"),
            "original_norman_pp_delta": original.get("norman_pp_delta"),
            "original_wessels_pp_delta": original.get("wessels_pp_delta"),
            "adjudication_status": adjud.get("status"),
        },
        "robustness_rows": robustness_rows,
        "reasons": reasons,
        "next_action": (
            "run row-reliability V2 CPU gate"
            if status.endswith("_next")
            else "do not launch Track C row-reliability GPU; first generate train_multi row-level LOO reliability artifact"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Track C Row-Reliability V2 Artifact Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only artifact sufficiency gate.",
        "- Reads existing safe trainselect support reports only.",
        "- Does not read canonical metrics, canonical multi, held-out Track C query, active logs, train, infer, or use GPU.",
        "",
        "## Artifact Summary",
        "",
        f"- train_multi rows: `{jack.get('n_train_multi_rows')}`",
        f"- support_val rows: `{jack.get('n_support_val_rows')}`",
        f"- CV summaries: `{len(cv_summaries)}`",
        f"- CV row-level tables present: `{cv_has_row_tables}`",
        f"- support_val row-level rows: `{len(support_rows)}`",
        f"- shuffled support_val rows: `{len(shuffled_rows)}`",
        f"- support rows have reliability features: `{support_rows_have_reliability_features}`",
        f"- abstention pass specs: `{abst.get('n_pass_specs')}/{abst.get('n_cv_specs')}`",
        f"- original support pp / Norman / Wessels: `{fmt(original.get('support_val_pp_delta'))}` / `{fmt(original.get('norman_pp_delta'))}` / `{fmt(original.get('wessels_pp_delta'))}`",
        f"- original enabled negative rows / min pp: `{original.get('enabled_negative_rows')}` / `{fmt(original.get('enabled_min_pp_delta'))}`",
        f"- adjudication status: `{adjud.get('status')}`",
        "",
        "## Decision",
        "",
        f"- reasons: `{reasons}`",
        "- GPU authorized: `False`",
        f"- next action: `{payload['next_action']}`",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "gpu_authorized": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
