#!/usr/bin/env python3
"""Protocol audit for a Track C anchor-gated support-teacher branch.

This is deliberately CPU-only and read-only. It records the next distinct
Track C mechanism after support-FiLM: keep the canonical anchor as default and
apply a frozen support-teacher residual only behind a train/support-derived
gate. The script does not claim that the mechanism has passed; it checks that
current artifacts are sufficient only for protocol design, not for an offline
blend evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
DECISION_JSON = (
    ROOT
    / "reports/latentfm_trackc_routed_distill_smoke_decision_"
    "xverse_trackc_support_film_absroute_2k_seed42_retry1.json"
)
ROUTE_GAP_JSON = (
    ROOT
    / "reports/latentfm_trackc_support_film_route_gap_gate_"
    "xverse_trackc_support_film_absroute_2k_seed42_retry1.json"
)
PRIOR_PROTOCOL = ROOT / "reports/LATENTFM_TRACKC_DISTINCT_SUPPORT_ABSORBABILITY_PROTOCOL_20260623.md"
OUT_JSON = ROOT / "reports/latentfm_trackc_anchor_gated_support_teacher_protocol_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_ANCHOR_GATED_SUPPORT_TEACHER_PROTOCOL_20260623.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested_has_key(obj: Any, needles: set[str], depth: int = 0) -> bool:
    if depth > 5:
        return False
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in needles:
                return True
            if nested_has_key(value, needles, depth + 1):
                return True
    elif isinstance(obj, list):
        for value in obj[:20]:
            if nested_has_key(value, needles, depth + 1):
                return True
    return False


def metric_row(decision: dict[str, Any], table: str, group: str, metric: str) -> dict[str, Any]:
    rows = decision.get("tables", {}).get(table) or {}
    if isinstance(rows, dict):
        rows = list(rows.values())
    for row in rows:
        if isinstance(row, dict) and str(row.get("group")) == group and str(row.get("metric")) == metric:
            return row
    return {}


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def main() -> int:
    decision = load_json(DECISION_JSON)
    route_gap = load_json(ROUTE_GAP_JSON)
    input_paths = {k: Path(v) for k, v in (decision.get("inputs") or {}).items()}
    missing = [str(p) for p in input_paths.values() if not p.is_file()]
    raw_pred_like = {}
    needles = {"predictions", "prediction", "pred", "per_condition", "condition_results", "rows"}
    for name, path in input_paths.items():
        if not path.is_file():
            raw_pred_like[name] = False
            continue
        raw_pred_like[name] = nested_has_key(load_json(path), needles)

    support_pp = metric_row(decision, "support_split", "test_multi", "pearson_pert")
    support_mmd = metric_row(decision, "support_split", "test_multi", "test_mmd_clamped")
    canonical_single_pp = metric_row(decision, "canonical_split", "test_single", "pearson_pert")
    canonical_family_pp = metric_row(decision, "canonical_family", "family_gene", "pearson_pert")
    route_gap_decision = route_gap.get("decision") or {}
    wessels_summary = next(
        (row for row in route_gap.get("summary", []) if str(row.get("dataset")) == "Wessels"),
        {},
    )

    payload: dict[str, Any] = {
        "status": "trackc_anchor_gated_support_teacher_protocol_ready_no_gpu",
        "gpu_authorization": "none",
        "mechanism": "anchor_gated_frozen_support_teacher_residual_calibrator",
        "formula": "pred = anchor_pred + g_trainonly(condition,dataset) * alpha * (support_teacher_pred - anchor_pred)",
        "source_decision_json": str(DECISION_JSON),
        "source_route_gap_json": str(ROUTE_GAP_JSON),
        "prior_protocol": str(PRIOR_PROTOCOL),
        "support_film_decision_status": decision.get("decision", {}).get("status"),
        "evidence": {
            "support_pp_delta": support_pp.get("delta_mean"),
            "support_pp_p_harm": support_pp.get("p_harm"),
            "support_mmd_delta": support_mmd.get("delta_mean"),
            "canonical_test_single_pp_p_harm": canonical_single_pp.get("p_harm"),
            "canonical_family_gene_pp_p_harm": canonical_family_pp.get("p_harm"),
            "wessels_route_gap_closure": route_gap_decision.get("wessels_route_gap_closure")
            or route_gap_decision.get("wessels_mean_pp_route_gap_closure")
            or wessels_summary.get("route_gap_closed_fraction"),
        },
        "artifact_audit": {
            "missing_inputs": missing,
            "posthoc_json_has_raw_prediction_like_keys": raw_pred_like,
            "offline_blend_eval_possible_from_current_artifacts": any(raw_pred_like.values()) and not missing,
        },
        "required_next_cpu_gate": {
            "leakage_boundary": [
                "safe trainselect support-val only for support selection",
                "canonical split single/background no-harm only after route is frozen",
                "no full v2 query, no held-out Track C query, no canonical multi selection",
            ],
            "grid": [
                "alpha in a tiny predeclared grid, e.g. 0.10/0.25/0.50",
                "gate threshold or reliability rule selected by train_multi CV only",
                "zero-support and shuffled-support controls",
            ],
            "support_gate": [
                "Wessels pp delta >= +0.02",
                "Wessels route-gap closure >= +0.05",
                "Norman pp delta >= -0.02",
                "support pp p_harm <= 0.20",
                "no MMD hard harm if evaluated",
            ],
            "canonical_gate": [
                "test_single pp p_harm <= 0.35",
                "family_gene pp p_harm <= 0.35",
                "canonical MMD p_harm <= 0.80",
                "near-anchor behavior on canonical rows where g=0",
            ],
            "stop_rule": "If this CPU gate cannot pass canonical no-harm, close Track C support-absorbability backup for now.",
        },
        "required_new_artifacts": [
            "per-condition anchor predictions or sufficient prediction summaries",
            "per-condition frozen support-teacher predictions",
            "train/support-only reliability gate values",
            "canonical no-harm per-condition gate mask proof",
        ],
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Track C Anchor-Gated Support-Teacher Protocol",
        "",
        f"Status: `{payload['status']}`",
        f"GPU authorization: `{payload['gpu_authorization']}`",
        "",
        "## Mechanism",
        "",
        "`pred = anchor_pred + g_trainonly(condition,dataset) * alpha * (support_teacher_pred - anchor_pred)`",
        "",
        "The anchor prediction remains the default.  The support residual is only",
        "applied when a train/support-derived reliability gate is high.  This is",
        "materially distinct from support-FiLM shift/scale because it does not",
        "modulate hidden state or change route labels.",
        "",
        "## Why This Branch Exists",
        "",
        f"- support-FiLM status: `{payload['support_film_decision_status']}`",
        f"- support pp delta: `{fmt(payload['evidence']['support_pp_delta'])}`",
        f"- support MMD delta: `{fmt(payload['evidence']['support_mmd_delta'])}`",
        f"- Wessels route-gap closure: `{fmt(payload['evidence']['wessels_route_gap_closure'])}`",
        f"- canonical test_single pp p_harm: `{fmt(payload['evidence']['canonical_test_single_pp_p_harm'])}`",
        f"- canonical family_gene pp p_harm: `{fmt(payload['evidence']['canonical_family_gene_pp_p_harm'])}`",
        "",
        "Interpretation: support signal exists, but the previous model-facing path",
        "moved canonical single/background behavior too much.  The next question is",
        "whether support can be applied selectively while preserving anchor behavior.",
        "",
        "## Artifact Audit",
        "",
        f"- missing posthoc inputs: `{payload['artifact_audit']['missing_inputs']}`",
        f"- raw prediction-like keys by input: `{payload['artifact_audit']['posthoc_json_has_raw_prediction_like_keys']}`",
        f"- offline blend eval possible from current aggregate posthoc: `{payload['artifact_audit']['offline_blend_eval_possible_from_current_artifacts']}`",
        "",
        "Current posthoc artifacts are aggregate metric reports, not sufficient raw",
        "prediction artifacts for a faithful offline residual blend.  A new CPU gate",
        "must first generate or read per-condition anchor/support-teacher outputs.",
        "",
        "## Required CPU Gate",
        "",
        "- Use only safe trainselect support-val for support route/gate selection.",
        "- Use canonical single/background no-harm only after route and grid are frozen.",
        "- Do not read full v2 query, held-out Track C query, or canonical multi for selection.",
        "- Include zero-support no-op and shuffled-support negative controls.",
        "- Pass Wessels pp delta `>= +0.02`, Wessels closure `>= +0.05`, Norman pp delta `>= -0.02`, and support pp p_harm `<= 0.20`.",
        "- Pass canonical `test_single` and `family_gene` pp p_harm `<= 0.35`, with canonical MMD p_harm `<= 0.80`.",
        "",
        "## Stop Rule",
        "",
        "If the CPU gate cannot pass canonical no-harm, close the Track C support",
        "absorbability backup for now rather than sweeping more FiLM/endpoint/replay",
        "variants.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
