#!/usr/bin/env python3
"""Seed-matched condition-count micro-matrix gate for scaling.

CPU/report-only gate. It checks whether the current moderate-exposure
condition-count signal is seed-stable enough to justify more GPU work. It reads
completed train-only internal and frozen no-harm reports only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_scaling_seed_matched_micro_matrix_gate_20260624.json"
OUT_MD = REPORTS / "LATENTFM_SCALING_SEED_MATCHED_MICRO_MATRIX_GATE_20260624.md"


def load_json(name: str) -> dict[str, Any]:
    path = REPORTS / name
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


def row_by_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("name") or row.get("run") or row.get("role") or ""): row for row in payload.get("rows") or []}


def canonical_row_by_run(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("run") or row.get("name") or ""): row for row in payload.get("rows") or []}


def metric_delta(row: dict[str, Any], metric: str) -> float | None:
    item = (row.get("metrics") or {}).get(metric) or {}
    value = item.get("delta_mean")
    return float(value) if value is not None else None


def metric_p_harm(row: dict[str, Any], metric: str) -> float | None:
    item = (row.get("metrics") or {}).get(metric) or {}
    value = item.get("p_harm")
    return float(value) if value is not None else None


def summarize_seed(row: dict[str, Any], canon: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics = row.get("metrics") or {}
    out = {
        "run": row.get("name") or row.get("run"),
        "role": row.get("role"),
        "seed": row.get("seed"),
        "steps": row.get("steps"),
        "anchor_replay_weight": row.get("anchor_replay_weight"),
        "internal_cross_pp_delta": metrics.get("cross_pp_delta_vs_anchor"),
        "internal_family_pp_delta": metrics.get("family_gene_pp_delta_vs_anchor"),
        "internal_family_mmd_delta": metrics.get("family_gene_mmd_delta_vs_anchor"),
        "internal_pass_thresholds": False,
    }
    if (
        out["internal_cross_pp_delta"] is not None
        and out["internal_family_pp_delta"] is not None
        and out["internal_family_mmd_delta"] is not None
    ):
        out["internal_pass_thresholds"] = (
            float(out["internal_cross_pp_delta"]) >= 0.010
            and float(out["internal_family_pp_delta"]) >= 0.008
            and float(out["internal_family_mmd_delta"]) <= 0.001
        )
    if canon:
        out.update(
            {
                "canonical_gate_status": canon.get("gate_status"),
                "canonical_gate_reasons": canon.get("gate_reasons") or [],
                "canonical_cross_pp_delta": metric_delta(canon, "cross_background_seen_gene:pearson_pert"),
                "canonical_family_pp_delta": metric_delta(canon, "family_gene:pearson_pert"),
                "canonical_max_pp_p_harm": max(
                    [
                        v
                        for v in [
                            metric_p_harm(canon, "cross_background_seen_gene:pearson_pert"),
                            metric_p_harm(canon, "all_test_single:pearson_pert"),
                            metric_p_harm(canon, "family_gene:pearson_pert"),
                        ]
                        if v is not None
                    ],
                    default=None,
                ),
            }
        )
    return out


def main() -> int:
    high = load_json("latentfm_scaling_highthroughput_smokes_decision_20260624.json")
    high_canon = load_json("latentfm_scaling_highthroughput_canonical_noharm_decision_20260624.json")
    refill = load_json("latentfm_scaling_highthroughput_smokes_refill_decision_20260624.json")
    refill_canon = load_json("latentfm_scaling_highthroughput_canonical_noharm_refill_decision_20260624.json")
    protocol = load_json("latentfm_scaling_protocol_matrix_decision_20260624.json")
    protocol_canon = load_json("latentfm_scaling_protocol_canonical_noharm_decision_20260624.json")

    high_rows = row_by_name(high)
    high_canon_rows = canonical_row_by_run(high_canon)
    seed_rows = []
    for run in [
        "xverse_scaling_cap60_6k_seed42",
        "xverse_scaling_cap60_6k_seed43",
        "xverse_scaling_cap60_replay05_4k_seed42",
    ]:
        if run in high_rows:
            seed_rows.append(summarize_seed(high_rows[run], high_canon_rows.get(run)))

    # Refill reports may duplicate the same run names but are kept as an
    # independent evidence source when present.
    refill_rows = row_by_name(refill)
    refill_canon_rows = canonical_row_by_run(refill_canon)
    for run, row in sorted(refill_rows.items()):
        if run and "cap60" in run:
            item = summarize_seed(row, refill_canon_rows.get(run))
            item["evidence_family"] = "refill"
            seed_rows.append(item)

    protocol_rows = row_by_name(protocol)
    protocol_canon_rows = canonical_row_by_run(protocol_canon)
    protocol_item = None
    run = "xverse_scaling_protocol_cap60_primary19_3k_seed42"
    if run in protocol_rows:
        protocol_item = summarize_seed(protocol_rows[run], protocol_canon_rows.get(run))
        protocol_item["evidence_family"] = "cap60_primary19_seed42_anchor"

    # De-duplicate rows by run while preserving the first occurrence.
    dedup = {}
    for row in seed_rows:
        key = row.get("run")
        if key and key not in dedup:
            dedup[key] = row
    seed_rows = list(dedup.values())

    cap60_6k = [row for row in seed_rows if str(row.get("run", "")).startswith("xverse_scaling_cap60_6k_seed")]
    pass_rows = [row for row in cap60_6k if row.get("internal_pass_thresholds")]
    fail_rows = [row for row in cap60_6k if not row.get("internal_pass_thresholds")]
    sign_flip = any(float(row.get("internal_cross_pp_delta") or 0.0) < 0 for row in cap60_6k)
    canonical_fail_rows = [
        row
        for row in seed_rows
        if row.get("canonical_gate_status")
        and "fail" in str(row.get("canonical_gate_status"))
    ]

    reasons = []
    if len(cap60_6k) < 2:
        reasons.append("fewer_than_two_seed_matched_cap60_6k_runs")
    if len(pass_rows) < len(cap60_6k):
        reasons.append("not_all_seed_matched_runs_pass_internal_thresholds")
    if sign_flip:
        reasons.append("seed_matched_cross_pp_sign_flip")
    if fail_rows:
        reasons.append("seed43_like_negative_or_weak_seed_present")
    if canonical_fail_rows:
        reasons.append("existing_seed42_cap60_family_failed_frozen_canonical_noharm")
    if protocol_item and protocol_item.get("canonical_gate_status") and "fail" in str(protocol_item.get("canonical_gate_status")):
        reasons.append("cap60_primary19_seed42_canonical_noharm_failed")

    status = "seed_matched_condition_count_micro_matrix_gate_fail_no_gpu"
    if not reasons:
        status = "seed_matched_condition_count_micro_matrix_gate_pass_one_seed_expansion_next"

    payload = {
        "status": status,
        "gpu_authorized": status.endswith("_next"),
        "boundary": {
            "reads_completed_reports_only": True,
            "canonical_noharm_used_as_veto_context": True,
            "canonical_multi_selection": False,
            "trackc_query_read": False,
            "training_or_inference": False,
            "active_logs_read": False,
            "gpu": False,
        },
        "gate_rule": {
            "seed_matched_family": "cap60_6k same split/steps",
            "required": [
                ">=2 completed seed-matched runs",
                "all seed-matched runs internal cross pp >= +0.010",
                "all seed-matched runs internal family pp >= +0.008",
                "all seed-matched runs family MMD <= +0.001",
                "no cross-pp sign flip",
                "no prior frozen canonical no-harm failure for same family",
            ],
        },
        "summary": {
            "n_cap60_6k_seed_matched": len(cap60_6k),
            "n_pass_internal": len(pass_rows),
            "n_fail_internal": len(fail_rows),
            "cross_pp_sign_flip": sign_flip,
            "n_canonical_fail_rows": len(canonical_fail_rows),
        },
        "reasons": reasons,
        "seed_rows": seed_rows,
        "protocol_anchor": protocol_item,
        "next_action": (
            "launch exactly one additional seed-matched bounded smoke after resource audit"
            if status.endswith("_next")
            else "do not launch more cap60/count-scaling seed expansion; treat current signal as seed-sensitive diagnostic"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Scaling Seed-Matched Condition-Count Micro-Matrix Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only synthesis of completed internal and frozen no-harm reports.",
        "- Canonical no-harm is used only as veto/context, not checkpoint selection.",
        "- Does not read canonical multi, Track C query, active logs, train, infer, or use GPU.",
        "",
        "## Gate Rule",
        "",
        "- Family: cap60_6k same split/steps.",
        "- All seed-matched runs must pass internal cross/family/MMD thresholds.",
        "- No seed can flip cross pp negative.",
        "- Existing frozen canonical no-harm failure for the same family vetoes expansion.",
        "",
        "## Seed Rows",
        "",
        "| run | seed | steps | replay | cross pp | family pp | family MMD | internal pass | canonical gate | canonical cross | canonical family pp |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in seed_rows:
        lines.append(
            f"| `{row.get('run')}` | {row.get('seed')} | {row.get('steps')} | {fmt(row.get('anchor_replay_weight'))} | "
            f"{fmt(row.get('internal_cross_pp_delta'))} | {fmt(row.get('internal_family_pp_delta'))} | "
            f"{fmt(row.get('internal_family_mmd_delta'))} | `{row.get('internal_pass_thresholds')}` | "
            f"`{row.get('canonical_gate_status', 'NA')}` | {fmt(row.get('canonical_cross_pp_delta'))} | "
            f"{fmt(row.get('canonical_family_pp_delta'))} |"
        )
    if protocol_item:
        lines.extend(
            [
                "",
                "## Protocol Anchor",
                "",
                f"- run: `{protocol_item.get('run')}`",
                f"- internal cross/family/MMD: `{fmt(protocol_item.get('internal_cross_pp_delta'))}` / `{fmt(protocol_item.get('internal_family_pp_delta'))}` / `{fmt(protocol_item.get('internal_family_mmd_delta'))}`",
                f"- canonical gate: `{protocol_item.get('canonical_gate_status')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- reasons: `{reasons}`",
            f"- GPU authorized: `{payload['gpu_authorized']}`",
            f"- next action: `{payload['next_action']}`",
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
