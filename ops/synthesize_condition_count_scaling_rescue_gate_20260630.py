#!/usr/bin/env python3
"""Condition-count scaling rescue/no-harm synthesis gate.

Reads completed scaling/count reports and decides whether any non-duplicate
count-scaling GPU route is currently justified. This is report-only and does
not train, infer, or re-evaluate canonical metrics.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "condition_count_scaling_rescue_gate_20260630"


INPUTS = {
    "mechanism_panel": REPORTS
    / "latentfm_run_outcome_mechanism_panel_20260630"
    / "latentfm_run_outcome_mechanism_panel_20260630.json",
    "count_smokes": REPORTS / "latentfm_xverse_scaling_count_smokes_decision_20260624.json",
    "count_canonical": REPORTS / "latentfm_xverse_scaling_canonical_noharm_decision_20260624.json",
    "residual_slate": REPORTS / "latentfm_condition_residual_scaling_slate_decision_20260628.json",
    "stabilizer_gate": REPORTS / "latentfm_scaling_noharm_stabilizer_design_gate_20260624.json",
    "highthroughput_internal": REPORTS / "latentfm_scaling_highthroughput_smokes_decision_20260624.json",
    "highthroughput_canonical": REPORTS / "latentfm_scaling_highthroughput_canonical_noharm_decision_20260624.json",
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def get(d: dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def row(
    source: str,
    route: str,
    status: str,
    cross_pp: float | None,
    family_pp: float | None,
    family_mmd: float | None,
    blockers: list[str],
    keep: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "route": route,
        "status": status,
        "cross_pp_delta": cross_pp,
        "family_pp_delta": family_pp,
        "family_mmd_delta": family_mmd,
        "blockers": ";".join(blockers),
        "keep_or_close": keep,
    }


def collect_rows(payloads: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    count = payloads["count_smokes"]
    rows.append(
        row(
            "count_smokes",
            "cap120_all",
            str(get(count, "decision", "status") or ""),
            get(count, "decision", "gate_checks", "cap120_crossbg_pp_minus_anchor"),
            get(count, "decision", "gate_checks", "cap120_family_pp_minus_anchor"),
            get(count, "decision", "gate_checks", "cap120_family_mmd_minus_anchor"),
            [],
            "keep_as_train_only_scaling_signal_not_gpu",
        )
    )
    for key, route in [
        ("full_extension_decision", "full_trainonly"),
        ("type_balance_extension_decision", "type_balanced"),
        ("jiang_exposure_extension_decision", "jiang_exposure"),
        ("general_exposure_extension_decision", "general_exposure"),
    ]:
        item = count.get(key) or {}
        checks = item.get("gate_checks") or {}
        prefix = route.split("_")[0]
        rows.append(
            row(
                "count_smokes",
                route,
                str(item.get("status") or ""),
                checks.get(f"{prefix}_crossbg_pp_minus_anchor")
                or checks.get("general_crossbg_pp_minus_anchor"),
                checks.get(f"{prefix}_family_pp_minus_anchor")
                or checks.get("general_family_pp_minus_anchor"),
                checks.get(f"{prefix}_family_mmd_minus_anchor")
                or checks.get("general_family_mmd_minus_anchor"),
                list(item.get("reasons") or []),
                "close_or_negative_control",
            )
        )
    for item in payloads["residual_slate"].get("rows") or []:
        rows.append(
            row(
                "residual_scaling_slate",
                str(item.get("arm") or item.get("name") or ""),
                str(item.get("status") or ""),
                item.get("cross_pp_delta"),
                item.get("family_pp_delta"),
                item.get("family_mmd_delta"),
                [],
                "keep_weak_only" if str(item.get("arm")) == "gene_cap120_allbg" else "close_or_negative_control",
            )
        )
    for item in payloads["highthroughput_internal"].get("rows") or []:
        metrics = item.get("metrics") or {}
        rows.append(
            row(
                "highthroughput_internal",
                str(item.get("name") or ""),
                str(item.get("status") or ""),
                metrics.get("cross_pp_delta_vs_anchor"),
                metrics.get("family_gene_pp_delta_vs_anchor"),
                metrics.get("family_gene_mmd_delta_vs_anchor"),
                [],
                "seed42_internal_only_not_robust" if int(item.get("seed") or 0) == 42 else "seed_robustness_fail",
            )
        )
    for item in payloads["count_canonical"].get("rows") or []:
        metrics = item.get("metrics") or {}
        rows.append(
            row(
                "count_canonical_noharm",
                str(item.get("run") or ""),
                str(item.get("gate_status") or item.get("status") or ""),
                get(metrics, "cross_background_seen_gene:pearson_pert", "delta_mean"),
                get(metrics, "family_gene:pearson_pert", "delta_mean"),
                get(metrics, "family_gene:test_mmd_clamped", "delta_mean"),
                list(item.get("gate_reasons") or []),
                "canonical_noharm_fail_close",
            )
        )
    for item in payloads["highthroughput_canonical"].get("rows") or []:
        metrics = item.get("metrics") or {}
        rows.append(
            row(
                "highthroughput_canonical_noharm",
                str(item.get("run") or ""),
                str(item.get("gate_status") or item.get("status") or ""),
                get(metrics, "cross_background_seen_gene:pearson_pert", "delta_mean"),
                get(metrics, "family_gene:pearson_pert", "delta_mean"),
                get(metrics, "family_gene:test_mmd_clamped", "delta_mean"),
                list(item.get("gate_reasons") or []),
                "canonical_noharm_fail_close",
            )
        )
    return rows


def decide(rows: list[dict[str, Any]], payloads: dict[str, Any]) -> dict[str, Any]:
    cap120 = [r for r in rows if r["source"] == "count_smokes" and r["route"] == "cap120_all"][0]
    canonical_failures = [
        r for r in rows if r["source"].endswith("canonical_noharm") and "fail" in str(r["status"]).lower()
    ]
    mmd_harm_routes = [
        r
        for r in rows
        if isinstance(r.get("family_mmd_delta"), (int, float)) and float(r["family_mmd_delta"]) > 0.001
    ]
    weak_routes = [
        r
        for r in rows
        if isinstance(r.get("cross_pp_delta"), (int, float))
        and isinstance(r.get("family_pp_delta"), (int, float))
        and float(r["cross_pp_delta"]) > 0
        and float(r["family_pp_delta"]) > 0
        and not (
            isinstance(r.get("family_mmd_delta"), (int, float))
            and float(r["family_mmd_delta"]) > 0.001
        )
    ]
    stabilizer = payloads["stabilizer_gate"]
    reasons = [
        "canonical_noharm_failed_for_existing_count_scaling_candidates",
        "simple_stabilizers_reported_failed_by_prior_design_gate",
    ]
    if mmd_harm_routes:
        reasons.append("positive_general_or_extended_count_routes_have_mmd_harm")
    if not weak_routes:
        reasons.append("no_small_noharm_count_signal_remains")
    status = "condition_count_scaling_rescue_blocks_gpu"
    return {
        "status": status,
        "gpu_authorized_next": False,
        "reasons": reasons,
        "cap120_internal_signal": cap120,
        "n_canonical_failures": len(canonical_failures),
        "mmd_harm_routes": mmd_harm_routes,
        "weak_noharm_routes": weak_routes,
        "prior_stabilizer_status": stabilizer.get("status"),
        "prior_stabilizer_reasons": stabilizer.get("reasons"),
        "next_action": (
            "do_not_launch_count_scaling_gpu; build train-only failure-localization/noharm "
            "predictor or pivot to a new biological/scaling variable"
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["source", "route", "status", "cross_pp_delta", "family_pp_delta", "family_mmd_delta", "blockers", "keep_or_close"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6f}"
    if v is None:
        return ""
    return str(v)


def write_report(payload: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "condition_count_scaling_rescue_gate_20260630.json"
    rows_path = OUT_DIR / "condition_count_scaling_rescue_rows_20260630.csv"
    md_path = OUT_DIR / "LATENTFM_CONDITION_COUNT_SCALING_RESCUE_GATE_20260630.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(rows_path, payload["rows"])
    dec = payload["decision"]
    lines = [
        "# LatentFM Condition-Count Scaling Rescue Gate 20260630",
        "",
        "## Boundary",
        "",
        "- CPU/report-only synthesis of completed condition-count scaling reports.",
        "- No training, inference, active-log polling, checkpoint selection, canonical multi selection, or Track C query access.",
        "- Canonical no-harm reports are used only as existing veto context.",
        "",
        "## Decision",
        "",
        f"- status: `{dec['status']}`",
        f"- gpu authorized next: `{dec['gpu_authorized_next']}`",
        f"- reasons: `{', '.join(dec['reasons'])}`",
        f"- prior stabilizer status: `{dec['prior_stabilizer_status']}`",
        f"- next action: `{dec['next_action']}`",
        "",
        "## Route Rows",
        "",
        "| source | route | status | cross pp | family pp | family MMD | keep/close | blockers |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {source} | `{route}` | `{status}` | {cross} | {family} | {mmd} | `{keep}` | {blockers} |".format(
                source=row["source"],
                route=row["route"],
                status=row["status"],
                cross=fmt(row.get("cross_pp_delta")),
                family=fmt(row.get("family_pp_delta")),
                mmd=fmt(row.get("family_mmd_delta")),
                keep=row["keep_or_close"],
                blockers=row["blockers"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Moderate capped exposure (`cap120_all` / all-background gene cap) remains a real scaling-law signal, but it is small and existing canonical no-harm reports failed.",
            "- More aggressive count/exposure routes can improve pp internally but repeatedly introduce MMD or canonical no-harm risk.",
            "- A future count-scaling run needs a new train-only failure-localization/no-harm predictor, not another cap/full/type/exposure sweep.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{json_path}`",
            f"- rows: `{rows_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    payloads = {name: load(path) for name, path in INPUTS.items()}
    rows = collect_rows(payloads)
    payload = {
        "boundary": {
            "reads_completed_reports_only": True,
            "training_or_inference": False,
            "checkpoint_selection": False,
            "canonical_noharm_used_as_veto_context": True,
            "canonical_multi_selection": False,
            "trackc_query_access": False,
        },
        "inputs": {name: str(path) for name, path in INPUTS.items()},
        "rows": rows,
    }
    payload["decision"] = decide(rows, payloads)
    write_report(payload)
    print(json.dumps(payload["decision"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
