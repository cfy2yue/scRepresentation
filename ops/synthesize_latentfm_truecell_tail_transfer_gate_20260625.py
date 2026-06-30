#!/usr/bin/env python3
"""Synthesize a true-cell tail/no-harm transfer gate.

CPU-only, query-blind synthesis from completed true-cell reports. This answers
whether any true-cell route is currently launchable/promotable, or whether the
branch should remain mechanism-only until a genuinely new hypothesis appears.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_truecell_tail_transfer_gate_20260625.json"
OUT_MD = REPORTS / "LATENTFM_TRUECELL_TAIL_TRANSFER_GATE_20260625.md"


def load(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    return json.loads(path.read_text())


def first_budget_row(payload: dict[str, Any]) -> dict[str, Any] | None:
    rows = payload.get("matrix_summary", {}).get("budget_rows", [])
    return rows[0] if rows else None


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:+.6f}"
    if isinstance(value, list):
        return "[" + ", ".join(fmt(v) for v in value) + "]"
    return str(value)


def route_row(
    *,
    route: str,
    report: str,
    payload: dict[str, Any],
    controls_report: str | None = None,
    controls_payload: dict[str, Any] | None = None,
    canonical_blocker: bool = False,
) -> dict[str, Any]:
    brow = first_budget_row(payload)
    if brow is None:
        return {
            "route": route,
            "status": "missing_or_empty",
            "report": report,
            "reasons": ["missing_budget_row"],
            "promotable_train_only": False,
            "launchable_new_route": False,
        }
    cross_boot = brow.get("cross_background_pp_condition_bootstrap", {})
    cross_tail = brow.get("cross_background_pp_dataset_tail", {})
    family_boot = brow.get("family_gene_pp_condition_bootstrap", {})
    controls_status = (controls_payload or {}).get("status") if controls_payload else None
    cross_ci = cross_boot.get("ci95") or [None, None]
    family_ci = family_boot.get("ci95") or [None, None]
    min_dataset = cross_tail.get("min_dataset") or {}

    reasons: list[str] = []
    cross_mean = brow.get("cross_background_pp_delta_mean")
    family_mean = brow.get("family_gene_pp_delta_mean")
    family_mmd = brow.get("family_gene_mmd_delta_mean")
    ci_low = cross_ci[0]
    dataset_min = min_dataset.get("mean")
    neg_tails = cross_tail.get("negative_tail_lt_minus_0p020")

    if cross_mean is None or cross_mean < 0.010:
        reasons.append("cross_mean_lt_0p010_or_missing")
    if family_mean is None or family_mean < 0.0:
        reasons.append("family_mean_negative_or_missing")
    if family_mmd is None or family_mmd > 0.001:
        reasons.append("family_mmd_gt_0p001_or_missing")
    if ci_low is None or ci_low <= 0.0:
        reasons.append("cross_ci_lower_not_positive")
    if dataset_min is None or dataset_min < -0.020:
        reasons.append("dataset_min_below_minus_0p020")
    if neg_tails is None or neg_tails != 0:
        reasons.append("negative_dataset_tails_present_or_missing")
    if controls_payload and controls_status != "nested_controls_pass_no_gpu":
        reasons.append("controls_not_pass")
    if canonical_blocker:
        reasons.append("frozen_canonical_noharm_failed")

    promotable_train_only = not [r for r in reasons if r != "frozen_canonical_noharm_failed"]
    launchable_new_route = promotable_train_only and not canonical_blocker

    return {
        "route": route,
        "status": payload.get("status", ""),
        "report": report,
        "controls_report": controls_report,
        "controls_status": controls_status,
        "cross_mean": cross_mean,
        "family_mean": family_mean,
        "family_mmd": family_mmd,
        "cross_ci": cross_ci,
        "family_ci": family_ci,
        "dataset_min": dataset_min,
        "dataset_min_name": min_dataset.get("dataset"),
        "negative_tails": neg_tails,
        "promotable_train_only": promotable_train_only,
        "canonical_blocker": canonical_blocker,
        "launchable_new_route": launchable_new_route,
        "reasons": reasons,
    }


def main() -> None:
    nested3k = load("reports/latentfm_true_cell_count_nested_matrix_decision_20260624.json")
    nested3k_controls = load("reports/latentfm_true_cell_count_nested_controls_gate_20260624.json")
    budget128 = load("reports/latentfm_true_cell_count_budget128_tail_stability_6k_decision_20260625.json")
    budget128_controls = load("reports/latentfm_true_cell_count_budget128_tail_stability_6k_controls_20260625.json")
    ar005 = load("reports/latentfm_true_cell_count_budget128_anchor_replay005_6k_decision_20260625.json")
    ar005_controls = load("reports/latentfm_true_cell_count_budget128_anchor_replay005_6k_controls_20260625.json")
    budget64 = load("reports/latentfm_true_cell_count_budget64_tail_stability_6k_decision_20260625.json")
    budget64_controls = load("reports/latentfm_true_cell_count_budget64_tail_stability_6k_controls_20260625.json")
    canonical = load("reports/latentfm_true_cell_count_budget128_6k_canonical_noharm_decision_20260625.json")

    routes: list[dict[str, Any]] = []
    for brow in nested3k.get("matrix_summary", {}).get("budget_rows", []):
        single = {
            "status": nested3k.get("status"),
            "matrix_summary": {"budget_rows": [brow]},
        }
        routes.append(
            route_row(
                route=f"nested_3k_budget{brow.get('budget')}",
                report="reports/LATENTFM_TRUE_CELL_COUNT_NESTED_MATRIX_DECISION_20260624.md",
                payload=single,
                controls_report="reports/LATENTFM_TRUE_CELL_COUNT_NESTED_CONTROLS_GATE_20260624.md",
                controls_payload=nested3k_controls,
            )
        )

    canonical_failed = canonical.get("decision", {}).get("status") == "canonical_noharm_fail_close_promotion"
    routes.append(
        route_row(
            route="nested_6k_budget128",
            report="reports/LATENTFM_TRUE_CELL_COUNT_BUDGET128_TAIL_STABILITY_6K_DECISION_20260625.md",
            payload=budget128,
            controls_report="reports/LATENTFM_TRUE_CELL_COUNT_BUDGET128_TAIL_STABILITY_6K_CONTROLS_20260625.md",
            controls_payload=budget128_controls,
            canonical_blocker=canonical_failed,
        )
    )
    routes.append(
        route_row(
            route="budget128_anchor_replay005_6k",
            report="reports/LATENTFM_TRUE_CELL_COUNT_BUDGET128_ANCHOR_REPLAY005_6K_DECISION_20260625.md",
            payload=ar005,
            controls_report="reports/LATENTFM_TRUE_CELL_COUNT_BUDGET128_ANCHOR_REPLAY005_6K_CONTROLS_20260625.md",
            controls_payload=ar005_controls,
        )
    )
    routes.append(
        route_row(
            route="nested_6k_budget64",
            report="reports/LATENTFM_TRUE_CELL_COUNT_BUDGET64_TAIL_STABILITY_6K_DECISION_20260625.md",
            payload=budget64,
            controls_report="reports/LATENTFM_TRUE_CELL_COUNT_BUDGET64_TAIL_STABILITY_6K_CONTROLS_20260625.md",
            controls_payload=budget64_controls,
        )
    )

    launchable = [r for r in routes if r.get("launchable_new_route")]
    train_only_pass = [r for r in routes if r.get("promotable_train_only")]
    reasons: list[str] = []
    if not launchable:
        reasons.append("no_truecell_route_passes_train_tail_gate_without_canonical_blocker")
    if canonical_failed:
        reasons.append("strongest_budget128_6k_route_failed_frozen_canonical_noharm")
    if budget64.get("status") == "nested_matrix_fail_or_mechanism_only":
        reasons.append("budget64_6k_failed_tail_gate_budget256_launch_condition_not_met")
    if ar005.get("status") == "nested_matrix_fail_or_mechanism_only":
        reasons.append("anchor_replay005_repair_failed_tail_gate")

    status = "truecell_tail_transfer_gate_fail_mechanism_only" if reasons else "truecell_tail_transfer_gate_pass"
    payload = {
        "boundary": {
            "cpu_only": True,
            "gpu": False,
            "reads_train_only_internal_truecell_reports": True,
            "reads_canonical_noharm_as_veto_context": True,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
        },
        "status": status,
        "gpu_authorized": False,
        "decision": {
            "launch_budget256": False,
            "launch_budget128_repair": False,
            "claim_scope": "mechanism_only",
            "next_action": "do_not_launch_truecell_gpu_until_new_nonduplicate_cpu_gate_passes",
        },
        "reasons": reasons,
        "routes": routes,
        "summary": {
            "n_routes": len(routes),
            "n_train_only_tail_pass": len(train_only_pass),
            "n_launchable_new_routes": len(launchable),
            "train_only_tail_pass_routes": [r["route"] for r in train_only_pass],
            "launchable_new_routes": [r["route"] for r in launchable],
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# LatentFM True-Cell Tail Transfer Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only synthesis of completed true-cell train-only/internal gates.",
        "- Canonical no-harm is used only as veto/blocker context for the already frozen budget128 6k route.",
        "- Does not read canonical multi, held-out Track C query, checkpoints, train, infer, or use GPU.",
        "",
        "## Route Table",
        "",
        "| route | status | cross pp | family pp | family MMD | cross CI | dataset min | neg tails | train-tail pass | canonical blocker | reasons |",
        "|---|---|---:|---:|---:|---|---:|---:|---|---|---|",
    ]
    for row in routes:
        lines.append(
            "| {route} | `{status}` | {cross} | {family} | {mmd} | {ci} | {dmin} ({dname}) | {neg} | {train_pass} | {canon} | {reasons} |".format(
                route=row.get("route"),
                status=row.get("status", ""),
                cross=fmt(row.get("cross_mean")),
                family=fmt(row.get("family_mean")),
                mmd=fmt(row.get("family_mmd")),
                ci=fmt(row.get("cross_ci")),
                dmin=fmt(row.get("dataset_min")),
                dname=row.get("dataset_min_name", ""),
                neg=fmt(row.get("negative_tails")),
                train_pass="yes" if row.get("promotable_train_only") else "no",
                canon="yes" if row.get("canonical_blocker") else "no",
                reasons=", ".join(row.get("reasons", [])) or "none",
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- GPU authorized: `False`.",
            "- Do not launch budget256 6k: budget64 6k failed the tail gate and therefore the prepared launch condition is unmet.",
            "- Do not launch another budget128 repair: AR005 failed the pre-canonical internal/tail screen.",
            "- Keep budget128 6k as mechanism-only true-cell/cell-cap evidence because it passes train-tail but failed frozen canonical no-harm.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": status, "out_json": str(OUT_JSON), "out_md": str(OUT_MD)}, indent=2))


if __name__ == "__main__":
    main()
