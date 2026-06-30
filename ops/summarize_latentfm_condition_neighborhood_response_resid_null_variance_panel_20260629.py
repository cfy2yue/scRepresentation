#!/usr/bin/env python3
"""Summarize pair-shuffle null variance for matched high/low LatentFM splits."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REAL_JSON = ROOT / "reports/condition_neighborhood_response_resid_highlow_smoke_20260629/latentfm_condition_neighborhood_response_resid_highlow_decision_20260629.json"
OUT_DIR = ROOT / "reports/condition_neighborhood_response_resid_null_variance_panel_20260629"
OUT_JSON = OUT_DIR / "latentfm_condition_neighborhood_response_resid_null_variance_panel_20260629.json"
OUT_MD = OUT_DIR / "LATENTFM_CONDITION_NEIGHBORHOOD_RESPONSE_RESID_NULL_VARIANCE_PANEL_20260629.md"
SEEDS = [43, 44, 45, 46]
RUN_ROOT_TEMPLATE = ROOT / "runs/latentfm_condition_neighborhood_response_resid_pairshuffle_seed{seed}_smoke_20260629"
RUN_NAME_TEMPLATE = "xverse_condition_neighborhood_response_resid_pairshuffle_seed{seed}_{arm}_2000step_seed42"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_exit(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def metric(payload: dict[str, Any] | None, group: str, key: str) -> float | None:
    if payload is None:
        return None
    value = ((payload.get("groups") or {}).get(group) or {}).get(key)
    return None if value is None else float(value)


def delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def arm_row(seed: int, arm: str) -> dict[str, Any]:
    run_root = Path(str(RUN_ROOT_TEMPLATE).format(seed=seed))
    run_name = RUN_NAME_TEMPLATE.format(seed=seed, arm=arm)
    run_dir = run_root / run_name
    eval_dir = run_dir / "posthoc_eval_internal"
    split_anchor = load_json(eval_dir / "split_group_eval_anchor_internal_ode20.json")
    split_cand = load_json(eval_dir / "split_group_eval_candidate_internal_ode20.json")
    fam_anchor = load_json(eval_dir / "condition_family_eval_anchor_internal_ode20.json")
    fam_cand = load_json(eval_dir / "condition_family_eval_candidate_internal_ode20.json")
    train_exit = read_exit(run_dir / "EXIT_CODE")
    posthoc_exit = read_exit(run_dir / "POSTHOC_EXIT_CODE")
    done = train_exit == 0 and posthoc_exit == 0
    cross_c = metric(split_cand, "internal_val_cross_background_seen_gene_proxy", "pearson_pert")
    cross_a = metric(split_anchor, "internal_val_cross_background_seen_gene_proxy", "pearson_pert")
    family_c = metric(split_cand, "internal_val_family_gene_proxy", "pearson_pert")
    family_a = metric(split_anchor, "internal_val_family_gene_proxy", "pearson_pert")
    family_mmd_c = metric(split_cand, "internal_val_family_gene_proxy", "test_mmd")
    family_mmd_a = metric(split_anchor, "internal_val_family_gene_proxy", "test_mmd")
    return {
        "seed": seed,
        "arm": arm,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "train_exit": train_exit,
        "posthoc_exit": posthoc_exit,
        "status": "done" if done else "pending_or_failed",
        "cross_pp_delta": delta(cross_c, cross_a),
        "family_pp_delta": delta(family_c, family_a),
        "family_mmd_delta": delta(family_mmd_c, family_mmd_a),
    }


def seed_summary(seed: int) -> dict[str, Any]:
    rows = [arm_row(seed, "high"), arm_row(seed, "low")]
    by = {row["arm"]: row for row in rows}
    high = by["high"]
    low = by["low"]
    cross_gap = delta(high["cross_pp_delta"], low["cross_pp_delta"])
    family_gap = delta(high["family_pp_delta"], low["family_pp_delta"])
    mmd_gap = delta(high["family_mmd_delta"], low["family_mmd_delta"])
    done = all(row["status"] == "done" for row in rows)
    return {
        "seed": seed,
        "status": "done" if done else "pending_or_failed",
        "rows": rows,
        "high_minus_low_cross_pp_delta": cross_gap,
        "high_minus_low_family_pp_delta": family_gap,
        "high_minus_low_family_mmd_delta": mmd_gap,
        "abs_high_minus_low_cross_pp_delta": abs(cross_gap) if cross_gap is not None else None,
        "abs_high_minus_low_family_pp_delta": abs(family_gap) if family_gap is not None else None,
    }


def decide(real: dict[str, Any] | None, seeds: list[dict[str, Any]]) -> dict[str, Any]:
    if real is None:
        return {"status": "null_variance_panel_missing_real", "action": "repair_inputs", "gpu_authorized_next": False}
    real_checks = (real.get("decision") or {}).get("checks") or {}
    real_cross = float(real_checks.get("high_minus_low_cross_pp_delta") or 0.0)
    real_family = float(real_checks.get("high_minus_low_family_pp_delta") or 0.0)
    if any(row["status"] != "done" for row in seeds):
        return {
            "status": "condition_neighborhood_response_resid_null_variance_panel_pending",
            "action": "wait_for_seed_completion",
            "gpu_authorized_next": False,
            "real_cross": real_cross,
            "real_family": real_family,
        }
    abs_cross = [float(row["abs_high_minus_low_cross_pp_delta"] or 0.0) for row in seeds]
    abs_family = [float(row["abs_high_minus_low_family_pp_delta"] or 0.0) for row in seeds]
    cross_p95 = percentile(abs_cross, 0.95)
    family_p95 = percentile(abs_family, 0.95)
    cross_max = max(abs_cross) if abs_cross else None
    family_max = max(abs_family) if abs_family else None
    cross_margin_over_p95 = real_cross - float(cross_p95 or 0.0)
    family_margin_over_p95 = real_family - float(family_p95 or 0.0)
    reasons: list[str] = []
    if cross_margin_over_p95 < 0.02:
        reasons.append("real_cross_margin_over_null_p95_below_0p02")
    if family_margin_over_p95 < 0.02:
        reasons.append("real_family_margin_over_null_p95_below_0p02")
    status = (
        "condition_neighborhood_response_resid_null_variance_panel_blocks_support_axis"
        if reasons
        else "condition_neighborhood_response_resid_null_variance_panel_axis_margin_pass"
    )
    return {
        "status": status,
        "action": "use_null_threshold_for_future_axes_not_promotion",
        "gpu_authorized_next": False,
        "reasons": reasons,
        "real_cross": real_cross,
        "real_family": real_family,
        "abs_null_cross": abs_cross,
        "abs_null_family": abs_family,
        "null_cross_p95": cross_p95,
        "null_family_p95": family_p95,
        "null_cross_max": cross_max,
        "null_family_max": family_max,
        "real_cross_margin_over_null_p95": cross_margin_over_p95,
        "real_family_margin_over_null_p95": family_margin_over_p95,
        "future_axis_required_cross_gap": float(cross_p95 or 0.0) + 0.02,
        "future_axis_required_family_gap": float(family_p95 or 0.0) + 0.02,
    }


def render(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# LatentFM Condition-Neighborhood Response-Resid Null Variance Panel",
        "",
        f"Status: `{decision['status']}`",
        "",
        "GPU authorized next: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only summary of pair-shuffle high/low null smokes.",
        "- Uses train-only/internal posthoc outputs only.",
        "- This is a protocol-calibration artifact; it does not authorize no-harm, canonical multi, Track C query, or promotion.",
        "",
        "## Real Reference",
        "",
        f"- real high-minus-low cross pp: `{fmt(decision.get('real_cross'))}`",
        f"- real high-minus-low family pp: `{fmt(decision.get('real_family'))}`",
        "",
        "## Null Seeds",
        "",
        "| seed | status | high-low cross pp | abs cross | high-low family pp | abs family | high-low family MMD |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["seeds"]:
        lines.append(
            f"| {row['seed']} | `{row['status']}` | {fmt(row['high_minus_low_cross_pp_delta'])} | "
            f"{fmt(row['abs_high_minus_low_cross_pp_delta'])} | {fmt(row['high_minus_low_family_pp_delta'])} | "
            f"{fmt(row['abs_high_minus_low_family_pp_delta'])} | {fmt(row['high_minus_low_family_mmd_delta'])} |"
        )
    lines.extend(["", "## Null Calibration", ""])
    for key in [
        "null_cross_p95",
        "null_family_p95",
        "null_cross_max",
        "null_family_max",
        "real_cross_margin_over_null_p95",
        "real_family_margin_over_null_p95",
        "future_axis_required_cross_gap",
        "future_axis_required_family_gap",
    ]:
        if key in decision:
            lines.append(f"- {key}: `{fmt(decision.get(key))}`")
    if decision.get("reasons"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {reason}" for reason in decision["reasons"])
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- action: `{decision['action']}`",
            "- Future high/low scaling axes must be compared against this null panel or a stricter axis-specific null before GPU promotion logic.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    real = load_json(REAL_JSON)
    seeds = [seed_summary(seed) for seed in SEEDS]
    payload = {
        "real_json": str(REAL_JSON),
        "seeds": seeds,
        "decision": decide(real, seeds),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "out": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
