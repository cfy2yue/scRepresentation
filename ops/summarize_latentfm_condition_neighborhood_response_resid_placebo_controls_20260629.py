#!/usr/bin/env python3
"""Summarize response-residualized support real high/low vs placebo controls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REAL_JSON = ROOT / "reports/condition_neighborhood_response_resid_highlow_smoke_20260629/latentfm_condition_neighborhood_response_resid_highlow_decision_20260629.json"
OUT_DIR = ROOT / "reports/condition_neighborhood_response_resid_placebo_controls_20260629"
OUT_JSON = OUT_DIR / "latentfm_condition_neighborhood_response_resid_placebo_controls_20260629.json"
OUT_MD = OUT_DIR / "LATENTFM_CONDITION_NEIGHBORHOOD_RESPONSE_RESID_PLACEBO_CONTROLS_20260629.md"
PLACEBO_RUNS = {
    43: ROOT / "runs/latentfm_condition_neighborhood_response_resid_pairshuffle_seed43_smoke_20260629",
    44: ROOT / "runs/latentfm_condition_neighborhood_response_resid_pairshuffle_seed44_smoke_20260629",
}
RUN_PREFIX_TEMPLATE = "xverse_condition_neighborhood_response_resid_pairshuffle_seed{seed}_{arm}_2000step_seed42"


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


def arm_row(run_root: Path, seed: int, arm: str) -> dict[str, Any]:
    run_name = RUN_PREFIX_TEMPLATE.format(seed=seed, arm=arm)
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
    test_single_c = metric(fam_cand, "test_single", "pearson_pert")
    test_single_a = metric(fam_anchor, "test_single", "pearson_pert")
    family_gene_c = metric(fam_cand, "family_gene", "pearson_pert")
    family_gene_a = metric(fam_anchor, "family_gene", "pearson_pert")
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
        "test_single_pp_delta": delta(test_single_c, test_single_a),
        "family_gene_pp_delta": delta(family_gene_c, family_gene_a),
    }


def seed_summary(seed: int, run_root: Path) -> dict[str, Any]:
    rows = [arm_row(run_root, seed, "high"), arm_row(run_root, seed, "low")]
    by = {row["arm"]: row for row in rows}
    high = by["high"]
    low = by["low"]
    return {
        "seed": seed,
        "status": "done" if all(row["status"] == "done" for row in rows) else "pending_or_failed",
        "rows": rows,
        "high_minus_low_cross_pp_delta": delta(high["cross_pp_delta"], low["cross_pp_delta"]),
        "high_minus_low_family_pp_delta": delta(high["family_pp_delta"], low["family_pp_delta"]),
        "high_minus_low_family_mmd_delta": delta(high["family_mmd_delta"], low["family_mmd_delta"]),
    }


def decide(real: dict[str, Any] | None, placebo: list[dict[str, Any]]) -> dict[str, Any]:
    if real is None:
        return {"status": "condition_neighborhood_response_resid_placebo_controls_missing_real", "action": "wait_or_repair", "gpu_authorized_next": False}
    real_checks = (real.get("decision") or {}).get("checks") or {}
    real_cross = real_checks.get("high_minus_low_cross_pp_delta")
    real_family = real_checks.get("high_minus_low_family_pp_delta")
    if any(row["status"] != "done" for row in placebo):
        return {
            "status": "condition_neighborhood_response_resid_placebo_controls_pending_wait_no_polling",
            "action": "wait_for_placebo_completion",
            "gpu_authorized_next": False,
            "real_cross": real_cross,
            "real_family": real_family,
        }
    reasons: list[str] = []
    placebo_cross = [row["high_minus_low_cross_pp_delta"] for row in placebo]
    placebo_family = [row["high_minus_low_family_pp_delta"] for row in placebo]
    max_abs_cross = max(abs(float(x or 0.0)) for x in placebo_cross)
    max_abs_family = max(abs(float(x or 0.0)) for x in placebo_family)
    real_cross_abs = abs(float(real_cross or 0.0))
    real_family_abs = abs(float(real_family or 0.0))
    cross_ratio = max_abs_cross / real_cross_abs if real_cross_abs else None
    family_ratio = max_abs_family / real_family_abs if real_family_abs else None
    if max_abs_cross > 0.01 and (cross_ratio is None or cross_ratio > 0.33):
        reasons.append("placebo_cross_did_not_collapse")
    if max_abs_family > 0.01 and (family_ratio is None or family_ratio > 0.33):
        reasons.append("placebo_family_did_not_collapse")
    if reasons:
        status = "condition_neighborhood_response_resid_placebo_controls_fail_demote_mechanism"
        action = "close_or_mutate_support_axis_before_noharm"
    else:
        status = "condition_neighborhood_response_resid_placebo_controls_pass_prepare_noharm_or_extra_control"
        action = "run_extra_control_or_frozen_dual_baseline_noharm"
    return {
        "status": status,
        "action": action,
        "gpu_authorized_next": False,
        "reasons": reasons,
        "real_cross": real_cross,
        "real_family": real_family,
        "placebo_cross": placebo_cross,
        "placebo_family": placebo_family,
        "max_abs_placebo_cross": max_abs_cross,
        "max_abs_placebo_family": max_abs_family,
        "placebo_over_real_cross": cross_ratio,
        "placebo_over_real_family": family_ratio,
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Condition-Neighborhood Response-Resid Placebo Controls",
        "",
        f"Status: `{payload['decision']['status']}`",
        "",
        "GPU authorized next: `False`",
        "",
        "## Boundary",
        "",
        "- Summarizes real response-residualized high/low and pair-shuffle placebo controls.",
        "- Reads only train-only/internal posthoc outputs.",
        "- Does not use canonical multi, Track C query, or checkpoint selection.",
        "",
        "## Real Signal",
        "",
        f"- high-minus-low cross pp: `{fmt(payload['decision'].get('real_cross'))}`",
        f"- high-minus-low family pp: `{fmt(payload['decision'].get('real_family'))}`",
        "",
        "## Placebo Seeds",
        "",
        "| seed | status | high-low cross pp | high-low family pp | high-low family MMD |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in payload["placebo"]:
        lines.append(
            f"| {row['seed']} | `{row['status']}` | {fmt(row['high_minus_low_cross_pp_delta'])} | "
            f"{fmt(row['high_minus_low_family_pp_delta'])} | {fmt(row['high_minus_low_family_mmd_delta'])} |"
        )
    decision = payload["decision"]
    lines.extend(["", "## Decision", "", f"- action: `{decision['action']}`"])
    for key in ["max_abs_placebo_cross", "placebo_over_real_cross", "max_abs_placebo_family", "placebo_over_real_family"]:
        if key in decision:
            lines.append(f"- {key}: `{fmt(decision[key])}`")
    for reason in decision.get("reasons", []) or []:
        lines.append(f"- reason: `{reason}`")
    lines.extend(["", "## Outputs", "", f"- JSON: `{OUT_JSON}`"])
    return "\n".join(lines) + "\n"


def main() -> int:
    real = load_json(REAL_JSON)
    placebo = [seed_summary(seed, run_root) for seed, run_root in PLACEBO_RUNS.items()]
    payload = {
        "real_json": str(REAL_JSON),
        "placebo": placebo,
        "decision": decide(real, placebo),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "out": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
