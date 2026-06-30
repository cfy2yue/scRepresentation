#!/usr/bin/env python3
"""Summarize observable-info condition-weight smoke vs random control."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_observable_information_condition_weight_smoke_20260629"
OUT_DIR = ROOT / "reports/observable_information_condition_weight_smoke_20260629"
OUT_JSON = OUT_DIR / "latentfm_observable_information_condition_weight_smoke_decision_20260629.json"
OUT_MD = OUT_DIR / "LATENTFM_OBSERVABLE_INFORMATION_CONDITION_WEIGHT_SMOKE_DECISION_20260629.md"
ARMS = {
    "observable": "xverse_obsinfo_condition_weight_observable_2000step_seed42",
    "random": "xverse_obsinfo_condition_weight_random_2000step_seed42",
}


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


def arm_row(arm: str, run_name: str) -> dict[str, Any]:
    run_dir = RUN_ROOT / run_name
    eval_dir = run_dir / "posthoc_eval_internal"
    split_anchor = load_json(eval_dir / "split_group_eval_anchor_internal_ode20.json")
    split_cand = load_json(eval_dir / "split_group_eval_candidate_internal_ode20.json")
    fam_anchor = load_json(eval_dir / "condition_family_eval_anchor_internal_ode20.json")
    fam_cand = load_json(eval_dir / "condition_family_eval_candidate_internal_ode20.json")
    train_exit = read_exit(run_dir / "EXIT_CODE")
    posthoc_exit = read_exit(run_dir / "POSTHOC_EXIT_CODE")
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
    done = train_exit == 0 and posthoc_exit == 0
    return {
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


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = {row["arm"]: row for row in rows}
    if any(row["status"] != "done" for row in rows):
        return {"status": "observable_information_condition_weight_smoke_pending", "gpu_authorized_next": False}
    obs = by["observable"]
    rnd = by["random"]
    cross_adv = delta(obs["cross_pp_delta"], rnd["cross_pp_delta"])
    family_adv = delta(obs["family_pp_delta"], rnd["family_pp_delta"])
    mmd_adv = delta(obs["family_mmd_delta"], rnd["family_mmd_delta"])
    reasons: list[str] = []
    if (cross_adv or 0.0) < 0.005:
        reasons.append("observable_cross_advantage_vs_random_below_0p005")
    if (family_adv or 0.0) < 0.005:
        reasons.append("observable_family_advantage_vs_random_below_0p005")
    if (obs["cross_pp_delta"] or 0.0) < 0.0:
        reasons.append("observable_cross_noharm_vs_anchor_failed")
    if (obs["family_pp_delta"] or 0.0) < 0.0:
        reasons.append("observable_family_noharm_vs_anchor_failed")
    if (obs["family_mmd_delta"] or 0.0) > 0.001:
        reasons.append("observable_family_mmd_harm_gt_0p001")
    status = (
        "observable_information_condition_weight_internal_pass_needs_controls_noharm"
        if not reasons
        else "observable_information_condition_weight_fail_or_mechanism_only_no_gpu"
    )
    return {
        "status": status,
        "gpu_authorized_next": False,
        "reasons": reasons,
        "observable_minus_random_cross_pp_delta": cross_adv,
        "observable_minus_random_family_pp_delta": family_adv,
        "observable_minus_random_family_mmd_delta": mmd_adv,
        "action": "external_audit_then_controls_noharm_if_pass_else_close_or_mutate",
    }


def render(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# LatentFM Observable-Information Condition Weight Smoke Decision",
        "",
        f"Status: `{decision['status']}`",
        "",
        "GPU authorized next: `False`",
        "",
        "## Boundary",
        "",
        "- Summarizes internal train-only split posthoc only.",
        "- Compares observable-information condition loss weights against same-marginal stratified random weights.",
        "- Does not authorize canonical multi, Track C query, checkpoint promotion, or scaling-law claims.",
        "",
        "## Rows",
        "",
        "| arm | status | cross pp delta | family pp delta | family MMD delta | test_single pp delta | family_gene pp delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| `{row['arm']}` | `{row['status']}` | {fmt(row['cross_pp_delta'])} | "
            f"{fmt(row['family_pp_delta'])} | {fmt(row['family_mmd_delta'])} | "
            f"{fmt(row['test_single_pp_delta'])} | {fmt(row['family_gene_pp_delta'])} |"
        )
    lines.extend(["", "## Observable Minus Random", ""])
    for key in [
        "observable_minus_random_cross_pp_delta",
        "observable_minus_random_family_pp_delta",
        "observable_minus_random_family_mmd_delta",
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
            f"- action: `{decision.get('action', 'wait_or_repair')}`",
            "- Passing this internal/random-control gate would still require external audit and a separate frozen no-harm route before any promotion claim.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    rows = [arm_row(arm, run_name) for arm, run_name in ARMS.items()]
    payload = {
        "run_root": str(RUN_ROOT),
        "rows": rows,
        "decision": decide(rows),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "out": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
