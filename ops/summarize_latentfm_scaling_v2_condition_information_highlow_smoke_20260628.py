#!/usr/bin/env python3
"""Summarize scaling-v2 condition-information high/low GPU smoke."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_scaling_v2_condition_information_highlow_smoke_20260628"
OUT_JSON = ROOT / "reports/scaling_v2_condition_information_highlow_smoke_20260628/latentfm_scaling_v2_condition_information_highlow_decision_20260628.json"
OUT_MD = ROOT / "reports/scaling_v2_condition_information_highlow_smoke_20260628/LATENTFM_SCALING_V2_CONDITION_INFORMATION_HIGHLOW_DECISION_20260628.md"
REPORT_TITLE = "LatentFM Scaling V2 Condition-Information High/Low Decision"
ARMS = {
    "high": "xverse_scaling_v2_info_high_2000step_seed42",
    "low": "xverse_scaling_v2_info_low_2000step_seed42",
}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_exit(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        return int(path.read_text().strip())
    except Exception:
        return None


def metric(payload: dict[str, Any] | None, group: str, key: str) -> float | None:
    if not payload:
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


def arm_row(run_root: Path, arm: str, run_name: str) -> dict[str, Any]:
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
    if any(row["status"] != "done" for row in rows):
        return {
            "status": "scaling_v2_condition_information_highlow_pending_wait_no_polling",
            "action": "wait_for_natural_completion",
            "gpu_authorized_next": False,
        }
    by = {row["arm"]: row for row in rows}
    high, low = by["high"], by["low"]
    checks = {
        "high_minus_low_cross_pp_delta": delta(high["cross_pp_delta"], low["cross_pp_delta"]),
        "high_minus_low_family_pp_delta": delta(high["family_pp_delta"], low["family_pp_delta"]),
        "high_minus_low_family_mmd_delta": delta(high["family_mmd_delta"], low["family_mmd_delta"]),
        "high_cross_pp_delta": high["cross_pp_delta"],
        "high_family_pp_delta": high["family_pp_delta"],
        "high_family_mmd_delta": high["family_mmd_delta"],
    }
    reasons = []
    if (checks["high_minus_low_cross_pp_delta"] or -999.0) <= 0.005:
        reasons.append("high_cross_advantage_too_small")
    if (checks["high_minus_low_family_pp_delta"] or -999.0) <= 0.005:
        reasons.append("high_family_advantage_too_small")
    if (checks["high_minus_low_family_mmd_delta"] or 999.0) > 0.002:
        reasons.append("high_family_mmd_harm_vs_low")
    if (checks["high_cross_pp_delta"] or -999.0) <= 0.0:
        reasons.append("high_cross_noharm_vs_anchor_failed")
    if (checks["high_family_pp_delta"] or -999.0) <= 0.0:
        reasons.append("high_family_noharm_vs_anchor_failed")
    if (checks["high_family_mmd_delta"] or 999.0) > 0.002:
        reasons.append("high_family_mmd_noharm_vs_anchor_failed")
    if reasons:
        status = "scaling_v2_condition_information_highlow_fail_or_mechanism_only_no_gpu"
        action = "close_or_mutate_after_review"
        gpu_next = False
    else:
        status = "scaling_v2_condition_information_highlow_internal_pass_needs_placebo_noharm"
        action = "design_pair_label_shuffle_or_random_placebo_before_any_promotion"
        gpu_next = False
    return {
        "status": status,
        "action": action,
        "gpu_authorized_next": gpu_next,
        "reasons": reasons,
        "checks": checks,
    }


def render(payload: dict[str, Any], *, out_json: Path, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"Status: `{payload['decision']['status']}`",
        "",
        "GPU authorized next: `False`",
        "",
        "## Boundary",
        "",
        "- Summarizes the bounded high-vs-low mechanism smoke only.",
        "- Uses train-only/internal split groups; no canonical multi or Track C query selection.",
        "- Even an internal pass only authorizes a placebo/random-label gate, not promotion.",
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
    decision = payload["decision"]
    lines.extend(["", "## Decision", "", f"- action: `{decision['action']}`"])
    if decision.get("reasons"):
        lines.extend(f"- reason: `{reason}`" for reason in decision["reasons"])
    for key, value in (decision.get("checks") or {}).items():
        lines.append(f"- {key}: `{fmt(value)}`")
    lines.extend(["", "## Outputs", "", f"- JSON: `{out_json}`"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--report-dir", type=Path, default=OUT_MD.parent)
    parser.add_argument("--run-prefix", default="xverse_scaling_v2_info")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title", default=REPORT_TITLE)
    parser.add_argument("--stem", default="latentfm_scaling_v2_condition_information_highlow_decision_20260628")
    args = parser.parse_args()

    arms = {
        "high": f"{args.run_prefix}_high_{args.steps}step_seed{args.seed}",
        "low": f"{args.run_prefix}_low_{args.steps}step_seed{args.seed}",
    }
    rows = [arm_row(args.run_root, arm, run_name) for arm, run_name in arms.items()]
    payload = {
        "rows": rows,
        "decision": decide(rows),
        "run_root": str(args.run_root),
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.report_dir / f"{args.stem}.json"
    out_md = args.report_dir / f"{args.stem}.md"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    out_md.write_text(render(payload, out_json=out_json, title=args.title), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "out": str(out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
