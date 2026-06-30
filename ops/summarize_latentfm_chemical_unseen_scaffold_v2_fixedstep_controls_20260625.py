#!/usr/bin/env python3
"""Decision summary for chemical unseen-scaffold V2 fixed-step controls."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_chemical_unseen_scaffold_v2_fixedstep_controls_20260625"
OUT_JSON = ROOT / "reports/latentfm_chemical_unseen_scaffold_v2_fixedstep_controls_decision_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_CHEMICAL_UNSEEN_SCAFFOLD_V2_FIXEDSTEP_CONTROLS_DECISION_20260625.md"
SEEDS = (43, 44)
ARMS = ("real_morgan512", "shuffled_morgan512", "random_morgan512")


def run_dir(arm: str, seed: int) -> Path:
    return RUN_ROOT / f"xverse_chemical_unseen_scaffold_v2_{arm}_fixedlatest_2500_seed{seed}"


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


def group(payload: dict[str, Any] | None, name: str) -> dict[str, Any]:
    if not payload:
        return {}
    return (payload.get("groups") or {}).get(name) or {}


def val(payload: dict[str, Any] | None, group_name: str, key: str) -> float | None:
    raw = group(payload, group_name).get(key)
    return None if raw is None else float(raw)


def delta(candidate: float | None, anchor: float | None) -> float | None:
    if candidate is None or anchor is None:
        return None
    return candidate - anchor


def summarize_arm_seed(arm: str, seed: int) -> dict[str, Any]:
    rd = run_dir(arm, seed)
    eval_dir = rd / "posthoc_eval_internal"
    fam_anchor = load_json(eval_dir / "condition_family_eval_anchor_internal_ode20.json")
    fam_candidate = load_json(eval_dir / "condition_family_eval_candidate_internal_ode20.json")
    train_exit = read_exit(rd / "EXIT_CODE")
    posthoc_exit = read_exit(rd / "POSTHOC_EXIT_CODE")
    ckpt = (rd / "FIXED_CANDIDATE_CHECKPOINT").read_text(encoding="utf-8").strip() if (rd / "FIXED_CANDIDATE_CHECKPOINT").is_file() else None
    metrics = {}
    for name in ("test_all", "family_gene", "family_drug", "type_drug"):
        metrics[name] = {
            "delta_pearson_pert": delta(val(fam_candidate, name, "pearson_pert"), val(fam_anchor, name, "pearson_pert")),
            "delta_mmd": delta(val(fam_candidate, name, "test_mmd"), val(fam_anchor, name, "test_mmd")),
            "n_conds": group(fam_candidate, name).get("n_conds"),
        }
    reasons = []
    if train_exit != 0 or posthoc_exit != 0:
        reasons.append("train_or_posthoc_not_complete")
    if not ckpt or not ckpt.endswith("/latest.pt"):
        reasons.append("fixed_latest_checkpoint_not_recorded")
    if metrics["family_drug"]["delta_pearson_pert"] is None or metrics["family_drug"]["delta_pearson_pert"] < 0.005:
        reasons.append("family_drug_pp_delta_lt_0p005")
    if metrics["type_drug"]["delta_pearson_pert"] is None or metrics["type_drug"]["delta_pearson_pert"] < 0.005:
        reasons.append("type_drug_pp_delta_lt_0p005")
    if metrics["test_all"]["delta_pearson_pert"] is None or metrics["test_all"]["delta_pearson_pert"] < 0.005:
        reasons.append("test_all_pp_delta_lt_0p005")
    if metrics["family_gene"]["delta_pearson_pert"] is None or metrics["family_gene"]["delta_pearson_pert"] < -0.002:
        reasons.append("family_gene_pp_delta_lt_minus_0p002")
    for name in ("test_all", "family_gene", "family_drug", "type_drug"):
        if metrics[name]["delta_mmd"] is None or metrics[name]["delta_mmd"] > 0.00025:
            reasons.append(f"{name}_mmd_delta_gt_0p00025")
    return {
        "arm": arm,
        "seed": seed,
        "run_dir": str(rd),
        "train_exit": train_exit,
        "posthoc_exit": posthoc_exit,
        "fixed_candidate_checkpoint": ckpt,
        "metrics": metrics,
        "status": "pass" if not reasons else "pending_or_fail",
        "reasons": reasons,
    }


def median(xs: list[float]) -> float | None:
    return None if not xs else float(statistics.median(xs))


def metric_values(rows: list[dict[str, Any]], arm: str, group_name: str, key: str) -> list[float]:
    out = []
    for row in rows:
        if row["arm"] != arm:
            continue
        raw = row["metrics"][group_name][key]
        if raw is not None:
            out.append(float(raw))
    return out


def fmt(x: Any) -> str:
    return "NA" if x is None else f"{float(x):+.6f}"


def main() -> int:
    rows = [summarize_arm_seed(arm, seed) for arm in ARMS for seed in SEEDS]
    complete = all(row["train_exit"] == 0 and row["posthoc_exit"] == 0 for row in rows)
    arm_summary = {}
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        arm_summary[arm] = {
            "pass_count": sum(1 for r in arm_rows if r["status"] == "pass"),
            "median_family_drug_pp_delta": median(metric_values(rows, arm, "family_drug", "delta_pearson_pert")),
            "median_type_drug_pp_delta": median(metric_values(rows, arm, "type_drug", "delta_pearson_pert")),
            "median_test_all_pp_delta": median(metric_values(rows, arm, "test_all", "delta_pearson_pert")),
            "median_family_gene_pp_delta": median(metric_values(rows, arm, "family_gene", "delta_pearson_pert")),
        }

    real = arm_summary["real_morgan512"]
    control_best_drug = max(
        arm_summary["shuffled_morgan512"]["median_family_drug_pp_delta"] or -999.0,
        arm_summary["random_morgan512"]["median_family_drug_pp_delta"] or -999.0,
    )
    real_margin = None if real["median_family_drug_pp_delta"] is None else real["median_family_drug_pp_delta"] - control_best_drug
    criteria = {
        "complete_full_2x3_matrix": complete,
        "real_pass_count_2_of_2": real["pass_count"] == 2,
        "real_median_family_drug_pp_ge_0p008": real["median_family_drug_pp_delta"] is not None and real["median_family_drug_pp_delta"] >= 0.008,
        "real_margin_over_best_control_ge_0p005": real_margin is not None and real_margin >= 0.005,
        "controls_do_not_pass": arm_summary["shuffled_morgan512"]["pass_count"] == 0 and arm_summary["random_morgan512"]["pass_count"] == 0,
    }
    if not complete:
        status = "chemical_unseen_scaffold_v2_fixedstep_controls_pending"
        action = "wait_without_frequent_polling_or_launch_remaining_arms_if_protocol_approved"
    elif all(criteria.values()):
        status = "chemical_unseen_scaffold_v2_fixedstep_controls_pass_mechanism_next_external_review"
        action = "external review before any larger chemical scaling matrix"
    else:
        status = "chemical_unseen_scaffold_v2_fixedstep_controls_fail_close"
        action = "close chemical scaffold GPU branch; preserve as scaling failure-map evidence"

    payload = {
        "status": status,
        "gpu_authorized": False,
        "criteria": criteria,
        "arm_summary": arm_summary,
        "real_margin_over_best_control_family_drug_pp": real_margin,
        "rows": rows,
        "action": action,
        "boundary": {
            "candidate_checkpoint_policy": "fixed latest.pt only",
            "canonical_multi_used": False,
            "trackc_query_used": False,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Chemical Unseen-Scaffold V2 Fixed-Step Controls Decision",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- Summarizes V2 seed43/44 real/shuffled/random descriptor controls.",
        "- Candidate checkpoint policy is fixed `latest.pt`; `best.pt` is not used for adjudication.",
        "- Uses train-only/internal V2 split posthoc outputs only.",
        "- Does not read canonical multi or Track C query.",
        "",
        "## Arm Summary",
        "",
        "| arm | pass count | median all pp | median drug pp | median type-drug pp | median gene pp |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        s = arm_summary[arm]
        lines.append(
            f"| `{arm}` | {s['pass_count']}/2 | {fmt(s['median_test_all_pp_delta'])} | "
            f"{fmt(s['median_family_drug_pp_delta'])} | {fmt(s['median_type_drug_pp_delta'])} | "
            f"{fmt(s['median_family_gene_pp_delta'])} |"
        )
    lines += [
        "",
        "## Rows",
        "",
        "| arm | seed | status | all pp | drug pp | type-drug pp | gene pp | drug MMD | reasons |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        m = row["metrics"]
        lines.append(
            f"| `{row['arm']}` | {row['seed']} | `{row['status']}` | "
            f"{fmt(m['test_all']['delta_pearson_pert'])} | {fmt(m['family_drug']['delta_pearson_pert'])} | "
            f"{fmt(m['type_drug']['delta_pearson_pert'])} | {fmt(m['family_gene']['delta_pearson_pert'])} | "
            f"{fmt(m['family_drug']['delta_mmd'])} | {', '.join(row['reasons']) or 'none'} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- real margin over best control family_drug pp: `{fmt(real_margin)}`",
        f"- criteria: `{criteria}`",
        f"- action: `{action}`",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
