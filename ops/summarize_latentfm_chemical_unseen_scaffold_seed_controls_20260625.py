#!/usr/bin/env python3
"""Seed-control decision for the chemical unseen-scaffold preliminary pass."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_chemical_unseen_drug_scaffold_smokes_20260625"
OUT_JSON = ROOT / "reports/latentfm_chemical_unseen_scaffold_seed_control_decision_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_CHEMICAL_UNSEEN_SCAFFOLD_SEED_CONTROL_DECISION_20260625.md"
SEEDS = (42, 43, 44)


def run_dir(seed: int) -> Path:
    return RUN_ROOT / f"xverse_chemical_unseen_scaffold_morgan512_2500_seed{seed}"


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


def value(payload: dict[str, Any] | None, name: str, key: str) -> float | None:
    raw = group(payload, name).get(key)
    return None if raw is None else float(raw)


def delta(candidate: float | None, anchor: float | None) -> float | None:
    if candidate is None or anchor is None:
        return None
    return candidate - anchor


def summarize_seed(seed: int) -> dict[str, Any]:
    rd = run_dir(seed)
    eval_dir = rd / "posthoc_eval_internal"
    fam_anchor = load_json(eval_dir / "condition_family_eval_anchor_internal_ode20.json")
    fam_candidate = load_json(eval_dir / "condition_family_eval_candidate_internal_ode20.json")
    train_exit = read_exit(rd / "EXIT_CODE")
    posthoc_exit = read_exit(rd / "POSTHOC_EXIT_CODE")
    metrics = {}
    for name in ("test_all", "family_gene", "family_drug", "type_drug"):
        metrics[name] = {
            "delta_pearson_pert": delta(value(fam_candidate, name, "pearson_pert"), value(fam_anchor, name, "pearson_pert")),
            "delta_mmd": delta(value(fam_candidate, name, "test_mmd"), value(fam_anchor, name, "test_mmd")),
            "n_conds": group(fam_candidate, name).get("n_conds"),
        }
    reasons = []
    if train_exit != 0 or posthoc_exit != 0:
        reasons.append("train_or_posthoc_not_complete")
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
        "seed": seed,
        "run_dir": str(rd),
        "train_exit": train_exit,
        "posthoc_exit": posthoc_exit,
        "metrics": metrics,
        "status": "pass" if not reasons else "pending_or_fail",
        "reasons": reasons,
    }


def median(values: list[float]) -> float | None:
    return None if not values else float(statistics.median(values))


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):+.6f}"


def main() -> int:
    rows = [summarize_seed(seed) for seed in SEEDS]
    complete = all(row["train_exit"] == 0 and row["posthoc_exit"] == 0 for row in rows)
    pass_rows = [row for row in rows if row["status"] == "pass"]
    drug_pp = [row["metrics"]["family_drug"]["delta_pearson_pert"] for row in rows if row["metrics"]["family_drug"]["delta_pearson_pert"] is not None]
    all_pp = [row["metrics"]["test_all"]["delta_pearson_pert"] for row in rows if row["metrics"]["test_all"]["delta_pearson_pert"] is not None]
    criteria = {
        "complete_3_seeds": complete,
        "pass_count_ge_2": len(pass_rows) >= 2,
        "median_family_drug_pp_ge_0p008": (median(drug_pp) is not None and median(drug_pp) >= 0.008),
        "median_test_all_pp_ge_0p005": (median(all_pp) is not None and median(all_pp) >= 0.005),
        "all_gene_pp_ge_minus_0p002": all(
            row["metrics"]["family_gene"]["delta_pearson_pert"] is not None
            and row["metrics"]["family_gene"]["delta_pearson_pert"] >= -0.002
            for row in rows
        ),
        "all_key_mmd_delta_le_0p00025": all(
            row["metrics"][name]["delta_mmd"] is not None and row["metrics"][name]["delta_mmd"] <= 0.00025
            for row in rows
            for name in ("test_all", "family_gene", "family_drug", "type_drug")
        ),
    }
    if not complete:
        status = "chemical_unseen_scaffold_seed_controls_pending"
        action = "wait_without_polling_until_seed_controls_complete"
    elif all(criteria.values()):
        status = "chemical_unseen_scaffold_seed_controls_pass_next_controls"
        action = "run fixed-step and negative-control layer before broader scaling claim"
    else:
        status = "chemical_unseen_scaffold_seed_controls_fail_close_or_mutate"
        action = "close or mutate before more chemical-scaling GPU"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "criteria": criteria,
        "summary": {
            "pass_count": len(pass_rows),
            "median_family_drug_pp_delta": median(drug_pp),
            "median_test_all_pp_delta": median(all_pp),
        },
        "rows": rows,
        "action": action,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM Chemical Unseen-Scaffold Seed-Control Decision",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- Summarizes same-split seed42/43/44 unseen-scaffold controls.",
        "- Uses only train-only/internal split posthoc outputs.",
        "- Does not read canonical multi or Track C query.",
        "- Does not authorize deployable claims or final scaling-law claims.",
        "",
        "## Seeds",
        "",
        "| seed | status | all pp | drug pp | type-drug pp | gene pp | drug MMD | reasons |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        m = row["metrics"]
        lines.append(
            f"| {row['seed']} | `{row['status']}` | {fmt(m['test_all']['delta_pearson_pert'])} | "
            f"{fmt(m['family_drug']['delta_pearson_pert'])} | {fmt(m['type_drug']['delta_pearson_pert'])} | "
            f"{fmt(m['family_gene']['delta_pearson_pert'])} | {fmt(m['family_drug']['delta_mmd'])} | "
            f"{', '.join(row['reasons']) or 'none'} |"
        )
    lines += [
        "",
        "## Criteria",
        "",
        f"- pass count: `{len(pass_rows)}/3`",
        f"- median family_drug pp delta: `{fmt(payload['summary']['median_family_drug_pp_delta'])}`",
        f"- median test_all pp delta: `{fmt(payload['summary']['median_test_all_pp_delta'])}`",
        f"- criteria: `{criteria}`",
        "",
        "## Decision",
        "",
        f"- action: `{action}`",
        "- GPU authorized by this report: `False`",
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
