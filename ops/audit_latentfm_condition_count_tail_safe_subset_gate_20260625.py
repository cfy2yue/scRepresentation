#!/usr/bin/env python3
"""CPU-only tail-safe subset gate for condition-count scaling.

This gate asks whether the failed cap120-vs-cap30 condition-count signal can be
reopened by a simple, predeclared, biologically/provenance-motivated subset
policy rather than posthoc cherry-picking.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean
from typing import Any, Callable

ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
INPUT_JSON = REPORTS / "latentfm_scaling_mixed_effect_lodo_condition_count_gate_20260624.json"
OUT_JSON = REPORTS / "latentfm_condition_count_tail_safe_subset_gate_20260625.json"
OUT_MD = REPORTS / "LATENTFM_CONDITION_COUNT_TAIL_SAFE_SUBSET_GATE_20260625.md"

SEED = 20260625


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def weighted_mean(rows: list[dict[str, Any]], key: str) -> float:
    n = sum(int(r.get("n") or 0) for r in rows)
    if n <= 0:
        return 0.0
    return sum(float(r.get(key) or 0.0) * int(r.get("n") or 0) for r in rows) / n


def bootstrap_ci(rows: list[dict[str, Any]], n_boot: int = 2000) -> dict[str, Any]:
    rng = random.Random(SEED)
    if not rows:
        return {"ci": [None, None], "p_le_zero": None, "n_boot": 0}
    vals = []
    for _ in range(n_boot):
        sample = [rng.choice(rows) for _ in rows]
        vals.append(weighted_mean(sample, "pp_delta_mean"))
    vals.sort()
    return {
        "ci": [vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]],
        "p_le_zero": sum(v <= 0.0 for v in vals) / len(vals),
        "n_boot": n_boot,
    }


def leave_one_min(rows: list[dict[str, Any]]) -> float | None:
    if len(rows) <= 1:
        return None
    vals = []
    for idx in range(len(rows)):
        kept = [row for j, row in enumerate(rows) if j != idx]
        vals.append(weighted_mean(kept, "pp_delta_mean"))
    return min(vals)


def evaluate(name: str, rows: list[dict[str, Any]], description: str) -> dict[str, Any]:
    n_conditions = sum(int(r.get("n") or 0) for r in rows)
    dataset_count = len(rows)
    pp = weighted_mean(rows, "pp_delta_mean")
    mmd = weighted_mean(rows, "mmd_delta_mean")
    ds_min = min((float(r["pp_delta_mean"]) for r in rows), default=0.0)
    neg_tail_count = sum(float(r["pp_delta_mean"]) < -0.020 for r in rows)
    severe_tail_count = sum(float(r["pp_delta_mean"]) < -0.050 for r in rows)
    max_weight = max((int(r.get("n") or 0) / n_conditions for r in rows), default=1.0)
    boot = bootstrap_ci(rows)
    lo = boot["ci"][0]
    lo_ok = lo is not None and float(lo) > 0.0
    loo = leave_one_min(rows)
    reasons = []
    if dataset_count < 5:
        reasons.append("too_few_datasets")
    if n_conditions < 60:
        reasons.append("too_few_conditions")
    if pp < 0.010:
        reasons.append("weighted_pp_below_0p010")
    if mmd > 0.001:
        reasons.append("weighted_mmd_above_0p001")
    if ds_min < -0.020:
        reasons.append("dataset_tail_below_minus_0p020")
    if neg_tail_count > 0:
        reasons.append("negative_dataset_tail_present")
    if severe_tail_count > 0:
        reasons.append("severe_dataset_tail_present")
    if not lo_ok:
        reasons.append("bootstrap_ci_lower_not_positive")
    if loo is None or loo < 0.005:
        reasons.append("leave_one_dataset_min_below_0p005")
    if max_weight > 0.35:
        reasons.append("single_dataset_weight_above_0p35")
    return {
        "policy": name,
        "description": description,
        "pass_gate": not reasons,
        "reasons": reasons,
        "dataset_count": dataset_count,
        "n_conditions": n_conditions,
        "weighted_pp_delta": pp,
        "weighted_mmd_delta": mmd,
        "dataset_min_pp_delta": ds_min,
        "negative_tail_count": neg_tail_count,
        "severe_tail_count": severe_tail_count,
        "bootstrap": boot,
        "leave_one_dataset_min_pp_delta": loo,
        "max_dataset_weight": max_weight,
        "datasets": [r["dataset"] for r in rows],
    }


def main() -> int:
    data = load_json(INPUT_JSON)
    rows = list(data.get("dataset_rows") or [])
    policies: list[tuple[str, str, Callable[[dict[str, Any]], bool]]] = [
        ("all_completed", "all completed cap120-vs-cap30 rows", lambda r: True),
        ("source_verified", "datasets with source_quality == source_verified", lambda r: r.get("source_quality") == "source_verified"),
        ("positive_cap_gain", "datasets where cap120 adds conditions over cap30", lambda r: int(r.get("cap_gain") or 0) > 0),
        (
            "source_verified_positive_cap_gain",
            "source-verified datasets with positive condition-count gain",
            lambda r: r.get("source_quality") == "source_verified" and int(r.get("cap_gain") or 0) > 0,
        ),
        ("crispri_only", "CRISPRi-only perturbation type", lambda r: r.get("perturbation_type") == "CRISPRi"),
        (
            "source_verified_crispri",
            "source-verified CRISPRi datasets",
            lambda r: r.get("source_quality") == "source_verified" and r.get("perturbation_type") == "CRISPRi",
        ),
        (
            "source_verified_crispri_positive_cap_gain",
            "source-verified CRISPRi datasets with positive cap gain",
            lambda r: r.get("source_quality") == "source_verified"
            and r.get("perturbation_type") == "CRISPRi"
            and int(r.get("cap_gain") or 0) > 0,
        ),
        (
            "k562_crispri",
            "K562 CRISPRi datasets only, predeclared because K562 is the largest repeated background",
            lambda r: r.get("background") == "K562" and r.get("perturbation_type") == "CRISPRi",
        ),
    ]
    evaluated = []
    for name, desc, predicate in policies:
        subset = [r for r in rows if predicate(r)]
        evaluated.append(evaluate(name, subset, desc))
    evaluated.sort(key=lambda r: (bool(r["pass_gate"]), float(r["weighted_pp_delta"]), -float(r["max_dataset_weight"])), reverse=True)
    passing = [r for r in evaluated if r["pass_gate"]]
    status = "condition_count_tail_safe_subset_pass_gpu_protocol_next" if passing else "condition_count_tail_safe_subset_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": bool(passing),
        "policies": evaluated,
        "best_policy": evaluated[0] if evaluated else None,
        "reasons": [] if passing else ["no_predeclared_tail_safe_subset_policy_passed"],
        "next_action": (
            "prepare bounded condition-count subset GPU protocol using best passing policy"
            if passing
            else "do not launch condition-count GPU; condition-count remains diagnostic/negative until a new non-posthoc mechanism is proposed"
        ),
        "boundary": {
            "cpu_only": True,
            "reads_completed_train_only_reports": True,
            "reads_canonical_multi": False,
            "reads_heldout_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Condition-Count Tail-Safe Subset Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only synthesis of completed cap120-vs-cap30 train-only reports.",
        "- Tests predeclared subset policies; it does not posthoc select arbitrary datasets.",
        "- Does not train, infer, use GPU, read canonical multi, or read held-out Track C query.",
        "",
        "## Policy Table",
        "",
        "| policy | pass | datasets | conditions | pp | MMD | min ds pp | CI lower | LOO min | max weight | reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in evaluated:
        ci0 = row["bootstrap"]["ci"][0]
        loo = row["leave_one_dataset_min_pp_delta"]
        lines.append(
            f"| `{row['policy']}` | `{row['pass_gate']}` | {row['dataset_count']} | {row['n_conditions']} | "
            f"{row['weighted_pp_delta']:+.6f} | {row['weighted_mmd_delta']:+.6f} | {row['dataset_min_pp_delta']:+.6f} | "
            f"{ci0 if ci0 is not None else 'NA'} | {loo if loo is not None else 'NA'} | {row['max_dataset_weight']:.3f} | `{row['reasons']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- GPU authorized: `{payload['gpu_authorized']}`",
            f"- best policy: `{payload['best_policy']['policy'] if payload['best_policy'] else 'NA'}`",
            f"- reasons: `{payload['reasons']}`",
            f"- next action: `{payload['next_action']}`",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorized": payload["gpu_authorized"], "out_md": str(OUT_MD)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
