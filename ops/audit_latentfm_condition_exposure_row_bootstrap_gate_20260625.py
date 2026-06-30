#!/usr/bin/env python3
"""CPU-only row bootstrap gate for LatentFM condition/exposure scaling.

Reads completed train-only/internal posthoc artifacts from scaling count smokes.
Does not read checkpoints, canonical multi, Track C query, train, infer, or use
GPU.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_xverse_scaling_count_smokes_20260624"
REPORTS = ROOT / "reports"
OUT_CSV = REPORTS / "latentfm_condition_exposure_row_bootstrap_rows_20260625.csv"
OUT_JSON = REPORTS / "latentfm_condition_exposure_row_bootstrap_gate_20260625.json"
OUT_MD = REPORTS / "LATENTFM_CONDITION_EXPOSURE_ROW_BOOTSTRAP_GATE_20260625.md"

SEED = 42
N_BOOT = 5000
CROSS_GROUP = "internal_val_cross_background_seen_gene_proxy"
FAMILY_GROUP = "internal_val_family_gene_proxy"
METRICS = ("pearson_pert", "test_mmd_clamped")

ARMS = {
    "cap30": RUN_ROOT / "xverse_scaling_cap30_all_3k_seed42",
    "cap120": RUN_ROOT / "xverse_scaling_cap120_all_3k_seed42",
    "full": RUN_ROOT / "xverse_scaling_full_trainonly_3k_seed42",
    "type_balanced": RUN_ROOT / "xverse_scaling_type_balanced_cap120_3k_seed42",
    "general_exposure": RUN_ROOT / "xverse_scaling_general_exposure_cap_v2_3k_seed42",
}

COMPARISONS = [
    {
        "name": "cap120_minus_cap30",
        "candidate_arm": "cap120",
        "baseline_arm": "cap30",
        "role": "moderate_exposure_signal",
        "expected": "positive",
    },
    {
        "name": "full_minus_cap120",
        "candidate_arm": "full",
        "baseline_arm": "cap120",
        "role": "nonmonotonic_full_exposure_check",
        "expected": "nonpositive_or_not_promotable",
    },
    {
        "name": "type_balanced_minus_cap120",
        "candidate_arm": "type_balanced",
        "baseline_arm": "cap120",
        "role": "type_balance_control",
        "expected": "not_promotable",
    },
    {
        "name": "general_exposure_minus_cap120",
        "candidate_arm": "general_exposure",
        "baseline_arm": "cap120",
        "role": "general_exposure_control",
        "expected": "not_promotable_if_mmd_harm",
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def posthoc_path(run: Path, kind: str) -> Path:
    return run / "posthoc_eval_internal" / f"split_group_eval_{kind}_internal_ode20.json"


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("dataset") or ""), str(row.get("condition") or "")


def load_arm_group_delta(run: Path, group: str) -> dict[tuple[str, str], dict[str, Any]]:
    cand_path = posthoc_path(run, "candidate")
    anchor_path = posthoc_path(run, "anchor")
    cand = load_json(cand_path)
    anchor = load_json(anchor_path)
    cand_rows = {
        row_key(row): row
        for row in ((cand.get("groups") or {}).get(group) or {}).get("condition_metrics") or []
        if all(row_key(row))
    }
    anchor_rows = {
        row_key(row): row
        for row in ((anchor.get("groups") or {}).get(group) or {}).get("condition_metrics") or []
        if all(row_key(row))
    }
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key in sorted(set(cand_rows) & set(anchor_rows)):
        c = cand_rows[key]
        a = anchor_rows[key]
        values: dict[str, Any] = {
            "dataset": key[0],
            "condition": key[1],
            "n_src_eval": c.get("n_src_eval"),
            "n_gt_eval": c.get("n_gt_eval"),
        }
        ok = True
        for metric in METRICS:
            if c.get(metric) is None or a.get(metric) is None:
                ok = False
                break
            values[f"{metric}_candidate"] = float(c[metric])
            values[f"{metric}_anchor"] = float(a[metric])
            values[f"{metric}_delta"] = float(c[metric]) - float(a[metric])
        if ok:
            out[key] = values
    return out


def mean_or_nan(vals: list[float]) -> float:
    return mean(vals) if vals else math.nan


def quantile(vals: list[float], q: float) -> float:
    if not vals:
        return math.nan
    xs = sorted(vals)
    idx = (len(xs) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return xs[int(idx)]
    return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)


def bootstrap_mean(vals: list[float], seed: int, n_boot: int = N_BOOT) -> dict[str, float]:
    rng = random.Random(seed)
    if not vals:
        return {"mean": math.nan, "ci_low": math.nan, "ci_high": math.nan, "p_le_zero": math.nan}
    n = len(vals)
    boots = []
    for _ in range(n_boot):
        boots.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    return {
        "mean": mean(vals),
        "ci_low": quantile(boots, 0.025),
        "ci_high": quantile(boots, 0.975),
        "p_le_zero": sum(x <= 0 for x in boots) / len(boots),
    }


def signflip_control(vals: list[float], actual: float, seed: int, n_boot: int = N_BOOT) -> dict[str, float]:
    rng = random.Random(seed)
    if not vals:
        return {"p_ge_actual": math.nan, "p95": math.nan, "mean": math.nan}
    null = []
    n = len(vals)
    for _ in range(n_boot):
        null.append(sum(v * (1 if rng.random() < 0.5 else -1) for v in vals) / n)
    return {
        "mean": mean(null),
        "p95": quantile(null, 0.95),
        "p_ge_actual": sum(x >= actual for x in null) / len(null),
    }


def summarize_pair(rows: list[dict[str, Any]], prefix: str, seed: int) -> dict[str, Any]:
    pp = [float(r[f"{prefix}_pp_diff"]) for r in rows]
    mmd = [float(r[f"{prefix}_mmd_diff"]) for r in rows]
    pp_boot = bootstrap_mean(pp, seed)
    mmd_boot = bootstrap_mean(mmd, seed + 17)
    pp_control = signflip_control(pp, pp_boot["mean"], seed + 101)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    dataset_rows = []
    for dataset, ds_rows in sorted(by_dataset.items()):
        ds_pp = [float(r[f"{prefix}_pp_diff"]) for r in ds_rows]
        ds_mmd = [float(r[f"{prefix}_mmd_diff"]) for r in ds_rows]
        dataset_rows.append(
            {
                "dataset": dataset,
                "n": len(ds_rows),
                "pp_mean": mean_or_nan(ds_pp),
                "mmd_mean": mean_or_nan(ds_mmd),
                "pp_hard_harm_frac": sum(v < -0.02 for v in ds_pp) / len(ds_pp) if ds_pp else math.nan,
            }
        )
    lodo = []
    for dataset in sorted(by_dataset):
        kept = [r for r in rows if r["dataset"] != dataset]
        kept_pp = [float(r[f"{prefix}_pp_diff"]) for r in kept]
        lodo.append({"left_out": dataset, "pp_mean": mean_or_nan(kept_pp), "n": len(kept)})
    return {
        "n_rows": len(rows),
        "n_datasets": len(by_dataset),
        "pp": pp_boot,
        "mmd": mmd_boot,
        "signflip_control": pp_control,
        "dataset_min_pp": min((r["pp_mean"] for r in dataset_rows), default=math.nan),
        "dataset_negative_tails_lt_minus_0p02": sum(r["pp_mean"] < -0.02 for r in dataset_rows),
        "row_hard_harm_frac_pp_lt_minus_0p02": sum(v < -0.02 for v in pp) / len(pp) if pp else math.nan,
        "mmd_mean": mean_or_nan(mmd),
        "mmd_max_dataset": max((r["mmd_mean"] for r in dataset_rows), default=math.nan),
        "lodo_min_pp": min((r["pp_mean"] for r in lodo), default=math.nan),
        "dataset_rows": dataset_rows,
        "lodo_rows": lodo,
    }


def gate_comparison(name: str, cross: dict[str, Any], family: dict[str, Any], expected: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if cross["n_rows"] < 100 or family["n_rows"] < 100:
        reasons.append("not_enough_condition_rows")
    if expected == "positive":
        if cross["pp"]["ci_low"] <= 0:
            reasons.append("cross_pp_ci_low_not_positive")
        if family["pp"]["ci_low"] <= 0:
            reasons.append("family_pp_ci_low_not_positive")
        if cross["signflip_control"]["p_ge_actual"] > 0.05:
            reasons.append("cross_signflip_control_not_separated")
        if family["signflip_control"]["p_ge_actual"] > 0.05:
            reasons.append("family_signflip_control_not_separated")
        if cross["dataset_min_pp"] < -0.02:
            reasons.append("cross_dataset_tail_below_minus_0p02")
        if family["dataset_min_pp"] < -0.02:
            reasons.append("family_dataset_tail_below_minus_0p02")
        if cross["row_hard_harm_frac_pp_lt_minus_0p02"] > 0.35:
            reasons.append("cross_row_hard_harm_frac_gt_0p35")
        if family["row_hard_harm_frac_pp_lt_minus_0p02"] > 0.35:
            reasons.append("family_row_hard_harm_frac_gt_0p35")
        if cross["mmd_mean"] > 0.001 or family["mmd_mean"] > 0.001:
            reasons.append("mmd_mean_above_0p001")
    else:
        reasons.append("control_or_nonpromotional_comparison")
    return len(reasons) == 0, reasons


def main() -> None:
    input_paths = []
    for run in ARMS.values():
        input_paths.extend([posthoc_path(run, "candidate"), posthoc_path(run, "anchor")])
    missing = [str(p) for p in input_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing posthoc inputs: " + ", ".join(missing))

    arm_rows: dict[str, dict[str, dict[tuple[str, str], dict[str, Any]]]] = {}
    for arm, run in ARMS.items():
        arm_rows[arm] = {
            "cross": load_arm_group_delta(run, CROSS_GROUP),
            "family": load_arm_group_delta(run, FAMILY_GROUP),
        }

    all_output_rows: list[dict[str, Any]] = []
    summaries = []
    for idx, comp in enumerate(COMPARISONS):
        cand = comp["candidate_arm"]
        base = comp["baseline_arm"]
        comp_rows_by_group: dict[str, list[dict[str, Any]]] = {}
        for group_name in ("cross", "family"):
            keys = sorted(set(arm_rows[cand][group_name]) & set(arm_rows[base][group_name]))
            rows = []
            for key in keys:
                c = arm_rows[cand][group_name][key]
                b = arm_rows[base][group_name][key]
                row = {
                    "comparison": comp["name"],
                    "role": comp["role"],
                    "group": group_name,
                    "dataset": key[0],
                    "condition": key[1],
                    f"{group_name}_pp_diff": c["pearson_pert_delta"] - b["pearson_pert_delta"],
                    f"{group_name}_mmd_diff": c["test_mmd_clamped_delta"] - b["test_mmd_clamped_delta"],
                    "candidate_arm": cand,
                    "baseline_arm": base,
                    "candidate_pp_delta": c["pearson_pert_delta"],
                    "baseline_pp_delta": b["pearson_pert_delta"],
                    "candidate_mmd_delta": c["test_mmd_clamped_delta"],
                    "baseline_mmd_delta": b["test_mmd_clamped_delta"],
                }
                rows.append(row)
                all_output_rows.append(row)
            comp_rows_by_group[group_name] = rows

        cross_summary = summarize_pair(comp_rows_by_group["cross"], "cross", SEED + idx * 1000)
        family_summary = summarize_pair(comp_rows_by_group["family"], "family", SEED + idx * 1000 + 200)
        passed, reasons = gate_comparison(comp["name"], cross_summary, family_summary, comp["expected"])
        summaries.append(
            {
                "comparison": comp["name"],
                "role": comp["role"],
                "candidate_arm": cand,
                "baseline_arm": base,
                "expected": comp["expected"],
                "pass": passed,
                "reasons": reasons,
                "cross": cross_summary,
                "family": family_summary,
            }
        )

    fieldnames = [
        "comparison",
        "role",
        "group",
        "dataset",
        "condition",
        "candidate_arm",
        "baseline_arm",
        "candidate_pp_delta",
        "baseline_pp_delta",
        "candidate_mmd_delta",
        "baseline_mmd_delta",
        "cross_pp_diff",
        "cross_mmd_diff",
        "family_pp_diff",
        "family_mmd_diff",
    ]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_output_rows:
            writer.writerow(row)

    primary = next(s for s in summaries if s["comparison"] == "cap120_minus_cap30")
    full = next(s for s in summaries if s["comparison"] == "full_minus_cap120")
    gpu_authorized = bool(primary["pass"])
    if full["cross"]["pp"]["mean"] > 0 and full["family"]["pp"]["mean"] > 0:
        gpu_authorized = False
    status = (
        "condition_exposure_row_bootstrap_pass_gpu_candidate"
        if gpu_authorized
        else "condition_exposure_row_bootstrap_fail_no_gpu"
    )
    decision_reasons = []
    if not primary["pass"]:
        decision_reasons.extend(primary["reasons"])
    if full["cross"]["pp"]["mean"] > 0 and full["family"]["pp"]["mean"] > 0:
        decision_reasons.append("full_exposure_row_means_positive_against_nonmonotonic_gate")
    if not decision_reasons:
        decision_reasons.append("primary_row_bootstrap_passed")

    payload = {
        "status": status,
        "gpu_authorized": gpu_authorized,
        "boundary": {
            "cpu_only": True,
            "reads_completed_trainonly_internal_posthoc": True,
            "reads_model_checkpoints": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "thresholds": {
            "primary_comparison": "cap120_minus_cap30",
            "pp_ci_low_must_be_positive": True,
            "signflip_p_ge_actual_max": 0.05,
            "dataset_min_pp_floor": -0.02,
            "row_hard_harm_frac_max": 0.35,
            "mmd_mean_max": 0.001,
            "full_exposure_must_not_dominate_moderate": True,
        },
        "inputs": {str(p): sha256(p) for p in input_paths},
        "row_csv": str(OUT_CSV),
        "comparisons": summaries,
        "decision": {
            "status": status,
            "gpu_authorized": gpu_authorized,
            "reasons": sorted(set(decision_reasons)),
            "next_action": (
                "if accepted, prepare a bounded moderate-exposure GPU smoke with row-bootstrap gate provenance"
                if gpu_authorized
                else "do not launch condition/exposure GPU; keep scaling exposure as mechanism/failure-map unless a new artifact/control changes the row gate"
            ),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# LatentFM Condition/Exposure Row Bootstrap Gate",
        "",
        f"Status: `{status}`",
        f"GPU authorized: `{gpu_authorized}`",
        "",
        "## Boundary",
        "",
        "- CPU-only paired condition-row bootstrap over completed train-only/internal scaling posthoc artifacts.",
        "- Does not read checkpoints, canonical multi, Track C held-out query, train, infer, or use GPU.",
        "",
        "## Primary Gate",
        "",
        "| comparison | group | n | pp mean | pp CI95 | signflip p>=actual | dataset min | hard-harm frac | mmd mean | pass/reason |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        for group_name in ("cross", "family"):
            g = summary[group_name]
            pass_text = "pass" if summary["pass"] else "; ".join(summary["reasons"])
            lines.append(
                "| `{}` | `{}` | {} | {:+.6f} | [{:+.6f}, {:+.6f}] | {:.4f} | {:+.6f} | {:.3f} | {:+.6f} | {} |".format(
                    summary["comparison"],
                    group_name,
                    g["n_rows"],
                    g["pp"]["mean"],
                    g["pp"]["ci_low"],
                    g["pp"]["ci_high"],
                    g["signflip_control"]["p_ge_actual"],
                    g["dataset_min_pp"],
                    g["row_hard_harm_frac_pp_lt_minus_0p02"],
                    g["mmd_mean"],
                    pass_text,
                )
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- reasons: `{sorted(set(decision_reasons))}`",
            f"- next action: `{payload['decision']['next_action']}`",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- row CSV: `{OUT_CSV}`",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
