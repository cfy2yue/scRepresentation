#!/usr/bin/env python3
"""Source/background/type matched scaling gate.

CPU/report-only synthesis over completed source-resolved scaling artifacts.
This is a dataset/source-row hierarchical gate, not a condition-level bootstrap:
the available source/background/type evidence is already aggregated to
dataset-level source-verified cap-gain rows.

It does not read checkpoints, canonical multi, Track C query outputs, expression
matrices, or launch training/inference/GPU work.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
SOURCE_V2 = REPORTS / "latentfm_scaling_source_resolved_estimand_v2_gate_20260625.json"
CROSSED_LODO = REPORTS / "latentfm_source_verified_crossed_background_type_lodo_gate_20260624.json"
POLICY_V2 = REPORTS / "latentfm_source_verified_background_type_v2_gate_20260625.json"
CONFOUND = REPORTS / "latentfm_scaling_matched_background_type_confound_gate_20260624.json"
LAW_READY = REPORTS / "latentfm_scaling_law_ready_evidence_table_20260626.json"

OUT_DIR = REPORTS / "source_background_type_hierarchical_matched_gate_20260626"
OUT_MD = REPORTS / "LATENTFM_SOURCE_BACKGROUND_TYPE_HIERARCHICAL_MATCHED_GATE_20260626.md"
OUT_JSON = REPORTS / "latentfm_source_background_type_hierarchical_matched_gate_20260626.json"
OUT_DATASET = OUT_DIR / "source_verified_dataset_rows.csv"
OUT_STRATA = OUT_DIR / "stratum_summary.csv"
OUT_CRITERIA = OUT_DIR / "criteria_matrix.csv"
OUT_INPUTS = OUT_DIR / "input_manifest.tsv"

N_BOOT = 5000
SEED = 260626


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fnum(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def weighted(rows: list[dict[str, Any]], key: str) -> float:
    denom = sum(max(0, int(fnum(row.get("n")))) for row in rows)
    if denom <= 0:
        return math.nan
    return sum(fnum(row.get(key)) * max(0, int(fnum(row.get("n")))) for row in rows) / denom


def quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def bootstrap(rows: list[dict[str, Any]], key: str, *, seed: int = SEED) -> dict[str, float]:
    rng = random.Random(seed)
    vals = []
    for _ in range(N_BOOT):
        sample = [rng.choice(rows) for _ in rows]
        vals.append(weighted(sample, key))
    return {
        "mean": weighted(rows, key),
        "ci_low": quantile(vals, 0.025),
        "ci_high": quantile(vals, 0.975),
        "p_le_zero": sum(v <= 0 for v in vals) / len(vals),
    }


def group_summary(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field) or "")].append(row)
    out = []
    for value, subset in sorted(groups.items()):
        out.append(
            {
                "stratum_type": field,
                "stratum": value,
                "dataset_count": len(subset),
                "datasets": ";".join(str(r.get("dataset")) for r in subset),
                "n": sum(int(fnum(r.get("n"))) for r in subset),
                "pp_delta_mean": weighted(subset, "pp_delta_mean"),
                "mmd_delta_mean": weighted(subset, "mmd_delta_mean"),
                "dataset_min_pp": min((fnum(r.get("pp_delta_mean")) for r in subset), default=math.nan),
            }
        )
    return out


def leave_one(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    values = sorted({str(row.get(field) or "") for row in rows})
    out = []
    for value in values:
        kept = [row for row in rows if str(row.get(field) or "") != value]
        out.append(
            {
                "left_out_type": field,
                "left_out": value,
                "remaining_dataset_count": len(kept),
                "pp_delta_mean": weighted(kept, "pp_delta_mean"),
                "mmd_delta_mean": weighted(kept, "mmd_delta_mean"),
            }
        )
    return out


def criterion(name: str, passed: bool, value: Any, threshold: str, fail_reason: str) -> dict[str, Any]:
    return {
        "criterion": name,
        "passed": bool(passed),
        "value": value,
        "threshold": threshold,
        "fail_reason": "" if passed else fail_reason,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str], *, delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M CST")
    source = read_json(SOURCE_V2)
    crossed = read_json(CROSSED_LODO)
    policy = read_json(POLICY_V2)
    confound = read_json(CONFOUND)
    law_ready = read_json(LAW_READY)

    rows = [dict(row) for row in source.get("primary_rows", [])]
    for row in rows:
        row["n"] = int(fnum(row.get("n")))
        row["pp_delta_mean"] = fnum(row.get("pp_delta_mean"))
        row["mmd_delta_mean"] = fnum(row.get("mmd_delta_mean"))

    pp_boot = bootstrap(rows, "pp_delta_mean")
    mmd_boot = bootstrap(rows, "mmd_delta_mean", seed=SEED + 17)
    bg_rows = group_summary(rows, "background")
    type_rows = group_summary(rows, "perturbation_type")
    leave_bg = leave_one(rows, "background")
    leave_type = leave_one(rows, "perturbation_type")

    n_total = sum(int(r["n"]) for r in rows)
    max_weight = max((int(r["n"]) / n_total for r in rows), default=math.nan)
    dataset_min = min((fnum(r["pp_delta_mean"]) for r in rows), default=math.nan)
    negative_tails = sum(fnum(r["pp_delta_mean"]) < -0.02 for r in rows)
    min_bg = min((fnum(r["pp_delta_mean"]) for r in bg_rows), default=math.nan)
    min_type = min((fnum(r["pp_delta_mean"]) for r in type_rows), default=math.nan)
    min_leave_bg = min((fnum(r["pp_delta_mean"]) for r in leave_bg), default=math.nan)
    min_leave_type = min((fnum(r["pp_delta_mean"]) for r in leave_type), default=math.nan)
    crossed_summary = crossed.get("summary", {})

    criteria = [
        criterion("dataset_count_sufficient", len(rows) >= 8, len(rows), ">= 8 source-verified cap-gain datasets", "too_few_source_verified_datasets"),
        criterion("condition_count_sufficient", n_total >= 100, n_total, ">= 100 source-verified rows", "too_few_source_verified_rows"),
        criterion("weighted_pp_positive_large_enough", pp_boot["mean"] >= 0.015, pp_boot["mean"], ">= +0.015", "source_resolved_pp_mean_lt_0p015"),
        criterion("bootstrap_ci_lower_positive", pp_boot["ci_low"] > 0, pp_boot["ci_low"], "> 0", "bootstrap_ci_lower_not_positive"),
        criterion("dataset_min_safe", dataset_min >= -0.02, dataset_min, ">= -0.02", "dataset_tail_below_minus_0p020"),
        criterion("no_negative_dataset_tails", negative_tails == 0, negative_tails, "0", "negative_dataset_tails_present"),
        criterion("mmd_mean_safe", abs(mmd_boot["mean"]) <= 0.001, mmd_boot["mean"], "abs <= 0.001", "mmd_mean_above_0p001"),
        criterion("background_stratum_min_nonnegative", min_bg >= 0, min_bg, ">= 0", "background_stratum_min_negative"),
        criterion("type_stratum_min_nonnegative", min_type >= 0, min_type, ">= 0", "type_stratum_min_negative"),
        criterion("leave_background_min_nonnegative", min_leave_bg >= 0, min_leave_bg, ">= 0", "leave_background_min_negative"),
        criterion("leave_type_min_nonnegative", min_leave_type >= 0, min_leave_type, ">= 0", "leave_type_min_negative"),
        criterion("max_dataset_weight_bounded", max_weight <= 0.35, max_weight, "<= 0.35", "max_dataset_weight_gt_0p35"),
        criterion(
            "background_type_confound_gate_not_failed",
            not str(confound.get("status", "")).endswith("fail_no_gpu"),
            confound.get("status"),
            "not fail_no_gpu",
            "background_type_confound_gate_failed",
        ),
        criterion(
            "tail_policy_v2_has_passing_policy",
            bool(policy.get("best_policy", {}).get("pass_gate")),
            policy.get("best_policy", {}).get("policy") if policy.get("best_policy") else None,
            "at least one passing policy",
            "no_predeclared_background_type_policy_passed_tail_safe_gate",
        ),
        criterion(
            "noharm_calibration_passed",
            crossed_summary.get("noharm_calibration_status") not in {None, "noharm_calibration_positive_controls_fail_no_gpu"},
            crossed_summary.get("noharm_calibration_status"),
            "positive controls pass",
            "noharm_calibration_not_passed",
        ),
    ]
    fail_reasons = [c["fail_reason"] for c in criteria if not c["passed"]]
    passed = not fail_reasons
    status = "source_background_type_hierarchical_matched_pass_external_review_no_gpu_yet" if passed else "source_background_type_hierarchical_matched_fail_no_gpu"

    dataset_rows = []
    for row in rows:
        dataset_rows.append(
            {
                "dataset": row.get("dataset"),
                "background": row.get("background"),
                "perturbation_type": row.get("perturbation_type"),
                "n": row.get("n"),
                "cap_gain": row.get("cap_gain"),
                "pp_delta_mean": row.get("pp_delta_mean"),
                "mmd_delta_mean": row.get("mmd_delta_mean"),
                "tail_flag": bool(row.get("pp_delta_mean") < -0.02),
                "source_quality": row.get("source_quality"),
            }
        )
    stratum_rows = bg_rows + type_rows + [
        {
            "stratum_type": f"leave_one_{r['left_out_type']}",
            "stratum": r["left_out"],
            "dataset_count": r["remaining_dataset_count"],
            "datasets": "",
            "n": "",
            "pp_delta_mean": r["pp_delta_mean"],
            "mmd_delta_mean": r["mmd_delta_mean"],
            "dataset_min_pp": "",
        }
        for r in (leave_bg + leave_type)
    ]

    input_paths = [SOURCE_V2, CROSSED_LODO, POLICY_V2, CONFOUND, LAW_READY]
    input_rows = [
        {
            "path": str(path),
            "exists": str(path.exists()).lower(),
            "size": path.stat().st_size if path.exists() else "",
            "sha256": sha256(path) if path.exists() else "",
        }
        for path in input_paths
    ]

    write_csv(OUT_DATASET, dataset_rows, ["dataset", "background", "perturbation_type", "n", "cap_gain", "pp_delta_mean", "mmd_delta_mean", "tail_flag", "source_quality"])
    write_csv(OUT_STRATA, stratum_rows, ["stratum_type", "stratum", "dataset_count", "datasets", "n", "pp_delta_mean", "mmd_delta_mean", "dataset_min_pp"])
    write_csv(OUT_CRITERIA, criteria, ["criterion", "passed", "value", "threshold", "fail_reason"])
    write_csv(OUT_INPUTS, input_rows, ["path", "exists", "size", "sha256"], delimiter="\t")

    payload = {
        "timestamp": timestamp,
        "status": status,
        "default_model": "xverse_8k_anchor",
        "gpu_authorized": False,
        "immediate_gpu_candidate_count": 0,
        "boundary": {
            "cpu_only": True,
            "reads_checkpoints": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "trains_or_infers": False,
            "uses_gpu": False,
            "bootstrap_unit": "dataset/source-verified cap-gain row",
        },
        "summary": {
            "primary_datasets": len(rows),
            "primary_n": n_total,
            "pp_delta_mean": pp_boot["mean"],
            "pp_bootstrap_ci95": [pp_boot["ci_low"], pp_boot["ci_high"]],
            "pp_bootstrap_p_le_zero": pp_boot["p_le_zero"],
            "mmd_delta_mean": mmd_boot["mean"],
            "mmd_bootstrap_ci95": [mmd_boot["ci_low"], mmd_boot["ci_high"]],
            "dataset_min_pp": dataset_min,
            "negative_tails_lt_minus_0p02": negative_tails,
            "background_count": len(bg_rows),
            "type_count": len(type_rows),
            "min_background_pp": min_bg,
            "min_type_pp": min_type,
            "min_leave_background_pp": min_leave_bg,
            "min_leave_type_pp": min_leave_type,
            "max_dataset_weight": max_weight,
            "background_shuffle_control": crossed_summary.get("background_shuffle_control"),
            "type_shuffle_control": crossed_summary.get("type_shuffle_control"),
        },
        "criteria": criteria,
        "fail_reasons": fail_reasons,
        "decision": {
            "action": "do_not_launch_source_background_type_gpu",
            "claim": "failure_map_only",
            "next_gate": "requires genuinely new source/background/type mechanism or condition-level artifact with safe tails, positive bootstrap, and no-harm",
        },
        "outputs": {
            "dataset_rows": str(OUT_DATASET),
            "strata": str(OUT_STRATA),
            "criteria": str(OUT_CRITERIA),
            "input_manifest": str(OUT_INPUTS),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# LatentFM Source/Background/Type Hierarchical Matched Gate",
        "",
        f"Timestamp: `{timestamp}`",
        "",
        f"Status: `{status}`",
        "",
        "Default/deployable model: `xverse_8k_anchor`",
        "",
        "Immediate non-ACK GPU candidate count: `0`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only synthesis over completed source/background/type scaling artifacts.",
        "- Bootstrap unit is a dataset/source-verified cap-gain row; no condition-level rows are available for this axis.",
        "- Does not train, infer, read checkpoints, read canonical multi, read Track C query, or use GPU.",
        "",
        "## Summary",
        "",
        f"- Source-verified cap-gain datasets / rows: `{len(rows)}` / `{n_total}`.",
        f"- Weighted pp mean / CI: `{fmt(pp_boot['mean'])}` / `[{fmt(pp_boot['ci_low'])}, {fmt(pp_boot['ci_high'])}]`.",
        f"- Weighted MMD mean / CI: `{fmt(mmd_boot['mean'])}` / `[{fmt(mmd_boot['ci_low'])}, {fmt(mmd_boot['ci_high'])}]`.",
        f"- Dataset min / negative tails: `{fmt(dataset_min)}` / `{negative_tails}`.",
        f"- Min background/type pp: `{fmt(min_bg)}` / `{fmt(min_type)}`.",
        f"- Min leave-background/type pp: `{fmt(min_leave_bg)}` / `{fmt(min_leave_type)}`.",
        f"- Max dataset weight: `{fmt(max_weight)}`.",
        "",
        "## Criteria",
        "",
        "| criterion | pass | value | threshold | fail reason |",
        "|---|---:|---:|---|---|",
    ]
    for c in criteria:
        lines.append(f"| `{c['criterion']}` | `{str(c['passed']).lower()}` | `{fmt(c['value'])}` | {c['threshold']} | {c['fail_reason'] or 'none'} |")
    lines += [
        "",
        "## Decision",
        "",
        "- Do not launch source/background/type GPU from current evidence.",
        "- The axis remains useful for failure localization, especially background/type tail analysis, but it is not a training or scaling-law authorization.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- Dataset rows: `{OUT_DATASET}`",
        f"- Strata: `{OUT_STRATA}`",
        f"- Criteria: `{OUT_CRITERIA}`",
        f"- Input manifest: `{OUT_INPUTS}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
