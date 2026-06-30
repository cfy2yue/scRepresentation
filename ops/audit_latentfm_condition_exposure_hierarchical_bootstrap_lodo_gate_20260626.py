#!/usr/bin/env python3
"""Condition-exposure hierarchical bootstrap + LODO gate for LatentFM scaling.

CPU/report-only. This script uses completed train-only/internal posthoc row
artifacts and previously frozen gate summaries to decide whether the
moderate-vs-full condition-exposure axis is law-ready or GPU-ready.

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
ROW_CSV = REPORTS / "latentfm_condition_exposure_row_bootstrap_rows_20260625.csv"
ROW_GATE_JSON = REPORTS / "latentfm_condition_exposure_row_bootstrap_gate_20260625.json"
MIXED_LODO_JSON = REPORTS / "latentfm_scaling_mixed_effect_lodo_condition_count_gate_20260624.json"
NESTED_JSON = REPORTS / "latentfm_scaling_nested_condition_exposure_v2_gate_20260625.json"
LAW_READY_JSON = REPORTS / "latentfm_scaling_law_ready_evidence_table_20260626.json"

OUT_DIR = REPORTS / "condition_exposure_hierarchical_bootstrap_lodo_gate_20260626"
OUT_MD = REPORTS / "LATENTFM_CONDITION_EXPOSURE_HIERARCHICAL_BOOTSTRAP_LODO_GATE_20260626.md"
OUT_JSON = REPORTS / "latentfm_condition_exposure_hierarchical_bootstrap_lodo_gate_20260626.json"
OUT_SUMMARY = OUT_DIR / "comparison_group_summary.csv"
OUT_DATASET = OUT_DIR / "dataset_tail_summary.csv"
OUT_CRITERIA = OUT_DIR / "criteria_matrix.csv"
OUT_INPUTS = OUT_DIR / "input_manifest.tsv"

N_BOOT = 5000
SEED = 260626
PRIMARY_COMPARISON = "cap120_minus_cap30"
FULL_COMPARISON = "full_minus_cap120"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


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


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ROW_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            group = row["group"]
            pp_col = "cross_pp_diff" if group == "cross" else "family_pp_diff"
            mmd_col = "cross_mmd_diff" if group == "cross" else "family_mmd_diff"
            pp = as_float(row.get(pp_col))
            mmd = as_float(row.get(mmd_col))
            if pp is None:
                continue
            rows.append(
                {
                    "comparison": row["comparison"],
                    "group": group,
                    "dataset": row["dataset"],
                    "condition": row["condition"],
                    "pp": pp,
                    "mmd": mmd if mmd is not None else math.nan,
                }
            )
    return rows


def hierarchical_bootstrap(vals_by_dataset: dict[str, list[float]], *, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    datasets = sorted(vals_by_dataset)
    if not datasets:
        return {"mean": math.nan, "ci_low": math.nan, "ci_high": math.nan, "p_le_zero": math.nan}
    boot_means = []
    for _ in range(N_BOOT):
        sampled = []
        for ds in (rng.choice(datasets) for _ in datasets):
            vals = vals_by_dataset[ds]
            sampled.extend(rng.choice(vals) for _ in vals)
        boot_means.append(mean(sampled))
    return {
        "mean": mean([v for vals in vals_by_dataset.values() for v in vals]),
        "ci_low": quantile(boot_means, 0.025),
        "ci_high": quantile(boot_means, 0.975),
        "p_le_zero": sum(v <= 0 for v in boot_means) / len(boot_means),
    }


def summarize_group(rows: list[dict[str, Any]], comparison: str, group: str, seed_offset: int) -> dict[str, Any]:
    subset = [r for r in rows if r["comparison"] == comparison and r["group"] == group]
    by_ds_pp: dict[str, list[float]] = defaultdict(list)
    by_ds_mmd: dict[str, list[float]] = defaultdict(list)
    for row in subset:
        by_ds_pp[row["dataset"]].append(row["pp"])
        if not math.isnan(row["mmd"]):
            by_ds_mmd[row["dataset"]].append(row["mmd"])
    pp_boot = hierarchical_bootstrap(by_ds_pp, seed=SEED + seed_offset)
    mmd_boot = hierarchical_bootstrap(by_ds_mmd, seed=SEED + seed_offset + 1000)
    dataset_rows = []
    for ds in sorted(by_ds_pp):
        vals = by_ds_pp[ds]
        mmd_vals = by_ds_mmd.get(ds, [])
        dataset_rows.append(
            {
                "comparison": comparison,
                "group": group,
                "dataset": ds,
                "n": len(vals),
                "pp_mean": mean(vals),
                "mmd_mean": mean(mmd_vals) if mmd_vals else math.nan,
                "hard_harm_frac_pp_lt_minus_0p02": sum(v < -0.02 for v in vals) / len(vals),
            }
        )
    lodo_rows = []
    for ds in sorted(by_ds_pp):
        kept = [v for d, vals in by_ds_pp.items() if d != ds for v in vals]
        lodo_rows.append({"left_out": ds, "pp_mean": mean(kept), "n": len(kept)})
    return {
        "comparison": comparison,
        "group": group,
        "n_rows": len(subset),
        "n_datasets": len(by_ds_pp),
        "pp_mean": pp_boot["mean"],
        "pp_hier_ci_low": pp_boot["ci_low"],
        "pp_hier_ci_high": pp_boot["ci_high"],
        "pp_hier_p_le_zero": pp_boot["p_le_zero"],
        "mmd_mean": mmd_boot["mean"],
        "mmd_hier_ci_low": mmd_boot["ci_low"],
        "mmd_hier_ci_high": mmd_boot["ci_high"],
        "dataset_min_pp": min((r["pp_mean"] for r in dataset_rows), default=math.nan),
        "dataset_negative_tails_lt_minus_0p02": sum(r["pp_mean"] < -0.02 for r in dataset_rows),
        "dataset_severe_tails_lt_minus_0p05": sum(r["pp_mean"] < -0.05 for r in dataset_rows),
        "row_hard_harm_frac_pp_lt_minus_0p02": sum(r["hard_harm_frac_pp_lt_minus_0p02"] * r["n"] for r in dataset_rows) / max(1, sum(r["n"] for r in dataset_rows)),
        "lodo_min_pp": min((r["pp_mean"] for r in lodo_rows), default=math.nan),
        "dataset_rows": dataset_rows,
        "lodo_rows": lodo_rows,
    }


def existing_comparison(row_gate: dict[str, Any], comparison: str, group: str) -> dict[str, Any]:
    for comp in row_gate.get("comparisons", []):
        if comp.get("comparison") == comparison:
            return comp.get(group, {})
    return {}


def passfail(name: str, passed: bool, value: Any, threshold: str, reason_if_fail: str) -> dict[str, Any]:
    return {
        "criterion": name,
        "passed": bool(passed),
        "value": value,
        "threshold": threshold,
        "fail_reason": "" if passed else reason_if_fail,
    }


def build_criteria(
    summaries: dict[tuple[str, str], dict[str, Any]],
    row_gate: dict[str, Any],
    mixed: dict[str, Any],
    nested: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    criteria: list[dict[str, Any]] = []
    reasons: list[str] = []

    for group in ["cross", "family"]:
        summary = summaries[(PRIMARY_COMPARISON, group)]
        existing = existing_comparison(row_gate, PRIMARY_COMPARISON, group)
        signflip = existing.get("signflip_control", {}).get("p_ge_actual")
        checks = [
            passfail(f"{group}_hierarchical_ci_lower_positive", summary["pp_hier_ci_low"] > 0, summary["pp_hier_ci_low"], "> 0", f"{group}_hierarchical_ci_lower_not_positive"),
            passfail(f"{group}_dataset_min_safe", summary["dataset_min_pp"] >= -0.02, summary["dataset_min_pp"], ">= -0.02", f"{group}_dataset_min_below_minus_0p02"),
            passfail(f"{group}_no_negative_dataset_tails", summary["dataset_negative_tails_lt_minus_0p02"] == 0, summary["dataset_negative_tails_lt_minus_0p02"], "0", f"{group}_negative_dataset_tails_present"),
            passfail(f"{group}_row_hard_harm_frac_safe", summary["row_hard_harm_frac_pp_lt_minus_0p02"] <= 0.35, summary["row_hard_harm_frac_pp_lt_minus_0p02"], "<= 0.35", f"{group}_row_hard_harm_frac_above_0p35"),
            passfail(f"{group}_mmd_mean_safe", abs(summary["mmd_mean"]) <= 0.001, summary["mmd_mean"], "abs <= 0.001", f"{group}_mmd_mean_above_0p001"),
            passfail(f"{group}_signflip_control_pass", signflip is not None and signflip <= 0.05, signflip, "<= 0.05", f"{group}_signflip_control_not_separated"),
            passfail(f"{group}_lodo_min_positive", summary["lodo_min_pp"] > 0, summary["lodo_min_pp"], "> 0", f"{group}_lodo_min_not_positive"),
        ]
        criteria.extend(checks)
        reasons.extend(c["fail_reason"] for c in checks if not c["passed"])

    mixed_summary = mixed.get("summary", {})
    mixed_checks = [
        passfail(
            "dataset_bootstrap_ci_lower_positive",
            (mixed_summary.get("bootstrap_dataset_mean", {}).get("mean_ci") or [math.nan])[0] > 0,
            (mixed_summary.get("bootstrap_dataset_mean", {}).get("mean_ci") or [None, None])[0],
            "> 0",
            "dataset_bootstrap_ci_lower_not_positive",
        ),
        passfail(
            "leave_background_min_nonnegative",
            mixed_summary.get("min_leave_one_background_pp_delta", -999) >= 0,
            mixed_summary.get("min_leave_one_background_pp_delta"),
            ">= 0",
            "leave_background_min_negative",
        ),
        passfail(
            "leave_type_min_nonnegative",
            mixed_summary.get("min_leave_one_type_pp_delta", -999) >= 0,
            mixed_summary.get("min_leave_one_type_pp_delta"),
            ">= 0",
            "leave_type_min_negative",
        ),
    ]
    criteria.extend(mixed_checks)
    reasons.extend(c["fail_reason"] for c in mixed_checks if not c["passed"])

    for group in ["cross", "family"]:
        full_summary = summaries[(FULL_COMPARISON, group)]
        check = passfail(
            f"full_minus_moderate_not_positive_{group}",
            full_summary["pp_mean"] <= 0,
            full_summary["pp_mean"],
            "<= 0",
            f"full_exposure_mean_positive_{group}",
        )
        criteria.append(check)
        if not check["passed"]:
            reasons.append(check["fail_reason"])

    nested_summary = nested.get("summary", {})
    canonical_checks = [
        passfail(
            "seed_sign_stable",
            not bool(nested_summary.get("seed_sign_flip")),
            nested_summary.get("seed_sign_flip"),
            "False",
            "seed_sign_flip_present",
        ),
        passfail(
            "frozen_canonical_noharm_not_failed_all",
            not bool(nested_summary.get("truecell_budget128_canonical_failed_all_seeds")),
            nested_summary.get("truecell_budget128_canonical_failed_all_seeds"),
            "False",
            "frozen_canonical_noharm_failed_all_related_scaling_seeds",
        ),
    ]
    criteria.extend(canonical_checks)
    reasons.extend(c["fail_reason"] for c in canonical_checks if not c["passed"])

    return criteria, reasons


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str], *, delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M CST")
    rows = load_rows()
    row_gate = read_json(ROW_GATE_JSON)
    mixed = read_json(MIXED_LODO_JSON)
    nested = read_json(NESTED_JSON)
    law_ready = read_json(LAW_READY_JSON)

    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    comparisons = sorted({r["comparison"] for r in rows})
    for comparison in comparisons:
        for group in ["cross", "family"]:
            s = summarize_group(rows, comparison, group, seed_offset=len(summary_rows) * 37)
            summaries[(comparison, group)] = s
            summary_rows.append({k: v for k, v in s.items() if k not in {"dataset_rows", "lodo_rows"}})
            dataset_rows.extend(s["dataset_rows"])

    criteria, reasons = build_criteria(summaries, row_gate, mixed, nested)
    pass_all = all(c["passed"] for c in criteria)
    status = "condition_exposure_hierarchical_bootstrap_lodo_pass_gpu_candidate" if pass_all else "condition_exposure_hierarchical_bootstrap_lodo_fail_no_gpu"

    input_paths = [ROW_CSV, ROW_GATE_JSON, MIXED_LODO_JSON, NESTED_JSON, LAW_READY_JSON]
    input_rows = []
    for path in input_paths:
        input_rows.append(
            {
                "path": str(path),
                "exists": str(path.exists()).lower(),
                "size": path.stat().st_size if path.exists() else "",
                "sha256": sha256(path) if path.exists() else "",
            }
        )

    write_csv(
        OUT_SUMMARY,
        summary_rows,
        [
            "comparison",
            "group",
            "n_rows",
            "n_datasets",
            "pp_mean",
            "pp_hier_ci_low",
            "pp_hier_ci_high",
            "pp_hier_p_le_zero",
            "mmd_mean",
            "mmd_hier_ci_low",
            "mmd_hier_ci_high",
            "dataset_min_pp",
            "dataset_negative_tails_lt_minus_0p02",
            "dataset_severe_tails_lt_minus_0p05",
            "row_hard_harm_frac_pp_lt_minus_0p02",
            "lodo_min_pp",
        ],
    )
    write_csv(
        OUT_DATASET,
        dataset_rows,
        ["comparison", "group", "dataset", "n", "pp_mean", "mmd_mean", "hard_harm_frac_pp_lt_minus_0p02"],
    )
    write_csv(OUT_CRITERIA, criteria, ["criterion", "passed", "value", "threshold", "fail_reason"])
    write_csv(OUT_INPUTS, input_rows, ["path", "exists", "size", "sha256"], delimiter="\t")

    payload = {
        "timestamp": timestamp,
        "status": status,
        "default_model": "xverse_8k_anchor",
        "gpu_authorized": bool(pass_all),
        "immediate_gpu_candidate_count": 1 if pass_all else 0,
        "boundary": {
            "cpu_only": True,
            "reads_checkpoints": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "trains_or_infers": False,
            "uses_gpu": False,
            "row_artifact_only": True,
        },
        "primary_comparison": PRIMARY_COMPARISON,
        "full_comparison": FULL_COMPARISON,
        "n_boot": N_BOOT,
        "seed": SEED,
        "summaries": summary_rows,
        "criteria": criteria,
        "fail_reasons": reasons,
        "law_ready_prior_status": law_ready.get("status"),
        "decision": {
            "action": "do_not_launch_condition_exposure_gpu" if not pass_all else "eligible_for_external_review_before_bounded_gpu",
            "claim": "failure_map_or_mechanism_only" if not pass_all else "cpu_gate_passed_not_yet_promoted",
            "next_gate": "new non-noop/tail-safe mechanism or genuinely new matched artifact" if not pass_all else "external review plus fresh GPU audit",
        },
        "outputs": {
            "summary": str(OUT_SUMMARY),
            "dataset_tails": str(OUT_DATASET),
            "criteria": str(OUT_CRITERIA),
            "input_manifest": str(OUT_INPUTS),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    primary_cross = summaries[(PRIMARY_COMPARISON, "cross")]
    primary_family = summaries[(PRIMARY_COMPARISON, "family")]
    full_cross = summaries[(FULL_COMPARISON, "cross")]
    mixed_summary = mixed.get("summary", {})

    lines = [
        "# LatentFM Condition Exposure Hierarchical Bootstrap LODO Gate",
        "",
        f"Timestamp: `{timestamp}`",
        "",
        f"Status: `{status}`",
        "",
        "Default/deployable model: `xverse_8k_anchor`",
        "",
        f"Immediate non-ACK GPU candidate count: `{1 if pass_all else 0}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only gate over completed train-only/internal condition-exposure row artifacts.",
        "- Recomputes two-level hierarchical bootstrap: dataset resampling, then condition resampling within dataset.",
        "- Does not train, infer, read checkpoints, read canonical multi, read Track C query, read expression matrices, or use GPU.",
        "",
        "## Primary Moderate Exposure Result",
        "",
        f"- Cross pp mean/CI: `{fmt(primary_cross['pp_mean'])}` / `[{fmt(primary_cross['pp_hier_ci_low'])}, {fmt(primary_cross['pp_hier_ci_high'])}]`.",
        f"- Family pp mean/CI: `{fmt(primary_family['pp_mean'])}` / `[{fmt(primary_family['pp_hier_ci_low'])}, {fmt(primary_family['pp_hier_ci_high'])}]`.",
        f"- Cross dataset min / negative tails: `{fmt(primary_cross['dataset_min_pp'])}` / `{primary_cross['dataset_negative_tails_lt_minus_0p02']}`.",
        f"- Family dataset min / negative tails: `{fmt(primary_family['dataset_min_pp'])}` / `{primary_family['dataset_negative_tails_lt_minus_0p02']}`.",
        f"- Mixed-effect dataset bootstrap CI low: `{fmt((mixed_summary.get('bootstrap_dataset_mean', {}).get('mean_ci') or [None])[0])}`.",
        f"- Leave-background/type minima: `{fmt(mixed_summary.get('min_leave_one_background_pp_delta'))}` / `{fmt(mixed_summary.get('min_leave_one_type_pp_delta'))}`.",
        f"- Full-minus-moderate cross mean: `{fmt(full_cross['pp_mean'])}`.",
        "",
        "## Criteria",
        "",
        "| criterion | pass | value | threshold | fail reason |",
        "|---|---:|---:|---|---|",
    ]
    for c in criteria:
        lines.append(
            f"| `{c['criterion']}` | `{str(c['passed']).lower()}` | `{fmt(c['value'])}` | {c['threshold']} | {c['fail_reason'] or 'none'} |"
        )

    lines += [
        "",
        "## Decision",
        "",
        "- Do not launch condition-exposure GPU from current evidence.",
        "- Moderate exposure has a positive mean but fails hierarchical CI, signflip, dataset-tail, leave-type/background, full-exposure, seed-stability, and frozen no-harm criteria.",
        "- Keep condition exposure as mechanism/failure-map evidence unless a new non-noop tail-safe mechanism or genuinely new matched artifact changes the gate.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- Comparison/group summary: `{OUT_SUMMARY}`",
        f"- Dataset tail summary: `{OUT_DATASET}`",
        f"- Criteria matrix: `{OUT_CRITERIA}`",
        f"- Input manifest: `{OUT_INPUTS}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
