#!/usr/bin/env python3
"""Dataset-stratified failure audit for the scaling-v2 high/low smoke."""

from __future__ import annotations

import csv
import json
import random
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_scaling_v2_condition_information_highlow_smoke_20260628"
REPORT_DIR = ROOT / "reports/scaling_v2_condition_information_failure_strata_20260628"
PACKET_JSON = ROOT / "reports/scaling_v2_condition_information_packet_audit_20260628/latentfm_scaling_v2_condition_information_packet_audit_20260628.json"
DECISION_JSON = ROOT / "reports/scaling_v2_condition_information_highlow_smoke_20260628/latentfm_scaling_v2_condition_information_highlow_decision_20260628.json"
ARMS = {
    "high": "xverse_scaling_v2_info_high_2000step_seed42",
    "low": "xverse_scaling_v2_info_low_2000step_seed42",
}
EVAL_FILES = {
    "split": ("split_group_eval_anchor_internal_ode20.json", "split_group_eval_candidate_internal_ode20.json"),
    "family": ("condition_family_eval_anchor_internal_ode20.json", "condition_family_eval_candidate_internal_ode20.json"),
}
GROUPS = [
    ("split", "internal_val_cross_background_seen_gene_proxy"),
    ("split", "internal_val_family_gene_proxy"),
    ("split", "test"),
    ("split", "test_single"),
    ("family", "family_gene"),
    ("family", "test_single"),
]


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def group_payload(arm: str, file_kind: str, group: str, role: str) -> dict[str, Any]:
    anchor_file, cand_file = EVAL_FILES[file_kind]
    filename = anchor_file if role == "anchor" else cand_file
    path = RUN_ROOT / ARMS[arm] / "posthoc_eval_internal" / filename
    payload = load_json(path)
    return (payload.get("groups") or {}).get(group) or {}


def dataset_rows(zero_train: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_kind, group in GROUPS:
        per_arm: dict[str, dict[str, dict[str, float]]] = {}
        datasets: set[str] = set()
        for arm in ARMS:
            anchor = group_payload(arm, file_kind, group, "anchor")
            cand = group_payload(arm, file_kind, group, "candidate")
            arm_map: dict[str, dict[str, float]] = {}
            for dataset, cand_pp in (cand.get("per_ds_p_pert") or {}).items():
                if dataset not in (anchor.get("per_ds_p_pert") or {}):
                    continue
                cand_mmd = (cand.get("per_ds_mmd") or {}).get(dataset)
                anchor_mmd = (anchor.get("per_ds_mmd") or {}).get(dataset)
                if cand_mmd is None or anchor_mmd is None:
                    continue
                arm_map[dataset] = {
                    "pp_delta": float(cand_pp) - float((anchor.get("per_ds_p_pert") or {})[dataset]),
                    "mmd_delta": float(cand_mmd) - float(anchor_mmd),
                }
                datasets.add(dataset)
            per_arm[arm] = arm_map
        for dataset in sorted(datasets):
            if dataset not in per_arm["high"] or dataset not in per_arm["low"]:
                continue
            high = per_arm["high"][dataset]
            low = per_arm["low"][dataset]
            rows.append(
                {
                    "file_kind": file_kind,
                    "group": group,
                    "dataset": dataset,
                    "zero_train_eval_dataset": dataset in zero_train,
                    "high_pp_delta": high["pp_delta"],
                    "low_pp_delta": low["pp_delta"],
                    "high_minus_low_pp_delta": high["pp_delta"] - low["pp_delta"],
                    "high_mmd_delta": high["mmd_delta"],
                    "low_mmd_delta": low["mmd_delta"],
                    "high_minus_low_mmd_delta": high["mmd_delta"] - low["mmd_delta"],
                }
            )
    return rows


def bootstrap_ci(values: list[float], seed: int = 20260628, repeats: int = 2000) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    rng = random.Random(seed + len(values))
    boots: list[float] = []
    for _ in range(repeats):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boots.append(statistics.fmean(sample))
    return statistics.fmean(values), quantile(boots, 0.025), quantile(boots, 0.975)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[(row["file_kind"], row["group"], "all")].append(row)
        by_group[(row["file_kind"], row["group"], "zero_train" if row["zero_train_eval_dataset"] else "nonzero_train")].append(row)
    for (file_kind, group, stratum), items in sorted(by_group.items()):
        pp = [float(row["high_minus_low_pp_delta"]) for row in items]
        mmd = [float(row["high_minus_low_mmd_delta"]) for row in items]
        pp_mean, pp_lo, pp_hi = bootstrap_ci(pp)
        mmd_mean, mmd_lo, mmd_hi = bootstrap_ci(mmd, seed=20260629)
        out.append(
            {
                "file_kind": file_kind,
                "group": group,
                "stratum": stratum,
                "n_datasets": len(items),
                "pp_mean": pp_mean,
                "pp_ci_low": pp_lo,
                "pp_ci_high": pp_hi,
                "pp_median": median(pp),
                "pp_positive_fraction": sum(1 for x in pp if x > 0) / len(pp) if pp else None,
                "mmd_mean": mmd_mean,
                "mmd_ci_low": mmd_lo,
                "mmd_ci_high": mmd_hi,
                "mmd_median": median(mmd),
                "mmd_nonharm_fraction": sum(1 for x in mmd if x <= 0.0005) / len(mmd) if mmd else None,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def decide(summary_rows: list[dict[str, Any]]) -> tuple[str, list[str], str]:
    reasons: list[str] = []
    primary_groups = {
        ("split", "internal_val_cross_background_seen_gene_proxy", "all"),
        ("split", "internal_val_family_gene_proxy", "all"),
        ("family", "family_gene", "all"),
    }
    primary = [row for row in summary_rows if (row["file_kind"], row["group"], row["stratum"]) in primary_groups]
    if not primary:
        return "scaling_v2_condition_information_failure_strata_incomplete_no_gpu", ["missing_primary_groups"], "repair_audit"
    for row in primary:
        if (row["pp_mean"] or -1.0) < 0:
            reasons.append(f"{row['group']}_dataset_mean_pp_negative")
        if (row["pp_positive_fraction"] or 0.0) < 0.5:
            reasons.append(f"{row['group']}_positive_fraction_below_half")
    nonzero = [row for row in summary_rows if row["stratum"] == "nonzero_train" and (row["file_kind"], row["group"]) in {("split", "internal_val_cross_background_seen_gene_proxy"), ("split", "internal_val_family_gene_proxy"), ("family", "family_gene")}]
    if nonzero and all((row["pp_mean"] or -1.0) < 0 for row in nonzero):
        reasons.append("nonzero_train_strata_still_negative")
    if reasons:
        return "scaling_v2_condition_information_failure_strata_supports_axis_demote_no_gpu", reasons, "await_replay_sweep_then_close_or_residualize_axis"
    return "scaling_v2_condition_information_failure_strata_points_to_localized_failure_no_gpu", reasons, "design_residualized_axis_or_stratum_exclusion_gate"


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    packet = load_json(PACKET_JSON)
    decision = load_json(DECISION_JSON)
    zero_train = set(packet.get("zero_train_eval_datasets") or [])
    rows = dataset_rows(zero_train)
    summary_rows = summarize(rows)
    status, reasons, next_action = decide(summary_rows)

    dataset_csv = REPORT_DIR / "scaling_v2_condition_information_failure_dataset_rows.csv"
    summary_csv = REPORT_DIR / "scaling_v2_condition_information_failure_summary_rows.csv"
    write_csv(dataset_csv, rows)
    write_csv(summary_csv, summary_rows)

    payload = {
        "timestamp": now_cst(),
        "status": status,
        "gpu_authorized": False,
        "reasons": reasons,
        "next_action": next_action,
        "decision_json": str(DECISION_JSON),
        "baseline_decision": decision.get("decision"),
        "zero_train_eval_datasets": sorted(zero_train),
        "dataset_rows": str(dataset_csv),
        "summary_rows": str(summary_csv),
    }
    out_json = REPORT_DIR / "latentfm_scaling_v2_condition_information_failure_strata_20260628.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    top_bad = sorted(
        [row for row in rows if row["group"] in {"internal_val_cross_background_seen_gene_proxy", "internal_val_family_gene_proxy", "family_gene"}],
        key=lambda r: float(r["high_minus_low_pp_delta"]),
    )[:12]
    report = REPORT_DIR / "LATENTFM_SCALING_V2_CONDITION_INFORMATION_FAILURE_STRATA_20260628.md"
    lines = [
        "# LatentFM Scaling V2 Condition-Information Failure Strata",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only failure analysis of the completed high/low smoke.",
        "- Uses existing internal/test proxy posthoc outputs only.",
        "- Does not train, infer, select checkpoints, use canonical multi, or use Track C query.",
        "",
        "## Primary Summary",
        "",
        "| group | stratum | n | high-low pp mean | pp CI | pp positive frac | high-low MMD mean | MMD CI | MMD nonharm frac |",
        "|---|---|---:|---:|---|---:|---:|---|---:|",
    ]
    for row in summary_rows:
        if row["group"] not in {"internal_val_cross_background_seen_gene_proxy", "internal_val_family_gene_proxy", "family_gene"}:
            continue
        lines.append(
            f"| `{row['group']}` | `{row['stratum']}` | {row['n_datasets']} | {fmt(row['pp_mean'])} | "
            f"[{fmt(row['pp_ci_low'])}, {fmt(row['pp_ci_high'])}] | {fmt(row['pp_positive_fraction'])} | "
            f"{fmt(row['mmd_mean'])} | [{fmt(row['mmd_ci_low'])}, {fmt(row['mmd_ci_high'])}] | {fmt(row['mmd_nonharm_fraction'])} |"
        )
    lines.extend(["", "## Worst High-Minus-Low Dataset Rows", "", "| group | dataset | zero-train | high-low pp | high-low MMD |", "|---|---|---:|---:|---:|"])
    for row in top_bad:
        lines.append(
            f"| `{row['group']}` | `{row['dataset']}` | {row['zero_train_eval_dataset']} | "
            f"{fmt(row['high_minus_low_pp_delta'])} | {fmt(row['high_minus_low_mmd_delta'])} |"
        )
    lines.extend(["", "## Decision", "", f"- next action: `{next_action}`"])
    lines.extend(f"- reason: `{reason}`" for reason in reasons)
    lines.extend(["", "## Outputs", "", f"- dataset rows: `{dataset_csv}`", f"- summary rows: `{summary_csv}`", f"- JSON: `{out_json}`"])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out": str(report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
