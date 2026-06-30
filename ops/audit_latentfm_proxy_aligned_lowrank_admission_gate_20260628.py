#!/usr/bin/env python3
"""CPU-only proxy-admission replay for low-rank residual failures.

This gate does not train, run inference, use GPU, read canonical multi, or
read Track C query.  It replays existing train-time update logs and asks
whether any training-visible, non-count rule could have vetoed all failed
low-rank trajectories while preserving the condition-delta viability reference.

The gate is deliberately strict.  A rule that is matched by count/attempt
controls or shuffled features is not evidence for a new GPU route.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
OUT_JSON = ROOT / "reports/latentfm_proxy_aligned_lowrank_admission_gate_20260628.json"
OUT_MD = ROOT / "reports/LATENTFM_PROXY_ALIGNED_LOWRANK_ADMISSION_GATE_20260628.md"
OUT_DIR = ROOT / "reports/proxy_aligned_lowrank_admission_gate_20260628"
OUT_ROWS = OUT_DIR / "step_rows.csv"
OUT_RULES = OUT_DIR / "rule_rows.csv"
SAFE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"

RUNS = {
    "lowrank_5accepted_internal_fail": {
        "path": ROOT / "runs/latentfm_lowrank_residual_adapter_smoke_20260627/xverse_lowrank_residual_adapter_seed42_5accepted_20260628_0000/train_metrics.csv",
        "role": "lowrank_fail",
    },
    "lowrank_10accepted_internal_fail": {
        "path": ROOT / "runs/latentfm_lowrank_residual_adapter_smoke_20260627/xverse_lowrank_residual_adapter_seed42_10accepted_20260628_0002/train_metrics.csv",
        "role": "lowrank_fail",
    },
    "lowrank_20accepted_internal_fail": {
        "path": ROOT / "runs/latentfm_lowrank_residual_adapter_smoke_20260627/xverse_lowrank_residual_adapter_seed42_20accepted_20260627_2356/train_metrics.csv",
        "role": "lowrank_fail",
    },
    "condition_delta_40accepted_internal_pass_canonical_fail": {
        "path": ROOT / "runs/latentfm_lookahead_trust_region_adapter_smoke_20260627/xverse_lookahead_trust_region_adapter_seed42_40accepted_20260627_2300/train_metrics.csv",
        "role": "viability_reference",
    },
}

NUMERIC_SOURCE = (
    "attempt",
    "accepted_step",
    "task_grad_norm",
    "anchor_grad0_norm",
    "base_task_loss",
    "base_noharm_anchor_loss",
    "best_step",
    "proj_task_delta",
    "proj_anchor_delta",
    "proj_footprint_mean_l2",
    "proj_material_row_frac",
    "task_retention_vs_unprojected",
    "projection_reduced_anchor_delta_frac",
)

COUNTLIKE = {
    "attempt",
    "accepted_step",
    "log1p_attempt",
    "log1p_accepted_step",
}

SEED = 20260628


@dataclass(frozen=True)
class Rule:
    name: str
    feature: str
    op: str
    threshold: float
    control: str


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_log1p_abs(value: float) -> float:
    return math.log1p(abs(value)) if math.isfinite(value) else float("nan")


def read_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run, meta in RUNS.items():
        path = Path(meta["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                if str(raw.get("accepted", "")).lower() != "true":
                    continue
                row: dict[str, Any] = dict(raw)
                row["run"] = run
                row["role"] = meta["role"]
                for key in NUMERIC_SOURCE:
                    row[key] = to_float(row.get(key))
                enrich_features(row)
                rows.append(row)
    return rows


def enrich_features(row: dict[str, Any]) -> None:
    for key in NUMERIC_SOURCE:
        row[f"feat_{key}"] = row[key]
        row[f"feat_log1p_abs_{key}"] = safe_log1p_abs(row[key])
    row["feat_abs_proj_task_delta"] = abs(row["proj_task_delta"])
    row["feat_abs_proj_anchor_delta"] = abs(row["proj_anchor_delta"])
    row["feat_log1p_abs_proj_task_delta"] = safe_log1p_abs(row["proj_task_delta"])
    row["feat_log1p_abs_proj_anchor_delta"] = safe_log1p_abs(row["proj_anchor_delta"])
    fp = max(abs(row["proj_footprint_mean_l2"]), 1e-12)
    row["feat_abs_task_delta_per_footprint"] = abs(row["proj_task_delta"]) / fp
    row["feat_abs_anchor_delta_per_footprint"] = abs(row["proj_anchor_delta"]) / fp
    row["feat_selected_update_norm_proxy"] = abs(row["best_step"]) * abs(row["task_grad_norm"])
    row["feat_log1p_selected_update_norm_proxy"] = safe_log1p_abs(row["feat_selected_update_norm_proxy"])
    row["feat_footprint_per_task_grad"] = abs(row["proj_footprint_mean_l2"]) / max(abs(row["task_grad_norm"]), 1e-12)
    row["feat_task_delta_per_update_proxy"] = abs(row["proj_task_delta"]) / max(row["feat_selected_update_norm_proxy"], 1e-12)


def numeric_feature_names(rows: list[dict[str, Any]], *, mode: str) -> list[str]:
    names = sorted(k.removeprefix("feat_") for row in rows for k in row if k.startswith("feat_"))
    out: list[str] = []
    for name in names:
        if mode == "count_only" and name not in COUNTLIKE:
            continue
        if mode == "noncount" and name in COUNTLIKE:
            continue
        vals = np.asarray([to_float(row.get(f"feat_{name}")) for row in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size >= 8 and float(vals.max() - vals.min()) > 1e-12:
            out.append(name)
    return out


def candidate_rules(rows: list[dict[str, Any]], *, mode: str) -> list[Rule]:
    rules: list[Rule] = []
    for feat in numeric_feature_names(rows, mode=mode):
        vals = np.asarray([to_float(row.get(f"feat_{feat}")) for row in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        for q in (0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9):
            thr = float(np.quantile(vals, q))
            rules.append(Rule(f"{feat}_le_q{q:.2f}", feat, "<=", thr, mode))
            rules.append(Rule(f"{feat}_ge_q{q:.2f}", feat, ">=", thr, mode))
    return rules


def admitted(row: dict[str, Any], rule: Rule) -> bool:
    value = to_float(row.get(f"feat_{rule.feature}"))
    if not math.isfinite(value):
        return False
    return value <= rule.threshold if rule.op == "<=" else value >= rule.threshold


def summarize_rule(rows: list[dict[str, Any]], rule: Rule) -> dict[str, Any]:
    run_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        run_rows.setdefault(str(row["run"]), []).append(row)
    run_summaries: list[dict[str, Any]] = []
    for run, rrows in sorted(run_rows.items()):
        rrows = sorted(rrows, key=lambda r: int(r["attempt"]))
        keep = [row for row in rrows if admitted(row, rule)]
        veto = [row for row in rrows if not admitted(row, rule)]
        first_five = [row for row in rrows if int(row["attempt"]) < 5]
        keep_first_five = [row for row in first_five if admitted(row, rule)]
        task_datasets = {str(row["task_dataset"]) for row in keep}
        run_summaries.append(
            {
                "run": run,
                "role": rrows[0]["role"],
                "n_total": len(rrows),
                "n_admitted": len(keep),
                "accepted_rate": len(keep) / max(len(rrows), 1),
                "n_first5": len(first_five),
                "n_first5_admitted": len(keep_first_five),
                "first_veto_attempt": min((int(row["attempt"]) for row in veto), default=None),
                "task_dataset_coverage": len(task_datasets),
                "mean_footprint": float(np.mean([to_float(row["proj_footprint_mean_l2"], 0.0) for row in keep])) if keep else 0.0,
                "max_abs_anchor_delta": float(max((abs(to_float(row["proj_anchor_delta"], 0.0)) for row in keep), default=0.0)),
            }
        )
    summary = {
        "control": rule.control,
        "rule": rule.name,
        "feature": rule.feature,
        "op": rule.op,
        "threshold": rule.threshold,
        "runs": run_summaries,
    }
    summary["decision"] = decision(summary)
    return summary


def decision(summary: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    lowrank = [row for row in summary["runs"] if row["role"] == "lowrank_fail"]
    viability = next((row for row in summary["runs"] if row["role"] == "viability_reference"), None)
    if not lowrank or viability is None:
        reasons.append("missing_required_runs")
    for row in lowrank:
        if row["first_veto_attempt"] is None:
            reasons.append(f"{row['run']}:no_veto")
        elif int(row["first_veto_attempt"]) > 4:
            reasons.append(f"{row['run']}:first_veto_after_negative_prefix")
        if int(row["n_first5_admitted"]) > 2:
            reasons.append(f"{row['run']}:too_many_first5_steps_admitted")
        if float(row["accepted_rate"]) > 0.5:
            reasons.append(f"{row['run']}:overall_admission_rate_above_0p50")
    if viability:
        if float(viability["accepted_rate"]) < 0.5:
            reasons.append("viability_accept_rate_below_0p50")
        if int(viability["task_dataset_coverage"]) < 4:
            reasons.append("viability_task_dataset_coverage_below_4")
        if float(viability["mean_footprint"]) < 5e-7:
            reasons.append("viability_mean_footprint_below_5e-7")
        if float(viability["max_abs_anchor_delta"]) > 1e-10:
            reasons.append("viability_anchor_delta_above_1e-10")
    if summary["control"] == "main" and summary["feature"] in COUNTLIKE:
        reasons.append("main_rule_is_countlike")
    return {"passes": not reasons, "reasons": reasons}


def score(summary: dict[str, Any]) -> tuple[int, int, float, int, float]:
    lowrank = [row for row in summary["runs"] if row["role"] == "lowrank_fail"]
    viability = next((row for row in summary["runs"] if row["role"] == "viability_reference"), {})
    blocked = sum(1 for row in lowrank if row["first_veto_attempt"] is not None and int(row["first_veto_attempt"]) <= 4)
    first5_blocked = sum(max(0, int(row["n_first5"]) - int(row["n_first5_admitted"])) for row in lowrank)
    return (
        1 if summary["decision"]["passes"] else 0,
        blocked,
        first5_blocked,
        float(viability.get("accepted_rate", 0.0)),
        int(viability.get("task_dataset_coverage", 0)),
    )


def transform_rows(rows: list[dict[str, Any]], control: str) -> list[dict[str, Any]]:
    copied = [dict(row) for row in rows]
    if control == "main":
        return copied
    feature_keys = sorted(k for row in copied for k in row if k.startswith("feat_"))
    rng = random.Random(SEED)
    if control == "feature_shuffle":
        for key in feature_keys:
            vals = [row.get(key) for row in copied]
            rng.shuffle(vals)
            for row, value in zip(copied, vals):
                row[key] = value
    elif control == "role_shuffle":
        roles = [row["role"] for row in copied]
        rng.shuffle(roles)
        for row, role in zip(copied, roles):
            row["role"] = role
    else:
        raise ValueError(control)
    return copied


def select_best(rows: list[dict[str, Any]], *, mode: str, control: str) -> dict[str, Any]:
    work = transform_rows(rows, control)
    rules = [Rule(r.name, r.feature, r.op, r.threshold, control if control != "main" else mode) for r in candidate_rules(work, mode=mode)]
    summaries = [summarize_rule(work, rule) for rule in rules]
    if not summaries:
        return {"control": control, "mode": mode, "missing": True, "decision": {"passes": False, "reasons": ["no_rules"]}}
    return max(summaries, key=score)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def flatten_rule(summary: dict[str, Any]) -> dict[str, Any]:
    out = {
        "control": summary.get("control"),
        "rule": summary.get("rule"),
        "feature": summary.get("feature"),
        "op": summary.get("op"),
        "threshold": summary.get("threshold"),
        "passes": (summary.get("decision") or {}).get("passes"),
        "reasons": ";".join((summary.get("decision") or {}).get("reasons") or []),
    }
    for row in summary.get("runs") or []:
        run = row["run"]
        out[f"{run}:accepted_rate"] = row["accepted_rate"]
        out[f"{run}:first_veto_attempt"] = row["first_veto_attempt"]
        out[f"{run}:n_first5_admitted"] = row["n_first5_admitted"]
        out[f"{run}:task_dataset_coverage"] = row["task_dataset_coverage"]
    return out


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Proxy-Aligned Lowrank Admission Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only replay over existing `train_metrics.csv` files.",
        f"- Safe train-only split boundary: `{SAFE_SPLIT}`.",
        "- No training, inference, GPU, canonical multi, canonical selection, or Track C query.",
        "",
        "## Decision",
        "",
        f"- reasons: `{payload['reasons']}`",
        "",
        "## Best Rules",
        "",
        "| control | rule | feature | op | threshold | pass | reasons |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for summary in payload["best_summaries"]:
        decision_row = summary.get("decision") or {}
        lines.append(
            f"| `{summary.get('control')}` | `{summary.get('rule')}` | `{summary.get('feature')}` | "
            f"`{summary.get('op')}` | {float(summary.get('threshold', 0.0)):.6g} | "
            f"`{decision_row.get('passes')}` | `{decision_row.get('reasons')}` |"
        )
    lines.extend(
        [
            "",
            "## Best Main Rule Replay",
            "",
            "| run | role | n | admitted | rate | first veto | first5 admitted | task datasets | mean footprint | max abs anchor delta |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["best_main"].get("runs") or []:
        lines.append(
            f"| `{row['run']}` | `{row['role']}` | {row['n_total']} | {row['n_admitted']} | "
            f"{row['accepted_rate']:.3f} | {row['first_veto_attempt']} | {row['n_first5_admitted']} | "
            f"{row['task_dataset_coverage']} | {row['mean_footprint']:.6g} | {row['max_abs_anchor_delta']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- rows: `{OUT_ROWS}`",
            f"- rule rows: `{OUT_RULES}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    rows = read_rows()
    write_csv(OUT_ROWS, rows)
    best_main = select_best(rows, mode="noncount", control="main")
    best_all = select_best(rows, mode="all", control="main")
    best_count = select_best(rows, mode="count_only", control="main")
    best_feature_shuffle = select_best(rows, mode="noncount", control="feature_shuffle")
    best_role_shuffle = select_best(rows, mode="noncount", control="role_shuffle")
    summaries = [best_main, best_all, best_count, best_feature_shuffle, best_role_shuffle]
    write_csv(OUT_RULES, [flatten_rule(s) for s in summaries])

    reasons: list[str] = []
    if not best_main["decision"]["passes"]:
        reasons.append("best_noncount_rule_fails_gate")
    if best_count["decision"]["passes"]:
        reasons.append("count_only_control_also_passes")
    if best_feature_shuffle["decision"]["passes"]:
        reasons.append("feature_shuffle_control_also_passes")
    if best_role_shuffle["decision"]["passes"]:
        reasons.append("role_shuffle_control_also_passes")
    if best_all["decision"]["passes"] and best_all.get("feature") in COUNTLIKE:
        reasons.append("best_all_rule_is_countlike")

    status = (
        "proxy_aligned_lowrank_admission_gate_pass_needs_metric_objective_dryrun"
        if not reasons
        else "proxy_aligned_lowrank_admission_gate_fail_no_gpu"
    )
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
        "status": status,
        "gpu_authorized": False,
        "reasons": reasons,
        "boundary": {
            "cpu_only": True,
            "uses_gpu": False,
            "trains_model": False,
            "runs_inference": False,
            "safe_split": str(SAFE_SPLIT),
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
        },
        "runs": {name: {"path": str(meta["path"]), "role": meta["role"]} for name, meta in RUNS.items()},
        "n_rows": len(rows),
        "best_main": best_main,
        "best_summaries": summaries,
        "outputs": {"json": str(OUT_JSON), "md": str(OUT_MD), "rows": str(OUT_ROWS), "rules": str(OUT_RULES)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "report": str(OUT_MD), "reasons": reasons}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
