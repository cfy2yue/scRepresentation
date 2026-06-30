#!/usr/bin/env python3
"""Nested true cell-count matrix decision with seed and bootstrap summaries.

This script consumes completed train-only/internal posthoc JSONs. It does not
read canonical multi, Track C query, train, infer, or use GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = Path(os.environ.get("LATENTFM_TRUE_CELL_COUNT_NESTED_RUN_ROOT", ROOT / "runs/latentfm_true_cell_count_nested_smokes_20260624"))
OUT_JSON = Path(os.environ.get("LATENTFM_TRUE_CELL_COUNT_NESTED_OUT_JSON", ROOT / "reports/latentfm_true_cell_count_nested_matrix_decision_20260624.json"))
OUT_MD = Path(os.environ.get("LATENTFM_TRUE_CELL_COUNT_NESTED_OUT_MD", ROOT / "reports/LATENTFM_TRUE_CELL_COUNT_NESTED_MATRIX_DECISION_20260624.md"))
EXPECTED_RUNS = int(os.environ.get("LATENTFM_TRUE_CELL_COUNT_NESTED_EXPECTED_RUNS", "9"))

GROUP_SPECS = {
    "cross_background": ("split_group", "internal_val_cross_background_seen_gene_proxy"),
    "family_gene": ("condition_family", "family_gene"),
    "test_single": ("condition_family", "test_single"),
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


def parse_budget_seed(name: str) -> tuple[int | None, int | None]:
    seed_match = re.search(r"_seed(\d+)", name)
    budget_matches = re.findall(r"_budget(\d+)", name[: seed_match.start()] if seed_match else name)
    if not seed_match or not budget_matches:
        return None, None
    return int(budget_matches[-1]), int(seed_match.group(1))


def group_payload(run_dir: Path, family: str, role: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    eval_dir = run_dir / "posthoc_eval_internal"
    if family == "split_group":
        anchor = load_json(eval_dir / "split_group_eval_anchor_internal_ode20.json")
        candidate = load_json(eval_dir / "split_group_eval_candidate_internal_ode20.json")
    else:
        anchor = load_json(eval_dir / "condition_family_eval_anchor_internal_ode20.json")
        candidate = load_json(eval_dir / "condition_family_eval_candidate_internal_ode20.json")
    return ((anchor or {}).get("groups") or {}).get(role), ((candidate or {}).get("groups") or {}).get(role)


def metric_delta(anchor: dict[str, Any] | None, candidate: dict[str, Any] | None, key: str) -> float | None:
    if not anchor or not candidate:
        return None
    a = anchor.get(key)
    c = candidate.get(key)
    if a is None or c is None:
        return None
    return float(c) - float(a)


def paired_condition_deltas(anchor: dict[str, Any] | None, candidate: dict[str, Any] | None, key: str) -> list[float]:
    return [r["delta"] for r in paired_condition_delta_records(anchor, candidate, key)]


def paired_condition_delta_records(anchor: dict[str, Any] | None, candidate: dict[str, Any] | None, key: str) -> list[dict[str, Any]]:
    if not anchor or not candidate:
        return []
    def to_map(payload: dict[str, Any]) -> dict[tuple[str, str], float]:
        out = {}
        for row in payload.get("condition_metrics") or []:
            value = row.get(key)
            if value is None:
                continue
            try:
                out[(str(row.get("dataset")), str(row.get("condition")))] = float(value)
            except (TypeError, ValueError):
                continue
        return out
    amap = to_map(anchor)
    cmap = to_map(candidate)
    keys = sorted(set(amap) & set(cmap))
    return [
        {
            "dataset": k[0],
            "condition": k[1],
            "delta": cmap[k] - amap[k],
        }
        for k in keys
    ]


def bootstrap_ci(values: list[float], *, n_boot: int, seed: int) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": None, "ci95": [None, None], "p_le_0": None}
    if arr.size == 1:
        mean = float(arr.mean())
        return {"n": 1, "mean": mean, "ci95": [mean, mean], "p_le_0": float(mean <= 0.0)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    means = arr[idx].mean(axis=1)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "p_le_0": float((means <= 0.0).mean()),
    }


def dataset_tail(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in records:
        by_ds[str(row["dataset"])].append(float(row["delta"]))
    ds_rows = []
    for ds, vals in sorted(by_ds.items()):
        arr = np.asarray(vals, dtype=np.float64)
        ds_rows.append({"dataset": ds, "n": int(arr.size), "mean": float(arr.mean())})
    values = [r["mean"] for r in ds_rows]
    min_row = min(ds_rows, key=lambda r: r["mean"]) if ds_rows else None
    loo = []
    total = sum(r["mean"] * r["n"] for r in ds_rows)
    n_total = sum(r["n"] for r in ds_rows)
    for row in ds_rows:
        rem_n = n_total - row["n"]
        if rem_n <= 0:
            continue
        loo.append({"left_out": row["dataset"], "mean": float((total - row["mean"] * row["n"]) / rem_n)})
    return {
        "n_datasets": len(ds_rows),
        "dataset_rows": ds_rows,
        "min_dataset": min_row,
        "negative_tail_lt_minus_0p020": sum(1 for r in ds_rows if r["mean"] < -0.020),
        "leave_one_dataset_min": min((r["mean"] for r in loo), default=None),
    }


def summarize_run(run_dir: Path) -> dict[str, Any]:
    budget, seed = parse_budget_seed(run_dir.name)
    train_exit = read_exit(run_dir / "EXIT_CODE")
    posthoc_exit = read_exit(run_dir / "POSTHOC_EXIT_CODE")
    groups = {}
    condition_deltas = {}
    for label, (family, role) in GROUP_SPECS.items():
        anchor, candidate = group_payload(run_dir, family, role)
        groups[label] = {
            "delta_pearson_pert": metric_delta(anchor, candidate, "pearson_pert"),
            "delta_mmd": metric_delta(anchor, candidate, "test_mmd"),
            "n_conds": (candidate or {}).get("n_conds"),
        }
        condition_deltas[label] = {
            "pearson_pert": paired_condition_deltas(anchor, candidate, "pearson_pert"),
            "test_mmd": paired_condition_deltas(anchor, candidate, "test_mmd"),
            "pearson_pert_records": paired_condition_delta_records(anchor, candidate, "pearson_pert"),
            "test_mmd_records": paired_condition_delta_records(anchor, candidate, "test_mmd"),
        }
    reasons = []
    if train_exit != 0 or posthoc_exit != 0:
        reasons.append("train_or_posthoc_not_complete")
    cross = groups["cross_background"]
    family = groups["family_gene"]
    test_single = groups["test_single"]
    if cross["delta_pearson_pert"] is None or cross["delta_pearson_pert"] < 0.010:
        reasons.append("cross_background_pp_delta_lt_0p010")
    if family["delta_pearson_pert"] is None or family["delta_pearson_pert"] < 0.0:
        reasons.append("family_gene_pp_negative")
    if family["delta_mmd"] is None or family["delta_mmd"] > 0.001:
        reasons.append("family_gene_mmd_delta_gt_0p001")
    if test_single["delta_pearson_pert"] is None or test_single["delta_pearson_pert"] < -0.005:
        reasons.append("test_single_pp_hard_harm")
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "budget": budget,
        "seed": seed,
        "train_exit": train_exit,
        "posthoc_exit": posthoc_exit,
        "groups": groups,
        "condition_deltas": condition_deltas,
        "status": "pass" if not reasons else "pending_or_fail",
        "reasons": reasons,
    }


def mean(xs: list[float]) -> float | None:
    vals = [x for x in xs if x is not None and math.isfinite(float(x))]
    return None if not vals else float(np.mean(vals))


def matrix_summary(rows: list[dict[str, Any]], *, n_boot: int) -> dict[str, Any]:
    complete = [r for r in rows if r["train_exit"] == 0 and r["posthoc_exit"] == 0]
    by_budget: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in complete:
        if row["budget"] is not None:
            by_budget[int(row["budget"])].append(row)
    budget_rows = []
    for budget in sorted(by_budget):
        br = by_budget[budget]
        item: dict[str, Any] = {"budget": budget, "n_complete": len(br), "seed_passes": sum(r["status"] == "pass" for r in br)}
        for label in GROUP_SPECS:
            item[f"{label}_pp_delta_mean"] = mean([r["groups"][label]["delta_pearson_pert"] for r in br])
            item[f"{label}_mmd_delta_mean"] = mean([r["groups"][label]["delta_mmd"] for r in br])
            pp_vals = [x for r in br for x in r["condition_deltas"][label]["pearson_pert"]]
            mmd_vals = [x for r in br for x in r["condition_deltas"][label]["test_mmd"]]
            pp_records = [x for r in br for x in r["condition_deltas"][label]["pearson_pert_records"]]
            mmd_records = [x for r in br for x in r["condition_deltas"][label]["test_mmd_records"]]
            item[f"{label}_pp_condition_bootstrap"] = bootstrap_ci(pp_vals, n_boot=n_boot, seed=20260624 + budget)
            item[f"{label}_mmd_condition_bootstrap"] = bootstrap_ci(mmd_vals, n_boot=n_boot, seed=20260625 + budget)
            item[f"{label}_pp_dataset_tail"] = dataset_tail(pp_records)
            item[f"{label}_mmd_dataset_tail"] = dataset_tail(mmd_records)
        budget_rows.append(item)
    cross_curve = [(r["budget"], r.get("cross_background_pp_delta_mean")) for r in budget_rows if r.get("cross_background_pp_delta_mean") is not None]
    peak_budget = None
    monotonic_increasing = None
    if cross_curve:
        peak_budget = max(cross_curve, key=lambda x: x[1])[0]
        vals = [x[1] for x in sorted(cross_curve)]
        monotonic_increasing = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
    reasons = []
    if len(complete) < EXPECTED_RUNS:
        reasons.append(f"matrix_incomplete:{len(complete)}_of_{EXPECTED_RUNS}")
    for item in budget_rows:
        if item["n_complete"] == 3:
            if item.get("cross_background_pp_delta_mean") is None or item["cross_background_pp_delta_mean"] < 0.010:
                reasons.append(f"budget{item['budget']}_mean_cross_pp_lt_0p010")
            if item.get("family_gene_pp_delta_mean") is None or item["family_gene_pp_delta_mean"] < 0.0:
                reasons.append(f"budget{item['budget']}_mean_family_pp_negative")
            if item.get("family_gene_mmd_delta_mean") is None or item["family_gene_mmd_delta_mean"] > 0.001:
                reasons.append(f"budget{item['budget']}_mean_family_mmd_gt_0p001")
            if item["seed_passes"] < 2:
                reasons.append(f"budget{item['budget']}_fewer_than_2_seed_passes")
            tail = item["cross_background_pp_dataset_tail"]
            if tail["negative_tail_lt_minus_0p020"]:
                reasons.append(f"budget{item['budget']}_cross_pp_negative_dataset_tails")
            loo_min = tail.get("leave_one_dataset_min")
            if loo_min is not None and loo_min < 0.005:
                reasons.append(f"budget{item['budget']}_cross_pp_leave_one_dataset_min_lt_0p005")
    status = "nested_matrix_pending"
    action = "wait_without_polling"
    if len(complete) == EXPECTED_RUNS:
        if reasons:
            status = "nested_matrix_fail_or_mechanism_only"
            action = "close_or_limit_claim_before_controls"
        else:
            status = "nested_matrix_internal_pass"
            action = "run count-only/dataset-ID controls and frozen no-harm only after route freeze"
    return {
        "n_runs": len(rows),
        "n_complete": len(complete),
        "n_expected": EXPECTED_RUNS,
        "budget_rows": budget_rows,
        "peak_budget_by_cross_pp_mean": peak_budget,
        "cross_pp_monotonic_increasing": monotonic_increasing,
        "reasons": reasons,
        "status": status,
        "action": action,
    }


def fmt(x: Any) -> str:
    if x is None:
        return "NA"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):+.6f}"


def render_md(payload: dict[str, Any]) -> str:
    summary = payload["matrix_summary"]
    lines = [
        "# LatentFM True Cell-Count Nested Matrix Decision",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- Summarizes nested-v2 true cell-count matrix only.",
        "- Uses train-only/internal posthoc outputs.",
        "- Does not read canonical multi, Track C query, train, infer, or use GPU.",
        "- Does not authorize deployable or final scaling-law claims.",
        "",
        "## Budget Summary",
        "",
        "| budget | complete | seed passes | cross pp mean | family pp mean | family MMD mean | cross pp CI | family pp CI | min cross dataset | neg tails |",
        "|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in summary["budget_rows"]:
        cross_ci = row["cross_background_pp_condition_bootstrap"]["ci95"]
        fam_ci = row["family_gene_pp_condition_bootstrap"]["ci95"]
        min_ds = (row["cross_background_pp_dataset_tail"].get("min_dataset") or {}).get("mean")
        neg_tails = row["cross_background_pp_dataset_tail"].get("negative_tail_lt_minus_0p020")
        lines.append(
            f"| {row['budget']} | {row['n_complete']} | {row['seed_passes']} | "
            f"{fmt(row.get('cross_background_pp_delta_mean'))} | {fmt(row.get('family_gene_pp_delta_mean'))} | "
            f"{fmt(row.get('family_gene_mmd_delta_mean'))} | "
            f"[{fmt(cross_ci[0])}, {fmt(cross_ci[1])}] | [{fmt(fam_ci[0])}, {fmt(fam_ci[1])}] | "
            f"{fmt(min_ds)} | {neg_tails} |"
        )
    lines.extend(
        [
            "",
            "## Shape",
            "",
            f"- peak budget by mean cross pp: `{summary['peak_budget_by_cross_pp_mean']}`",
            f"- monotonic increasing cross pp: `{summary['cross_pp_monotonic_increasing']}`",
            "",
            "## Run Rows",
            "",
            "| run | budget | seed | status | cross pp | family pp | family MMD | reasons |",
            "|---|---:|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            f"| `{row['run_name']}` | {row.get('budget')} | {row.get('seed')} | `{row['status']}` | "
            f"{fmt(row['groups']['cross_background']['delta_pearson_pert'])} | "
            f"{fmt(row['groups']['family_gene']['delta_pearson_pert'])} | "
            f"{fmt(row['groups']['family_gene']['delta_mmd'])} | {', '.join(row['reasons']) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- action: `{payload['action']}`",
            f"- reasons: `{payload['reasons'] or 'none'}`",
            f"- GPU authorized by this report: `{payload['gpu_authorized']}`",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    args = ap.parse_args()
    run_dirs = sorted(p for p in RUN_ROOT.iterdir() if p.is_dir() and (p / "RUN_STATUS.md").is_file()) if RUN_ROOT.exists() else []
    rows = [summarize_run(p) for p in run_dirs]
    summary = matrix_summary(rows, n_boot=args.n_bootstrap)
    payload = {
        "status": summary["status"],
        "action": summary["action"],
        "reasons": summary["reasons"],
        "run_root": str(RUN_ROOT),
        "rows": rows,
        "matrix_summary": summary,
        "gpu_authorized": False,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_json": str(OUT_JSON), "out_md": str(OUT_MD)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
