#!/usr/bin/env python3
"""CPU-only route-safety controls for budget128 6k true-cell-count candidate.

Checks whether the 6k budget128 train-only/internal signal is tail-safe and not
obviously explained by eval cell counts or one dominating dataset. This gate is
query-blind and canonical-multi-blind; it does not train, infer, or use GPU.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_true_cell_count_budget128_tail_stability_6k_20260625"
DECISION_JSON = ROOT / "reports/latentfm_true_cell_count_budget128_tail_stability_6k_decision_20260625.json"
ARTIFACT_CONTROL_JSON = ROOT / "reports/latentfm_true_cell_count_budget128_tail_stability_6k_artifact_control_20260625.json"
OUT_JSON = ROOT / "reports/latentfm_true_cell_count_budget128_6k_route_safety_controls_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_TRUE_CELL_COUNT_BUDGET128_6K_ROUTE_SAFETY_CONTROLS_20260625.md"

GROUPS = {
    "cross_background": ("split_group", "internal_val_cross_background_seen_gene_proxy"),
    "family_gene": ("condition_family", "family_gene"),
    "test_single": ("condition_family", "test_single"),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    ax = np.asarray(xs, dtype=np.float64)
    ay = np.asarray(ys, dtype=np.float64)
    if float(ax.var()) <= 0.0 or float(ay.var()) <= 0.0:
        return None
    return float(np.corrcoef(ax, ay)[0, 1])


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return corr(rank(xs), rank(ys))


def perm_p_abs_spearman(xs: list[float], ys: list[float], *, n_perm: int = 2000, seed: int = 20260625) -> float | None:
    actual = spearman(xs, ys)
    if actual is None:
        return None
    rng = np.random.default_rng(seed)
    y = np.asarray(ys, dtype=np.float64)
    hits = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        val = spearman(xs, [float(v) for v in yp])
        if val is not None and abs(val) >= abs(actual):
            hits += 1
    return float(hits / n_perm)


def group_payload(run_dir: Path, family: str, group: str, role: str) -> dict[str, Any] | None:
    eval_dir = run_dir / "posthoc_eval_internal"
    if family == "split_group":
        path = eval_dir / f"split_group_eval_{role}_internal_ode20.json"
    else:
        path = eval_dir / f"condition_family_eval_{role}_internal_ode20.json"
    if not path.is_file():
        return None
    return (load_json(path).get("groups") or {}).get(group)


def metric_map(payload: dict[str, Any] | None, metric: str) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    if not payload:
        return out
    for row in payload.get("condition_metrics") or []:
        value = row.get(metric)
        if value is None:
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        out[(str(row.get("dataset")), str(row.get("condition")))] = {
            "value": val,
            "n_src_eval": row.get("n_src_eval"),
            "n_gt_eval": row.get("n_gt_eval"),
        }
    return out


def collect_delta_records(label: str, metric: str) -> list[dict[str, Any]]:
    family, group = GROUPS[label]
    records = []
    for seed in [42, 43, 44]:
        run_dir = RUN_ROOT / f"xverse_truecell_nested_budget128_tailstable_seed{seed}_6000"
        anchor = group_payload(run_dir, family, group, "anchor")
        cand = group_payload(run_dir, family, group, "candidate")
        amap = metric_map(anchor, metric)
        cmap = metric_map(cand, metric)
        for key in sorted(set(amap) & set(cmap)):
            meta = cmap[key]
            records.append(
                {
                    "seed": seed,
                    "dataset": key[0],
                    "condition": key[1],
                    "delta": float(cmap[key]["value"] - amap[key]["value"]),
                    "n_src_eval": meta.get("n_src_eval"),
                    "n_gt_eval": meta.get("n_gt_eval"),
                }
            )
    return records


def finite_log_count(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v <= 0.0 or not math.isfinite(v):
        return None
    return math.log1p(v)


def count_control(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for field in ["n_src_eval", "n_gt_eval"]:
        xs, ys = [], []
        for row in records:
            x = finite_log_count(row.get(field))
            if x is None:
                continue
            xs.append(x)
            ys.append(float(row["delta"]))
        rho = spearman(xs, ys)
        rows.append(
            {
                "field": field,
                "n": len(xs),
                "spearman": rho,
                "perm_p_abs": perm_p_abs_spearman(xs, ys, seed=20260625 + len(rows)) if rho is not None else None,
            }
        )
    return {"rows": rows}


def dataset_control(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in records:
        by_ds[str(row["dataset"])].append(float(row["delta"]))
    ds_rows = []
    for ds, vals in sorted(by_ds.items()):
        arr = np.asarray(vals, dtype=np.float64)
        ds_rows.append({"dataset": ds, "n": int(arr.size), "mean": float(arr.mean())})
    total_n = sum(r["n"] for r in ds_rows)
    total = sum(r["mean"] * r["n"] for r in ds_rows)
    overall = total / total_n if total_n else None
    loo = []
    for row in ds_rows:
        rem_n = total_n - row["n"]
        if rem_n <= 0:
            continue
        loo.append({"left_out": row["dataset"], "mean": float((total - row["mean"] * row["n"]) / rem_n)})
    pos_total = sum(max(0.0, r["mean"] * r["n"]) for r in ds_rows)
    max_pos_share = None
    if pos_total > 0.0:
        max_pos_share = max(max(0.0, r["mean"] * r["n"]) / pos_total for r in ds_rows)
    return {
        "overall": overall,
        "dataset_rows": ds_rows,
        "min_dataset": min(ds_rows, key=lambda r: r["mean"]) if ds_rows else None,
        "negative_tail_lt_minus_0p020": sum(1 for r in ds_rows if r["mean"] < -0.020),
        "leave_one_dataset_min": min((r["mean"] for r in loo), default=None),
        "max_positive_contribution_share": max_pos_share,
    }


def analyze_group(label: str, metric: str) -> dict[str, Any]:
    records = collect_delta_records(label, metric)
    return {
        "label": label,
        "metric": metric,
        "n_records": len(records),
        "mean_delta": float(np.mean([r["delta"] for r in records])) if records else None,
        "count_control": count_control(records),
        "dataset_control": dataset_control(records),
    }


def main() -> int:
    decision = load_json(DECISION_JSON)
    artifact = load_json(ARTIFACT_CONTROL_JSON)
    analyses = [analyze_group(label, metric) for label in ["cross_background", "family_gene", "test_single"] for metric in ["pearson_pert", "test_mmd"]]

    reasons = []
    warnings = []
    if decision.get("status") != "nested_matrix_internal_pass":
        reasons.append("sixk_decision_not_internal_pass")
    if artifact.get("status") != "budget128_tail_stability_artifact_control_pass_no_gpu":
        reasons.append("artifact_control_not_pass")
    primary = next((a for a in analyses if a["label"] == "cross_background" and a["metric"] == "pearson_pert"), None)
    family = next((a for a in analyses if a["label"] == "family_gene" and a["metric"] == "pearson_pert"), None)
    for name, row in [("cross_background", primary), ("family_gene", family)]:
        dc = (row or {}).get("dataset_control") or {}
        if dc.get("negative_tail_lt_minus_0p020") not in (0, None):
            reasons.append(f"{name}_negative_dataset_tail_present")
        loo = dc.get("leave_one_dataset_min")
        if loo is None or float(loo) < 0.005:
            reasons.append(f"{name}_leave_one_dataset_min_lt_0p005")
        share = dc.get("max_positive_contribution_share")
        if share is not None and float(share) > 0.50:
            warnings.append(f"{name}_single_dataset_positive_contribution_gt_0p50")
        for c in (row or {}).get("count_control", {}).get("rows", []):
            rho = c.get("spearman")
            pval = c.get("perm_p_abs")
            if rho is not None and abs(float(rho)) >= 0.30 and (pval is None or float(pval) < 0.05):
                reasons.append(f"{name}_{c['field']}_count_correlation_material")

    if reasons:
        status = "budget128_6k_route_safety_controls_fail_no_gpu"
    elif warnings:
        status = "budget128_6k_route_safety_controls_pass_with_warnings_no_gpu"
    else:
        status = "budget128_6k_route_safety_controls_pass_no_gpu"
    payload = {
        "status": status,
        "boundary": {
            "cpu_only": True,
            "reads_train_only_internal_posthoc": True,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "analyses": analyses,
        "reasons": reasons,
        "warnings": warnings,
        "gpu_authorized": False,
        "next_action": "route-freeze memo then frozen canonical single/family no-harm veto may be launched" if not reasons else "do not launch frozen no-harm until route-safety controls are addressed",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM True Cell-Count Budget128 6k Route-Safety Controls",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only route-safety controls for the 6k budget128 true-cell-count candidate.",
        "- Reads train-only/internal posthoc and completed CPU-control reports only.",
        "- Does not read canonical multi, Track C query, train, infer, or use GPU.",
        "",
        "## Controls",
        "",
        "| group | metric | n | mean delta | min dataset | neg tails | LODO min | max positive dataset share | count controls |",
        "|---|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in analyses:
        dc = row["dataset_control"]
        min_ds = dc.get("min_dataset") or {}
        count_bits = []
        for c in row["count_control"]["rows"]:
            rho = c.get("spearman")
            pval = c.get("perm_p_abs")
            count_bits.append(f"{c['field']}:rho={'NA' if rho is None else f'{rho:+.3f}'},p={'NA' if pval is None else f'{pval:.3f}'}")
        loo = dc.get("leave_one_dataset_min")
        loo_text = "NA" if loo is None else f"{float(loo):+.6f}"
        share = dc.get("max_positive_contribution_share")
        share_text = "NA" if share is None else f"{float(share):+.3f}"
        lines.append(
            f"| `{row['label']}` | `{row['metric']}` | {row['n_records']} | "
            f"{float(row['mean_delta'] or 0.0):+.6f} | `{min_ds.get('dataset')}` {float(min_ds.get('mean') or 0.0):+.6f} | "
            f"{dc.get('negative_tail_lt_minus_0p020')} | "
            f"{loo_text} | "
            f"{share_text} | "
            f"`{'; '.join(count_bits)}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- reasons: `{reasons or 'none'}`",
            f"- warnings: `{warnings or 'none'}`",
            f"- next action: `{payload['next_action']}`",
            "- GPU authorized: `False`",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
