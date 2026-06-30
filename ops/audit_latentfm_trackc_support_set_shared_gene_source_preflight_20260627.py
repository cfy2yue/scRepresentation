#!/usr/bin/env python3
"""CPU preflight for a non-duplicate Track C support-set source.

The old support-set summary gate used one dataset-level mean residual and
failed. This gate asks whether a condition-specific, permutation-invariant
support set built from train_multi rows that share a gene with the query pair
has enough query-free signal to justify implementing a trainable set/token
source.

It reads only safe trainselect condition-mean artifacts. It does not train,
infer, use GPU, read canonical multi for selection, or read Track C query.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = (
    ROOT
    / "runs/latentfm_trackc_support_set_task_input_artifacts_20260623/"
    "xverse_support_film_retry1_trainmulti_condition_means"
)
ANCHOR_PATH = RUN_ROOT / "condition_means/trainselect_anchor_train_support_multi_condition_means_ode20.json"
CANDIDATE_PATH = RUN_ROOT / "condition_means/trainselect_candidate_train_support_multi_condition_means_ode20.json"
SAFE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
FULL_V2 = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2.json"
OUT_JSON = ROOT / "reports/latentfm_trackc_support_set_shared_gene_source_preflight_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_SUPPORT_SET_SHARED_GENE_SOURCE_PREFLIGHT_20260627.md"
ALPHAS = (0.25, 0.50, 0.75, 1.00)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pair(cond: str) -> tuple[str, str] | None:
    parts = [p.strip().upper() for p in str(cond).split("+") if p.strip()]
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return None
    x = x[mask] - x[mask].mean()
    y = y[mask] - y[mask].mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return None
    return float(np.dot(x, y) / denom)


def rows(payload: dict[str, Any], group: str) -> list[dict[str, Any]]:
    return ((payload.get("groups") or {}).get(group) or {}).get("condition_metrics") or []


def vec(row: dict[str, Any], key: str) -> np.ndarray:
    arr = np.asarray(row[key], dtype=np.float32)
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise ValueError(f"bad {key} vector for {row.get('dataset')}:{row.get('condition')}")
    return arr


def paired_rows(anchor: dict[str, Any], candidate: dict[str, Any], group: str) -> list[dict[str, Any]]:
    a = {(str(r["dataset"]), str(r["condition"])): r for r in rows(anchor, group)}
    c = {(str(r["dataset"]), str(r["condition"])): r for r in rows(candidate, group)}
    if set(a) != set(c):
        raise ValueError(f"anchor/candidate mismatch for {group}")
    out = []
    for key in sorted(a):
        ar = a[key]
        cr = c[key]
        pred_anchor = vec(ar, "pred_mean")
        pred_candidate = vec(cr, "pred_mean")
        gt = vec(ar, "gt_mean")
        pert = vec(ar, "pert_mean")
        p = pair(key[1])
        if p is None:
            continue
        out.append(
            {
                "dataset": key[0],
                "condition": key[1],
                "pair": p,
                "pred_anchor": pred_anchor,
                "pred_candidate": pred_candidate,
                "gt": gt,
                "pert": pert,
                "residual": pred_candidate - pred_anchor,
                "anchor_pp": pearson(pred_anchor - pert, gt - pert),
                "candidate_pp": pearson(pred_candidate - pert, gt - pert),
            }
        )
    return out


def support_rows_for(query: dict[str, Any], train: list[dict[str, Any]], *, leave_condition: str | None) -> list[dict[str, Any]]:
    qgenes = set(query["pair"])
    out = []
    for row in train:
        if row["dataset"] != query["dataset"]:
            continue
        if leave_condition is not None and row["condition"] == leave_condition:
            continue
        rgenes = set(row["pair"])
        if qgenes & rgenes:
            out.append(row)
    return out


def summarize_support(rows_: list[dict[str, Any]]) -> np.ndarray | None:
    if not rows_:
        return None
    # Mean is permutation-invariant; this source differs from the closed route
    # by using query-conditioned shared-gene support subsets rather than all
    # train_multi rows from the dataset.
    return np.stack([r["residual"] for r in rows_], axis=0).mean(axis=0)


def score_rows(train: list[dict[str, Any]], queries: list[dict[str, Any]], alpha: float, *, loo: bool, shuffle: bool = False) -> list[dict[str, Any]]:
    support_map = {}
    for row in queries:
        support = support_rows_for(row, train, leave_condition=row["condition"] if loo else None)
        support_map[(row["dataset"], row["condition"])] = summarize_support(support)
    if shuffle:
        keys = sorted(support_map)
        vals = [support_map[k] for k in keys]
        rng = np.random.default_rng(20260627)
        order = rng.permutation(len(vals))
        support_map = {k: vals[int(order[i])] for i, k in enumerate(keys)}
    scored = []
    for row in queries:
        summary = support_map[(row["dataset"], row["condition"])]
        if summary is None:
            summary = np.zeros_like(row["residual"], dtype=np.float32)
        pred = row["pred_anchor"] + float(alpha) * summary
        task_pp = pearson(pred - row["pert"], row["gt"] - row["pert"])
        delta = None if task_pp is None or row["anchor_pp"] is None else float(task_pp - row["anchor_pp"])
        scored.append(
            {
                "dataset": row["dataset"],
                "condition": row["condition"],
                "support_count": len(support_rows_for(row, train, leave_condition=row["condition"] if loo else None)),
                "anchor_pp": row["anchor_pp"],
                "candidate_pp": row["candidate_pp"],
                "task_pp": task_pp,
                "delta_vs_anchor": delta,
            }
        )
    return scored


def dataset_deltas(scored: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in scored:
        d = row.get("delta_vs_anchor")
        if d is not None and np.isfinite(float(d)):
            grouped[str(row["dataset"])].append(float(d))
    return {ds: mean(vals) for ds, vals in grouped.items() if vals}


def boot(scored: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    ds_delta = dataset_deltas(scored)
    if not ds_delta:
        return {"delta_mean": None, "ci95": [None, None], "p_harm": None, "n_datasets": 0, "n_rows": 0}
    names = sorted(ds_delta)
    rng = np.random.default_rng(seed)
    vals = [mean(ds_delta[str(ds)] for ds in rng.choice(names, size=len(names), replace=True)) for _ in range(2000)]
    arr = np.asarray(vals)
    return {
        "delta_mean": float(mean(ds_delta.values())),
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_harm": float(np.mean(arr < 0.0)),
        "p_improvement": float(np.mean(arr > 0.0)),
        "n_datasets": len(names),
        "n_rows": len(scored),
        "dataset_deltas": ds_delta,
    }


def gate_train(summary: dict[str, Any], scored: list[dict[str, Any]]) -> bool:
    ds = summary.get("dataset_deltas") or {}
    min_support = (
        min((int(r["support_count"]) for r in scored), default=0)
        if scored
        else int(summary.get("min_support_count") or 0)
    )
    p_harm = summary.get("p_harm")
    return (
        summary.get("delta_mean") is not None
        and float(summary["delta_mean"]) >= 0.02
        and float(1.0 if p_harm is None else p_harm) <= 0.20
        and min(ds.values() or [-999.0]) >= -0.01
        and min_support >= 1
    )


def gate_support(actual: dict[str, Any], zero: dict[str, Any], shuffled: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    ds = actual.get("dataset_deltas") or {}
    actual_delta = float(actual.get("delta_mean") or -999.0)
    zero_delta = float(zero.get("delta_mean") or 0.0)
    shuffle_delta = float(shuffled.get("delta_mean") or 0.0)
    p_harm = actual.get("p_harm")
    if actual_delta < 0.04:
        reasons.append("support_actual_delta_lt_0p04")
    if float(1.0 if p_harm is None else p_harm) > 0.20:
        reasons.append("support_p_harm_gt_0p20")
    if min(ds.values() or [-999.0]) < -0.01:
        reasons.append("support_dataset_min_lt_minus_0p01")
    if actual_delta - zero_delta < 0.02:
        reasons.append("zero_control_not_0p02_below_actual")
    if actual_delta - shuffle_delta < 0.02:
        reasons.append("shuffle_control_not_0p02_below_actual")
    return not reasons, reasons


def main() -> None:
    anchor = load_json(ANCHOR_PATH)
    candidate = load_json(CANDIDATE_PATH)
    if str(anchor.get("split_file")) != str(SAFE_SPLIT) or str(candidate.get("split_file")) != str(SAFE_SPLIT):
        raise RuntimeError("condition-mean artifacts are not from the safe trainselect split")
    if str(FULL_V2) in json.dumps(anchor) or str(FULL_V2) in json.dumps(candidate):
        raise RuntimeError("full v2 query split appeared in support-set source artifacts")
    train = paired_rows(anchor, candidate, "train_multi")
    support = paired_rows(anchor, candidate, "support_val_multi")

    train_summaries = []
    selected = None
    for alpha in ALPHAS:
        scored_all = score_rows(train, train, alpha, loo=True)
        scored = [r for r in scored_all if int(r["support_count"]) > 0]
        summary = {"alpha": alpha, **boot(scored, seed=1000 + int(alpha * 100))}
        summary["min_support_count"] = min((int(r["support_count"]) for r in scored), default=0)
        summary["supported_rows"] = len(scored)
        summary["all_rows"] = len(scored_all)
        summary["supported_fraction"] = len(scored) / max(1, len(scored_all))
        train_summaries.append(summary)
        if selected is None and gate_train(summary, scored):
            selected = {"summary": summary, "scored": scored}

    support_summary = zero_summary = shuffle_summary = None
    support_reasons = ["no_alpha_passed_train_loo_gate"]
    if selected is not None:
        alpha = float(selected["summary"]["alpha"])
        actual_rows_all = score_rows(train, support, alpha, loo=False)
        zero_rows_all = score_rows(train, support, 0.0, loo=False)
        shuffle_rows_all = score_rows(train, support, alpha, loo=False, shuffle=True)
        actual_rows = [r for r in actual_rows_all if int(r["support_count"]) > 0]
        supported_keys = {(r["dataset"], r["condition"]) for r in actual_rows}
        zero_rows = [r for r in zero_rows_all if (r["dataset"], r["condition"]) in supported_keys]
        shuffle_rows = [r for r in shuffle_rows_all if (r["dataset"], r["condition"]) in supported_keys]
        support_summary = {"alpha": alpha, **boot(actual_rows, seed=2000 + int(alpha * 100))}
        support_summary["min_support_count"] = min((int(r["support_count"]) for r in actual_rows), default=0)
        support_summary["supported_rows"] = len(actual_rows)
        support_summary["all_rows"] = len(actual_rows_all)
        support_summary["supported_fraction"] = len(actual_rows) / max(1, len(actual_rows_all))
        zero_summary = {"alpha": 0.0, **boot(zero_rows, seed=2001)}
        shuffle_summary = {"alpha": alpha, **boot(shuffle_rows, seed=2002 + int(alpha * 100))}
        _, support_reasons = gate_support(support_summary, zero_summary, shuffle_summary)

    status = (
        "shared_gene_support_source_pass_encoder_unit_next_no_gpu"
        if selected is not None and not support_reasons
        else "shared_gene_support_source_near_signal_encoder_unit_next_no_gpu"
        if (
            selected is not None
            and support_reasons == ["support_p_harm_gt_0p20"]
            and support_summary is not None
            and float(support_summary.get("delta_mean") or -999.0) >= 0.04
            and min((support_summary.get("dataset_deltas") or {}).values() or [-999.0]) >= -0.01
            and zero_summary is not None
            and shuffle_summary is not None
            and float(support_summary.get("delta_mean") or 0.0) - float(zero_summary.get("delta_mean") or 0.0) >= 0.02
            and float(support_summary.get("delta_mean") or 0.0) - float(shuffle_summary.get("delta_mean") or 0.0) >= 0.02
        )
        else "shared_gene_support_source_fail_no_gpu"
    )
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "safe_trainselect_split": str(SAFE_SPLIT),
            "full_v2_query_used": False,
            "canonical_multi_selection_used": False,
            "training": False,
            "inference": False,
            "gpu": False,
        },
        "inputs": {"anchor": str(ANCHOR_PATH), "candidate": str(CANDIDATE_PATH)},
        "rows": {"train_multi": len(train), "support_val_multi": len(support)},
        "source": "query_condition_shared_gene_train_multi_mean_residual",
        "target_policy": "enable_only_when_query_has_at_least_one_same_dataset_train_multi_shared_gene_support_row",
        "alpha_grid": list(ALPHAS),
        "train_loo_summaries": train_summaries,
        "selected_train_loo_summary": None if selected is None else selected["summary"],
        "support_val_summary": support_summary,
        "zero_support_control": zero_summary,
        "shuffled_support_control": shuffle_summary,
        "decision_reasons": support_reasons,
        "next_action": (
            "implement encoder/source unit gate before GPU" if status.endswith("next_no_gpu")
            else "close shared-gene mean residual source; do not implement GPU launcher"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Track C Shared-Gene Support-Set Source Preflight",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "CPU-only scoring over safe-trainselect condition-mean artifacts. No training, inference, canonical multi selection, full v2 query, or GPU.",
        "",
        "## Rows",
        "",
        f"* train_multi: `{len(train)}`",
        f"* support_val_multi: `{len(support)}`",
        "",
        "## Train LOO",
        "",
        "| alpha | delta | CI95 | p_harm | min dataset | rows | coverage | min support | pass |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in train_summaries:
        ds_vals = list((s.get("dataset_deltas") or {}).values())
        min_ds = min(ds_vals) if ds_vals else None
        lines.append(
            f"| {s['alpha']:.2f} | {s.get('delta_mean')} | {s.get('ci95')} | {s.get('p_harm')} | "
            f"{min_ds} | {s.get('supported_rows')}/{s.get('all_rows')} | {s.get('supported_fraction')} | "
            f"{s.get('min_support_count')} | `{gate_train(s, [])}` |"
        )
    lines.extend(["", "## Support-Val", ""])
    if support_summary is None:
        lines.append("- not evaluated because no train LOO alpha passed")
    else:
        lines.append(f"* selected alpha: `{support_summary['alpha']}`")
        lines.append(f"* actual delta: `{support_summary.get('delta_mean')}`")
        lines.append(f"* supported rows: `{support_summary.get('supported_rows')}/{support_summary.get('all_rows')}`")
        lines.append(f"* zero delta: `{zero_summary.get('delta_mean') if zero_summary else None}`")
        lines.append(f"* shuffled delta: `{shuffle_summary.get('delta_mean') if shuffle_summary else None}`")
        lines.append(f"* actual dataset deltas: `{support_summary.get('dataset_deltas')}`")
    lines.extend(["", "## Decision Reasons", ""])
    lines.extend(f"- `{r}`" for r in support_reasons)
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This preflight can only authorize the next CPU encoder/source unit gate. It cannot authorize GPU training.",
            "",
            "## Outputs",
            "",
            f"* JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "md": str(OUT_MD), "json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
