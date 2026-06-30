#!/usr/bin/env python3
"""Summarize the Track C support-set task summary CPU gate.

This gate consumes safe-trainselect condition-mean artifacts for
``train_multi`` and ``support_val_multi``. It tests a distinct support-set task
mechanism: build a per-dataset task residual summary from train_multi support
rows, select a small predeclared alpha using train_multi leave-one-condition
scoring only, then score support_val_multi. Held-out query and canonical
test_multi are forbidden.

The evaluated offline rule is:

    pred = anchor_pred + alpha * mean_train_multi(candidate_pred - anchor_pred)

for each dataset. This is not a trained support-set adapter and does not
authorize GPU training by itself. Passing would only authorize a later
query-free posthoc/code gate with MMD/no-harm checks.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_RUN_ROOT = (
    ROOT
    / "runs/latentfm_trackc_support_set_task_input_artifacts_20260623/"
    "xverse_support_film_retry1_trainmulti_condition_means"
)
CPU_ROUTE_GAP_JSON = ROOT / "reports/latentfm_trackc_alternative_support_conditioning_cpu_gate_20260623.json"
OUT_JSON = ROOT / "reports/latentfm_trackc_support_set_task_summary_gate_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_SUPPORT_SET_TASK_SUMMARY_GATE_20260623.md"
ALPHAS = (0.25, 0.50, 0.75, 1.00)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def pearson_np(x: np.ndarray, y: np.ndarray) -> float | None:
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


def group_rows(payload: dict[str, Any], group: str) -> list[dict[str, Any]]:
    return ((payload.get("groups") or {}).get(group) or {}).get("condition_metrics") or []


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("dataset")), str(row.get("condition"))


def require_mean(row: dict[str, Any], key: str) -> np.ndarray:
    value = row.get(key)
    if value is None:
        raise ValueError(f"row missing {key}: {row_key(row)}")
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"row {row_key(row)} {key} must be a vector")
    if not np.isfinite(arr).all():
        raise ValueError(f"row {row_key(row)} {key} contains non-finite values")
    return arr


def paired_rows(anchor_payload: dict[str, Any], candidate_payload: dict[str, Any], group: str) -> list[dict[str, Any]]:
    anchors = {row_key(row): row for row in group_rows(anchor_payload, group)}
    candidates = {row_key(row): row for row in group_rows(candidate_payload, group)}
    missing = sorted(set(anchors) ^ set(candidates))
    if missing:
        raise ValueError(f"anchor/candidate condition mismatch for {group}: {len(missing)}")
    out: list[dict[str, Any]] = []
    for key in sorted(anchors):
        a = anchors[key]
        c = candidates[key]
        pred_anchor = require_mean(a, "pred_mean")
        pred_candidate = require_mean(c, "pred_mean")
        gt_mean = require_mean(a, "gt_mean")
        pert_mean = require_mean(a, "pert_mean")
        if pred_anchor.shape != pred_candidate.shape or pred_anchor.shape != gt_mean.shape:
            raise ValueError(f"mean shape mismatch for {key}")
        out.append(
            {
                "dataset": key[0],
                "condition": key[1],
                "anchor_pp": pearson_np(pred_anchor - pert_mean, gt_mean - pert_mean),
                "candidate_pp": pearson_np(pred_candidate - pert_mean, gt_mean - pert_mean),
                "pred_anchor": pred_anchor,
                "pred_candidate": pred_candidate,
                "gt_mean": gt_mean,
                "pert_mean": pert_mean,
                "residual": pred_candidate - pred_anchor,
            }
        )
    if not out:
        raise ValueError(f"no rows for group {group}")
    return out


def route_gap_by_dataset(path: Path) -> dict[str, float]:
    payload = load_json(path)
    out: dict[str, float] = {}
    for row in ((payload.get("real") or {}).get("dataset_breakdown") or []):
        ds = str(row.get("dataset"))
        route = row.get("support_selected_route")
        target = row.get("candidate")
        if route is not None and target is not None:
            out[ds] = float(target) - float(route)
    return out


def by_dataset(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row["dataset"])].append(row)
    return dict(out)


def mean_residual(rows: list[dict[str, Any]]) -> np.ndarray | None:
    residuals = [np.asarray(row["residual"], dtype=np.float32) for row in rows]
    if not residuals:
        return None
    return np.stack(residuals, axis=0).mean(axis=0)


def train_loo_score(train_rows: list[dict[str, Any]], alpha: float) -> list[dict[str, Any]]:
    grouped = by_dataset(train_rows)
    scored: list[dict[str, Any]] = []
    for ds, rows in grouped.items():
        for idx, row in enumerate(rows):
            support_rows = rows[:idx] + rows[idx + 1 :]
            summary = mean_residual(support_rows)
            if summary is None:
                summary = np.zeros_like(row["residual"], dtype=np.float32)
            pred = row["pred_anchor"] + float(alpha) * summary
            pp = pearson_np(pred - row["pert_mean"], row["gt_mean"] - row["pert_mean"])
            scored.append(
                {
                    "dataset": ds,
                    "condition": row["condition"],
                    "anchor_pp": row["anchor_pp"],
                    "task_pp": pp,
                    "delta_vs_anchor": None if pp is None or row["anchor_pp"] is None else pp - row["anchor_pp"],
                }
            )
    return scored


def support_score(
    train_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
    alpha: float,
    *,
    shuffle: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    summaries = {ds: mean_residual(rows) for ds, rows in by_dataset(train_rows).items()}
    if shuffle:
        keys = sorted(summaries)
        vals = [summaries[key] for key in keys]
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(vals))
        summaries = {key: vals[int(order[i])] for i, key in enumerate(keys)}
    scored: list[dict[str, Any]] = []
    for row in support_rows:
        summary = summaries.get(str(row["dataset"]))
        if summary is None:
            summary = np.zeros_like(row["residual"], dtype=np.float32)
        pred = row["pred_anchor"] + float(alpha) * summary
        pp = pearson_np(pred - row["pert_mean"], row["gt_mean"] - row["pert_mean"])
        scored.append(
            {
                "dataset": row["dataset"],
                "condition": row["condition"],
                "anchor_pp": row["anchor_pp"],
                "candidate_pp": row["candidate_pp"],
                "task_pp": pp,
                "delta_vs_anchor": None if pp is None or row["anchor_pp"] is None else pp - row["anchor_pp"],
            }
        )
    return scored


def dataset_means(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is not None and np.isfinite(float(value)):
            grouped[str(row["dataset"])].append(float(value))
    return {ds: mean(vals) for ds, vals in grouped.items() if vals}


def bootstrap_delta(rows: list[dict[str, Any]], candidate: str, baseline: str, *, n_boot: int, seed: int) -> dict[str, Any]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        c = row.get(candidate)
        b = row.get(baseline)
        if c is not None and b is not None and np.isfinite(float(c)) and np.isfinite(float(b)):
            grouped[str(row["dataset"])].append((float(c), float(b)))
    ds_items = sorted((ds, vals) for ds, vals in grouped.items() if vals)
    if not ds_items:
        return {"status": "missing", "delta_mean": None, "p_harm": None, "ci95": [None, None], "n_datasets": 0}
    ds_delta = {ds: mean(c - b for c, b in vals) for ds, vals in ds_items}
    names = [ds for ds, _ in ds_items]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n_boot)):
        sample = rng.choice(names, size=len(names), replace=True)
        vals.append(mean(ds_delta[str(ds)] for ds in sample))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "status": "ok",
        "delta_mean": float(mean(ds_delta.values())),
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_improvement": float(np.mean(arr > 0.0)),
        "p_harm": float(np.mean(arr < 0.0)),
        "n_datasets": len(names),
        "n_conditions": sum(len(v) for _, v in ds_items),
        "dataset_deltas": ds_delta,
    }


def summary_for(rows: list[dict[str, Any]], route_gaps: dict[str, float], alpha: float, *, seed: int) -> dict[str, Any]:
    by_ds = dataset_means(rows, "delta_vs_anchor")
    ds_rows = []
    for ds, delta in sorted(by_ds.items()):
        gap = route_gaps.get(ds)
        ds_rows.append(
            {
                "dataset": ds,
                "mean_delta_pp": delta,
                "route_gap_pp": gap,
                "route_gap_closed_fraction": None if gap is None or abs(gap) <= 1e-12 else delta / gap,
                "n_conditions": sum(1 for row in rows if row["dataset"] == ds),
            }
        )
    return {
        "alpha": float(alpha),
        "paired": bootstrap_delta(rows, "task_pp", "anchor_pp", n_boot=2000, seed=seed),
        "dataset_summary": ds_rows,
        "scored_rows": rows,
    }


def find_dataset(summary: dict[str, Any], dataset: str) -> dict[str, Any]:
    for row in summary.get("dataset_summary") or []:
        if row.get("dataset") == dataset:
            return row
    return {}


def support_gate_passes(summary: dict[str, Any]) -> bool:
    wessels = find_dataset(summary, "Wessels")
    norman = find_dataset(summary, "NormanWeissman2019_filtered")
    paired = summary.get("paired") or {}
    return (
        float(wessels.get("mean_delta_pp") if wessels.get("mean_delta_pp") is not None else -999.0) >= 0.02
        and float(wessels.get("route_gap_closed_fraction") if wessels.get("route_gap_closed_fraction") is not None else -999.0) >= 0.05
        and float(norman.get("mean_delta_pp") if norman.get("mean_delta_pp") is not None else -999.0) >= -0.02
        and float(paired.get("p_harm") if paired.get("p_harm") is not None else 1.0) <= 0.20
    )


def select_alpha(train_rows: list[dict[str, Any]], route_gaps: dict[str, float]) -> dict[str, Any] | None:
    for alpha in ALPHAS:
        scored = train_loo_score(train_rows, alpha)
        summary = summary_for(scored, route_gaps, alpha, seed=500 + int(alpha * 1000))
        if support_gate_passes(summary):
            return summary
    return None


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    selected_train = payload.get("selected_train_loo_summary")
    support = payload.get("support_val_summary")
    zero = payload.get("zero_support_control")
    shuffled = payload.get("shuffled_support_control")
    if not selected_train:
        reasons.append("no_alpha_passed_train_multi_loo_gate")
    if not support or not support_gate_passes(support):
        reasons.append("support_val_gate_failed")
    if zero and support_gate_passes(zero):
        reasons.append("zero_support_control_passed_unexpectedly")
    if shuffled and support_gate_passes(shuffled):
        reasons.append("shuffled_support_control_passed_unexpectedly")
    if support and shuffled:
        support_w = float((find_dataset(support, "Wessels") or {}).get("mean_delta_pp") or 0.0)
        shuffled_w = float((find_dataset(shuffled, "Wessels") or {}).get("mean_delta_pp") or 0.0)
        if shuffled_w >= 0.02 or shuffled_w >= 0.5 * support_w:
            reasons.append("shuffled_support_control_did_not_lose_wessels_signal")
    status = (
        "trackc_support_set_task_summary_gate_pass_posthoc_mmd_gate_next_no_gpu"
        if not reasons
        else "trackc_support_set_task_summary_gate_fail_no_gpu"
    )
    return {
        "status": status,
        "gpu_authorization": "none",
        "next_authorization": "query_free_posthoc_mmd_gate_only" if not reasons else "none",
        "reasons": reasons,
        "rules": [
            "select smallest alpha passing train_multi leave-one-condition gate",
            "support_val Wessels pp delta >= +0.02",
            "support_val Wessels route-gap closure >= +0.05",
            "support_val Norman pp delta >= -0.02",
            "support_val paired pp p_harm <= 0.20",
            "zero support control must fail support_val gate",
            "shuffled support control must fail and lose Wessels signal",
            "mean-vector gate does not evaluate MMD; passing only authorizes a query-free MMD/no-harm posthoc gate",
        ],
    }


def build_payload(run_root: Path, route_gap_json: Path) -> dict[str, Any]:
    cm_dir = run_root / "condition_means"
    anchor_path = cm_dir / "trainselect_anchor_train_support_multi_condition_means_ode20.json"
    candidate_path = cm_dir / "trainselect_candidate_train_support_multi_condition_means_ode20.json"
    anchor = load_json(anchor_path)
    candidate = load_json(candidate_path)
    train_rows = paired_rows(anchor, candidate, "train_multi")
    support_rows = paired_rows(anchor, candidate, "support_val_multi")
    route_gaps = route_gap_by_dataset(route_gap_json)
    train_summaries = []
    selected_train = None
    for alpha in ALPHAS:
        summary = summary_for(
            train_loo_score(train_rows, alpha),
            route_gaps,
            alpha,
            seed=500 + int(alpha * 1000),
        )
        train_summaries.append(summary)
        if selected_train is None and support_gate_passes(summary):
            selected_train = summary
    if selected_train is None:
        support = None
        zero = None
        shuffled = None
    else:
        alpha = float(selected_train["alpha"])
        support = summary_for(
            support_score(train_rows, support_rows, alpha),
            route_gaps,
            alpha,
            seed=900 + int(alpha * 1000),
        )
        zero = summary_for(
            support_score(train_rows, support_rows, 0.0),
            route_gaps,
            0.0,
            seed=901,
        )
        shuffled = summary_for(
            support_score(train_rows, support_rows, alpha, shuffle=True),
            route_gaps,
            alpha,
            seed=902 + int(alpha * 1000),
        )
    payload = {
        "run_root": str(run_root),
        "inputs": {
            "anchor_condition_means": str(anchor_path),
            "candidate_condition_means": str(candidate_path),
            "route_gap_json": str(route_gap_json),
        },
        "n_rows": {"train_multi": len(train_rows), "support_val_multi": len(support_rows)},
        "alpha_grid": list(ALPHAS),
        "train_loo_summaries": train_summaries,
        "selected_train_loo_summary": selected_train,
        "support_val_summary": support,
        "zero_support_control": zero,
        "shuffled_support_control": shuffled,
        "heldout_query_used": False,
        "canonical_multi_selection_used": False,
        "mmd_evaluated": False,
    }
    payload["decision"] = decide(payload)
    return payload


def render(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# Track C Support-Set Task Summary Gate",
        "",
        f"Status: `{decision['status']}`",
        f"GPU authorization: `{decision['gpu_authorization']}`",
        f"Next authorization: `{decision['next_authorization']}`",
        "",
        "## Scope",
        "",
        "This query-free CPU gate uses safe-trainselect train_multi to build a dataset-level support-set task residual summary and scores support_val_multi. It does not read held-out query or canonical test_multi.",
        "",
        "## Row Counts",
        "",
        f"- train_multi: `{payload['n_rows']['train_multi']}`",
        f"- support_val_multi: `{payload['n_rows']['support_val_multi']}`",
        "",
    ]
    selected = payload.get("selected_train_loo_summary")
    if selected:
        lines.extend(
            [
                "## Selected Train-Multi LOO Alpha",
                "",
                f"- alpha: `{selected['alpha']}`",
                f"- train LOO pp delta: `{fmt((selected.get('paired') or {}).get('delta_mean'))}`",
                f"- train LOO p_harm: `{fmt((selected.get('paired') or {}).get('p_harm'))}`",
                "",
            ]
        )
    else:
        lines.extend(["## Selected Train-Multi LOO Alpha", "", "- none", ""])
    for title, key in (
        ("Support-Val Summary", "support_val_summary"),
        ("Zero-Support Control", "zero_support_control"),
        ("Shuffled-Support Control", "shuffled_support_control"),
    ):
        summary = payload.get(key)
        lines.extend([f"## {title}", ""])
        if not summary:
            lines.append("- not evaluated because no train alpha passed")
            lines.append("")
            continue
        lines.append(
            f"- paired pp delta: `{fmt((summary.get('paired') or {}).get('delta_mean'))}`, "
            f"p_harm `{fmt((summary.get('paired') or {}).get('p_harm'))}`"
        )
        lines.extend(["", "| dataset | n | mean delta pp | route gap | closure |", "|---|---:|---:|---:|---:|"])
        for row in summary.get("dataset_summary") or []:
            lines.append(
                f"| {row['dataset']} | {row['n_conditions']} | {fmt(row['mean_delta_pp'])} | "
                f"{fmt(row.get('route_gap_pp'))} | {fmt(row.get('route_gap_closed_fraction'))} |"
            )
        lines.append("")
    lines.extend(["## Decision Reasons", ""])
    if decision["reasons"]:
        lines.extend(f"- `{reason}`" for reason in decision["reasons"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Passing this mean-vector gate would not authorize GPU training or held-out query.",
            "- MMD is not evaluated here; a later query-free posthoc MMD/no-harm gate is required.",
            "- Canonical test_multi remains forbidden for selection.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--route-gap-json", type=Path, default=CPU_ROUTE_GAP_JSON)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_payload(args.run_root, args.route_gap_json)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "out_md": str(args.out_md), "out_json": str(args.out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
