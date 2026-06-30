#!/usr/bin/env python3
"""CPU policy sweep for Track C support-set aggregation.

This is a query-free, CPU-only gate over safe-trainselect condition-mean
artifacts. It asks whether a materially different permutation-invariant
support-set source policy can beat the fixed shared-gene mean residual before
we implement another GPU launcher.

Policy/alpha selection uses train_multi leave-one-condition-out only.
support_val_multi is then evaluated once with zero and shuffled controls. This
script does not train, run inference, use GPU, use canonical multi selection,
or read held-out Track C query.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

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
OUT_JSON = ROOT / "reports/latentfm_trackc_support_set_policy_sweep_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_SUPPORT_SET_POLICY_SWEEP_20260627.md"
ALPHAS = (0.10, 0.25, 0.50, 0.75, 1.00)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pair(cond: str) -> tuple[str, str] | None:
    parts = [p.strip().upper() for p in str(cond).split("+") if p.strip()]
    return (parts[0], parts[1]) if len(parts) == 2 else None


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
    out: list[dict[str, Any]] = []
    for key in sorted(a):
        genes = pair(key[1])
        if genes is None:
            continue
        ar = a[key]
        cr = c[key]
        pred_anchor = vec(ar, "pred_mean")
        pred_candidate = vec(cr, "pred_mean")
        gt = vec(ar, "gt_mean")
        pert = vec(ar, "pert_mean")
        out.append(
            {
                "dataset": key[0],
                "condition": key[1],
                "pair": genes,
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


def support_rows_for(query: dict[str, Any], train: list[dict[str, Any]], *, loo: bool) -> list[dict[str, Any]]:
    qgenes = set(query["pair"])
    out = []
    for row in train:
        if row["dataset"] != query["dataset"]:
            continue
        if loo and row["condition"] == query["condition"]:
            continue
        if qgenes & set(row["pair"]):
            out.append(row)
    return out


def _stack(rows_: list[dict[str, Any]]) -> np.ndarray:
    return np.stack([row["residual"] for row in rows_], axis=0).astype(np.float32)


def policy_mean(query: dict[str, Any], support: list[dict[str, Any]]) -> np.ndarray | None:
    if not support:
        return None
    return _stack(support).mean(axis=0)


def policy_median(query: dict[str, Any], support: list[dict[str, Any]]) -> np.ndarray | None:
    if not support:
        return None
    return np.median(_stack(support), axis=0).astype(np.float32)


def policy_overlap_weighted(query: dict[str, Any], support: list[dict[str, Any]]) -> np.ndarray | None:
    if not support:
        return None
    qgenes = set(query["pair"])
    vals = _stack(support)
    weights = np.asarray([max(1, len(qgenes & set(row["pair"]))) for row in support], dtype=np.float32)
    return np.average(vals, axis=0, weights=weights).astype(np.float32)


def policy_gene_component_mean(query: dict[str, Any], support: list[dict[str, Any]]) -> np.ndarray | None:
    if not support:
        return None
    parts = []
    for gene in query["pair"]:
        matched = [row for row in support if gene in set(row["pair"])]
        if matched:
            parts.append(_stack(matched).mean(axis=0))
    if not parts:
        return None
    return np.stack(parts, axis=0).mean(axis=0).astype(np.float32)


def policy_norm_clipped_mean(query: dict[str, Any], support: list[dict[str, Any]]) -> np.ndarray | None:
    if not support:
        return None
    vals = _stack(support)
    norms = np.linalg.norm(vals, axis=1)
    finite = norms[np.isfinite(norms)]
    if finite.size == 0:
        return None
    cap = float(np.median(finite))
    if cap <= 1e-12:
        return vals.mean(axis=0)
    scale = np.minimum(1.0, cap / np.maximum(norms, 1e-12)).astype(np.float32)
    return (vals * scale[:, None]).mean(axis=0).astype(np.float32)


def shrink_policy(base: Callable[[dict[str, Any], list[dict[str, Any]]], np.ndarray | None], lam: float):
    def wrapped(query: dict[str, Any], support: list[dict[str, Any]]) -> np.ndarray | None:
        token = base(query, support)
        if token is None:
            return None
        return (token * (len(support) / (len(support) + lam))).astype(np.float32)

    return wrapped


POLICIES: dict[str, Callable[[dict[str, Any], list[dict[str, Any]]], np.ndarray | None]] = {
    "mean": policy_mean,
    "median": policy_median,
    "overlap_weighted_mean": policy_overlap_weighted,
    "gene_component_mean": policy_gene_component_mean,
    "norm_clipped_mean": policy_norm_clipped_mean,
    "mean_shrink_lambda1": shrink_policy(policy_mean, 1.0),
    "mean_shrink_lambda2": shrink_policy(policy_mean, 2.0),
    "mean_shrink_lambda4": shrink_policy(policy_mean, 4.0),
}


def score_rows(
    train: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    *,
    policy_name: str,
    alpha: float,
    loo: bool,
    shuffle: bool = False,
) -> list[dict[str, Any]]:
    policy = POLICIES[policy_name]
    token_map: dict[tuple[str, str], np.ndarray | None] = {}
    count_map: dict[tuple[str, str], int] = {}
    for query in queries:
        key = (str(query["dataset"]), str(query["condition"]))
        support = support_rows_for(query, train, loo=loo)
        token_map[key] = policy(query, support)
        count_map[key] = len(support)
    if shuffle:
        rng = np.random.default_rng(20260627)
        keys = sorted(token_map)
        vals = [token_map[k] for k in keys]
        order = rng.permutation(len(vals))
        token_map = {k: vals[int(order[i])] for i, k in enumerate(keys)}
    scored = []
    for row in queries:
        key = (str(row["dataset"]), str(row["condition"]))
        token = token_map[key]
        if token is None:
            token = np.zeros_like(row["residual"], dtype=np.float32)
        pred = row["pred_anchor"] + float(alpha) * token
        task_pp = pearson(pred - row["pert"], row["gt"] - row["pert"])
        delta = None if task_pp is None or row["anchor_pp"] is None else float(task_pp - row["anchor_pp"])
        scored.append(
            {
                "dataset": row["dataset"],
                "condition": row["condition"],
                "support_count": count_map[key],
                "anchor_pp": row["anchor_pp"],
                "task_pp": task_pp,
                "delta_vs_anchor": delta,
            }
        )
    return scored


def dataset_deltas(scored: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in scored:
        delta = row.get("delta_vs_anchor")
        if delta is not None and np.isfinite(float(delta)):
            grouped[str(row["dataset"])].append(float(delta))
    return {ds: mean(vals) for ds, vals in grouped.items() if vals}


def boot(scored: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    ds_delta = dataset_deltas(scored)
    if not ds_delta:
        return {"delta_mean": None, "ci95": [None, None], "p_harm": None, "n_datasets": 0, "n_rows": len(scored)}
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


def summarize(scored_all: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    scored = [row for row in scored_all if int(row["support_count"]) > 0]
    out = boot(scored, seed=seed)
    out["supported_rows"] = len(scored)
    out["all_rows"] = len(scored_all)
    out["supported_fraction"] = len(scored) / max(1, len(scored_all))
    out["min_support_count"] = min((int(row["support_count"]) for row in scored), default=0)
    return out


def train_gate(summary: dict[str, Any]) -> bool:
    ds = summary.get("dataset_deltas") or {}
    p_harm = summary.get("p_harm")
    return (
        summary.get("delta_mean") is not None
        and float(summary["delta_mean"]) >= 0.02
        and float(1.0 if p_harm is None else p_harm) <= 0.20
        and min(ds.values() or [-999.0]) >= -0.01
        and int(summary.get("min_support_count") or 0) >= 1
    )


def support_gate(actual: dict[str, Any], zero: dict[str, Any], shuffled: dict[str, Any]) -> tuple[bool, list[str]]:
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
        raise RuntimeError("full v2 query split appeared in support-set artifacts")
    train = paired_rows(anchor, candidate, "train_multi")
    support = paired_rows(anchor, candidate, "support_val_multi")

    train_summaries = []
    for pidx, policy_name in enumerate(POLICIES):
        for alpha in ALPHAS:
            scored = score_rows(train, train, policy_name=policy_name, alpha=alpha, loo=True)
            summary = summarize(scored, seed=10000 + pidx * 100 + int(alpha * 100))
            summary.update({"policy": policy_name, "alpha": alpha, "train_gate_pass": train_gate(summary)})
            train_summaries.append(summary)

    train_passes = [row for row in train_summaries if row["train_gate_pass"]]
    selected = None
    if train_passes:
        selected = max(
            train_passes,
            key=lambda row: (
                min((row.get("dataset_deltas") or {}).values() or [-999.0]),
                float(row.get("delta_mean") or -999.0),
                -float(row["alpha"]),
            ),
        )

    actual_summary = zero_summary = shuffle_summary = None
    support_reasons = ["no_policy_alpha_passed_train_loo_gate"]
    if selected is not None:
        policy_name = str(selected["policy"])
        alpha = float(selected["alpha"])
        actual_all = score_rows(train, support, policy_name=policy_name, alpha=alpha, loo=False)
        supported_keys = {(row["dataset"], row["condition"]) for row in actual_all if int(row["support_count"]) > 0}
        actual_rows = [row for row in actual_all if (row["dataset"], row["condition"]) in supported_keys]
        zero_rows = [
            row
            for row in score_rows(train, support, policy_name=policy_name, alpha=0.0, loo=False)
            if (row["dataset"], row["condition"]) in supported_keys
        ]
        shuffle_rows = [
            row
            for row in score_rows(train, support, policy_name=policy_name, alpha=alpha, loo=False, shuffle=True)
            if (row["dataset"], row["condition"]) in supported_keys
        ]
        actual_summary = summarize(actual_rows, seed=20000)
        actual_summary.update({"policy": policy_name, "alpha": alpha})
        zero_summary = summarize(zero_rows, seed=20001)
        zero_summary.update({"policy": policy_name, "alpha": 0.0})
        shuffle_summary = summarize(shuffle_rows, seed=20002)
        shuffle_summary.update({"policy": policy_name, "alpha": alpha})
        _, support_reasons = support_gate(actual_summary, zero_summary, shuffle_summary)

    status = (
        "trackc_support_set_policy_sweep_pass_source_impl_next_no_gpu"
        if selected is not None and not support_reasons
        else "trackc_support_set_policy_sweep_fail_no_gpu"
    )
    if status.endswith("fail_no_gpu") and actual_summary is not None:
        actual_delta = float(actual_summary.get("delta_mean") or -999.0)
        if actual_delta >= 0.04 and support_reasons == ["support_p_harm_gt_0p20"]:
            status = "trackc_support_set_policy_sweep_near_signal_no_gpu"

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
            "policy_selected_on": "train_multi_leave_one_condition_out_only",
            "support_val_role": "one_time_validation_with_controls",
        },
        "inputs": {"anchor": str(ANCHOR_PATH), "candidate": str(CANDIDATE_PATH)},
        "rows": {"train_multi": len(train), "support_val_multi": len(support)},
        "policies": sorted(POLICIES),
        "alphas": list(ALPHAS),
        "train_summaries": train_summaries,
        "selected_train_policy": selected,
        "support_val_summary": actual_summary,
        "zero_support_control": zero_summary,
        "shuffled_support_control": shuffle_summary,
        "decision_reasons": support_reasons,
        "next_action": (
            "implement selected policy source and rerun source/control gate before GPU"
            if status == "trackc_support_set_policy_sweep_pass_source_impl_next_no_gpu"
            else "do not launch another shared-gene aggregation GPU smoke; require materially new set encoder"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    top = sorted(
        train_summaries,
        key=lambda row: (
            bool(row.get("train_gate_pass")),
            min((row.get("dataset_deltas") or {}).values() or [-999.0]),
            float(row.get("delta_mean") or -999.0),
        ),
        reverse=True,
    )[:10]
    lines = [
        "# Track C Support-Set Aggregation Policy Sweep",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "CPU-only over safe-trainselect condition-mean artifacts. Policy/alpha selection uses train_multi leave-one-condition-out only; support-val is evaluated once with zero/shuffle controls. No training, inference, canonical multi selection, full v2 query, or GPU.",
        "",
        "## Top Train-LOO Policies",
        "",
        "| policy | alpha | delta | CI95 | p_harm | min dataset | rows | pass |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in top:
        ds_vals = list((row.get("dataset_deltas") or {}).values())
        min_ds = min(ds_vals) if ds_vals else None
        lines.append(
            f"| `{row['policy']}` | {float(row['alpha']):.2f} | {row.get('delta_mean')} | "
            f"{row.get('ci95')} | {row.get('p_harm')} | {min_ds} | "
            f"{row.get('supported_rows')}/{row.get('all_rows')} | `{row.get('train_gate_pass')}` |"
        )
    lines.extend(["", "## Selected Policy", ""])
    if selected is None:
        lines.append("- none")
    else:
        lines.append(f"- policy: `{selected['policy']}`")
        lines.append(f"- alpha: `{selected['alpha']}`")
        lines.append(f"- train delta: `{selected.get('delta_mean')}`")
        lines.append(f"- train dataset deltas: `{selected.get('dataset_deltas')}`")
    lines.extend(["", "## Support-Val Check", ""])
    if actual_summary is None:
        lines.append("- not evaluated because no train policy passed")
    else:
        lines.append(f"- actual delta: `{actual_summary.get('delta_mean')}`")
        lines.append(f"- actual CI95: `{actual_summary.get('ci95')}`")
        lines.append(f"- actual p_harm: `{actual_summary.get('p_harm')}`")
        lines.append(f"- actual dataset deltas: `{actual_summary.get('dataset_deltas')}`")
        lines.append(f"- zero delta: `{zero_summary.get('delta_mean') if zero_summary else None}`")
        lines.append(f"- shuffled delta: `{shuffle_summary.get('delta_mean') if shuffle_summary else None}`")
    lines.extend(["", "## Decision Reasons", ""])
    lines.extend(f"- `{reason}`" for reason in support_reasons)
    lines.extend(["", "## Outputs", "", f"- JSON: `{OUT_JSON}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "md": str(OUT_MD), "json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
