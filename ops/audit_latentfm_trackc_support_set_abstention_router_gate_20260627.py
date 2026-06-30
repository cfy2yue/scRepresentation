#!/usr/bin/env python3
"""CPU gate for a Track C support-set abstention/no-harm router.

This tests whether a support-set source can be used only when support evidence
looks safe, with exact anchor no-op otherwise. It uses safe-trainselect
condition-mean artifacts only. It does not train LatentFM, run inference, read
held-out Track C query, read canonical multi for selection, or use GPU.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_trackc_support_set_task_input_artifacts_20260623/xverse_support_film_retry1_trainmulti_condition_means"
ANCHOR_PATH = RUN_ROOT / "condition_means/trainselect_anchor_train_support_multi_condition_means_ode20.json"
CANDIDATE_PATH = RUN_ROOT / "condition_means/trainselect_candidate_train_support_multi_condition_means_ode20.json"
SAFE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
FULL_V2 = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2.json"
OUT_JSON = ROOT / "reports/latentfm_trackc_support_set_abstention_router_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_SUPPORT_SET_ABSTENTION_ROUTER_GATE_20260627.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pair(cond: str) -> tuple[str, str] | None:
    parts = [p.strip().upper() for p in str(cond).split("+") if p.strip()]
    return (parts[0], parts[1]) if len(parts) == 2 else None


def vec(row: dict[str, Any], key: str) -> np.ndarray:
    arr = np.asarray(row[key], dtype=np.float32)
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise ValueError(f"bad {key} for {row.get('dataset')}:{row.get('condition')}")
    return arr


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return None
    x = x[mask] - x[mask].mean()
    y = y[mask] - y[mask].mean()
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    if den <= 1e-12:
        return None
    return float(np.dot(x, y) / den)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / den)


def rows(payload: dict[str, Any], group: str) -> list[dict[str, Any]]:
    return ((payload.get("groups") or {}).get(group) or {}).get("condition_metrics") or []


def paired_rows(anchor: dict[str, Any], candidate: dict[str, Any], group: str) -> list[dict[str, Any]]:
    a = {(str(r["dataset"]), str(r["condition"])): r for r in rows(anchor, group)}
    c = {(str(r["dataset"]), str(r["condition"])): r for r in rows(candidate, group)}
    if set(a) != set(c):
        raise ValueError(f"anchor/candidate mismatch for {group}")
    out = []
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
        anchor_shape = pred_anchor - pert
        out.append(
            {
                "dataset": key[0],
                "condition": key[1],
                "pair": genes,
                "pred_anchor": pred_anchor,
                "gt": gt,
                "pert": pert,
                "anchor_shape": anchor_shape,
                "residual": pred_candidate - pred_anchor,
                "anchor_pp": pearson(anchor_shape, gt - pert),
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


def policy_delta(query: dict[str, Any], support: list[dict[str, Any]], policy: str, beta: float) -> np.ndarray:
    if not support:
        return np.zeros_like(query["residual"], dtype=np.float32)
    residuals = np.stack([r["residual"] for r in support]).astype(np.float32)
    if policy == "mean":
        return residuals.mean(axis=0)
    if policy == "shared_gene_component":
        scores = np.asarray([len(set(query["pair"]) & set(r["pair"])) for r in support], dtype=np.float64)
    elif policy == "positive_cosine":
        scores = np.asarray([max(0.0, cosine(query["anchor_shape"], r["anchor_shape"])) for r in support], dtype=np.float64)
    elif policy == "softmax_cosine":
        scores = np.asarray([cosine(query["anchor_shape"], r["anchor_shape"]) for r in support], dtype=np.float64) * float(beta)
        scores = np.exp(scores - scores.max())
    else:
        raise ValueError(policy)
    if float(scores.sum()) <= 1e-12:
        return np.zeros_like(query["residual"], dtype=np.float32)
    weights = (scores / scores.sum()).astype(np.float32)
    return np.sum(residuals * weights[:, None], axis=0)


def support_confidence(query: dict[str, Any], support: list[dict[str, Any]]) -> float:
    if not support:
        return -1.0
    return max(cosine(query["anchor_shape"], row["anchor_shape"]) for row in support)


def score_rows(
    rows_: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    *,
    policy: str,
    beta: float,
    min_support: int,
    min_confidence: float,
    loo: bool,
    control: str,
) -> list[dict[str, Any]]:
    base_support = {}
    for row in rows_:
        support = support_rows_for(row, train_rows, loo=loo)
        if support:
            base_support[(row["dataset"], row["condition"])] = support
    support_map = dict(base_support)
    if control == "shuffle":
        keys = sorted(support_map)
        vals = [support_map[k] for k in keys]
        rng = np.random.default_rng(20260627)
        order = rng.permutation(len(vals))
        support_map = {k: vals[int(order[i])] for i, k in enumerate(keys)}
    scored = []
    for row in rows_:
        key = (row["dataset"], row["condition"])
        if key not in base_support:
            continue
        support = support_map.get(key, [])
        confidence = support_confidence(row, support)
        enabled = len(support) >= min_support and confidence >= min_confidence and control not in {"zero", "absent"}
        delta_vec = policy_delta(row, support, policy, beta) if enabled else np.zeros_like(row["residual"], dtype=np.float32)
        pred = row["pred_anchor"] + delta_vec
        task_pp = pearson(pred - row["pert"], row["gt"] - row["pert"])
        delta = None if task_pp is None or row["anchor_pp"] is None else float(task_pp - row["anchor_pp"])
        scored.append(
            {
                "dataset": row["dataset"],
                "condition": row["condition"],
                "enabled": enabled,
                "support_count": len(support),
                "confidence": confidence,
                "delta_vs_anchor": delta,
            }
        )
    return scored


def dataset_deltas(scored: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in scored:
        val = row.get("delta_vs_anchor")
        if val is not None and np.isfinite(float(val)):
            grouped[str(row["dataset"])].append(float(val))
    return {ds: mean(vals) for ds, vals in grouped.items() if vals}


def boot(scored: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    ds_delta = dataset_deltas(scored)
    enabled = sum(1 for row in scored if row.get("enabled"))
    if not ds_delta:
        return {"delta_mean": None, "ci95": [None, None], "p_harm": None, "n_datasets": 0, "n_rows": len(scored), "enabled_rows": enabled}
    names = sorted(ds_delta)
    rng = np.random.default_rng(seed)
    vals = [mean(ds_delta[str(ds)] for ds in rng.choice(names, size=len(names), replace=True)) for _ in range(2000)]
    arr = np.asarray(vals)
    return {
        "delta_mean": float(mean(ds_delta.values())),
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_harm": float(np.mean(arr < 0.0)),
        "n_datasets": len(names),
        "n_rows": len(scored),
        "enabled_rows": enabled,
        "dataset_deltas": ds_delta,
    }


def eval_candidate(train_rows: list[dict[str, Any]], support_val_rows: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    kwargs = {
        "policy": spec["policy"],
        "beta": spec["beta"],
        "min_support": spec["min_support"],
        "min_confidence": spec["min_confidence"],
    }
    train = boot(score_rows(train_rows, train_rows, loo=True, control="actual", **kwargs), seed=101)
    actual_rows = score_rows(support_val_rows, train_rows, loo=False, control="actual", **kwargs)
    actual = boot(actual_rows, seed=201)
    zero = boot(score_rows(support_val_rows, train_rows, loo=False, control="zero", **kwargs), seed=202)
    shuffled = boot(score_rows(support_val_rows, train_rows, loo=False, control="shuffle", **kwargs), seed=203)
    absent = boot(score_rows(support_val_rows, train_rows, loo=False, control="absent", **kwargs), seed=204)
    return {**spec, "train_loo": train, "support_val": actual, "zero_control": zero, "shuffle_control": shuffled, "absent_control": absent}


def gate(selected: dict[str, Any]) -> list[str]:
    actual = selected["support_val"]
    zero = selected["zero_control"]
    shuffle = selected["shuffle_control"]
    absent = selected["absent_control"]
    reasons = []
    actual_delta = float(actual.get("delta_mean") or 0.0)
    if int(actual.get("enabled_rows") or 0) < 6:
        reasons.append("support_enabled_rows_lt_6")
    if actual_delta < 0.04:
        reasons.append("support_actual_delta_lt_0p04")
    if float(actual.get("p_harm") if actual.get("p_harm") is not None else 1.0) > 0.20:
        reasons.append("support_p_harm_gt_0p20")
    if min((actual.get("dataset_deltas") or {}).values() or [-999.0]) < 0.0:
        reasons.append("support_dataset_min_lt_0")
    if actual_delta - float(shuffle.get("delta_mean") or 0.0) < 0.02:
        reasons.append("shuffle_control_not_0p02_below_actual")
    if abs(float(zero.get("delta_mean") or 0.0)) > 1e-8:
        reasons.append("zero_control_not_exact_anchor")
    if abs(float(absent.get("delta_mean") or 0.0)) > 1e-8:
        reasons.append("absent_control_not_exact_anchor")
    return reasons


def fmt(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v):+.6f}"
    except (TypeError, ValueError):
        return str(v)


def main() -> int:
    anchor = load_json(ANCHOR_PATH)
    candidate = load_json(CANDIDATE_PATH)
    if str(anchor.get("split_file")) != str(SAFE_SPLIT) or str(candidate.get("split_file")) != str(SAFE_SPLIT):
        raise RuntimeError("condition-mean artifacts are not from the safe trainselect split")
    if str(FULL_V2) in json.dumps(anchor) or str(FULL_V2) in json.dumps(candidate):
        raise RuntimeError("full v2 query split appeared in support-set artifacts")

    train_rows = paired_rows(anchor, candidate, "train_multi")
    support_val_rows = paired_rows(anchor, candidate, "support_val_multi")
    specs = []
    for policy in ("mean", "shared_gene_component", "positive_cosine", "softmax_cosine"):
        betas = (1.0, 3.0) if policy == "softmax_cosine" else (1.0,)
        for beta in betas:
            for min_support in (1, 2, 3):
                for min_confidence in (-1.0, 0.0, 0.25, 0.5):
                    specs.append({"policy": policy, "beta": beta, "min_support": min_support, "min_confidence": min_confidence})
    candidates = [eval_candidate(train_rows, support_val_rows, spec) for spec in specs]
    candidates.sort(
        key=lambda row: (
            float(row["support_val"].get("delta_mean") or -999.0),
            -float(row["support_val"].get("p_harm") if row["support_val"].get("p_harm") is not None else 1.0),
            int(row["support_val"].get("enabled_rows") or 0),
        ),
        reverse=True,
    )
    selected = candidates[0]
    reasons = gate(selected)
    status = "trackc_support_set_abstention_router_gate_pass_launcher_design_next_no_gpu" if not reasons else "trackc_support_set_abstention_router_gate_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "decision_reasons": reasons,
        "inputs": {"anchor": str(ANCHOR_PATH), "candidate": str(CANDIDATE_PATH), "split": str(SAFE_SPLIT)},
        "rows": {"train_multi": len(train_rows), "support_val_multi": len(support_val_rows)},
        "selected": selected,
        "top_candidates": candidates[:12],
        "boundary": {
            "safe_trainselect_only": True,
            "full_trackc_query_used": False,
            "canonical_multi_selection_used": False,
            "latentfm_training": False,
            "gpu": False,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Track C Support-Set Abstention Router Gate 2026-06-27",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "CPU-only abstention/no-harm router over safe-trainselect condition-mean residuals. No LatentFM training, no inference, no held-out Track C query, no canonical multi selection, and no GPU.",
        "",
        "## Selected",
        "",
        f"- policy: `{selected['policy']}`",
        f"- beta: `{selected['beta']}`",
        f"- min support: `{selected['min_support']}`",
        f"- min confidence: `{selected['min_confidence']}`",
        f"- support dataset deltas: `{selected['support_val'].get('dataset_deltas')}`",
        "",
        "## Top Candidates",
        "",
        "| policy | beta | min support | min conf | train delta | support delta | enabled | p_harm | dataset min | shuffle | absent |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cand in candidates[:12]:
        ds = cand["support_val"].get("dataset_deltas") or {}
        lines.append(
            f"| `{cand['policy']}` | {cand['beta']:.1f} | {cand['min_support']} | {cand['min_confidence']:.2f} | "
            f"{fmt(cand['train_loo'].get('delta_mean'))} | {fmt(cand['support_val'].get('delta_mean'))} | "
            f"{cand['support_val'].get('enabled_rows')} | {cand['support_val'].get('p_harm')} | "
            f"{fmt(min(ds.values()) if ds else None)} | {fmt(cand['shuffle_control'].get('delta_mean'))} | "
            f"{fmt(cand['absent_control'].get('delta_mean'))} |"
        )
    lines += ["", "## Decision Reasons", ""]
    lines.extend(f"- `{reason}`" for reason in reasons) if reasons else lines.append("- none")
    lines += [
        "",
        "## Decision",
        "",
        "A pass would authorize only external audit plus launcher design for a default-off support router. A fail closes this CPU abstention idea as a GPU unlock.",
        "",
        f"- JSON: `{OUT_JSON}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "reasons": reasons, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
