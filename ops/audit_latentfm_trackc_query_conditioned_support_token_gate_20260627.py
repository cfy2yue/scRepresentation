#!/usr/bin/env python3
"""CPU gate for a query-conditioned Track C support-token idea.

This gate tests a non-neural, query-conditioned support-token preflight before
touching the LatentFM training stack. Unlike fixed mean/median aggregation or
the previous DeepSets sketch, support residuals are weighted by similarity
between the query's anchor-visible shape and each support condition's
anchor-visible shape.

Only safe-trainselect condition-mean artifacts are used. Held-out Track C query
and canonical multi are not read.
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
RUN_ROOT = (
    ROOT
    / "runs/latentfm_trackc_support_set_task_input_artifacts_20260623/"
    "xverse_support_film_retry1_trainmulti_condition_means"
)
ANCHOR_PATH = RUN_ROOT / "condition_means/trainselect_anchor_train_support_multi_condition_means_ode20.json"
CANDIDATE_PATH = RUN_ROOT / "condition_means/trainselect_candidate_train_support_multi_condition_means_ode20.json"
SAFE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
FULL_V2 = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2.json"
OUT_JSON = ROOT / "reports/latentfm_trackc_query_conditioned_support_token_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_QUERY_CONDITIONED_SUPPORT_TOKEN_GATE_20260627.md"


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


def rows(payload: dict[str, Any], group: str) -> list[dict[str, Any]]:
    return ((payload.get("groups") or {}).get(group) or {}).get("condition_metrics") or []


def paired_rows(anchor: dict[str, Any], candidate: dict[str, Any], group: str) -> list[dict[str, Any]]:
    a = {(str(r["dataset"]), str(r["condition"])): r for r in rows(anchor, group)}
    c = {(str(r["dataset"]), str(r["condition"])): r for r in rows(candidate, group)}
    if set(a) != set(c):
        raise ValueError(f"anchor/candidate mismatch for group {group}")
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


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / den)


def policy_delta(query: dict[str, Any], support: list[dict[str, Any]], policy: str, beta: float) -> np.ndarray:
    if not support:
        return np.zeros_like(query["residual"], dtype=np.float32)
    residuals = np.stack([r["residual"] for r in support]).astype(np.float32)
    if policy == "mean":
        return residuals.mean(axis=0)
    if policy == "positive_cosine":
        scores = np.asarray([max(0.0, cosine(query["anchor_shape"], r["anchor_shape"])) for r in support], dtype=np.float64)
    elif policy == "softmax_cosine":
        scores = np.asarray([cosine(query["anchor_shape"], r["anchor_shape"]) for r in support], dtype=np.float64) * float(beta)
        scores = np.exp(scores - scores.max())
    elif policy == "shared_gene_component":
        qgenes = set(query["pair"])
        scores = np.asarray([len(qgenes & set(r["pair"])) for r in support], dtype=np.float64)
    else:
        raise ValueError(policy)
    if float(scores.sum()) <= 1e-12:
        return residuals.mean(axis=0)
    w = (scores / scores.sum()).astype(np.float32)
    return np.sum(residuals * w[:, None], axis=0)


def score_rows(
    rows_: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    *,
    policy: str,
    beta: float = 1.0,
    loo: bool,
    control: str = "actual",
) -> list[dict[str, Any]]:
    support_map = {}
    for row in rows_:
        support = support_rows_for(row, train_rows, loo=loo)
        if support:
            support_map[(row["dataset"], row["condition"])] = support
    if control == "shuffle":
        keys = sorted(support_map)
        vals = [support_map[k] for k in keys]
        rng = np.random.default_rng(20260627)
        order = rng.permutation(len(vals))
        support_map = {k: vals[int(order[i])] for i, k in enumerate(keys)}
    scored = []
    for row in rows_:
        key = (row["dataset"], row["condition"])
        support = support_map.get(key, [])
        if control in {"zero", "absent"}:
            delta_vec = np.zeros_like(row["residual"], dtype=np.float32)
        else:
            delta_vec = policy_delta(row, support, policy=policy, beta=beta)
        pred = row["pred_anchor"] + delta_vec
        pp = pearson(pred - row["pert"], row["gt"] - row["pert"])
        delta = None if pp is None or row["anchor_pp"] is None else float(pp - row["anchor_pp"])
        scored.append(
            {
                "dataset": row["dataset"],
                "condition": row["condition"],
                "support_count": len(support),
                "delta_vs_anchor": delta,
            }
        )
    supported_keys = set(support_map)
    return [r for r in scored if (r["dataset"], r["condition"]) in supported_keys]


def dataset_deltas(scored: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in scored:
        val = row.get("delta_vs_anchor")
        if val is not None and np.isfinite(float(val)):
            grouped[str(row["dataset"])].append(float(val))
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
        "n_datasets": len(names),
        "n_rows": len(scored),
        "dataset_deltas": ds_delta,
    }


def eval_policy(train_rows: list[dict[str, Any]], support_val_rows: list[dict[str, Any]], policy: str, beta: float) -> dict[str, Any]:
    train_actual = boot(score_rows(train_rows, train_rows, policy=policy, beta=beta, loo=True), seed=101)
    support_actual_rows = score_rows(support_val_rows, train_rows, policy=policy, beta=beta, loo=False)
    support_actual = boot(support_actual_rows, seed=201)
    support_subset = [
        row for row in support_val_rows
        if (row["dataset"], row["condition"]) in {(r["dataset"], r["condition"]) for r in support_actual_rows}
    ]
    zero = boot(score_rows(support_subset, train_rows, policy=policy, beta=beta, loo=False, control="zero"), seed=202)
    shuffled = boot(score_rows(support_subset, train_rows, policy=policy, beta=beta, loo=False, control="shuffle"), seed=203)
    absent = boot(score_rows(support_subset, train_rows, policy=policy, beta=beta, loo=False, control="absent"), seed=204)
    return {
        "policy": policy,
        "beta": beta,
        "train_loo": train_actual,
        "support_val": support_actual,
        "zero_control": zero,
        "shuffle_control": shuffled,
        "absent_control": absent,
    }


def gate(payload: dict[str, Any]) -> list[str]:
    actual = payload["support_val"]
    zero = payload["zero_control"]
    shuffle = payload["shuffle_control"]
    absent = payload["absent_control"]
    reasons = []
    actual_delta = float(actual.get("delta_mean") or -999.0)
    p_harm = actual.get("p_harm")
    ds = actual.get("dataset_deltas") or {}
    if actual_delta < 0.10:
        reasons.append("support_actual_delta_lt_0p10")
    if float(1.0 if p_harm is None else p_harm) > 0.20:
        reasons.append("support_p_harm_gt_0p20")
    if min(ds.values() or [-999.0]) < 0.0:
        reasons.append("support_dataset_min_lt_0")
    for name, control in (("zero", zero), ("shuffle", shuffle), ("absent", absent)):
        cdelta = float(control.get("delta_mean") or 0.0)
        if actual_delta - cdelta < 0.05:
            reasons.append(f"{name}_control_not_0p05_below_actual")
    if abs(float(absent.get("delta_mean") or 0.0)) > 1e-8:
        reasons.append("absent_control_not_exact_anchor")
    return reasons


def main() -> None:
    anchor = load_json(ANCHOR_PATH)
    candidate = load_json(CANDIDATE_PATH)
    if str(anchor.get("split_file")) != str(SAFE_SPLIT) or str(candidate.get("split_file")) != str(SAFE_SPLIT):
        raise RuntimeError("condition-mean artifacts are not from the safe trainselect split")
    if str(FULL_V2) in json.dumps(anchor) or str(FULL_V2) in json.dumps(candidate):
        raise RuntimeError("full v2 query split appeared in support-set artifacts")

    train_rows = paired_rows(anchor, candidate, "train_multi")
    support_val_rows = paired_rows(anchor, candidate, "support_val_multi")
    candidates = []
    for policy in ("mean", "shared_gene_component", "positive_cosine"):
        candidates.append(eval_policy(train_rows, support_val_rows, policy, beta=1.0))
    for beta in (1.0, 3.0, 5.0):
        candidates.append(eval_policy(train_rows, support_val_rows, "softmax_cosine", beta=beta))
    candidates = sorted(
        candidates,
        key=lambda x: (
            float(x["train_loo"].get("delta_mean") or -999.0),
            float(x["support_val"].get("delta_mean") or -999.0),
        ),
        reverse=True,
    )
    selected = candidates[0]
    reasons = gate(selected)
    status = "trackc_query_conditioned_support_token_gate_fail_no_gpu"
    if not reasons:
        status = "trackc_query_conditioned_support_token_gate_pass_unit_impl_next_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "safe_trainselect_split": str(SAFE_SPLIT),
            "full_v2_query_used": False,
            "canonical_multi_selection_used": False,
            "latentfm_training": False,
            "inference": False,
            "gpu": False,
            "model_class": "cpu_only_query_conditioned_attention_policy_not_integrated_into_latentfm",
        },
        "inputs": {"anchor": str(ANCHOR_PATH), "candidate": str(CANDIDATE_PATH)},
        "rows": {"train_multi": len(train_rows), "support_val_multi": len(support_val_rows)},
        "all_candidates": candidates,
        "selected": selected,
        "decision_reasons": reasons,
        "next_action": (
            "implement default-off query-conditioned support-token source/unit tests before any GPU"
            if not reasons
            else "close current query-conditioned support-token preflight unless a materially new support source appears"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def fmt(v: Any) -> str:
        if v is None:
            return "NA"
        if isinstance(v, float):
            return f"{v:+.6f}" if math.isfinite(v) else str(v)
        return str(v)

    lines = [
        "# Track C Query-Conditioned Support-Token Gate",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "CPU-only query-conditioned support policy over safe-trainselect condition-mean residuals. No LatentFM training, no inference, no held-out query, no canonical multi selection, and no GPU.",
        "",
        "## Candidate Policies",
        "",
        "| policy | beta | train delta | support delta | support p_harm | support dataset min | zero | shuffle | absent |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cand in candidates:
        ds = cand["support_val"].get("dataset_deltas") or {}
        lines.append(
            f"| `{cand['policy']}` | {cand['beta']:.1f} | "
            f"{fmt(cand['train_loo'].get('delta_mean'))} | {fmt(cand['support_val'].get('delta_mean'))} | "
            f"{cand['support_val'].get('p_harm')} | {fmt(min(ds.values()) if ds else None)} | "
            f"{fmt(cand['zero_control'].get('delta_mean'))} | {fmt(cand['shuffle_control'].get('delta_mean'))} | "
            f"{fmt(cand['absent_control'].get('delta_mean'))} |"
        )
    lines.extend(
        [
            "",
            "## Selected Policy",
            "",
            f"- policy: `{selected['policy']}`",
            f"- beta: `{selected['beta']}`",
            f"- support dataset deltas: `{selected['support_val'].get('dataset_deltas')}`",
            "",
            "## Decision Reasons",
            "",
        ]
    )
    lines.extend(f"- `{r}`" for r in reasons) if reasons else lines.append("- none")
    lines.extend(["", "## Outputs", "", f"- JSON: `{OUT_JSON}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "md": str(OUT_MD), "json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
