#!/usr/bin/env python3
"""CPU-only DeepSets sketch gate for Track C support-set sources.

This gate tests a materially different support-set source before touching the
LatentFM training stack. Unlike fixed aggregation policies, it learns a tiny
permutation-invariant map from the same-dataset shared-gene support residual
set to a query residual. It uses only safe-trainselect condition-mean artifacts.

This script does not train LatentFM, run inference, use GPU, read canonical
multi for selection, or read held-out Track C query. Passing this gate would
only authorize a default-off source/unit implementation gate, not a GPU smoke.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from torch import nn


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
OUT_JSON = ROOT / "reports/latentfm_trackc_support_set_deepset_sketch_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_SUPPORT_SET_DEEPSET_SKETCH_GATE_20260627.md"


class DeepSetSketch(nn.Module):
    def __init__(self, dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.rho = nn.Linear(hidden, dim)

    def forward(self, support: torch.Tensor) -> torch.Tensor:
        if support.ndim != 2:
            raise ValueError("support must be shaped (n_support, dim)")
        pooled = self.phi(support).mean(dim=0)
        return self.rho(pooled)


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


def support_tensor(rows_: list[dict[str, Any]], residual_mean: np.ndarray, residual_std: np.ndarray) -> torch.Tensor:
    arr = np.stack([row["residual"] for row in rows_], axis=0)
    return torch.from_numpy(((arr - residual_mean) / residual_std).astype(np.float32))


def target_tensor(row: dict[str, Any], residual_mean: np.ndarray, residual_std: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(((row["residual"] - residual_mean) / residual_std).astype(np.float32))


def train_sketch(train_rows: list[dict[str, Any]], residual_mean: np.ndarray, residual_std: np.ndarray) -> tuple[DeepSetSketch, dict[str, Any]]:
    torch.manual_seed(20260627)
    dim = int(train_rows[0]["residual"].shape[0])
    model = DeepSetSketch(dim=dim, hidden=32)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    examples = []
    for row in train_rows:
        support = support_rows_for(row, train_rows, loo=True)
        if support:
            examples.append((row, support_tensor(support, residual_mean, residual_std), target_tensor(row, residual_mean, residual_std)))
    if not examples:
        raise RuntimeError("no train examples with support rows")
    losses = []
    for epoch in range(800):
        total = torch.zeros(())
        for _row, support, target in examples:
            pred = model(support)
            total = total + torch.mean((pred - target) ** 2)
        loss = total / len(examples)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if epoch in {0, 99, 199, 399, 799}:
            losses.append({"epoch": epoch + 1, "loss": float(loss.detach().item())})
    return model.eval(), {"n_examples": len(examples), "loss_trace": losses}


@torch.no_grad()
def predict_delta(
    model: DeepSetSketch,
    support: list[dict[str, Any]],
    residual_mean: np.ndarray,
    residual_std: np.ndarray,
) -> np.ndarray | None:
    if not support:
        return None
    pred_norm = model(support_tensor(support, residual_mean, residual_std)).cpu().numpy()
    return (pred_norm * residual_std + residual_mean).astype(np.float32)


def score_rows(
    rows_: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    model: DeepSetSketch,
    residual_mean: np.ndarray,
    residual_std: np.ndarray,
    *,
    mode: str,
    loo: bool,
) -> list[dict[str, Any]]:
    supported = []
    actual_supports: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows_:
        support = support_rows_for(row, train_rows, loo=loo)
        if support:
            supported.append(row)
            actual_supports[(row["dataset"], row["condition"])] = support
    if mode == "shuffle":
        keys = sorted(actual_supports)
        vals = [actual_supports[k] for k in keys]
        rng = np.random.default_rng(20260627)
        order = rng.permutation(len(vals))
        actual_supports = {k: vals[int(order[i])] for i, k in enumerate(keys)}

    scored = []
    for row in rows_:
        key = (row["dataset"], row["condition"])
        support = actual_supports.get(key, [])
        if mode in {"zero", "absent"}:
            delta_vec = np.zeros_like(row["residual"], dtype=np.float32)
        elif mode in {"actual", "shuffle"}:
            pred = predict_delta(model, support, residual_mean, residual_std)
            delta_vec = np.zeros_like(row["residual"], dtype=np.float32) if pred is None else pred
        else:
            raise ValueError(f"unknown mode: {mode}")
        pred_mean = row["pred_anchor"] + delta_vec
        task_pp = pearson(pred_mean - row["pert"], row["gt"] - row["pert"])
        delta = None if task_pp is None or row["anchor_pp"] is None else float(task_pp - row["anchor_pp"])
        scored.append(
            {
                "dataset": row["dataset"],
                "condition": row["condition"],
                "support_count": len(support),
                "task_pp": task_pp,
                "delta_vs_anchor": delta,
            }
        )
    return [row for row in scored if int(row["support_count"]) > 0 or mode in {"zero", "absent"} and (row["dataset"], row["condition"]) in {(r["dataset"], r["condition"]) for r in supported}]


def gate_support(actual: dict[str, Any], zero: dict[str, Any], shuffled: dict[str, Any], absent: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    actual_delta = float(actual.get("delta_mean") or -999.0)
    p_harm = actual.get("p_harm")
    ds = actual.get("dataset_deltas") or {}
    zero_delta = float(zero.get("delta_mean") or 0.0)
    shuffle_delta = float(shuffled.get("delta_mean") or 0.0)
    absent_delta = float(absent.get("delta_mean") or 0.0)
    if actual_delta < 0.04:
        reasons.append("support_actual_delta_lt_0p04")
    if float(1.0 if p_harm is None else p_harm) > 0.20:
        reasons.append("support_p_harm_gt_0p20")
    if min(ds.values() or [-999.0]) < -0.01:
        reasons.append("support_dataset_min_lt_minus_0p01")
    for name, control_delta in (("zero", zero_delta), ("shuffle", shuffle_delta), ("absent", absent_delta)):
        if actual_delta - control_delta < 0.02:
            reasons.append(f"{name}_control_not_0p02_below_actual")
    if abs(absent_delta) > 1e-8:
        reasons.append("absent_control_not_exact_anchor")
    return not reasons, reasons


def main() -> None:
    torch.set_num_threads(1)
    anchor = load_json(ANCHOR_PATH)
    candidate = load_json(CANDIDATE_PATH)
    if str(anchor.get("split_file")) != str(SAFE_SPLIT) or str(candidate.get("split_file")) != str(SAFE_SPLIT):
        raise RuntimeError("condition-mean artifacts are not from the safe trainselect split")
    if str(FULL_V2) in json.dumps(anchor) or str(FULL_V2) in json.dumps(candidate):
        raise RuntimeError("full v2 query split appeared in support-set artifacts")
    train_rows = paired_rows(anchor, candidate, "train_multi")
    support_val_rows = paired_rows(anchor, candidate, "support_val_multi")
    residuals = np.stack([row["residual"] for row in train_rows], axis=0).astype(np.float32)
    residual_mean = residuals.mean(axis=0)
    residual_std = residuals.std(axis=0)
    residual_std = np.where(residual_std < 1e-6, 1.0, residual_std).astype(np.float32)

    model, train_info = train_sketch(train_rows, residual_mean, residual_std)
    train_actual = boot(score_rows(train_rows, train_rows, model, residual_mean, residual_std, mode="actual", loo=True), seed=101)
    support_actual_rows = score_rows(support_val_rows, train_rows, model, residual_mean, residual_std, mode="actual", loo=False)
    supported_keys = {(row["dataset"], row["condition"]) for row in support_actual_rows}
    support_subset = [row for row in support_val_rows if (row["dataset"], row["condition"]) in supported_keys]
    actual = boot(support_actual_rows, seed=201)
    zero = boot(score_rows(support_subset, train_rows, model, residual_mean, residual_std, mode="zero", loo=False), seed=202)
    shuffled = boot(score_rows(support_subset, train_rows, model, residual_mean, residual_std, mode="shuffle", loo=False), seed=203)
    absent = boot(score_rows(support_subset, train_rows, model, residual_mean, residual_std, mode="absent", loo=False), seed=204)
    gate_pass, reasons = gate_support(actual, zero, shuffled, absent)
    status = (
        "trackc_support_set_deepset_sketch_pass_source_unit_next_no_gpu"
        if gate_pass
        else "trackc_support_set_deepset_sketch_fail_no_gpu"
    )
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
            "model_class": "cpu_only_deepset_sketch_not_integrated_into_latentfm",
        },
        "inputs": {"anchor": str(ANCHOR_PATH), "candidate": str(CANDIDATE_PATH)},
        "rows": {"train_multi": len(train_rows), "support_val_multi": len(support_val_rows)},
        "train_info": train_info,
        "train_loo_summary": train_actual,
        "support_val_summary": actual,
        "zero_support_control": zero,
        "shuffled_support_control": shuffled,
        "absent_support_control": absent,
        "decision_reasons": reasons,
        "next_action": (
            "implement default-off set-encoder source/unit gate before any GPU"
            if gate_pass
            else "close current Track C set-encoder sketch and pivot to Track A exact failure-cluster gate"
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
        "# Track C DeepSets Sketch Gate",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "CPU-only learned permutation-invariant sketch over safe-trainselect condition-mean residuals. This is not LatentFM training and does not use held-out query or canonical multi selection.",
        "",
        "## Train Sketch",
        "",
        f"- train examples: `{train_info['n_examples']}`",
        f"- loss trace: `{train_info['loss_trace']}`",
        f"- train LOO delta: `{fmt(train_actual.get('delta_mean'))}`",
        f"- train LOO dataset deltas: `{train_actual.get('dataset_deltas')}`",
        "",
        "## Support-Val",
        "",
        f"- actual delta: `{fmt(actual.get('delta_mean'))}`",
        f"- actual CI95: `{actual.get('ci95')}`",
        f"- actual p_harm: `{actual.get('p_harm')}`",
        f"- actual dataset deltas: `{actual.get('dataset_deltas')}`",
        f"- zero delta: `{fmt(zero.get('delta_mean'))}`",
        f"- shuffle delta: `{fmt(shuffled.get('delta_mean'))}`",
        f"- absent delta: `{fmt(absent.get('delta_mean'))}`",
        "",
        "## Decision Reasons",
        "",
    ]
    lines.extend(f"- `{reason}`" for reason in reasons) if reasons else lines.append("- none")
    lines.extend(["", "## Outputs", "", f"- JSON: `{OUT_JSON}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "md": str(OUT_MD), "json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
