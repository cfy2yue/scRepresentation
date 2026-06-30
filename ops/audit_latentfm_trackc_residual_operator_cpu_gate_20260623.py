#!/usr/bin/env python3
"""CPU-only Track C support-conditioned residual-operator gate.

The gate asks whether a frozen, transparent residual operator can absorb the
Wessels support route gap before launching any new GPU smoke.  It uses only the
safe trainselect split: train_multi/train_single for fitting and support_val_multi
for scoring.  Held-out query rows and canonical Track A/test_multi outputs are
not read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
SUPPORT_MODULE_PATH = ROOT / "ops/audit_latentfm_trackc_support_route_readiness_20260622.py"
DEFAULT_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
DEFAULT_PERT_MEANS = (
    ROOT
    / "runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/"
    "xverse_trainonly_pert_means_split_seed42_crossbgval_v2.npz"
)
DEFAULT_READOUT_JSON = ROOT / "reports/latentfm_trackc_trainonly_memory_readout_gate_20260622.json"
DEFAULT_BOTTLENECK_SUMMARY = ROOT / "reports/latentfm_trackc_memory_transfer_bottleneck_summary_20260622.csv"
OUT_JSON = ROOT / "reports/latentfm_trackc_residual_operator_cpu_gate_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_RESIDUAL_OPERATOR_CPU_GATE_20260623.md"
EXPECTED_TRAINSELECT_SHA256 = "5f29dd5b582a40da3736770ca29950c12c54e46b590c3f7705c1d37da89f4f20"
FOCUS_DATASETS = ("NormanWeissman2019_filtered", "Wessels")
BASELINES = ("support_selected_route", "dataset_multi_mean", "additive_single_sum")


@dataclass(frozen=True)
class MemorySpec:
    name: str
    mode: str
    k: int
    same_dataset: bool
    min_score: float


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    kind: str
    rank: int = 0
    ridge: float = 0.0


def load_support_module() -> Any:
    spec = importlib.util.spec_from_file_location("trackc_support_route_readiness", SUPPORT_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {SUPPORT_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def as_float(value: Any) -> float | None:
    if value is None or value == "" or str(value).lower() == "none":
        return None
    return float(value)


def fmt(value: Any) -> str:
    value = as_float(value)
    return "NA" if value is None else f"{value:+.6f}"


def condition_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["condition"])


def gene_set(row: dict[str, Any]) -> set[str]:
    return {str(g).strip().upper() for g in row.get("genes") or [] if str(g).strip()}


def gene_score(a: dict[str, Any], b: dict[str, Any], mode: str) -> float:
    ga = gene_set(a)
    gb = gene_set(b)
    if mode == "overlap":
        return float(len(ga & gb))
    if mode == "jaccard":
        return float(len(ga & gb) / max(len(ga | gb), 1))
    raise ValueError(mode)


def memory_specs() -> list[MemorySpec]:
    return [
        MemorySpec("mem_jaccard_k3_same_ds_min0p25", "jaccard", 3, True, 0.25),
        MemorySpec("mem_jaccard_k1_same_ds_min0p25", "jaccard", 1, True, 0.25),
        MemorySpec("mem_jaccard_k5_same_ds_min0p25", "jaccard", 5, True, 0.25),
        MemorySpec("mem_overlap_k3_same_ds_min1", "overlap", 3, True, 1.0),
        MemorySpec("mem_jaccard_k3_all_ds_min0p25", "jaccard", 3, False, 0.25),
        MemorySpec("mem_overlap_k3_all_ds_min1", "overlap", 3, False, 1.0),
    ]


def operator_specs() -> list[OperatorSpec]:
    specs = [OperatorSpec("scalar_ridge0", "scalar", ridge=0.0)]
    for rank in (1, 2, 4, 8):
        for ridge in (0.1, 1.0, 10.0):
            specs.append(OperatorSpec(f"lowrank_r{rank}_ridge{ridge:g}", "lowrank", rank=rank, ridge=ridge))
    return specs


def weighted_memory_prediction(target: dict[str, Any], memory: list[dict[str, Any]], spec: MemorySpec) -> np.ndarray | None:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for row in memory:
        if condition_key(row) == condition_key(target):
            continue
        if spec.same_dataset and str(row["dataset"]) != str(target["dataset"]):
            continue
        score = gene_score(target, row, spec.mode)
        if score >= spec.min_score:
            candidates.append((score, row))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], str(item[1]["dataset"]), str(item[1]["condition"])), reverse=True)
    selected = candidates[: max(spec.k, 1)]
    weights = np.asarray([max(score, 1e-6) for score, _ in selected], dtype=np.float64)
    weights = weights / weights.sum()
    residuals = np.vstack([np.asarray(row["residual"], dtype=np.float32) for _, row in selected])
    return (weights[:, None] * residuals).sum(axis=0).astype(np.float32)


def route_residual(row: dict[str, Any], single: dict[str, Any], multi: dict[str, Any], support: Any) -> np.ndarray:
    return np.asarray(support.predict_baselines(row, single, multi)["support_selected_route"], dtype=np.float32)


def make_samples(
    targets: list[dict[str, Any]],
    memory: list[dict[str, Any]],
    mem_spec: MemorySpec,
    single: dict[str, Any],
    multi: dict[str, Any],
    support: Any,
) -> list[dict[str, Any]]:
    out = []
    for row in targets:
        mem = weighted_memory_prediction(row, memory, mem_spec)
        if mem is None:
            continue
        route = route_residual(row, single, multi, support)
        out.append(
            {
                "dataset": str(row["dataset"]),
                "condition": str(row["condition"]),
                "genes": list(row.get("genes") or []),
                "row": row,
                "route": route,
                "context_delta": (mem - route).astype(np.float32),
                "target_delta": (np.asarray(row["residual"], dtype=np.float32) - route).astype(np.float32),
            }
        )
    return out


def fit_operator(samples: list[dict[str, Any]], spec: OperatorSpec) -> dict[str, Any]:
    if not samples:
        raise ValueError("no samples to fit")
    x = np.vstack([s["context_delta"] for s in samples]).astype(np.float64)
    y = np.vstack([s["target_delta"] for s in samples]).astype(np.float64)
    if spec.kind == "scalar":
        denom = float(np.sum(x * x) + spec.ridge)
        alpha = 0.0 if denom <= 1e-12 else float(np.sum(x * y) / denom)
        alpha = float(np.clip(alpha, -1.0, 1.5))
        return {"kind": "scalar", "alpha": alpha}
    if spec.kind == "lowrank":
        rank = max(1, min(int(spec.rank), x.shape[0], x.shape[1]))
        _u, _s, vt = np.linalg.svd(x, full_matrices=False)
        basis = vt[:rank].T
        feat = x @ basis
        lhs = feat.T @ feat + float(spec.ridge) * np.eye(rank)
        rhs = feat.T @ y
        coef = np.linalg.solve(lhs, rhs)
        return {"kind": "lowrank", "basis": basis.astype(np.float32), "coef": coef.astype(np.float32), "rank": rank}
    raise ValueError(spec.kind)


def apply_operator(route: np.ndarray, context_delta: np.ndarray, fitted: dict[str, Any]) -> np.ndarray:
    if fitted["kind"] == "scalar":
        correction = float(fitted["alpha"]) * context_delta
    elif fitted["kind"] == "lowrank":
        basis = np.asarray(fitted["basis"], dtype=np.float32)
        coef = np.asarray(fitted["coef"], dtype=np.float32)
        correction = (context_delta @ basis) @ coef
    else:
        raise ValueError(fitted["kind"])
    return (route + correction).astype(np.float32)


def score_prediction(
    sample: dict[str, Any],
    pred: np.ndarray,
    pert_means: dict[str, np.ndarray],
    support: Any,
    *,
    compute_mmd: bool,
) -> dict[str, Any]:
    row = sample["row"]
    out = {
        "dataset": str(row["dataset"]),
        "condition": str(row["condition"]),
        "genes": list(row.get("genes") or []),
        "candidate": support.pp_score(row, pred, pert_means),
        "support_selected_route": support.pp_score(row, sample["route"], pert_means),
    }
    if compute_mmd:
        for metric, value in support.mmd_scores(row, pred).items():
            out[f"candidate__{metric}"] = value
        for metric, value in support.mmd_scores(row, sample["route"]).items():
            out[f"support_selected_route__{metric}"] = value
    return out


def score_noop_row(
    row: dict[str, Any],
    single: dict[str, Any],
    multi: dict[str, Any],
    pert_means: dict[str, np.ndarray],
    support: Any,
    *,
    compute_mmd: bool,
) -> dict[str, Any]:
    route = route_residual(row, single, multi, support)
    out = {
        "dataset": str(row["dataset"]),
        "condition": str(row["condition"]),
        "genes": list(row.get("genes") or []),
        "candidate": support.pp_score(row, route, pert_means),
        "support_selected_route": support.pp_score(row, route, pert_means),
        "no_context_noop": True,
    }
    if compute_mmd:
        for metric, value in support.mmd_scores(row, route).items():
            out[f"candidate__{metric}"] = value
            out[f"support_selected_route__{metric}"] = value
    return out


def train_cv_rows(
    train_rows: list[dict[str, Any]],
    mem_spec: MemorySpec,
    op_spec: OperatorSpec,
    single: dict[str, Any],
    multi: dict[str, Any],
    pert_means: dict[str, np.ndarray],
    support: Any,
) -> list[dict[str, Any]]:
    out = []
    for heldout in train_rows:
        fit_rows = [r for r in train_rows if condition_key(r) != condition_key(heldout)]
        fit_samples = make_samples(fit_rows, fit_rows, mem_spec, single, multi, support)
        if len(fit_samples) < 3:
            row = score_noop_row(heldout, single, multi, pert_means, support, compute_mmd=False)
            row.update({"memory_spec": mem_spec.name, "operator_spec": op_spec.name})
            out.append(row)
            continue
        heldout_samples = make_samples([heldout], fit_rows, mem_spec, single, multi, support)
        if not heldout_samples:
            row = score_noop_row(heldout, single, multi, pert_means, support, compute_mmd=False)
            row.update({"memory_spec": mem_spec.name, "operator_spec": op_spec.name})
            out.append(row)
            continue
        fitted = fit_operator(fit_samples, op_spec)
        sample = heldout_samples[0]
        pred = apply_operator(sample["route"], sample["context_delta"], fitted)
        row = score_prediction(sample, pred, pert_means, support, compute_mmd=False)
        row.update({"memory_spec": mem_spec.name, "operator_spec": op_spec.name})
        out.append(row)
    return out


def equal_dataset_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        val = row.get(key)
        if val is not None:
            by_ds[str(row["dataset"])].append(float(val))
    vals = [float(np.mean(v)) for v in by_ds.values() if v]
    return None if not vals else float(np.mean(vals))


def dataset_delta(rows: list[dict[str, Any]], candidate: str, baseline: str) -> dict[str, float]:
    out = {}
    for ds in sorted({str(r["dataset"]) for r in rows}):
        vals = [
            float(r[candidate]) - float(r[baseline])
            for r in rows
            if str(r["dataset"]) == ds and r.get(candidate) is not None and r.get(baseline) is not None
        ]
        if vals:
            out[ds] = float(np.mean(vals))
    return out


def paired_bootstrap(
    rows: list[dict[str, Any]],
    candidate: str,
    baseline: str,
    *,
    metric: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    if metric == "pp":
        ck, bk = candidate, baseline
        improve_is_positive = True
    elif metric == "mmd_clamped":
        ck, bk = f"{candidate}__test_mmd_clamped", f"{baseline}__test_mmd_clamped"
        improve_is_positive = False
    else:
        raise ValueError(metric)
    diffs_by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        a = row.get(ck)
        b = row.get(bk)
        if a is not None and b is not None:
            diffs_by_ds[str(row["dataset"])].append(float(a) - float(b))
    datasets = sorted(ds for ds, vals in diffs_by_ds.items() if vals)
    if not datasets:
        return {"status": "missing", "candidate": candidate, "baseline": baseline, "metric": metric}
    point = float(np.mean([np.mean(diffs_by_ds[ds]) for ds in datasets]))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        sample_ds = rng.choice(datasets, size=len(datasets), replace=True)
        means = []
        for ds in sample_ds:
            vals = np.asarray(diffs_by_ds[str(ds)], dtype=np.float64)
            means.append(float(np.mean(rng.choice(vals, size=len(vals), replace=True))))
        boot.append(float(np.mean(means)))
    arr = np.asarray(boot, dtype=np.float64)
    if improve_is_positive:
        p_improve = float(np.mean(arr > 0.0))
        p_harm = float(np.mean(arr < 0.0))
    else:
        p_improve = float(np.mean(arr < 0.0))
        p_harm = float(np.mean(arr > 0.0))
    return {
        "status": "ok",
        "candidate": candidate,
        "baseline": baseline,
        "metric": metric,
        "n_conditions": int(sum(len(diffs_by_ds[d]) for d in datasets)),
        "n_datasets": int(len(datasets)),
        "delta_mean": point,
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_improve": p_improve,
        "p_harm": p_harm,
        "by_dataset": {ds: float(np.mean(vals)) for ds, vals in diffs_by_ds.items()},
    }


def summarize_cv(rows_by_spec: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    summary = []
    for name, rows in sorted(rows_by_spec.items()):
        by_ds = dataset_delta(rows, "candidate", "support_selected_route")
        summary.append(
            {
                "spec": name,
                "n_rows": len(rows),
                "equal_dataset_pp_delta": equal_dataset_mean(rows, "candidate") - equal_dataset_mean(rows, "support_selected_route")
                if equal_dataset_mean(rows, "candidate") is not None and equal_dataset_mean(rows, "support_selected_route") is not None
                else None,
                "norman_pp_delta": by_ds.get("NormanWeissman2019_filtered"),
                "wessels_pp_delta": by_ds.get("Wessels"),
            }
        )
    return sorted(
        summary,
        key=lambda row: (
            as_float(row.get("wessels_pp_delta")) if as_float(row.get("wessels_pp_delta")) is not None else -999,
            as_float(row.get("equal_dataset_pp_delta")) if as_float(row.get("equal_dataset_pp_delta")) is not None else -999,
        ),
        reverse=True,
    )


def select_spec(cv_summary: list[dict[str, Any]]) -> str:
    eligible = [
        row
        for row in cv_summary
        if (as_float(row.get("wessels_pp_delta")) or -999) > 0.0
        and (as_float(row.get("norman_pp_delta")) or 0.0) >= -0.02
    ]
    pool = eligible or cv_summary
    return str(pool[0]["spec"])


def load_closed_wessels_delta(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    candidates = [
        row
        for row in rows
        if row.get("dataset") == "Wessels" and row.get("finetune_trainable_scope") == "pairwise_condition_adapter"
    ]
    if not candidates:
        return 0.0
    best = max(candidates, key=lambda row: as_float(row.get("mean_delta_pp")) or -999.0)
    return float(as_float(best.get("mean_delta_pp")) or 0.0)


def readout_wessels_route_gap(path: Path) -> float | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = str(payload.get("selected_model"))
    for row in payload.get("dataset_breakdown") or []:
        if row.get("dataset") == "Wessels":
            selected_pp = as_float(row.get(selected))
            route_pp = as_float(row.get("support_selected_route"))
            if selected_pp is not None and route_pp is not None:
                return selected_pp - route_pp
    return None


def split_guard(path: Path, split: dict[str, Any]) -> dict[str, Any]:
    support_total = 0
    heldout_total = 0
    datasets = {}
    for ds in FOCUS_DATASETS:
        obj = split.get(ds) or {}
        support_n = len(obj.get("support_val_multi") or [])
        heldout_n = len(obj.get("heldout_query_multi_final_only") or [])
        support_total += support_n
        heldout_total += heldout_n
        datasets[ds] = {
            "train_multi": len(obj.get("train_multi") or []),
            "support_val_multi": support_n,
            "test_multi": len(obj.get("test_multi") or []),
            "heldout_query_multi_final_only": heldout_n,
        }
    return {
        "split_file": str(path),
        "sha256": sha256(path),
        "expected_sha256": EXPECTED_TRAINSELECT_SHA256,
        "support_val_multi_total_focus": support_total,
        "heldout_query_multi_final_only_total_focus_metadata_only": heldout_total,
        "datasets": datasets,
        "leakage_status": "used_train_multi_train_single_for_fit_and_support_val_multi_for_scoring_no_heldout_query_no_canonical_outputs",
    }


def decide(
    eval_rows: list[dict[str, Any]],
    pp_delta: dict[str, Any],
    mmd_delta: dict[str, Any],
    *,
    closed_wessels_delta: float,
    wessels_route_gap: float | None,
    wiring_delta_l2: float,
    split: dict[str, Any],
) -> dict[str, Any]:
    reasons = []
    by_ds = pp_delta.get("by_dataset") or {}
    wessels_delta = float(by_ds.get("Wessels", -999.0))
    norman_delta = float(by_ds.get("NormanWeissman2019_filtered", -999.0))
    wessels_vs_closed = wessels_delta - closed_wessels_delta
    closure = None if not wessels_route_gap or abs(wessels_route_gap) < 1e-12 else wessels_delta / wessels_route_gap
    if split["sha256"] != EXPECTED_TRAINSELECT_SHA256:
        reasons.append("trainselect_split_hash_mismatch")
    if wessels_vs_closed < 0.02:
        reasons.append("wessels_delta_vs_best_closed_family_below_0p02")
    if closure is None or closure < 0.05:
        reasons.append("wessels_route_gap_closure_below_0p05")
    if norman_delta < -0.02:
        reasons.append("norman_material_pp_loss")
    if float(pp_delta.get("p_harm") if pp_delta.get("p_harm") is not None else 1.0) > 0.20:
        reasons.append("bootstrap_pp_harm_probability_above_0p20")
    if mmd_delta.get("status") != "ok":
        reasons.append("mmd_comparison_missing")
    elif float(mmd_delta.get("p_harm") if mmd_delta.get("p_harm") is not None else 1.0) > 0.80:
        reasons.append("mmd_hard_harm_vs_route")
    if wiring_delta_l2 <= 1e-8:
        reasons.append("wiring_test_context_does_not_change_prediction")
    if len(eval_rows) != 24:
        reasons.append("support_val_coverage_not_complete")
    status = "residual_operator_cpu_gate_pass_authorize_one_capped_gpu_smoke" if not reasons else "residual_operator_cpu_gate_fail_no_gpu"
    return {
        "status": status,
        "gpu_authorization": "one_capped_trackc_support_only_smoke" if not reasons else "none",
        "reasons": reasons,
        "wessels_delta_vs_route": wessels_delta,
        "wessels_best_closed_family_delta": closed_wessels_delta,
        "wessels_delta_vs_best_closed_family": wessels_vs_closed,
        "wessels_route_gap_from_readout": wessels_route_gap,
        "wessels_route_gap_closure": closure,
        "norman_delta_vs_route": norman_delta,
        "wiring_delta_l2": wiring_delta_l2,
        "n_eval_rows": len(eval_rows),
        "n_no_context_noop_eval_rows": sum(1 for row in eval_rows if row.get("no_context_noop")),
    }


def render(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    selected = payload["selected_spec"]
    lines = [
        "# Track C Residual-Operator CPU Gate",
        "",
        f"Status: `{decision['status']}`",
        f"GPU authorization: `{decision['gpu_authorization']}`",
        "",
        "## Provenance",
        "",
        f"- split_file: `{payload['split_guard']['split_file']}`",
        f"- split SHA256: `{payload['split_guard']['sha256']}`",
        f"- data_dir: `{payload['data_dir']}`",
        f"- pert_means_file: `{payload['pert_means_file']}`",
        f"- leakage_status: `{payload['split_guard']['leakage_status']}`",
        f"- train_multi rows: `{payload['n_train_multi_rows']}`",
        f"- support_val_multi rows: `{payload['n_support_val_rows']}`",
        f"- no-context no-op support_val rows: `{decision['n_no_context_noop_eval_rows']}`",
        f"- selected spec: `{selected}`",
        "",
        "## Gate Criteria",
        "",
        f"- Wessels delta vs best closed family: `{fmt(decision['wessels_delta_vs_best_closed_family'])}` (gate `>= +0.020000`)",
        f"- Wessels route-gap closure: `{fmt(decision['wessels_route_gap_closure'])}` (gate `>= +0.050000`)",
        f"- Norman delta vs route: `{fmt(decision['norman_delta_vs_route'])}` (gate `>= -0.020000`)",
        f"- bootstrap pp p_harm: `{fmt(payload['paired_pp_delta'].get('p_harm'))}` (gate `<= 0.200000`)",
        f"- MMD p_harm vs route: `{fmt(payload['paired_mmd_delta'].get('p_harm'))}` (hard-harm gate `<= 0.800000`)",
        f"- wiring delta L2: `{fmt(decision['wiring_delta_l2'])}`",
        "",
        "## Support-Val Dataset Breakdown",
        "",
        "| dataset | n | candidate pp | route pp | delta | candidate MMD | route MMD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["dataset_breakdown"]:
        lines.append(
            f"| {row['dataset']} | {row['n_conditions']} | {fmt(row.get('candidate'))} | "
            f"{fmt(row.get('support_selected_route'))} | {fmt(row.get('delta_pp'))} | "
            f"{fmt(row.get('candidate_mmd_clamped'))} | {fmt(row.get('route_mmd_clamped'))} |"
        )
    lines.extend(
        [
            "",
            "## Train-CV Selection Summary",
            "",
            "Selection used train_multi leave-one-out only; support-val was not used to choose this spec.",
            "",
            "| spec | n | equal-dataset delta | Norman delta | Wessels delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["cv_summary"][:12]:
        marker = " (selected)" if row["spec"] == selected else ""
        lines.append(
            f"| `{row['spec']}`{marker} | {row['n_rows']} | {fmt(row.get('equal_dataset_pp_delta'))} | "
            f"{fmt(row.get('norman_pp_delta'))} | {fmt(row.get('wessels_pp_delta'))} |"
        )
    lines.extend(
        [
            "",
            "## Paired Support-Val Delta",
            "",
            f"- pp delta vs route: `{fmt(payload['paired_pp_delta'].get('delta_mean'))}` "
            f"CI `[{fmt((payload['paired_pp_delta'].get('ci95') or [None, None])[0])}, "
            f"{fmt((payload['paired_pp_delta'].get('ci95') or [None, None])[1])}]`",
            f"- MMD delta vs route: `{fmt(payload['paired_mmd_delta'].get('delta_mean'))}` "
            f"CI `[{fmt((payload['paired_mmd_delta'].get('ci95') or [None, None])[0])}, "
            f"{fmt((payload['paired_mmd_delta'].get('ci95') or [None, None])[1])}]`",
            "",
            "## Decision Reasons",
            "",
        ]
    )
    reasons = decision.get("reasons") or []
    lines.extend([f"- `{reason}`" for reason in reasons] if reasons else ["- none"])
    lines.extend(
        [
            "",
            "## Usage Rule",
            "",
            "- Passing authorizes at most one capped Track C support-only GPU smoke.",
            "- It does not authorize held-out query evaluation or any formal multi-success claim.",
            "- Failure closes this residual-operator CPU gate until a new leakage-safe hypothesis is documented.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--pert-means-file", type=Path, default=DEFAULT_PERT_MEANS)
    parser.add_argument("--readout-json", type=Path, default=DEFAULT_READOUT_JSON)
    parser.add_argument("--bottleneck-summary", type=Path, default=DEFAULT_BOTTLENECK_SUMMARY)
    parser.add_argument("--max-cells-per-condition", type=int, default=256)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    args = parser.parse_args()

    support = load_support_module()
    data_dir = args.data_dir.resolve()
    split = support.load_json(args.split_file)
    manifest = support.load_json(data_dir / "manifest.json")
    metadata = support.load_json(Path(manifest["condition_metadata_file"]))
    pert_means = {k: v.astype(np.float32) for k, v in np.load(args.pert_means_file).items()}

    guard = split_guard(args.split_file, split)
    if guard["sha256"] != EXPECTED_TRAINSELECT_SHA256:
        raise RuntimeError(f"unexpected trainselect split hash: {guard['sha256']}")
    for ds in FOCUS_DATASETS:
        obj = split.get(ds) or {}
        if set(obj.get("support_val_multi") or []) & set(obj.get("heldout_query_multi_final_only") or []):
            raise RuntimeError(f"{ds}: support_val_multi overlaps heldout query")

    train_rows = support.collect_role_rows(data_dir, split, metadata, "train_multi", max_cells=args.max_cells_per_condition)
    support_val = support.collect_role_rows(data_dir, split, metadata, "support_val_multi", max_cells=args.max_cells_per_condition)
    single = support.train_single_components(data_dir, split, metadata, max_cells=args.max_cells_per_condition)
    multi = support.train_multi_components(train_rows)

    cv_by_spec: dict[str, list[dict[str, Any]]] = {}
    spec_lookup: dict[str, tuple[MemorySpec, OperatorSpec]] = {}
    for mem_spec in memory_specs():
        for op_spec in operator_specs():
            name = f"{mem_spec.name}__{op_spec.name}"
            spec_lookup[name] = (mem_spec, op_spec)
            cv_by_spec[name] = train_cv_rows(train_rows, mem_spec, op_spec, single, multi, pert_means, support)
    cv_summary = summarize_cv(cv_by_spec)
    selected_name = select_spec(cv_summary)
    selected_mem, selected_op = spec_lookup[selected_name]

    fit_samples = make_samples(train_rows, train_rows, selected_mem, single, multi, support)
    fitted = fit_operator(fit_samples, selected_op)
    eval_samples = make_samples(support_val, train_rows, selected_mem, single, multi, support)
    eval_by_key = {condition_key(sample["row"]): sample for sample in eval_samples}
    eval_rows = []
    for target in support_val:
        sample = eval_by_key.get(condition_key(target))
        if sample is None:
            row = score_noop_row(target, single, multi, pert_means, support, compute_mmd=True)
        else:
            pred = apply_operator(sample["route"], sample["context_delta"], fitted)
            row = score_prediction(sample, pred, pert_means, support, compute_mmd=True)
        row["memory_spec"] = selected_mem.name
        row["operator_spec"] = selected_op.name
        eval_rows.append(row)

    # Model-facing wiring test: changing the support context must change output.
    wiring_delta_l2 = 0.0
    if eval_samples:
        sample = eval_samples[0]
        pred_real = apply_operator(sample["route"], sample["context_delta"], fitted)
        pred_zero = apply_operator(sample["route"], np.zeros_like(sample["context_delta"]), fitted)
        wiring_delta_l2 = float(np.linalg.norm(pred_real - pred_zero))

    by_dataset = []
    for ds in sorted({str(r["dataset"]) for r in eval_rows}):
        sub = [r for r in eval_rows if str(r["dataset"]) == ds]
        by_dataset.append(
            {
                "dataset": ds,
                "n_conditions": len(sub),
                "candidate": float(np.mean([r["candidate"] for r in sub if r.get("candidate") is not None])),
                "support_selected_route": float(np.mean([r["support_selected_route"] for r in sub if r.get("support_selected_route") is not None])),
                "delta_pp": dataset_delta(sub, "candidate", "support_selected_route").get(ds),
                "candidate_mmd_clamped": float(np.mean([r["candidate__test_mmd_clamped"] for r in sub])),
                "route_mmd_clamped": float(np.mean([r["support_selected_route__test_mmd_clamped"] for r in sub])),
            }
        )

    pp_delta = paired_bootstrap(
        eval_rows,
        "candidate",
        "support_selected_route",
        metric="pp",
        n_boot=args.n_boot,
        seed=args.seed,
    )
    mmd_delta = paired_bootstrap(
        eval_rows,
        "candidate",
        "support_selected_route",
        metric="mmd_clamped",
        n_boot=args.n_boot,
        seed=args.seed + 100,
    )
    closed_delta = load_closed_wessels_delta(args.bottleneck_summary)
    route_gap = readout_wessels_route_gap(args.readout_json)
    decision = decide(
        eval_rows,
        pp_delta,
        mmd_delta,
        closed_wessels_delta=closed_delta,
        wessels_route_gap=route_gap,
        wiring_delta_l2=wiring_delta_l2,
        split=guard,
    )

    payload = {
        "data_dir": str(data_dir),
        "split_guard": guard,
        "pert_means_file": str(args.pert_means_file),
        "readout_json": str(args.readout_json),
        "bottleneck_summary": str(args.bottleneck_summary),
        "max_cells_per_condition": args.max_cells_per_condition,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_train_multi_rows": len(train_rows),
        "n_support_val_rows": len(support_val),
        "selected_spec": selected_name,
        "selected_memory_spec": selected_mem.__dict__,
        "selected_operator_spec": selected_op.__dict__,
        "fitted_operator_summary": {
            key: (float(value) if isinstance(value, (float, int, np.floating)) else str(np.asarray(value).shape))
            for key, value in fitted.items()
        },
        "cv_summary": cv_summary,
        "dataset_breakdown": by_dataset,
        "paired_pp_delta": pp_delta,
        "paired_mmd_delta": mmd_delta,
        "eval_rows": eval_rows,
        "decision": decision,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "gpu_authorization": decision["gpu_authorization"], "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
