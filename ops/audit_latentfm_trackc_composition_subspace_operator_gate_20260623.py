#!/usr/bin/env python3
"""Query-free Track C composition subspace/operator CPU gate.

This gate follows the failed route-share/additive composition family.  It asks
whether the train_multi interaction subspace can filter or transform the naive
additive correction before any GPU smoke.  Selection is train_multi
leave-one-condition-out only; support_val_multi is final scoring only.

Held-out query, canonical test, canonical multi, active logs, and GPU
artifacts are not read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
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
OUT_JSON = ROOT / "reports/latentfm_trackc_composition_subspace_operator_gate_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_COMPOSITION_SUBSPACE_OPERATOR_GATE_20260623.md"
EXPECTED_TRAINSELECT_SHA256 = "5f29dd5b582a40da3736770ca29950c12c54e46b590c3f7705c1d37da89f4f20"
FOCUS_DATASETS = ("NormanWeissman2019_filtered", "Wessels")


@dataclass(frozen=True)
class Spec:
    name: str
    kind: str
    beta: float
    rank: int = 0
    ridge: float = 0.0


def load_support_module() -> Any:
    spec = importlib.util.spec_from_file_location("trackc_support_route_readiness", SUPPORT_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {SUPPORT_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def genes(row: dict[str, Any]) -> list[str]:
    return [str(g).strip().upper() for g in (row.get("genes") or []) if str(g).strip()]


def condition_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["condition"])


def specs() -> list[Spec]:
    out = [Spec("noop_beta0", "noop", 0.0)]
    for beta in (0.25, 0.50, 0.75, 1.00):
        for rank in (1, 2, 4, 8, 16):
            out.append(Spec(f"project_r{rank}_beta{beta:g}", "project", beta, rank=rank))
        out.append(Spec(f"scalar_beta{beta:g}_ridge1", "scalar", beta, ridge=1.0))
        for rank in (1, 2, 4, 8):
            for ridge in (0.1, 1.0, 10.0):
                out.append(Spec(f"lowrank_r{rank}_beta{beta:g}_ridge{ridge:g}", "lowrank", beta, rank=rank, ridge=ridge))
    return out


def route_vector(support: Any, row: dict[str, Any], single: dict[str, Any], multi: dict[str, Any]) -> np.ndarray:
    return np.asarray(support.predict_baselines(row, single, multi)["support_selected_route"], dtype=np.float32)


def additive_vector(row: dict[str, Any], route: np.ndarray, single: dict[str, Any]) -> tuple[np.ndarray, int, int, str]:
    bank = single.get("gene_raw_mean") or {}
    gs = genes(row)
    parts = []
    raw = 0
    fallback = 0
    for gene in gs:
        value = bank.get(gene)
        if value is None:
            parts.append(np.asarray(route, dtype=np.float32) / max(len(gs), 1))
            fallback += 1
        else:
            parts.append(np.asarray(value, dtype=np.float32))
            raw += 1
    if not parts:
        return np.asarray(route, dtype=np.float32), 0, 0, "empty"
    if raw == len(gs):
        stratum = "full_raw"
    elif raw == 0:
        stratum = "zero_raw"
    else:
        stratum = "partial_raw"
    return np.sum(np.stack(parts, axis=0), axis=0).astype(np.float32), raw, fallback, stratum


def make_xy(rows: list[dict[str, Any]], single: dict[str, Any], multi: dict[str, Any], support: Any) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for row in rows:
        route = route_vector(support, row, single, multi)
        add, _raw, _fallback, _stratum = additive_vector(row, route, single)
        xs.append((add - route).astype(np.float32))
        ys.append((np.asarray(row["residual"], dtype=np.float32) - route).astype(np.float32))
    return np.vstack(xs).astype(np.float64), np.vstack(ys).astype(np.float64)


def fit_transform(rows: list[dict[str, Any]], single: dict[str, Any], multi: dict[str, Any], support: Any, spec: Spec) -> dict[str, Any]:
    x, y = make_xy(rows, single, multi, support)
    if spec.kind == "noop":
        return {"kind": "noop"}
    if spec.kind == "project":
        basis_source = y if np.linalg.norm(y) > 1e-12 else x
        _u, _s, vt = np.linalg.svd(basis_source, full_matrices=False)
        rank = max(1, min(int(spec.rank), vt.shape[0]))
        return {"kind": "project", "basis": vt[:rank].T.astype(np.float32)}
    if spec.kind == "scalar":
        denom = float(np.sum(x * x) + float(spec.ridge))
        alpha = 0.0 if denom <= 1e-12 else float(np.sum(x * y) / denom)
        return {"kind": "scalar", "alpha": float(np.clip(alpha, -1.0, 1.5))}
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


def apply_transform(raw_delta: np.ndarray, fitted: dict[str, Any]) -> np.ndarray:
    if fitted["kind"] == "noop":
        return np.zeros_like(raw_delta, dtype=np.float32)
    if fitted["kind"] == "project":
        basis = np.asarray(fitted["basis"], dtype=np.float32)
        return ((raw_delta @ basis) @ basis.T).astype(np.float32)
    if fitted["kind"] == "scalar":
        return (float(fitted["alpha"]) * raw_delta).astype(np.float32)
    if fitted["kind"] == "lowrank":
        basis = np.asarray(fitted["basis"], dtype=np.float32)
        coef = np.asarray(fitted["coef"], dtype=np.float32)
        return ((raw_delta @ basis) @ coef).astype(np.float32)
    raise ValueError(fitted["kind"])


def score_row(
    row: dict[str, Any],
    single: dict[str, Any],
    multi: dict[str, Any],
    pert_means: dict[str, np.ndarray],
    support: Any,
    spec: Spec,
    fitted: dict[str, Any],
    *,
    compute_mmd: bool,
) -> dict[str, Any]:
    route = route_vector(support, row, single, multi)
    add, raw, fallback, stratum = additive_vector(row, route, single)
    raw_delta = (add - route).astype(np.float32)
    pred = (route + float(spec.beta) * apply_transform(raw_delta, fitted)).astype(np.float32)
    out: dict[str, Any] = {
        "dataset": str(row["dataset"]),
        "condition": str(row["condition"]),
        "genes": genes(row),
        "candidate": support.pp_score(row, pred, pert_means),
        "support_selected_route": support.pp_score(row, route, pert_means),
        "raw_gene_covered": int(raw),
        "fallback_genes": int(fallback),
        "coverage_stratum": stratum,
        "raw_delta_norm": float(np.linalg.norm(raw_delta)),
        "applied_delta_norm": float(np.linalg.norm(pred - route)),
    }
    if compute_mmd:
        for metric, value in support.mmd_scores(row, pred).items():
            out[f"candidate__{metric}"] = value
        for metric, value in support.mmd_scores(row, route).items():
            out[f"support_selected_route__{metric}"] = value
    return out


def train_loo_rows(
    train_rows: list[dict[str, Any]],
    single: dict[str, Any],
    pert_means: dict[str, np.ndarray],
    support: Any,
    spec: Spec,
) -> list[dict[str, Any]]:
    out = []
    for heldout in train_rows:
        fit_rows = [row for row in train_rows if condition_key(row) != condition_key(heldout)]
        multi = support.train_multi_components(fit_rows)
        fitted = fit_transform(fit_rows, single, multi, support, spec)
        row = score_row(heldout, single, multi, pert_means, support, spec, fitted, compute_mmd=False)
        out.append(row)
    return out


def dataset_delta(rows: list[dict[str, Any]], candidate: str = "candidate", baseline: str = "support_selected_route") -> dict[str, float]:
    out = {}
    for ds in sorted({str(row["dataset"]) for row in rows}):
        vals = [float(row[candidate]) - float(row[baseline]) for row in rows if str(row["dataset"]) == ds]
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
        improve_positive = True
    elif metric == "mmd_clamped":
        ck, bk = f"{candidate}__test_mmd_clamped", f"{baseline}__test_mmd_clamped"
        improve_positive = False
    else:
        raise ValueError(metric)
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get(ck) is not None and row.get(bk) is not None:
            by_ds[str(row["dataset"])].append(float(row[ck]) - float(row[bk]))
    datasets = sorted(ds for ds, vals in by_ds.items() if vals)
    if not datasets:
        return {"status": "missing", "candidate": candidate, "baseline": baseline, "metric": metric}
    point = float(np.mean([np.mean(by_ds[ds]) for ds in datasets]))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        sampled = rng.choice(datasets, size=len(datasets), replace=True)
        vals = []
        for ds in sampled:
            arr = np.asarray(by_ds[str(ds)], dtype=np.float64)
            vals.append(float(np.mean(rng.choice(arr, size=len(arr), replace=True))))
        boot.append(float(np.mean(vals)))
    arr = np.asarray(boot, dtype=np.float64)
    if improve_positive:
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
        "delta_mean": point,
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_improve": p_improve,
        "p_harm": p_harm,
        "by_dataset": {ds: float(np.mean(vals)) for ds, vals in by_ds.items()},
    }


def readout_wessels_route_gap(path: Path) -> float | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = str(payload.get("selected_model"))
    for row in payload.get("dataset_breakdown") or []:
        if row.get("dataset") == "Wessels":
            selected_pp = row.get(selected)
            route_pp = row.get("support_selected_route")
            if selected_pp is not None and route_pp is not None:
                return float(selected_pp) - float(route_pp)
    return None


def summarize_train(rows: list[dict[str, Any]], spec: Spec) -> dict[str, Any]:
    ds = dataset_delta(rows)
    return {
        "spec": spec.name,
        "kind": spec.kind,
        "beta": float(spec.beta),
        "rank": int(spec.rank),
        "ridge": float(spec.ridge),
        "n_rows": len(rows),
        "paired_pp_delta": float(np.mean(list(ds.values()))) if ds else None,
        "norman_delta": ds.get("NormanWeissman2019_filtered"),
        "wessels_delta": ds.get("Wessels"),
        "min_dataset_delta": float(min(ds.values())) if ds else None,
        "strata": stratum_summary(rows),
    }


def stratum_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for key in sorted({str(row.get("coverage_stratum")) for row in rows}):
        sub = [row for row in rows if str(row.get("coverage_stratum")) == key]
        vals = [float(row["candidate"]) - float(row["support_selected_route"]) for row in sub]
        out[key] = {
            "n": len(sub),
            "mean_delta": float(np.mean(vals)) if vals else None,
            "min_delta": float(np.min(vals)) if vals else None,
            "n_negative": int(sum(v < 0.0 for v in vals)),
        }
    return out


def select_spec(summaries: list[dict[str, Any]]) -> str:
    eligible = [
        row
        for row in summaries
        if float(row.get("wessels_delta") if row.get("wessels_delta") is not None else -999.0) >= 0.02
        and float(row.get("norman_delta") if row.get("norman_delta") is not None else -999.0) >= -0.01
        and float(row.get("paired_pp_delta") if row.get("paired_pp_delta") is not None else -999.0) > 0.0
    ]
    pool = eligible or summaries
    return str(
        sorted(
            pool,
            key=lambda row: (
                float(row.get("wessels_delta") if row.get("wessels_delta") is not None else -999.0),
                float(row.get("norman_delta") if row.get("norman_delta") is not None else -999.0),
                float(row.get("paired_pp_delta") if row.get("paired_pp_delta") is not None else -999.0),
            ),
            reverse=True,
        )[0]["spec"]
    )


def dataset_breakdown(rows: list[dict[str, Any]], *, wessels_gap: float | None) -> list[dict[str, Any]]:
    ds_pp = dataset_delta(rows)
    out = []
    for ds in sorted({str(row["dataset"]) for row in rows}):
        sub = [row for row in rows if str(row["dataset"]) == ds]
        delta = ds_pp.get(ds)
        gap = wessels_gap if ds == "Wessels" else None
        out.append(
            {
                "dataset": ds,
                "n_conditions": len(sub),
                "candidate": float(np.mean([float(row["candidate"]) for row in sub])),
                "support_selected_route": float(np.mean([float(row["support_selected_route"]) for row in sub])),
                "delta_pp": delta,
                "route_gap_pp": gap,
                "route_gap_closed_fraction": None if gap is None or abs(gap) <= 1e-12 or delta is None else float(delta / gap),
                "candidate_mmd_clamped": float(np.mean([float(row["candidate__test_mmd_clamped"]) for row in sub])),
                "route_mmd_clamped": float(np.mean([float(row["support_selected_route__test_mmd_clamped"]) for row in sub])),
                "strata": stratum_summary(sub),
            }
        )
    return out


def find_dataset(summary: list[dict[str, Any]], dataset: str) -> dict[str, Any]:
    for row in summary:
        if row.get("dataset") == dataset:
            return row
    return {}


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    guard = payload["split_guard"]
    selected = payload["selected_support_summary"]
    pp = payload["paired_pp_delta"]
    mmd = payload["paired_mmd_delta"]
    w = find_dataset(selected["dataset_breakdown"], "Wessels")
    n = find_dataset(selected["dataset_breakdown"], "NormanWeissman2019_filtered")
    if guard["sha256"] != EXPECTED_TRAINSELECT_SHA256:
        reasons.append("trainselect_split_hash_mismatch")
    if float(w.get("delta_pp") if w.get("delta_pp") is not None else -999.0) < 0.02:
        reasons.append("support_wessels_delta_below_0p02")
    if float(w.get("route_gap_closed_fraction") if w.get("route_gap_closed_fraction") is not None else -999.0) < 0.05:
        reasons.append("wessels_route_gap_closure_below_0p05")
    if float(n.get("delta_pp") if n.get("delta_pp") is not None else -999.0) < -0.01:
        reasons.append("support_norman_delta_below_minus_0p01")
    if float(pp.get("p_harm") if pp.get("p_harm") is not None else 1.0) > 0.20:
        reasons.append("bootstrap_pp_harm_above_0p20")
    if float(mmd.get("delta_mean") if mmd.get("delta_mean") is not None else 999.0) > 0.005:
        reasons.append("mmd_delta_hard_harm_above_0p005")
    if float(mmd.get("p_harm") if mmd.get("p_harm") is not None else 1.0) > 0.80:
        reasons.append("mmd_harm_probability_above_0p80")
    if payload["n_support_val_rows"] != 24:
        reasons.append("support_val_coverage_not_complete")
    status = "trackc_composition_subspace_operator_gate_pass_authorize_one_capped_gpu_smoke" if not reasons else "trackc_composition_subspace_operator_gate_fail_no_gpu"
    return {
        "status": status,
        "gpu_authorization": "one_capped_trackc_support_only_smoke" if not reasons else "none",
        "reasons": reasons,
    }


def split_guard(path: Path, split: dict[str, Any]) -> dict[str, Any]:
    return {
        "split_file": str(path),
        "sha256": sha256(path),
        "expected_sha256": EXPECTED_TRAINSELECT_SHA256,
        "datasets": {
            ds: {
                "train_multi": len((split.get(ds) or {}).get("train_multi") or []),
                "support_val_multi": len((split.get(ds) or {}).get("support_val_multi") or []),
                "heldout_query_multi_final_only": len((split.get(ds) or {}).get("heldout_query_multi_final_only") or []),
            }
            for ds in FOCUS_DATASETS
        },
        "leakage_status": "train_multi_for_selection_support_val_multi_for_final_scoring_no_query_no_canonical_outputs",
    }


def render(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    selected = payload["selected_spec"]
    support_summary = payload["selected_support_summary"]
    lines = [
        "# Track C Composition Subspace/Operator CPU Gate",
        "",
        f"Status: `{decision['status']}`",
        f"GPU authorization: `{decision['gpu_authorization']}`",
        "",
        "## Hypothesis",
        "",
        "Naive additive/route-share composition has a real Wessels signal but a Norman/tail no-harm failure. "
        "A transform learned only from train_multi interaction residual geometry may preserve in-subspace signal while suppressing harmful additive components.",
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
        f"- selected spec: `{selected}`",
        "",
        "## Gate Criteria",
        "",
        f"- Wessels pp delta: `{fmt(find_dataset(support_summary['dataset_breakdown'], 'Wessels').get('delta_pp'))}` (gate `>= +0.020000`)",
        f"- Wessels route-gap closure: `{fmt(find_dataset(support_summary['dataset_breakdown'], 'Wessels').get('route_gap_closed_fraction'))}` (gate `>= +0.050000`)",
        f"- Norman pp delta: `{fmt(find_dataset(support_summary['dataset_breakdown'], 'NormanWeissman2019_filtered').get('delta_pp'))}` (gate `>= -0.010000`)",
        f"- bootstrap pp p_harm: `{fmt(payload['paired_pp_delta'].get('p_harm'))}` (gate `<= 0.200000`)",
        f"- MMD delta: `{fmt(payload['paired_mmd_delta'].get('delta_mean'))}` (hard-harm gate `<= +0.005000`)",
        f"- MMD p_harm: `{fmt(payload['paired_mmd_delta'].get('p_harm'))}` (gate `<= 0.800000`)",
        "",
        "## Support-Val Dataset Breakdown",
        "",
        "| dataset | n | candidate pp | route pp | delta pp | route-gap closure | candidate MMD | route MMD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in support_summary["dataset_breakdown"]:
        lines.append(
            f"| {row['dataset']} | {row['n_conditions']} | {fmt(row.get('candidate'))} | "
            f"{fmt(row.get('support_selected_route'))} | {fmt(row.get('delta_pp'))} | "
            f"{fmt(row.get('route_gap_closed_fraction'))} | {fmt(row.get('candidate_mmd_clamped'))} | "
            f"{fmt(row.get('route_mmd_clamped'))} |"
        )
    lines.extend(
        [
            "",
            "## Train-Only Selection Summary",
            "",
            "Selection used train_multi leave-one-condition-out only; support_val was not used to choose the spec.",
            "",
            "| spec | n | paired pp delta | Norman delta | Wessels delta | min dataset delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["train_summaries"][:15]:
        marker = " (selected)" if row["spec"] == selected else ""
        lines.append(
            f"| `{row['spec']}`{marker} | {row['n_rows']} | {fmt(row.get('paired_pp_delta'))} | "
            f"{fmt(row.get('norman_delta'))} | {fmt(row.get('wessels_delta'))} | {fmt(row.get('min_dataset_delta'))} |"
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
            "- Failure closes this subspace/operator gate until a new leakage-safe hypothesis is documented.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--pert-means-file", type=Path, default=DEFAULT_PERT_MEANS)
    parser.add_argument("--readout-json", type=Path, default=DEFAULT_READOUT_JSON)
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
    full_multi = support.train_multi_components(train_rows)

    train_by_spec = {}
    train_summaries = []
    all_specs = specs()
    for spec in all_specs:
        rows = train_loo_rows(train_rows, single, pert_means, support, spec)
        train_by_spec[spec.name] = rows
        train_summaries.append(summarize_train(rows, spec))
    train_summaries = sorted(
        train_summaries,
        key=lambda row: (
            float(row.get("wessels_delta") if row.get("wessels_delta") is not None else -999.0),
            float(row.get("norman_delta") if row.get("norman_delta") is not None else -999.0),
            float(row.get("paired_pp_delta") if row.get("paired_pp_delta") is not None else -999.0),
        ),
        reverse=True,
    )
    selected_name = select_spec(train_summaries)
    selected_spec = next(spec for spec in all_specs if spec.name == selected_name)
    fitted = fit_transform(train_rows, single, full_multi, support, selected_spec)
    eval_rows = [
        score_row(row, single, full_multi, pert_means, support, selected_spec, fitted, compute_mmd=True)
        for row in support_val
    ]
    wessels_gap = readout_wessels_route_gap(args.readout_json)
    support_summary = {
        "spec": selected_name,
        "dataset_breakdown": dataset_breakdown(eval_rows, wessels_gap=wessels_gap),
        "strata": stratum_summary(eval_rows),
        "rows": eval_rows,
    }
    pp_delta = paired_bootstrap(eval_rows, "candidate", "support_selected_route", metric="pp", n_boot=args.n_boot, seed=args.seed)
    mmd_delta = paired_bootstrap(
        eval_rows,
        "candidate",
        "support_selected_route",
        metric="mmd_clamped",
        n_boot=args.n_boot,
        seed=args.seed + 100,
    )

    payload: dict[str, Any] = {
        "data_dir": str(data_dir),
        "split_guard": guard,
        "pert_means_file": str(args.pert_means_file),
        "readout_json": str(args.readout_json),
        "max_cells_per_condition": args.max_cells_per_condition,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_train_multi_rows": len(train_rows),
        "n_support_val_rows": len(support_val),
        "selected_spec": selected_name,
        "selected_spec_params": selected_spec.__dict__,
        "fitted_transform_summary": {
            key: (float(value) if isinstance(value, (float, int, np.floating)) else str(np.asarray(value).shape))
            for key, value in fitted.items()
        },
        "train_summaries": train_summaries,
        "selected_train_rows": train_by_spec[selected_name],
        "selected_support_summary": support_summary,
        "paired_pp_delta": pp_delta,
        "paired_mmd_delta": mmd_delta,
    }
    payload["decision"] = decide(payload)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["decision"]["status"],
                "gpu_authorization": payload["decision"]["gpu_authorization"],
                "selected_spec": selected_name,
                "out_md": str(args.out_md),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
