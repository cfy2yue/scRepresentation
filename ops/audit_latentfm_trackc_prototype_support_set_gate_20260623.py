#!/usr/bin/env python3
"""Query-free Track C prototype support-set CPU gate.

This gate tests a support-set task rule that is different from gene-overlap
neighbor transfer and composition filters.  Train_multi route/target residual
geometry is compressed into response prototypes.  A support_val row receives a
small correction from prototypes whose route centroid is close to its route.

Selection is train_multi leave-one-condition-out only; support_val_multi is
final scoring only.  Held-out query rows, canonical test/multi outputs, active
logs, and GPU artifacts are not read.
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
OUT_JSON = ROOT / "reports/latentfm_trackc_prototype_support_set_gate_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_PROTOTYPE_SUPPORT_SET_GATE_20260623.md"
EXPECTED_TRAINSELECT_SHA256 = "5f29dd5b582a40da3736770ca29950c12c54e46b590c3f7705c1d37da89f4f20"
FOCUS_DATASETS = ("NormanWeissman2019_filtered", "Wessels")


@dataclass(frozen=True)
class Spec:
    name: str
    k: int
    alpha: float
    temperature: float
    same_dataset: bool


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


def condition_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["condition"])


def specs() -> list[Spec]:
    out = []
    for same_dataset in (False, True):
        for k in (2, 4, 8, 12):
            for alpha in (0.25, 0.50, 0.75, 1.00):
                for temp in (0.05, 0.10, 0.20):
                    ds = "same" if same_dataset else "all"
                    out.append(Spec(f"proto_{ds}_k{k}_a{alpha:g}_t{temp:g}", k, alpha, temp, same_dataset))
    return out


def route_vector(support: Any, row: dict[str, Any], single: dict[str, Any], multi: dict[str, Any]) -> np.ndarray:
    return np.asarray(support.predict_baselines(row, single, multi)["support_selected_route"], dtype=np.float32)


def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = float(np.linalg.norm(x))
    return x / n if n > 1e-12 else np.zeros_like(x)


def build_samples(rows: list[dict[str, Any]], single: dict[str, Any], multi: dict[str, Any], support: Any) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        route = route_vector(support, row, single, multi)
        out.append(
            {
                "dataset": str(row["dataset"]),
                "condition": str(row["condition"]),
                "genes": list(row.get("genes") or []),
                "row": row,
                "route": route,
                "route_unit": unit(route),
                "target_delta": (np.asarray(row["residual"], dtype=np.float32) - route).astype(np.float32),
            }
        )
    return out


def fit_prototypes(samples: list[dict[str, Any]], spec: Spec) -> dict[str, Any]:
    if not samples:
        return {"route_centroids": np.zeros((0, 1), dtype=np.float32), "delta_centroids": np.zeros((0, 1), dtype=np.float32), "datasets": []}
    k = max(1, min(int(spec.k), len(samples)))
    routes = np.vstack([s["route_unit"] for s in samples]).astype(np.float32)
    deltas = np.vstack([s["target_delta"] for s in samples]).astype(np.float32)
    # Deterministic farthest-point initialization in route space.
    centers = [0]
    dist = np.full(len(samples), np.inf, dtype=np.float32)
    for _ in range(1, k):
        last = routes[centers[-1]]
        dist = np.minimum(dist, np.sum((routes - last[None, :]) ** 2, axis=1))
        centers.append(int(np.argmax(dist)))
    centroids = routes[centers].copy()
    labels = np.zeros(len(samples), dtype=int)
    for _ in range(8):
        sim = routes @ centroids.T
        labels = np.argmax(sim, axis=1)
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centroids[j] = unit(np.mean(routes[mask], axis=0))
    delta_centroids = []
    proto_datasets = []
    for j in range(k):
        mask = labels == j
        if not np.any(mask):
            delta_centroids.append(np.zeros(deltas.shape[1], dtype=np.float32))
            proto_datasets.append("empty")
        else:
            delta_centroids.append(np.mean(deltas[mask], axis=0).astype(np.float32))
            ds_counts = defaultdict(int)
            for idx in np.where(mask)[0]:
                ds_counts[str(samples[int(idx)]["dataset"])] += 1
            proto_datasets.append(max(ds_counts.items(), key=lambda item: (item[1], item[0]))[0])
    return {
        "route_centroids": centroids.astype(np.float32),
        "delta_centroids": np.vstack(delta_centroids).astype(np.float32),
        "datasets": proto_datasets,
    }


def prototype_correction(sample: dict[str, Any], fitted: dict[str, Any], spec: Spec) -> tuple[np.ndarray, dict[str, Any]]:
    routes = np.asarray(fitted["route_centroids"], dtype=np.float32)
    deltas = np.asarray(fitted["delta_centroids"], dtype=np.float32)
    if routes.size == 0 or deltas.size == 0:
        return np.zeros_like(sample["route"], dtype=np.float32), {"n_active_prototypes": 0, "max_similarity": None}
    sims = sample["route_unit"] @ routes.T
    if spec.same_dataset:
        mask = np.asarray([ds == sample["dataset"] for ds in fitted["datasets"]], dtype=bool)
        if np.any(mask):
            sims = np.where(mask, sims, -np.inf)
    finite = np.isfinite(sims)
    if not np.any(finite):
        return np.zeros_like(sample["route"], dtype=np.float32), {"n_active_prototypes": 0, "max_similarity": None}
    z = sims[finite] / max(float(spec.temperature), 1e-6)
    z = z - float(np.max(z))
    w = np.exp(z)
    w = w / max(float(np.sum(w)), 1e-12)
    corr = (w[:, None] * deltas[finite]).sum(axis=0).astype(np.float32)
    return corr, {"n_active_prototypes": int(np.sum(finite)), "max_similarity": float(np.max(sims[finite]))}


def score_sample(
    sample: dict[str, Any],
    fitted: dict[str, Any],
    spec: Spec,
    pert_means: dict[str, np.ndarray],
    support: Any,
    *,
    compute_mmd: bool,
    shuffle_delta_seed: int | None = None,
) -> dict[str, Any]:
    fit_obj = fitted
    if shuffle_delta_seed is not None:
        rng = np.random.default_rng(shuffle_delta_seed)
        fit_obj = dict(fitted)
        deltas = np.asarray(fitted["delta_centroids"], dtype=np.float32)
        if len(deltas):
            fit_obj["delta_centroids"] = deltas[rng.permutation(len(deltas))]
    corr, meta = prototype_correction(sample, fit_obj, spec)
    pred = (sample["route"] + float(spec.alpha) * corr).astype(np.float32)
    row = sample["row"]
    out = {
        "dataset": str(row["dataset"]),
        "condition": str(row["condition"]),
        "genes": list(row.get("genes") or []),
        "candidate": support.pp_score(row, pred, pert_means),
        "support_selected_route": support.pp_score(row, sample["route"], pert_means),
        "correction_norm": float(np.linalg.norm(float(spec.alpha) * corr)),
        **meta,
    }
    if compute_mmd:
        for metric, value in support.mmd_scores(row, pred).items():
            out[f"candidate__{metric}"] = value
        for metric, value in support.mmd_scores(row, sample["route"]).items():
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
        fit_samples = build_samples(fit_rows, single, multi, support)
        fitted = fit_prototypes(fit_samples, spec)
        sample = build_samples([heldout], single, multi, support)[0]
        out.append(score_sample(sample, fitted, spec, pert_means, support, compute_mmd=False))
    return out


def dataset_delta(rows: list[dict[str, Any]], candidate: str, baseline: str) -> dict[str, float]:
    out = {}
    for ds in sorted({str(row["dataset"]) for row in rows}):
        vals = [float(row[candidate]) - float(row[baseline]) for row in rows if str(row["dataset"]) == ds]
        if vals:
            out[ds] = float(np.mean(vals))
    return out


def paired_bootstrap(rows: list[dict[str, Any]], candidate: str, baseline: str, *, metric: str, n_boot: int, seed: int) -> dict[str, Any]:
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
    return {
        "status": "ok",
        "candidate": candidate,
        "baseline": baseline,
        "metric": metric,
        "delta_mean": point,
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_improve": float(np.mean(arr > 0.0)) if improve_positive else float(np.mean(arr < 0.0)),
        "p_harm": float(np.mean(arr < 0.0)) if improve_positive else float(np.mean(arr > 0.0)),
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


def summarize(rows: list[dict[str, Any]], spec: Spec, *, n_boot: int, seed: int, wessels_gap: float | None, include_mmd: bool) -> dict[str, Any]:
    pp = paired_bootstrap(rows, "candidate", "support_selected_route", metric="pp", n_boot=n_boot, seed=seed)
    mmd = paired_bootstrap(rows, "candidate", "support_selected_route", metric="mmd_clamped", n_boot=n_boot, seed=seed + 100) if include_mmd else None
    ds_pp = dataset_delta(rows, "candidate", "support_selected_route")
    ds_mmd = dataset_delta(rows, "candidate__test_mmd_clamped", "support_selected_route__test_mmd_clamped") if include_mmd else {}
    breakdown = []
    for ds in sorted({str(row["dataset"]) for row in rows}):
        sub = [row for row in rows if str(row["dataset"]) == ds]
        delta = ds_pp.get(ds)
        gap = wessels_gap if ds == "Wessels" else None
        breakdown.append(
            {
                "dataset": ds,
                "n_conditions": len(sub),
                "delta_pp": delta,
                "delta_mmd_clamped": ds_mmd.get(ds),
                "route_gap_pp": gap,
                "route_gap_closed_fraction": None if gap is None or abs(gap) <= 1e-12 or delta is None else float(delta / gap),
                "mean_correction_norm": float(np.mean([float(row.get("correction_norm") or 0.0) for row in sub])),
                "zero_context_rows": int(sum(int(row.get("n_active_prototypes") or 0) == 0 for row in sub)),
            }
        )
    return {
        "spec": spec.name,
        "spec_params": spec.__dict__,
        "paired_pp_delta": pp,
        "paired_mmd_delta": mmd,
        "dataset_breakdown": breakdown,
        "rows": rows,
    }


def find_dataset(summary: dict[str, Any], dataset: str) -> dict[str, Any]:
    for row in summary.get("dataset_breakdown") or []:
        if row.get("dataset") == dataset:
            return row
    return {}


def select_spec(train_summaries: list[dict[str, Any]]) -> str:
    eligible = []
    for row in train_summaries:
        w = find_dataset(row, "Wessels")
        n = find_dataset(row, "NormanWeissman2019_filtered")
        if (
            float(w.get("delta_pp") if w.get("delta_pp") is not None else -999.0) >= 0.02
            and float(n.get("delta_pp") if n.get("delta_pp") is not None else -999.0) >= -0.02
            and float(row["paired_pp_delta"].get("p_harm") if row["paired_pp_delta"].get("p_harm") is not None else 1.0) <= 0.20
        ):
            eligible.append(row)
    pool = eligible or train_summaries
    return str(
        sorted(
            pool,
            key=lambda row: (
                float(find_dataset(row, "Wessels").get("delta_pp") if find_dataset(row, "Wessels").get("delta_pp") is not None else -999.0),
                float(find_dataset(row, "NormanWeissman2019_filtered").get("delta_pp") if find_dataset(row, "NormanWeissman2019_filtered").get("delta_pp") is not None else -999.0),
                float(row["paired_pp_delta"].get("delta_mean") if row["paired_pp_delta"].get("delta_mean") is not None else -999.0),
            ),
            reverse=True,
        )[0]["spec"]
    )


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    support = payload["selected_support_summary"]
    shuffled = payload["shuffled_prototype_control"]
    pp = support["paired_pp_delta"]
    mmd = support["paired_mmd_delta"] or {}
    w = find_dataset(support, "Wessels")
    n = find_dataset(support, "NormanWeissman2019_filtered")
    if payload["split_guard"]["sha256"] != EXPECTED_TRAINSELECT_SHA256:
        reasons.append("trainselect_split_hash_mismatch")
    if float(w.get("delta_pp") if w.get("delta_pp") is not None else -999.0) < 0.02:
        reasons.append("support_wessels_delta_below_0p02")
    if float(w.get("route_gap_closed_fraction") if w.get("route_gap_closed_fraction") is not None else -999.0) < 0.05:
        reasons.append("wessels_route_gap_closure_below_0p05")
    if float(n.get("delta_pp") if n.get("delta_pp") is not None else -999.0) < -0.02:
        reasons.append("support_norman_material_pp_loss")
    if float(pp.get("p_harm") if pp.get("p_harm") is not None else 1.0) > 0.20:
        reasons.append("bootstrap_pp_harm_above_0p20")
    if float(mmd.get("delta_mean") if mmd.get("delta_mean") is not None else 999.0) > 0.005:
        reasons.append("mmd_delta_hard_harm_above_0p005")
    if float(mmd.get("p_harm") if mmd.get("p_harm") is not None else 1.0) > 0.80:
        reasons.append("mmd_harm_probability_above_0p80")
    real = float(pp.get("delta_mean") if pp.get("delta_mean") is not None else 0.0)
    shuf = float(shuffled["paired_pp_delta"].get("delta_mean") if shuffled["paired_pp_delta"].get("delta_mean") is not None else 0.0)
    if shuf > real - 0.02:
        reasons.append("shuffled_prototype_control_not_separated")
    if sum(int(row.get("zero_context_rows") or 0) for row in support["dataset_breakdown"]) > 0:
        reasons.append("support_zero_context_rows_present")
    status = "trackc_prototype_support_set_gate_pass_authorize_one_capped_gpu_smoke" if not reasons else "trackc_prototype_support_set_gate_fail_no_gpu"
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
        "leakage_status": "train_multi_loo_selection_support_val_final_no_query_no_canonical_outputs",
        "datasets": {
            ds: {
                "train_multi": len((split.get(ds) or {}).get("train_multi") or []),
                "support_val_multi": len((split.get(ds) or {}).get("support_val_multi") or []),
                "heldout_query_multi_final_only": len((split.get(ds) or {}).get("heldout_query_multi_final_only") or []),
            }
            for ds in FOCUS_DATASETS
        },
    }


def render(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    support = payload["selected_support_summary"]
    shuffled = payload["shuffled_prototype_control"]
    lines = [
        "# Track C Prototype Support-Set CPU Gate",
        "",
        f"Status: `{decision['status']}`",
        f"GPU authorization: `{decision['gpu_authorization']}`",
        "",
        "## Boundary",
        "",
        f"- split_file: `{payload['split_guard']['split_file']}`",
        f"- split SHA256: `{payload['split_guard']['sha256']}`",
        f"- leakage_status: `{payload['split_guard']['leakage_status']}`",
        f"- selected spec: `{payload['selected_spec']}`",
        "- Train_multi leave-one-condition-out selected the prototype rule; support_val_multi was final scoring only.",
        "- No held-out query, canonical test, canonical multi, active logs, or GPU artifacts were read.",
        "",
        "## Gate Criteria",
        "",
        f"- Wessels pp delta: `{fmt(find_dataset(support, 'Wessels').get('delta_pp'))}` (gate `>= +0.020000`)",
        f"- Wessels route-gap closure: `{fmt(find_dataset(support, 'Wessels').get('route_gap_closed_fraction'))}` (gate `>= +0.050000`)",
        f"- Norman pp delta: `{fmt(find_dataset(support, 'NormanWeissman2019_filtered').get('delta_pp'))}` (gate `>= -0.020000`)",
        f"- bootstrap pp p_harm: `{fmt(support['paired_pp_delta'].get('p_harm'))}` (gate `<= 0.200000`)",
        f"- MMD delta: `{fmt((support['paired_mmd_delta'] or {}).get('delta_mean'))}` (hard-harm gate `<= +0.005000`)",
        f"- shuffled prototype pp delta: `{fmt(shuffled['paired_pp_delta'].get('delta_mean'))}` (must be at least `0.020000` below real)",
        "",
        "## Support-Val Dataset Breakdown",
        "",
        "| dataset | n | pp delta | MMD delta | closure | correction norm | zero rows |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in support["dataset_breakdown"]:
        lines.append(
            f"| {row['dataset']} | {row['n_conditions']} | {fmt(row.get('delta_pp'))} | "
            f"{fmt(row.get('delta_mmd_clamped'))} | {fmt(row.get('route_gap_closed_fraction'))} | "
            f"{fmt(row.get('mean_correction_norm'))} | {row.get('zero_context_rows')} |"
        )
    lines.extend(["", "## Train-Only Selection Summary", "", "| spec | pp delta | Norman | Wessels | p_harm |", "|---|---:|---:|---:|---:|"])
    for row in payload["train_summaries"][:15]:
        marker = " (selected)" if row["spec"] == payload["selected_spec"] else ""
        lines.append(
            f"| `{row['spec']}`{marker} | {fmt(row['paired_pp_delta'].get('delta_mean'))} | "
            f"{fmt(find_dataset(row, 'NormanWeissman2019_filtered').get('delta_pp'))} | "
            f"{fmt(find_dataset(row, 'Wessels').get('delta_pp'))} | {fmt(row['paired_pp_delta'].get('p_harm'))} |"
        )
    lines.extend(["", "## Decision Reasons", ""])
    reasons = decision.get("reasons") or []
    lines.extend([f"- `{reason}`" for reason in reasons] if reasons else ["- none"])
    lines.extend(
        [
            "",
            "## Usage Rule",
            "",
            "- Passing authorizes at most one capped Track C support-only GPU smoke.",
            "- It does not authorize held-out query evaluation or any formal multi-success claim.",
            "- Failure closes this prototype gate until a new predeclared prototype representation is documented.",
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
    split = support.load_json(args.split_file)
    manifest = support.load_json(args.data_dir / "manifest.json")
    metadata = support.load_json(Path(manifest["condition_metadata_file"]))
    guard = split_guard(args.split_file, split)
    if guard["sha256"] != EXPECTED_TRAINSELECT_SHA256:
        raise RuntimeError(f"unexpected trainselect split hash: {guard['sha256']}")
    for ds in FOCUS_DATASETS:
        obj = split.get(ds) or {}
        if set(obj.get("support_val_multi") or []) & set(obj.get("heldout_query_multi_final_only") or []):
            raise RuntimeError(f"{ds}: support_val_multi overlaps heldout query")

    pert_means = {k: v.astype(np.float32) for k, v in np.load(args.pert_means_file).items()}
    train_rows = support.collect_role_rows(args.data_dir, split, metadata, "train_multi", max_cells=args.max_cells_per_condition)
    support_val = support.collect_role_rows(args.data_dir, split, metadata, "support_val_multi", max_cells=args.max_cells_per_condition)
    single = support.train_single_components(args.data_dir, split, metadata, max_cells=args.max_cells_per_condition)
    full_multi = support.train_multi_components(train_rows)
    wessels_gap = readout_wessels_route_gap(args.readout_json)

    all_specs = specs()
    train_summaries = []
    train_by_spec = {}
    for spec in all_specs:
        rows = train_loo_rows(train_rows, single, pert_means, support, spec)
        train_by_spec[spec.name] = rows
        train_summaries.append(summarize(rows, spec, n_boot=args.n_boot, seed=args.seed, wessels_gap=wessels_gap, include_mmd=False))
    train_summaries = sorted(
        train_summaries,
        key=lambda row: (
            float(find_dataset(row, "Wessels").get("delta_pp") if find_dataset(row, "Wessels").get("delta_pp") is not None else -999.0),
            float(find_dataset(row, "NormanWeissman2019_filtered").get("delta_pp") if find_dataset(row, "NormanWeissman2019_filtered").get("delta_pp") is not None else -999.0),
            float(row["paired_pp_delta"].get("delta_mean") if row["paired_pp_delta"].get("delta_mean") is not None else -999.0),
        ),
        reverse=True,
    )
    selected = select_spec(train_summaries)
    selected_spec = next(spec for spec in all_specs if spec.name == selected)
    fit_samples = build_samples(train_rows, single, full_multi, support)
    fitted = fit_prototypes(fit_samples, selected_spec)
    eval_samples = build_samples(support_val, single, full_multi, support)
    eval_rows = [score_sample(sample, fitted, selected_spec, pert_means, support, compute_mmd=True) for sample in eval_samples]
    shuf_rows = [
        score_sample(sample, fitted, selected_spec, pert_means, support, compute_mmd=True, shuffle_delta_seed=args.seed + 77)
        for sample in eval_samples
    ]
    support_summary = summarize(eval_rows, selected_spec, n_boot=args.n_boot, seed=args.seed, wessels_gap=wessels_gap, include_mmd=True)
    shuffled_summary = summarize(shuf_rows, selected_spec, n_boot=args.n_boot, seed=args.seed + 200, wessels_gap=wessels_gap, include_mmd=True)

    payload = {
        "data_dir": str(args.data_dir),
        "split_guard": guard,
        "pert_means_file": str(args.pert_means_file),
        "readout_json": str(args.readout_json),
        "max_cells_per_condition": args.max_cells_per_condition,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_train_multi_rows": len(train_rows),
        "n_support_val_rows": len(support_val),
        "selected_spec": selected,
        "selected_train_summary": next(row for row in train_summaries if row["spec"] == selected),
        "selected_support_summary": support_summary,
        "shuffled_prototype_control": shuffled_summary,
        "train_summaries": train_summaries,
    }
    payload["decision"] = decide(payload)
    payload["status"] = payload["decision"]["status"]
    payload["gpu_authorization"] = payload["decision"]["gpu_authorization"]
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "gpu_authorization": payload["gpu_authorization"], "selected_spec": selected, "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
