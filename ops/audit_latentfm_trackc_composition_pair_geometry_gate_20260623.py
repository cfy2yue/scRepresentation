#!/usr/bin/env python3
"""Query-free Track C pairwise single-response geometry gate.

This gate uses train_single gene response vectors to decide whether to enable
the already-selected no-harm calibrated composition correction.  It is a
different gate from module priors and low-rank subspace transforms: the
hypothesis is that pairwise single-response geometry identifies when additive
composition is safe.  Rule selection uses existing train_multi LOO rows only;
support_val_multi is final scoring only.

Held-out query, canonical test, canonical multi, active logs, and GPU
artifacts are not read.
"""

from __future__ import annotations

import argparse
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
IN_JSON = ROOT / "reports/latentfm_trackc_composition_noharm_calibrated_gate_20260623.json"
DEFAULT_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
OUT_JSON = ROOT / "reports/latentfm_trackc_composition_pair_geometry_gate_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_COMPOSITION_PAIR_GEOMETRY_GATE_20260623.md"
EXPECTED_TRAINSELECT_SHA256 = "5f29dd5b582a40da3736770ca29950c12c54e46b590c3f7705c1d37da89f4f20"


@dataclass(frozen=True)
class Spec:
    name: str
    metric: str
    op: str
    threshold: float
    partial_only: bool


def load_support_module() -> Any:
    spec = importlib.util.spec_from_file_location("trackc_support_route_readiness", SUPPORT_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {SUPPORT_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def genes(row: dict[str, Any]) -> list[str]:
    return [str(g).strip().upper() for g in (row.get("genes") or []) if str(g).strip()]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def geometry_features(row: dict[str, Any], bank: dict[str, np.ndarray]) -> dict[str, float]:
    gs = genes(row)
    vecs = [np.asarray(bank[g], dtype=np.float32) for g in gs if g in bank]
    coverage = float(len(vecs) / max(len(gs), 1))
    if len(vecs) < 2:
        return {
            "raw_vector_coverage": coverage,
            "mean_cosine": 0.0,
            "max_cosine": 0.0,
            "min_cosine": 0.0,
            "mean_abs_cosine": 0.0,
            "norm_balance": 0.0,
            "mean_norm": float(np.mean([np.linalg.norm(v) for v in vecs])) if vecs else 0.0,
        }
    cos = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            cos.append(cosine(vecs[i], vecs[j]))
    norms = [float(np.linalg.norm(v)) for v in vecs]
    return {
        "raw_vector_coverage": coverage,
        "mean_cosine": float(np.mean(cos)),
        "max_cosine": float(max(cos)),
        "min_cosine": float(min(cos)),
        "mean_abs_cosine": float(np.mean([abs(v) for v in cos])),
        "norm_balance": float(min(norms) / max(max(norms), 1e-12)),
        "mean_norm": float(np.mean(norms)),
    }


def shuffled_bank(bank: dict[str, np.ndarray], genes_to_map: list[str], seed: int) -> dict[str, np.ndarray]:
    keys = sorted({g for g in genes_to_map if g in bank})
    vals = [bank[g] for g in keys]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(vals))
    out = dict(bank)
    for i, gene in enumerate(keys):
        out[gene] = vals[int(order[i])]
    return out


def specs() -> list[Spec]:
    out = [
        Spec("always_off_route", "constant", "ge", 1.0, False),
        Spec("always_on_composition", "constant", "ge", 0.0, False),
    ]
    grid = [
        ("raw_vector_coverage", "ge", (1.0,)),
        ("mean_cosine", "ge", (-0.25, 0.0, 0.25, 0.50)),
        ("max_cosine", "ge", (0.0, 0.25, 0.50, 0.75)),
        ("min_cosine", "ge", (-0.50, -0.25, 0.0, 0.25)),
        ("mean_abs_cosine", "le", (0.25, 0.50, 0.75)),
        ("norm_balance", "ge", (0.25, 0.50, 0.75)),
        ("mean_norm", "ge", (0.25, 0.50, 0.75, 1.0)),
    ]
    for metric, op, thresholds in grid:
        for partial_only in (False, True):
            for threshold in thresholds:
                clean = str(threshold).replace("-", "m").replace(".", "p")
                suffix = "partial" if partial_only else "all"
                out.append(Spec(f"{metric}_{op}{clean}_{suffix}", metric, op, float(threshold), partial_only))
    return out


def enabled(row: dict[str, Any], feats: dict[str, float], spec: Spec) -> bool:
    if spec.name == "always_off_route":
        return False
    if spec.name == "always_on_composition":
        return True
    if spec.partial_only and str(row.get("coverage_stratum")) != "partial_raw":
        return False
    val = float(feats[spec.metric])
    if spec.op == "ge":
        return val >= float(spec.threshold)
    if spec.op == "le":
        return val <= float(spec.threshold)
    raise ValueError(spec.op)


def score_rows(rows: list[dict[str, Any]], features: list[dict[str, float]], spec: Spec) -> list[dict[str, Any]]:
    out = []
    for row, feats in zip(rows, features, strict=True):
        use_comp = enabled(row, feats, spec)
        item = dict(row)
        item["pair_geometry_spec"] = spec.name
        item["pair_geometry_enabled"] = bool(use_comp)
        item["candidate_pair_geometry_gated"] = float(row["candidate"]) if use_comp else float(row["support_selected_route"])
        if "candidate__test_mmd_clamped" in row:
            item["candidate_pair_geometry_gated__test_mmd_clamped"] = (
                float(row["candidate__test_mmd_clamped"]) if use_comp else float(row["support_selected_route__test_mmd_clamped"])
            )
        item["feature_value"] = 0.0 if spec.metric == "constant" else float(feats[spec.metric])
        item["features"] = feats
        out.append(item)
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


def find_dataset(summary: dict[str, Any], dataset: str) -> dict[str, Any]:
    for row in summary.get("dataset_breakdown") or []:
        if row.get("dataset") == dataset:
            return row
    return {}


def summarize(rows: list[dict[str, Any]], spec: Spec, *, n_boot: int, seed: int, wessels_gap: float, include_mmd: bool) -> dict[str, Any]:
    pp = paired_bootstrap(rows, "candidate_pair_geometry_gated", "support_selected_route", metric="pp", n_boot=n_boot, seed=seed)
    mmd = (
        paired_bootstrap(rows, "candidate_pair_geometry_gated", "support_selected_route", metric="mmd_clamped", n_boot=n_boot, seed=seed + 100)
        if include_mmd and rows and "candidate_pair_geometry_gated__test_mmd_clamped" in rows[0]
        else None
    )
    ds_pp = dataset_delta(rows, "candidate_pair_geometry_gated", "support_selected_route")
    ds_mmd = dataset_delta(rows, "candidate_pair_geometry_gated__test_mmd_clamped", "support_selected_route__test_mmd_clamped") if mmd else {}
    breakdown = []
    for ds in sorted({str(row["dataset"]) for row in rows}):
        sub = [row for row in rows if str(row["dataset"]) == ds]
        delta = ds_pp.get(ds)
        gap = wessels_gap if ds == "Wessels" else None
        breakdown.append(
            {
                "dataset": ds,
                "n_conditions": len(sub),
                "enabled_fraction": float(np.mean([bool(row["pair_geometry_enabled"]) for row in sub])),
                "delta_pp": delta,
                "delta_mmd_clamped": ds_mmd.get(ds),
                "route_gap_pp": gap,
                "route_gap_closed_fraction": None if gap is None or abs(gap) <= 1e-12 or delta is None else float(delta / gap),
            }
        )
    return {
        "spec": spec.name,
        "spec_params": spec.__dict__,
        "paired_pp_delta": pp,
        "paired_mmd_delta": mmd,
        "dataset_breakdown": breakdown,
        "enabled_fraction": float(np.mean([bool(row["pair_geometry_enabled"]) for row in rows])) if rows else 0.0,
        "rows": rows,
    }


def select_spec(train_summaries: list[dict[str, Any]]) -> str:
    eligible = []
    for row in train_summaries:
        w = find_dataset(row, "Wessels")
        n = find_dataset(row, "NormanWeissman2019_filtered")
        pp = row["paired_pp_delta"]
        if (
            float(w.get("delta_pp") if w.get("delta_pp") is not None else -999.0) >= 0.02
            and float(n.get("delta_pp") if n.get("delta_pp") is not None else -999.0) >= -0.01
            and float(pp.get("p_harm") if pp.get("p_harm") is not None else 1.0) <= 0.20
            and float(row.get("enabled_fraction") or 0.0) > 0.0
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
    shuffled = payload["shuffled_geometry_control"]
    pp = support["paired_pp_delta"]
    mmd = support["paired_mmd_delta"] or {}
    w = find_dataset(support, "Wessels")
    n = find_dataset(support, "NormanWeissman2019_filtered")
    if payload["split_guard"].get("sha256") != EXPECTED_TRAINSELECT_SHA256:
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
    real_delta = float(pp.get("delta_mean") if pp.get("delta_mean") is not None else 0.0)
    shuf_delta = float(shuffled["paired_pp_delta"].get("delta_mean") if shuffled["paired_pp_delta"].get("delta_mean") is not None else 0.0)
    if shuf_delta > real_delta - 0.02:
        reasons.append("shuffled_geometry_control_does_not_collapse")
    status = "trackc_composition_pair_geometry_gate_pass_authorize_one_capped_gpu_smoke" if not reasons else "trackc_composition_pair_geometry_gate_fail_no_gpu"
    return {
        "status": status,
        "gpu_authorization": "one_capped_trackc_support_only_smoke" if not reasons else "none",
        "reasons": reasons,
    }


def render(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    support = payload["selected_support_summary"]
    shuffled = payload["shuffled_geometry_control"]
    lines = [
        "# Track C Composition Pair Geometry Gate",
        "",
        f"Status: `{decision['status']}`",
        f"GPU authorization: `{decision['gpu_authorization']}`",
        "",
        "## Provenance",
        "",
        f"- source rows: `{payload['input_json']}`",
        f"- split SHA256: `{payload['split_guard'].get('sha256')}`",
        f"- data_dir: `{payload['data_dir']}`",
        f"- leakage_status: `{payload['boundary']}`",
        f"- selected spec: `{payload['selected_spec']}`",
        "",
        "## Gate Criteria",
        "",
        f"- Wessels pp delta: `{fmt(find_dataset(support, 'Wessels').get('delta_pp'))}` (gate `>= +0.020000`)",
        f"- Wessels route-gap closure: `{fmt(find_dataset(support, 'Wessels').get('route_gap_closed_fraction'))}` (gate `>= +0.050000`)",
        f"- Norman pp delta: `{fmt(find_dataset(support, 'NormanWeissman2019_filtered').get('delta_pp'))}` (gate `>= -0.010000`)",
        f"- bootstrap pp p_harm: `{fmt(support['paired_pp_delta'].get('p_harm'))}` (gate `<= 0.200000`)",
        f"- MMD delta: `{fmt((support['paired_mmd_delta'] or {}).get('delta_mean'))}` (hard-harm gate `<= +0.005000`)",
        f"- shuffled geometry pp delta: `{fmt(shuffled['paired_pp_delta'].get('delta_mean'))}` (must be at least `0.020000` below real)",
        "",
        "## Support-Val Dataset Breakdown",
        "",
        "| dataset | n | enabled fraction | pp delta | MMD delta | route-gap closure |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in support["dataset_breakdown"]:
        lines.append(
            f"| {row['dataset']} | {row['n_conditions']} | {fmt(row.get('enabled_fraction'))} | "
            f"{fmt(row.get('delta_pp'))} | {fmt(row.get('delta_mmd_clamped'))} | {fmt(row.get('route_gap_closed_fraction'))} |"
        )
    lines.extend(["", "## Train-Only Selection Summary", "", "| spec | enabled | pp delta | Norman delta | Wessels delta | p_harm |", "|---|---:|---:|---:|---:|---:|"])
    for row in payload["train_summaries"][:15]:
        marker = " (selected)" if row["spec"] == payload["selected_spec"] else ""
        lines.append(
            f"| `{row['spec']}`{marker} | {fmt(row.get('enabled_fraction'))} | {fmt(row['paired_pp_delta'].get('delta_mean'))} | "
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
            "- Failure closes this pair-geometry gate unless a new train-only geometry feature family is introduced.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=IN_JSON)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--max-cells-per-condition", type=int, default=256)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    args = parser.parse_args()

    support = load_support_module()
    payload_in = load_json(args.input_json)
    split = support.load_json(args.split_file)
    manifest = support.load_json(args.data_dir / "manifest.json")
    metadata = support.load_json(Path(manifest["condition_metadata_file"]))
    single = support.train_single_components(args.data_dir, split, metadata, max_cells=args.max_cells_per_condition)
    bank = {str(k).upper(): np.asarray(v, dtype=np.float32) for k, v in (single.get("gene_raw_mean") or {}).items()}
    train_rows = payload_in["selected_train_summary"]["rows"]
    support_rows = payload_in["support_val_summary"]["rows"]
    wessels_gap = 0.15360754709757318

    train_features = [geometry_features(row, bank) for row in train_rows]
    support_features = [geometry_features(row, bank) for row in support_rows]
    all_specs = specs()
    train_summaries = []
    train_by_spec = {}
    for spec in all_specs:
        scored = score_rows(train_rows, train_features, spec)
        train_by_spec[spec.name] = scored
        train_summaries.append(summarize(scored, spec, n_boot=args.n_boot, seed=args.seed, wessels_gap=wessels_gap, include_mmd=False))
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
    support_scored = score_rows(support_rows, support_features, selected_spec)
    support_summary = summarize(support_scored, selected_spec, n_boot=args.n_boot, seed=args.seed, wessels_gap=wessels_gap, include_mmd=True)

    all_genes = [g for row in train_rows + support_rows for g in genes(row)]
    shuf = shuffled_bank(bank, all_genes, args.seed + 41)
    support_shuf_features = [geometry_features(row, shuf) for row in support_rows]
    shuf_scored = score_rows(support_rows, support_shuf_features, selected_spec)
    shuf_summary = summarize(shuf_scored, selected_spec, n_boot=args.n_boot, seed=args.seed + 300, wessels_gap=wessels_gap, include_mmd=True)

    out = {
        "input_json": str(args.input_json),
        "data_dir": str(args.data_dir),
        "boundary": "train_single_gene_vectors_plus_existing_noharm_train_multi_loo_rows_for_selection_support_val_final_once_no_query_no_canonical_outputs",
        "split_guard": payload_in.get("split_guard", {}),
        "n_train_rows": len(train_rows),
        "n_support_rows": len(support_rows),
        "selected_spec": selected,
        "selected_train_summary": next(row for row in train_summaries if row["spec"] == selected),
        "selected_support_summary": support_summary,
        "shuffled_geometry_control": shuf_summary,
        "train_summaries": train_summaries,
    }
    out["decision"] = decide(out)
    out["status"] = out["decision"]["status"]
    out["gpu_authorization"] = out["decision"]["gpu_authorization"]
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render(out), encoding="utf-8")
    print(json.dumps({"status": out["status"], "gpu_authorization": out["gpu_authorization"], "selected_spec": selected, "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
