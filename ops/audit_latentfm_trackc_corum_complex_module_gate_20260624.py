#!/usr/bin/env python3
"""Track C CORUM complex-module gated composition CPU gate."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
IN_JSON = REPORTS / "latentfm_trackc_composition_noharm_calibrated_gate_20260623.json"
CORUM_COMPLEXES = ROOT / "dataset/external_priors/corum_complexes_20260624/corum_human_complexes_normalized.tsv"
CORUM_SUMMARY = ROOT / "dataset/external_priors/corum_complexes_20260624/corum_human_complex_prior_summary.json"
OUT_JSON = REPORTS / "latentfm_trackc_corum_complex_module_gate_20260624.json"
OUT_MD = REPORTS / "LATENTFM_TRACKC_CORUM_COMPLEX_MODULE_GATE_20260624.md"
EXPECTED_TRAINSELECT_SHA256 = "5f29dd5b582a40da3736770ca29950c12c54e46b590c3f7705c1d37da89f4f20"


@dataclass(frozen=True)
class Spec:
    name: str
    metric: str
    threshold: float
    partial_only: bool


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


def split_semicolon(value: str) -> list[str]:
    return [x.strip() for x in str(value or "").split(";") if x.strip()]


def load_corum(path: Path) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    gene_to_complexes: dict[str, set[str]] = defaultdict(set)
    gene_to_fcg: dict[str, set[str]] = defaultdict(set)
    complex_to_genes: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            cid = str(row["complex_id"])
            gs = {g.upper() for g in split_semicolon(row.get("genes", ""))}
            fcgs = {x for x in split_semicolon(row.get("fcgs_name", "")) + split_semicolon(row.get("fcgs_category_name", ""))}
            if len(gs) < 2:
                continue
            complex_to_genes[cid] = gs
            for gene in gs:
                gene_to_complexes[gene].add(cid)
                gene_to_fcg[gene].update(fcgs)
    return gene_to_complexes, complex_to_genes, gene_to_fcg


def shuffled_sets(source: dict[str, set[str]], genes_to_map: list[str], seed: int) -> dict[str, set[str]]:
    keys = sorted({g for g in genes_to_map if g in source})
    vals = [source[g] for g in keys]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(vals))
    out = dict(source)
    for i, gene in enumerate(keys):
        out[gene] = set(vals[int(order[i])])
    return out


def features_for(row: dict[str, Any], gene_to_complexes: dict[str, set[str]], gene_to_fcg: dict[str, set[str]]) -> dict[str, float]:
    gs = genes(row)
    covered = [g for g in gs if gene_to_complexes.get(g)]
    shared_counts = []
    shared_jaccards = []
    fcg_shared_counts = []
    degrees = []
    for g in gs:
        degrees.append(len(gene_to_complexes.get(g, set())))
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            a = gene_to_complexes.get(gs[i], set())
            b = gene_to_complexes.get(gs[j], set())
            fa = gene_to_fcg.get(gs[i], set())
            fb = gene_to_fcg.get(gs[j], set())
            if a or b:
                inter = len(a & b)
                union = len(a | b)
                shared_counts.append(float(inter))
                shared_jaccards.append(float(inter / max(union, 1)))
            if fa or fb:
                fcg_shared_counts.append(float(len(fa & fb)))
    return {
        "corum_gene_coverage": float(len(covered) / max(len(gs), 1)),
        "corum_min_gene_degree": float(min(degrees)) if degrees else 0.0,
        "corum_mean_gene_degree": float(np.mean(degrees)) if degrees else 0.0,
        "corum_max_shared_complexes": float(max(shared_counts)) if shared_counts else 0.0,
        "corum_max_shared_jaccard": float(max(shared_jaccards)) if shared_jaccards else 0.0,
        "corum_max_shared_fcg": float(max(fcg_shared_counts)) if fcg_shared_counts else 0.0,
    }


def specs() -> list[Spec]:
    out = [Spec("always_off_route", "constant_off", 1.0, False), Spec("always_on_composition", "constant_on", 0.0, False)]
    grid = [
        ("corum_max_shared_complexes", (1.0, 2.0, 3.0)),
        ("corum_max_shared_jaccard", (0.01, 0.05, 0.10, 0.20)),
        ("corum_gene_coverage", (0.5, 1.0)),
        ("corum_min_gene_degree", (1.0, 2.0, 5.0, 10.0)),
        ("corum_max_shared_fcg", (1.0, 2.0)),
    ]
    for metric, thresholds in grid:
        for partial_only in (False, True):
            for threshold in thresholds:
                suffix = "partial" if partial_only else "all"
                clean = str(threshold).replace(".", "p")
                out.append(Spec(f"{metric}_ge{clean}_{suffix}", metric, float(threshold), partial_only))
    return out


def enabled(row: dict[str, Any], feats: dict[str, float], spec: Spec) -> bool:
    if spec.name == "always_off_route":
        return False
    if spec.name == "always_on_composition":
        return True
    if spec.partial_only and str(row.get("coverage_stratum")) != "partial_raw":
        return False
    return float(feats.get(spec.metric, 0.0)) >= spec.threshold


def score_rows(rows: list[dict[str, Any]], feature_rows: list[dict[str, float]], spec: Spec) -> list[dict[str, Any]]:
    out = []
    for row, feats in zip(rows, feature_rows, strict=True):
        use_comp = enabled(row, feats, spec)
        item = dict(row)
        item["corum_gate_enabled"] = bool(use_comp)
        item["corum_gate_spec"] = spec.name
        item["candidate_corum_gated"] = float(row["candidate"]) if use_comp else float(row["support_selected_route"])
        if "candidate__test_mmd_clamped" in row:
            item["candidate_corum_gated__test_mmd_clamped"] = (
                float(row["candidate__test_mmd_clamped"]) if use_comp else float(row["support_selected_route__test_mmd_clamped"])
            )
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
        sample_ds = rng.choice(datasets, size=len(datasets), replace=True)
        vals = []
        for ds in sample_ds:
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


def summarize(rows: list[dict[str, Any]], spec: Spec, *, n_boot: int, seed: int, wessels_gap: float | None, include_mmd: bool) -> dict[str, Any]:
    pp = paired_bootstrap(rows, "candidate_corum_gated", "support_selected_route", metric="pp", n_boot=n_boot, seed=seed)
    mmd = (
        paired_bootstrap(rows, "candidate_corum_gated", "support_selected_route", metric="mmd_clamped", n_boot=n_boot, seed=seed + 100)
        if include_mmd and rows and "candidate_corum_gated__test_mmd_clamped" in rows[0]
        else None
    )
    ds_pp = dataset_delta(rows, "candidate_corum_gated", "support_selected_route")
    ds_mmd = dataset_delta(rows, "candidate_corum_gated__test_mmd_clamped", "support_selected_route__test_mmd_clamped") if mmd else {}
    breakdown = []
    for ds in sorted({str(row["dataset"]) for row in rows}):
        sub = [row for row in rows if str(row["dataset"]) == ds]
        delta = ds_pp.get(ds)
        gap = wessels_gap if ds == "Wessels" else None
        breakdown.append(
            {
                "dataset": ds,
                "n_conditions": len(sub),
                "enabled_fraction": float(np.mean([bool(row["corum_gate_enabled"]) for row in sub])),
                "delta_pp": delta,
                "delta_mmd_clamped": ds_mmd.get(ds),
                "route_gap_pp": gap,
                "route_gap_closed_fraction": None if gap is None or abs(gap) <= 1e-12 or delta is None else float(delta / gap),
            }
        )
    return {"spec": spec.name, "spec_params": spec.__dict__, "paired_pp_delta": pp, "paired_mmd_delta": mmd, "dataset_breakdown": breakdown, "enabled_fraction": float(np.mean([bool(row["corum_gate_enabled"]) for row in rows])) if rows else 0.0, "rows": rows}


def find_dataset(summary: dict[str, Any], dataset: str) -> dict[str, Any]:
    for row in summary.get("dataset_breakdown") or []:
        if row.get("dataset") == dataset:
            return row
    return {}


def read_wessels_gap(payload: dict[str, Any]) -> float | None:
    for row in (payload.get("support_val_summary") or {}).get("dataset_breakdown") or []:
        if row.get("dataset") == "Wessels" and row.get("route_gap_pp") is not None:
            return float(row["route_gap_pp"])
    return 0.1536066201897797


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
    return str(sorted(pool, key=lambda row: (float(find_dataset(row, "Wessels").get("delta_pp") or -999.0), float(find_dataset(row, "NormanWeissman2019_filtered").get("delta_pp") or -999.0), float(row["paired_pp_delta"].get("delta_mean") or -999.0)), reverse=True)[0]["spec"])


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    split = payload.get("split_guard") or {}
    support = payload["selected_support_summary"]
    shuffled = payload["shuffled_prior_control"]
    pp = support["paired_pp_delta"]
    mmd = support["paired_mmd_delta"] or {}
    w = find_dataset(support, "Wessels")
    n = find_dataset(support, "NormanWeissman2019_filtered")
    if split.get("sha256") != EXPECTED_TRAINSELECT_SHA256:
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
        reasons.append("shuffled_corum_control_does_not_collapse")
    status = "trackc_corum_complex_module_gate_pass_authorize_one_capped_gpu_smoke" if not reasons else "trackc_corum_complex_module_gate_fail_no_gpu"
    return {"status": status, "gpu_authorization": "one_capped_trackc_support_only_smoke" if not reasons else "none", "reasons": reasons}


def render(payload: dict[str, Any]) -> str:
    support = payload["selected_support_summary"]
    shuffled = payload["shuffled_prior_control"]
    lines = [
        "# Track C CORUM Complex-Module Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"GPU authorization: `{payload['decision']['gpu_authorization']}`",
        "",
        "## Boundary",
        "",
        "- Uses only safe trainselect rows from the existing no-harm calibrated composition gate.",
        "- Train_multi leave-one-condition rows select the CORUM rule; support_val_multi is final scoring.",
        "- Does not read held-out query, canonical test, canonical multi, active logs, or GPU artifacts.",
        "",
        "## Provenance",
        "",
        f"- CORUM summary: `{payload['corum_summary_path']}`",
        f"- CORUM release: `{payload['corum_summary'].get('source', {}).get('release')}`",
        f"- split SHA256: `{payload['split_guard'].get('sha256')}`",
        f"- selected spec: `{payload['selected_spec']}`",
        "",
        "## Gate Criteria",
        "",
        f"- Wessels pp delta: `{fmt(find_dataset(support, 'Wessels').get('delta_pp'))}`",
        f"- Wessels route-gap closure: `{fmt(find_dataset(support, 'Wessels').get('route_gap_closed_fraction'))}`",
        f"- Norman pp delta: `{fmt(find_dataset(support, 'NormanWeissman2019_filtered').get('delta_pp'))}`",
        f"- bootstrap pp p_harm: `{fmt(support['paired_pp_delta'].get('p_harm'))}`",
        f"- MMD delta: `{fmt((support['paired_mmd_delta'] or {}).get('delta_mean'))}`",
        f"- shuffled CORUM pp delta: `{fmt(shuffled['paired_pp_delta'].get('delta_mean'))}`",
        "",
        "## Support-Val Breakdown",
        "",
        "| dataset | n | enabled | pp delta | MMD delta | closure |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in support["dataset_breakdown"]:
        lines.append(f"| {row['dataset']} | {row['n_conditions']} | {fmt(row.get('enabled_fraction'))} | {fmt(row.get('delta_pp'))} | {fmt(row.get('delta_mmd_clamped'))} | {fmt(row.get('route_gap_closed_fraction'))} |")
    lines.extend(["", "## Train Selection Top Rows", "", "| spec | enabled | pp delta | Norman | Wessels | p_harm |", "|---|---:|---:|---:|---:|---:|"])
    for row in payload["train_summaries"][:20]:
        marker = " (selected)" if row["spec"] == payload["selected_spec"] else ""
        lines.append(f"| `{row['spec']}`{marker} | {fmt(row.get('enabled_fraction'))} | {fmt(row['paired_pp_delta'].get('delta_mean'))} | {fmt(find_dataset(row, 'NormanWeissman2019_filtered').get('delta_pp'))} | {fmt(find_dataset(row, 'Wessels').get('delta_pp'))} | {fmt(row['paired_pp_delta'].get('p_harm'))} |")
    lines.extend(["", "## Decision Reasons", ""])
    lines.extend([f"- `{r}`" for r in payload["decision"].get("reasons", [])] or ["- none"])
    return "\n".join(lines) + "\n"


def main() -> int:
    payload_in = load_json(IN_JSON)
    train_rows = payload_in["selected_train_summary"]["rows"]
    support_rows = payload_in["support_val_summary"]["rows"]
    gene_to_complexes, _complex_to_genes, gene_to_fcg = load_corum(CORUM_COMPLEXES)
    all_row_genes = [g for row in train_rows + support_rows for g in genes(row)]
    train_features = [features_for(row, gene_to_complexes, gene_to_fcg) for row in train_rows]
    support_features = [features_for(row, gene_to_complexes, gene_to_fcg) for row in support_rows]
    train_summaries = []
    train_rows_by_spec = {}
    wessels_gap = read_wessels_gap(payload_in)
    for spec in specs():
        scored = score_rows(train_rows, train_features, spec)
        train_rows_by_spec[spec.name] = scored
        train_summaries.append(summarize(scored, spec, n_boot=2000, seed=20260624, wessels_gap=wessels_gap, include_mmd=False))
    selected = select_spec(train_summaries)
    selected_spec = next(s for s in specs() if s.name == selected)
    support_scored = score_rows(support_rows, support_features, selected_spec)
    support_summary = summarize(support_scored, selected_spec, n_boot=2000, seed=20260625, wessels_gap=wessels_gap, include_mmd=True)
    shuffled_complexes = shuffled_sets(gene_to_complexes, all_row_genes, 20260624 + 99)
    shuffled_fcg = shuffled_sets(gene_to_fcg, all_row_genes, 20260624 + 199)
    shuffled_features = [features_for(row, shuffled_complexes, shuffled_fcg) for row in support_rows]
    shuffled_scored = score_rows(support_rows, shuffled_features, selected_spec)
    shuffled_summary = summarize(shuffled_scored, selected_spec, n_boot=2000, seed=20260626, wessels_gap=wessels_gap, include_mmd=True)
    payload = {
        "status": "pending",
        "timestamp": "2026-06-24 00:35 CST",
        "input_json": str(IN_JSON),
        "corum_complexes": str(CORUM_COMPLEXES),
        "corum_summary_path": str(CORUM_SUMMARY),
        "corum_summary": load_json(CORUM_SUMMARY),
        "boundary": {"safe_trainselect_only": True, "train_multi_loo_selection_only": True, "support_val_final_scoring_only": True, "heldout_query_read": False, "canonical_test_read": False, "canonical_multi_read": False, "active_log_read": False, "gpu_artifact_read": False},
        "split_guard": payload_in.get("split_guard"),
        "selected_spec": selected,
        "train_summaries": sorted(train_summaries, key=lambda row: (float(find_dataset(row, "Wessels").get("delta_pp") or -999.0), float(find_dataset(row, "NormanWeissman2019_filtered").get("delta_pp") or -999.0), float(row["paired_pp_delta"].get("delta_mean") or -999.0)), reverse=True),
        "selected_train_rows": train_rows_by_spec[selected],
        "selected_support_summary": support_summary,
        "shuffled_prior_control": shuffled_summary,
    }
    payload["decision"] = decide(payload)
    payload["status"] = payload["decision"]["status"]
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "gpu_authorization": payload["decision"]["gpu_authorization"], "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
