#!/usr/bin/env python3
"""Query-free Track C module-coherence gated composition CPU gate.

This gate tests a materially different way to protect the composition
near-miss: only enable the already-selected no-harm calibrated composition
correction when frozen external priors indicate that the perturbation genes are
module-coherent.  Rule selection uses existing train_multi leave-one-condition
rows only; support_val_multi is final scoring only.

Held-out query, canonical test, canonical multi, active logs, and GPU
artifacts are not read.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
IN_JSON = ROOT / "reports/latentfm_trackc_composition_noharm_calibrated_gate_20260623.json"
REACTOME_TSV = ROOT / "dataset/external_priors/reactome_pathways_current_20260623/reactome_gene_pathways.tsv"
GOA_TSV = ROOT / "dataset/external_priors/goa_human_20260519/goa_human_gene_terms.tsv"
OMNIPATH_TSV = ROOT / "dataset/external_priors/omnipath_tf_20260623/omnipath_tf_target_gene_features.tsv"
OUT_JSON = ROOT / "reports/latentfm_trackc_composition_module_coherence_gate_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_COMPOSITION_MODULE_COHERENCE_GATE_20260623.md"
EXPECTED_TRAINSELECT_SHA256 = "5f29dd5b582a40da3736770ca29950c12c54e46b590c3f7705c1d37da89f4f20"


@dataclass(frozen=True)
class Spec:
    name: str
    source: str
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


def load_gene_sets(path: Path, term_col: str, sep: str = ";") -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            gene = str(row.get("gene") or "").strip().upper()
            if not gene:
                continue
            terms = {str(x).strip() for x in str(row.get(term_col) or "").split(sep) if str(x).strip()}
            if terms:
                out[gene] = terms
    return out


def load_omnipath(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            gene = str(row.get("gene") or "").strip().upper()
            if not gene:
                continue
            vals = {}
            for key, value in row.items():
                if key == "gene":
                    continue
                try:
                    vals[key] = float(value)
                except (TypeError, ValueError):
                    vals[key] = 0.0
            out[gene] = vals
    return out


def shuffled_sets(sets: dict[str, set[str]], genes_to_map: list[str], seed: int) -> dict[str, set[str]]:
    keys = sorted({g for g in genes_to_map if g in sets})
    vals = [sets[g] for g in keys]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(vals))
    out = dict(sets)
    for i, gene in enumerate(keys):
        out[gene] = set(vals[int(order[i])])
    return out


def shuffled_omni(omni: dict[str, dict[str, float]], genes_to_map: list[str], seed: int) -> dict[str, dict[str, float]]:
    keys = sorted({g for g in genes_to_map if g in omni})
    vals = [omni[g] for g in keys]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(vals))
    out = dict(omni)
    for i, gene in enumerate(keys):
        out[gene] = dict(vals[int(order[i])])
    return out


def pair_scores(row: dict[str, Any], term_sets: dict[str, set[str]]) -> dict[str, float]:
    gs = genes(row)
    covered = [g for g in gs if g in term_sets]
    vals_j = []
    vals_o = []
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            a = term_sets.get(gs[i], set())
            b = term_sets.get(gs[j], set())
            if not a or not b:
                continue
            vals_o.append(float(len(a & b)))
            vals_j.append(float(len(a & b) / max(len(a | b), 1)))
    return {
        "coverage": float(len(covered) / max(len(gs), 1)),
        "max_overlap": float(max(vals_o)) if vals_o else 0.0,
        "mean_overlap": float(np.mean(vals_o)) if vals_o else 0.0,
        "max_jaccard": float(max(vals_j)) if vals_j else 0.0,
        "mean_jaccard": float(np.mean(vals_j)) if vals_j else 0.0,
    }


def omni_scores(row: dict[str, Any], omni: dict[str, dict[str, float]]) -> dict[str, float]:
    gs = genes(row)
    vals = [omni.get(g) for g in gs if omni.get(g) is not None]
    if not vals:
        return {"coverage": 0.0, "tf_or_target_coverage": 0.0, "max_log_degree_diff": 0.0, "mean_log_degree": 0.0}
    degrees = []
    active = 0
    for vals_i in vals:
        deg = float(vals_i.get("tf_out_degree", 0.0)) + float(vals_i.get("target_in_degree", 0.0))
        degrees.append(np.log1p(max(deg, 0.0)))
        active += int(deg > 0.0)
    diffs = []
    for i in range(len(degrees)):
        for j in range(i + 1, len(degrees)):
            diffs.append(abs(degrees[i] - degrees[j]))
    return {
        "coverage": float(len(vals) / max(len(gs), 1)),
        "tf_or_target_coverage": float(active / max(len(gs), 1)),
        "max_log_degree_diff": float(max(diffs)) if diffs else 0.0,
        "mean_log_degree": float(np.mean(degrees)) if degrees else 0.0,
    }


def make_features(row: dict[str, Any], reactome: dict[str, set[str]], goa: dict[str, set[str]], omni: dict[str, dict[str, float]]) -> dict[str, float]:
    r = pair_scores(row, reactome)
    g = pair_scores(row, goa)
    o = omni_scores(row, omni)
    out = {}
    for key, value in r.items():
        out[f"reactome_{key}"] = value
    for key, value in g.items():
        out[f"goa_{key}"] = value
    for key, value in o.items():
        out[f"omni_{key}"] = value
    out["either_max_overlap"] = max(out["reactome_max_overlap"], out["goa_max_overlap"])
    out["either_max_jaccard"] = max(out["reactome_max_jaccard"], out["goa_max_jaccard"])
    out["both_prior_coverage"] = min(out["reactome_coverage"], out["goa_coverage"])
    return out


def specs() -> list[Spec]:
    out = [Spec("always_off_route", "constant", "always", 1.0, False), Spec("always_on_composition", "constant", "always", 0.0, False)]
    grid = [
        ("reactome", "max_overlap", (1.0, 2.0, 3.0)),
        ("reactome", "max_jaccard", (0.01, 0.03, 0.05, 0.10)),
        ("goa", "max_overlap", (1.0, 2.0, 3.0, 5.0)),
        ("goa", "max_jaccard", (0.01, 0.03, 0.05, 0.10)),
        ("either", "max_overlap", (1.0, 2.0, 3.0, 5.0)),
        ("either", "max_jaccard", (0.01, 0.03, 0.05, 0.10)),
        ("omni", "tf_or_target_coverage", (0.5, 1.0)),
        ("omni", "mean_log_degree", (1.0, 2.0, 3.0)),
    ]
    for source, metric, thresholds in grid:
        for partial_only in (False, True):
            for threshold in thresholds:
                suffix = "partial" if partial_only else "all"
                clean = str(threshold).replace(".", "p")
                out.append(Spec(f"{source}_{metric}_ge{clean}_{suffix}", source, metric, float(threshold), partial_only))
    return out


def feature_value(features: dict[str, float], spec: Spec) -> float:
    if spec.source == "constant":
        return 0.0
    if spec.source == "reactome":
        return float(features[f"reactome_{spec.metric}"])
    if spec.source == "goa":
        return float(features[f"goa_{spec.metric}"])
    if spec.source == "either":
        return float(features[f"either_{spec.metric}"])
    if spec.source == "omni":
        return float(features[f"omni_{spec.metric}"])
    raise ValueError(spec.source)


def enabled(row: dict[str, Any], features: dict[str, float], spec: Spec) -> bool:
    if spec.name == "always_off_route":
        return False
    if spec.name == "always_on_composition":
        return True
    if spec.partial_only and str(row.get("coverage_stratum")) != "partial_raw":
        return False
    return feature_value(features, spec) >= float(spec.threshold)


def score_rows(rows: list[dict[str, Any]], feature_rows: list[dict[str, float]], spec: Spec) -> list[dict[str, Any]]:
    out = []
    for row, feats in zip(rows, feature_rows, strict=True):
        use_comp = enabled(row, feats, spec)
        item = dict(row)
        item["module_gate_enabled"] = bool(use_comp)
        item["module_gate_spec"] = spec.name
        item["candidate_module_gated"] = float(row["candidate"]) if use_comp else float(row["support_selected_route"])
        if "candidate__test_mmd_clamped" in row:
            item["candidate_module_gated__test_mmd_clamped"] = (
                float(row["candidate__test_mmd_clamped"]) if use_comp else float(row["support_selected_route__test_mmd_clamped"])
            )
        item["feature_value"] = feature_value(feats, spec)
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


def summarize(rows: list[dict[str, Any]], spec: Spec, *, n_boot: int, seed: int, wessels_gap: float | None, include_mmd: bool) -> dict[str, Any]:
    pp = paired_bootstrap(rows, "candidate_module_gated", "support_selected_route", metric="pp", n_boot=n_boot, seed=seed)
    mmd = (
        paired_bootstrap(rows, "candidate_module_gated", "support_selected_route", metric="mmd_clamped", n_boot=n_boot, seed=seed + 100)
        if include_mmd and rows and "candidate_module_gated__test_mmd_clamped" in rows[0]
        else None
    )
    ds_pp = dataset_delta(rows, "candidate_module_gated", "support_selected_route")
    ds_mmd = dataset_delta(rows, "candidate_module_gated__test_mmd_clamped", "support_selected_route__test_mmd_clamped") if mmd else {}
    breakdown = []
    for ds in sorted({str(row["dataset"]) for row in rows}):
        sub = [row for row in rows if str(row["dataset"]) == ds]
        delta = ds_pp.get(ds)
        gap = wessels_gap if ds == "Wessels" else None
        breakdown.append(
            {
                "dataset": ds,
                "n_conditions": len(sub),
                "enabled_fraction": float(np.mean([bool(row["module_gate_enabled"]) for row in sub])),
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
        "enabled_fraction": float(np.mean([bool(row["module_gate_enabled"]) for row in rows])) if rows else 0.0,
        "rows": rows,
    }


def find_dataset(summary: dict[str, Any], dataset: str) -> dict[str, Any]:
    for row in summary.get("dataset_breakdown") or []:
        if row.get("dataset") == dataset:
            return row
    return {}


def read_wessels_gap(payload: dict[str, Any]) -> float | None:
    support = payload.get("support_val_summary") or {}
    for row in support.get("dataset_breakdown") or []:
        if row.get("dataset") == "Wessels":
            gap = row.get("route_gap_pp")
            if gap is not None:
                return float(gap)
    # no-harm calibrated report stores route_gap in support rows only for some
    # upstream gates; use the known previous support route gap if absent.
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
    shuffled_pp = shuffled["paired_pp_delta"]
    real_delta = float(pp.get("delta_mean") if pp.get("delta_mean") is not None else 0.0)
    shuf_delta = float(shuffled_pp.get("delta_mean") if shuffled_pp.get("delta_mean") is not None else 0.0)
    if shuf_delta > real_delta - 0.02:
        reasons.append("shuffled_prior_control_does_not_collapse")
    status = "trackc_composition_module_coherence_gate_pass_authorize_one_capped_gpu_smoke" if not reasons else "trackc_composition_module_coherence_gate_fail_no_gpu"
    return {
        "status": status,
        "gpu_authorization": "one_capped_trackc_support_only_smoke" if not reasons else "none",
        "reasons": reasons,
    }


def render(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    selected = payload["selected_spec"]
    support = payload["selected_support_summary"]
    shuffled = payload["shuffled_prior_control"]
    lines = [
        "# Track C Composition Module-Coherence Gate",
        "",
        f"Status: `{decision['status']}`",
        f"GPU authorization: `{decision['gpu_authorization']}`",
        "",
        "## Hypothesis",
        "",
        "The composition correction has real Wessels signal, but it should only be enabled when frozen external priors indicate module coherence between perturbation genes.",
        "",
        "## Provenance",
        "",
        f"- source rows: `{payload['input_json']}`",
        f"- split SHA256: `{payload['split_guard'].get('sha256')}`",
        f"- Reactome TSV: `{payload['reactome_tsv']}`",
        f"- GOA TSV: `{payload['goa_tsv']}`",
        f"- OmniPath TSV: `{payload['omnipath_tsv']}`",
        f"- leakage_status: `{payload['boundary']}`",
        f"- selected spec: `{selected}`",
        "",
        "## Gate Criteria",
        "",
        f"- Wessels pp delta: `{fmt(find_dataset(support, 'Wessels').get('delta_pp'))}` (gate `>= +0.020000`)",
        f"- Wessels route-gap closure: `{fmt(find_dataset(support, 'Wessels').get('route_gap_closed_fraction'))}` (gate `>= +0.050000`)",
        f"- Norman pp delta: `{fmt(find_dataset(support, 'NormanWeissman2019_filtered').get('delta_pp'))}` (gate `>= -0.010000`)",
        f"- bootstrap pp p_harm: `{fmt(support['paired_pp_delta'].get('p_harm'))}` (gate `<= 0.200000`)",
        f"- MMD delta: `{fmt((support['paired_mmd_delta'] or {}).get('delta_mean'))}` (hard-harm gate `<= +0.005000`)",
        f"- shuffled prior pp delta: `{fmt(shuffled['paired_pp_delta'].get('delta_mean'))}` (must be at least `0.020000` below real)",
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
    lines.extend(
        [
            "",
            "## Train-Only Selection Summary",
            "",
            "| spec | enabled | pp delta | Norman delta | Wessels delta | p_harm |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["train_summaries"][:15]:
        marker = " (selected)" if row["spec"] == selected else ""
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
            "- Failure closes this module-coherence gate unless a new frozen prior or predeclared feature family is introduced.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=IN_JSON)
    parser.add_argument("--reactome-tsv", type=Path, default=REACTOME_TSV)
    parser.add_argument("--goa-tsv", type=Path, default=GOA_TSV)
    parser.add_argument("--omnipath-tsv", type=Path, default=OMNIPATH_TSV)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    args = parser.parse_args()

    payload_in = load_json(args.input_json)
    train_rows = payload_in["selected_train_summary"]["rows"]
    support_rows = payload_in["support_val_summary"]["rows"]
    reactome = load_gene_sets(args.reactome_tsv, "reactome_pathways")
    goa = load_gene_sets(args.goa_tsv, "go_terms")
    omni = load_omnipath(args.omnipath_tsv)
    all_row_genes = [g for row in train_rows + support_rows for g in genes(row)]
    train_features = [make_features(row, reactome, goa, omni) for row in train_rows]
    support_features = [make_features(row, reactome, goa, omni) for row in support_rows]
    wessels_gap = read_wessels_gap(payload_in)

    all_specs = specs()
    train_summaries = []
    train_rows_by_spec = {}
    for spec in all_specs:
        scored = score_rows(train_rows, train_features, spec)
        train_rows_by_spec[spec.name] = scored
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
    selected_name = select_spec(train_summaries)
    selected_spec = next(spec for spec in all_specs if spec.name == selected_name)
    support_scored = score_rows(support_rows, support_features, selected_spec)
    support_summary = summarize(support_scored, selected_spec, n_boot=args.n_boot, seed=args.seed, wessels_gap=wessels_gap, include_mmd=True)

    reactome_shuf = shuffled_sets(reactome, all_row_genes, args.seed + 11)
    goa_shuf = shuffled_sets(goa, all_row_genes, args.seed + 13)
    omni_shuf = shuffled_omni(omni, all_row_genes, args.seed + 17)
    support_shuf_features = [make_features(row, reactome_shuf, goa_shuf, omni_shuf) for row in support_rows]
    support_shuf_scored = score_rows(support_rows, support_shuf_features, selected_spec)
    shuffled_summary = summarize(support_shuf_scored, selected_spec, n_boot=args.n_boot, seed=args.seed + 200, wessels_gap=wessels_gap, include_mmd=True)

    out = {
        "status": None,
        "gpu_authorization": None,
        "input_json": str(args.input_json),
        "reactome_tsv": str(args.reactome_tsv),
        "goa_tsv": str(args.goa_tsv),
        "omnipath_tsv": str(args.omnipath_tsv),
        "boundary": "uses_existing_noharm_train_multi_loo_rows_for_selection_support_val_final_once_no_query_no_canonical_outputs",
        "split_guard": payload_in.get("split_guard", {}),
        "n_train_rows": len(train_rows),
        "n_support_rows": len(support_rows),
        "selected_spec": selected_name,
        "selected_train_summary": next(row for row in train_summaries if row["spec"] == selected_name),
        "selected_support_summary": support_summary,
        "shuffled_prior_control": shuffled_summary,
        "train_summaries": train_summaries,
    }
    out["decision"] = decide(out)
    out["status"] = out["decision"]["status"]
    out["gpu_authorization"] = out["decision"]["gpu_authorization"]
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render(out), encoding="utf-8")
    print(json.dumps({"status": out["status"], "gpu_authorization": out["gpu_authorization"], "selected_spec": selected_name, "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
