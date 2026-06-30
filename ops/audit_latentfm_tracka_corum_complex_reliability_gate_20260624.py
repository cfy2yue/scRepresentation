#!/usr/bin/env python3
"""Track A CORUM complex-membership reliability CPU gate."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
SCF_ROWS = REPORTS / "latentfm_crosslatent_scfoundation_tracka_anchor_internal_val_20260622.json"
CORUM_GENE = ROOT / "dataset/external_priors/corum_complexes_20260624/corum_human_gene_complexes.tsv"
CORUM_SUMMARY = ROOT / "dataset/external_priors/corum_complexes_20260624/corum_human_complex_prior_summary.json"
OUT_JSON = REPORTS / "latentfm_tracka_corum_complex_reliability_gate_20260624.json"
OUT_MD = REPORTS / "LATENTFM_TRACKA_CORUM_COMPLEX_RELIABILITY_GATE_20260624.md"
GROUPS = ("internal_val_cross_background_seen_gene_proxy", "internal_val_family_gene_proxy")
BOOT_N = 2000
SEED = 20260624


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.6f}"


def load_gene_degrees(path: Path) -> dict[str, int]:
    out = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            gene = str(row.get("gene") or "").strip().upper()
            if gene:
                out[gene] = int(float(row.get("n_complexes") or 0))
    return out


def paired_bootstrap(rows: list[dict[str, Any]], candidate: str, baseline: str, *, seed: int) -> dict[str, Any]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        cand = as_float(row.get(candidate))
        base = as_float(row.get(baseline))
        if np.isfinite(cand) and np.isfinite(base):
            by_ds[str(row["dataset"])].append(float(cand - base))
    keys = sorted(ds for ds, vals in by_ds.items() if vals)
    point_by_ds = {ds: float(np.mean(by_ds[ds])) for ds in keys}
    point = float(np.mean(list(point_by_ds.values()))) if point_by_ds else float("nan")
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(BOOT_N):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        vals = []
        for ds in sampled:
            arr = np.asarray(by_ds[str(ds)], dtype=float)
            vals.append(float(np.mean(rng.choice(arr, size=len(arr), replace=True))))
        boot.append(float(np.mean(vals)))
    arr = np.asarray(boot, dtype=float)
    return {"candidate": candidate, "baseline": baseline, "delta_mean": point, "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))], "p_improve": float(np.mean(arr > 0.0)), "p_harm": float(np.mean(arr < 0.0)), "dataset_deltas": point_by_ds, "dataset_min": float(min(point_by_ds.values())) if point_by_ds else float("nan")}


def rule_names() -> list[str]:
    out = ["always_anchor", "always_gene"]
    for threshold in (1, 2, 5, 10, 20):
        out.append(f"degree_ge{threshold}_use_gene")
        out.append(f"degree_lt{threshold}_use_gene")
    return out


def rule_score(row: dict[str, Any], degree: int, rule: str) -> tuple[float, bool]:
    if rule == "always_anchor":
        return as_float(row["anchor_pearson_pert"]), False
    if rule == "always_gene":
        return as_float(row["gene_raw_mean"]), True
    if rule.startswith("degree_ge"):
        threshold = int(rule.split("_use_gene")[0].replace("degree_ge", ""))
        flagged = degree >= threshold
    elif rule.startswith("degree_lt"):
        threshold = int(rule.split("_use_gene")[0].replace("degree_lt", ""))
        flagged = degree < threshold
    else:
        raise ValueError(rule)
    return (as_float(row["gene_raw_mean"]) if flagged else as_float(row["anchor_pearson_pert"])), flagged


def evaluate_group(rows: list[dict[str, Any]], degrees: dict[str, int], shuffled_degrees: dict[str, int], group: str, rule: str) -> dict[str, Any]:
    out = []
    for row in [r for r in rows if r.get("group") == group]:
        gene = str(row["gene"]).upper()
        score, flagged = rule_score(row, degrees.get(gene, 0), rule)
        shuf_score, shuf_flagged = rule_score(row, shuffled_degrees.get(gene, 0), rule)
        item = dict(row)
        item["corum_degree"] = int(degrees.get(gene, 0))
        item["corum_flagged"] = bool(flagged)
        item["corum_anchor_or_gene"] = score
        item["shuffled_corum_degree"] = int(shuffled_degrees.get(gene, 0))
        item["shuffled_corum_flagged"] = bool(shuf_flagged)
        item["shuffled_corum_anchor_or_gene"] = shuf_score
        out.append(item)
    paired = [paired_bootstrap(out, "corum_anchor_or_gene", baseline, seed=SEED + i) for i, baseline in enumerate(("gene_raw_mean", "anchor_pearson_pert", "dataset_mean", "global_mean", "shuffled_corum_anchor_or_gene"))]
    return {"group": group, "rule": rule, "n_rows": len(out), "flagged_fraction": float(np.mean([r["corum_flagged"] for r in out])) if out else 0.0, "paired_deltas": paired, "scored_rows": out}


def paired_row(result: dict[str, Any], baseline: str) -> dict[str, Any]:
    return next(row for row in result["paired_deltas"] if row["baseline"] == baseline)


def select_rule(rows: list[dict[str, Any]], degrees: dict[str, int], shuffled_degrees: dict[str, int]) -> str:
    candidates = []
    for rule in rule_names():
        results = [evaluate_group(rows, degrees, shuffled_degrees, group, rule) for group in GROUPS]
        ok = True
        score = 0.0
        for result in results:
            vs_gene = paired_row(result, "gene_raw_mean")
            vs_anchor = paired_row(result, "anchor_pearson_pert")
            if result["flagged_fraction"] <= 0.0:
                ok = False
            if vs_gene["delta_mean"] < 0.0 or vs_gene["p_harm"] > 0.35 or vs_gene["dataset_min"] < -0.05:
                ok = False
            if vs_anchor["delta_mean"] < -0.03:
                ok = False
            score += float(vs_gene["delta_mean"]) + 0.25 * float(vs_anchor["delta_mean"])
        candidates.append((ok, score, rule, results))
    pool = [x for x in candidates if x[0]] or candidates
    return sorted(pool, key=lambda x: x[1], reverse=True)[0][2]


def decide(results: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = []
    for result in results:
        group = result["group"]
        vs_gene = paired_row(result, "gene_raw_mean")
        vs_anchor = paired_row(result, "anchor_pearson_pert")
        vs_shuf = paired_row(result, "shuffled_corum_anchor_or_gene")
        if float(result["flagged_fraction"]) < 0.02:
            reasons.append(f"{group}_flags_too_few_rows")
        if float(vs_gene["delta_mean"]) < 0.02:
            reasons.append(f"{group}_delta_vs_gene_below_0p02")
        if float(vs_gene["p_harm"]) > 0.20:
            reasons.append(f"{group}_harm_vs_gene_above_0p20")
        if float(vs_gene["dataset_min"]) < -0.02:
            reasons.append(f"{group}_dataset_min_vs_gene_below_minus_0p02")
        if float(vs_anchor["delta_mean"]) < -0.005:
            reasons.append(f"{group}_material_loss_vs_anchor")
        if float(vs_shuf["delta_mean"]) < 0.002:
            reasons.append(f"{group}_shuffled_corum_not_beaten_by_0p002")
    status = "tracka_corum_complex_reliability_gate_pass_code_gate_next_no_gpu" if not reasons else "tracka_corum_complex_reliability_gate_fail_no_gpu"
    return {"status": status, "gpu_authorization": "none", "next_authorization": "code_gate_only_if_pass_else_none", "reasons": reasons}


def render(payload: dict[str, Any]) -> str:
    lines = ["# Track A CORUM Complex Reliability Gate", "", f"Status: `{payload['decision']['status']}`", "GPU authorization: `none`", "", "## Boundary", "", "- Uses scFoundation internal proxy rows and frozen CORUM human complex degree features.", "- Does not read canonical Track A outcomes, canonical multi, held-out query, active logs, or GPU artifacts.", "", "## Provenance", "", f"- CORUM summary: `{payload['corum_summary_path']}`", f"- CORUM release: `{payload['corum_summary'].get('source', {}).get('release')}`", f"- selected rule: `{payload['selected_rule']}`", "", "## Group Results", "", "| group | flagged | delta vs gene | p harm | dataset min | delta vs anchor | delta vs shuffled |", "|---|---:|---:|---:|---:|---:|---:|"]
    for result in payload["results"]:
        vs_gene = paired_row(result, "gene_raw_mean")
        vs_anchor = paired_row(result, "anchor_pearson_pert")
        vs_shuf = paired_row(result, "shuffled_corum_anchor_or_gene")
        lines.append(f"| {result['group']} | {result['flagged_fraction']:.3f} | {fmt(vs_gene['delta_mean'])} | {fmt(vs_gene['p_harm'])} | {fmt(vs_gene['dataset_min'])} | {fmt(vs_anchor['delta_mean'])} | {fmt(vs_shuf['delta_mean'])} |")
    lines.extend(["", "## Decision Reasons", ""])
    lines.extend([f"- `{r}`" for r in payload["decision"].get("reasons", [])] or ["- none"])
    return "\n".join(lines) + "\n"


def main() -> int:
    rows_payload = load_json(SCF_ROWS)
    rows = rows_payload["condition_rows"]
    degrees = load_gene_degrees(CORUM_GENE)
    row_genes = sorted({str(row["gene"]).upper() for row in rows})
    degree_values = [degrees.get(g, 0) for g in row_genes]
    rng = np.random.default_rng(SEED + 123)
    shuffled_values = list(degree_values)
    rng.shuffle(shuffled_values)
    shuffled_degrees = dict(zip(row_genes, shuffled_values))
    selected = select_rule(rows, degrees, shuffled_degrees)
    results = [evaluate_group(rows, degrees, shuffled_degrees, group, selected) for group in GROUPS]
    decision = decide(results)
    payload = {
        "status": decision["status"],
        "timestamp": "2026-06-24 00:35 CST",
        "inputs": {"scfoundation_internal_proxy": str(SCF_ROWS), "corum_gene_complexes": str(CORUM_GENE)},
        "corum_summary_path": str(CORUM_SUMMARY),
        "corum_summary": load_json(CORUM_SUMMARY),
        "boundary": {"canonical_test_read": False, "canonical_multi_read": False, "heldout_query_read": False, "active_log_read": False, "gpu_artifact_read": False},
        "selected_rule": selected,
        "row_gene_degree_summary": {"n_row_genes": len(row_genes), "n_corum_covered": int(sum(1 for g in row_genes if degrees.get(g, 0) > 0)), "max_degree": int(max(degree_values) if degree_values else 0)},
        "results": results,
        "decision": decision,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "gpu_authorization": "none", "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
