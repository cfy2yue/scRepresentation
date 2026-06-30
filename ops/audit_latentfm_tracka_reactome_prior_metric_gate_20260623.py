#!/usr/bin/env python3
"""Train-only Track A CPU gate for the Reactome pathway external prior."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from audit_latentfm_tracka_goa_prior_metric_gate_20260623 import (
    GROUPS,
    SEED,
    as_float,
    build_feature_matrix,
    lodo_kernel_ridge,
    load_json,
    paired_bootstrap,
)


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
PRIOR_DIR = ROOT / "dataset" / "external_priors" / "reactome_pathways_current_20260623"

REACTOME_SUMMARY = PRIOR_DIR / "reactome_pathway_prior_summary.json"
REACTOME_GENE_PATHWAYS = PRIOR_DIR / "reactome_gene_pathways.tsv"
XVERSE_ROWS = REPORTS / "latentfm_xverse_tracka_residual_forensics_20260622.json"

OUT_JSON = REPORTS / "latentfm_tracka_reactome_prior_metric_gate_20260623.json"
OUT_MD = REPORTS / "LATENTFM_TRACKA_REACTOME_PRIOR_METRIC_GATE_20260623.md"


def load_gene_pathways(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            pathways = set(filter(None, str(row.get("reactome_pathways") or "").split(";")))
            if pathways:
                out[str(row["gene"]).upper()] = pathways
    return out


def evaluate_group(all_rows: list[dict[str, Any]], gene_pathways: dict[str, set[str]], group: str) -> dict[str, Any]:
    base_rows = [dict(row) for row in all_rows if row.get("group") == group]
    X, terms, feature_meta = build_feature_matrix(base_rows, gene_pathways, shuffled=False)
    pred, lodo = lodo_kernel_ridge(base_rows, X)
    Xs, _, shuffled_meta = build_feature_matrix(base_rows, gene_pathways, shuffled=True)
    spred, slodo = lodo_kernel_ridge(base_rows, Xs)
    scored = []
    for i, row in enumerate(base_rows):
        item = dict(row)
        item["reactome_pred_anchor_minus_gene"] = float(pred[i])
        item["reactome_shuffled_pred_anchor_minus_gene"] = float(spred[i])
        item["reactome_routed_xverse_or_gene"] = as_float(row["anchor_pearson_pert"]) if pred[i] > 0 else as_float(row["gene_raw_mean"])
        item["reactome_shuffled_routed_xverse_or_gene"] = as_float(row["anchor_pearson_pert"]) if spred[i] > 0 else as_float(row["gene_raw_mean"])
        scored.append(item)
    paired = [
        paired_bootstrap(scored, "reactome_routed_xverse_or_gene", baseline, seed=SEED + 100 + i)
        for i, baseline in enumerate(
            (
                "gene_raw_mean",
                "dataset_mean",
                "global_mean",
                "anchor_pearson_pert",
                "reactome_shuffled_routed_xverse_or_gene",
            )
        )
    ]
    return {
        "group": group,
        "feature_meta": feature_meta,
        "shuffled_feature_meta": shuffled_meta,
        "kept_pathways_preview": terms[:20],
        "lodo": lodo,
        "shuffled_lodo": slodo,
        "paired_deltas": paired,
    }


def paired_row(result: dict[str, Any], baseline: str) -> dict[str, Any]:
    return next(row for row in result["paired_deltas"] if row["baseline"] == baseline)


def decide(results: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = []
    for result in results:
        group = result["group"]
        if result["feature_meta"]["coverage_fraction"] < 0.80:
            reasons.append(f"{group}_coverage_below_0p80")
        if (result["lodo"].get("spearman") or 0.0) < 0.20 and (result["lodo"].get("r2") or -999.0) < 0.05:
            reasons.append(f"{group}_predictive_signal_below_gate")
        vs_gene = paired_row(result, "gene_raw_mean")
        vs_shuffled = paired_row(result, "reactome_shuffled_routed_xverse_or_gene")
        if float(vs_gene["delta_mean"]) < 0.02:
            reasons.append(f"{group}_delta_vs_gene_below_0p02")
        if float(vs_gene["p_harm"]) > 0.20:
            reasons.append(f"{group}_harm_vs_gene_above_0p20")
        if float(vs_gene["dataset_min"]) < -0.02:
            reasons.append(f"{group}_dataset_min_vs_gene_below_minus_0p02")
        if float(vs_shuffled["delta_mean"]) < 0.02:
            reasons.append(f"{group}_shuffled_control_not_separated")
    status = "tracka_reactome_prior_metric_gate_pass_code_gate_next_no_gpu" if not reasons else "tracka_reactome_prior_metric_gate_fail_no_gpu"
    return {"status": status, "gpu_authorization": "none", "next_authorization": "code_gate_only_if_pass_else_none", "reasons": reasons}


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.6f}"


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track A Reactome Prior Metric Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        "GPU authorization: `none`",
        "",
        "## Boundary",
        "",
        "- Uses only Reactome pathway prior and xverse train-only/internal proxy residual rows.",
        "- Does not read canonical test, canonical multi, held-out query, active logs, or GPU artifacts.",
        "- Passing would authorize only a later code/provenance gate, not immediate GPU.",
        "",
        "## Prior",
        "",
        f"- source report: `{payload['prior']['acquisition_report']}`",
        f"- raw ZIP SHA256: `{payload['prior']['raw_zip_sha256']}`",
        f"- pathways: `{payload['prior']['n_pathways']}`",
        f"- genes: `{payload['prior']['n_genes']}`",
        "",
        "## Group Results",
        "",
        "| group | coverage | kept pathways | LODO Spearman | LODO R2 | delta vs gene | p harm | dataset min | delta vs shuffled |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        vs_gene = paired_row(result, "gene_raw_mean")
        vs_shuf = paired_row(result, "reactome_shuffled_routed_xverse_or_gene")
        lines.append(
            f"| {result['group']} | {result['feature_meta']['coverage_fraction']:.3f} | "
            f"{result['feature_meta']['n_kept_terms']} | {fmt(result['lodo'].get('spearman'))} | "
            f"{fmt(result['lodo'].get('r2'))} | {fmt(vs_gene['delta_mean'])} | "
            f"{fmt(vs_gene['p_harm'])} | {fmt(vs_gene['dataset_min'])} | {fmt(vs_shuf['delta_mean'])} |"
        )
    lines.extend(["", "## Gate Reasons", ""])
    reasons = payload["decision"].get("reasons") or []
    lines.extend(f"- `{reason}`" for reason in reasons) if reasons else lines.append("- none")
    lines.extend(["", "## Decision", "", payload["decision_text"], ""])
    return "\n".join(lines)


def main() -> int:
    prior_summary = load_json(REACTOME_SUMMARY)
    rows_payload = load_json(XVERSE_ROWS)
    all_rows = rows_payload["condition_rows"]
    gene_pathways = load_gene_pathways(REACTOME_GENE_PATHWAYS)
    results = [evaluate_group(all_rows, gene_pathways, group) for group in GROUPS]
    decision = decide(results)
    payload = {
        "status": decision["status"],
        "timestamp": "2026-06-23 13:07 CST",
        "boundary": {
            "query_free": True,
            "canonical_test_read": False,
            "canonical_multi_read": False,
            "heldout_query_read": False,
            "active_log_read": False,
            "gpu_authorization": "none",
            "selection_or_tuning": False,
        },
        "prior": {
            "acquisition_report": str(REPORTS / "LATENTFM_REACTOME_PATHWAY_PRIOR_ACQUISITION_20260623.md"),
            "raw_zip_sha256": prior_summary["hashes"]["raw_zip"],
            "raw_gmt_sha256": prior_summary["hashes"]["raw_gmt"],
            "gene_pathways_sha256": prior_summary["hashes"]["gene_pathways_tsv"],
            "n_genes": prior_summary["n_genes"],
            "n_pathways": prior_summary["n_pathways"],
        },
        "results": results,
        "decision": decision,
        "decision_text": (
            "The Reactome prior metric gate may proceed only if both internal proxy groups pass coverage, "
            "predictive-signal, improvement, no-harm, and shuffled-control gates. A failure keeps this "
            "external prior as a hashed input/negative result and does not authorize GPU."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
