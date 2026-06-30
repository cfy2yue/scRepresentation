#!/usr/bin/env python3
"""Track A cytokine/IFN/TNF signature reliability CPU gate.

This CPU gate tests one predeclared biological reliability rule for the
scFoundation Track A near-miss: use a frozen GOA/Reactome cytokine/IFN/TNF/NFkB
signature to decide when to abstain from the scFoundation anchor and keep the
train-only gene baseline.  It uses only internal proxy rows; canonical Track A
outcomes are not read for selection/scoring.
"""

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
REACTOME_TSV = ROOT / "dataset/external_priors/reactome_pathways_current_20260623/reactome_gene_pathways.tsv"
GOA_TSV = ROOT / "dataset/external_priors/goa_human_20260519/goa_human_gene_terms.tsv"
OUT_JSON = REPORTS / "latentfm_tracka_cytokine_signature_reliability_gate_20260623.json"
OUT_MD = REPORTS / "LATENTFM_TRACKA_CYTOKINE_SIGNATURE_RELIABILITY_GATE_20260623.md"
GROUPS = ("internal_val_cross_background_seen_gene_proxy", "internal_val_family_gene_proxy")
BOOT_N = 2000
SEED = 20260623
KEYWORDS = (
    "interferon",
    "ifn-",
    "ifn ",
    "tumor necrosis factor",
    "tnf",
    "cytokine",
    "nf-kappa",
    "nf kappa",
    "inflammatory",
    "inflammation",
)


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


def term_hit(text: str) -> bool:
    lowered = text.lower()
    return any(key in lowered for key in KEYWORDS)


def load_reactome_signature(path: Path) -> tuple[set[str], dict[str, Any]]:
    genes = set()
    n_terms = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            gene = str(row.get("gene") or "").strip().upper()
            terms = [str(x).strip() for x in str(row.get("reactome_pathways") or "").split(";") if str(x).strip()]
            hits = [term for term in terms if term_hit(term)]
            if gene and hits:
                genes.add(gene)
                n_terms += len(hits)
    return genes, {"source": str(path), "n_genes": len(genes), "n_gene_term_hits": n_terms}


def load_goa_signature(path: Path) -> tuple[set[str], dict[str, Any]]:
    # GOA normalized TSV stores GO IDs, not names.  Keep this as a provenance
    # placeholder unless a GO-name table is introduced; do not infer names from
    # IDs.
    return set(), {"source": str(path), "n_genes": 0, "n_gene_term_hits": 0, "note": "GOA normalized file has GO IDs only; not used for keyword signature"}


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
    return {
        "candidate": candidate,
        "baseline": baseline,
        "delta_mean": point,
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_improve": float(np.mean(arr > 0.0)),
        "p_harm": float(np.mean(arr < 0.0)),
        "dataset_deltas": point_by_ds,
        "dataset_min": float(min(point_by_ds.values())) if point_by_ds else float("nan"),
    }


def score_rows(rows: list[dict[str, Any]], signature: set[str], *, label: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        gene = str(row["gene"]).upper()
        flagged = gene in signature
        item = dict(row)
        item[f"{label}_flagged"] = bool(flagged)
        item[f"{label}_anchor_or_gene"] = as_float(row["gene_raw_mean"]) if flagged else as_float(row["anchor_pearson_pert"])
        out.append(item)
    return out


def evaluate_group(rows: list[dict[str, Any]], signature: set[str], shuffled_signature: set[str], group: str) -> dict[str, Any]:
    group_rows = [dict(row) for row in rows if row.get("group") == group]
    scored = score_rows(group_rows, signature, label="cytokine")
    shuffled = score_rows(group_rows, shuffled_signature, label="shuffled")
    for i, row in enumerate(scored):
        row["shuffled_anchor_or_gene"] = shuffled[i]["shuffled_anchor_or_gene"]
        row["shuffled_flagged"] = shuffled[i]["shuffled_flagged"]
    paired = [
        paired_bootstrap(scored, "cytokine_anchor_or_gene", baseline, seed=SEED + i)
        for i, baseline in enumerate(("gene_raw_mean", "anchor_pearson_pert", "dataset_mean", "global_mean", "shuffled_anchor_or_gene"))
    ]
    flagged_by_dataset: dict[str, int] = defaultdict(int)
    total_by_dataset: dict[str, int] = defaultdict(int)
    for row in scored:
        total_by_dataset[str(row["dataset"])] += 1
        flagged_by_dataset[str(row["dataset"])] += int(row["cytokine_flagged"])
    return {
        "group": group,
        "n_rows": len(scored),
        "flagged_fraction": float(np.mean([bool(row["cytokine_flagged"]) for row in scored])) if scored else 0.0,
        "shuffled_flagged_fraction": float(np.mean([bool(row["shuffled_flagged"]) for row in scored])) if scored else 0.0,
        "flagged_by_dataset": {ds: {"flagged": flagged_by_dataset[ds], "n": total_by_dataset[ds]} for ds in sorted(total_by_dataset)},
        "paired_deltas": paired,
        "scored_rows": scored,
    }


def paired_row(result: dict[str, Any], baseline: str) -> dict[str, Any]:
    return next(row for row in result["paired_deltas"] if row["baseline"] == baseline)


def decide(results: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = []
    for result in results:
        group = result["group"]
        vs_gene = paired_row(result, "gene_raw_mean")
        vs_anchor = paired_row(result, "anchor_pearson_pert")
        vs_shuf = paired_row(result, "shuffled_anchor_or_gene")
        if float(result["flagged_fraction"]) < 0.02:
            reasons.append(f"{group}_signature_flags_too_few_rows")
        if float(vs_gene["delta_mean"]) < 0.02:
            reasons.append(f"{group}_delta_vs_gene_below_0p02")
        if float(vs_gene["p_harm"]) > 0.20:
            reasons.append(f"{group}_harm_vs_gene_above_0p20")
        if float(vs_gene["dataset_min"]) < -0.02:
            reasons.append(f"{group}_dataset_min_vs_gene_below_minus_0p02")
        if float(vs_anchor["delta_mean"]) < -0.005:
            reasons.append(f"{group}_material_loss_vs_anchor")
        if float(vs_shuf["delta_mean"]) < 0.002:
            reasons.append(f"{group}_shuffled_signature_not_beaten_by_0p002")
    status = "tracka_cytokine_signature_reliability_gate_pass_code_gate_next_no_gpu" if not reasons else "tracka_cytokine_signature_reliability_gate_fail_no_gpu"
    return {
        "status": status,
        "gpu_authorization": "none",
        "next_authorization": "code_gate_only_if_pass_else_none",
        "reasons": reasons,
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track A Cytokine Signature Reliability Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        "GPU authorization: `none`",
        "",
        "## Boundary",
        "",
        "- Uses scFoundation internal proxy rows and frozen Reactome keyword-derived cytokine/IFN/TNF/NFkB signature.",
        "- Does not read canonical Track A outcomes, canonical multi, held-out query, active logs, or GPU artifacts.",
        "- GOA normalized terms are GO IDs only and are not used for keyword matching.",
        "",
        "## Signature",
        "",
        f"- Reactome genes: `{payload['signature']['reactome']['n_genes']}`",
        f"- GOA genes used: `{payload['signature']['goa']['n_genes']}`",
        f"- Union genes: `{payload['signature']['n_union_genes']}`",
        "",
        "## Group Results",
        "",
        "| group | flagged | delta vs gene | p harm | dataset min | delta vs anchor | delta vs shuffled |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        vs_gene = paired_row(result, "gene_raw_mean")
        vs_anchor = paired_row(result, "anchor_pearson_pert")
        vs_shuf = paired_row(result, "shuffled_anchor_or_gene")
        lines.append(
            f"| {result['group']} | {result['flagged_fraction']:.3f} | {fmt(vs_gene['delta_mean'])} | "
            f"{fmt(vs_gene['p_harm'])} | {fmt(vs_gene['dataset_min'])} | {fmt(vs_anchor['delta_mean'])} | {fmt(vs_shuf['delta_mean'])} |"
        )
    lines.extend(["", "## Gate Reasons", ""])
    reasons = payload["decision"].get("reasons") or []
    lines.extend(f"- `{reason}`" for reason in reasons) if reasons else lines.append("- none")
    lines.extend(["", "## Decision", "", payload["decision_text"], ""])
    return "\n".join(lines)


def main() -> int:
    rows_payload = load_json(SCF_ROWS)
    all_rows = rows_payload["condition_rows"]
    reactome, reactome_meta = load_reactome_signature(REACTOME_TSV)
    goa, goa_meta = load_goa_signature(GOA_TSV)
    signature = reactome | goa
    row_genes = sorted({str(row["gene"]).upper() for row in all_rows})
    row_signature_hits = sorted(set(row_genes) & signature)
    rng = np.random.default_rng(SEED + 55)
    shuffled = set(rng.choice(row_genes, size=min(len(row_signature_hits), len(row_genes)), replace=False))
    results = [evaluate_group(all_rows, signature, shuffled, group) for group in GROUPS]
    decision = decide(results)
    payload = {
        "status": decision["status"],
        "timestamp": "2026-06-23 15:15 CST",
        "boundary": {
            "canonical_test_read": False,
            "canonical_multi_read": False,
            "heldout_query_read": False,
            "active_log_read": False,
            "gpu_authorization": "none",
        },
        "inputs": {
            "scfoundation_internal_proxy": str(SCF_ROWS),
            "reactome_tsv": str(REACTOME_TSV),
            "goa_tsv": str(GOA_TSV),
        },
        "keywords": KEYWORDS,
        "signature": {
            "reactome": reactome_meta,
            "goa": goa_meta,
            "n_union_genes": len(signature),
            "n_row_genes": len(row_genes),
            "n_row_signature_hits": len(row_signature_hits),
            "n_shuffled_signature_genes": len(shuffled),
        },
        "results": results,
        "decision": decision,
        "decision_text": (
            "The cytokine signature reliability rule is not promoted unless it "
            "beats gene baseline on both proxy groups with low harm and separates "
            "from a size-matched shuffled signature. Failure keeps the scFoundation "
            "near-miss closed as a GPU target."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "gpu_authorization": "none", "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
