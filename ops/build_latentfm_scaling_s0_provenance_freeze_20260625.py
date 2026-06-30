#!/usr/bin/env python3
"""Build a frozen S0 provenance table for LatentFM scaling.

CPU-only. This script reads completed metadata/split artifacts and writes an
auditable condition-level table for scaling-law figure/protocol work. It does
not read model predictions, canonical multi metrics, Track C query, train, or
use GPU.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_TSV = REPORTS / "latentfm_scaling_s0_provenance_freeze_20260625.tsv"
OUT_JSON = REPORTS / "latentfm_scaling_s0_provenance_freeze_20260625.json"
OUT_MD = REPORTS / "LATENTFM_SCALING_S0_PROVENANCE_FREEZE_20260625.md"

CONDITION_TABLE = REPORTS / "latentfm_scaling_law_condition_table_20260624.json"
CANONICAL_SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"

SPLIT_ARTIFACTS = {
    "canonical_seed42": CANONICAL_SPLIT,
    "scaling_cap30_all_v2": ROOT
    / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap30_all_v2.json",
    "scaling_cap120_all_v2": ROOT
    / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json",
    "scaling_gene_cap120_allbg_v2": ROOT
    / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_gene_cap120_allbg_v2.json",
    "scaling_gene_cap120_k562bg_v2": ROOT
    / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_gene_cap120_k562bg_v2.json",
    "scaling_type_balanced_cap120_v2": ROOT
    / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_type_balanced_cap120_v2.json",
    "truecell_nested_budget128_seed42": ROOT
    / "dataset/biFlow_data/xverse_true_cell_count_scaling_nested_splits_20260624/split_gene_only_fixed256_budget64_128_256_budget128_seed42.json",
    "allmod_doseaware_budget32_seed42_loader": ROOT
    / "dataset/biFlow_data/xverse_true_cell_count_allmodality_doseaware_loader_splits_20260625/loader_split_all_modality_doseaware_fixed64_budget16_32_64_budget32_seed42.json",
}

ARTIFACTS = {
    "condition_table_json": CONDITION_TABLE,
    "condition_table_tsv": REPORTS / "latentfm_scaling_law_condition_table_20260624.tsv",
    "metainfo_audit_json": REPORTS / "latentfm_condition_level_metainfo_scaling_audit_20260624.json",
    "provenance_estimand_json": REPORTS / "latentfm_scaling_provenance_estimand_matrix_gate_20260624.json",
    "allmod_materializer_json": REPORTS / "latentfm_true_cell_count_allmodality_doseaware_materializer_gate_20260625.json",
    "allmod_design_json": REPORTS / "latentfm_true_cell_count_allmodality_doseaware_design_controls_20260625.json",
    **{f"split_{k}": v for k, v in SPLIT_ARTIFACTS.items()},
}


FIELDS = [
    "stable_condition_id",
    "dataset",
    "condition",
    "modality",
    "perturbation_type",
    "bucket",
    "nperts",
    "perturbation",
    "gene",
    "dose",
    "pathway",
    "cell_background_source",
    "n_backgrounds",
    "n_cells",
    "source_quality",
    "source_check_status",
    "source_label",
    "source_url",
    "source_h5ad",
    "has_obs_cell_background_axis",
    "has_obs_dose_axis",
    "has_obs_condition_axis",
    "has_obs_perturbation_type_axis",
    "canonical_seed42_membership",
    "scaling_cap30_all_v2_membership",
    "scaling_cap120_all_v2_membership",
    "scaling_gene_cap120_allbg_v2_membership",
    "scaling_gene_cap120_k562bg_v2_membership",
    "scaling_type_balanced_cap120_v2_membership",
    "truecell_nested_budget128_seed42_membership",
    "allmod_doseaware_budget32_seed42_loader_membership",
    "scaling_claim_inclusion",
    "exclusion_reason",
]


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_id(dataset: str, condition: str) -> str:
    return hashlib.sha1(f"{dataset}\t{condition}".encode("utf-8")).hexdigest()[:16]


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def load_split_membership(path: Path) -> dict[tuple[str, str], str]:
    if not path.is_file():
        return {}
    split = load_json(path)
    memberships: dict[tuple[str, str], list[str]] = defaultdict(list)
    for ds, groups in split.items():
        if not isinstance(groups, dict):
            continue
        for group, conds in groups.items():
            if not isinstance(conds, list):
                continue
            for cond in conds:
                memberships[(str(ds), str(cond))].append(str(group))
    return {k: ";".join(sorted(set(v))) for k, v in memberships.items()}


def as_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return "|".join(str(x) for x in v)
    return str(v)


def row_inclusion(row: dict[str, Any]) -> tuple[str, str]:
    reasons = []
    if row.get("source_quality") != "source_verified":
        reasons.append("source_not_verified")
    if not row.get("cell_background_source"):
        reasons.append("missing_cell_background_source")
    if not row.get("perturbation_type"):
        reasons.append("missing_perturbation_type")
    if row.get("modality") == "chemical":
        if not row.get("dose"):
            reasons.append("chemical_missing_dose")
        # Current condition table has pathway/dose but not scaffold/SMILES; this
        # is explicitly unresolved for S6 chemical/scaffold claims.
        reasons.append("chemical_scaffold_unresolved_in_s0_table")
    if reasons:
        return "diagnostic_or_excluded_until_resolved", ";".join(reasons)
    return "s0_resolved_for_gene_or_nonchemical_axes", ""


def main() -> int:
    source = load_json(CONDITION_TABLE)
    rows_in = source["rows"]
    split_maps = {name: load_split_membership(path) for name, path in SPLIT_ARTIFACTS.items()}
    rows_out = []
    for row in rows_in:
        dataset = str(row.get("dataset", ""))
        condition = str(row.get("condition", ""))
        out = {
            "stable_condition_id": stable_id(dataset, condition),
            "dataset": dataset,
            "condition": condition,
        }
        for field in FIELDS:
            if field in out or field.endswith("_membership") or field in {"scaling_claim_inclusion", "exclusion_reason"}:
                continue
            out[field] = as_str(row.get(field, ""))
        for name, membership in split_maps.items():
            out[f"{name}_membership"] = membership.get((dataset, condition), "")
        inclusion, reason = row_inclusion(row)
        out["scaling_claim_inclusion"] = inclusion
        out["exclusion_reason"] = reason
        rows_out.append(out)

    with OUT_TSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, dialect="excel-tab")
        writer.writeheader()
        writer.writerows(rows_out)

    summary = {
        "n_rows": len(rows_out),
        "n_datasets": len({r["dataset"] for r in rows_out}),
        "n_source_verified": sum(1 for r in rows_out if r["source_quality"] == "source_verified"),
        "n_s0_resolved": sum(1 for r in rows_out if r["scaling_claim_inclusion"] == "s0_resolved_for_gene_or_nonchemical_axes"),
        "claim_inclusion_counts": Counter(r["scaling_claim_inclusion"] for r in rows_out),
        "modality_counts": Counter(r["modality"] for r in rows_out),
        "perturbation_type_counts": Counter(r["perturbation_type"] for r in rows_out),
        "dataset_counts": Counter(r["dataset"] for r in rows_out),
        "missing_or_unresolved_reasons": Counter(
            reason
            for r in rows_out
            for reason in r["exclusion_reason"].split(";")
            if reason
        ),
        "split_membership_counts": {
            name: sum(1 for r in rows_out if r[f"{name}_membership"])
            for name in SPLIT_ARTIFACTS
        },
    }
    artifacts = {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256(path),
        }
        for name, path in ARTIFACTS.items()
    }
    payload = {
        "status": "scaling_s0_provenance_freeze_materialized",
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "reads_model_outputs": False,
            "reads_canonical_metrics": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
        },
        "outputs": {"tsv": str(OUT_TSV), "json": str(OUT_JSON), "md": str(OUT_MD)},
        "artifacts": artifacts,
        "summary": json.loads(json.dumps(summary, default=lambda x: dict(x))),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    reason_rows = sorted(summary["missing_or_unresolved_reasons"].items(), key=lambda kv: (-kv[1], kv[0]))
    lines = [
        "# LatentFM Scaling S0 Provenance Freeze",
        "",
        "Status: `scaling_s0_provenance_freeze_materialized`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only provenance/materialization gate.",
        "- Does not read model outputs, canonical metrics, canonical multi, or Track C held-out query.",
        "- Writes a frozen condition-level S0 table plus input artifact SHA256 hashes.",
        "",
        "## Outputs",
        "",
        f"- TSV: `{OUT_TSV}`",
        f"- JSON: `{OUT_JSON}`",
        "",
        "## Summary",
        "",
        f"- rows: `{summary['n_rows']}`",
        f"- datasets: `{summary['n_datasets']}`",
        f"- source-verified rows: `{summary['n_source_verified']}`",
        f"- S0 resolved rows for gene/nonchemical axes: `{summary['n_s0_resolved']}`",
        f"- modality counts: `{dict(summary['modality_counts'])}`",
        f"- perturbation type counts: `{dict(summary['perturbation_type_counts'])}`",
        "",
        "## Unresolved / Exclusion Reasons",
        "",
        "| reason | rows |",
        "|---|---:|",
    ]
    for reason, n in reason_rows:
        lines.append(f"| `{reason}` | {n} |")
    lines += [
        "",
        "## Split Membership Coverage",
        "",
        "| split artifact | rows with membership |",
        "|---|---:|",
    ]
    for name, n in sorted(summary["split_membership_counts"].items()):
        lines.append(f"| `{name}` | {n} |")
    lines += [
        "",
        "## Decision",
        "",
        "- This artifact is suitable as the S0 input for scaling/failure-map figures and protocol audits.",
        "- It does not authorize GPU. GPU unlock still requires an axis-specific CPU gate with a non-no-op route and strict no-leakage boundary.",
        "- Chemical/scaffold claims remain unresolved at S0 because the current condition table has dose/pathway but no scaffold/SMILES field; S6 needs a dedicated chemical protocol gate.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(OUT_MD)
    print(OUT_JSON)
    print(OUT_TSV)
    print(payload["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
