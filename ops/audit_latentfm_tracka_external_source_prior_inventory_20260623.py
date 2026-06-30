#!/usr/bin/env python3
"""Audit whether a genuinely new Track A external/source prior exists locally.

This is a query-free, CPU-only inventory/preflight. It does not train, evaluate
models, inspect active logs, read held-out query artifacts, or authorize GPU
work. Its job is to decide whether the deferred Track A external/source-prior
branch has a materially new feature source beyond already-consumed
dataset/source/cell-line/Jiang/archetype evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
DATASET = ROOT / "dataset"

OUT_JSON = REPORTS / "latentfm_tracka_external_source_prior_inventory_20260623.json"
OUT_MD = REPORTS / "LATENTFM_TRACKA_EXTERNAL_SOURCE_PRIOR_INVENTORY_20260623.md"

METADATA_INPUTS = {
    "xverse_condition_metadata": DATASET / "latentfm_full" / "xverse" / "condition_metadata.json",
    "scfoundation_condition_metadata": DATASET / "latentfm" / "scfoundation" / "condition_metadata.json",
    "raw_genepert_bench_metainfo": DATASET / "raw" / "genepert_bench" / "metainfo.json",
    "raw_genepert_de5000_metainfo": DATASET / "raw" / "genepert_DE5000" / "metainfo.json",
    "raw_genepert_bench_brief_meta": DATASET / "raw" / "genepert_bench" / "brief_benchmark_meta.json",
}

CONSUMED_REPORTS = {
    "condition_source_agreement_gate": REPORTS / "latentfm_xverse_condition_source_agreement_covariate_gate_20260622.json",
    "condition_source_agreement_fulltrain_gate": REPORTS / "latentfm_xverse_condition_source_agreement_covariate_gate_fulltrain_20260622.json",
    "crosslatent_deployable_source_gate": REPORTS / "latentfm_xverse_crosslatent_deployable_source_gate_20260622.json",
    "tracka_jiang_abstain_gate": REPORTS / "latentfm_tracka_jiang_abstain_router_cpu_gate_20260623.json",
    "archetype_pocket_triage": REPORTS / "latentfm_soft_archetype_pocket_reopen_triage_20260623.json",
    "post_support_set_portfolio": REPORTS / "latentfm_post_support_set_summary_portfolio_decision_20260623.json",
    "compact_reporting_bundle": REPORTS / "latentfm_trackc_support_context_v2_compact_reporting_bundle_20260623.json",
}

EXTERNAL_PRIOR_PATTERNS = [
    "*pathway*",
    "*ontology*",
    "*geneset*",
    "*gene_set*",
    "*GO*.json",
    "*complex*",
    "*reactome*",
    "*kegg*",
    "*hallmark*",
    "*msig*",
    "*cytokine*",
    "*source_prior*",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def status_of(path: Path) -> str | None:
    if not path.is_file():
        return None
    obj = load_json(path)
    if isinstance(obj, dict):
        if obj.get("status") is not None:
            return str(obj["status"])
        decision = obj.get("decision")
        if isinstance(decision, dict) and decision.get("status") is not None:
            return str(decision["status"])
    return None


def condition_metadata_summary(path: Path) -> dict[str, Any]:
    obj = load_json(path)
    datasets = list(obj) if isinstance(obj, dict) else []
    fields: set[str] = set()
    perturbation_types: set[str] = set()
    condition_cols: set[str] = set()
    n_conditions = 0
    n_multi = 0
    n_single = 0
    examples: dict[str, Any] = {}
    for dataset, conds in obj.items():
        if not isinstance(conds, dict):
            continue
        for condition, meta in conds.items():
            if not isinstance(meta, dict):
                continue
            n_conditions += 1
            fields.update(meta)
            genes = meta.get("genes") or []
            if len(genes) > 1:
                n_multi += 1
            elif len(genes) == 1:
                n_single += 1
            if meta.get("perturbation_type_raw") is not None:
                perturbation_types.add(str(meta["perturbation_type_raw"]))
            if meta.get("condition_col") is not None:
                condition_cols.add(str(meta["condition_col"]))
            if len(examples) < 3:
                examples[f"{dataset}/{condition}"] = meta
    return {
        "path": str(path),
        "exists": path.is_file(),
        "datasets": len(datasets),
        "n_conditions": n_conditions,
        "n_single": n_single,
        "n_multi": n_multi,
        "fields": sorted(fields),
        "perturbation_type_raw_values": sorted(perturbation_types),
        "condition_col_values": sorted(condition_cols),
        "examples": examples,
    }


def dataset_metainfo_summary(path: Path) -> dict[str, Any]:
    obj = load_json(path)
    rows = obj if isinstance(obj, list) else obj.get("datasets", []) if isinstance(obj, dict) else []
    fields: set[str] = set()
    perturbation_types: set[str] = set()
    cell_lines: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        fields.update(row)
        if row.get("perturbation_type") is not None:
            perturbation_types.add(str(row["perturbation_type"]))
        if row.get("cell_line") is not None:
            cell_lines.add(str(row["cell_line"]))
    return {
        "path": str(path),
        "exists": path.is_file(),
        "rows": len(rows),
        "fields": sorted(fields),
        "perturbation_type_values": sorted(perturbation_types),
        "cell_line_values": sorted(cell_lines),
    }


def discover_external_prior_files() -> list[dict[str, Any]]:
    found: dict[str, Path] = {}
    roots = [DATASET, ROOT / "pretrainckpt", REPORTS]
    for root in roots:
        if not root.exists():
            continue
        for pattern in EXTERNAL_PRIOR_PATTERNS:
            for path in root.rglob(pattern):
                if path.is_file():
                    found[str(path)] = path
    records = []
    for path in sorted(found.values(), key=lambda p: str(p)):
        rel = str(path.relative_to(ROOT))
        is_report = rel.startswith("reports/")
        records.append(
            {
                "path": str(path),
                "relative": rel,
                "size_bytes": path.stat().st_size,
                "is_report_or_generated": is_report,
                "looks_like_independent_prior": (not is_report and path.suffix.lower() in {".json", ".csv", ".tsv", ".txt", ".gmt"}),
            }
        )
    return records


def build_payload() -> dict[str, Any]:
    metadata_summaries: dict[str, Any] = {}
    for name, path in METADATA_INPUTS.items():
        if not path.is_file():
            metadata_summaries[name] = {"path": str(path), "exists": False}
        elif "condition_metadata" in name:
            metadata_summaries[name] = condition_metadata_summary(path)
        else:
            metadata_summaries[name] = dataset_metainfo_summary(path)

    consumed = {
        name: {"path": str(path), "exists": path.is_file(), "status": status_of(path)}
        for name, path in CONSUMED_REPORTS.items()
    }
    external_files = discover_external_prior_files()
    independent_candidates = [
        row for row in external_files
        if row["looks_like_independent_prior"] and not row["relative"].startswith("reports/")
    ]

    observed_local_feature_families = [
        "condition genes",
        "perturbation_type_raw",
        "condition_col",
        "dataset-level perturbation_type",
        "dataset-level cell_line",
        "benchmark/source metadata",
    ]
    consumed_feature_families = [
        "condition/source agreement with scGPT and CellNavi caches",
        "cross-latent deployable source disagreement",
        "Jiang/cytokine abstain/lowcount/simple fallback rules",
        "dataset-negative fallback rules",
        "same-feature archetype/state-prior variants",
    ]

    independent_non_generated = [
        row for row in independent_candidates
        if not any(token in row["relative"].lower() for token in ["condition_metadata", "metainfo", "brief_benchmark_meta"])
    ]

    failed_reasons: list[str] = []
    if independent_non_generated:
        failed_reasons.append("independent_prior_files_present_needs_manual_protocol")
    else:
        failed_reasons.append("no_new_independent_local_prior_beyond_metadata_and_consumed_reports")

    status = (
        "tracka_external_source_prior_inventory_no_new_local_feature_no_gpu"
        if not independent_non_generated
        else "tracka_external_source_prior_inventory_found_candidate_protocol_needed_no_gpu"
    )

    return {
        "status": status,
        "timestamp": "2026-06-23 12:56 CST",
        "boundary": {
            "query_free": True,
            "canonical_test_read": False,
            "canonical_multi_read": False,
            "heldout_query_read": False,
            "active_log_read": False,
            "gpu_authorization": "none",
            "selection_or_tuning": False,
        },
        "hypothesis_checked": (
            "A Track A external/source prior can be reopened only if local artifacts contain "
            "a genuinely new biology/source feature beyond dataset labels, lowcount/Jiang rules, "
            "source-agreement features, and archetype/state-prior variants."
        ),
        "metadata_summaries": metadata_summaries,
        "discovered_external_prior_like_files": external_files[:80],
        "independent_candidate_files": independent_non_generated,
        "consumed_reports": consumed,
        "observed_local_feature_families": observed_local_feature_families,
        "consumed_feature_families": consumed_feature_families,
        "decision_reasons": failed_reasons,
        "decision": (
            "No Track A GPU or metric CPU gate is authorized from local metadata alone. "
            "The branch may reopen only after adding a real independent prior such as curated "
            "gene sets/pathways/complexes/source reliability labels, followed by a train-only "
            "internal proxy gate against the closed baselines."
        ),
        "next_valid_gate_requirements": [
            "new independent prior provenance with frozen hash",
            "train-only/internal proxy discovery and disjoint confirmation",
            "beat lowcount, dataset-negative, Jiang-abstain, source-agreement, and shuffled-prior controls",
            "cross-background and family proxy delta >= +0.02",
            "p_harm <= 0.20 and dataset min >= -0.02",
            "no canonical multi, no held-out query, and no GPU before pass",
        ],
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track A External/Source Prior Inventory",
        "",
        f"Status: `{payload['status']}`",
        "GPU authorization: `none`",
        "",
        "## Boundary",
        "",
    ]
    for key, value in payload["boundary"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Hypothesis Checked", "", payload["hypothesis_checked"], "", "## Local Metadata Summary", ""])
    lines.append("| source | exists | datasets/rows | conditions | fields | key values |")
    lines.append("|---|---:|---:|---:|---|---|")
    for name, row in payload["metadata_summaries"].items():
        count = row.get("datasets", row.get("rows", "NA"))
        conds = row.get("n_conditions", "NA")
        fields = ", ".join(row.get("fields", []))
        values = ", ".join(row.get("perturbation_type_raw_values", row.get("perturbation_type_values", []))[:8])
        if row.get("cell_line_values"):
            values += "; cell lines: " + ", ".join(row["cell_line_values"][:8])
        lines.append(f"| `{name}` | `{row.get('exists')}` | {count} | {conds} | {fields} | {values} |")

    lines.extend(["", "## Consumed Evidence", "", "| report | exists | status | path |", "|---|---:|---|---|"])
    for name, row in payload["consumed_reports"].items():
        lines.append(f"| `{name}` | `{row['exists']}` | `{row['status']}` | `{row['path']}` |")

    lines.extend(["", "## External-Prior-Like Files", ""])
    if payload["independent_candidate_files"]:
        lines.extend(["| file | size |", "|---|---:|"])
        for row in payload["independent_candidate_files"]:
            lines.append(f"| `{row['relative']}` | {row['size_bytes']} |")
    else:
        lines.append("No independent non-generated pathway/ontology/gene-set/complex/source-prior file was found beyond metadata and generated reports.")

    lines.extend(["", "## Decision Reasons", ""])
    lines.extend(f"- `{item}`" for item in payload["decision_reasons"])
    lines.extend(["", "## Decision", "", payload["decision"], "", "## Next Valid Gate Requirements", ""])
    lines.extend(f"- {item}" for item in payload["next_valid_gate_requirements"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    payload = build_payload()
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
