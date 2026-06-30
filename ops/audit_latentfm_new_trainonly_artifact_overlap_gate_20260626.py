#!/usr/bin/env python3
"""Preflight genuinely new train-only artifact overlap for LatentFM.

Short CPU task. Uses existing obs-schema, condition inventory, and completed
train-only/internal row-metric artifacts. It does not read expression matrices,
checkpoints, canonical multi, Track C query, train, infer, or use GPU.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"

OBS_SCHEMA_JSON = REPORTS / "latentfm_dataset_obs_schema_audit_20260624.json"
CONDITION_INV_JSON = REPORTS / "latentfm_condition_level_inventory_20260624.json"
OUT_JSON = REPORTS / "latentfm_new_trainonly_artifact_overlap_gate_20260626.json"
OUT_CSV = REPORTS / "latentfm_new_trainonly_artifact_overlap_columns_20260626.csv"
OUT_MD = REPORTS / "LATENTFM_NEW_TRAINONLY_ARTIFACT_OVERLAP_GATE_20260626.md"

ROW_METRIC_FILES = [
    REPORTS / "latentfm_condition_exposure_row_bootstrap_rows_20260625.csv",
    REPORTS / "latentfm_background_target_actionability_rows_20260625.csv",
    REPORTS / "latentfm_lodo_domain_conflict_rows_20260625.csv",
    REPORTS / "latentfm_multiprior_tailrisk_mask_rows_20260625.csv",
    REPORTS / "latentfm_response_program_projection_rows_20260625.csv",
    REPORTS / "latentfm_truecell_riskrow_complementarity_rows_20260625.csv",
    REPORTS / "latentfm_qc_support_reliability_rows_20260625.csv",
    REPORTS / "latentfm_jiang_guide_cytokine_context_rows_20260625.csv",
]

FAMILY_KEYWORDS = {
    "qc_support_consumed": [
        "n_genes",
        "ngenes",
        "ncounts",
        "total_counts",
        "pct_counts",
        "percent_mito",
        "percent_ribo",
    ],
    "jiang_context_consumed": ["guide", "sgrna", "cytokine", "mixscale"],
    "source_background_type_consumed": [
        "cell_line",
        "celltype",
        "cell_type",
        "tissue",
        "disease",
        "cancer",
        "organism",
        "perturbation_type",
    ],
    "target_actionability_consumed": ["target"],
    "chemical_semantics_ack_gated": [
        "dose",
        "pathway",
        "chembl",
        "smiles",
        "drug_dose",
    ],
    "technical_batch_protocol": ["plate", "well", "replicate", "batch", "time", "donor", "sample"],
    "identifier_or_split_forbidden": [
        "condition",
        "control",
        "cov",
        "gene",
        "perturbation",
        "nperts",
        "split",
    ],
}

FAMILY_DECISIONS = {
    "qc_support_consumed": "closed_by_qc_support_reliability_gate",
    "jiang_context_consumed": "supplement_only_after_jiang_context_gate",
    "source_background_type_consumed": "closed_by_source_background_type_tail_gates",
    "target_actionability_consumed": "closed_by_target_actionability_tail_gates",
    "chemical_semantics_ack_gated": "chemical_v2_ack_required_not_non_ack_unlock",
    "technical_batch_protocol": "protocol_covariate_only_not_model_signal",
    "identifier_or_split_forbidden": "forbidden_leakage_or_label_column",
    "unclassified_candidate": "candidate_only_if_overlap_and_controls_pass",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    return text


def load_outcome_keys() -> tuple[set[tuple[str, str]], dict[str, int]]:
    keys: set[tuple[str, str]] = set()
    by_file: dict[str, int] = {}
    for path in ROW_METRIC_FILES:
        if not path.is_file():
            continue
        local: set[tuple[str, str]] = set()
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "dataset" not in (reader.fieldnames or []) or "condition" not in (reader.fieldnames or []):
                continue
            for row in reader:
                ds = norm(row.get("dataset"))
                cond = norm(row.get("condition"))
                if ds and cond:
                    local.add((ds, cond))
        keys.update(local)
        by_file[path.name] = len(local)
    return keys, by_file


def load_dataset_conditions() -> dict[str, set[str]]:
    payload = read_json(CONDITION_INV_JSON)
    out: dict[str, set[str]] = defaultdict(set)
    for row in payload.get("rows", []):
        ds = norm(row.get("dataset"))
        cond = norm(row.get("condition"))
        if ds and cond:
            out[ds].add(cond)
    return out


def classify_column(column: str) -> str:
    low = column.lower()
    for family, keywords in FAMILY_KEYWORDS.items():
        if any(keyword in low for keyword in keywords):
            return family
    return "unclassified_candidate"


def build_rows() -> tuple[list[dict[str, Any]], dict[str, int]]:
    schema = read_json(OBS_SCHEMA_JSON)
    dataset_conditions = load_dataset_conditions()
    outcome_keys, outcome_by_file = load_outcome_keys()

    rows: list[dict[str, Any]] = []
    for ds_row in schema.get("rows", []):
        if not ds_row.get("exists", False):
            continue
        dataset = norm(ds_row.get("dataset"))
        if not dataset:
            continue
        dataset_keys = {(dataset, c) for c in dataset_conditions.get(dataset, set())}
        overlap_conditions = sorted(c for _, c in (dataset_keys & outcome_keys))
        for column in sorted(str(c) for c in ds_row.get("obs_columns", [])):
            family = classify_column(column)
            rows.append(
                {
                    "dataset": dataset,
                    "bucket": norm(ds_row.get("bucket")),
                    "modality": norm(ds_row.get("modality")),
                    "column": column,
                    "family": family,
                    "decision": FAMILY_DECISIONS[family],
                    "dataset_conditions": len(dataset_conditions.get(dataset, set())),
                    "outcome_overlap_conditions": len(overlap_conditions),
                    "example_overlap_conditions": ";".join(overlap_conditions[:8]),
                    "source_h5ad": norm(ds_row.get("path")),
                }
            )
    return rows, outcome_by_file


def summarize(rows: list[dict[str, Any]], outcome_by_file: dict[str, int]) -> dict[str, Any]:
    by_family: dict[str, dict[str, Any]] = {}
    for family in sorted(FAMILY_DECISIONS):
        local = [r for r in rows if r["family"] == family]
        datasets = sorted({r["dataset"] for r in local})
        columns = sorted({r["column"] for r in local})
        modalities = Counter(r["modality"] for r in local)
        overlap = sum(int(r["outcome_overlap_conditions"]) for r in local)
        by_family[family] = {
            "datasets": len(datasets),
            "dataset_names": datasets,
            "columns": columns,
            "modalities": dict(sorted(modalities.items())),
            "summed_outcome_overlap_conditions": int(overlap),
            "decision": FAMILY_DECISIONS[family],
        }

    candidate_rows = [r for r in rows if r["family"] == "unclassified_candidate"]
    candidate_datasets = sorted({r["dataset"] for r in candidate_rows})
    candidate_columns = sorted({r["column"] for r in candidate_rows})
    candidate_overlap = sum(int(r["outcome_overlap_conditions"]) for r in candidate_rows)

    pass_candidate = (
        len(candidate_datasets) >= 3
        and candidate_overlap >= 50
        and len(candidate_columns) > 0
    )
    status = (
        "new_trainonly_artifact_overlap_candidate_needs_controls_no_gpu"
        if pass_candidate
        else "new_trainonly_artifact_overlap_fail_no_gpu"
    )
    reasons = []
    if len(candidate_columns) == 0:
        reasons.append("no_unclassified_nonconsumed_obs_columns")
    if len(candidate_datasets) < 3:
        reasons.append("unclassified_candidate_dataset_count_below_3")
    if candidate_overlap < 50:
        reasons.append("unclassified_candidate_overlap_below_50")
    if pass_candidate:
        reasons.append("candidate_requires_control_gate_before_gpu")

    return {
        "status": status,
        "gpu_authorized": False,
        "outcome_metric_files": outcome_by_file,
        "families": by_family,
        "unclassified_candidate": {
            "datasets": len(candidate_datasets),
            "dataset_names": candidate_datasets,
            "columns": candidate_columns,
            "summed_outcome_overlap_conditions": int(candidate_overlap),
        },
        "decision": {
            "default_model": "xverse_8k_anchor",
            "immediate_gpu_candidate": False,
            "reasons": reasons,
            "next_action": (
                "no non-ACK GPU; if new external metadata/artifact is added, rerun this "
                "preflight and then require bootstrap/shuffle/source/count/tail controls"
            ),
        },
    }


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "family",
        "decision",
        "column",
        "dataset",
        "bucket",
        "modality",
        "dataset_conditions",
        "outcome_overlap_conditions",
        "example_overlap_conditions",
        "source_h5ad",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM New Train-Only Artifact Overlap Gate",
        "",
        f"Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M CST')}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"GPU authorized: `{payload['gpu_authorized']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only preflight over existing obs schema, condition inventory, and completed train-only/internal row metrics.",
        "- Does not read expression matrices, checkpoints, canonical multi, Track C query, train, infer, or use GPU.",
        "- Explicitly excludes artifact families already consumed by QC/Jiang/source/background/type/target/chemical/OT/no-harm gates.",
        "",
        "## Family Summary",
        "",
        "| family | datasets | columns | summed overlap | decision |",
        "|---|---:|---:|---:|---|",
    ]
    for family, info in payload["families"].items():
        lines.append(
            f"| `{family}` | {info['datasets']} | {len(info['columns'])} | "
            f"{info['summed_outcome_overlap_conditions']} | `{info['decision']}` |"
        )
    cand = payload["unclassified_candidate"]
    lines += [
        "",
        "## Unclassified Candidate Columns",
        "",
        f"- datasets: `{cand['datasets']}`",
        f"- columns: `{', '.join(cand['columns']) if cand['columns'] else 'none'}`",
        f"- summed outcome-overlap conditions: `{cand['summed_outcome_overlap_conditions']}`",
        "",
        "## Decision",
        "",
        f"- immediate GPU candidate: `{payload['decision']['immediate_gpu_candidate']}`",
        f"- reasons: `{payload['decision']['reasons']}`",
        f"- next action: {payload['decision']['next_action']}",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- column table: `{OUT_CSV}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    rows, outcome_by_file = build_rows()
    payload = summarize(rows, outcome_by_file)
    payload["outputs"] = {"json": str(OUT_JSON), "csv": str(OUT_CSV), "md": str(OUT_MD)}
    payload["rows"] = rows
    write_csv(rows)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "gpu_authorized": payload["gpu_authorized"],
                "candidate_columns": payload["unclassified_candidate"]["columns"],
                "out_md": str(OUT_MD),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
