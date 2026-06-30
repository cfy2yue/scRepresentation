#!/usr/bin/env python3
"""Design-control gate for dose-aware all-modality scaling artifacts.

CPU-only. This gate separates "usable for a bounded GPU smoke" from
"sufficient for a Nature Methods-level scaling-law claim".
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
MATERIALIZER_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_materializer_gate_20260625.json"
SCHEMA_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_schema_gate_20260625.json"
DRYLOAD_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_dryload_gate_20260625.json"
FEASIBILITY_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_feasibility_gate_20260625.json"
OUT_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_design_controls_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_TRUE_CELL_COUNT_ALLMODALITY_DOSEAWARE_DESIGN_CONTROLS_20260625.md"

EXPECTED_BUDGETS = {16, 32, 64}
EXPECTED_SEEDS = {42, 43, 44}
SCIPLEX_DATASETS = {"sciplex3_A549", "sciplex3_K562", "sciplex3_MCF7"}


def load_json(path: Path) -> Any:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def load_sampled_summary(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "sampled_indices_summary.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def condition_signature(split: dict[str, Any]) -> dict[str, Any]:
    signature = {}
    for ds, groups in sorted(split.items()):
        signature[ds] = {
            "train": sorted(str(x) for x in groups.get("train") or []),
            "internal_val_allmodality_doseaware": sorted(
                str(x) for x in groups.get("internal_val_allmodality_doseaware") or []
            ),
        }
    return signature


def row_design(row: dict[str, Any], reference_signature: dict[str, Any] | None) -> dict[str, Any]:
    reasons: list[str] = []
    warnings: list[str] = []
    data_dir = Path(row["data_dir"])
    split_file = Path(row["split_file"])
    if not split_file.exists():
        return {"run_id": row["run_id"], "status": "fail", "reasons": ["missing_split_file"]}
    if not (data_dir / "sampled_indices_summary.json.gz").exists():
        return {"run_id": row["run_id"], "status": "fail", "reasons": ["missing_sampled_summary"]}
    split = load_json(split_file)
    summary = load_sampled_summary(data_dir)
    sig = condition_signature(split)
    if reference_signature is not None and sig != reference_signature:
        reasons.append("condition_identity_differs_across_budget_seed_rows")
    modality_counts = {"train_gene": 0, "eval_gene": 0, "train_chemical": 0, "eval_chemical": 0}
    train_eval_overlaps = []
    canonical_keys = []
    chemical_drug_train: dict[str, set[str]] = defaultdict(set)
    chemical_drug_eval: dict[str, set[str]] = defaultdict(set)
    for ds, groups in sorted(split.items()):
        if "canonical_test_reference" in groups:
            canonical_keys.append(ds)
        train = {str(x) for x in groups.get("train") or []}
        eval_set = {str(x) for x in groups.get("internal_val_allmodality_doseaware") or []}
        overlap = sorted(train & eval_set)
        if overlap:
            train_eval_overlaps.append({"dataset": ds, "n": len(overlap), "preview": overlap[:5]})
        is_chem = ds in SCIPLEX_DATASETS
        modality_counts["train_chemical" if is_chem else "train_gene"] += len(train)
        modality_counts["eval_chemical" if is_chem else "eval_gene"] += len(eval_set)
        if is_chem:
            ds_summary = summary.get(ds) or {}
            for cond, meta in ds_summary.items():
                cov_drug = str(meta.get("cov_drug", ""))
                role = str(meta.get("role", ""))
                if role == "train":
                    chemical_drug_train[ds].add(cov_drug)
                elif role == "eval":
                    chemical_drug_eval[ds].add(cov_drug)
    if canonical_keys:
        reasons.append(f"canonical_reference_key_present:{canonical_keys}")
    if train_eval_overlaps:
        reasons.append(f"train_eval_condition_overlap:{len(train_eval_overlaps)}")
    if any(v <= 0 for v in modality_counts.values()):
        reasons.append(f"empty_modality_axis:{modality_counts}")
    chemical_drug_overlap = {
        ds: sorted(chemical_drug_train[ds] & chemical_drug_eval[ds])
        for ds in sorted(SCIPLEX_DATASETS)
        if chemical_drug_train[ds] & chemical_drug_eval[ds]
    }
    if chemical_drug_overlap:
        warnings.append("chemical_internal_eval_is_dose_generalization_not_drug_holdout")
    return {
        "run_id": row["run_id"],
        "status": "ok" if not reasons else "fail",
        "reasons": reasons,
        "warnings": warnings,
        "budget": int(row["budget"]),
        "seed": int(row["seed"]),
        "modality_counts": modality_counts,
        "chemical_drug_train_eval_overlap_counts": {k: len(v) for k, v in chemical_drug_overlap.items()},
    }


def nested_budget_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect whether sampled keys are explicitly nested across budgets.

    The current materializer seeds sampling with the budget value, so this is
    expected to warn. We do not load all index arrays here; this is a design
    provenance check, not a heavy content comparison.
    """

    source = Path("/data/cyx/1030/scLatent/ops/materialize_latentfm_true_cell_count_allmodality_doseaware_capped_h5_20260625.py")
    text = source.read_text(encoding="utf-8") if source.exists() else ""
    budget_seeded = "|{budget}" in text or "seed}|{budget}" in text
    return {
        "status": "warn_not_nested" if budget_seeded else "unknown_check_content_if_claiming_law",
        "evidence": "sample_indices keys include budget, so samples are budget-specific rather than guaranteed nested"
        if budget_seeded
        else "could not prove budget-specific sampling from source text",
        "blocks_nm_scaling_law_claim": bool(budget_seeded),
        "blocks_bounded_gpu_smoke": False,
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM All-Modality Dose-Aware Design Controls",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only design/provenance gate.",
        "- Does not train, infer, read canonical metrics, read canonical multi, read held-out Track C query, or use GPU.",
        "- A pass here can support a bounded smoke only; NM-level scaling-law claims need nested/sample-identity controls and final strict statistics.",
        "",
        "## Matrix",
        "",
        f"- expected budgets: `{sorted(EXPECTED_BUDGETS)}`",
        f"- observed budgets: `{payload['observed_budgets']}`",
        f"- expected seeds: `{sorted(EXPECTED_SEEDS)}`",
        f"- observed seeds: `{payload['observed_seeds']}`",
        f"- full budget x seed grid: `{payload['full_grid']}`",
        f"- same condition identities across rows: `{payload['same_condition_identity_across_rows']}`",
        "",
        "## Budget Nesting",
        "",
        f"- status: `{payload['nested_budget_audit']['status']}`",
        f"- evidence: {payload['nested_budget_audit']['evidence']}",
        "",
        "## Rows",
        "",
        "| run id | status | modality counts | warnings | reasons |",
        "|---|---|---|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| `{row['run_id']}` | `{row['status']}` | `{row.get('modality_counts', {})}` | {', '.join(row.get('warnings') or []) or 'none'} | {', '.join(row.get('reasons') or []) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- GPU authorized by this gate alone: `{payload['gpu_authorized']}`",
            f"- smoke ready after schema/dryload pass: `{payload['smoke_ready_after_schema_dryload']}`",
            f"- NM scaling law ready: `{payload['nm_scaling_law_ready']}`",
            f"- next action: `{payload['next_action']}`",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    materializer = load_json(MATERIALIZER_JSON)
    schema = load_json(SCHEMA_JSON)
    dryload = load_json(DRYLOAD_JSON)
    feasibility = load_json(FEASIBILITY_JSON)
    materialized_rows = materializer.get("materialized_rows") or []
    reasons: list[str] = []
    if not materializer.get("materialized"):
        reasons.append("materializer_not_materialized")
    if not materialized_rows:
        reasons.append("no_materialized_rows")
    reference_signature = None
    if materialized_rows:
        first_split = load_json(Path(materialized_rows[0]["split_file"]))
        reference_signature = condition_signature(first_split)
    rows = [row_design(row, reference_signature) for row in materialized_rows]
    observed_budgets = sorted({int(r.get("budget")) for r in materialized_rows})
    observed_seeds = sorted({int(r.get("seed")) for r in materialized_rows})
    full_grid = {(b, s) for b in EXPECTED_BUDGETS for s in EXPECTED_SEEDS} == {
        (int(r.get("budget")), int(r.get("seed"))) for r in materialized_rows
    }
    same_condition_identity = all(r.get("status") == "ok" or "condition_identity_differs_across_budget_seed_rows" not in r.get("reasons", []) for r in rows)
    nested = nested_budget_audit(rows)
    if set(observed_budgets) != EXPECTED_BUDGETS:
        reasons.append(f"budget_grid_mismatch:{observed_budgets}")
    if set(observed_seeds) != EXPECTED_SEEDS:
        reasons.append(f"seed_grid_mismatch:{observed_seeds}")
    if not full_grid:
        reasons.append("budget_seed_grid_incomplete")
    failed_rows = [r for r in rows if r.get("status") != "ok"]
    if failed_rows:
        reasons.append(f"row_design_failures:{len(failed_rows)}")
    if feasibility.get("status") != "allmodality_doseaware_feasibility_pass_cpu_materializer_next":
        reasons.append(f"feasibility_not_pass:{feasibility.get('status')}")

    schema_pass = schema.get("status") == "allmodality_doseaware_schema_pass_no_gpu"
    dryload_pass = dryload.get("status") == "allmodality_doseaware_dryload_pass_no_gpu"
    structural_pass = not reasons
    smoke_ready_after_schema_dryload = bool(structural_pass and schema_pass and dryload_pass)
    nm_ready = bool(smoke_ready_after_schema_dryload and not nested.get("blocks_nm_scaling_law_claim"))
    if reasons:
        status = "allmodality_doseaware_design_not_ready_or_fail_no_gpu"
        next_action = "wait_for_materialization_or_fix_design_failures"
    elif not (schema_pass and dryload_pass):
        status = "allmodality_doseaware_design_pass_wait_schema_dryload_no_gpu"
        next_action = "run schema and dryload gates after materialization"
    elif nested.get("blocks_nm_scaling_law_claim"):
        status = "allmodality_doseaware_design_pass_smoke_nm_nested_warning_no_gpu"
        next_action = "bounded_gpu_smoke_allowed_after_resource_audit; build nested materializer before final NM scaling-law claim"
    else:
        status = "allmodality_doseaware_design_pass_no_gpu"
        next_action = "resource_audit_then_bounded_gpu_smoke"
    payload = {
        "status": status,
        "reasons": reasons,
        "inputs": {
            "materializer_json": str(MATERIALIZER_JSON),
            "schema_json": str(SCHEMA_JSON),
            "dryload_json": str(DRYLOAD_JSON),
            "feasibility_json": str(FEASIBILITY_JSON),
        },
        "observed_budgets": observed_budgets,
        "observed_seeds": observed_seeds,
        "full_grid": full_grid,
        "same_condition_identity_across_rows": same_condition_identity,
        "nested_budget_audit": nested,
        "rows": rows,
        "gpu_authorized": False,
        "smoke_ready_after_schema_dryload": smoke_ready_after_schema_dryload,
        "nm_scaling_law_ready": nm_ready,
        "next_action": next_action,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
