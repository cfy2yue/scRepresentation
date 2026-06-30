#!/usr/bin/env python3
"""Reconcile requested Track A metric provenance across existing artifacts.

CPU/report-only. It distinguishes canonical frozen posthoc groups from
train-only/internal-val proxy groups and candidate gate reports, so missing
canonical JSON labels are not confused with absent Track A evidence.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "reports/tracka_requested_metric_provenance_reconciliation_20260627"
OUT_JSON = ROOT / "reports/latentfm_tracka_requested_metric_provenance_reconciliation_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_REQUESTED_METRIC_PROVENANCE_RECONCILIATION_20260627.md"
OUT_CSV = OUT_DIR / "metric_provenance_matrix.csv"

ANCHOR_INTERNAL = ROOT / "reports/latentfm_xverse_tracka_anchor_internal_val_error_map_20260622.json"
DEPLOYABLE_TAXONOMY = ROOT / "reports/latentfm_tracka_deployable_benchmark_failure_taxonomy_20260627.json"
SCLDM_DECISION = ROOT / "reports/latentfm_tracka_scldm_guarded_fallback_adapter_scldm_tracka_gene_shrink_k4_dataset_negative_adapter_2k_seed42_gate_20260623.json"
TRACKA_FAILURE_GATE = ROOT / "reports/latentfm_tracka_failure_cluster_conditioned_trust_region_gate_20260627.json"
SIMPLE_SINGLE_EXACT = ROOT / "reports/latentfm_tracka_simple_single_unseen_exact_20260627.json"
ALL_TEST_SINGLE_EXACT = ROOT / "reports/latentfm_tracka_all_test_single_exact_20260627.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v):+.6f}"
    except Exception:
        return str(v)


def internal_abs_scores(anchor_internal: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in anchor_internal.get("dataset_summary", []):
        group = row.get("group")
        if not group:
            continue
        rec = out.setdefault(
            group,
            {
                "n_conditions": 0,
                "n_datasets": 0,
                "pp_dataset_sum": 0.0,
                "mmd_dataset_sum": 0.0,
                "aggregation": "equal_dataset_mean",
            },
        )
        n = int(row.get("n_conditions") or 0)
        rec["n_conditions"] += n
        rec["n_datasets"] += 1
        rec["pp_dataset_sum"] += float(row.get("anchor_pearson_pert") or 0.0)
        rec["mmd_dataset_sum"] += float(row.get("anchor_mmd_clamped") or 0.0)
    for rec in out.values():
        n = rec["n_datasets"]
        rec["anchor_pearson_pert"] = rec["pp_dataset_sum"] / n if n else None
        rec["anchor_mmd_clamped"] = rec["mmd_dataset_sum"] / n if n else None
        rec.pop("pp_dataset_sum", None)
        rec.pop("mmd_dataset_sum", None)
    return out


def group_lookup(taxonomy: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    out = {}
    for row in taxonomy.get("group_summary", []):
        out[(row.get("seed"), row.get("source"), row.get("group"))] = row
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    anchor_internal = load_json(ANCHOR_INTERNAL)
    taxonomy = load_json(DEPLOYABLE_TAXONOMY)
    failure_gate = load_json(TRACKA_FAILURE_GATE) if TRACKA_FAILURE_GATE.exists() else {}
    scldm = load_json(SCLDM_DECISION) if SCLDM_DECISION.exists() else {}
    simple_exact = load_json(SIMPLE_SINGLE_EXACT) if SIMPLE_SINGLE_EXACT.exists() else {}
    all_single_exact = load_json(ALL_TEST_SINGLE_EXACT) if ALL_TEST_SINGLE_EXACT.exists() else {}
    internal = internal_abs_scores(anchor_internal)
    groups = group_lookup(taxonomy)
    simple_seed42 = (
        simple_exact.get("summaries", {})
        .get("seed42", {})
        .get("simple_single_unseen", {})
        .get("metrics", {})
    )
    simple_seed43 = (
        simple_exact.get("summaries", {})
        .get("seed43", {})
        .get("simple_single_unseen", {})
        .get("metrics", {})
    )
    simple_seed42_meta = simple_exact.get("summaries", {}).get("seed42", {}).get("simple_single_unseen", {})
    all_single_summaries = {row.get("seed"): row for row in all_single_exact.get("summaries", [])}
    all_single_seed42 = all_single_summaries.get("seed42", {})
    all_single_seed43 = all_single_summaries.get("seed43", {})

    rows = [
        {
            "requested_metric": "cross_background_seen_gene",
            "exact_or_proxy": "proxy",
            "available_in_canonical_seed42_seed43_full_eval_json": False,
            "available_as_trainonly_internal_val": True,
            "authoritative_artifact": str(ANCHOR_INTERNAL),
            "group_or_label": "internal_val_cross_background_seen_gene_proxy",
            "selection_role": "train-only/internal-val mechanism diagnostic; not canonical selection",
            "anchor_seed42_value": internal.get("internal_val_cross_background_seen_gene_proxy", {}).get("anchor_pearson_pert"),
            "anchor_seed43_value": None,
            "mmd_value": internal.get("internal_val_cross_background_seen_gene_proxy", {}).get("anchor_mmd_clamped"),
            "n_conditions": internal.get("internal_val_cross_background_seen_gene_proxy", {}).get("n_conditions"),
            "n_datasets": internal.get("internal_val_cross_background_seen_gene_proxy", {}).get("n_datasets"),
            "current_decision": anchor_internal.get("decision", {}).get("status"),
            "gpu_authorized": False,
            "note": "Not present as an explicit canonical frozen full-eval group; available as train-only internal-val proxy used by Track A gates. Value is equal-dataset mean, matching the source report aggregation.",
        },
        {
            "requested_metric": "all_test_single",
            "exact_or_proxy": "exact_evaluator",
            "available_in_canonical_seed42_seed43_full_eval_json": True,
            "available_as_trainonly_internal_val": False,
            "authoritative_artifact": str(ALL_TEST_SINGLE_EXACT if all_single_exact else DEPLOYABLE_TAXONOMY),
            "group_or_label": "test_single",
            "selection_role": "canonical all-test-single descriptive/no-harm context",
            "anchor_seed42_value": all_single_seed42.get("pearson_pert_mean") or groups.get(("seed42", "family", "test_single"), {}).get("reported_pearson_pert"),
            "anchor_seed43_value": all_single_seed43.get("pearson_pert_mean") or groups.get(("seed43", "family", "test_single"), {}).get("reported_pearson_pert"),
            "mmd_value": all_single_seed42.get("test_mmd_clamped_mean") or groups.get(("seed42", "family", "test_single"), {}).get("reported_test_mmd_clamped"),
            "n_conditions": all_single_seed42.get("n_conditions") or groups.get(("seed42", "family", "test_single"), {}).get("n_conds"),
            "n_datasets": all_single_seed42.get("n_datasets"),
            "current_decision": all_single_exact.get("status") or "proxy_available_as_test_single",
            "gpu_authorized": False,
            "note": "Exact CPU-only provenance audit verified frozen posthoc test_single rows match split_seed42.json test_single 540/540 for seed42 and seed43.",
        },
        {
            "requested_metric": "family_gene",
            "exact_or_proxy": "exact",
            "available_in_canonical_seed42_seed43_full_eval_json": True,
            "available_as_trainonly_internal_val": True,
            "authoritative_artifact": str(DEPLOYABLE_TAXONOMY),
            "group_or_label": "family_gene",
            "selection_role": "Track A primary/no-harm context",
            "anchor_seed42_value": groups.get(("seed42", "family", "family_gene"), {}).get("reported_pearson_pert"),
            "anchor_seed43_value": groups.get(("seed43", "family", "family_gene"), {}).get("reported_pearson_pert"),
            "mmd_value": groups.get(("seed42", "family", "family_gene"), {}).get("reported_test_mmd_clamped"),
            "n_conditions": groups.get(("seed42", "family", "family_gene"), {}).get("n_conds"),
            "n_datasets": None,
            "current_decision": "available_and_stable_seed42_seed43",
            "gpu_authorized": False,
            "note": "Canonical full-eval exact group; seed replicate stable.",
        },
        {
            "requested_metric": "simple_single_unseen",
            "exact_or_proxy": "exact_evaluator",
            "available_in_canonical_seed42_seed43_full_eval_json": bool(simple_exact),
            "available_as_trainonly_internal_val": False,
            "authoritative_artifact": str(SIMPLE_SINGLE_EXACT if simple_exact else DEPLOYABLE_TAXONOMY),
            "group_or_label": "simple_single_unseen",
            "selection_role": "canonical Track A descriptive evaluator; not candidate selection by itself",
            "anchor_seed42_value": simple_seed42.get("pearson_pert", {}).get("mean"),
            "anchor_seed43_value": simple_seed43.get("pearson_pert", {}).get("mean"),
            "mmd_value": simple_seed42.get("test_mmd_clamped", {}).get("mean"),
            "n_conditions": simple_seed42_meta.get("n_conditions"),
            "n_datasets": simple_seed42_meta.get("n_datasets"),
            "current_decision": simple_exact.get("status") or "not_available_in_existing_anchor_json_or_internal_val_artifacts",
            "gpu_authorized": False,
            "note": "Exact CPU-only evaluator built from canonical test_single, one non-drug gene target, and canonical train target visibility; no canonical multi or Track C query.",
        },
        {
            "requested_metric": "canonical_multi",
            "exact_or_proxy": "diagnostic",
            "available_in_canonical_seed42_seed43_full_eval_json": True,
            "available_as_trainonly_internal_val": False,
            "authoritative_artifact": str(DEPLOYABLE_TAXONOMY),
            "group_or_label": "test_multi*",
            "selection_role": "diagnostic_zero_selection_weight",
            "anchor_seed42_value": groups.get(("seed42", "family", "test_multi_unseen2"), {}).get("reported_pearson_pert"),
            "anchor_seed43_value": groups.get(("seed43", "family", "test_multi_unseen2"), {}).get("reported_pearson_pert"),
            "mmd_value": groups.get(("seed42", "family", "test_multi_unseen2"), {}).get("reported_test_mmd_clamped"),
            "n_conditions": groups.get(("seed42", "family", "test_multi_unseen2"), {}).get("n_conds"),
            "n_datasets": None,
            "current_decision": "diagnostic_zero_selection_weight",
            "gpu_authorized": False,
            "note": "Only zero-shot composition diagnostic. Never Track A checkpoint/model selection.",
        },
    ]

    fields = list(rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "status": "tracka_requested_metric_provenance_reconciled_no_gpu",
        "gpu_authorized": False,
        "default_model": "xverse_8k_anchor",
        "boundary": {
            "cpu_report_only": True,
            "no_training": True,
            "no_inference": True,
            "no_checkpoint_selection": True,
            "canonical_multi_selection_weight": 0,
            "trackc_query_read": False,
        },
        "inputs": {
            "anchor_internal_val_error_map": str(ANCHOR_INTERNAL),
            "deployable_taxonomy": str(DEPLOYABLE_TAXONOMY),
            "failure_cluster_gate": str(TRACKA_FAILURE_GATE),
            "scldm_decision_optional": str(SCLDM_DECISION),
            "simple_single_unseen_exact": str(SIMPLE_SINGLE_EXACT),
            "all_test_single_exact": str(ALL_TEST_SINGLE_EXACT),
        },
        "metric_rows": rows,
        "failure_cluster_gate_status": failure_gate.get("status"),
        "scldm_crossbg_delta_if_available": scldm.get("summary", {}).get("cross_background_seen_gene", {}).get("pearson_pert_delta") if isinstance(scldm.get("summary"), dict) else None,
        "outputs": {"csv": str(OUT_CSV), "json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Track A Requested Metric Provenance Reconciliation",
        "",
        "Status: `tracka_requested_metric_provenance_reconciled_no_gpu`",
        "",
        "GPU authorized: `False`",
        "",
        "Default/deployable model: `xverse_8k_anchor`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only synthesis of existing reports.",
        "- Separates canonical frozen posthoc groups from train-only internal-val proxy groups.",
        "- Does not train, infer, select checkpoints, read Track C query, or use canonical multi for selection.",
        "",
        "## Metric Matrix",
        "",
        "| requested metric | exact/proxy | canonical full-eval | train-only internal-val | group | seed42/proxy pp | seed43 pp | MMD | decision |",
        "|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['requested_metric']}` | `{row['exact_or_proxy']}` | "
            f"`{row['available_in_canonical_seed42_seed43_full_eval_json']}` | "
            f"`{row['available_as_trainonly_internal_val']}` | `{row['group_or_label']}` | "
            f"{fmt(row['anchor_seed42_value'])} | {fmt(row['anchor_seed43_value'])} | {fmt(row['mmd_value'])} | "
            f"`{row['current_decision']}` |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "- `cross_background_seen_gene` is not an explicit canonical full-eval group, but it is available as the train-only internal-val proxy used by Track A gates.",
        "- `all_test_single` now has exact provenance: frozen `test_single` rows match canonical split `test_single` 540/540 for both seed42 and seed43.",
        "- `simple_single_unseen` now has an exact CPU-only evaluator over canonical `test_single` rows and train-target visibility.",
        "- Canonical `test_multi*` remains diagnostic with selection weight `0`.",
        "- No GPU is authorized by this reconciliation.",
        "",
        "## Outputs",
        "",
        f"- CSV: `{OUT_CSV}`",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
