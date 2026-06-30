#!/usr/bin/env python3
"""Triage whether soft-archetype pockets justify a new CPU gate.

This is intentionally read-only over existing train-only/internal-proxy
archetype diagnostics. It does not rerun model fitting, does not read canonical
test outputs, and does not authorize GPU. The purpose is to distinguish a
possibly useful pocket signal from a deployable route-selection rule.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
DATASET_EFFECTS = ROOT / "reports/latentfm_soft_archetype_dataset_effects_20260623.json"
OUT_JSON = ROOT / "reports/latentfm_soft_archetype_pocket_reopen_triage_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_SOFT_ARCHETYPE_POCKET_REOPEN_TRIAGE_20260623.md"

GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
BASELINE_FIELDS = (
    "delta_vs_dataset_mean",
    "delta_vs_gene_only_ridge",
    "delta_vs_gene_raw_mean",
    "delta_vs_soft_archetype_gene_shuffled_ridge",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.6f}"


def passes_pocket_row(row: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if int(row.get("n_conditions") or 0) < 5:
        reasons.append("n_conditions_below_5")
    for field in BASELINE_FIELDS:
        if float(row.get(field) if row.get(field) is not None else -999.0) < 0.02:
            reasons.append(f"{field}_below_0p02")
    if float(row.get("negative_fraction_vs_dataset_mean") or 1.0) > 0.25:
        reasons.append("negative_fraction_vs_dataset_mean_above_0p25")
    return not reasons, reasons


def build_payload() -> dict[str, Any]:
    effects = load_json(DATASET_EFFECTS)
    rows = effects.get("dataset_summary") or []
    by_dataset: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        group = str(row.get("group"))
        if group in GROUPS:
            by_dataset[str(row.get("dataset"))][group] = row

    pocket_rows: list[dict[str, Any]] = []
    diagnostic_candidates: list[dict[str, Any]] = []
    near_misses: list[dict[str, Any]] = []
    for dataset, group_rows in sorted(by_dataset.items()):
        if not all(group in group_rows for group in GROUPS):
            continue
        group_status = []
        all_pass = True
        for group in GROUPS:
            row = group_rows[group]
            passed, reasons = passes_pocket_row(row)
            all_pass = all_pass and passed
            group_status.append(
                {
                    "group": group,
                    "passed": passed,
                    "reasons": reasons,
                    "n_conditions": row.get("n_conditions"),
                    "delta_vs_dataset_mean": row.get("delta_vs_dataset_mean"),
                    "delta_vs_gene_only_ridge": row.get("delta_vs_gene_only_ridge"),
                    "delta_vs_gene_raw_mean": row.get("delta_vs_gene_raw_mean"),
                    "delta_vs_soft_archetype_gene_shuffled_ridge": row.get(
                        "delta_vs_soft_archetype_gene_shuffled_ridge"
                    ),
                    "negative_fraction_vs_dataset_mean": row.get("negative_fraction_vs_dataset_mean"),
                }
            )
        record = {"dataset": dataset, "all_groups_pass": all_pass, "groups": group_status}
        pocket_rows.append(record)
        if all_pass:
            diagnostic_candidates.append(record)
        elif any(
            float(g.get("delta_vs_dataset_mean") or 0.0) >= 0.10
            and int(g.get("n_conditions") or 0) >= 3
            for g in group_status
        ):
            near_misses.append(record)

    reasons = [
        "selection_would_be_internal_validation_discovered_not_train_discovered",
        "existing_aggregate_soft_archetype_predictive_gate_failed_noharm",
        "conditional_and_orthogonal_routers_failed_raw_or_shuffled_controls",
    ]
    if not diagnostic_candidates:
        reasons.append("no_reproducible_positive_pocket_under_strict_cross_group_rules")

    status = "soft_archetype_pocket_reopen_triage_close_gpu_candidate_diagnostic_only"
    return {
        "status": status,
        "gpu_authorization": "none",
        "next_authorization": "none",
        "leakage_status": "read_existing_trainonly_internal_proxy_diagnostics_no_canonical_no_query_no_active_gpu",
        "source": str(DATASET_EFFECTS),
        "rules": {
            "groups_required": list(GROUPS),
            "n_conditions_min_per_group": 5,
            "required_deltas": {field: ">= +0.02" for field in BASELINE_FIELDS},
            "negative_fraction_vs_dataset_mean": "<= 0.25",
        },
        "diagnostic_candidates": diagnostic_candidates,
        "near_misses": near_misses,
        "all_dataset_records": pocket_rows,
        "decision_reasons": reasons,
        "independent_review": {
            "agent": "Rawls/019ef2ac-13b5-79f1-af97-854c5b39488f",
            "conclusion": "close archetype/state-prior for now as a GPU/smoke candidate; same-feature threshold/router/ridge/abstain variants are not materially new",
        },
        "future_reopen_prerequisites": {
            "name": "materially_new_archetype_state_prior",
            "allowed_inputs": [
                "a genuinely new independent information source or mechanism beyond the existing hard/soft archetype, conditional router, orthogonal router, and multi-latent ridge feature families",
                "train-only/internal-proxy rows for discovery and a disjoint confirmation slice before any GPU authorization",
                "frozen provenance for any newly introduced biological/state prior",
            ],
            "forbidden_inputs": [
                "canonical test outputs for selection",
                "canonical test_multi",
                "Track C held-out query rows",
                "active GPU run predictions/logs",
                "dataset pockets selected directly from internal validation aggregate tables",
                "renamed threshold/router/ridge/abstain variants over the already failed feature families",
            ],
            "pass_criteria": [
                "the new source/mechanism explains why previously harmed pockets such as Dixit, Jiang_TGFB, Norman, and Papalexi are abstained or improved without validation-target selection",
                "confirmed deltas >= +0.02 vs dataset_mean, gene_raw, gene_only, and shuffled/permuted controls on disjoint train-only/internal confirmation rows",
                "bootstrap p_harm <= 0.20, leave-one-dataset or leave-one-pocket min >= -0.02, and no pocket chosen solely from internal validation aggregate effects",
            ],
            "stop_rule": "until such a genuinely new source/mechanism exists, archetype remains diagnostic-only and no GPU smoke is authorized",
        },
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Soft-Archetype Pocket Reopen Triage",
        "",
        f"Status: `{payload['status']}`",
        f"GPU authorization: `{payload['gpu_authorization']}`",
        f"Next authorization: `{payload['next_authorization']}`",
        "",
        "## Scope",
        "",
        "This triage reads only the existing train-only/internal-proxy dataset-effect diagnostic. It does not rerun fitting, read canonical test outputs, read Track C query rows, or inspect active GPU artifacts.",
        "",
        "## Diagnostic Candidates",
        "",
    ]
    candidates = payload.get("diagnostic_candidates") or []
    if not candidates:
        lines.append("- none")
    else:
        lines.extend(["| dataset | group | n | dataset | gene_only | gene_raw | shuffled | neg frac |", "|---|---|---:|---:|---:|---:|---:|---:|"])
        for record in candidates:
            for group in record["groups"]:
                lines.append(
                    f"| {record['dataset']} | {group['group']} | {group['n_conditions']} | "
                    f"{fmt(group['delta_vs_dataset_mean'])} | {fmt(group['delta_vs_gene_only_ridge'])} | "
                    f"{fmt(group['delta_vs_gene_raw_mean'])} | "
                    f"{fmt(group['delta_vs_soft_archetype_gene_shuffled_ridge'])} | "
                    f"{fmt(group['negative_fraction_vs_dataset_mean'])} |"
                )
    lines.extend(["", "## Near Misses", ""])
    near = payload.get("near_misses") or []
    if not near:
        lines.append("- none")
    else:
        lines.extend(["| dataset | group | pass | reasons | n | dataset delta | neg frac |", "|---|---|---|---|---:|---:|---:|"])
        for record in near:
            for group in record["groups"]:
                lines.append(
                    f"| {record['dataset']} | {group['group']} | {group['passed']} | "
                    f"`{';'.join(group['reasons']) or 'none'}` | {group['n_conditions']} | "
                    f"{fmt(group['delta_vs_dataset_mean'])} | {fmt(group['negative_fraction_vs_dataset_mean'])} |"
                )
    lines.extend(["", "## Decision Reasons", ""])
    lines.extend(f"- `{reason}`" for reason in payload.get("decision_reasons") or [])
    review = payload.get("independent_review") or {}
    lines.extend(
        [
            "",
            "## Independent Review",
            "",
            f"- agent: `{review.get('agent')}`",
            f"- conclusion: {review.get('conclusion')}",
        ]
    )
    next_gate = payload["future_reopen_prerequisites"]
    lines.extend(["", "## Future Reopen Prerequisites", "", f"- name: `{next_gate['name']}`", ""])
    lines.append("Allowed inputs:")
    lines.extend(f"- {item}" for item in next_gate["allowed_inputs"])
    lines.append("")
    lines.append("Forbidden inputs:")
    lines.extend(f"- {item}" for item in next_gate["forbidden_inputs"])
    lines.append("")
    lines.append("Pass criteria:")
    lines.extend(f"- {item}" for item in next_gate["pass_criteria"])
    lines.extend(["", f"Stop rule: {next_gate['stop_rule']}", ""])
    return "\n".join(lines)


def main() -> int:
    payload = build_payload()
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
