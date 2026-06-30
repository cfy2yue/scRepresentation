#!/usr/bin/env python3
"""CPU-only train-only audit for the scFoundation Jiang near-miss harm mode."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
CPU_GATE = ROOT / "reports/latentfm_crosslatent_scfoundation_gene_reliability_router_gate_20260622.json"
CANON_GATE = ROOT / (
    "reports/latentfm_crosslatent_tracka_gene_reliability_adapter_"
    "scfoundation_tracka_gene_shrink_k2_adapter_2k_seed42_gate_20260623.json"
)
OUT_JSON = ROOT / "reports/latentfm_tracka_scf_jian_harm_trainonly_gate_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_SCF_JIANG_HARM_TRAINONLY_GATE_20260623.md"
HARM_DATASETS = ("Jiang_IFNG", "Jiang_TNFA")


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:+.6f}"


def trainonly_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("val_condition_rows") or []
    out = []
    for group in ("internal_val_cross_background_seen_gene_proxy", "internal_val_family_gene_proxy"):
        for ds in HARM_DATASETS:
            items = [r for r in rows if r.get("group") == group and r.get("dataset") == ds]
            if not items:
                continue
            shrink = mean(float(r["shrink_k2"]) for r in items)
            gene = mean(float(r["gene_raw_mean"]) for r in items)
            dataset = mean(float(r["dataset_mean"]) for r in items)
            out.append(
                {
                    "group": group,
                    "dataset": ds,
                    "n_conditions": len(items),
                    "shrink_k2": shrink,
                    "gene_raw_mean": gene,
                    "dataset_mean": dataset,
                    "shrink_minus_gene": shrink - gene,
                    "shrink_minus_dataset": shrink - dataset,
                }
            )
    return out


def canonical_harm_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in payload.get("paired_deltas") or []:
        if row.get("metric") != "pearson_pert":
            continue
        if row.get("stratum") not in {"cross_background_seen_gene", "all_test_single", "family_gene"}:
            continue
        by_ds = row.get("by_dataset") or {}
        for ds in HARM_DATASETS:
            if ds in by_ds:
                out.append(
                    {
                        "stratum": row["stratum"],
                        "dataset": ds,
                        "canonical_delta": float(by_ds[ds]),
                    }
                )
    return out


def decide(train_rows: list[dict[str, Any]], canonical_rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    if any(int(row["n_conditions"]) < 5 for row in train_rows):
        reasons.append("jiang_trainonly_support_too_small_for_gpu_unlock")
    if any(float(row["shrink_minus_dataset"]) < 0.0 for row in train_rows):
        reasons.append("trainonly_jiang_shrink_not_better_than_dataset_mean")
    if any(float(row["canonical_delta"]) < -0.02 for row in canonical_rows):
        reasons.append("canonical_jiang_material_harm_observed")
    return {
        "status": "jiang_harm_trainonly_gate_partial_no_gpu",
        "action": "design_cpu_targeted_correction_or_wait_for_scldm;do_not_launch_scf_gpu_followup_yet",
        "reasons": reasons,
        "gpu_authorization": "none",
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track A scFoundation Jiang Harm Train-Only Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        "GPU authorization: `none`",
        "",
        "## Scope",
        "",
        "This audit compares the frozen scFoundation train-only gene-reliability",
        "CPU gate against the completed canonical Track A near-miss gate. It does",
        "not read canonical multi or Track C query artifacts.",
        "",
        "## Train-Only Jiang Signal",
        "",
        "| group | dataset | n | shrink_k2 | gene_raw_mean | dataset_mean | shrink - gene | shrink - dataset |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["trainonly_rows"]:
        lines.append(
            f"| `{row['group']}` | `{row['dataset']}` | {row['n_conditions']} | "
            f"{fmt(row['shrink_k2'])} | {fmt(row['gene_raw_mean'])} | {fmt(row['dataset_mean'])} | "
            f"{fmt(row['shrink_minus_gene'])} | {fmt(row['shrink_minus_dataset'])} |"
        )
    lines.extend(
        [
            "",
            "## Canonical Harm Rows",
            "",
            "| stratum | dataset | canonical pp delta |",
            "|---|---|---:|",
        ]
    )
    for row in payload["canonical_harm_rows"]:
        lines.append(f"| `{row['stratum']}` | `{row['dataset']}` | {fmt(row['canonical_delta'])} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Train-only evidence already warns that `shrink_k2` is not better than",
            "`dataset_mean` for the harmed Jiang_IFNG/Jiang_TNFA backgrounds, but",
            "the support is only three internal-val conditions per dataset. This",
            "explains the canonical harm direction but does not authorize a GPU",
            "follow-up. A future scFoundation follow-up needs a predeclared",
            "CPU-targeted Jiang correction with stronger no-harm evidence.",
            "",
            "Reasons:",
            "",
        ]
    )
    lines.extend(f"- `{reason}`" for reason in payload["decision"]["reasons"])
    return "\n".join(lines) + "\n"


def main() -> None:
    train_payload = json.loads(CPU_GATE.read_text(encoding="utf-8"))
    canon_payload = json.loads(CANON_GATE.read_text(encoding="utf-8"))
    payload = {
        "cpu_gate": str(CPU_GATE),
        "canonical_gate": str(CANON_GATE),
        "leakage_status": "trainonly_gate_plus_frozen_canonical_gate_no_multi_no_query",
        "trainonly_rows": trainonly_rows(train_payload),
        "canonical_harm_rows": canonical_harm_rows(canon_payload),
    }
    payload["decision"] = decide(payload["trainonly_rows"], payload["canonical_harm_rows"])
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
