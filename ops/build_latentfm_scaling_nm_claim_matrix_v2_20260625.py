#!/usr/bin/env python3
"""Build Nature Methods-level scaling claim matrix v2.

CPU-only synthesis from completed gate reports. This is a claim/provenance
matrix, not a training launcher.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
EVIDENCE_JSON = REPORTS / "latentfm_scaling_evidence_table_20260625.json"
ALLMOD_JSON = REPORTS / "latentfm_true_cell_count_allmodality_label_compatibility_gate_20260625.json"
OUT_JSON = REPORTS / "latentfm_scaling_nm_claim_matrix_v2_20260625.json"
OUT_MD = REPORTS / "LATENTFM_SCALING_NM_CLAIM_MATRIX_V2_20260625.md"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def rows_by_axis(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row.get("axis") or "unknown"), []).append(row)
    return out


def claim_rows(evidence_rows: list[dict[str, Any]], allmod: dict[str, Any]) -> list[dict[str, Any]]:
    by_axis = rows_by_axis(evidence_rows)
    truecell_128 = next((r for r in by_axis.get("true_cell_count", []) if r.get("estimand") == "nested_6k_budget128"), {})
    truecell_64 = next((r for r in by_axis.get("true_cell_count", []) if r.get("estimand") == "nested_6k_budget64"), {})
    noharm_rows = by_axis.get("canonical_noharm", [])
    condition_rows = by_axis.get("condition_count_or_breadth", [])
    source_rows = by_axis.get("background_type_source", [])
    target_rows = by_axis.get("target_observability", [])
    chemical_rows = by_axis.get("chemical_holdout", [])
    noharm_transfer = by_axis.get("noharm_transfer", [])

    return [
        {
            "axis": "true_cell_cell_budget",
            "claim_status": "mechanism_supported_not_deployable",
            "best_evidence": "budget128 6k internal train-tail pass",
            "key_metric": {
                "cross_pp": truecell_128.get("primary_metric"),
                "family_pp": truecell_128.get("secondary_metric"),
                "neg_tails": truecell_128.get("tail_metric"),
            },
            "blocker": "frozen canonical no-harm failed for all 3 seeds; budget64 6k tail failure blocks budget256 curve completion",
            "source_reports": [
                truecell_128.get("source_report"),
                truecell_64.get("source_report"),
                "reports/LATENTFM_TRUE_CELL_COUNT_BUDGET128_6K_CANONICAL_NOHARM_DECISION_20260625.md",
                "reports/LATENTFM_TRUECELL_TAIL_TRANSFER_GATE_20260625.md",
            ],
            "gpu_authorized": False,
            "next_gate": "new non-duplicate CPU gate explaining tail/no-harm transfer before any budget256 or repair GPU",
        },
        {
            "axis": "condition_count_exposure",
            "claim_status": "diagnostic_only",
            "best_evidence": "moderate cap arms show local internal gains, but full exposure and seed-matched checks undermine monotonic law",
            "key_metric": {"arms": len(condition_rows)},
            "blocker": "seed sign flips and dataset tails; no deployable no-harm transfer",
            "source_reports": sorted({r.get("source_report") for r in condition_rows if r.get("source_report")}),
            "gpu_authorized": False,
            "next_gate": "CPU matched mixed-effect/LODO v2 with shuffled/count-only controls",
        },
        {
            "axis": "background_dataset_breadth",
            "claim_status": "not_supported_currently",
            "best_evidence": "source-resolved matched estimand remains confounded",
            "key_metric": source_rows[0] if source_rows else {},
            "blocker": "background/type NMI high and dataset min tail unsafe",
            "source_reports": sorted({r.get("source_report") for r in source_rows if r.get("source_report")}),
            "gpu_authorized": False,
            "next_gate": "source-verified crossed background/type estimand with label-shuffle and LODO tails",
        },
        {
            "axis": "target_gene_coverage",
            "claim_status": "hint_only_not_supported",
            "best_evidence": "target observability has local signal but unsafe tails and weak explanatory correlation",
            "key_metric": target_rows[0] if target_rows else {},
            "blocker": "target coverage/activity gate failed tail/control criteria",
            "source_reports": sorted({r.get("source_report") for r in target_rows if r.get("source_report")}),
            "gpu_authorized": False,
            "next_gate": "target observability v2 only if permutation/control and dataset-tail gate passes",
        },
        {
            "axis": "perturbation_type_and_all_modality",
            "claim_status": "blocked_by_label_compatibility",
            "best_evidence": "all-modality protocol contains dose-level chemical rows but current xverse H5/split are drug-level",
            "key_metric": allmod.get("summary", {}),
            "blocker": "direct dose-level H5/train/eval overlap 0/0/0; drug rollup collapses dose estimand and has no query-blind chemical eval",
            "source_reports": [
                "reports/LATENTFM_TRUE_CELL_COUNT_ALLMODALITY_LABEL_COMPATIBILITY_GATE_20260625.md",
                "reports/LATENTFM_TRUE_CELL_COUNT_ALLMODALITY_READINESS_GATE_20260625.md",
            ],
            "gpu_authorized": False,
            "next_gate": "dose-aware artifact path with query-blind chemical train/internal split, or separate drug-level branch without dose claim",
        },
        {
            "axis": "drug_dose_scaffold",
            "claim_status": "negative_currently",
            "best_evidence": "Morgan cache/provenance exists, but chemical holdout and residual semantic gates fail",
            "key_metric": {"chemical_rows": chemical_rows},
            "blocker": "chemical pp negative and descriptor/dose signals below promotion thresholds",
            "source_reports": sorted({r.get("source_report") for r in chemical_rows if r.get("source_report")}) + [
                "reports/LATENTFM_SCIPLEX_MORGAN_DESCRIPTOR_CACHE_20260624.md",
                "reports/LATENTFM_SCIPLEX_RESIDUAL_SEMANTIC_GATE_20260624.md",
                "reports/LATENTFM_SCIPLEX_DOSE_RANK_SEMANTIC_GATE_20260624.md",
            ],
            "gpu_authorized": False,
            "next_gate": "fresh chemical protocol only after dose-aware split/artifact compatibility is fixed",
        },
        {
            "axis": "noharm_transfer",
            "claim_status": "veto_only",
            "best_evidence": "internal-pass-like scaling candidates repeatedly fail frozen canonical no-harm",
            "key_metric": {"rows": noharm_transfer, "budget128_noharm_rows": noharm_rows},
            "blocker": "no positive no-harm calibration candidate from current scaling family",
            "source_reports": sorted({r.get("source_report") for r in noharm_transfer if r.get("source_report")}) + [
                "reports/LATENTFM_TRUE_CELL_COUNT_BUDGET128_6K_CANONICAL_NOHARM_DECISION_20260625.md",
                "reports/LATENTFM_NOHARM_CALIBRATION_POSITIVE_CONTROLS_GATE_20260624.md",
            ],
            "gpu_authorized": False,
            "next_gate": "route-freeze then canonical single/family no-harm veto only after train-tail/control pass",
        },
    ]


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Scaling NM Claim Matrix V2",
        "",
        "Status: `scaling_claim_matrix_v2_no_immediate_gpu`",
        "",
        "## Boundary",
        "",
        "- CPU-only synthesis of completed scaling reports and compatibility gates.",
        "- Does not train, infer, read canonical multi, read held-out Track C query, or use GPU.",
        "- Canonical single/family metrics appear only as frozen no-harm veto context.",
        "",
        "## Matrix",
        "",
        "| axis | claim status | best evidence | blocker | GPU authorized | next gate |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload["claim_rows"]:
        lines.append(
            "| `{axis}` | `{claim_status}` | {best_evidence} | {blocker} | `{gpu}` | {next_gate} |".format(
                axis=row["axis"],
                claim_status=row["claim_status"],
                best_evidence=row["best_evidence"],
                blocker=row["blocker"],
                gpu=row["gpu_authorized"],
                next_gate=row["next_gate"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Immediate scaling GPU authorized: `{payload['immediate_scaling_gpu_authorized']}`",
            "- Current default/deployable model remains `xverse_8k_anchor`.",
            "- Scaling remains active as mechanism/training-data construction evidence.",
            "- The shortest reopen path is not a duplicate GPU run; it is a dose-aware artifact/split gate or another fresh CPU gate that removes a specific blocker.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- Evidence input: `{EVIDENCE_JSON}`",
            f"- All-modality compatibility input: `{ALLMOD_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    evidence = load_json(EVIDENCE_JSON)
    allmod = load_json(ALLMOD_JSON)
    rows = claim_rows(evidence.get("rows", []), allmod)
    payload = {
        "status": "scaling_claim_matrix_v2_no_immediate_gpu",
        "boundary": {
            "cpu_only": True,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "uses_gpu": False,
        },
        "inputs": {"evidence_json": str(EVIDENCE_JSON), "allmodality_label_json": str(ALLMOD_JSON)},
        "claim_rows": rows,
        "immediate_scaling_gpu_authorized": any(bool(r.get("gpu_authorized")) for r in rows),
        "current_default_model": "xverse_8k_anchor",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
