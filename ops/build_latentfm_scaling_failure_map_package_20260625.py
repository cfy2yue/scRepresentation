#!/usr/bin/env python3
"""Build a LatentFM scaling/failure-map package.

CPU-only reporting package. It consolidates the current scaling evidence,
negative gates, claim boundaries, and mainline-use guidance into a single
provenance-indexed report. It does not read checkpoints, canonical multi,
Track C query, train, infer, or use GPU.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_scaling_failure_map_package_20260625.json"
OUT_MD = REPORTS / "LATENTFM_SCALING_FAILURE_MAP_PACKAGE_20260625.md"

INPUTS = {
    "evidence_table_json": REPORTS / "latentfm_scaling_evidence_table_20260625.json",
    "s0_json": REPORTS / "latentfm_scaling_s0_provenance_freeze_20260625.json",
    "scaling_status": REPORTS / "LATENTFM_SCALING_STATUS_COMPLETION_AND_MAINLINE_USE_20260625.md",
    "nested_condition_exposure_v2": REPORTS / "latentfm_scaling_nested_condition_exposure_v2_gate_20260625.json",
    "source_resolved_v2": REPORTS / "latentfm_scaling_source_resolved_estimand_v2_gate_20260625.json",
    "tail_noharm_reopenability": REPORTS / "latentfm_tail_noharm_reopenability_audit_20260625.json",
    "truecell_budget128_6k": REPORTS / "latentfm_true_cell_count_budget128_tail_stability_6k_decision_20260625.json",
    "truecell_budget128_noharm": REPORTS / "latentfm_true_cell_count_budget128_6k_canonical_noharm_decision_20260625.json",
    "allmod_family": REPORTS / "latentfm_allmodality_family_stratified_protocol_gate_20260625.json",
    "chemical_v2_ack": REPORTS / "LATENTFM_CHEMICAL_V2_LAUNCH_ACK_EXTERNAL_AUDIT_HERSCHEL_20260625.md",
}


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"_missing": True}
    with path.open() as f:
        return json.load(f)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def evidence_rows() -> list[dict[str, Any]]:
    data = load_json(INPUTS["evidence_table_json"])
    return data.get("rows", [])


def main() -> int:
    s0 = load_json(INPUTS["s0_json"])
    nested = load_json(INPUTS["nested_condition_exposure_v2"])
    source = load_json(INPUTS["source_resolved_v2"])
    tail = load_json(INPUTS["tail_noharm_reopenability"])
    truecell = load_json(INPUTS["truecell_budget128_6k"])
    noharm = load_json(INPUTS["truecell_budget128_noharm"])
    allmod = load_json(INPUTS["allmod_family"])

    truecell_row = ((truecell.get("matrix_summary") or {}).get("budget_rows") or [{}])[0]
    noharm_rows = noharm.get("rows") or []
    noharm_cross = [
        (((r.get("metrics") or {}).get("cross_background_seen_gene:pearson_pert") or {}).get("delta_mean"))
        for r in noharm_rows
    ]
    noharm_family_pharm = [
        (((r.get("metrics") or {}).get("family_gene:pearson_pert") or {}).get("p_harm"))
        for r in noharm_rows
    ]

    axes = [
        {
            "axis": "true_cell_budget",
            "claim": "strongest mechanism signal",
            "support": f"budget128 6k internal cross/family/MMD {fmt(truecell_row.get('cross_background_pp_delta_mean'))}/{fmt(truecell_row.get('family_gene_pp_delta_mean'))}/{fmt(truecell_row.get('family_gene_mmd_delta_mean'))}",
            "boundary": "mechanism only; frozen canonical no-harm failed all seeds",
            "next_gate": "new non-noop tail/no-harm mechanism before any GPU promotion",
        },
        {
            "axis": "condition_count_exposure",
            "claim": "moderate exposure local signal, not monotonic law",
            "support": f"cap120-cap30 {fmt((nested.get('summary') or {}).get('cap120_minus_cap30_cross_pp'))}; full-cap120 {fmt((nested.get('summary') or {}).get('full_minus_cap120_cross_pp'))}",
            "boundary": "seed flip, LODO tails, and canonical no-harm veto block GPU",
            "next_gate": "do not rerun old cap/full/breadth arms; new mechanism only",
        },
        {
            "axis": "background_type_source",
            "claim": "failure map, no clean scaling claim",
            "support": f"source-resolved pp {fmt((source.get('summary') or {}).get('pp_delta_mean'))}; min {fmt((source.get('summary') or {}).get('dataset_min_pp'))}",
            "boundary": "S0 source-resolved filtering does not rescue background/type scaling",
            "next_gate": "only a new matched estimand with positive CI/tails could reopen",
        },
        {
            "axis": "target_actionability",
            "claim": "hint only, unsafe tails",
            "support": "high-actionability positive mean but CI/control/tails fail",
            "boundary": "do not use target/actionability as GPU route",
            "next_gate": "new target gate must beat permutation and dataset-tail veto",
        },
        {
            "axis": "chemical_semantics",
            "claim": "protocol-ready branch, not robust claim",
            "support": "V2 fixed-step controls prepared; exact ACK required",
            "boundary": "seed controls and hint controls make current chemical evidence diagnostic",
            "next_gate": "chemical V2 real seed43/44 only after exact ACK, then shuffled/random controls",
        },
        {
            "axis": "allmodality_type",
            "claim": "failure map",
            "support": f"family-stratified status {allmod.get('status')}",
            "boundary": "0 passing family/dose/pathway policies under hard-harm/tail/shuffle criteria",
            "next_gate": "do not expand allmod GPU without new CPU mechanism",
        },
        {
            "axis": "tail_noharm_mechanisms",
            "claim": "no current reopenability",
            "support": f"{(tail.get('summary') or {}).get('n_gpu_authorized', 0)} gates authorize GPU",
            "boundary": f"closed families {(tail.get('summary') or {}).get('closed_families', [])}",
            "next_gate": "invent materially new CPU-first mechanism or move to reporting",
        },
    ]

    figure_candidates = [
        "S0 provenance table: dataset/source/background/type/modality coverage and unresolved exclusions",
        "True-cell budget curve: budget64/128/256 and budget128 6k with seed bootstrap/tails",
        "Condition/exposure non-monotonicity: cap30/cap60/cap120/full plus breadth arms",
        "No-harm veto panel: budget128 6k canonical seed42/43/44 cross/family p_harm",
        "Failure-map heat table: background/type/source, target, chemical, allmodality, OT, tail/no-harm gates",
    ]

    payload = {
        "status": "scaling_failure_map_package_ready_no_gpu",
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "reads_completed_reports": True,
            "reads_checkpoints": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "default_model": "xverse_8k_anchor",
        "claim_boundary": {
            "allowed": "condition-level data-axis mechanism/failure map with true-cell and moderate-exposure local signals",
            "not_allowed": "deployable scaling law, checkpoint replacement, monotonic more-data law, background/type law, or chemical specificity claim",
        },
        "s0_summary": s0.get("summary", {}),
        "axes": axes,
        "canonical_noharm_veto": {
            "cross_background_pp_deltas": noharm_cross,
            "family_gene_p_harm": noharm_family_pharm,
        },
        "mainline_guidance": [
            "keep xverse_8k_anchor as default",
            "prefer moderate per-condition cell budget and controlled exposure",
            "require source/background/type tail audits before adding breadth",
            "do not default to naive full-data, hard balancing, generic weighted loss, response normalization, or OT sweeps",
            "treat failed scaling axes as hard-negative controls for future sampler/loss design",
        ],
        "figure_candidates": figure_candidates,
        "inputs": {
            k: {"path": str(v), "exists": v.exists(), "sha256": sha256(v)}
            for k, v in INPUTS.items()
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# LatentFM Scaling Failure-Map Package",
        "",
        "Status: `scaling_failure_map_package_ready_no_gpu`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only reporting package over completed reports.",
        "- Does not read checkpoints, canonical multi, Track C held-out query, train, infer, or use GPU.",
        "",
        "## Claim Boundary",
        "",
        f"- Allowed: {payload['claim_boundary']['allowed']}.",
        f"- Not allowed: {payload['claim_boundary']['not_allowed']}.",
        "- Default/deployable model remains `xverse_8k_anchor`.",
        "",
        "## S0 Provenance",
        "",
        f"- rows: `{(s0.get('summary') or {}).get('n_rows')}`",
        f"- datasets: `{(s0.get('summary') or {}).get('n_datasets')}`",
        f"- source-verified rows: `{(s0.get('summary') or {}).get('n_source_verified')}`",
        f"- resolved gene/nonchemical rows: `{(s0.get('summary') or {}).get('n_s0_resolved')}`",
        "",
        "## Axis Summary",
        "",
        "| axis | claim | support | boundary | next gate |",
        "|---|---|---|---|---|",
    ]
    for axis in axes:
        lines.append(
            f"| `{axis['axis']}` | {axis['claim']} | {axis['support']} | {axis['boundary']} | {axis['next_gate']} |"
        )
    lines.extend(
        [
            "",
            "## Mainline Guidance",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["mainline_guidance"])
    lines.extend(
        [
            "",
            "## Figure / Table Candidates",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in figure_candidates)
    lines.extend(
        [
            "",
            "## Input Artifacts",
            "",
            "| input | exists | sha256 | path |",
            "|---|---:|---|---|",
        ]
    )
    for name, meta in payload["inputs"].items():
        lines.append(f"| `{name}` | `{meta['exists']}` | `{meta['sha256']}` | `{meta['path']}` |")
    lines.extend(["", "## JSON", "", f"`{OUT_JSON}`"])
    OUT_MD.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
