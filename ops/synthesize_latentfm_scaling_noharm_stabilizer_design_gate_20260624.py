#!/usr/bin/env python3
"""Synthesize whether a scaling no-harm stabilizer is GPU-ready.

This is a CPU/report-only gate. It uses completed train-only and frozen
canonical no-harm reports as evidence, but does not use canonical metrics for
new checkpoint selection and does not read canonical multi or Track C query.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_scaling_noharm_stabilizer_design_gate_20260624.json"
OUT_MD = REPORTS / "LATENTFM_SCALING_NOHARM_STABILIZER_DESIGN_GATE_20260624.md"


def load_json(name: str) -> dict[str, Any]:
    path = REPORTS / name
    if not path.exists():
        return {"missing": True, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def get_status(payload: dict[str, Any]) -> str:
    decision = payload.get("decision")
    if isinstance(decision, dict) and decision.get("status"):
        return str(decision.get("status"))
    if isinstance(decision, dict) and decision.get("overall_status"):
        return str(decision.get("overall_status"))
    return str(payload.get("status", "missing" if payload.get("missing") else "unknown"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def first_row(payload: dict[str, Any], name_contains: str = "") -> dict[str, Any]:
    for row in payload.get("rows") or []:
        if not name_contains or name_contains in str(row.get("name") or row.get("run") or ""):
            return row
    return {}


def main() -> int:
    reports = {
        "count_internal": load_json("latentfm_xverse_scaling_count_smokes_decision_20260624.json"),
        "count_canonical": load_json("latentfm_xverse_scaling_canonical_noharm_decision_20260624.json"),
        "cap60_internal": load_json("latentfm_scaling_highthroughput_smokes_decision_20260624.json"),
        "cap60_canonical": load_json("latentfm_scaling_highthroughput_canonical_noharm_decision_20260624.json"),
        "response_internal": load_json("latentfm_scaling_cap60_response_repair_decision_20260624.json"),
        "response_seed": load_json("latentfm_scaling_cap60_response_seed_robustness_decision_20260624.json"),
        "response_canonical": load_json("latentfm_scaling_cap60_response_canonical_noharm_decision_20260624.json"),
        "noot_replay": load_json("latentfm_scaling_cap60_noot_replay_interaction_decision_20260624.json"),
        "pathway_mmd_smoke": load_json("latentfm_modality_pathway_mmd_preservation_smoke_decision_20260624.json"),
        "randomcount_mmd_gate": load_json("latentfm_randomcount_mmd_preservation_gate_20260624.json"),
        "training_closure": load_json("latentfm_training_data_normalization_closure_20260624.json"),
    }

    candidates = [
        {
            "stabilizer": "none/moderate cap count scaling",
            "train_only_signal": get_status(reports["count_internal"]),
            "frozen_noharm": get_status(reports["count_canonical"]),
            "metric": "cap120 internal cross +0.013077 vs anchor; canonical no-harm failed",
            "evidence": "reports/LATENTFM_XVERSE_SCALING_COUNT_SMOKES_DECISION_20260624.md; reports/LATENTFM_XVERSE_SCALING_CANONICAL_NOHARM_DECISION_20260624.md",
            "gpu_ready": False,
        },
        {
            "stabilizer": "more steps / light replay",
            "train_only_signal": get_status(reports["cap60_internal"]),
            "frozen_noharm": get_status(reports["cap60_canonical"]),
            "metric": "seed42 cap60/replay passed internal, seed43 failed; canonical no-harm failed",
            "evidence": "reports/LATENTFM_SCALING_HIGH_THROUGHPUT_SMOKES_DECISION_20260624.md; reports/LATENTFM_SCALING_HIGH_THROUGHPUT_CANONICAL_NOHARM_DECISION_20260624.md",
            "gpu_ready": False,
        },
        {
            "stabilizer": "response normalization + replay",
            "train_only_signal": get_status(reports["response_internal"]),
            "frozen_noharm": f"{get_status(reports['response_canonical'])}; seed robustness {get_status(reports['response_seed'])}",
            "metric": "seed42 internal pass, seed43 internal fail, canonical no-harm failed",
            "evidence": "reports/LATENTFM_SCALING_CAP60_RESPONSE_REPAIR_DECISION_20260624.md; reports/LATENTFM_SCALING_CAP60_RESPONSE_SEED_ROBUSTNESS_DECISION_20260624.md; reports/LATENTFM_SCALING_CAP60_RESPONSE_CANONICAL_NOHARM_DECISION_20260624.md",
            "gpu_ready": False,
        },
        {
            "stabilizer": "remove OT plus replay",
            "train_only_signal": get_status(reports["noot_replay"]),
            "frozen_noharm": "not reached",
            "metric": "cross/family pp hard harm and MMD harm",
            "evidence": "reports/LATENTFM_SCALING_CAP60_NOOT_REPLAY_INTERACTION_DECISION_20260624.md",
            "gpu_ready": False,
        },
        {
            "stabilizer": "pathway MMD-preserving composition",
            "train_only_signal": get_status(reports["pathway_mmd_smoke"]),
            "frozen_noharm": "not reached",
            "metric": "family pp/MMD improved but cross pp +0.007219 below +0.010 gate",
            "evidence": "reports/LATENTFM_MODALITY_PATHWAY_MMD_PRESERVATION_SMOKE_DECISION_20260624.md",
            "gpu_ready": False,
        },
        {
            "stabilizer": "random-count/downsampling control",
            "train_only_signal": get_status(reports["randomcount_mmd_gate"]),
            "frozen_noharm": "not reached",
            "metric": "Pearson clue but MMD tail harm across multiple datasets",
            "evidence": "reports/LATENTFM_RANDOMCOUNT_MMD_PRESERVATION_GATE_20260624.md",
            "gpu_ready": False,
        },
    ]

    reasons = [
        "all_completed_stabilizers_either_failed_internal_gate_or_failed_frozen_canonical_noharm",
        "seed42_internal_signals_are_not_seed_robust",
        "MMD-preserving_composition_trades_off_cross_background_signal",
        "randomcount_like_Pearson_gain_is_MMD_unsafe",
    ]
    payload = {
        "status": "scaling_noharm_stabilizer_design_gate_fail_no_gpu",
        "gpu_authorized": False,
        "boundary": {
            "reads_completed_reports_only": True,
            "canonical_noharm_used_as_veto_context": True,
            "canonical_multi_selection": False,
            "trackc_query_read": False,
            "training_or_inference": False,
        },
        "candidates": candidates,
        "reasons": reasons,
        "next_action": {
            "no_immediate_gpu": True,
            "shortest_path_to_gpu": [
                "Build a train-only failure-case localization gate that predicts canonical no-harm failure from internal strata without reading new canonical/query data.",
                "Require seed-robust internal cross pp >= +0.010, family pp no regression, family MMD <= +0.001, dataset-min pp >= -0.020, and negative controls collapse.",
                "Only then launch one bounded scaling smoke with RUN_STATUS and frozen posthoc no-harm veto.",
            ],
            "mainline_feedback": [
                "Use moderate capped exposure as discovery/default, not full data by default.",
                "Avoid hard type balance, random-count downsampling, simple replay/response normalization, and OT/no-OT pair-mode sweeps as already closed simple stabilizers.",
                "If scaling returns to GPU, it needs a new stabilizer that explicitly handles row-tail/canonical no-harm risk rather than only increasing train coverage.",
            ],
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Scaling No-Harm Stabilizer Design Gate",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only synthesis of completed train-only and frozen canonical no-harm reports.",
        "- Canonical no-harm is used as a veto/context signal, not as checkpoint selection.",
        "- Does not read canonical multi, Track C query, active logs, train, infer, or use GPU.",
        "",
        "## Stabilizer Evidence",
        "",
        "| stabilizer | train-only signal | no-harm / robustness | metric | GPU-ready | evidence |",
        "|---|---|---|---|---|---|",
    ]
    for row in candidates:
        lines.append(
            f"| {row['stabilizer']} | `{row['train_only_signal']}` | `{row['frozen_noharm']}` | "
            f"{row['metric']} | `{row['gpu_ready']}` | `{row['evidence']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- reasons: `{reasons}`",
            "- GPU authorized: `False`",
            "",
            "## Shortest Path Back To GPU",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["next_action"]["shortest_path_to_gpu"])
    lines.extend(["", "## Mainline Feedback", ""])
    lines.extend(f"- {item}" for item in payload["next_action"]["mainline_feedback"])
    lines.extend(["", "## JSON", "", f"`{OUT_JSON}`", ""])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "gpu_authorized": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
