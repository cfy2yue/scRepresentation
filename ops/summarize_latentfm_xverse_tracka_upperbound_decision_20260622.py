#!/usr/bin/env python3
"""Summarize Track A train-only upper-bound evidence after P10/P11.

This is a decision report, not a new model fit. It combines:

* the representation-normalization CPU gate that authorized P10;
* the actual P10 stablecaps decision;
* the full-train condition-source agreement covariate CPU gate.

The purpose is to decide whether more same-family Track A GPU smokes are
justified, or whether we should pause and require a new mechanism/information
source before spending more GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_REP_JSON = ROOT / "reports/latentfm_xverse_representation_normalization_covariate_gate_20260622.json"
DEFAULT_P10_DECISION_JSON = (
    ROOT
    / "reports/latentfm_xverse_response_repair_xverse_response_dscale_v2_aux025_replay1_4k_stablecaps_decision_20260622.json"
)
DEFAULT_SOURCE_JSON = ROOT / "reports/latentfm_xverse_condition_source_agreement_covariate_gate_fulltrain_20260622.json"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_tracka_upperbound_decision_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_TRACKA_UPPERBOUND_DECISION_20260622.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def find_rep_decision(rep: dict[str, Any], mode: str) -> dict[str, Any] | None:
    for row in rep.get("decision", {}).get("decisions", []):
        if row.get("mode") == mode:
            return row
    return None


def find_source_delta(source: dict[str, Any], group: str, baseline: str) -> dict[str, Any] | None:
    for row in source.get("paired_deltas") or []:
        if row.get("group") == group and row.get("baseline") == baseline:
            return row
    return None


def build_decision(rep: dict[str, Any], p10: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    rep_status = rep.get("decision", {}).get("overall_status")
    p10_status = p10.get("decision", {}).get("status")
    source_status = source.get("decision", {}).get("status")

    if rep_status == "cpu_gate_pass":
        reasons.append("representation_cpu_gate_passed_but_authorized_only_p10")
    else:
        reasons.append("representation_cpu_gate_not_pass")

    if p10_status != "stablecaps_ready_for_uncapped_posthoc":
        reasons.append(f"p10_not_ready_for_uncapped:{p10_status}")

    if source_status != "cpu_gate_pass_launch_one_tiny_gpu_smoke":
        reasons.append(f"condition_source_gate_failed:{source_status}")

    source_cross_dataset = find_source_delta(source, "internal_val_cross_background_seen_gene_proxy", "dataset_mean") or {}
    source_family_dataset = find_source_delta(source, "internal_val_family_gene_proxy", "dataset_mean") or {}
    if source_cross_dataset.get("delta_mean") is not None and float(source_cross_dataset["delta_mean"]) < 0:
        reasons.append("condition_source_agreement_worse_than_dataset_mean_control_cross_background")
    if source_family_dataset.get("delta_mean") is not None and float(source_family_dataset["delta_mean"]) < 0:
        reasons.append("condition_source_agreement_worse_than_dataset_mean_control_family")

    status = "stop_same_family_tracka_gpu_smokes_pending_new_mechanism"
    action = "mechanism_review_or_new_information_source_before_gpu"
    return {"status": status, "action": action, "reasons": reasons}


def render(payload: dict[str, Any]) -> str:
    rep_dec = find_rep_decision(payload["representation_gate"], "dataset_scale_ridge") or {}
    p10_dec = payload["p10_stablecaps_decision"].get("decision", {})
    source = payload["condition_source_gate"]
    src_cross_sc = find_source_delta(source, "internal_val_cross_background_seen_gene_proxy", "scgpt_ridge") or {}
    src_cross_dataset = find_source_delta(source, "internal_val_cross_background_seen_gene_proxy", "dataset_mean") or {}
    src_family_sc = find_source_delta(source, "internal_val_family_gene_proxy", "scgpt_ridge") or {}
    src_family_dataset = find_source_delta(source, "internal_val_family_gene_proxy", "dataset_mean") or {}

    lines = [
        "# LatentFM xverse Track A Upper-Bound Decision",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"Recommended action: `{payload['decision']['action']}`",
        "",
        "## Inputs",
        "",
        f"- representation-normalization CPU gate: `{payload['representation_gate_json']}`",
        f"- P10 stablecaps decision: `{payload['p10_stablecaps_decision_json']}`",
        f"- condition-source full-train CPU gate: `{payload['condition_source_gate_json']}`",
        "",
        "## Key Evidence",
        "",
        "| evidence | result | interpretation |",
        "|---|---|---|",
    ]
    c = rep_dec.get("cross_background_delta") or {}
    f = rep_dec.get("family_delta") or {}
    lines.append(
        "| representation CPU gate | "
        f"dataset_scale vs raw crossbg delta {fmt(c.get('delta'))}, family delta {fmt(f.get('delta'))} | "
        "authorized exactly one P10 smoke, not a sweep |"
    )
    lines.append(
        "| P10 stablecaps | "
        f"status `{p10_dec.get('status')}`, reasons `{', '.join(p10_dec.get('reasons') or []) or 'none'}` | "
        "MMD improves, but test/family pp harm risk blocks uncapped promotion |"
    )
    lines.append(
        "| condition-source agreement | "
        f"crossbg vs scGPT {fmt(src_cross_sc.get('delta_mean'))}, family vs scGPT {fmt(src_family_sc.get('delta_mean'))} | "
        "weak/unstable support, no GPU source adapter |"
    )
    lines.append(
        "| dataset-only control | "
        f"agreement vs dataset_mean crossbg {fmt(src_cross_dataset.get('delta_mean'))}, family {fmt(src_family_dataset.get('delta_mean'))} | "
        "train-only dataset/background residual dominates source-feature upper bound |"
    )
    lines += [
        "",
        "## Decision Reasons",
        "",
    ]
    lines.extend([f"- `{r}`" for r in payload["decision"].get("reasons") or []])
    lines += [
        "",
        "## Consequence",
        "",
        "- Do not launch another Track A GPU smoke from the same response-normalization, metric/loss, sampling, condition-source swap, or simple train-only covariate family.",
        "- The current stage result remains xverse/top-latent + 8k anchor; canonical multi stays diagnostic.",
        "- Next GPU requires a genuinely new CPU-gated mechanism, new latent-source evidence, or the separate Track C true-multi support/query protocol.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation-gate-json", type=Path, default=DEFAULT_REP_JSON)
    parser.add_argument("--p10-stablecaps-decision-json", type=Path, default=DEFAULT_P10_DECISION_JSON)
    parser.add_argument("--condition-source-gate-json", type=Path, default=DEFAULT_SOURCE_JSON)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    rep = load_json(args.representation_gate_json)
    p10 = load_json(args.p10_stablecaps_decision_json)
    source = load_json(args.condition_source_gate_json)
    payload = {
        "representation_gate_json": str(args.representation_gate_json),
        "p10_stablecaps_decision_json": str(args.p10_stablecaps_decision_json),
        "condition_source_gate_json": str(args.condition_source_gate_json),
        "representation_gate": rep,
        "p10_stablecaps_decision": p10,
        "condition_source_gate": source,
        "decision": build_decision(rep, p10, source),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
