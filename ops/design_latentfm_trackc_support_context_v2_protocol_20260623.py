#!/usr/bin/env python3
"""Build the next Track C support-context v2 protocol/gate specification.

This is a design artifact, not an experiment launcher. It converts the current
negative and positive evidence into a narrow next-step protocol that can be
implemented only after CPU/code boundary checks pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"

SUPPORT_PROTOCOL_AUDIT = REPORTS / "latentfm_trackc_support_protocol_legitimacy_audit_20260623.json"
FROZEN_PACKAGE = REPORTS / "latentfm_frozen_diagnostic_reporting_package_20260623.json"
LEARNED_GATE = REPORTS / "latentfm_trackc_learned_anchor_gate_cpu_gate_20260623.json"
BIO_GATE = REPORTS / "latentfm_trackc_biological_prior_separability_20260623.json"
ROUTED_SMOKE_DECISION_MD = REPORTS / "LATENTFM_TRACKC_ROUTED_DISTILL_SMOKE_DECISION_xverse_trackc_route_condprior_w05_replay1_2k_seed42.md"
OUT_JSON = REPORTS / "latentfm_trackc_support_context_v2_protocol_20260623.json"
OUT_MD = REPORTS / "LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_PROTOCOL_20260623.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def md_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    for line in path.read_text(encoding="utf-8").splitlines()[:20]:
        if line.startswith("Status:"):
            return line.split("`")[1] if "`" in line else line.replace("Status:", "").strip()
    return "present"


def build_payload() -> dict[str, Any]:
    protocol = load_json(SUPPORT_PROTOCOL_AUDIT)
    frozen = load_json(FROZEN_PACKAGE)
    learned = load_json(LEARNED_GATE)
    bio = load_json(BIO_GATE)
    support = protocol["evidence"]["support_metrics"]
    collision = protocol["evidence"]["condition_only_collision"]

    return {
        "status": "trackc_support_context_v2_protocol_ready_cpu_code_gate_next_no_gpu",
        "gpu_authorization": "none",
        "hypothesis": (
            "Formal Track C progress should use an explicit support-set task interface, "
            "not a condition-only residual gate. A model may consume support context when "
            "the protocol supplies it, and must reduce to the anchor/no-op when support "
            "context is absent."
        ),
        "evidence": {
            "support_protocol_status": protocol["decision"]["status"],
            "frozen_package_status": frozen["status"],
            "routed_distill_status": md_status(ROUTED_SMOKE_DECISION_MD),
            "learned_gate_status": learned["decision"]["status"],
            "biological_gate_status": bio["decision"]["status"],
            "support_metrics": support,
            "condition_only_collision": collision,
        },
        "v2_task_definition": {
            "inputs": [
                "anchor condition/cell inputs",
                "explicit support set S from safe trainselect support protocol",
                "query condition to predict",
                "support-present flag supplied by protocol, not inferred from held-out outcome",
            ],
            "support_absent_behavior": "exact anchor/no-op for canonical Track A groups",
            "selection_split": str(ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"),
            "forbidden_inputs": [
                "held-out Track C query during selection",
                "canonical multi as a selection objective",
                "evaluation-scope labels as trainable features",
                "post-query failure labels",
                "condition-only GO/scGPT/CellNavi thresholds as residual gates",
            ],
        },
        "required_cpu_code_gate": [
            {
                "name": "support_context_interface_static_audit",
                "pass_rule": "training/eval config must expose explicit support-context-present and support-context-absent paths; support-absent path must be exact anchor/no-op in a synthetic forward test",
            },
            {
                "name": "split_boundary_audit",
                "pass_rule": "all selection and support banks read only split_seed42_multi_support_v2_trainselect.json; full v2 query and canonical multi selection are absent from launcher/config",
            },
            {
                "name": "negative_controls",
                "pass_rule": "zero-support and shuffled-support controls fail support material-gain gate before any GPU launch",
            },
            {
                "name": "frozen_route_plan",
                "pass_rule": "route/checkpoint/alpha/support protocol must be frozen before any one-shot query; query launcher must fail closed until support and uncapped no-harm pass",
            },
        ],
        "promotion_gate_after_one_capped_gpu": {
            "support_val_multi": [
                "Wessels pp delta >= +0.02",
                "Wessels route-gap closure >= +0.05",
                "Norman pp delta >= -0.02",
                "support paired pp p_harm <= 0.20",
                "support MMD p_harm <= 0.80",
            ],
            "canonical_noharm": [
                "test_single support-absent exact no-op or pp p_harm <= 0.35 and MMD p_harm <= 0.80",
                "family_gene support-absent exact no-op or pp p_harm <= 0.35 and MMD p_harm <= 0.80",
            ],
            "query": "held-out query allowed once only after route/checkpoint freeze and uncapped canonical no-harm pass",
        },
        "resource_plan_if_future_gpu_is_authorized": {
            "classification": "long task",
            "launch": "tmux/nohup with RUN_STATUS.md",
            "gpu_budget": "one capped smoke first; obey current AGENTS/user cap and fresh multi-sample audit",
            "cpu_budget": "<= 8 dataloader/eval workers for the capped smoke unless a new audit justifies more",
            "polling": "one lightweight startup verification, then no repeated checks before the AGENTS interval",
        },
        "stop_rules": [
            "CPU/code boundary gate fails -> no GPU",
            "support material gain fails -> close v2 support-context smoke",
            "canonical no-harm fails -> do not run query; close or redesign",
            "query is evaluated once only after all gates pass; never tune from query",
        ],
        "next_action": "Implement the CPU/code boundary gate for this v2 protocol; do not launch GPU from this protocol document alone.",
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track C Support-Context V2 Protocol",
        "",
        f"Status: `{payload['status']}`",
        f"GPU authorization: `{payload['gpu_authorization']}`",
        "",
        "## Hypothesis",
        "",
        payload["hypothesis"],
        "",
        "## Why This Is Not A Closed Threshold Rerun",
        "",
        "- Condition/train-metadata gates are closed.",
        "- GO/scGPT/CellNavi condition-only biological gates are closed.",
        "- This protocol requires support context as an explicit task input, and canonical Track A is support-context absent.",
        "",
        "## Evidence Snapshot",
        "",
    ]
    evidence = payload["evidence"]
    for key in (
        "support_protocol_status",
        "frozen_package_status",
        "routed_distill_status",
        "learned_gate_status",
        "biological_gate_status",
    ):
        lines.append(f"- `{key}`: `{evidence[key]}`")
    support = evidence["support_metrics"]
    collision = evidence["condition_only_collision"]
    lines.extend(
        [
            f"- support rows `{support['n_rows']}`, pp delta `{fmt(support['pearson_pert_delta'])}`, pp p_harm `{fmt(support['pearson_pert_p_harm'])}`, MMD delta `{fmt(support['mmd_delta'])}`, MMD p_harm `{fmt(support['mmd_p_harm'])}`",
            f"- condition-only collision usable support rows under exact family no-op: `{collision['max_support_rows_under_exact_family_noop']}`",
            "",
            "## V2 Task Definition",
            "",
        ]
    )
    for item in payload["v2_task_definition"]["inputs"]:
        lines.append(f"- input: {item}")
    lines.append(f"- support-absent behavior: {payload['v2_task_definition']['support_absent_behavior']}")
    lines.append(f"- selection split: `{payload['v2_task_definition']['selection_split']}`")
    lines.extend(["", "Forbidden inputs:"])
    for item in payload["v2_task_definition"]["forbidden_inputs"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Required CPU/Code Gate Before GPU", ""])
    for gate in payload["required_cpu_code_gate"]:
        lines.append(f"- `{gate['name']}`: {gate['pass_rule']}")
    lines.extend(["", "## Promotion Gate After One Capped GPU", ""])
    for section, rules in payload["promotion_gate_after_one_capped_gpu"].items():
        if isinstance(rules, list):
            lines.append(f"{section}:")
            for rule in rules:
                lines.append(f"- {rule}")
        else:
            lines.append(f"- {section}: {rules}")
    lines.extend(["", "## Resource Plan If Future GPU Is Authorized", ""])
    for key, value in payload["resource_plan_if_future_gpu_is_authorized"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Stop Rules", ""])
    for rule in payload["stop_rules"]:
        lines.append(f"- {rule}")
    lines.extend(["", "## Next Action", "", payload["next_action"], ""])
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.6f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_payload()
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(args.out_md), "out_json": str(args.out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
