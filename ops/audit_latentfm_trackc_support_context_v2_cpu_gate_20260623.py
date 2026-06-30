#!/usr/bin/env python3
"""CPU gate for Track C support-context v2 after exact-noop code boundary.

This combines the protocol spec, code-boundary audit, frozen support-teacher
CPU gate, and full posthoc MMD/no-harm gate. It does not train, evaluate query,
or authorize GPU directly; passing means the next step is a launcher/provenance
gate with AGENTS.md resource audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
PROTOCOL_JSON = REPORTS / "latentfm_trackc_support_context_v2_protocol_20260623.json"
CODE_BOUNDARY_JSON = REPORTS / "latentfm_trackc_support_context_v2_code_boundary_20260623.json"
SUPPORT_CPU_JSON = REPORTS / "latentfm_trackc_anchor_gated_support_teacher_cpu_gate_20260623.json"
POSTHOC_JSON = REPORTS / "latentfm_trackc_anchor_gated_support_teacher_blend_posthoc_gate_20260623.json"
BIO_JSON = REPORTS / "latentfm_trackc_biological_prior_separability_20260623.json"
OUT_JSON = REPORTS / "latentfm_trackc_support_context_v2_cpu_gate_20260623.json"
OUT_MD = REPORTS / "LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_CPU_GATE_20260623.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_path(obj: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def support_dataset(summary: dict[str, Any], dataset: str) -> dict[str, Any]:
    for row in summary.get("dataset_summary") or []:
        if row.get("dataset") == dataset:
            return row
    return {}


def build_payload() -> dict[str, Any]:
    protocol = load_json(PROTOCOL_JSON)
    code = load_json(CODE_BOUNDARY_JSON)
    support_cpu = load_json(SUPPORT_CPU_JSON)
    posthoc = load_json(POSTHOC_JSON)
    bio = load_json(BIO_JSON)

    support_summary = support_cpu.get("selected_support_alpha_summary", {})
    shuffled = support_cpu.get("selected_shuffled_summary", {})
    wessels = support_dataset(support_summary, "Wessels")
    norman = support_dataset(support_summary, "NormanWeissman2019_filtered")
    shuffled_wessels = support_dataset(shuffled, "Wessels")
    support_paired = support_summary.get("paired", {})
    posthoc_support = posthoc.get("support", {})
    pp = posthoc_support.get("pearson_pert_delta", {})
    mmd = posthoc_support.get("test_mmd_delta", {})
    canonical_noop = posthoc.get("canonical_noop", {})
    collision = bio.get("collision", {})

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, evidence: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    add(
        "protocol_ready_no_gpu",
        protocol.get("status") == "trackc_support_context_v2_protocol_ready_cpu_code_gate_next_no_gpu"
        and protocol.get("gpu_authorization") == "none",
        f"protocol status={protocol.get('status')} gpu={protocol.get('gpu_authorization')}",
    )
    add(
        "code_boundary_pass",
        code.get("status") == "trackc_support_context_v2_code_boundary_pass_cpu_gate_next"
        and code.get("next_authorization") == "cpu_gate_only"
        and not code.get("hard_failures"),
        f"code status={code.get('status')} next={code.get('next_authorization')} hard_failures={len(code.get('hard_failures') or [])}",
    )
    add(
        "condition_only_gates_closed",
        collision.get("max_support_rows_under_exact_family_noop") == 0,
        f"condition-only usable support rows={collision.get('max_support_rows_under_exact_family_noop')}",
    )
    add(
        "mean_vector_support_gate_pass",
        get_path(support_cpu, ["decision", "status"]) == "trackc_anchor_gated_support_teacher_cpu_gate_pass_code_gate_next",
        f"support CPU gate status={get_path(support_cpu, ['decision', 'status'])}",
    )
    add(
        "support_material_gain",
        fnum(wessels.get("mean_delta_pp")) >= 0.02
        and fnum(wessels.get("route_gap_closed_fraction")) >= 0.05
        and fnum(norman.get("mean_delta_pp")) >= -0.02
        and fnum(support_paired.get("p_harm"), 1.0) <= 0.20,
        (
            f"Wessels delta={wessels.get('mean_delta_pp')} closure={wessels.get('route_gap_closed_fraction')}; "
            f"Norman delta={norman.get('mean_delta_pp')}; paired p_harm={support_paired.get('p_harm')}"
        ),
    )
    add(
        "negative_controls_fail",
        fnum(shuffled_wessels.get("mean_delta_pp"), 999.0) < 0.02
        and fnum(shuffled_wessels.get("route_gap_closed_fraction"), 999.0) < 0.05,
        f"shuffled Wessels delta={shuffled_wessels.get('mean_delta_pp')} closure={shuffled_wessels.get('route_gap_closed_fraction')}",
    )
    add(
        "posthoc_support_pp_mmd_pass",
        fnum(pp.get("observed")) >= 0.02
        and fnum(pp.get("p_harm_pp"), 1.0) <= 0.20
        and fnum(mmd.get("observed"), 999.0) <= 0.0
        and fnum(mmd.get("p_harm_mmd"), 1.0) <= 0.80,
        (
            f"posthoc pp={pp.get('observed')} pp_harm={pp.get('p_harm_pp')} "
            f"mmd={mmd.get('observed')} mmd_harm={mmd.get('p_harm_mmd')}"
        ),
    )
    add(
        "canonical_support_absent_exact_noop",
        fnum(get_path(canonical_noop, ["test_single_max_abs_delta", "blend_delta_vs_anchor_pearson_pert"]), 999.0) == 0.0
        and fnum(get_path(canonical_noop, ["family_gene_max_abs_delta", "blend_delta_vs_anchor_pearson_pert"]), 999.0) == 0.0
        and fnum(get_path(canonical_noop, ["test_single_max_abs_delta", "blend_delta_vs_anchor_test_mmd"]), 999.0) == 0.0
        and fnum(get_path(canonical_noop, ["family_gene_max_abs_delta", "blend_delta_vs_anchor_test_mmd"]), 999.0) == 0.0,
        "canonical test_single/family_gene max deltas are exact zero for Pearson and MMD under support-absent protocol",
    )
    add(
        "query_not_used",
        support_cpu.get("heldout_query_used") is False
        and support_cpu.get("canonical_multi_selection_used") is False,
        f"heldout_query_used={support_cpu.get('heldout_query_used')} canonical_multi_selection_used={support_cpu.get('canonical_multi_selection_used')}",
    )

    failed = [c for c in checks if not c["passed"]]
    status = (
        "trackc_support_context_v2_cpu_gate_pass_launcher_gate_next_no_gpu"
        if not failed
        else "trackc_support_context_v2_cpu_gate_fail_no_gpu"
    )
    return {
        "status": status,
        "gpu_authorization": "none",
        "next_authorization": "launcher_provenance_gate_only" if not failed else "none",
        "checks": checks,
        "failed_checks": failed,
        "selected_alpha": support_summary.get("alpha"),
        "support_summary": {
            "wessels": wessels,
            "norman": norman,
            "paired": support_paired,
            "posthoc_pearson_pert": pp,
            "posthoc_mmd": mmd,
        },
        "canonical_noop": canonical_noop,
        "negative_control": {"shuffled_wessels": shuffled_wessels},
        "boundaries": {
            "heldout_query_used": False,
            "canonical_multi_selection_used": False,
            "gpu_authorized": False,
            "next_step": "Build launcher/provenance gate and AGENTS resource audit only if choosing to run one capped v2 smoke.",
        },
        "inputs": {
            "protocol_json": str(PROTOCOL_JSON),
            "code_boundary_json": str(CODE_BOUNDARY_JSON),
            "support_cpu_json": str(SUPPORT_CPU_JSON),
            "posthoc_json": str(POSTHOC_JSON),
            "biological_prior_json": str(BIO_JSON),
        },
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def render(payload: dict[str, Any]) -> str:
    s = payload["support_summary"]
    lines = [
        "# Track C Support-Context V2 CPU Gate",
        "",
        f"Status: `{payload['status']}`",
        f"GPU authorization: `{payload['gpu_authorization']}`",
        f"Next authorization: `{payload['next_authorization']}`",
        "",
        "## Scope",
        "",
        "This gate reads frozen JSON/condition-mean artifacts only. It does not train, evaluate held-out query, or authorize GPU. It checks whether v2 is ready for a launcher/provenance gate.",
        "",
        "## Key Evidence",
        "",
        f"- selected alpha: `{payload.get('selected_alpha')}`",
        f"- Wessels support delta: `{fmt(s['wessels'].get('mean_delta_pp'))}`, closure `{fmt(s['wessels'].get('route_gap_closed_fraction'))}`",
        f"- Norman support delta: `{fmt(s['norman'].get('mean_delta_pp'))}`",
        f"- support paired pp delta: `{fmt(s['paired'].get('delta_mean'))}`, p_harm `{fmt(s['paired'].get('p_harm'))}`",
        f"- posthoc support pp: `{fmt(s['posthoc_pearson_pert'].get('observed'))}`, p_harm `{fmt(s['posthoc_pearson_pert'].get('p_harm_pp'))}`",
        f"- posthoc support MMD: `{fmt(s['posthoc_mmd'].get('observed'))}`, p_harm `{fmt(s['posthoc_mmd'].get('p_harm_mmd'))}`",
        f"- shuffled Wessels delta: `{fmt(payload['negative_control']['shuffled_wessels'].get('mean_delta_pp'))}`",
        "",
        "## Checks",
        "",
        "| check | passed | evidence |",
        "|---|---:|---|",
    ]
    for check in payload["checks"]:
        lines.append(f"| `{check['name']}` | `{check['passed']}` | {check['evidence']} |")
    lines.extend(["", "## Failed Checks", ""])
    if payload["failed_checks"]:
        for check in payload["failed_checks"]:
            lines.append(f"- `{check['name']}`: {check['evidence']}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Decision Boundary",
            "",
            "- Passing this gate does not authorize GPU.",
            "- Next step is a launcher/provenance gate with RUN_STATUS template and fresh AGENTS.md resource audit.",
            "- Held-out query remains forbidden until route/checkpoint freeze and uncapped canonical no-harm pass.",
            "",
        ]
    )
    return "\n".join(lines)


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
