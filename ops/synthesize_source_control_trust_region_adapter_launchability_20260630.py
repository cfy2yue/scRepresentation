#!/usr/bin/env python3
"""Synthesize source/control trust-region adapter launchability evidence.

This is a CPU/report-only gate. It does not train, run inference, read
canonical multi for selection, read Track C query, or use GPU.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "reports" / "source_control_trust_region_adapter_launchability_20260630"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUTS = {
    "pcgrad_unit": ROOT / "reports" / "latentfm_noharm_pcgrad_adapter_unit_gate_20260627.json",
    "lookahead_unit": ROOT / "reports" / "latentfm_lookahead_trust_region_adapter_unit_gate_20260627.json",
    "lookahead_trainbatch": ROOT
    / "reports"
    / "latentfm_lookahead_trust_region_trainbatch_checkpoint_unit_gate_20260627.json",
    "adapter_reliability": ROOT / "reports" / "latentfm_adapter_support_reliability_admission_gate_20260628.json",
    "failure_cluster_trust_region": ROOT
    / "reports"
    / "latentfm_tracka_failure_cluster_conditioned_trust_region_gate_20260627.json",
    "dual_baseline": ROOT
    / "reports"
    / "tracka_benchmark_control_consolidation_20260630"
    / "tracka_benchmark_control_consolidation_20260630.json",
}


def load_json(path: Path) -> dict:
    if not path.exists():
        return {"_missing": True, "path": str(path)}
    with path.open() as f:
        return json.load(f)


def status_of(obj: dict) -> str:
    if obj.get("_missing"):
        return "missing"
    return str(obj.get("status") or obj.get("decision", {}).get("status") or "unknown")


def bool_of(obj: dict, key: str, default: bool = False) -> bool:
    val = obj.get(key, default)
    if isinstance(val, bool):
        return val
    return bool(val)


def main() -> None:
    data = {name: load_json(path) for name, path in INPUTS.items()}

    mechanical_passes = [
        status_of(data["lookahead_unit"]).startswith("lookahead_trust_region_adapter_unit_gate_pass"),
        status_of(data["lookahead_trainbatch"]).startswith(
            "lookahead_trust_region_trainbatch_checkpoint_unit_gate_pass"
        ),
    ]

    hard_fail_reasons = []
    if not all(mechanical_passes):
        hard_fail_reasons.append("lookahead_trust_region_unit_or_trainbatch_gate_missing_or_failed")

    reliability = data["adapter_reliability"]
    reliability_reasons = reliability.get("reasons", [])
    if status_of(reliability).endswith("fail_no_gpu") or reliability.get("gpu_authorized") is False:
        hard_fail_reasons.append("adapter_support_reliability_gate_failed_negative_controls")

    failure_cluster = data["failure_cluster_trust_region"]
    fc_reasons = failure_cluster.get("reasons") or failure_cluster.get("decision", {}).get("reasons", [])
    if status_of(failure_cluster).endswith("fail_no_gpu") or failure_cluster.get("gpu_authorized") is False:
        hard_fail_reasons.append("failure_cluster_trust_region_gate_failed_dataset_tail_ci_or_mmd")

    dual = data["dual_baseline"]
    dual_candidates = dual.get("dual_baseline_summary", [])
    dual_any_pass = dual.get("decision", {}).get("dual_baseline_candidates_with_any_pass")
    if dual_any_pass is None:
        dual_any_pass = 0
        for row in dual_candidates:
            try:
                dual_any_pass += int(row.get("pass_groups", 0) > 0)
            except Exception:
                pass
    if dual_any_pass == 0:
        hard_fail_reasons.append("no_existing_candidate_passes_dual_baseline_anchor_source_control_gate")

    pcgrad_status = status_of(data["pcgrad_unit"])
    if pcgrad_status.endswith("fail_no_gpu"):
        hard_fail_reasons.append("vanilla_pcgrad_failed_anchor_gradient_zero_noop")

    status = "source_control_trust_region_adapter_launchability_fail_no_gpu"
    gpu_authorized = False

    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "status": status,
        "gpu_authorized": gpu_authorized,
        "boundary": {
            "cpu_only": True,
            "trains_model": False,
            "runs_inference": False,
            "uses_gpu": False,
            "reads_canonical_multi_for_selection": False,
            "reads_trackc_query": False,
        },
        "inputs": {name: str(path) for name, path in INPUTS.items()},
        "input_statuses": {name: status_of(obj) for name, obj in data.items()},
        "hard_fail_reasons": hard_fail_reasons,
        "detail": {
            "mechanical_unit_gates_pass": all(mechanical_passes),
            "adapter_reliability_reasons": reliability_reasons,
            "failure_cluster_reasons": fc_reasons,
            "dual_baseline_any_pass": dual_any_pass,
        },
        "decision": (
            "Do not launch a source/control or trust-region adapter GPU smoke from the current "
            "evidence. The lookahead mechanics are plausible, but reliability controls and "
            "dual-baseline Track A admission are not satisfied."
        ),
        "next_gate": {
            "name": "source_control_specific_trust_region_cpu_admission",
            "requirements": [
                "use safe train-only split and frozen xverse_8k_anchor provenance",
                "compare candidate against max(anchor, source/control) before GPU",
                "negative controls must collapse: count-only, shuffled-control, inverted-control",
                "dataset-bootstrap CI low > 0 for primary Track A internal groups",
                "dataset-min dominance >= -0.02 on cross_background_seen_gene and family_gene",
                "candidate MMD harm <= +0.001 row mean and no unsafe dataset tail",
            ],
        },
    }

    json_path = OUT_DIR / "source_control_trust_region_adapter_launchability_20260630.json"
    md_path = OUT_DIR / "LATENTFM_SOURCE_CONTROL_TRUST_REGION_ADAPTER_LAUNCHABILITY_20260630.md"
    with json_path.open("w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    lines = [
        "# LatentFM Source/Control Trust-Region Adapter Launchability",
        "",
        f"Created: `{result['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        f"GPU authorized: `{gpu_authorized}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only synthesis of existing adapter, reliability, failure-cluster, and dual-baseline evidence.",
        "- No training, inference, canonical multi selection, Track C query access, checkpoint selection, or GPU.",
        "",
        "## Input Statuses",
        "",
        "| artifact | status | path |",
        "|---|---|---|",
    ]
    for name, path in INPUTS.items():
        lines.append(f"| `{name}` | `{status_of(data[name])}` | `{path}` |")

    lines += [
        "",
        "## Decision",
        "",
        result["decision"],
        "",
        "## Why GPU Is Not Authorized",
        "",
    ]
    for reason in hard_fail_reasons:
        lines.append(f"- `{reason}`")

    lines += [
        "",
        "## Key Evidence",
        "",
        "- Lookahead trust-region mechanics passed frozen-means and real train-batch/checkpoint unit gates, but both reports explicitly label the result as external-audit only.",
        f"- Adapter support reliability failed with reasons: `{', '.join(map(str, reliability_reasons))}`.",
        f"- Failure-cluster trust-region gate failed with reasons: `{', '.join(map(str, fc_reasons))}`.",
        f"- Dual-baseline consolidation shows existing candidates with any pass: `{dual_any_pass}`.",
        "- Vanilla PCGrad is closed because the default-off anchor/no-harm gradient is zero at initialization.",
        "",
        "## Next Legal Gate",
        "",
        "`source_control_specific_trust_region_cpu_admission` should be built only if this branch is revived.",
        "",
    ]
    for req in result["next_gate"]["requirements"]:
        lines.append(f"- {req}")

    lines += [
        "",
        "## Outputs",
        "",
        f"- JSON: `{json_path}`",
        f"- Markdown: `{md_path}`",
        "",
    ]

    md_path.write_text("\n".join(lines))
    print(md_path)


if __name__ == "__main__":
    main()
