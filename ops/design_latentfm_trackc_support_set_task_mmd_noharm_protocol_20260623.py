#!/usr/bin/env python3
"""Protocol for the support-set task summary MMD/no-harm posthoc gate.

This is a planning/provenance artifact only. It does not inspect the active
input-artifact long job and does not authorize GPU. It defines the next
query-free posthoc if, and only if, the mean-vector support-set summary gate
passes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
SUMMARY_GATE = ROOT / "reports/latentfm_trackc_support_set_task_summary_gate_20260623.json"
SUMMARY_SCRIPT = ROOT / "ops/summarize_latentfm_trackc_support_set_task_summary_gate_20260623.py"
CHECKER = ROOT / "ops/check_latentfm_trackc_support_set_task_inputs_after_1231_20260623.sh"
OUT_JSON = ROOT / "reports/latentfm_trackc_support_set_task_mmd_noharm_protocol_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_SUPPORT_SET_TASK_MMD_NOHARM_PROTOCOL_20260623.md"


def file_status(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "is_file": path.is_file()}


def build_payload() -> dict[str, Any]:
    summary_payload = None
    if SUMMARY_GATE.is_file():
        summary_payload = json.loads(SUMMARY_GATE.read_text(encoding="utf-8"))
    summary_status = None
    if isinstance(summary_payload, dict):
        summary_status = ((summary_payload.get("decision") or {}).get("status") or summary_payload.get("status"))
    summary_pass = summary_status == "trackc_support_set_task_summary_gate_pass_posthoc_mmd_gate_next_no_gpu"
    summary_available = summary_status is not None

    reasons = []
    if not SUMMARY_SCRIPT.is_file():
        reasons.append("summary_gate_script_missing")
    if not CHECKER.is_file():
        reasons.append("post_1231_checker_missing")
    if not summary_pass:
        reasons.append("summary_gate_not_passed_or_not_yet_available")
    status = (
        "support_set_task_mmd_noharm_protocol_blocked_summary_gate_failed_no_gpu"
        if summary_available and not summary_pass
        else "support_set_task_mmd_noharm_protocol_ready_wait_summary_pass_no_gpu"
    )

    return {
        "status": status,
        "gpu_authorization": "none",
        "next_authorization": "none" if summary_available and not summary_pass else "implement_and_resource_audit_only_if_summary_gate_passes",
        "summary_gate_status": summary_status,
        "summary_gate_pass": summary_pass,
        "blocking_reasons_for_launch_now": reasons,
        "inputs": {
            "summary_gate_json": file_status(SUMMARY_GATE),
            "summary_script": file_status(SUMMARY_SCRIPT),
            "post_1231_checker": file_status(CHECKER),
            "safe_trainselect_split": str(ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"),
            "canonical_noharm_split": str(ROOT / "dataset/biFlow_data/split_seed42.json"),
        },
        "hypothesis": (
            "A train_multi-derived per-dataset support-set task residual summary can improve "
            "support_val_multi distributional metrics beyond anchor while preserving canonical "
            "single/background no-harm when the summary is absent or gate=0."
        ),
        "required_evaluator": {
            "mechanism": "for each evaluated support condition, compute anchor per-cell endpoint predictions and add alpha * dataset_train_multi_mean(candidate_pred_mean - anchor_pred_mean) as a fixed endpoint residual; canonical no-harm uses gate=0 exact no-op",
            "support_scope": {
                "split": "split_seed42_multi_support_v2_trainselect.json",
                "groups": ["support_val_multi"],
                "gate": 1,
            },
            "canonical_noharm_scope": {
                "split": "split_seed42.json",
                "groups": ["test_single", "family_gene"],
                "gate": 0,
                "forbidden": ["test_multi", "test_multi_seen", "test_multi_unseen1", "test_multi_unseen2"],
            },
            "forbidden": [
                "full v2 held-out query",
                "canonical test_multi selection",
                "checkpoint/alpha changes after summary gate",
                "MMD estimated from condition means only",
            ],
        },
        "resource_plan_if_authorized": {
            "runtime_classification": "long GPU posthoc if launched",
            "gpus": "1 physical GPU initially; may parallelize support/canonical evals only after fresh AGENTS resource audit",
            "cpu_threads": "<= 8 for this posthoc block; total LatentFM cap <= 48 cores",
            "detached_required": True,
            "run_status_required": True,
        },
        "promotion_gate": [
            "summary gate must pass before implementation/launch",
            "support_val_multi Pearson_pert equal-dataset delta >= +0.02",
            "support_val_multi Pearson_pert bootstrap p(delta < -0.02) <= 0.10",
            "support_val_multi unbiased and biased MMD deltas <= +0.005",
            "support_val_multi MMD bootstrap p(delta > +0.005) <= 0.10",
            "canonical test_single/family_gene gate=0 max absolute metric delta <= 1e-8",
            "payload safety flags prove no held-out query and no canonical multi selection",
        ],
        "failure_close_rule": (
            "If summary gate fails, or if MMD/no-harm posthoc fails any support/canonical/safety rule, "
            "close this support-set task summary branch and do not launch GPU training or query."
        ),
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track C Support-Set Task MMD/No-Harm Protocol",
        "",
        f"Status: `{payload['status']}`",
        f"GPU authorization: `{payload['gpu_authorization']}`",
        f"Next authorization: `{payload['next_authorization']}`",
        f"Summary gate status: `{payload['summary_gate_status']}`",
        "",
        "## Hypothesis",
        "",
        payload["hypothesis"],
        "",
        "## Required Evaluator",
        "",
        f"- mechanism: {payload['required_evaluator']['mechanism']}",
        f"- support scope: `{payload['required_evaluator']['support_scope']}`",
        f"- canonical no-harm scope: `{payload['required_evaluator']['canonical_noharm_scope']}`",
        "",
        "Forbidden:",
    ]
    lines.extend(f"- {item}" for item in payload["required_evaluator"]["forbidden"])
    lines.extend(["", "## Resource Plan If Authorized", ""])
    for key, value in payload["resource_plan_if_authorized"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Promotion Gate", ""])
    lines.extend(f"- {item}" for item in payload["promotion_gate"])
    lines.extend(["", "## Blocking Reasons For Launch Now", ""])
    if payload["blocking_reasons_for_launch_now"]:
        lines.extend(f"- `{reason}`" for reason in payload["blocking_reasons_for_launch_now"])
    else:
        lines.append("- none")
    lines.extend(["", "## Failure Close Rule", "", payload["failure_close_rule"], ""])
    return "\n".join(lines)


def main() -> int:
    payload = build_payload()
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
