#!/usr/bin/env python3
"""Protocol audit for the response-program projection artifact builder."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
SCRIPT = ROOT / "ops/build_latentfm_response_program_projection_artifact_20260625.py"
RUN_STATUS = ROOT / (
    "runs/latentfm_response_program_projection_artifact_20260625/"
    "truecell_budget128_seed42_internal_projection_v2/RUN_STATUS.md"
)
DRY_JSON = ROOT / "reports/latentfm_response_program_projection_gate_DRYRUN_20260625.json"
DRY_MD = ROOT / "reports/LATENTFM_RESPONSE_PROGRAM_PROJECTION_GATE_DRYRUN_20260625.md"
OUT_JSON = ROOT / "reports/latentfm_response_program_projection_protocol_audit_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_RESPONSE_PROGRAM_PROJECTION_PROTOCOL_AUDIT_20260625.md"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) is not None


def main() -> int:
    script = SCRIPT.read_text(encoding="utf-8")
    status = RUN_STATUS.read_text(encoding="utf-8") if RUN_STATUS.exists() else ""
    dry = load_json(DRY_JSON)
    checks = {
        "script_exists": SCRIPT.exists(),
        "run_status_exists": RUN_STATUS.exists(),
        "dry_run_json_exists": DRY_JSON.exists(),
        "dry_run_md_exists": DRY_MD.exists(),
        "dry_run_completed": dry.get("status") == "response_program_projection_gate_fail_no_gpu",
        "dry_run_gpu_authorized_false": dry.get("gpu_authorized") is False,
        "boundary_declares_no_canonical_performance": "reads_canonical_performance" in script
        and "reads_canonical_performance\": False" in script,
        "boundary_declares_no_canonical_multi": "reads_canonical_multi" in script
        and "reads_canonical_multi\": False" in script,
        "boundary_declares_no_trackc_query": "reads_trackc_query" in script
        and "reads_trackc_query\": False" in script,
        "uses_internal_truecell_default_jsons": "latentfm_true_cell_count_budget128_tail_stability_6k_20260625" in script
        and "posthoc_eval_internal" in script,
        "uses_train_conditions_for_axes": "parts.get(\"train\"" in script
        and "build_train_axes" in script,
        "uses_internal_scalar_pairs_for_vector_eval": "selected_pairs=scalar_pairs" in script
        and "metric_group" in script
        and "internal_val_family_gene_proxy" in script,
        "threshold_zero_values_not_treated_as_missing": "def missing_as" in script
        and "hard_harm_frac\"] or 1.0" not in script
        and "mmd_max\"] or 999.0" not in script,
        "uses_eval_condition_residuals_private_rows": "_pred_residual" in script
        and "_target_residual" in script,
        "has_random_axis_control": "random_axis" in script and "np.linalg.qr" in script,
        "has_fail_closed_thresholds": all(
            token in script
            for token in (
                "supported_pp_below_0p025",
                "supported_ci_lower_not_positive",
                "supported_minus_unsupported_gap_below_0p020",
                "dataset_min_below_minus_0p010",
                "hard_harm_frac_above_0p15",
                "mmd_max_above_0p001",
                "random_axis_control_not_collapsed",
            )
        ),
        "run_status_records_long_task": "Long task" in status or "unknown-runtime GPU inference" in status,
        "run_status_records_no_canonical_multi_query": "no canonical performance" in status
        and "no canonical multi" in status
        and "no Track C query" in status,
        "run_status_records_resource_plan": "GPU 5" in status and "4 CPU threads" in status,
    }

    risks: list[str] = []
    if "canonical_noharm" in script:
        risks.append("script_mentions_canonical_noharm_path_text")
    if has(r"heldout_query|trackc.*query|test_multi", script):
        # The script may contain boundary text only; keep as risk note rather than hard fail.
        risks.append("script_contains_query_or_multi_terms_in_boundary_or_text")
    if "metric_group" in script and "family_gene" in script:
        risks.append("gate_uses_existing_internal_scalar_posthoc_metrics_for pp/mmd deltas")

    reasons = [k for k, v in checks.items() if not v]
    status_value = (
        "response_program_projection_protocol_audit_pass_running_job_not_polled"
        if not reasons
        else "response_program_projection_protocol_audit_fail"
    )
    payload = {
        "status": status_value,
        "checks": checks,
        "reasons": reasons,
        "risks": risks,
        "decision": {
            "gpu_training_authorized_by_this_audit": False,
            "polling_running_job": False,
            "requires_external_audit_if_gate_passes": True,
            "next_action": (
                "wait for artifact gate natural completion; if pass, require external audit before GPU training"
                if not reasons
                else "fix protocol audit failures before relying on artifact gate"
            ),
        },
        "paths": {
            "script": str(SCRIPT),
            "run_status": str(RUN_STATUS),
            "dry_json": str(DRY_JSON),
            "dry_md": str(DRY_MD),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# LatentFM Response-Program Projection Protocol Audit",
        "",
        f"Status: `{status_value}`",
        "",
        "## Boundary",
        "",
        "- CPU-only audit of script/protocol/dry-run files.",
        "- Does not read the running long-job log or EXIT_CODE.",
        "- Does not train, infer, use GPU, read canonical multi, or read Track C query.",
        "",
        "## Checks",
        "",
        "| check | pass |",
        "|---|---:|",
    ]
    for key, value in checks.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines += [
        "",
        "## Decision",
        "",
        f"- reasons: `{reasons}`",
        f"- risks: `{risks}`",
        "- GPU training authorized by this audit: `False`",
        "- If the artifact gate passes, require external audit before training launch.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status_value, "reasons": reasons, "out": str(OUT_MD)}, indent=2))
    return 0 if not reasons else 1


if __name__ == "__main__":
    raise SystemExit(main())
