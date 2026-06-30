#!/usr/bin/env python3
"""Design the post-v2 Track C support-set task-adapter protocol.

This is a query-free protocol/preflight artifact. It reads source text and
existing reports only; it does not run training, evaluate held-out query, or
authorize GPU work.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
COUPLEDFM = ROOT / "CoupledFM"

SAFE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
FULL_V2_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2.json"
CANONICAL_SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"

INPUT_REPORTS = {
    "post_v2_next_gate": REPORTS / "latentfm_post_v2_next_modeling_gate_decision_20260623.json",
    "v2_reporting_package": REPORTS / "latentfm_trackc_support_context_v2_reporting_package_20260623.json",
    "v2_claim_readiness": REPORTS / "latentfm_trackc_support_context_v2_claim_readiness_audit_20260623.json",
    "residual_operator_gpu_gate": REPORTS
    / "latentfm_trackc_residual_operator_route_gap_gate_xverse_trackc_residual_operator_memall_resid_ep050_replay2_2k_seed42_retry1.json",
    "residual_operator_negative_controls": REPORTS / "latentfm_trackc_residual_operator_negative_controls_20260623.json",
    "archetype_multilatent_gate": REPORTS / "latentfm_soft_archetype_multilatent_state_cpu_gate_20260623.json",
}

CODE_FILES = {
    "config": COUPLEDFM / "model/latent/config.py",
    "train": COUPLEDFM / "model/latent/train.py",
    "mlp": COUPLEDFM / "model/latent/models/mlp.py",
    "latent_condition_tests": COUPLEDFM / "model/tests/test_latent_condition_embedding_sources.py",
    "route_gap_tests": COUPLEDFM / "model/tests/test_trackc_residual_operator_route_gap_gate.py",
}

OUT_JSON = REPORTS / "latentfm_trackc_support_set_task_adapter_protocol_20260623.json"
OUT_MD = REPORTS / "LATENTFM_TRACKC_SUPPORT_SET_TASK_ADAPTER_PROTOCOL_20260623.md"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested_status(payload: dict[str, Any]) -> str:
    if payload.get("status") is not None:
        return str(payload["status"])
    if isinstance(payload.get("decision"), dict) and payload["decision"].get("status") is not None:
        return str(payload["decision"]["status"])
    return "present_json_no_status"


def count_split(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    counts: dict[str, int] = {}

    def walk(obj: Any, key_hint: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, list):
                    counts[key] = len(value)
                walk(value, key)
        elif isinstance(obj, list):
            counts[key_hint or "list"] = len(obj)

    walk(payload)
    return {"sha256": sha256_file(path), "counts": counts}


def contains(path: Path, pattern: str) -> bool:
    return pattern in path.read_text(encoding="utf-8", errors="replace")


def main() -> int:
    report_payloads = {name: load_json(path) for name, path in INPUT_REPORTS.items()}
    report_statuses = {name: nested_status(payload) for name, payload in report_payloads.items()}
    code_presence = {
        "existing_support_context_adapter": contains(CODE_FILES["train"], "support_context_adapter")
        and contains(CODE_FILES["mlp"], "support_context_to_c"),
        "existing_support_residual_adapter": contains(CODE_FILES["train"], "support_residual_adapter")
        and contains(CODE_FILES["mlp"], "support_context_to_v"),
        "existing_support_film_adapter": contains(CODE_FILES["train"], "support_film_adapter")
        and contains(CODE_FILES["mlp"], "support_context_to_v_scale"),
        "new_support_set_task_adapter_absent": not any(
            contains(path, "support_set_task_adapter") or contains(path, "trackc_support_set")
            for path in CODE_FILES.values()
        ),
    }
    code_hashes = {name: sha256_file(path) for name, path in CODE_FILES.items()}
    split_info = {
        "safe_trainselect": count_split(SAFE_SPLIT),
        "full_v2_query_split": {"sha256": sha256_file(FULL_V2_SPLIT)},
        "canonical": {"sha256": sha256_file(CANONICAL_SPLIT)},
    }

    checks = [
        {
            "name": "post_v2_decision_requires_new_cpu_protocol",
            "passed": report_statuses["post_v2_next_gate"] == "post_v2_no_gpu_new_cpu_protocol_required",
            "evidence": report_statuses["post_v2_next_gate"],
        },
        {
            "name": "v2_reporting_ready",
            "passed": report_statuses["v2_reporting_package"] == "support_context_v2_reporting_package_ready",
            "evidence": report_statuses["v2_reporting_package"],
        },
        {
            "name": "residual_gpu_gate_failed_closure",
            "passed": report_statuses["residual_operator_gpu_gate"] == "residual_route_gap_gate_fail_close_branch",
            "evidence": report_statuses["residual_operator_gpu_gate"],
        },
        {
            "name": "safe_split_hash_matches_freeze",
            "passed": split_info["safe_trainselect"]["sha256"]
            == "5f29dd5b582a40da3736770ca29950c12c54e46b590c3f7705c1d37da89f4f20",
            "evidence": split_info["safe_trainselect"]["sha256"],
        },
        {
            "name": "current_code_has_only_consumed_support_modes",
            "passed": code_presence["existing_support_context_adapter"]
            and code_presence["existing_support_residual_adapter"]
            and code_presence["existing_support_film_adapter"]
            and code_presence["new_support_set_task_adapter_absent"],
            "evidence": code_presence,
        },
    ]
    failed = [row for row in checks if not row["passed"]]
    status = (
        "trackc_support_set_task_adapter_protocol_ready_code_gate_next_no_gpu"
        if not failed
        else "trackc_support_set_task_adapter_protocol_needs_review"
    )

    protocol = {
        "status": status,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S CST"),
        "checks": checks,
        "failed_checks": failed,
        "input_reports": {name: str(path) for name, path in INPUT_REPORTS.items()},
        "report_statuses": report_statuses,
        "split_info": split_info,
        "code_hashes": code_hashes,
        "code_presence": code_presence,
        "hypothesis": (
            "A permutation-invariant support-set task adapter can condition model computation on a support set as a task, "
            "rather than on a single routed-distill vector, and may absorb support signal without collapsing to residual, FiLM, context-c, endpoint, replay, or memory-dose variants."
        ),
        "allowed_inputs": [
            "safe trainselect split only",
            "train_multi for leave-one-task or predeclared small-grid fitting",
            "support_val_multi for gate scoring only",
            "train_single/pert means as anchor/background metadata only",
            "closed-family route-gap reports as baselines only",
        ],
        "forbidden_inputs": [
            "full v2 held-out query examples or metrics",
            "canonical test_multi as a selection signal",
            "query-selected thresholds/features/checkpoints",
            "residual query or second query in the current v2 family",
        ],
        "required_code_gate": [
            "add default-off Config flag for support-set task adapter",
            "add explicit support-set/task tensor or encoded support summary API distinct from support_context",
            "add fail-closed validation for missing/malformed support-set task input",
            "add trainable scope for the new adapter only if the new mode is enabled",
            "prove fixed-condition outputs change when support set changes and remain unchanged under zero-support control",
            "prove old support_context/residual/film paths are unchanged when new flag is off",
        ],
        "required_cpu_gate": {
            "Wessels pp delta": ">= +0.02",
            "Wessels route-gap closure": ">= +0.05",
            "Norman pp delta": ">= -0.02",
            "support pp p_harm": "<= 0.20",
            "MMD hard harm": "none",
            "zero-support control": "must fail support gate",
            "shuffled-support control": "must fail support gate and lose Wessels closure",
            "wiring proof": "fixed-condition outputs change with support-set task input",
        },
        "gpu_consequence_if_all_gates_pass": "at most one capped support-only smoke after fresh AGENTS resource audit and RUN_STATUS; no held-out query",
        "next_action": "implement code-boundary gate only; do not launch GPU",
    }
    OUT_JSON.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Track C Support-Set Task Adapter Protocol",
        "",
        f"Timestamp: `{protocol['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "This is a query-free protocol/preflight artifact. It does not implement the model, run training, read held-out query, or authorize GPU work.",
        "",
        "## Rationale",
        "",
        protocol["hypothesis"],
        "",
        "Current code already has consumed support-context modes (`support_context_adapter`, `support_residual_adapter`, `support_film_adapter`). The new protocol is allowed only if it introduces a genuinely distinct support-set task interface.",
        "",
        "## Checks",
        "",
        "| check | passed | evidence |",
        "|---|---:|---|",
    ]
    for row in checks:
        evidence = json.dumps(row["evidence"], sort_keys=True) if isinstance(row["evidence"], dict) else str(row["evidence"])
        lines.append(f"| `{row['name']}` | `{row['passed']}` | {evidence[:240]} |")
    lines.extend(
        [
            "",
            "## Allowed Inputs",
            "",
        ]
    )
    for item in protocol["allowed_inputs"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Forbidden Inputs", ""])
    for item in protocol["forbidden_inputs"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Required Code Gate", ""])
    for item in protocol["required_code_gate"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Required CPU Gate", "", "| criterion | threshold |", "|---|---|"])
    for key, value in protocol["required_cpu_gate"].items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- safe trainselect split hash: `{split_info['safe_trainselect']['sha256']}`",
            f"- full v2 split hash, forbidden for selection: `{split_info['full_v2_query_split']['sha256']}`",
            f"- canonical split hash, support-absent no-harm only: `{split_info['canonical']['sha256']}`",
            "- code hashes are recorded in the JSON artifact.",
            "",
            "## Next Action",
            "",
            "Implement a code-boundary gate for this protocol only. No GPU launch is allowed until code-boundary and query-free CPU gates both pass.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
