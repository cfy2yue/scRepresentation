#!/usr/bin/env python3
"""Readiness gate for Track C support-set task input artifact generation.

This prepares, but does not launch, a long GPU posthoc job that would generate
safe-trainselect train_multi/support_val_multi condition-mean artifacts. Those
artifacts are only inputs for a later support-set task summary/metric gate; they
are not themselves a model metric pass or GPU training authorization.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
COUPLED = ROOT / "CoupledFM"
SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
FULL_V2 = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2.json"
LAUNCHER = ROOT / "ops/launch_latentfm_trackc_support_set_task_input_artifacts_20260623.sh"
EVAL_SPLIT = COUPLED / "model/latent/eval_split_groups.py"
PROTOCOL_JSON = REPORTS / "latentfm_trackc_support_set_task_adapter_protocol_20260623.json"
CODE_JSON = REPORTS / "latentfm_trackc_support_set_task_adapter_code_boundary_20260623.json"
CPU_GATE_JSON = REPORTS / "latentfm_trackc_support_set_task_adapter_cpu_metric_gate_20260623.json"
ANCHOR_CKPT = (
    COUPLED
    / "output/latentfm_runs/xverse_8k_full_eval_20260620/"
    "xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt"
)
CANDIDATE_CKPT = (
    COUPLED
    / "output/latentfm_runs/xverse_trackc_support_film_20260623/"
    "xverse_trackc_support_film_absroute_2k_seed42_retry1/best.pt"
)
RUN_ROOT = (
    ROOT
    / "runs/latentfm_trackc_support_set_task_input_artifacts_20260623/"
    "xverse_support_film_retry1_trainmulti_condition_means"
)
OUT_JSON = REPORTS / "latentfm_trackc_support_set_task_input_generation_gate_20260623.json"
OUT_MD = REPORTS / "LATENTFM_TRACKC_SUPPORT_SET_TASK_INPUT_GENERATION_GATE_20260623.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_counts(split: dict[str, Any]) -> dict[str, Any]:
    focus: dict[str, dict[str, int]] = {}
    totals = {"train_multi": 0, "support_val_multi": 0, "test_multi": 0}
    for ds, groups in split.items():
        if not isinstance(groups, dict):
            continue
        row = {
            key: len(groups.get(key) or [])
            for key in ("train_multi", "support_val_multi", "test_multi")
        }
        for key in totals:
            totals[key] += row[key]
        if row["train_multi"] or row["support_val_multi"]:
            focus[str(ds)] = row
    return {"totals": totals, "nonempty": focus}


def run_cmd(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {"cmd": cmd, "returncode": proc.returncode, "stdout_tail": proc.stdout[-3000:]}


def main() -> int:
    protocol = load_json(PROTOCOL_JSON)
    code = load_json(CODE_JSON)
    cpu_gate = load_json(CPU_GATE_JSON)
    split = load_json(SPLIT)
    eval_text = EVAL_SPLIT.read_text()
    launcher_text = LAUNCHER.read_text()
    counts = split_counts(split)
    bash_syntax = run_cmd(["bash", "-n", str(LAUNCHER)])

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, evidence: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    add(
        "protocol_and_code_ready",
        protocol.get("status") == "trackc_support_set_task_adapter_protocol_ready_code_gate_next_no_gpu"
        and code.get("status") == "trackc_support_set_task_adapter_code_boundary_pass_cpu_metric_gate_next_no_gpu",
        f"protocol={protocol.get('status')} code={code.get('status')}",
    )
    add(
        "previous_cpu_metric_failed_for_missing_inputs",
        cpu_gate.get("status") == "trackc_support_set_task_adapter_cpu_metric_gate_fail_no_gpu_missing_train_multi_task_inputs",
        f"status={cpu_gate.get('status')}",
    )
    add(
        "safe_split_has_required_rows",
        counts["totals"]["train_multi"] == 49 and counts["totals"]["support_val_multi"] == 24,
        f"counts={counts}",
    )
    add(
        "full_v2_query_not_in_launcher",
        str(FULL_V2) not in launcher_text and "query_multi" not in launcher_text,
        "launcher references only safe trainselect split and train/support groups",
    )
    add(
        "canonical_multi_not_in_launcher",
        "test_multi" not in launcher_text and "split_seed42.json" not in launcher_text,
        "launcher does not request canonical test_multi or canonical split",
    )
    add(
        "eval_split_groups_supports_needed_cli",
        "--pert-means-file" in eval_text and "args.groups" in eval_text and "_group_as_test_split" in eval_text,
        "eval_split_groups supports arbitrary split groups, pert_means override, and condition means",
    )
    add("anchor_checkpoint_exists", ANCHOR_CKPT.is_file(), str(ANCHOR_CKPT))
    add("candidate_checkpoint_exists", CANDIDATE_CKPT.is_file(), str(CANDIDATE_CKPT))
    add("launcher_bash_syntax", bash_syntax["returncode"] == 0, bash_syntax["stdout_tail"])
    add(
        "output_dir_not_populated",
        not (RUN_ROOT / "condition_means/trainselect_anchor_train_support_multi_condition_means_ode20.json").exists()
        and not (RUN_ROOT / "condition_means/trainselect_candidate_train_support_multi_condition_means_ode20.json").exists(),
        str(RUN_ROOT),
    )

    failed = [row for row in checks if not row["passed"]]
    status = (
        "trackc_support_set_task_input_generation_gate_ready_resource_audit_next_no_gpu"
        if not failed
        else "trackc_support_set_task_input_generation_gate_fail"
    )
    payload = {
        "status": status,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "gpu_authorization": "none",
        "next_authorization": "fresh_resource_audit_and_detached_long_posthoc_only" if not failed else "none",
        "checks": checks,
        "failed_checks": failed,
        "split": {
            "path": str(SPLIT),
            "sha256": sha256(SPLIT),
            "counts": counts,
        },
        "launcher": {
            "path": str(LAUNCHER),
            "sha256": sha256(LAUNCHER),
            "bash_syntax": bash_syntax,
            "runtime_classification": "Long GPU posthoc artifact-generation task if launched.",
            "expected_outputs": [
                str(RUN_ROOT / "condition_means/trainselect_anchor_train_support_multi_condition_means_ode20.json"),
                str(RUN_ROOT / "condition_means/trainselect_candidate_train_support_multi_condition_means_ode20.json"),
            ],
            "resource_plan": {
                "physical_gpus": 1,
                "cpu_threads": 4,
                "requires_prelaunch_audit": [
                    "multi-sample GPU audit per AGENTS.md",
                    "free -h",
                    "df -h",
                    "CPU load/RAM check",
                ],
                "detach_required": True,
                "run_status_required": True,
            },
        },
        "boundaries": {
            "does_not_launch_now": True,
            "does_not_train": True,
            "does_not_use_full_v2_query": True,
            "does_not_use_canonical_test_multi_selection": True,
            "does_not_authorize_gpu_training": True,
            "artifact_use": "input_generation_only_for_later_support_set_task_summary_gate",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Track C Support-Set Task Input-Generation Gate",
        "",
        f"Status: `{status}`",
        "GPU authorization: `none`",
        f"Next authorization: `{payload['next_authorization']}`",
        "",
        "## Purpose",
        "",
        "Prepare a long, detached GPU posthoc artifact-generation job for safe-trainselect `train_multi` and `support_val_multi` condition means. These artifacts are inputs only; they do not constitute a metric pass.",
        "",
        "## Checks",
        "",
        "| check | passed | evidence |",
        "|---|---:|---|",
    ]
    for row in checks:
        evidence = str(row["evidence"]).replace("\n", " ")
        lines.append(f"| `{row['name']}` | `{row['passed']}` | {evidence} |")
    lines.extend(
        [
            "",
            "## Split Counts",
            "",
            f"- safe split: `{SPLIT}`",
            f"- totals: `{counts['totals']}`",
            f"- nonempty multi datasets: `{counts['nonempty']}`",
            "",
            "## Launcher",
            "",
            f"- path: `{LAUNCHER}`",
            "- runtime if launched: long GPU posthoc artifact-generation task",
            "- expected outputs:",
            f"  - `{payload['launcher']['expected_outputs'][0]}`",
            f"  - `{payload['launcher']['expected_outputs'][1]}`",
            "",
            "## Boundary",
            "",
            "- This gate does not launch the job.",
            "- Launching requires fresh AGENTS.md resource audit and detached tmux/nohup RUN_STATUS.",
            "- The artifacts are for a later query-free support-set task summary gate only.",
            "- Full v2 query and canonical `test_multi` remain forbidden for selection.",
            "",
        ]
    )
    if failed:
        lines.extend(["## Failed Checks", ""])
        lines.extend(f"- `{row['name']}`: {row['evidence']}" for row in failed)
        lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print(json.dumps({"status": status, "failed": len(failed), "report": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
