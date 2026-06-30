#!/usr/bin/env python3
"""Audit the Track C support-set task adapter code-boundary gate.

This is CPU-only. It checks that the new adapter is default-off, distinct from
the consumed routed support_context path, fail-closed, and covered by focused
unit tests. It does not read held-out query artifacts or launch GPU work.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
COUPLED = ROOT / "CoupledFM"
REPORT_DIR = ROOT / "reports"
OUT_JSON = REPORT_DIR / "latentfm_trackc_support_set_task_adapter_code_boundary_20260623.json"
OUT_MD = REPORT_DIR / "LATENTFM_TRACKC_SUPPORT_SET_TASK_ADAPTER_CODE_BOUNDARY_20260623.md"

FILES = {
    "config": COUPLED / "model/latent/config.py",
    "mlp": COUPLED / "model/latent/models/mlp.py",
    "train": COUPLED / "model/latent/train.py",
    "tests": COUPLED / "model/tests/test_latent_condition_embedding_sources.py",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: list[str]) -> dict[str, object]:
    proc = subprocess.run(
        cmd,
        cwd=str(COUPLED),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
    }


def require_contains(name: str, text: str, markers: list[str], failed: list[str]) -> None:
    for marker in markers:
        if marker not in text:
            failed.append(f"{name}:missing:{marker}")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    texts = {name: path.read_text() for name, path in FILES.items()}

    require_contains(
        "config",
        texts["config"],
        [
            "trackc_support_set_task_use_in_model: bool = False",
            "trackc_support_set_task_dim: int = 0",
            "support_set_task_adapter",
        ],
        failed,
    )
    require_contains(
        "mlp",
        texts["mlp"],
        [
            "trackc_support_set_task_use_in_model",
            "trackc_support_set_task_dim",
            "support_set_task_to_c",
            "support_set_task_present",
            "_validate_support_set_task",
            "_support_set_task_projection",
            "support_set_task was passed but trackc_support_set_task_use_in_model=False",
        ],
        failed,
    )
    require_contains(
        "train",
        texts["train"],
        [
            "support_set_task_enabled",
            "trackc_support_set_task_use_in_model=support_set_task_enabled",
            "trackc_support_set_task_dim=int",
            "support_set_task_adapter",
            "support_set_task_to_c.",
        ],
        failed,
    )
    require_contains(
        "tests",
        texts["tests"],
        [
            "test_trackc_support_set_task_default_off_has_no_state_and_rejects_task",
            "test_trackc_support_set_task_fail_closed_validation",
            "test_trackc_support_set_task_forward_consumes_task_signal",
            "test_trackc_support_set_task_zero_task_exact_noop_after_nonzero_weights",
            "test_support_set_task_adapter_finetune_scope_only_trains_task_bridge",
        ],
        failed,
    )

    py_compile = run_cmd(
        [
            sys.executable,
            "-m",
            "py_compile",
            str(FILES["mlp"]),
            str(FILES["config"]),
            str(FILES["train"]),
            str(FILES["tests"]),
        ]
    )
    if py_compile["returncode"] != 0:
        failed.append("py_compile_failed")

    focused_pytest = run_cmd(
        [
            sys.executable,
            "-m",
            "pytest",
            str(FILES["tests"]),
            "-q",
            "-k",
            "support_set_task or support_context or support_residual or support_film",
        ]
    )
    if focused_pytest["returncode"] != 0:
        failed.append("focused_support_pytest_failed")

    full_pytest = run_cmd(
        [sys.executable, "-m", "pytest", str(FILES["tests"]), "-q"]
    )
    if full_pytest["returncode"] != 0:
        failed.append("full_condition_embedding_pytest_failed")

    status = (
        "trackc_support_set_task_adapter_code_boundary_pass_cpu_metric_gate_next_no_gpu"
        if not failed
        else "trackc_support_set_task_adapter_code_boundary_fail"
    )
    payload = {
        "status": status,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "failed_checks": failed,
        "scope": "cpu_only_default_off_code_boundary_no_query_no_gpu",
        "files": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in FILES.items()
        },
        "validation": {
            "py_compile": py_compile,
            "focused_support_pytest": focused_pytest,
            "full_condition_embedding_pytest": full_pytest,
        },
        "claim_boundary": {
            "allowed": [
                "A distinct default-off support-set task adapter code path exists.",
                "Disabled models have no support_set_task_to_c state and reject support_set_task input.",
                "Enabled models fail closed on missing, wrong-shape, or non-finite support_set_task.",
                "Zero or masked support_set_task is exact no-op in the unit gate.",
                "support_set_task_adapter finetune scope trains only support_set_task_to_c.",
            ],
            "not_allowed": [
                "No CPU metric gate has passed yet.",
                "No GPU training is authorized by this audit.",
                "No held-out query or canonical test_multi selection is authorized.",
                "This is not a formal multi-perturbation success claim.",
            ],
        },
        "next_gate": {
            "name": "trackc_support_set_task_adapter_cpu_metric_gate",
            "requirements": [
                "safe trainselect split only",
                "permutation-invariant support-set/task summary source",
                "zero and shuffled support controls",
                "support-val scoring only",
                "predeclared Wessels/Norman/no-harm gates before any capped GPU smoke",
            ],
        },
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Track C Support-Set Task Adapter Code-Boundary Audit",
        "",
        f"Status: `{status}`",
        f"Failed checks: `{len(failed)}`",
        "",
        "## Scope",
        "",
        "CPU-only default-off code-boundary audit. No GPU, no held-out query, no canonical test_multi selection.",
        "",
        "## Validation",
        "",
        f"- py_compile returncode: `{py_compile['returncode']}`",
        f"- focused support pytest returncode: `{focused_pytest['returncode']}`",
        f"- full condition embedding pytest returncode: `{full_pytest['returncode']}`",
        "",
        "## Boundary Evidence",
        "",
        "- New config flags: `trackc_support_set_task_use_in_model`, `trackc_support_set_task_dim`.",
        "- New explicit forward API: `support_set_task`, `support_set_task_present`.",
        "- New parameter prefix only when enabled: `support_set_task_to_c`.",
        "- New finetune scope: `support_set_task_adapter`.",
        "- Unit tests cover default-off no state, fail-closed validation, signal wiring, zero/masked no-op, and trainable scope.",
        "",
        "## Claim Boundary",
        "",
        "- Allowed: distinct default-off support-set task adapter boundary exists and passes CPU unit tests.",
        "- Not allowed: CPU metric success, GPU authorization, query evaluation, or formal multi success.",
        "",
        "## Next Gate",
        "",
        "`trackc_support_set_task_adapter_cpu_metric_gate`: safe trainselect only, support-val scoring only, zero/shuffled controls, Wessels/Norman/no-harm thresholds.",
        "",
        "## Files",
        "",
    ]
    for name, meta in payload["files"].items():
        lines.append(f"- `{name}`: `{meta['path']}` sha256 `{meta['sha256']}`")
    if failed:
        lines.extend(["", "## Failed Checks", ""])
        lines.extend(f"- `{item}`" for item in failed)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": status, "failed_checks": failed, "report": str(OUT_MD)}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
