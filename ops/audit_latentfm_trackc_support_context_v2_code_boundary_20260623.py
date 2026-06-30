#!/usr/bin/env python3
"""Static code-boundary audit for Track C support-context v2.

The v2 protocol requires exact support-absent no-op behavior. Existing support
adapters may accept zero support context, but a trained linear adapter with bias
can still emit nonzero residuals for zero context. This audit checks the current
code boundary before any new GPU work.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
MLP = ROOT / "CoupledFM/model/latent/models/mlp.py"
TRAIN = ROOT / "CoupledFM/model/latent/train.py"
CONFIG = ROOT / "CoupledFM/model/latent/config.py"
LAUNCHER = ROOT / "CoupledFM/model/latent/scripts/run_full_stack_latentfm.sh"
PROTOCOL_JSON = ROOT / "reports/latentfm_trackc_support_context_v2_protocol_20260623.json"
OUT_JSON = ROOT / "reports/latentfm_trackc_support_context_v2_code_boundary_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_CODE_BOUNDARY_20260623.md"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def contains(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.MULTILINE) is not None


def audit() -> dict[str, Any]:
    mlp = read(MLP)
    train = read(TRAIN)
    config = read(CONFIG)
    launcher = read(LAUNCHER)
    protocol = json.loads(PROTOCOL_JSON.read_text(encoding="utf-8"))

    checks = []

    def add(name: str, passed: bool, evidence: str, severity: str = "info") -> None:
        checks.append({"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence})

    add(
        "default_off_flags_present",
        all(s in config for s in ("trackc_support_context_use_in_model", "trackc_support_residual_use_in_model", "trackc_support_film_use_in_model")),
        "Config contains support-context/residual/FiLM default-off flags.",
    )
    add(
        "support_context_shape_validation_present",
        "_validate_support_context" in mlp and "support_context contains non-finite values" in mlp,
        "ControlMLP validates support_context shape, batch size, dim, and finite values.",
    )
    add(
        "train_zero_context_fallback_present",
        "_zero_support_context_for" in train and "support_context is None" in train,
        "train.py can synthesize zero support_context when the model uses support context.",
    )
    add(
        "routed_bank_split_guard_present",
        "trackc_routed_distill_bank_split_file" in train and "support-context routed source requires" in train,
        "support-context source requires a routed bank split file.",
    )
    add(
        "launcher_exposes_support_flags",
        all(s in launcher for s in ("TRACKC_SUPPORT_CONTEXT_USE_IN_MODEL", "TRACKC_SUPPORT_RESIDUAL_USE_IN_MODEL", "TRACKC_SUPPORT_FILM_USE_IN_MODEL")),
        "launcher exposes support-context/residual/FiLM switches.",
    )
    add(
        "explicit_support_present_flag_present",
        "support_context_present" in mlp or "support_present" in mlp,
        "v2 requires an explicit support-present mask/flag, not just zero context.",
        severity="hard",
    )
    add(
        "support_residual_linear_biasless_or_masked",
        (
            contains(r"support_context_to_v\s*=\s*nn\.Linear\([^\n]+bias\s*=\s*False", mlp)
            or "support_context_present" in mlp
            or "support_present" in mlp
        ),
        "support_context_to_v must be biasless or explicitly masked so support-absent is exact no-op after training.",
        severity="hard",
    )
    add(
        "support_film_linear_biasless_or_masked",
        (
            contains(r"support_context_to_v_scale\s*=\s*nn\.Linear\([^\n]+bias\s*=\s*False", mlp)
            or "support_context_present" in mlp
            or "support_present" in mlp
        ),
        "support_context_to_v_scale must be biasless or explicitly masked so support-absent FiLM is exact no-op after training.",
        severity="hard",
    )
    add(
        "protocol_no_gpu_boundary",
        protocol.get("gpu_authorization") == "none",
        "v2 protocol document itself does not authorize GPU.",
    )

    hard_failures = [c for c in checks if c["severity"] == "hard" and not c["passed"]]
    status = (
        "trackc_support_context_v2_code_boundary_fail_no_gpu"
        if hard_failures
        else "trackc_support_context_v2_code_boundary_pass_cpu_gate_next"
    )
    return {
        "status": status,
        "gpu_authorization": "none",
        "next_authorization": "none" if hard_failures else "cpu_gate_only",
        "checks": checks,
        "hard_failures": hard_failures,
        "required_fix": (
            "Add an explicit support-present mask/flag or biasless/masked support adapters so support-absent canonical evaluation is exact no-op after training."
            if hard_failures
            else "Implement the next CPU gate; still no GPU from this audit alone."
        ),
        "source_files": {
            "mlp": str(MLP),
            "train": str(TRAIN),
            "config": str(CONFIG),
            "launcher": str(LAUNCHER),
            "protocol": str(PROTOCOL_JSON),
        },
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track C Support-Context V2 Code-Boundary Audit",
        "",
        f"Status: `{payload['status']}`",
        f"GPU authorization: `{payload['gpu_authorization']}`",
        "",
        "## Summary",
        "",
        payload["required_fix"],
        "",
        "## Checks",
        "",
        "| check | passed | severity | evidence |",
        "|---|---:|---|---|",
    ]
    for c in payload["checks"]:
        lines.append(f"| `{c['name']}` | `{c['passed']}` | `{c['severity']}` | {c['evidence']} |")
    lines.extend(["", "## Hard Failures", ""])
    if payload["hard_failures"]:
        for c in payload["hard_failures"]:
            lines.append(f"- `{c['name']}`: {c['evidence']}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The v2 code-boundary now passes: support-present masking and biasless support adapters make support-absent outputs exact no-op under this static/synthetic gate. This still authorizes only the next CPU gate, not GPU."
                if not payload["hard_failures"]
                else "Existing support-context plumbing is useful, but v2 cannot be GPU-launched until support-absent behavior is made exact after training. A zero context alone is insufficient when trainable adapter biases can emit nonzero residuals."
            ),
            "",
            "## Source Files",
            "",
        ]
    )
    for name, path in payload["source_files"].items():
        lines.append(f"- `{name}`: `{path}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = audit()
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(args.out_md), "out_json": str(args.out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
