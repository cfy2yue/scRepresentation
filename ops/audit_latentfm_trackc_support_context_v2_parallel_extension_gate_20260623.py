#!/usr/bin/env python3
"""Gate a small high-throughput extension of Track C support-context v2.

This does not inspect the active v2 smoke's training/log/exit state.  It only
checks that the already-frozen v2 protocol/code/CPU gates support two
mechanistically distinct capped variants under the same leakage boundary.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path("/data/cyx/1030/scLatent")
CPU_GATE = ROOT / "reports/latentfm_trackc_support_context_v2_cpu_gate_20260623.json"
CODE_GATE = ROOT / "reports/latentfm_trackc_support_context_v2_code_boundary_20260623.json"
LAUNCH_GATE = ROOT / "reports/latentfm_trackc_support_context_v2_launcher_provenance_gate_20260623.json"
GENERIC_LAUNCHER = ROOT / "ops/launch_latentfm_xverse_trackc_routed_distill_smoke_20260622.sh"
EXT_LAUNCHER = ROOT / "ops/launch_latentfm_trackc_support_context_v2_parallel_variants_20260623.sh"
TRAIN_PY = ROOT / "CoupledFM/model/latent/train.py"
MLP_PY = ROOT / "CoupledFM/model/latent/models/mlp.py"
CONFIG_PY = ROOT / "CoupledFM/model/latent/config.py"
EVAL_SPLIT = ROOT / "CoupledFM/model/latent/eval_split_groups.py"
EVAL_FAMILY = ROOT / "CoupledFM/model/latent/eval_condition_families.py"
SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
CANONICAL_SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
ROUTE_FILE = ROOT / "reports/latentfm_trackc_residual_operator_route_teacher_20260623.json"
ANCHOR = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620"
    / "xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt"
)
RUN_ROOT = ROOT / "runs/latentfm_xverse_trackc_support_context_v2_parallel_20260623"
OUT_ROOT = ROOT / "CoupledFM/output/latentfm_runs/xverse_trackc_support_context_v2_parallel_20260623"
OUT_JSON = ROOT / "reports/latentfm_trackc_support_context_v2_parallel_extension_gate_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_PARALLEL_EXTENSION_GATE_20260623.md"

VARIANTS = [
    {
        "run_name": "xverse_trackc_support_context_v2_residual_ep050_replay2_2k_seed42",
        "scope": "support_residual_adapter",
        "support_context": "0",
        "support_residual": "1",
        "support_film": "0",
        "hypothesis": "direct support-context velocity residual may absorb support teacher signal with less canonical harm than FiLM scale",
    },
    {
        "run_name": "xverse_trackc_support_context_v2_contextc_ep050_replay2_2k_seed42",
        "scope": "support_context_adapter",
        "support_context": "1",
        "support_residual": "0",
        "support_film": "0",
        "hypothesis": "conditioning-vector support injection may steer the dynamics through c without direct output scaling",
    },
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def bash_n(path: Path) -> tuple[bool, str]:
    proc = subprocess.run(["bash", "-n", str(path)], text=True, capture_output=True, check=False)
    return proc.returncode == 0, (proc.stderr or proc.stdout).strip()


def main() -> int:
    checks: list[dict] = []
    required = {
        "cpu_gate": CPU_GATE,
        "code_gate": CODE_GATE,
        "launcher_gate": LAUNCH_GATE,
        "generic_launcher": GENERIC_LAUNCHER,
        "extension_launcher": EXT_LAUNCHER,
        "train_py": TRAIN_PY,
        "mlp_py": MLP_PY,
        "config_py": CONFIG_PY,
        "eval_split": EVAL_SPLIT,
        "eval_family": EVAL_FAMILY,
        "safe_trainselect_split": SPLIT,
        "canonical_split": CANONICAL_SPLIT,
        "route_file": ROUTE_FILE,
        "anchor_checkpoint": ANCHOR,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    checks.append({"name": "required_artifacts_exist", "passed": not missing, "evidence": missing})

    cpu = load(CPU_GATE) if CPU_GATE.exists() else {}
    code = load(CODE_GATE) if CODE_GATE.exists() else {}
    launch = load(LAUNCH_GATE) if LAUNCH_GATE.exists() else {}
    checks.append(
        {
            "name": "v2_cpu_gate_passed",
            "passed": cpu.get("status") == "trackc_support_context_v2_cpu_gate_pass_launcher_gate_next_no_gpu",
            "evidence": {"status": cpu.get("status"), "failed": cpu.get("failed_checks")},
        }
    )
    checks.append(
        {
            "name": "v2_code_boundary_passed",
            "passed": code.get("status") == "trackc_support_context_v2_code_boundary_pass_cpu_gate_next"
            and not code.get("hard_failures"),
            "evidence": {"status": code.get("status"), "hard_failures": code.get("hard_failures")},
        }
    )
    checks.append(
        {
            "name": "base_launcher_gate_passed",
            "passed": launch.get("status") == "trackc_support_context_v2_launcher_provenance_gate_pass_launch_allowed",
            "evidence": {"status": launch.get("status"), "gpu": launch.get("gpu_authorization")},
        }
    )

    train_s = text(TRAIN_PY) if TRAIN_PY.exists() else ""
    mlp_s = text(MLP_PY) if MLP_PY.exists() else ""
    config_s = text(CONFIG_PY) if CONFIG_PY.exists() else ""
    generic_s = text(GENERIC_LAUNCHER) if GENERIC_LAUNCHER.exists() else ""
    support_modes_ok = all(
        token in train_s and token in config_s
        for token in ["support_context_adapter", "support_residual_adapter", "support_film_adapter"]
    ) and all(
        token in mlp_s
        for token in [
            "support_context_to_c = nn.Linear",
            "support_context_to_v = nn.Linear",
            "bias=False",
            "support_context_present",
        ]
    )
    checks.append(
        {
            "name": "mechanism_variants_supported_and_masked",
            "passed": support_modes_ok,
            "evidence": "context-c, residual-v, and film scopes exist; support projections are biasless/masked",
        }
    )
    forced_absent_ok = (
        "--force-support-context-absent" in text(EVAL_SPLIT)
        and "--force-support-context-absent" in text(EVAL_FAMILY)
        and "--force-support-context-absent" in generic_s
        and "Canonical posthoc forces support context absent" in generic_s
    )
    checks.append(
        {
            "name": "canonical_posthoc_forces_support_absent",
            "passed": forced_absent_ok,
            "evidence": "split/family eval and generic launcher force support absent for canonical posthoc",
        }
    )
    generic_syntax, generic_err = bash_n(GENERIC_LAUNCHER) if GENERIC_LAUNCHER.exists() else (False, "missing")
    ext_syntax, ext_err = bash_n(EXT_LAUNCHER) if EXT_LAUNCHER.exists() else (False, "missing")
    checks.append({"name": "generic_launcher_bash_n", "passed": generic_syntax, "evidence": generic_err})
    checks.append({"name": "extension_launcher_bash_n", "passed": ext_syntax, "evidence": ext_err})

    preexisting = []
    for variant in VARIANTS:
        run_name = variant["run_name"]
        for path in [RUN_ROOT / run_name, OUT_ROOT / run_name]:
            if path.exists():
                preexisting.append(str(path))
    checks.append({"name": "variant_outputs_not_preexisting", "passed": not preexisting, "evidence": preexisting})

    failed = [c for c in checks if not c["passed"]]
    status = (
        "trackc_support_context_v2_parallel_extension_gate_pass_two_capped_smokes_allowed"
        if not failed
        else "trackc_support_context_v2_parallel_extension_gate_fail_no_launch"
    )
    payload = {
        "status": status,
        "gpu_authorization": "two_capped_v2_variants_after_fresh_resource_audit" if not failed else "none",
        "cpu_cap_cores": 48,
        "physical_gpu_cap": 5,
        "launcher": str(EXT_LAUNCHER),
        "variants": VARIANTS,
        "boundaries": [
            "safe trainselect split only for training/selection/support context",
            "canonical Track A posthoc must force support context absent",
            "held-out Track C query remains forbidden",
            "each variant must use the generic launcher's fresh multi-sample GPU/CPU/RAM audit and RUN_STATUS",
            "selection is capped-smoke support/canonical gate only; no seed expansion unless a variant passes and no-harm is verified",
        ],
        "promotion_gate": {
            "support": "support-val pp/MMD pass in generic smoke decision",
            "canonical": "canonical test_single/family_gene no-harm with support_context_forced_absent=true",
            "next": "passing variant only proceeds to uncapped canonical no-harm; no query before frozen route/checkpoint/no-harm",
        },
        "fail_close": [
            "any failed gate check blocks launch",
            "training/posthoc nonzero exit closes or debugs the specific variant",
            "support gate failure or canonical harm closes the variant",
            "do not treat a passing capped variant as formal multi success",
        ],
        "checks": checks,
        "failed_checks": [c["name"] for c in failed],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Track C Support-Context V2 Parallel Extension Gate",
        "",
        f"Status: `{status}`",
        f"GPU authorization: `{payload['gpu_authorization']}`",
        "",
        "## Variants",
        "",
        "| run | scope | support_c | residual_v | film_v | hypothesis |",
        "|---|---|---:|---:|---:|---|",
    ]
    for v in VARIANTS:
        lines.append(
            f"| `{v['run_name']}` | `{v['scope']}` | `{v['support_context']}` | "
            f"`{v['support_residual']}` | `{v['support_film']}` | {v['hypothesis']} |"
        )
    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| check | passed | evidence |",
            "|---|---:|---|",
        ]
    )
    for check in checks:
        evidence = check["evidence"]
        if isinstance(evidence, (dict, list)):
            evidence_s = json.dumps(evidence, sort_keys=True)
        else:
            evidence_s = str(evidence)
        lines.append(f"| `{check['name']}` | `{check['passed']}` | {evidence_s} |")
    lines.extend(["", "## Failed Checks", ""])
    lines.extend([f"- `{name}`" for name in payload["failed_checks"]] or ["- none"])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This gate does not inspect the active v2 smoke's training/log/exit state.",
            "- It authorizes only two capped support-context v2 mechanism variants after fresh resource audit.",
            "- Held-out query remains forbidden.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
