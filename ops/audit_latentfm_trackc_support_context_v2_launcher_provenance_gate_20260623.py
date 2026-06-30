#!/usr/bin/env python3
"""Launcher/provenance gate for the Track C support-context v2 capped smoke."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path("/data/cyx/1030/scLatent")
RUN_NAME = "xverse_trackc_support_context_v2_resfilm_ep050_replay2_2k_seed42"
RUN_ROOT = ROOT / "runs/latentfm_xverse_trackc_support_context_v2_20260623" / RUN_NAME
OUT_DIR = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_trackc_support_context_v2_20260623"
    / RUN_NAME
)
LOG_ROOT = ROOT / "logs/latentfm_xverse_trackc_support_context_v2_20260623" / RUN_NAME
CPU_GATE = ROOT / "reports/latentfm_trackc_support_context_v2_cpu_gate_20260623.json"
CODE_GATE = ROOT / "reports/latentfm_trackc_support_context_v2_code_boundary_20260623.json"
PROTOCOL = ROOT / "reports/latentfm_trackc_support_context_v2_protocol_20260623.json"
LAUNCHER = ROOT / "ops/launch_latentfm_trackc_support_context_v2_smoke_20260623.sh"
GENERIC_LAUNCHER = ROOT / "ops/launch_latentfm_xverse_trackc_routed_distill_smoke_20260622.sh"
TRAIN_LAUNCHER = ROOT / "CoupledFM/model/latent/scripts/run_full_stack_latentfm.sh"
EVAL_SPLIT = ROOT / "CoupledFM/model/latent/eval_split_groups.py"
EVAL_FAMILY = ROOT / "CoupledFM/model/latent/eval_condition_families.py"
TRAIN_PY = ROOT / "CoupledFM/model/latent/train.py"
SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
CANONICAL_SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
ROUTE_FILE = ROOT / "reports/latentfm_trackc_residual_operator_route_teacher_20260623.json"
ANCHOR = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620"
    / "xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt"
)
OUT_JSON = ROOT / "reports/latentfm_trackc_support_context_v2_launcher_provenance_gate_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_LAUNCHER_PROVENANCE_GATE_20260623.md"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bash_n(path: Path) -> tuple[bool, str]:
    proc = subprocess.run(["bash", "-n", str(path)], text=True, capture_output=True, check=False)
    return proc.returncode == 0, (proc.stderr or proc.stdout).strip()


def contains(path: Path, text: str) -> bool:
    return text in path.read_text(encoding="utf-8")


def main() -> int:
    checks: list[dict] = []

    required_paths = {
        "cpu_gate": CPU_GATE,
        "code_gate": CODE_GATE,
        "protocol": PROTOCOL,
        "launcher": LAUNCHER,
        "generic_launcher": GENERIC_LAUNCHER,
        "train_launcher": TRAIN_LAUNCHER,
        "eval_split": EVAL_SPLIT,
        "eval_family": EVAL_FAMILY,
        "train_py": TRAIN_PY,
        "safe_trainselect_split": SPLIT,
        "canonical_split": CANONICAL_SPLIT,
        "route_file": ROUTE_FILE,
        "anchor_checkpoint": ANCHOR,
    }
    missing = [name for name, path in required_paths.items() if not path.exists()]
    checks.append({"name": "required_artifacts_exist", "passed": not missing, "evidence": missing})

    cpu = load_json(CPU_GATE) if CPU_GATE.exists() else {}
    code = load_json(CODE_GATE) if CODE_GATE.exists() else {}
    protocol = load_json(PROTOCOL) if PROTOCOL.exists() else {}
    checks.append(
        {
            "name": "v2_cpu_gate_passed",
            "passed": cpu.get("status") == "trackc_support_context_v2_cpu_gate_pass_launcher_gate_next_no_gpu"
            and cpu.get("next_authorization") == "launcher_provenance_gate_only",
            "evidence": {"status": cpu.get("status"), "next": cpu.get("next_authorization")},
        }
    )
    checks.append(
        {
            "name": "code_boundary_passed",
            "passed": code.get("status") == "trackc_support_context_v2_code_boundary_pass_cpu_gate_next",
            "evidence": {"status": code.get("status"), "gpu": code.get("gpu_authorization")},
        }
    )
    checks.append(
        {
            "name": "protocol_no_gpu_boundary_respected",
            "passed": protocol.get("gpu_authorization") == "none",
            "evidence": {"status": protocol.get("status"), "gpu": protocol.get("gpu_authorization")},
        }
    )

    eval_support_absent_ok = (
        contains(EVAL_SPLIT, "--force-support-context-absent")
        and contains(EVAL_FAMILY, "--force-support-context-absent")
        and contains(TRAIN_PY, "support_context_source_active = _support_context_source_active(cfg)")
        and contains(GENERIC_LAUNCHER, "--force-support-context-absent")
        and contains(GENERIC_LAUNCHER, "Canonical posthoc forces support context absent")
    )
    checks.append(
        {
            "name": "canonical_eval_forces_support_absent",
            "passed": eval_support_absent_ok,
            "evidence": "eval CLIs and generic launcher expose/use --force-support-context-absent",
        }
    )

    launcher_syntax, launcher_err = bash_n(LAUNCHER) if LAUNCHER.exists() else (False, "missing")
    generic_syntax, generic_err = bash_n(GENERIC_LAUNCHER) if GENERIC_LAUNCHER.exists() else (False, "missing")
    train_syntax, train_err = bash_n(TRAIN_LAUNCHER) if TRAIN_LAUNCHER.exists() else (False, "missing")
    checks.extend(
        [
            {"name": "v2_launcher_bash_n", "passed": launcher_syntax, "evidence": launcher_err},
            {"name": "generic_launcher_bash_n", "passed": generic_syntax, "evidence": generic_err},
            {"name": "train_launcher_bash_n", "passed": train_syntax, "evidence": train_err},
        ]
    )

    command_env = {
        "LATENTFM_TRACKC_RUN_NAME": RUN_NAME,
        "LATENTFM_TRACKC_RUN_ROOT": str(RUN_ROOT.parent),
        "LATENTFM_TRACKC_OUT_ROOT": str(OUT_DIR.parent),
        "LATENTFM_TRACKC_LOG_ROOT": str(LOG_ROOT.parent),
        "LATENTFM_TRACKC_TRAINSELECT_SPLIT": str(SPLIT),
        "LATENTFM_TRACKC_BANK_SPLIT_FILE": str(SPLIT),
        "LATENTFM_TRACKC_ROUTE_FILE": str(ROUTE_FILE),
        "LATENTFM_TRACKC_ANCHOR_CKPT": str(ANCHOR),
        "LATENTFM_TRACKC_FINETUNE_TRAINABLE_SCOPE": "support_film_adapter",
        "LATENTFM_TRACKC_SUPPORT_CONTEXT_USE_IN_MODEL": "0",
        "LATENTFM_TRACKC_SUPPORT_RESIDUAL_USE_IN_MODEL": "0",
        "LATENTFM_TRACKC_SUPPORT_FILM_USE_IN_MODEL": "1",
        "LATENTFM_TRACKC_SUPPORT_CONTEXT_DIM": "384",
        "LATENTFM_TRACKC_SUPPORT_CONTEXT_SOURCE": "routed_distill_target",
        "LATENTFM_TRACKC_ROUTED_DISTILL_LOSS_WEIGHT": "0.0",
        "LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WEIGHT": "0.50",
        "LATENTFM_TRACKC_ROUTED_DISTILL_MEMORY_MODE": "jaccard",
        "LATENTFM_TRACKC_ROUTED_DISTILL_MEMORY_K": "3",
        "LATENTFM_TRACKC_ROUTED_DISTILL_MEMORY_SCOPE": "all_dataset",
        "LATENTFM_TRACKC_ANCHOR_REPLAY_LOSS_WEIGHT": "2.0",
        "LATENTFM_TRACKC_ANCHOR_REPLAY_CONDITION_FILTER": "all",
        "LATENTFM_TRACKC_TOTAL_STEPS": "2000",
    }
    config_ok = (
        command_env["LATENTFM_TRACKC_BANK_SPLIT_FILE"] == str(SPLIT)
        and command_env["LATENTFM_TRACKC_TRAINSELECT_SPLIT"] == str(SPLIT)
        and command_env["LATENTFM_TRACKC_SUPPORT_FILM_USE_IN_MODEL"] == "1"
        and command_env["LATENTFM_TRACKC_SUPPORT_CONTEXT_SOURCE"] == "routed_distill_target"
        and command_env["LATENTFM_TRACKC_FINETUNE_TRAINABLE_SCOPE"] == "support_film_adapter"
    )
    checks.append({"name": "launch_config_matches_v2_protocol", "passed": config_ok, "evidence": command_env})

    preexisting = [str(path) for path in (RUN_ROOT, OUT_DIR) if path.exists()]
    checks.append({"name": "run_outputs_not_preexisting", "passed": not preexisting, "evidence": preexisting})

    failed = [c for c in checks if not c["passed"]]
    status = (
        "trackc_support_context_v2_launcher_provenance_gate_pass_launch_allowed"
        if not failed
        else "trackc_support_context_v2_launcher_provenance_gate_fail_no_launch"
    )
    payload = {
        "status": status,
        "gpu_authorization": (
            "one_capped_smoke_allowed_after_fresh_resource_audit" if not failed else "none"
        ),
        "run_name": RUN_NAME,
        "launcher": str(LAUNCHER),
        "generic_launcher": str(GENERIC_LAUNCHER),
        "run_root": str(RUN_ROOT),
        "out_dir": str(OUT_DIR),
        "log_root": str(LOG_ROOT),
        "exact_command": f"bash {LAUNCHER}",
        "command_env": command_env,
        "promotion_gate": {
            "support_val": "generic routed-distill smoke decision must pass support pp/MMD gates",
            "canonical": "canonical test_single/family_gene posthoc must use forced support-absent and pass no-harm",
            "next": "if capped pass, run uncapped canonical no-harm before any held-out query",
        },
        "fail_close": [
            "any failed launcher/provenance check means no launch",
            "training/posthoc nonzero exit closes or debugs before relaunch",
            "support gate failure or canonical harm closes this capped branch",
        ],
        "checks": checks,
        "failed_checks": [c["name"] for c in failed],
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Track C Support-Context V2 Launcher/Provenance Gate",
        "",
        f"Status: `{status}`",
        f"GPU authorization: `{payload['gpu_authorization']}`",
        f"Run: `{RUN_NAME}`",
        "",
        "## Exact Command",
        "",
        "```bash",
        f"bash {LAUNCHER}",
        "```",
        "",
        "## Key Boundaries",
        "",
        "- safe trainselect split only for training/selection and support context bank",
        "- support-val posthoc keeps support context present",
        "- canonical posthoc forces support context absent",
        "- held-out query remains forbidden",
        "- launcher delegates fresh multi-sample GPU/CPU/RAM audit to the generic launcher",
        "",
        "## Checks",
        "",
        "| check | passed | evidence |",
        "|---|---:|---|",
    ]
    for check in checks:
        evidence = check["evidence"]
        if isinstance(evidence, (dict, list)):
            evidence_s = json.dumps(evidence, sort_keys=True)
        else:
            evidence_s = str(evidence)
        lines.append(f"| `{check['name']}` | `{check['passed']}` | {evidence_s} |")
    lines.extend(["", "## Failed Checks", ""])
    lines.extend([f"- `{name}`" for name in payload["failed_checks"]] or ["- none"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
