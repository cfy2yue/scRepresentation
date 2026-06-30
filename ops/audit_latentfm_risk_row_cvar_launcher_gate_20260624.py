#!/usr/bin/env python3
"""Fail-closed launcher/provenance gate for one risk-row CVaR train-only smoke."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
LAUNCHER = ROOT / "ops/launch_latentfm_risk_row_cvar_trainonly_smoke_20260624.sh"
TRAIN_LAUNCHER = ROOT / "CoupledFM/model/latent/scripts/run_full_stack_latentfm.sh"
TRAIN = ROOT / "CoupledFM/model/latent/train.py"
CODE_GATE_JSON = ROOT / "reports/latentfm_risk_row_cvar_loss_code_gate_20260624.json"
EXTERNAL_AUDIT = ROOT / "reports/LATENTFM_RISK_ROW_CVAR_EXTERNAL_AUDIT_BERNOULLI_20260624.md"
SPLIT = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_general_exposure_cap_v2.json"
PERT_MEANS = ROOT / "runs/latentfm_xverse_scaling_splits_v2_20260624/artifacts/xverse_trainonly_scaling_general_exposure_cap_v2_pert_means.npz"
ANCHOR = ROOT / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt"
OUT_JSON = ROOT / "reports/latentfm_risk_row_cvar_launcher_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_RISK_ROW_CVAR_LAUNCHER_GATE_20260624.md"

RUN_NAME = "xverse_risk_row_cvar_allrisk_w020_2k_seed42"
RISK_FILTER = (
    "Nadig_hepg2,Nadig_jurket,NormanWeissman2019_filtered,"
    "ReplogleWeissman2022_K562_gwps,Replogle_RPE1essential,TianActivation"
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def add(checks: list[dict[str, Any]], name: str, passed: bool, evidence: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "evidence": evidence})


def contains_all(text: str, needles: list[str]) -> bool:
    return all(needle in text for needle in needles)


def main() -> int:
    launcher = read(LAUNCHER)
    train_launcher = read(TRAIN_LAUNCHER)
    train = read(TRAIN)
    code_gate = json.loads(CODE_GATE_JSON.read_text(encoding="utf-8"))
    split = json.loads(SPLIT.read_text(encoding="utf-8"))

    checks: list[dict[str, Any]] = []
    add(
        checks,
        "required_artifacts_exist",
        all(p.exists() for p in [LAUNCHER, TRAIN_LAUNCHER, TRAIN, CODE_GATE_JSON, EXTERNAL_AUDIT, SPLIT, PERT_MEANS, ANCHOR]),
        "Launcher, shared train launcher, source, code gate, audit, split, pert means, and anchor exist.",
    )
    add(
        checks,
        "code_gate_pass_no_gpu",
        code_gate.get("status") == "risk_row_cvar_loss_code_gate_pass_unit_validated_no_gpu",
        f"code gate status={code_gate.get('status')!r}",
    )
    add(
        checks,
        "exactly_one_run_name",
        f"RUN_NAME=${{RISK_ROW_RUN_NAME:-{RUN_NAME}}}" in launcher
        and "for arm" not in launcher
        and "POSTHOC" not in launcher
        and "SUMMARIZER" not in launcher,
        f"default run name is {RUN_NAME}; no posthoc/summarizer branch detected.",
    )
    add(
        checks,
        "fresh_output_dirs_fail_closed",
        "Run/output/log dir already exists" in launcher
        and "FORCE_LATENTFM_RISK_ROW_RERUN" not in launcher,
        "Launcher rejects existing run/output/log dirs and has no force-rerun bypass.",
    )
    add(
        checks,
        "trainonly_xverse_split_and_pert_means",
        "split_seed42_xverse_trainonly_scaling_general_exposure_cap_v2.json" in launcher
        and "xverse_trainonly_scaling_general_exposure_cap_v2_pert_means.npz" in launcher
        and "/dataset/biFlow_data/split_seed42.json" not in launcher,
        "Uses train-only xverse general-exposure v2 split and matching train-only pert means.",
    )
    add(
        checks,
        "exact_fixed_six_dataset_risk_filter",
        f"RISK_FILTER=${{RISK_ROW_CVAR_DATASET_FILTER_VALUE:-{RISK_FILTER}}}" in launcher,
        f"Risk filter is exact fixed six-dataset set: {RISK_FILTER}",
    )
    risk_counts = {
        ds: len((split.get(ds) or {}).get("train") or [])
        for ds in RISK_FILTER.split(",")
    }
    add(
        checks,
        "risk_filter_datasets_have_train_rows",
        all(count > 0 for count in risk_counts.values()),
        f"train counts={risk_counts}",
    )
    add(
        checks,
        "required_train_env_flags",
        contains_all(
            launcher,
            [
                "export TOTAL_STEPS=2000",
                "export TRAIN_EVAL_ENABLED=0",
                "export RISK_ROW_CVAR_LOSS_WEIGHT=0.20",
                "export RISK_ROW_CVAR_LOSS_WARMUP_END=500",
                "export INIT_CHECKPOINT_USE_EMA=1",
                "export INIT_CHECKPOINT=${ANCHOR_CKPT}",
                "export MMD_DATASET_FILTER=",
            ],
        ),
        "Capped 2k, train-eval disabled, nonzero risk-row weight, warmup before cap, EMA anchor, scalar MMD filter empty.",
    )
    add(
        checks,
        "shared_launcher_forwards_flags",
        contains_all(
            train_launcher,
            [
                "RISK_ROW_CVAR_LOSS_WEIGHT",
                "--risk-row-cvar-loss-weight",
                "--risk-row-cvar-dataset-filter",
                "TRAIN_EVAL_ENABLED",
                "--no-train-eval-enabled",
            ],
        ),
        "Shared launcher forwards risk-row and train-eval-disable flags.",
    )
    add(
        checks,
        "train_no_eval_source_guard",
        "if is_rank0 and train_eval_enabled:" in train
        and "Test  conditions (IID): skipped because train_eval_enabled=False" in train
        and "Final IID/OOD evaluation skipped because train_eval_enabled=False" in train,
        "No-eval mode skips IID test dataset construction and final eval.",
    )
    add(
        checks,
        "risk_row_log_signals_present",
        "risk_row_obs=" in train
        and "risk_row_apply=" in train
        and "avg_risk_row_cvar_w=" in train
        and "risk_row_cvar_weight=" in train_launcher,
        "Train/shared launcher logs expose risk-row config, observe/apply counts, and active tail weight.",
    )
    add(
        checks,
        "provenance_snapshot_present",
        "provenance.json" in launcher
        and "git_status_short.txt" in launcher
        and "sha256" in launcher
        and "git\", \"-C\", str(repo), \"status\", \"--short\"" in launcher,
        "Launcher snapshots git status and SHA256 hashes for key files.",
    )
    add(
        checks,
        "run_status_before_tmux",
        launcher.find("cat > \"${RUN_DIR}/RUN_STATUS.md\"") != -1
        and launcher.find("tmux new -d -s") != -1
        and launcher.find("cat > \"${RUN_DIR}/RUN_STATUS.md\"") < launcher.find("tmux new -d -s"),
        "RUN_STATUS is written before tmux launch.",
    )
    add(
        checks,
        "resource_audit_three_sample_policy",
        contains_all(
            launcher,
            [
                "--samples 3",
                "--interval-seconds 10",
                "--memory-threshold-mib 4096",
                "--util-threshold-pct 10",
                "--max-user-gpus 4",
                "--max-jobs-per-gpu 4",
                "ps -u cyx",
            ],
        ),
        "Launcher performs required repeated GPU audit and records CPU/RAM/process snapshot.",
    )
    add(
        checks,
        "thread_budget_small",
        contains_all(
            launcher,
            [
                "export OMP_NUM_THREADS=4",
                "export MKL_NUM_THREADS=4",
                "export OPENBLAS_NUM_THREADS=4",
                "export NUMEXPR_NUM_THREADS=4",
                "export BLIS_NUM_THREADS=4",
            ],
        ),
        "Single smoke uses 4-thread env settings, well under 48-core project cap.",
    )
    add(
        checks,
        "no_canonical_multi_or_trackc_query_strings",
        not re.search(
            r"eval_split_groups|eval_condition_families|POSTHOC_EXIT_CODE|posthoc_script|SUMMARIZER|trackc",
            launcher,
            re.I,
        ),
        "Launcher source does not call canonical/posthoc evaluators or Track C artifacts.",
    )

    failed = [row for row in checks if not row["passed"]]
    status = "risk_row_cvar_launcher_gate_pass_one_trainonly_smoke_allowed" if not failed else "risk_row_cvar_launcher_gate_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": status.endswith("_allowed"),
        "authorized_run_name": RUN_NAME if not failed else None,
        "authorized_command": (
            f"LATENTFM_RISK_ROW_CVAR_ACK=trainonly_noeval_one_capped_smoke bash {LAUNCHER}"
            if not failed
            else None
        ),
        "boundary": {
            "exactly_one_run": True,
            "train_only": True,
            "train_eval_enabled": False,
            "canonical_metrics_read": False,
            "canonical_multi_read": False,
            "trackc_query_read": False,
            "posthoc_chained": False,
        },
        "checks": checks,
        "failed_checks": failed,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM Risk-Row CVaR Launcher Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Checks",
        "",
        "| check | pass | evidence |",
        "|---|---:|---|",
    ]
    for row in checks:
        evidence = str(row["evidence"]).replace("|", "\\|")
        lines.append(f"| `{row['name']}` | `{row['passed']}` | {evidence} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- GPU authorized: `{payload['gpu_authorized']}`.",
            "- Authorization is limited to exactly one capped train-only smoke.",
            "- No canonical no-harm, canonical multi, Track C query, or posthoc evaluation is authorized by this gate.",
            "- Before promotion or extension, the completed training log must show `risk_row_obs>0` and `risk_row_apply>0` or `avg_risk_row_cvar_w>0`.",
            "",
            "## Output",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
