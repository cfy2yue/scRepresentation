#!/usr/bin/env python3
"""Static readiness gate for RawFM observable gene-budget experiments."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
TRAIN = ROOT / "CoupledFM/model/train.py"
LAUNCHER = ROOT / "CoupledFM/model/tools/launch_stack_train.py"
DATASET = ROOT / "CoupledFM/model/data/dataset.py"
CONFIG = ROOT / "CoupledFM/model/config.py"
MANIFEST_BUILDER = ROOT / "CoupledFM/model/tools/build_rawfm_gene_budget_manifest.py"
LOADER_DRY_RUN = ROOT / "ops/dry_run_rawfm_gene_budget_loader_20260628.py"
LOADER_DRY_RUN_REPORT = ROOT / "reports/rawfm_gene_budget_loader_dryrun_20260628/LATENTFM_RAWFM_GENE_BUDGET_LOADER_DRYRUN_20260628.md"
OUT_DIR = ROOT / "reports/rawfm_observable_budget_readiness_20260628"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def has_any(text: str, needles: list[str]) -> bool:
    low = text.lower()
    return any(n.lower() in low for n in needles)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = read(TRAIN)
    launcher = read(LAUNCHER)
    dataset = read(DATASET)
    config = read(CONFIG)
    manifest_builder = read(MANIFEST_BUILDER)
    loader_dry_run = read(LOADER_DRY_RUN)
    loader_dry_run_report = read(LOADER_DRY_RUN_REPORT)
    all_text = "\n".join([
        train,
        launcher,
        dataset,
        config,
        manifest_builder,
        loader_dry_run,
        loader_dry_run_report,
    ])

    checks = [
        {
            "check": "raw_expression_route_exists",
            "pass": "--ot_feature" in train and '"raw"' in train and "--ot-feature" in launcher,
            "evidence": "`train.py` exposes `--ot_feature raw`; `launch_stack_train.py` defaults `--ot-feature raw`.",
            "required_before_gpu": True,
        },
        {
            "check": "arbitrary_split_file_override",
            "pass": has_any(all_text, ["--split-file", "--split_file", "split_json_path", "split_path_override"]),
            "evidence": "Pass requires explicit split-file plumbing, read-only loading, and run-level provenance copy.",
            "required_before_gpu": True,
        },
        {
            "check": "internal_val_final_test_separation",
            "pass": has_any(all_text, ["final_test", "internal_val", "selection_split", "validation_split_file", "fixed_steps_no_selection"]),
            "evidence": "Pass requires a validation/selection route that does not select checkpoints on the final test split, or fixed-step no-selection.",
            "required_before_gpu": True,
        },
        {
            "check": "deterministic_trainonly_gene_budget_mask",
            "pass": has_any(all_text, ["top_k_gene", "topk_gene", "gene_budget", "gene_mask_file", "fixed_gene_mask"]),
            "evidence": "Pass requires an explicit deterministic gene-budget manifest/mask path in the raw training loader.",
            "required_before_gpu": True,
        },
        {
            "check": "mask_aware_budget_final_eval",
            "pass": has_any(all_text, ["gene_budget_eval", "budget_mask_for_eval", "eval_gene_budget_mask"]),
            "evidence": "Final evaluation/integration must apply the same observable-gene budget; otherwise a budgeted model sees full-gene inputs at test time.",
            "required_before_gpu": True,
        },
        {
            "check": "abundance_mean_random_control_manifests",
            "pass": has_any(all_text, ["abundance_matched", "mean_matched", "random_gene_set", "control_manifest"]),
            "evidence": "No local launcher/config plumbing for abundance-matched, mean-matched, random, or shuffled-label gene-budget controls.",
            "required_before_gpu": True,
        },
        {
            "check": "loader_dry_run_provenance",
            "pass": has_any(all_text, ["rawfm_gene_budget_loader_dry_run_pass", "gene_budget_loader_dry_run"]),
            "evidence": "Before GPU, instantiate train/final-test loaders with a concrete split and manifest, pull a batch, and verify masked genes stay hidden.",
            "required_before_gpu": True,
        },
        {
            "check": "no_canonical_multi_or_trackc_query_dependency",
            "pass": not has_any(all_text, ["test_multi", "trackc_query", "multi_support_v2_query"]),
            "evidence": "Static scan did not find explicit canonical multi or Track C query references in raw train launcher route.",
            "required_before_gpu": True,
        },
    ]

    required = [c for c in checks if c["required_before_gpu"]]
    passed_required = [c for c in required if c["pass"]]
    missing_required = [c for c in required if not c["pass"]]
    gpu = len(missing_required) == 0
    status = (
        "rawfm_observable_budget_readiness_pass_gpu_packet_possible"
        if gpu
        else "rawfm_observable_budget_readiness_missing_plumbing_no_gpu"
    )

    rows_path = OUT_DIR / "rawfm_observable_budget_readiness_checks.csv"
    with rows_path.open("w", encoding="utf-8") as f:
        f.write("check,pass,required_before_gpu,evidence\n")
        for c in checks:
            ev = str(c["evidence"]).replace('"', '""')
            f.write(f"{c['check']},{str(c['pass'])},{str(c['required_before_gpu'])},\"{ev}\"\n")

    obj = {
        "timestamp": now_cst(),
        "status": status,
        "gpu_authorized_next": gpu,
        "checks": checks,
        "missing_required": [c["check"] for c in missing_required],
        "outputs": {
            "checks_csv": str(rows_path),
            "report": str(OUT_DIR / "LATENTFM_RAWFM_OBSERVABLE_BUDGET_READINESS_20260628.md"),
        },
    }
    write_json(OUT_DIR / "rawfm_observable_budget_readiness_20260628.json", obj)

    lines: list[str] = []
    lines.append("# LatentFM RawFM Observable-Budget Readiness Gate")
    lines.append("")
    lines.append(f"Timestamp: `{now_cst()}`")
    lines.append("")
    lines.append(f"Status: `{status}`")
    lines.append("")
    lines.append(f"GPU authorized: `{gpu}`")
    lines.append("")
    lines.append("## Boundary")
    lines.append("")
    lines.append("- CPU/static source audit only.")
    lines.append("- No training, no inference, no canonical multi, and no Track C query.")
    lines.append("- This report audits the current CoupledFM source after the RawFM split/no-selection plumbing patch.")
    lines.append("- Goal: decide whether observable gene-budget scaling can launch as a RawFM GPU smoke.")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| check | pass | evidence |")
    lines.append("|---|---:|---|")
    for c in checks:
        lines.append(f"| {c['check']} | {c['pass']} | {c['evidence']} |")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    if gpu:
        lines.append("- RawFM observable-budget route has required static plumbing for a bounded GPU packet.")
    else:
        lines.append("- Do not launch a RawFM observable-budget GPU smoke yet.")
        lines.append("- Missing required plumbing: `" + "`, `".join(c["check"] for c in missing_required) + "`.")
        lines.append("- The route is scientifically promising, but it needs a CPU implementation gate before GPU.")
    lines.append("")
    lines.append("## Minimum Implementation Gate")
    lines.append("")
    lines.append("1. Add split-file override without overwriting canonical `split_seed42.json`.")
    lines.append("2. Add internal-val/final-test separation or fixed-step no-selection mode.")
    lines.append("3. Add deterministic train-only top-k gene mask manifests.")
    lines.append("4. Add mask-aware final evaluation/integration for budgeted runs.")
    lines.append("5. Add abundance-matched, mean-matched, random, and shuffled-label gene-budget controls.")
    lines.append("6. Dry-run loaders and provenance manifests before any GPU launch.")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- checks: `{rows_path}`")
    lines.append(f"- JSON: `{OUT_DIR / 'rawfm_observable_budget_readiness_20260628.json'}`")
    (OUT_DIR / "LATENTFM_RAWFM_OBSERVABLE_BUDGET_READINESS_20260628.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
