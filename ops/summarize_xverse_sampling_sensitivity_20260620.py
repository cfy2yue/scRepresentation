#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BASELINE_RUN = (
    "/data/cyx/1030/scLatent/CoupledFM/output/latentfm_runs/"
    "xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval"
)
SAMPLING_ROOT = (
    "/data/cyx/1030/scLatent/CoupledFM/output/latentfm_runs/"
    "xverse_sampling_sensitivity_20260620"
)
DEFAULT_RUNS = [
    "xverse_comp006_endpoint5_visitcap8_power05_floor32_4k_seed42",
    "xverse_comp006_endpoint5_visitcap8_power05_floor32_dsloss05_4k_seed42",
]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def get_group(path: Path, group: str) -> dict[str, Any] | None:
    data = read_json(path)
    if data is None:
        return None
    groups = data.get("groups") or data.get("results") or data
    if isinstance(groups, dict):
        return groups.get(group)
    for item in groups:
        name = item.get("group") or item.get("family") or item.get("name")
        if name == group:
            return item
    return None


def metric(data: dict[str, Any] | None, key: str) -> float | None:
    if data is None or key not in data or data[key] is None:
        return None
    return float(data[key])


def passes_threshold(value: float | None, threshold: float, op: str) -> bool | None:
    if value is None:
        return None
    if op == ">=":
        return value >= threshold
    if op == "<=":
        return value <= threshold
    raise ValueError(op)


def status_label(value: bool | None) -> str:
    if value is None:
        return "missing"
    return "pass" if value else "fail"


def summarize_run(run_root: Path, baseline: dict[str, float]) -> dict[str, Any]:
    iid = read_json(run_root / "iid_eval_results.json")
    split_posthoc = run_root / "posthoc_eval/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json"
    family_posthoc = run_root / "posthoc_eval/condition_family_eval_best_ode20_mse1024_mmd1024_stablecaps.json"
    test = get_group(split_posthoc, "test")
    unseen2 = get_group(split_posthoc, "test_multi_unseen2")
    family_gene = get_group(family_posthoc, "family_gene")

    artifacts = {
        "best.pt": (run_root / "best.pt").exists(),
        "latest.pt": (run_root / "latest.pt").exists(),
        "config.json": (run_root / "config.json").exists(),
        "iid_eval_results.json": iid is not None,
        "split_posthoc": split_posthoc.exists(),
        "family_posthoc": family_posthoc.exists(),
    }
    has_core_artifacts = all(
        artifacts[name]
        for name in ["best.pt", "latest.pt", "config.json", "iid_eval_results.json"]
    )

    iid_summary = {
        "n_conds": metric(iid, "n_conds"),
        "n_available_conditions": metric(iid, "n_available_conditions"),
        "test_mmd": metric(iid, "test_mmd"),
        "test_mmd_biased": metric(iid, "test_mmd_biased"),
        "test_mmd_clamped": metric(iid, "test_mmd_clamped"),
        "pearson_pert": metric(iid, "pearson_pert"),
        "pearson_ctrl": metric(iid, "pearson_ctrl"),
        "direct_pearson": metric(iid, "direct_pearson"),
        "test_mse": metric(iid, "test_mse"),
        "test_mae": metric(iid, "test_mae"),
    }
    posthoc_summary = {
        "test": {
            "n_conds": metric(test, "n_conds"),
            "test_mmd": metric(test, "test_mmd"),
            "test_mmd_clamped": metric(test, "test_mmd_clamped"),
            "pearson_pert": metric(test, "pearson_pert"),
            "pearson_ctrl": metric(test, "pearson_ctrl"),
            "direct_pearson": metric(test, "direct_pearson"),
        },
        "test_multi_unseen2": {
            "n_conds": metric(unseen2, "n_conds"),
            "test_mmd": metric(unseen2, "test_mmd"),
            "test_mmd_clamped": metric(unseen2, "test_mmd_clamped"),
            "pearson_pert": metric(unseen2, "pearson_pert"),
            "pearson_ctrl": metric(unseen2, "pearson_ctrl"),
            "direct_pearson": metric(unseen2, "direct_pearson"),
        },
        "family_gene": {
            "n_conds": metric(family_gene, "n_conds"),
            "test_mmd": metric(family_gene, "test_mmd"),
            "test_mmd_clamped": metric(family_gene, "test_mmd_clamped"),
            "pearson_pert": metric(family_gene, "pearson_pert"),
            "pearson_ctrl": metric(family_gene, "pearson_ctrl"),
            "direct_pearson": metric(family_gene, "direct_pearson"),
        },
    }

    posthoc_gate = {
        "core_artifacts": has_core_artifacts,
        "test_multi_unseen2_pp": passes_threshold(
            posthoc_summary["test_multi_unseen2"]["pearson_pert"],
            baseline["test_multi_unseen2_pp"] + 0.02,
            ">=",
        ),
        "test_pp": passes_threshold(
            posthoc_summary["test"]["pearson_pert"],
            baseline["test_pp"],
            ">=",
        ),
        "family_gene_pp": passes_threshold(
            posthoc_summary["family_gene"]["pearson_pert"],
            baseline["family_gene_pp"] - 0.01,
            ">=",
        ),
        "test_mmd": passes_threshold(
            posthoc_summary["test"]["test_mmd"],
            baseline["test_mmd"] * 1.15,
            "<=",
        ),
    }
    posthoc_gate["all_declared_posthoc_checks"] = all(
        value is True for value in posthoc_gate.values()
    )

    iid_gate = {
        "core_artifacts": has_core_artifacts,
        "iid_pp_ge_baseline": passes_threshold(
            iid_summary["pearson_pert"], baseline["iid_pp"], ">="
        ),
        "iid_pc_ge_baseline": passes_threshold(
            iid_summary["pearson_ctrl"], baseline["iid_pc"], ">="
        ),
        "iid_mmd_le_1p15x_baseline": passes_threshold(
            iid_summary["test_mmd"], baseline["iid_mmd"] * 1.15, "<="
        ),
        "direct_not_collapsed": passes_threshold(
            iid_summary["direct_pearson"], baseline["iid_direct"] - 0.005, ">="
        ),
    }
    iid_gate["triage_posthoc_allowed"] = all(value is True for value in iid_gate.values())

    if not has_core_artifacts:
        recommendation = "wait_for_completion"
    elif not iid_gate["triage_posthoc_allowed"]:
        recommendation = "do_not_posthoc_close_or_rethink"
    elif not artifacts["split_posthoc"] or not artifacts["family_posthoc"]:
        recommendation = "run_stablecaps_split_family_posthoc"
    elif posthoc_gate["all_declared_posthoc_checks"]:
        recommendation = "triage_pass_uncapped_required_before_seeds"
    else:
        recommendation = "posthoc_failed_do_not_seed_expand"

    return {
        "run": run_root.name,
        "root": str(run_root),
        "artifacts": artifacts,
        "iid": iid_summary,
        "posthoc": posthoc_summary,
        "iid_gate": iid_gate,
        "posthoc_gate": posthoc_gate,
        "recommendation": recommendation,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# xverse Sampling Sensitivity Decision",
        "",
        "Baseline: `xverse_comp006_endpoint5_8k_seed42_fulleval` stable-caps posthoc.",
        "",
        "## Baseline Gate Values",
        "",
    ]
    for key, value in summary["baseline"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Runs", ""])
    for run in summary["runs"]:
        lines.append(f"### {run['run']}")
        lines.append("")
        lines.append(f"Recommendation: `{run['recommendation']}`")
        lines.append("")
        lines.append("IID:")
        for key, value in run["iid"].items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
        lines.append("IID gate:")
        for key, value in run["iid_gate"].items():
            lines.append(f"- `{key}`: `{status_label(value)}`")
        lines.append("")
        lines.append("Posthoc gate:")
        for key, value in run["posthoc_gate"].items():
            lines.append(f"- `{key}`: `{status_label(value)}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", default=BASELINE_RUN)
    parser.add_argument("--sampling-root", default=SAMPLING_ROOT)
    parser.add_argument("--run", action="append", default=None)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    baseline_root = Path(args.baseline_run)
    baseline_iid = read_json(baseline_root / "iid_eval_results.json")
    baseline_split = baseline_root / "posthoc_eval/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json"
    baseline_family = baseline_root / "posthoc_eval/condition_family_eval_best_ode20_mse1024_mmd1024_stablecaps.json"
    baseline_test = get_group(baseline_split, "test")
    baseline_unseen2 = get_group(baseline_split, "test_multi_unseen2")
    baseline_gene = get_group(baseline_family, "family_gene")

    baseline = {
        "iid_pp": metric(baseline_iid, "pearson_pert"),
        "iid_pc": metric(baseline_iid, "pearson_ctrl"),
        "iid_mmd": metric(baseline_iid, "test_mmd"),
        "iid_direct": metric(baseline_iid, "direct_pearson"),
        "test_pp": metric(baseline_test, "pearson_pert"),
        "test_mmd": metric(baseline_test, "test_mmd"),
        "test_multi_unseen2_pp": metric(baseline_unseen2, "pearson_pert"),
        "family_gene_pp": metric(baseline_gene, "pearson_pert"),
    }
    missing = [key for key, value in baseline.items() if value is None]
    if missing:
        raise SystemExit(f"Missing baseline metrics: {missing}")

    run_names = args.run or DEFAULT_RUNS
    summary = {
        "baseline_run": str(baseline_root),
        "sampling_root": args.sampling_root,
        "baseline": baseline,
        "runs": [
            summarize_run(Path(args.sampling_root) / run_name, baseline)
            for run_name in run_names
        ],
    }
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(summary, out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
