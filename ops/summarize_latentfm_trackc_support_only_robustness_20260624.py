#!/usr/bin/env python3
"""Summarize Track C fixed support-only robustness smoke.

This decision layer is intentionally query-free and canonical-free. It reads
only safe trainselect support-val posthoc JSONs and support-context ablations
for the same frozen checkpoint.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
BOOTSTRAP = ROOT / "ops/bootstrap_latentfm_paired_posthoc_20260621.py"
DEFAULT_EXPECTED_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
METRICS = ["pearson_pert", "test_mmd_clamped"]
SPLIT_GROUPS = ["test_multi", "test"]
FAMILY_GROUPS = ["test_all", "family_gene", "structure_multi", "test_multi"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_expected_split(paths: dict[str, Path], expected_split: Path) -> None:
    expected = str(expected_split.expanduser().resolve())
    bad: list[str] = []
    for key, path in paths.items():
        payload = load_json(path)
        got_raw = str(payload.get("split_file") or "")
        got = str(Path(got_raw).expanduser().resolve()) if got_raw else ""
        if got != expected:
            bad.append(f"{key}: {got_raw or '<missing>'}")
    if bad:
        raise RuntimeError(
            "support-only robustness summarizer expected all posthoc JSONs "
            f"to use safe trainselect split {expected}, but found: {bad}"
        )


def run_bootstrap(
    *,
    baseline: Path,
    candidate: Path,
    groups: list[str],
    out_json: Path,
    out_md: Path,
    title: str,
    n_boot: int,
    seed: int,
    python: str,
) -> dict[str, Any]:
    if not baseline.is_file():
        raise FileNotFoundError(f"missing baseline JSON: {baseline}")
    if not candidate.is_file():
        raise FileNotFoundError(f"missing candidate JSON: {candidate}")
    subprocess.check_call(
        [
            python,
            str(BOOTSTRAP),
            "--baseline-json",
            str(baseline),
            "--candidate-json",
            str(candidate),
            "--groups",
            *groups,
            "--metrics",
            *METRICS,
            "--n-boot",
            str(int(n_boot)),
            "--seed",
            str(int(seed)),
            "--title",
            title,
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )
    return load_json(out_json)


def index_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in payload.get("rows") or []:
        out[f"{row.get('group')}:{row.get('metric')}"] = row
    return out


def first_row(table: dict[str, dict[str, Any]], groups: list[str], metric: str) -> dict[str, Any] | None:
    for group in groups:
        row = table.get(f"{group}:{metric}")
        if row and row.get("status") == "ok" and int(row.get("n_matched_conditions") or 0) > 0:
            return row
    for group in groups:
        row = table.get(f"{group}:{metric}")
        if row:
            return row
    return None


def condition_metric_rows(path: Path, group: str) -> list[dict[str, Any]]:
    payload = load_json(path)
    groups = payload.get("groups") or {}
    group_payload = groups.get(group) or {}
    rows = group_payload.get("condition_metrics") or []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def dataset_deltas(*, baseline: Path, candidate: Path, group: str, metric: str) -> dict[str, float]:
    base_rows = {
        (str(row.get("dataset")), str(row.get("condition"))): row
        for row in condition_metric_rows(baseline, group)
    }
    cand_rows = {
        (str(row.get("dataset")), str(row.get("condition"))): row
        for row in condition_metric_rows(candidate, group)
    }
    by_dataset: dict[str, list[float]] = {}
    for dataset, condition in sorted(set(base_rows) & set(cand_rows)):
        b_value = base_rows[(dataset, condition)].get(metric)
        c_value = cand_rows[(dataset, condition)].get(metric)
        if b_value is None or c_value is None:
            continue
        by_dataset.setdefault(dataset, []).append(float(c_value) - float(b_value))
    return {
        dataset: sum(values) / len(values)
        for dataset, values in sorted(by_dataset.items())
        if values
    }


def f(row: dict[str, Any] | None, key: str, default: float) -> float:
    if not row:
        return default
    value = row.get(key)
    if value is None:
        return default
    return float(value)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def ci(row: dict[str, Any] | None) -> str:
    if not row:
        return "[NA, NA]"
    return f"[{fmt(row.get('ci95_low'))}, {fmt(row.get('ci95_high'))}]"


def row_line(role: str, row: dict[str, Any] | None) -> str:
    if not row:
        return f"| {role} | NA | NA | 0 | 0 | NA | [NA, NA] | NA | NA | missing |"
    return (
        f"| {role} | {row.get('group')} | {row.get('metric')} | "
        f"{row.get('n_matched_conditions', 0)} | {row.get('n_matched_datasets', 0)} | "
        f"{fmt(row.get('delta_mean'))} | {ci(row)} | "
        f"{fmt(row.get('p_improvement'))} | {fmt(row.get('p_harm'))} | {row.get('status', 'NA')} |"
    )


def decide(tables: dict[str, dict[str, dict[str, Any]]], paths: dict[str, Path]) -> dict[str, Any]:
    actual_pp = first_row(tables["actual_split"], SPLIT_GROUPS, "pearson_pert")
    actual_mmd = first_row(tables["actual_split"], SPLIT_GROUPS, "test_mmd_clamped")
    family_pp = first_row(tables["actual_family"], ["family_gene", "test_multi", "test_all"], "pearson_pert")
    family_mmd = first_row(tables["actual_family"], ["family_gene", "test_multi", "test_all"], "test_mmd_clamped")
    controls = {
        "zero_pp": first_row(tables["zero_split"], SPLIT_GROUPS, "pearson_pert"),
        "shuffle_pp": first_row(tables["shuffle_split"], SPLIT_GROUPS, "pearson_pert"),
        "absent_pp": first_row(tables["absent_split"], SPLIT_GROUPS, "pearson_pert"),
    }
    support_group = str(actual_pp.get("group") or "test_multi") if actual_pp else "test_multi"
    support_dataset_deltas = dataset_deltas(
        baseline=paths["support_anchor_split"],
        candidate=paths["support_candidate_split"],
        group=support_group,
        metric="pearson_pert",
    )

    reasons: list[str] = []
    required = {
        "actual_pp": actual_pp,
        "actual_mmd": actual_mmd,
        "family_pp": family_pp,
        "family_mmd": family_mmd,
        **controls,
    }
    for name, row in required.items():
        if not row or row.get("status") != "ok":
            reasons.append(f"missing_or_bad_{name}")
    if not reasons:
        actual_delta = f(actual_pp, "delta_mean", -999.0)
        support_dataset_min = min(support_dataset_deltas.values()) if support_dataset_deltas else -999.0
        if actual_delta < 0.04:
            reasons.append("support_pp_delta_below_0p04")
        if support_dataset_min < 0.0:
            reasons.append("actual_support_dataset_min_delta_below_0")
        if f(actual_pp, "p_harm", 1.0) > 0.20:
            reasons.append("support_pp_harm_above_0p20")
        if f(actual_mmd, "delta_mean", 999.0) > 0.0:
            reasons.append("support_mmd_positive_delta")
        if f(family_pp, "delta_mean", -999.0) < -0.01:
            reasons.append("family_gene_or_alias_pp_harm")
        if f(family_mmd, "delta_mean", 999.0) > 0.0:
            reasons.append("family_gene_or_alias_mmd_harm")
        for name, row in controls.items():
            c_delta = f(row, "delta_mean", 999.0)
            if c_delta > 0.02:
                reasons.append(f"{name}_did_not_collapse_below_0p02")
            if actual_delta - c_delta < 0.02:
                reasons.append(f"{name}_not_0p02_below_actual")

    if not reasons:
        status = "trackc_support_only_robustness_pass_support_gate"
        action = "prepare external review and uncapped support-only no-harm protocol; held-out query remains forbidden"
    elif any(r.startswith("missing_or_bad") for r in reasons):
        status = "trackc_support_only_robustness_missing_metrics_close"
        action = "inspect posthoc artifacts once; do not query-evaluate"
    else:
        status = "trackc_support_only_robustness_fail_close"
        action = "close fixed support-only robustness branch; consider CPU gates for pair-type/coverage-floor only"
    return {
        "status": status,
        "action": action,
        "reasons": reasons,
        "key_rows": {name: row for name, row in required.items()},
        "support_dataset_deltas": support_dataset_deltas,
        "rules": {
            "actual_support_pp_delta_min": 0.04,
            "actual_support_dataset_min_delta_min": 0.0,
            "actual_support_pp_p_harm_max": 0.20,
            "actual_support_mmd_delta_max": 0.0,
            "family_pp_delta_floor": -0.01,
            "family_mmd_delta_max": 0.0,
            "control_pp_delta_max": 0.02,
            "actual_minus_control_pp_delta_min": 0.02,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--expected-split-file", type=Path, default=DEFAULT_EXPECTED_SPLIT)
    args = parser.parse_args()

    if not BOOTSTRAP.is_file():
        raise FileNotFoundError(BOOTSTRAP)
    posthoc = args.run_root / "posthoc_eval"
    paths = {
        "support_anchor_split": posthoc / "support_anchor_split_ode20.json",
        "support_anchor_family": posthoc / "support_anchor_family_ode20.json",
        "support_candidate_split": posthoc / "support_candidate_split_ode20.json",
        "support_candidate_family": posthoc / "support_candidate_family_ode20.json",
        "support_zero_split": posthoc / "support_zero_candidate_split_ode20.json",
        "support_shuffle_split": posthoc / "support_shuffle_condition_candidate_split_ode20.json",
        "support_absent_split": posthoc / "support_absent_support_candidate_split_ode20.json",
    }
    assert_expected_split(paths, args.expected_split_file)
    boot_dir = posthoc / "bootstrap_support_only_decision"
    boot_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "actual_split": run_bootstrap(
            baseline=paths["support_anchor_split"],
            candidate=paths["support_candidate_split"],
            groups=SPLIT_GROUPS,
            out_json=boot_dir / "actual_split.json",
            out_md=boot_dir / "actual_split.md",
            title="Track C Support-Only Actual Split",
            n_boot=args.n_boot,
            seed=args.seed,
            python=args.python,
        ),
        "actual_family": run_bootstrap(
            baseline=paths["support_anchor_family"],
            candidate=paths["support_candidate_family"],
            groups=FAMILY_GROUPS,
            out_json=boot_dir / "actual_family.json",
            out_md=boot_dir / "actual_family.md",
            title="Track C Support-Only Actual Family",
            n_boot=args.n_boot,
            seed=args.seed + 1,
            python=args.python,
        ),
        "zero_split": run_bootstrap(
            baseline=paths["support_anchor_split"],
            candidate=paths["support_zero_split"],
            groups=SPLIT_GROUPS,
            out_json=boot_dir / "zero_split.json",
            out_md=boot_dir / "zero_split.md",
            title="Track C Support-Only Zero-Control Split",
            n_boot=args.n_boot,
            seed=args.seed + 2,
            python=args.python,
        ),
        "shuffle_split": run_bootstrap(
            baseline=paths["support_anchor_split"],
            candidate=paths["support_shuffle_split"],
            groups=SPLIT_GROUPS,
            out_json=boot_dir / "shuffle_split.json",
            out_md=boot_dir / "shuffle_split.md",
            title="Track C Support-Only Shuffle-Control Split",
            n_boot=args.n_boot,
            seed=args.seed + 3,
            python=args.python,
        ),
        "absent_split": run_bootstrap(
            baseline=paths["support_anchor_split"],
            candidate=paths["support_absent_split"],
            groups=SPLIT_GROUPS,
            out_json=boot_dir / "absent_split.json",
            out_md=boot_dir / "absent_split.md",
            title="Track C Support-Only Absent-Control Split",
            n_boot=args.n_boot,
            seed=args.seed + 4,
            python=args.python,
        ),
    }
    tables = {name: index_rows(payload) for name, payload in payloads.items()}
    decision = decide(tables, paths)
    json_payload = {
        "run_root": str(args.run_root),
        "boundary": {
            "selection_split": "split_seed42_multi_support_v2_trainselect support-val only",
            "expected_split_file": str(args.expected_split_file.expanduser().resolve()),
            "heldout_query_read": False,
            "canonical_metrics_read": False,
            "canonical_multi_selection": False,
            "gpu_launched_by_this_script": False,
        },
        "inputs": {key: str(path) for key, path in paths.items()},
        "bootstrap_dir": str(boot_dir),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "decision": decision,
        "tables": tables,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(json_payload, indent=2, sort_keys=True), encoding="utf-8")

    rows = decision["key_rows"]
    lines = [
        "# LatentFM Track C Support-Only Robustness Decision",
        "",
        f"Run root: `{args.run_root}`",
        f"Status: `{decision['status']}`",
        f"Action: `{decision['action']}`",
        "",
        "## Boundary",
        "",
        "- Reads only safe trainselect support-val posthoc and support ablation controls.",
        "- Does not read held-out Track C query, canonical metrics, or canonical multi for selection.",
        "- This is not a final multi claim.",
        "",
        "## Gate Rows",
        "",
        "| role | group | metric | n cond | n ds | delta | 95% CI | p improve | p harm | status |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for role in ["actual_pp", "actual_mmd", "family_pp", "family_mmd", "zero_pp", "shuffle_pp", "absent_pp"]:
        lines.append(row_line(role, rows.get(role)))
    lines.extend(["", "## Support Dataset Deltas", ""])
    support_dataset_deltas = decision.get("support_dataset_deltas") or {}
    if support_dataset_deltas:
        lines.extend(["| dataset | pearson_pert delta |", "|---|---:|"])
        for dataset, delta in sorted(support_dataset_deltas.items()):
            lines.append(f"| {dataset} | {fmt(delta)} |")
    else:
        lines.append("- no matched per-dataset condition rows")
    lines.extend(["", "## Reasons", ""])
    if decision["reasons"]:
        lines.extend(f"- `{r}`" for r in decision["reasons"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Output",
            "",
            f"- JSON: `{args.out_json}`",
            f"- bootstrap dir: `{boot_dir}`",
            "",
        ]
    )
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "action": decision["action"], "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
