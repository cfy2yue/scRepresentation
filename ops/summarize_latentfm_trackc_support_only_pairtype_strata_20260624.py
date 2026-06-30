#!/usr/bin/env python3
"""Summarize pair-type support-control robustness across Track C seeds.

This is query-free and canonical-free. It reads safe trainselect support-val
posthoc JSONs from completed support-only runs and reports target/non-target
pair-type strata. It does not launch GPU work.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path("/data/cyx/1030/scLatent")
SAFE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
DEFAULT_RUN_ROOT = ROOT / "runs/latentfm_trackc_support_only_robustness_20260624"
OUT_JSON = ROOT / "reports/latentfm_trackc_support_only_pairtype_strata_summary_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_SUPPORT_ONLY_PAIRTYPE_STRATA_SUMMARY_20260624.md"

FILES = {
    "anchor": "support_anchor_split_ode20.json",
    "actual": "support_candidate_split_ode20.json",
    "zero": "support_zero_candidate_split_ode20.json",
    "shuffle": "support_shuffle_condition_candidate_split_ode20.json",
    "absent": "support_absent_support_candidate_split_ode20.json",
}

TARGET_JOINT = "none_train_single|both_train_multi_gene"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pair_genes(condition: str) -> tuple[str, str] | None:
    parts = [x.strip() for x in condition.split("+") if x.strip()]
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def split_strata(split: dict[str, Any], *, target_label: str) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for ds, groups in split.items():
        train_single = set(str(x) for x in groups.get("train_single") or [])
        train_multi_pairs = [pair_genes(str(x)) for x in groups.get("train_multi") or []]
        train_multi_genes = {g for pair in train_multi_pairs if pair for g in pair}
        for cond in groups.get("support_val_multi") or []:
            cond_s = str(cond)
            pair = pair_genes(cond_s)
            if not pair:
                continue
            single_cov = sum(g in train_single for g in pair)
            multi_gene_cov = sum(g in train_multi_genes for g in pair)
            single_label = "both_train_single" if single_cov == 2 else "one_train_single" if single_cov == 1 else "none_train_single"
            multi_label = "both_train_multi_gene" if multi_gene_cov == 2 else "one_train_multi_gene" if multi_gene_cov == 1 else "none_train_multi_gene"
            joint_label = f"{single_label}|{multi_label}"
            out[(str(ds), cond_s)] = {
                "dataset": str(ds),
                "condition": cond_s,
                "single_label": single_label,
                "multi_label": multi_label,
                "joint_label": joint_label,
                "is_target": target_label in {joint_label, single_label, multi_label},
            }
    return out


def assert_safe_payload(path: Path, payload: dict[str, Any]) -> None:
    split_file = Path(str(payload.get("split_file") or ""))
    if split_file != SAFE_SPLIT:
        raise ValueError(f"{path} split_file mismatch: {split_file} != {SAFE_SPLIT}")


def condition_rows(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    groups = payload.get("groups") or {}
    rows = (groups.get("test_multi") or groups.get("test") or {}).get("condition_metrics") or []
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        ds = str(row.get("dataset") or "")
        cond = str(row.get("condition") or "")
        if ds and cond:
            out[(ds, cond)] = row
    return out


def fnum(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def delta_table(
    *,
    anchor: dict[tuple[str, str], dict[str, Any]],
    candidate: dict[tuple[str, str], dict[str, Any]],
    strata: dict[tuple[str, str], dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    buckets: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for key, meta in strata.items():
        if key not in anchor or key not in candidate:
            continue
        a = fnum(anchor[key].get(metric))
        c = fnum(candidate[key].get(metric))
        if a is None or c is None:
            continue
        buckets["target" if meta["is_target"] else "non_target"].append((str(meta["dataset"]), c - a))
        buckets[f"joint:{meta['joint_label']}"].append((str(meta["dataset"]), c - a))
    out: dict[str, Any] = {}
    for label, rows in sorted(buckets.items()):
        by_ds: dict[str, list[float]] = defaultdict(list)
        for ds, delta in rows:
            by_ds[ds].append(delta)
        ds_means = {ds: sum(vals) / len(vals) for ds, vals in by_ds.items() if vals}
        out[label] = {
            "n_conditions": len(rows),
            "n_datasets": len(ds_means),
            "dataset_means": ds_means,
            "equal_dataset_mean_delta": sum(ds_means.values()) / max(1, len(ds_means)),
            "min_dataset_delta": min(ds_means.values()) if ds_means else None,
            "max_dataset_delta": max(ds_means.values()) if ds_means else None,
        }
    return out


def summarize_run(run_name: str, run_root: Path, strata: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    posthoc = run_root / run_name / "posthoc_eval"
    missing = [name for name, rel in FILES.items() if not (posthoc / rel).is_file()]
    if missing:
        return {"run_name": run_name, "status": "incomplete", "missing": missing, "posthoc_dir": str(posthoc)}
    payloads = {}
    for name, rel in FILES.items():
        path = posthoc / rel
        payload = load_json(path)
        assert_safe_payload(path, payload)
        payloads[name] = payload
    rows = {name: condition_rows(payload) for name, payload in payloads.items()}
    anchor = rows["anchor"]
    pp = {name: delta_table(anchor=anchor, candidate=table, strata=strata, metric="pearson_pert") for name, table in rows.items() if name != "anchor"}
    mmd = {name: delta_table(anchor=anchor, candidate=table, strata=strata, metric="test_mmd_clamped") for name, table in rows.items() if name != "anchor"}

    target_actual = pp["actual"].get("target") or {}
    target_mmd = mmd["actual"].get("target") or {}
    non_target_actual = pp["actual"].get("non_target") or {}
    reasons: list[str] = []
    target_pp = float(target_actual.get("equal_dataset_mean_delta") or 0.0)
    target_mmd_delta = float(target_mmd.get("equal_dataset_mean_delta") or 999.0)
    target_min = target_actual.get("min_dataset_delta")
    if int(target_actual.get("n_conditions") or 0) < 4:
        reasons.append("target_n_conditions_lt_4")
    if int(target_actual.get("n_datasets") or 0) < 2:
        reasons.append("target_n_datasets_lt_2")
    if target_pp < 0.04:
        reasons.append("target_pp_delta_lt_0p04")
    if target_mmd_delta > 0.0:
        reasons.append("target_mmd_positive")
    if target_min is None or float(target_min) < -0.01:
        reasons.append("target_dataset_tail_harm")
    for ctrl in ("zero", "shuffle", "absent"):
        ctrl_pp = (pp[ctrl].get("target") or {}).get("equal_dataset_mean_delta")
        if ctrl_pp is None:
            reasons.append(f"{ctrl}_target_missing")
            continue
        ctrl_pp = float(ctrl_pp)
        if ctrl_pp > 0.02:
            reasons.append(f"{ctrl}_target_pp_gt_0p02")
        if target_pp - ctrl_pp < 0.02:
            reasons.append(f"{ctrl}_target_not_0p02_below_actual")
    non_target_pp = non_target_actual.get("equal_dataset_mean_delta")
    non_target_min = non_target_actual.get("min_dataset_delta")
    if non_target_pp is not None and float(non_target_pp) < -0.02:
        reasons.append("non_target_mean_pp_harm")
    if non_target_min is not None and float(non_target_min) < -0.05:
        reasons.append("non_target_tail_pp_harm")
    status = "pass_pairtype_target_support_control" if not reasons else "fail_pairtype_target_support_control"
    return {
        "run_name": run_name,
        "status": status,
        "reasons": reasons,
        "posthoc_dir": str(posthoc),
        "target": {
            "actual_pp": target_actual,
            "actual_mmd": target_mmd,
            "zero_pp": pp["zero"].get("target"),
            "shuffle_pp": pp["shuffle"].get("target"),
            "absent_pp": pp["absent"].get("target"),
        },
        "non_target": {
            "actual_pp": non_target_actual,
            "actual_mmd": mmd["actual"].get("non_target"),
            "zero_pp": pp["zero"].get("non_target"),
            "shuffle_pp": pp["shuffle"].get("non_target"),
            "absent_pp": pp["absent"].get("non_target"),
        },
        "joint": {
            "actual_pp": {k: v for k, v in pp["actual"].items() if k.startswith("joint:")},
            "actual_mmd": {k: v for k, v in mmd["actual"].items() if k.startswith("joint:")},
            "control_pp": {
                ctrl: {k: v for k, v in pp[ctrl].items() if k.startswith("joint:")}
                for ctrl in ("zero", "shuffle", "absent")
            },
        },
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def render_md(payload: dict[str, Any], *, out_json: Path) -> str:
    lines = [
        "# Track C Support-Only Pair-Type Strata Summary",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- Query-free and canonical-free.",
        "- Reads only safe trainselect support-val posthoc/control JSONs.",
        "- Asserts all posthoc JSON `split_file` equals the safe trainselect split.",
        "- Does not launch GPU.",
        f"- Target label: `{payload['boundary']['target_label']}`.",
        "",
        "## Seed Rows",
        "",
        "| run | status | target pp | target MMD | target min ds pp | zero pp | shuffle pp | absent pp | non-target pp | non-target min ds pp | reasons |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["runs"]:
        if row["status"] == "incomplete":
            lines.append(f"| `{row['run_name']}` | `incomplete` | NA | NA | NA | NA | NA | NA | NA | NA | `{row.get('missing')}` |")
            continue
        target = row["target"]
        non_target = row["non_target"]
        lines.append(
            f"| `{row['run_name']}` | `{row['status']}` | "
            f"{fmt((target.get('actual_pp') or {}).get('equal_dataset_mean_delta'))} | "
            f"{fmt((target.get('actual_mmd') or {}).get('equal_dataset_mean_delta'))} | "
            f"{fmt((target.get('actual_pp') or {}).get('min_dataset_delta'))} | "
            f"{fmt((target.get('zero_pp') or {}).get('equal_dataset_mean_delta'))} | "
            f"{fmt((target.get('shuffle_pp') or {}).get('equal_dataset_mean_delta'))} | "
            f"{fmt((target.get('absent_pp') or {}).get('equal_dataset_mean_delta'))} | "
            f"{fmt((non_target.get('actual_pp') or {}).get('equal_dataset_mean_delta'))} | "
            f"{fmt((non_target.get('actual_pp') or {}).get('min_dataset_delta'))} | "
            f"`{row.get('reasons')}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- completed runs: `{payload['n_completed']}`",
            f"- passing runs: `{payload['n_pass']}`",
            f"- stability status: `{payload['stability_status']}`",
            f"- frozen seed rule: `{payload['frozen_seed_rule']}`",
            f"- next action: `{payload['next_action']}`",
            "",
            "## JSON",
            "",
            f"`{out_json}`",
        ]
    )
    return "\n".join(lines) + "\n"


def suffixed_output(path: Path, target_label: str) -> Path:
    if target_label == TARGET_JOINT:
        return path
    safe = target_label.replace("|", "__").replace("/", "_")
    return path.with_name(f"{path.stem}_{safe}{path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    parser.add_argument("--target-joint-label", "--target-label", dest="target_label", default=TARGET_JOINT)
    args = parser.parse_args()

    args.out_json = suffixed_output(args.out_json, args.target_label)
    args.out_md = suffixed_output(args.out_md, args.target_label)
    split = load_json(SAFE_SPLIT)
    strata = split_strata(split, target_label=args.target_label)
    run_rows = [summarize_run(run, args.run_root, strata) for run in args.runs]
    completed = [row for row in run_rows if row["status"] != "incomplete"]
    passed = [row for row in completed if row["status"] == "pass_pairtype_target_support_control"]
    hard_failed = [row for row in completed if row["status"].startswith("fail_") and any("harm" in r or "control" in r or "not_0p02" in r for r in row.get("reasons", []))]
    stability_status = "pending"
    next_action = "wait for remaining seed posthoc before no-harm"
    if len(completed) >= 3:
        if len(passed) >= 2 and not hard_failed:
            stability_status = "pass_2_of_3_no_hard_fail"
            next_action = "freeze lowest-number passing seed for canonical single/family no-harm veto"
        else:
            stability_status = "fail_close_pairtype_branch"
            next_action = "close pair-type branch; do not launch canonical no-harm or query"
    elif completed and len(passed) == 0 and len(completed) >= 2:
        stability_status = "early_fail_no_passing_seeds"
        next_action = "close pair-type branch unless a predeclared remaining seed passes without hard fail"

    payload = {
        "status": "trackc_support_only_pairtype_strata_summary_ready",
        "boundary": {
            "safe_split": str(SAFE_SPLIT),
            "reads_heldout_query": False,
            "reads_canonical_metrics": False,
            "reads_canonical_multi": False,
            "launches_gpu": False,
            "target_label": args.target_label,
            "target_matching": "row is target if target_label matches joint_label, single_label, or multi_label",
        },
        "gate_rules": {
            "target_pp_delta_min": 0.04,
            "target_mmd_delta_max": 0.0,
            "target_min_dataset_pp_floor": -0.01,
            "target_control_pp_max": 0.02,
            "target_actual_minus_control_pp_min": 0.02,
            "non_target_mean_pp_floor": -0.02,
            "non_target_min_dataset_pp_floor": -0.05,
            "stability": "need at least 2 of 3 seeds passing with no hard failed seed before canonical no-harm",
        },
        "runs": run_rows,
        "n_completed": len(completed),
        "n_pass": len(passed),
        "hard_failed_runs": [row["run_name"] for row in hard_failed],
        "stability_status": stability_status,
        "frozen_seed_rule": "lowest-number seed among support/control passing seeds after 2/3 stability; no canonical metric selection",
        "next_action": next_action,
    }
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.write_text(render_md(payload, out_json=args.out_json), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "stability_status": stability_status, "n_completed": len(completed), "n_pass": len(passed), "out_md": str(args.out_md)}, indent=2))


if __name__ == "__main__":
    main()
