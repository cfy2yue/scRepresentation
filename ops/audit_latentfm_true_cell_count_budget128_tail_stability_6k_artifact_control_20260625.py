#!/usr/bin/env python3
"""Artifact/paired-tail control for budget128 6k true-cell-count follow-up.

This CPU-only gate compares the completed 3k and 6k budget128 train-only/internal
posthoc outputs. It verifies that the 6k follow-up used the same nested capped
artifacts/splits and that the tail improvement is paired over the same
condition identities. It does not read canonical multi, Track C query, train,
infer, or use GPU.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
RUN3_ROOT = ROOT / "runs/latentfm_true_cell_count_nested_smokes_20260624"
RUN6_ROOT = ROOT / "runs/latentfm_true_cell_count_budget128_tail_stability_6k_20260625"
DEC3 = ROOT / "reports/latentfm_true_cell_count_nested_matrix_decision_20260624.json"
DEC6 = ROOT / "reports/latentfm_true_cell_count_budget128_tail_stability_6k_decision_20260625.json"
OUT_JSON = ROOT / "reports/latentfm_true_cell_count_budget128_tail_stability_6k_artifact_control_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_TRUE_CELL_COUNT_BUDGET128_TAIL_STABILITY_6K_ARTIFACT_CONTROL_20260625.md"

GROUPS = {
    "cross_background": ("split_group", "internal_val_cross_background_seen_gene_proxy"),
    "family_gene": ("condition_family", "family_gene"),
    "test_single": ("condition_family", "test_single"),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_exit(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def run_dir_for(root: Path, seed: int, steps: int) -> Path:
    if steps == 3000:
        return root / f"xverse_truecell_nested_gene_only_fixed256_budget64_128_256_budget128_seed{seed}_3000"
    return root / f"xverse_truecell_nested_budget128_tailstable_seed{seed}_6000"


def status_artifacts(run_dir: Path) -> dict[str, str | None]:
    text = (run_dir / "RUN_STATUS.md").read_text(encoding="utf-8")
    fields = {}
    patterns = {
        "data_dir": r"Capped DATA_DIR: `([^`]+)`",
        "split": r"Split: `([^`]+)`",
        "pert_means": r"Train-only pert means: `([^`]+)`",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        fields[key] = m.group(1) if m else None
    return fields


def group_payload(run_dir: Path, family: str, group: str, role: str) -> dict[str, Any] | None:
    eval_dir = run_dir / "posthoc_eval_internal"
    if family == "split_group":
        path = eval_dir / f"split_group_eval_{role}_internal_ode20.json"
    else:
        path = eval_dir / f"condition_family_eval_{role}_internal_ode20.json"
    if not path.is_file():
        return None
    return ((load_json(path).get("groups") or {}).get(group))


def condition_map(payload: dict[str, Any] | None, metric: str) -> dict[tuple[str, str], float]:
    out = {}
    if not payload:
        return out
    for row in payload.get("condition_metrics") or []:
        value = row.get(metric)
        if value is None:
            continue
        try:
            out[(str(row.get("dataset")), str(row.get("condition")))] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def candidate_minus_anchor_records(run_dir: Path, *, label: str, metric: str) -> list[dict[str, Any]]:
    family, group = GROUPS[label]
    anchor = group_payload(run_dir, family, group, "anchor")
    candidate = group_payload(run_dir, family, group, "candidate")
    amap = condition_map(anchor, metric)
    cmap = condition_map(candidate, metric)
    records = []
    for key in sorted(set(amap) & set(cmap)):
        records.append({"dataset": key[0], "condition": key[1], "delta": float(cmap[key] - amap[key])})
    return records


def by_key(records: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    return {(r["dataset"], r["condition"]): float(r["delta"]) for r in records}


def dataset_tail(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in records:
        by_ds[str(row["dataset"])].append(float(row["delta"]))
    rows = []
    for ds, vals in sorted(by_ds.items()):
        arr = np.asarray(vals, dtype=np.float64)
        rows.append({"dataset": ds, "n": int(arr.size), "mean": float(arr.mean())})
    return {
        "dataset_rows": rows,
        "min_dataset": min(rows, key=lambda r: r["mean"]) if rows else None,
        "negative_tail_lt_minus_0p020": sum(1 for r in rows if r["mean"] < -0.020),
    }


def paired_increment(seed: int, *, label: str, metric: str) -> dict[str, Any]:
    r3 = run_dir_for(RUN3_ROOT, seed, 3000)
    r6 = run_dir_for(RUN6_ROOT, seed, 6000)
    m3 = by_key(candidate_minus_anchor_records(r3, label=label, metric=metric))
    m6 = by_key(candidate_minus_anchor_records(r6, label=label, metric=metric))
    common = sorted(set(m3) & set(m6))
    records = [{"dataset": k[0], "condition": k[1], "delta": float(m6[k] - m3[k])} for k in common]
    return {
        "seed": seed,
        "label": label,
        "metric": metric,
        "n_3k": len(m3),
        "n_6k": len(m6),
        "n_common": len(common),
        "identity_match": len(common) == len(m3) == len(m6) and len(common) > 0,
        "mean_increment": float(np.mean([r["delta"] for r in records])) if records else None,
        "tail": dataset_tail(records),
    }


def budget_row(decision: dict[str, Any], budget: int) -> dict[str, Any] | None:
    for row in (decision.get("matrix_summary") or {}).get("budget_rows", []):
        if int(row.get("budget")) == budget:
            return row
    return None


def main() -> int:
    seeds = [42, 43, 44]
    dec3 = load_json(DEC3)
    dec6 = load_json(DEC6)
    row3 = budget_row(dec3, 128)
    row6 = budget_row(dec6, 128)

    artifact_rows = []
    for seed in seeds:
        r3 = run_dir_for(RUN3_ROOT, seed, 3000)
        r6 = run_dir_for(RUN6_ROOT, seed, 6000)
        a3 = status_artifacts(r3)
        a6 = status_artifacts(r6)
        artifact_rows.append(
            {
                "seed": seed,
                "run3": str(r3),
                "run6": str(r6),
                "train_exit_3k": read_exit(r3 / "EXIT_CODE"),
                "posthoc_exit_3k": read_exit(r3 / "POSTHOC_EXIT_CODE"),
                "train_exit_6k": read_exit(r6 / "EXIT_CODE"),
                "posthoc_exit_6k": read_exit(r6 / "POSTHOC_EXIT_CODE"),
                "same_data_dir": a3.get("data_dir") == a6.get("data_dir"),
                "same_split": a3.get("split") == a6.get("split"),
                "same_pert_means": a3.get("pert_means") == a6.get("pert_means"),
                "artifacts_3k": a3,
                "artifacts_6k": a6,
            }
        )

    increment_rows = []
    for label in ["cross_background", "family_gene", "test_single"]:
        for metric in ["pearson_pert", "test_mmd"]:
            seed_rows = [paired_increment(seed, label=label, metric=metric) for seed in seeds]
            vals = [r["mean_increment"] for r in seed_rows if r["mean_increment"] is not None]
            increment_rows.append(
                {
                    "label": label,
                    "metric": metric,
                    "seed_rows": seed_rows,
                    "mean_increment_over_seeds": float(np.mean(vals)) if vals else None,
                    "all_identity_match": all(r["identity_match"] for r in seed_rows),
                }
            )

    reasons = []
    if not all(r["train_exit_3k"] == 0 and r["posthoc_exit_3k"] == 0 and r["train_exit_6k"] == 0 and r["posthoc_exit_6k"] == 0 for r in artifact_rows):
        reasons.append("paired_runs_not_complete")
    if not all(r["same_data_dir"] and r["same_split"] and r["same_pert_means"] for r in artifact_rows):
        reasons.append("paired_artifacts_or_splits_differ")
    if not all(row["all_identity_match"] for row in increment_rows):
        reasons.append("paired_condition_identity_mismatch")
    if not row6 or row6.get("cross_background_pp_dataset_tail", {}).get("negative_tail_lt_minus_0p020") != 0:
        reasons.append("sixk_cross_background_negative_tail_present")
    if row6 and float(row6.get("cross_background_pp_delta_mean") or -999.0) < float((row3 or {}).get("cross_background_pp_delta_mean") or 999.0):
        reasons.append("sixk_cross_background_mean_below_3k")
    if row6 and float((row6.get("cross_background_pp_condition_bootstrap") or {}).get("ci95", [-999.0])[0]) <= 0.0:
        reasons.append("sixk_cross_background_ci_lower_not_positive")

    status = "budget128_tail_stability_artifact_control_pass_no_gpu" if not reasons else "budget128_tail_stability_artifact_control_fail_no_gpu"
    payload = {
        "status": status,
        "boundary": {
            "cpu_only": True,
            "reads_train_only_internal_posthoc": True,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "artifact_rows": artifact_rows,
        "increment_rows": increment_rows,
        "budget128_3k_summary": row3,
        "budget128_6k_summary": row6,
        "reasons": reasons,
        "gpu_authorized": False,
        "next_action": "external audit then frozen canonical single/family no-harm may be considered" if not reasons else "fix control failure or close 6k tail-stability branch",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM True Cell-Count Budget128 6k Artifact Control",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only paired artifact/tail control for 3k vs 6k budget128 true-cell-count runs.",
        "- Reads train-only/internal posthoc JSON and RUN_STATUS provenance only.",
        "- Does not read canonical multi, Track C query, train, infer, or use GPU.",
        "",
        "## Artifact Identity",
        "",
        "| seed | exits 3k | exits 6k | same data dir | same split | same pert means |",
        "|---:|---|---|---|---|---|",
    ]
    for row in artifact_rows:
        lines.append(
            f"| {row['seed']} | {row['train_exit_3k']}/{row['posthoc_exit_3k']} | "
            f"{row['train_exit_6k']}/{row['posthoc_exit_6k']} | `{row['same_data_dir']}` | "
            f"`{row['same_split']}` | `{row['same_pert_means']}` |"
        )
    lines.extend(
        [
            "",
            "## Budget128 Summary",
            "",
            "| source | cross pp mean | family pp mean | family MMD mean | min cross dataset | neg tails |",
            "|---|---:|---:|---:|---|---:|",
        ]
    )
    for label, row in [("3k", row3), ("6k", row6)]:
        tail = (row or {}).get("cross_background_pp_dataset_tail") or {}
        min_ds = tail.get("min_dataset") or {}
        lines.append(
            f"| `{label}` | {float((row or {}).get('cross_background_pp_delta_mean') or 0.0):+.6f} | "
            f"{float((row or {}).get('family_gene_pp_delta_mean') or 0.0):+.6f} | "
            f"{float((row or {}).get('family_gene_mmd_delta_mean') or 0.0):+.6f} | "
            f"`{min_ds.get('dataset')}` {float(min_ds.get('mean') or 0.0):+.6f} | "
            f"{tail.get('negative_tail_lt_minus_0p020')} |"
        )
    lines.extend(
        [
            "",
            "## Paired 6k Minus 3k Increments",
            "",
            "| group | metric | all condition identity match | mean increment over seeds |",
            "|---|---|---|---:|",
        ]
    )
    for row in increment_rows:
        inc = row["mean_increment_over_seeds"]
        lines.append(
            f"| `{row['label']}` | `{row['metric']}` | `{row['all_identity_match']}` | "
            f"{'NA' if inc is None else f'{inc:+.6f}'} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- reasons: `{reasons or 'none'}`",
            f"- next action: `{payload['next_action']}`",
            "- GPU authorized: `False`",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
