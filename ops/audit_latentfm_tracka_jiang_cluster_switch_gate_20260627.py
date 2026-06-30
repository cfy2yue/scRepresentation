#!/usr/bin/env python3
"""CPU gate for a predeclared Jiang-cluster Track A switch policy.

The policy is intentionally simple and non-oracle: use the existing
``xverse_conddelta_seed42`` candidate only for Jiang_* cytokine single-gene
failure-cluster rows, and use anchor elsewhere. It reads only existing paired
posthoc rows and does not train, infer, select checkpoints, read canonical
multi, or read Track C query.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
PAIRED = ROOT / "reports/tracka_exact_tail_candidate_gate_20260627/xverse_conddelta_seed42_paired_rows.csv"
OUT_JSON = ROOT / "reports/latentfm_tracka_jiang_cluster_switch_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_JIANG_CLUSTER_SWITCH_GATE_20260627.md"


GROUPS = (
    "canonical_test_single",
    "canonical_family_gene",
    "exact_simple_single_unseen",
    "exact_cross_background_seen_gene",
    "recurrent_simple_hard_tail",
    "recurrent_cross_background_hard_tail",
)


def is_jiang_cluster(dataset: str, condition: str) -> bool:
    del condition
    return str(dataset).startswith("Jiang_")


def fnum(v: Any) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def bootstrap(vals: list[float], *, seed: int, n_boot: int = 5000) -> dict[str, float | None]:
    if not vals:
        return {"ci_low": None, "ci_high": None, "p_gt0": None, "p_lt0": None}
    arr = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boots = arr[idx].mean(axis=1)
    return {
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
        "p_gt0": float(np.mean(boots > 0.0)),
        "p_lt0": float(np.mean(boots < 0.0)),
    }


def main() -> None:
    grouped: dict[str, list[dict[str, Any]]] = {group: [] for group in GROUPS}
    with PAIRED.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            group = str(row.get("group", ""))
            if group not in grouped:
                continue
            ds = str(row.get("dataset", ""))
            cond = str(row.get("condition", ""))
            pp_delta = fnum(row.get("delta_pearson_pert"))
            mmd_delta = fnum(row.get("delta_test_mmd_clamped"))
            if pp_delta is None or mmd_delta is None:
                continue
            active = is_jiang_cluster(ds, cond)
            grouped[group].append(
                {
                    "dataset": ds,
                    "condition": cond,
                    "active": active,
                    "delta_pearson_pert": pp_delta if active else 0.0,
                    "delta_test_mmd_clamped": mmd_delta if active else 0.0,
                }
            )

    summaries = []
    for idx, group in enumerate(GROUPS):
        rows = grouped[group]
        for metric, lower_is_better in (("delta_pearson_pert", False), ("delta_test_mmd_clamped", True)):
            vals = [float(row[metric]) for row in rows]
            bs = bootstrap(vals, seed=20260627 + idx * 17 + (1 if lower_is_better else 0))
            summaries.append(
                {
                    "group": group,
                    "metric": metric.replace("delta_", ""),
                    "n": len(vals),
                    "n_active": sum(1 for row in rows if row["active"]),
                    "delta_mean": float(np.mean(vals)) if vals else None,
                    "ci_low": bs["ci_low"],
                    "ci_high": bs["ci_high"],
                    "p_improve": bs["p_lt0"] if lower_is_better else bs["p_gt0"],
                    "p_harm": bs["p_gt0"] if lower_is_better else bs["p_lt0"],
                    "lower_is_better": lower_is_better,
                }
            )

    lookup = {(row["group"], row["metric"]): row for row in summaries}
    reasons = []
    exact_cross_pp = lookup[("exact_cross_background_seen_gene", "pearson_pert")]
    exact_cross_mmd = lookup[("exact_cross_background_seen_gene", "test_mmd_clamped")]
    recurrent_cross_pp = lookup[("recurrent_cross_background_hard_tail", "pearson_pert")]
    for group in ("canonical_test_single", "canonical_family_gene", "exact_simple_single_unseen"):
        pp = lookup[(group, "pearson_pert")]
        mmd = lookup[(group, "test_mmd_clamped")]
        if float(pp["delta_mean"] or 0.0) < -0.002 or float(pp["p_harm"] or 0.0) > 0.35:
            reasons.append(f"{group}_pp_noharm_fail")
        if float(mmd["delta_mean"] or 0.0) > 0.001 or float(mmd["p_harm"] or 0.0) > 0.80:
            reasons.append(f"{group}_mmd_noharm_fail")
    if float(exact_cross_pp["delta_mean"] or 0.0) < 0.01 or float(exact_cross_pp["p_improve"] or 0.0) < 0.75:
        reasons.append("exact_cross_material_gain_fail")
    if float(exact_cross_mmd["delta_mean"] or 0.0) > 0.001:
        reasons.append("exact_cross_mmd_noharm_fail")
    if float(recurrent_cross_pp["delta_mean"] or 0.0) < 0.005:
        reasons.append("recurrent_cross_tail_gain_fail")

    status = "tracka_jiang_cluster_switch_pass_trainonly_gate_next_no_gpu" if not reasons else "tracka_jiang_cluster_switch_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_report_only": True,
            "training": False,
            "inference": False,
            "canonical_multi_selection_used": False,
            "trackc_query_used": False,
            "policy": "xverse_conddelta_seed42_only_for_dataset_prefix_Jiang_else_anchor",
        },
        "summaries": summaries,
        "decision_reasons": reasons,
        "next_action": (
            "external audit plus launcher/unit preflight for Jiang-only condition-delta route"
            if not reasons
            else "close Jiang cluster switch unless a materially new train-only selector/no-harm gate is proposed"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Track A Jiang-Cluster Switch Gate",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "CPU/report-only switch over existing paired rows: use `xverse_conddelta_seed42` only for datasets with prefix `Jiang_`; use anchor elsewhere. No training, inference, canonical multi selection, or Track C query.",
        "",
        "## Summaries",
        "",
        "| group | metric | n | active | delta | CI95 | p improve | p harm |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| `{row['group']}` | `{row['metric']}` | {row['n']} | {row['n_active']} | "
            f"{float(row['delta_mean'] or 0):+.6f} | [{row['ci_low']}, {row['ci_high']}] | "
            f"{float(row['p_improve'] or 0):+.6f} | {float(row['p_harm'] or 0):+.6f} |"
        )
    lines.extend(["", "## Decision Reasons", ""])
    lines.extend(f"- `{r}`" for r in reasons) if reasons else lines.append("- none")
    lines.extend(["", "## Outputs", "", f"- JSON: `{OUT_JSON}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "md": str(OUT_MD), "json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
