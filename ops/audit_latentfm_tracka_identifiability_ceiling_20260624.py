#!/usr/bin/env python3
"""Track A identifiability / ceiling audit after failed model-search gates."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
MEAN_DIR = ROOT / "reports/latentfm_xverse_nuisance_condition_means_20260624"
ANCHOR_MEANS = MEAN_DIR / "split_group_eval_anchor_internal_means_ode20.json"
CAP120_MEANS = MEAN_DIR / "split_group_eval_cap120_internal_means_ode20.json"
OUT_JSON = ROOT / "reports/latentfm_tracka_identifiability_ceiling_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_IDENTIFIABILITY_CEILING_20260624.md"

GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
BOOT_N = 1000
SEED = 42

GATE_REPORTS = {
    "reliability_condition_cap120": ROOT / "reports/latentfm_trainonly_reliability_condition_gate_20260624.json",
    "control_state_support_cap120": ROOT / "reports/latentfm_control_state_support_gate_20260624.json",
    "signed_neighborhood_cap120": ROOT / "reports/latentfm_signed_neighborhood_consistency_gate_20260624.json",
    "composite_safe_subset_cap120": ROOT / "reports/latentfm_composite_safe_subset_gate_20260624.json",
    "bootstrap_target_noise_cap120": ROOT / "reports/latentfm_bootstrap_target_noise_gate_20260624.json",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def condition_rows(group: str) -> list[dict[str, Any]]:
    anchor = {
        (str(r["dataset"]), str(r["condition"])): r
        for r in load_json(ANCHOR_MEANS)["groups"][group]["condition_metrics"]
    }
    cap = {
        (str(r["dataset"]), str(r["condition"])): r
        for r in load_json(CAP120_MEANS)["groups"][group]["condition_metrics"]
    }
    rows = []
    for key in sorted(set(anchor) & set(cap)):
        rows.append(
            {
                "dataset": key[0],
                "condition": key[1],
                "delta_pp": float(cap[key]["pearson_pert"] - anchor[key]["pearson_pert"]),
                "delta_mmd": float(cap[key]["test_mmd_clamped"] - anchor[key]["test_mmd_clamped"]),
            }
        )
    return rows


def bootstrap(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(SEED)
    arr = np.asarray(values, dtype=np.float64)
    means = []
    for _ in range(BOOT_N):
        idx = [rng.randrange(len(arr)) for _ in arr]
        means.append(float(np.mean(arr[idx])))
    means_arr = np.asarray(means, dtype=np.float64)
    return float(np.quantile(means_arr, 0.025)), float(np.quantile(means_arr, 0.975)), float(np.mean(means_arr < 0.0))


def summarize_applied(rows: list[dict[str, Any]], alphas: dict[tuple[str, str], float]) -> dict[str, float]:
    vals = []
    mmd = []
    by_ds: dict[str, list[float]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["condition"]))
        alpha = float(alphas.get(key, 0.0))
        val = alpha * float(row["delta_pp"])
        vals.append(val)
        mmd.append(alpha * float(row["delta_mmd"]))
        by_ds.setdefault(str(row["dataset"]), []).append(val)
    lo, hi, p_harm = bootstrap(vals)
    return {
        "n": len(rows),
        "mean_pp_delta": float(np.mean(vals)),
        "ci95_low": lo,
        "ci95_high": hi,
        "bootstrap_p_harm": p_harm,
        "dataset_min_pp_delta": float(min(sum(v) / len(v) for v in by_ds.values())),
        "mean_mmd_delta": float(np.mean(mmd)),
        "mean_alpha": float(np.mean(list(alphas.values()))) if alphas else 0.0,
    }


def oracle_ladder_for_group(group: str) -> list[dict[str, Any]]:
    rows = condition_rows(group)
    all_keys = {(r["dataset"], r["condition"]): 1.0 for r in rows}
    noop = {(r["dataset"], r["condition"]): 0.0 for r in rows}
    condition_pp = {(r["dataset"], r["condition"]): 1.0 if r["delta_pp"] > 0.0 else 0.0 for r in rows}
    condition_pp_mmd = {
        (r["dataset"], r["condition"]): 1.0 if r["delta_pp"] > 0.0 and r["delta_mmd"] <= 0.0 else 0.0
        for r in rows
    }
    by_ds: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_ds.setdefault(str(row["dataset"]), []).append(row)
    dataset_pp = {}
    for ds, ds_rows in by_ds.items():
        use_candidate = float(np.mean([r["delta_pp"] for r in ds_rows])) > 0.0
        for row in ds_rows:
            dataset_pp[(row["dataset"], row["condition"])] = 1.0 if use_candidate else 0.0
    strategies = [
        ("deployable_noop_anchor", "deployable", noop),
        ("all_cap120_candidate", "candidate", all_keys),
        ("forbidden_dataset_outcome_oracle", "forbidden_oracle", dataset_pp),
        ("forbidden_condition_pp_oracle", "forbidden_oracle", condition_pp),
        ("forbidden_condition_pp_mmd_oracle", "forbidden_oracle", condition_pp_mmd),
    ]
    out = []
    for name, tier, alphas in strategies:
        summary = summarize_applied(rows, alphas)
        summary.update({"name": name, "tier": tier, "group": group})
        out.append(summary)
    return out


def gate_summary_rows() -> list[dict[str, Any]]:
    rows = []
    for name, path in GATE_REPORTS.items():
        if not path.exists():
            continue
        payload = load_json(path)
        for result in payload.get("results", []):
            if result.get("control") != "main":
                continue
            group = result.get("group")
            if group not in GROUPS:
                continue
            if "cap120" in name and result.get("run") not in {None, "cap120"}:
                continue
            summary = dict(result["summary"])
            summary.update({"name": name, "tier": "train_only_gate", "group": group})
            rows.append(summary)
    return rows


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = {group: [r for r in rows if r["group"] == group] for group in GROUPS}
    decisions = []
    for group, group_rows in by_group.items():
        oracle = max((r for r in group_rows if r["tier"] == "forbidden_oracle"), key=lambda r: r["mean_pp_delta"])
        gates = [r for r in group_rows if r["tier"] == "train_only_gate"]
        safe_gates = [
            r
            for r in gates
            if r["mean_pp_delta"] >= 0.010
            and r["dataset_min_pp_delta"] >= -0.020
            and r["mean_mmd_delta"] <= 0.0005
            and r["bootstrap_p_harm"] <= 0.35
        ]
        best_gate = max(gates, key=lambda r: r["mean_pp_delta"]) if gates else None
        recovered = 0.0
        if oracle["mean_pp_delta"] > 1e-12 and safe_gates:
            recovered = max(r["mean_pp_delta"] for r in safe_gates) / oracle["mean_pp_delta"]
        decisions.append(
            {
                "group": group,
                "oracle_name": oracle["name"],
                "oracle_mean_pp_delta": oracle["mean_pp_delta"],
                "best_gate_name": None if best_gate is None else best_gate["name"],
                "best_gate_mean_pp_delta": None if best_gate is None else best_gate["mean_pp_delta"],
                "n_safe_gates": len(safe_gates),
                "safe_gate_recovered_oracle_fraction": recovered,
            }
        )
    has_safe_gate = any(d["n_safe_gates"] > 0 for d in decisions)
    status = "tracka_identifiability_ceiling_stop_model_search_no_gpu"
    if has_safe_gate:
        status = "tracka_identifiability_ceiling_partial_reopen_cpu_gate_needed"
    return {
        "status": status,
        "gpu_authorized": False,
        "group_decisions": decisions,
        "next_action": "Stop Track A GPU model-search; use ceiling audit in paper framing and only reopen with a genuinely new CPU gate." if not has_safe_gate else "Do not launch GPU directly; design a focused CPU gate for the safe signal.",
    }


def render_md(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# LatentFM Track A Identifiability Ceiling Audit",
        "",
        f"Status: `{decision['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only stop/continue audit.",
        "- Reads completed cap120/anchor internal condition means and completed gate JSON reports only.",
        "- Does not read canonical outcomes, canonical multi, Track C query, active logs, new GPU artifacts, or use GPU.",
        "",
        "## Decision",
        "",
        f"- GPU authorized: `{decision['gpu_authorized']}`",
        f"- next action: {decision['next_action']}",
        "",
        "## Oracle Recovery",
        "",
        "| group | oracle | oracle pp delta | best gate | best gate pp delta | safe gates | recovered oracle fraction |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for row in decision["group_decisions"]:
        best = "NA" if row["best_gate_name"] is None else f"`{row['best_gate_name']}`"
        best_val = "NA" if row["best_gate_mean_pp_delta"] is None else f"{row['best_gate_mean_pp_delta']:.6f}"
        lines.append(
            f"| `{row['group']}` | `{row['oracle_name']}` | {row['oracle_mean_pp_delta']:.6f} | {best} | {best_val} | {row['n_safe_gates']} | {row['safe_gate_recovered_oracle_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Ladder Rows",
            "",
            "| tier | name | group | n | mean pp delta | 95% CI | p_harm | dataset min | mean MMD delta | mean alpha |",
            "|---|---|---|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            f"| `{row['tier']}` | `{row['name']}` | `{row['group']}` | {row['n']} | {row['mean_pp_delta']:.6f} | [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] | {row['bootstrap_p_harm']:.3f} | {row['dataset_min_pp_delta']:.6f} | {row['mean_mmd_delta']:.6f} | {row['mean_alpha']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Forbidden outcome oracles quantify available headroom, while the completed train-only gates quantify how much of that headroom deployable signals recovered under strict worst-dataset and no-harm constraints.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    rows = []
    for group in GROUPS:
        rows.extend(oracle_ladder_for_group(group))
    rows.extend(gate_summary_rows())
    rows = sorted(rows, key=lambda r: (r["group"], r["tier"], r["name"]))
    payload = {
        "boundary": {
            "anchor_means": str(ANCHOR_MEANS),
            "cap120_means": str(CAP120_MEANS),
            "gate_reports": {k: str(v) for k, v in GATE_REPORTS.items()},
            "seed": SEED,
        },
        "rows": rows,
        "decision": decide(rows),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    print(OUT_MD)


if __name__ == "__main__":
    main()
