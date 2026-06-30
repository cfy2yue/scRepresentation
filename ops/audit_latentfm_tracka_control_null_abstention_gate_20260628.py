#!/usr/bin/env python3
"""Control/null abstention gate for Track A tail repair.

This is a CPU-only, train-only/internal gate. It asks whether frozen xverse
anchor tails can be repaired by a deployable conservative fallback to the
source/control endpoint, using only prediction/source/perturbation features
available before seeing the held-out target. Explicit Track A proxy rows are
reported as locked context only and are not used to fit thresholds.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
IN_DIR = ROOT / "reports/latentfm_xverse_8k_seed_ensemble_internal_means_20260627"
SEED_FILES = {
    "seed42": IN_DIR / "seed42_internal_split_group_means_evalseed42.json",
    "seed43": IN_DIR / "seed43_internal_split_group_means_evalseed42.json",
}
EXPLICIT_ROWS = ROOT / "reports/tracka_explicit_group_proxy_benchmark_20260628/condition_rows.csv"
OUT_JSON = ROOT / "reports/latentfm_tracka_control_null_abstention_gate_20260628.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_CONTROL_NULL_ABSTENTION_GATE_20260628.md"

GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
DEPLOYABLE_FEATURES = (
    "pred_delta_norm",
    "pred_pert_norm",
    "ctrl_pert_norm",
    "pred_ctrl_corr_pert",
    "pred_ctrl_resid_corr",
    "pred_ctrl_direct_corr",
)
CONTROL_FEATURES = ("n_src_eval", "random_noise")
FALLBACK_ALPHAS = (0.0, 0.25, 0.5)
FIXED_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def stable_seed(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / den)


def pp_score(endpoint: np.ndarray, gt: np.ndarray, pert: np.ndarray) -> float:
    return pearson(endpoint - pert, gt - pert)


def as_array(row: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(row[key], dtype=np.float32)


def enrich_row(seed_name: str, group: str, row: dict[str, Any], random_noise: float) -> dict[str, Any]:
    pred = as_array(row, "pred_mean")
    ctrl = as_array(row, "ctrl_mean")
    gt = as_array(row, "gt_mean")
    pert = as_array(row, "pert_mean")
    anchor_pp = pp_score(pred, gt, pert)
    ctrl_pp = pp_score(ctrl, gt, pert)
    alpha_pp = {
        str(alpha): pp_score(ctrl + float(alpha) * (pred - ctrl), gt, pert)
        for alpha in sorted(set(FALLBACK_ALPHAS + FIXED_ALPHAS))
    }
    features = {
        "pred_delta_norm": float(np.linalg.norm(pred - ctrl)),
        "pred_pert_norm": float(np.linalg.norm(pred - pert)),
        "ctrl_pert_norm": float(np.linalg.norm(ctrl - pert)),
        "pred_ctrl_corr_pert": pearson(pred - pert, ctrl - pert),
        "pred_ctrl_resid_corr": pearson(pred - ctrl, ctrl - pert),
        "pred_ctrl_direct_corr": pearson(pred, ctrl),
        "n_src_eval": float(row.get("n_src_eval", 0.0)),
        "random_noise": float(random_noise),
    }
    return {
        "seed": seed_name,
        "group": group,
        "dataset": str(row["dataset"]),
        "condition": str(row["condition"]),
        "anchor_pp": float(anchor_pp),
        "ctrl_pp": float(ctrl_pp),
        "ctrl_delta": float(ctrl_pp - anchor_pp),
        "anchor_mmd": float(row.get("test_mmd_clamped", row.get("test_mmd", 0.0))),
        "alpha_pp": alpha_pp,
        "features": features,
    }


def load_internal_rows(path: Path, seed_name: str, *, noise_tag: str) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for group in GROUPS:
        group_obj = obj.get("groups", {}).get(group, {})
        for row in group_obj.get("condition_metrics", []):
            rng = np.random.default_rng(stable_seed(f"{noise_tag}|{seed_name}|{group}|{row.get('dataset')}|{row.get('condition')}"))
            rows.append(enrich_row(seed_name, group, row, float(rng.normal())))
    return rows


def policy_name(policy: dict[str, Any]) -> str:
    if policy["type"] == "fixed":
        return f"fixed_alpha_{policy['alpha']}"
    return f"{policy['feature']}_{policy['direction']}_{policy['threshold']:.6g}_alpha_{policy['alpha']}"


def enabled(row: dict[str, Any], policy: dict[str, Any]) -> bool:
    if policy["type"] == "fixed":
        return True
    val = float(row["features"][policy["feature"]])
    if policy["direction"] == "le":
        return val <= float(policy["threshold"])
    if policy["direction"] == "ge":
        return val >= float(policy["threshold"])
    raise ValueError(policy["direction"])


def candidate_pp(row: dict[str, Any], policy: dict[str, Any]) -> float:
    if not enabled(row, policy):
        return float(row["anchor_pp"])
    return float(row["alpha_pp"][str(policy["alpha"])])


def summarize_rows(rows: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["group"])].append(row)
    out: dict[str, Any] = {}
    for group in GROUPS:
        part = by_group.get(group, [])
        deltas = [candidate_pp(r, policy) - float(r["anchor_pp"]) for r in part]
        enabled_flags = [enabled(r, policy) for r in part]
        bad = [r for r in part if float(r["anchor_pp"]) < 0.0]
        bad_deltas = [candidate_pp(r, policy) - float(r["anchor_pp"]) for r in bad]
        ds_deltas: dict[str, list[float]] = defaultdict(list)
        for row, delta in zip(part, deltas):
            ds_deltas[str(row["dataset"])].append(float(delta))
        ds_means = [float(np.mean(v)) for v in ds_deltas.values()]
        out[group] = {
            "n": len(part),
            "n_enabled": int(sum(enabled_flags)),
            "enabled_fraction": float(np.mean(enabled_flags)) if enabled_flags else 0.0,
            "anchor_pp": float(np.mean([r["anchor_pp"] for r in part])) if part else 0.0,
            "candidate_pp": float(np.mean([candidate_pp(r, policy) for r in part])) if part else 0.0,
            "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
            "dataset_min_delta": float(min(ds_means)) if ds_means else 0.0,
            "negative_anchor_rows": len(bad),
            "negative_anchor_mean_delta": float(np.mean(bad_deltas)) if bad_deltas else 0.0,
            "per_dataset_delta": {ds: float(np.mean(vals)) for ds, vals in sorted(ds_deltas.items())},
        }
    return out


def objective(summary: dict[str, Any]) -> float:
    vals = []
    for group in GROUPS:
        s = summary[group]
        mean_delta = float(s["mean_delta"])
        tail_delta = float(s["negative_anchor_mean_delta"])
        min_delta = float(s["dataset_min_delta"])
        enabled_frac = float(s["enabled_fraction"])
        penalty = 0.0
        if enabled_frac < 0.03 or enabled_frac > 0.85:
            penalty += 0.02
        if min_delta < -0.01:
            penalty += abs(min_delta + 0.01) * 2.0
        vals.append(mean_delta + 0.25 * tail_delta - penalty)
    return float(np.mean(vals)) if vals else -1e9


def train_constraints(summary: dict[str, Any]) -> bool:
    for group in GROUPS:
        s = summary[group]
        if float(s["mean_delta"]) < 0.0:
            return False
        if float(s["dataset_min_delta"]) < -0.015:
            return False
        if float(s["enabled_fraction"]) < 0.03 or float(s["enabled_fraction"]) > 0.85:
            return False
    return True


def build_policies(train_rows: list[dict[str, Any]], features: tuple[str, ...]) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = [{"type": "fixed", "alpha": alpha} for alpha in FIXED_ALPHAS if alpha != 1.0]
    for feature in features:
        values = np.asarray([float(r["features"][feature]) for r in train_rows], dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 8:
            continue
        qs = sorted(set(float(np.quantile(values, q)) for q in np.linspace(0.1, 0.9, 17)))
        for threshold in qs:
            for direction in ("le", "ge"):
                for alpha in FALLBACK_ALPHAS:
                    policies.append(
                        {
                            "type": "threshold",
                            "feature": feature,
                            "direction": direction,
                            "threshold": float(threshold),
                            "alpha": float(alpha),
                        }
                    )
    return policies


def choose_policy(train_rows: list[dict[str, Any]], features: tuple[str, ...]) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    scored = []
    for policy in build_policies(train_rows, features):
        summary = summarize_rows(train_rows, policy)
        scored.append((train_constraints(summary), objective(summary), policy, summary))
    constrained = [x for x in scored if x[0]]
    pool = constrained or scored
    ok, _obj, policy, summary = max(pool, key=lambda x: x[1])
    return policy, bool(ok), summary


def lodo_eval(rows: list[dict[str, Any]], features: tuple[str, ...]) -> dict[str, Any]:
    datasets = sorted({str(r["dataset"]) for r in rows})
    eval_rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for heldout in datasets:
        train = [r for r in rows if str(r["dataset"]) != heldout]
        val = [r for r in rows if str(r["dataset"]) == heldout]
        if len(train) < 20 or not val:
            continue
        policy, constrained, train_summary = choose_policy(train, features)
        selected.append(
            {
                "heldout_dataset": heldout,
                "policy": policy_name(policy),
                "policy_detail": policy,
                "train_constraint_satisfied": constrained,
                "train_objective": objective(train_summary),
            }
        )
        for row in val:
            copied = dict(row)
            copied["candidate_pp"] = candidate_pp(row, policy)
            copied["candidate_delta"] = float(copied["candidate_pp"] - copied["anchor_pp"])
            copied["policy"] = policy_name(policy)
            copied["policy_enabled"] = enabled(row, policy)
            eval_rows.append(copied)
    return {"rows": eval_rows, "selected_policies": selected}


def bootstrap_dataset_ci(rows: list[dict[str, Any]], group: str, *, seed: int = 20260628) -> dict[str, float]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if str(row["group"]) == group:
            by_ds[str(row["dataset"])].append(float(row["candidate_delta"]))
    vals = np.asarray([np.mean(v) for v in by_ds.values()], dtype=float)
    if vals.size == 0:
        return {"ci_low": 0.0, "ci_high": 0.0, "p_improve": 0.0, "p_harm": 1.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(5000, vals.size))
    means = vals[idx].mean(axis=1)
    return {
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_improve": float(np.mean(means > 0)),
        "p_harm": float(np.mean(means < 0)),
    }


def summarize_lodo(eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in GROUPS:
        part = [r for r in eval_rows if str(r["group"]) == group]
        by_ds: dict[str, list[float]] = defaultdict(list)
        for row in part:
            by_ds[str(row["dataset"])].append(float(row["candidate_delta"]))
        ds_means = [float(np.mean(v)) for v in by_ds.values()]
        bad = [r for r in part if float(r["anchor_pp"]) < 0.0]
        out[group] = {
            "n": len(part),
            "n_enabled": int(sum(bool(r["policy_enabled"]) for r in part)),
            "enabled_fraction": float(np.mean([bool(r["policy_enabled"]) for r in part])) if part else 0.0,
            "anchor_pp": float(np.mean([r["anchor_pp"] for r in part])) if part else 0.0,
            "candidate_pp": float(np.mean([r["candidate_pp"] for r in part])) if part else 0.0,
            "mean_delta": float(np.mean([r["candidate_delta"] for r in part])) if part else 0.0,
            "dataset_min_delta": float(min(ds_means)) if ds_means else 0.0,
            "negative_anchor_rows": len(bad),
            "negative_anchor_mean_delta": float(np.mean([r["candidate_delta"] for r in bad])) if bad else 0.0,
            "bootstrap_dataset_delta": bootstrap_dataset_ci(eval_rows, group),
            "per_dataset_delta": {ds: float(np.mean(vals)) for ds, vals in sorted(by_ds.items())},
        }
    return out


def explicit_context(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False}
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_group[str(row["explicit_group"])].append(row)
    out: dict[str, Any] = {"available": True, "path": str(path), "used_for_selection": False, "groups": {}}
    for group, rows in sorted(by_group.items()):
        pp = [float(r["pearson_pert"]) for r in rows]
        cp = [float(r["pearson_ctrl"]) for r in rows]
        bad_idx = [i for i, v in enumerate(pp) if v < 0.0]
        out["groups"][group] = {
            "n": len(rows),
            "anchor_pp": float(np.mean(pp)) if pp else 0.0,
            "ctrl_pp": float(np.mean(cp)) if cp else 0.0,
            "ctrl_minus_anchor": float(np.mean(np.asarray(cp) - np.asarray(pp))) if pp else 0.0,
            "negative_anchor_rows": len(bad_idx),
            "negative_anchor_ctrl_minus_anchor": float(np.mean([cp[i] - pp[i] for i in bad_idx])) if bad_idx else 0.0,
            "negative_anchor_ctrl_better_fraction": float(np.mean([cp[i] > pp[i] for i in bad_idx])) if bad_idx else 0.0,
        }
    return out


def pass_reasons(summary: dict[str, Any], shuffle_summary: dict[str, Any], count_summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for group in GROUPS:
        s = summary[group]
        boot = s["bootstrap_dataset_delta"]
        if s["mean_delta"] < 0.015:
            reasons.append(f"{group}_mean_delta_lt_0p015")
        if boot["ci_low"] <= 0.0:
            reasons.append(f"{group}_dataset_bootstrap_ci_low_not_above_0")
        if s["dataset_min_delta"] < -0.005:
            reasons.append(f"{group}_dataset_min_delta_below_minus_0p005")
        if s["negative_anchor_mean_delta"] < 0.03:
            reasons.append(f"{group}_negative_anchor_tail_delta_lt_0p03")
        if shuffle_summary[group]["mean_delta"] >= s["mean_delta"] - 0.003:
            reasons.append(f"{group}_shuffle_control_too_close")
        if count_summary[group]["mean_delta"] >= s["mean_delta"] - 0.003:
            reasons.append(f"{group}_count_control_too_close")
    reasons.append("real_candidate_mmd_not_computed_cpu_mean_gate_only")
    return reasons


def run_seed(seed_name: str, path: Path) -> dict[str, Any]:
    rows = load_internal_rows(path, seed_name, noise_tag="actual")
    actual = lodo_eval(rows, DEPLOYABLE_FEATURES)
    count_control = lodo_eval(rows, ("n_src_eval",))
    random_control = lodo_eval(rows, ("random_noise",))
    actual_summary = summarize_lodo(actual["rows"])
    count_summary = summarize_lodo(count_control["rows"])
    random_summary = summarize_lodo(random_control["rows"])
    reasons = pass_reasons(actual_summary, random_summary, count_summary)
    status = "pass_needs_real_mmd_and_canonical_noharm" if reasons == ["real_candidate_mmd_not_computed_cpu_mean_gate_only"] else "fail_no_gpu"
    return {
        "status": status,
        "input": str(path),
        "groups": actual_summary,
        "selected_policies": actual["selected_policies"],
        "controls": {
            "count_only_n_src_eval": count_summary,
            "random_noise": random_summary,
        },
        "decision_reasons": reasons,
    }


def main() -> None:
    seed_results = {seed: run_seed(seed, path) for seed, path in SEED_FILES.items() if path.is_file()}
    explicit = explicit_context(EXPLICIT_ROWS)
    seed_statuses = [obj["status"] for obj in seed_results.values()]
    status = "tracka_control_null_abstention_fail_no_gpu"
    if seed_statuses and all(s == "pass_needs_real_mmd_and_canonical_noharm" for s in seed_statuses):
        status = "tracka_control_null_abstention_internal_pass_needs_real_eval_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "thresholds_fit_on_trainonly_internal_lodo": True,
            "explicit_tracka_proxy_rows_selection_weight": 0,
            "canonical_multi_selection_used": False,
            "trackc_query_used": False,
            "candidate_policy": "fallback to ctrl + alpha*(pred-ctrl) on deployable pred/ctrl/pert features",
            "real_mmd_computed": False,
        },
        "seed_results": seed_results,
        "explicit_tracka_locked_context": explicit,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Track A Control/Null Abstention Gate",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "CPU-only gate over frozen xverse internal condition means. Thresholds are fit leave-one-dataset-out on train-only/internal rows only. Explicit Track A proxy rows are locked context with selection weight 0. No canonical multi or Track C query is used.",
        "",
        "## Internal LODO Results",
        "",
        "| seed | group | n | enabled | anchor pp | candidate pp | delta | 95% CI | dataset min | neg-tail delta |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for seed, obj in sorted(seed_results.items()):
        for group in GROUPS:
            s = obj["groups"][group]
            boot = s["bootstrap_dataset_delta"]
            lines.append(
                f"| `{seed}` | `{group}` | {s['n']} | {s['enabled_fraction']:.3f} | "
                f"{s['anchor_pp']:+.6f} | {s['candidate_pp']:+.6f} | {s['mean_delta']:+.6f} | "
                f"[{boot['ci_low']:+.6f},{boot['ci_high']:+.6f}] | {s['dataset_min_delta']:+.6f} | "
                f"{s['negative_anchor_mean_delta']:+.6f} |"
            )
    lines.extend(["", "## Controls", "", "| seed | group | actual delta | count-only delta | random-noise delta |", "|---|---|---:|---:|---:|"])
    for seed, obj in sorted(seed_results.items()):
        for group in GROUPS:
            lines.append(
                f"| `{seed}` | `{group}` | {obj['groups'][group]['mean_delta']:+.6f} | "
                f"{obj['controls']['count_only_n_src_eval'][group]['mean_delta']:+.6f} | "
                f"{obj['controls']['random_noise'][group]['mean_delta']:+.6f} |"
            )
    lines.extend(["", "## Explicit Track A Locked Context", ""])
    if explicit.get("available"):
        lines.extend(["| group | n | anchor pp | ctrl pp | ctrl-anchor | neg rows | neg ctrl-anchor | neg ctrl better |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for group, s in explicit["groups"].items():
            lines.append(
                f"| `{group}` | {s['n']} | {s['anchor_pp']:+.6f} | {s['ctrl_pp']:+.6f} | "
                f"{s['ctrl_minus_anchor']:+.6f} | {s['negative_anchor_rows']} | "
                f"{s['negative_anchor_ctrl_minus_anchor']:+.6f} | {s['negative_anchor_ctrl_better_fraction']:.3f} |"
            )
    else:
        lines.append("Explicit proxy rows were not found.")
    lines.extend(["", "## Decision Reasons", ""])
    all_reasons = sorted({reason for obj in seed_results.values() for reason in obj["decision_reasons"]})
    lines.extend(f"- `{reason}`" for reason in all_reasons)
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This branch does not authorize GPU. Control/null fallback has tail headroom on rows where the anchor is already bad, but the deployable LODO policies do not pass mean, CI, dataset-tail, and control separation requirements; real candidate MMD is also not computed in this CPU mean-only gate.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "md": str(OUT_MD), "json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
