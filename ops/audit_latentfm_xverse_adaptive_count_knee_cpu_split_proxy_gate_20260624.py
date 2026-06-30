#!/usr/bin/env python3
"""Adaptive count-knee CPU split/proxy gate for xverse Track A.

This gate does not launch training.  It uses existing train-only internal
posthoc metrics from cap30/cap120/full/type-balanced smokes to test whether a
deployable per-dataset cap policy can be selected by nested leave-one-dataset
evaluation.  The policy may only use train-side count/type features.

Canonical metrics are not used for selection; held-out query and Track C data
are not read.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_xverse_scaling_count_smokes_20260624"
SPLIT_DIR = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624"
BASE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
METADATA = ROOT / "dataset/latentfm_full/xverse/condition_metadata.json"
OUT_JSON = ROOT / "reports/latentfm_xverse_adaptive_count_knee_cpu_split_proxy_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_XVERSE_ADAPTIVE_COUNT_KNEE_CPU_SPLIT_PROXY_GATE_20260624.md"

ARM_RUNS = {
    "cap30": "xverse_scaling_cap30_all_3k_seed42",
    "cap120": "xverse_scaling_cap120_all_3k_seed42",
    "full": "xverse_scaling_full_trainonly_3k_seed42",
    "type_balanced": "xverse_scaling_type_balanced_cap120_3k_seed42",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_float(seed: int, key: str) -> float:
    raw = hashlib.sha256(f"{seed}\t{key}".encode("utf-8")).hexdigest()
    return int(raw[:12], 16) / float(16**12 - 1)


def perturbation_type(metadata: dict[str, Any], ds: str, cond: str) -> str:
    entry = ((metadata.get(ds) or {}).get(cond) or {})
    raw = str(entry.get("perturbation_type_raw", entry.get("perturbation_type", ""))).strip()
    if raw:
        return raw
    if "sciplex" in ds.lower():
        return "drug"
    return "unknown"


def feature_table(base_split: dict[str, Any], metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for ds, groups in sorted(base_split.items()):
        train = [str(c) for c in groups.get("train") or []]
        ptypes = Counter(perturbation_type(metadata, str(ds), cond) for cond in train)
        n = len(train)
        out[str(ds)] = {
            "train_count": n,
            "ptype_counts": dict(sorted(ptypes.items())),
            "drug_share": float(ptypes.get("drug", 0) / max(1, n)),
            "crispri_share": float(ptypes.get("CRISPRi", 0) / max(1, n)),
            "gene_share": float(sum(v for k, v in ptypes.items() if k != "drug") / max(1, n)),
            "is_jiang": str(ds).startswith("Jiang_"),
            "is_sciplex": str(ds).startswith("sciplex3_"),
        }
    return out


def load_group(run: str, filename: str, group: str) -> dict[str, Any]:
    path = RUN_ROOT / run / "posthoc_eval_internal" / filename
    payload = load_json(path)
    return payload["groups"][group]


def load_arm_metrics() -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for arm, run in ARM_RUNS.items():
        cross = load_group(run, "split_group_eval_candidate_internal_ode20.json", "internal_val_cross_background_seen_gene_proxy")
        family = load_group(run, "split_group_eval_candidate_internal_ode20.json", "internal_val_family_gene_proxy")
        ds_names = sorted(set(cross.get("per_ds_p_pert") or {}) & set(family.get("per_ds_p_pert") or {}))
        out[arm] = {}
        for ds in ds_names:
            out[arm][ds] = {
                "cross_pp": float(cross["per_ds_p_pert"][ds]),
                "cross_mmd": float(cross["per_ds_mmd"][ds]),
                "family_pp": float(family["per_ds_p_pert"][ds]),
                "family_mmd": float(family["per_ds_mmd"][ds]),
            }
    return out


class Spec:
    def __init__(self, name: str, fn: Callable[[str, dict[str, Any]], str], *, control: bool = False):
        self.name = name
        self.fn = fn
        self.control = control

    def arm(self, ds: str, feat: dict[str, Any]) -> str:
        return self.fn(ds, feat)


def specs() -> list[Spec]:
    out: list[Spec] = [
        Spec("always_cap30", lambda ds, f: "cap30"),
        Spec("always_cap120_equal_count_baseline", lambda ds, f: "cap120", control=True),
        Spec("always_full", lambda ds, f: "full"),
    ]
    for t in (30, 60, 120, 240, 480, 960):
        out.append(Spec(f"small_count_le{t}_full_else_cap120", lambda ds, f, t=t: "full" if f["train_count"] <= t else "cap120"))
        out.append(Spec(f"large_count_ge{t}_full_else_cap120", lambda ds, f, t=t: "full" if f["train_count"] >= t else "cap120"))
        out.append(Spec(f"small_count_le{t}_cap30_else_cap120", lambda ds, f, t=t: "cap30" if f["train_count"] <= t else "cap120"))
        out.append(Spec(f"large_count_ge{t}_cap30_else_cap120", lambda ds, f, t=t: "cap30" if f["train_count"] >= t else "cap120"))
    for s in (0.25, 0.50, 0.75):
        out.append(Spec(f"drug_share_ge{s:.2f}_full_else_cap120", lambda ds, f, s=s: "full" if f["drug_share"] >= s else "cap120"))
        out.append(Spec(f"drug_share_ge{s:.2f}_cap30_else_cap120", lambda ds, f, s=s: "cap30" if f["drug_share"] >= s else "cap120"))
        out.append(Spec(f"crispri_share_ge{s:.2f}_cap30_else_cap120", lambda ds, f, s=s: "cap30" if f["crispri_share"] >= s else "cap120"))
    out.extend(
        [
            Spec("jiang_full_else_cap120", lambda ds, f: "full" if f["is_jiang"] else "cap120"),
            Spec("jiang_cap30_else_cap120", lambda ds, f: "cap30" if f["is_jiang"] else "cap120"),
            Spec("sciplex_full_else_cap120", lambda ds, f: "full" if f["is_sciplex"] else "cap120"),
            Spec("sciplex_cap30_else_cap120", lambda ds, f: "cap30" if f["is_sciplex"] else "cap120"),
        ]
    )
    for seed in range(8):
        out.append(
            Spec(
                f"random_policy_seed{seed}",
                lambda ds, f, seed=seed: ("cap30", "cap120", "full")[int(stable_float(seed, ds) * 3.0) % 3],
                control=True,
            )
        )
    return out


def metric_delta(arm_metrics: dict[str, dict[str, dict[str, float]]], spec: Spec, ds: str, feat: dict[str, Any]) -> dict[str, float]:
    arm = spec.arm(ds, feat)
    base = arm_metrics["cap120"][ds]
    cand = arm_metrics[arm][ds]
    return {
        "arm": arm,
        "cross_pp_delta": cand["cross_pp"] - base["cross_pp"],
        "family_pp_delta": cand["family_pp"] - base["family_pp"],
        "family_mmd_delta": cand["family_mmd"] - base["family_mmd"],
    }


def train_score(rows: list[dict[str, float]]) -> float:
    cross = float(np.mean([r["cross_pp_delta"] for r in rows])) if rows else -999.0
    fam = float(np.mean([r["family_pp_delta"] for r in rows])) if rows else -999.0
    mmd = float(np.mean([r["family_mmd_delta"] for r in rows])) if rows else 999.0
    min_delta = min((r["cross_pp_delta"] for r in rows), default=-999.0)
    penalty = 0.0
    if fam < -0.002:
        penalty += 10.0 * abs(fam + 0.002)
    if mmd > 0.0:
        penalty += 10.0 * mmd
    if min_delta < -0.02:
        penalty += 2.0 * abs(min_delta + 0.02)
    return cross - penalty


def aggregate(rows: list[dict[str, float]], seed: int = 42) -> dict[str, Any]:
    if not rows:
        return {"status": "missing"}
    vals = {k: np.asarray([r[k] for r in rows], dtype=np.float64) for k in ("cross_pp_delta", "family_pp_delta", "family_mmd_delta")}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, len(rows), size=len(rows))
        boot.append(float(np.mean(vals["cross_pp_delta"][idx])))
    boot_a = np.asarray(boot, dtype=np.float64)
    return {
        "n_datasets": len(rows),
        "cross_pp_delta": float(np.mean(vals["cross_pp_delta"])),
        "cross_pp_ci95": [float(np.quantile(boot_a, 0.025)), float(np.quantile(boot_a, 0.975))],
        "cross_pp_p_harm": float(np.mean(boot_a < 0.0)),
        "family_pp_delta": float(np.mean(vals["family_pp_delta"])),
        "family_mmd_delta": float(np.mean(vals["family_mmd_delta"])),
        "min_dataset_cross_pp_delta": float(np.min(vals["cross_pp_delta"])),
    }


def nested_lodo(arm_metrics: dict[str, dict[str, dict[str, float]]], features: dict[str, dict[str, Any]], spec_list: list[Spec], *, controls: bool) -> dict[str, Any]:
    datasets = sorted(set(features) & set(arm_metrics["cap120"]))
    candidate_specs = [s for s in spec_list if bool(s.control) == controls]
    heldout_rows = []
    selected_counts: Counter[str] = Counter()
    for holdout in datasets:
        train_ds = [ds for ds in datasets if ds != holdout]
        best = None
        best_score = -1e9
        for spec in candidate_specs:
            rows = [metric_delta(arm_metrics, spec, ds, features[ds]) for ds in train_ds]
            score = train_score(rows)
            if score > best_score:
                best_score = score
                best = spec
        assert best is not None
        row = metric_delta(arm_metrics, best, holdout, features[holdout])
        row["dataset"] = holdout
        row["selected_spec"] = best.name
        row["selection_score"] = float(best_score)
        heldout_rows.append(row)
        selected_counts[best.name] += 1
    return {
        "rows": heldout_rows,
        "summary": aggregate(heldout_rows, seed=123 if not controls else 321),
        "selected_spec_counts": dict(sorted(selected_counts.items())),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def main() -> None:
    base_split = load_json(BASE_SPLIT)
    metadata = load_json(METADATA)
    features = feature_table(base_split, metadata)
    arm_metrics = load_arm_metrics()
    spec_list = specs()
    main_lodo = nested_lodo(arm_metrics, features, spec_list, controls=False)
    control_lodo = nested_lodo(arm_metrics, features, spec_list, controls=True)
    s = main_lodo["summary"]
    c = control_lodo["summary"]

    reasons = []
    if s["cross_pp_delta"] < 0.003:
        reasons.append("lodo_cross_pp_delta_below_cap120_plus_0p003")
    if s["family_pp_delta"] < -0.002:
        reasons.append("lodo_family_pp_delta_below_minus_0p002")
    if s["family_mmd_delta"] > 0.0:
        reasons.append("lodo_family_mmd_delta_above_cap120")
    if s["min_dataset_cross_pp_delta"] < -0.02:
        reasons.append("lodo_min_dataset_cross_pp_delta_below_minus_0p02")
    if c["cross_pp_delta"] >= 0.003:
        reasons.append("random_or_equal_count_control_does_not_collapse")
    if s["cross_pp_delta"] - c["cross_pp_delta"] < 0.003:
        reasons.append("main_policy_not_separated_from_controls_by_0p003")

    status = "adaptive_count_knee_cpu_split_proxy_gate_fail_no_gpu" if reasons else "adaptive_count_knee_cpu_split_proxy_gate_pass_gpu_smoke_candidate"
    result = {
        "status": status,
        "gpu_authorization": "none" if reasons else "one_capped_tracka_smoke_after_resource_audit_and_launcher_review",
        "decision_reasons": reasons,
        "boundary": {
            "cpu_only_existing_internal_posthoc": True,
            "canonical_selection_used": False,
            "heldout_query_read": False,
            "active_log_read": False,
            "gpu_launch": False,
            "proxy_limitation": "mixed per-dataset metrics from existing cap30/cap120/full trained models; does not prove a new jointly trained adaptive split will behave identically",
        },
        "main_lodo": main_lodo,
        "control_lodo": control_lodo,
        "feature_table": features,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM xverse Adaptive Count-Knee CPU Split Proxy Gate",
        "",
        f"Status: `{status}`",
        f"GPU authorization: `{result['gpu_authorization']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only nested leave-one-dataset proxy using existing internal posthoc metrics from cap30/cap120/full/type-balanced smokes.",
        "- Candidate policies use only train-side count/type features.",
        "- Canonical metrics, held-out query, active logs, and new GPU artifacts are not read.",
        "- This is a proxy for split-policy usefulness; it is not a claim about a jointly trained adaptive model.",
        "",
        "## LODO Summary",
        "",
        "| policy set | n datasets | cross pp delta vs cap120 | cross CI95 | family pp delta | family MMD delta | min dataset cross |",
        "|---|---:|---:|---|---:|---:|---:|",
        f"| main deployable specs | {s['n_datasets']} | {fmt(s['cross_pp_delta'])} | [{fmt(s['cross_pp_ci95'][0])}, {fmt(s['cross_pp_ci95'][1])}] | {fmt(s['family_pp_delta'])} | {fmt(s['family_mmd_delta'])} | {fmt(s['min_dataset_cross_pp_delta'])} |",
        f"| random/equal-count controls | {c['n_datasets']} | {fmt(c['cross_pp_delta'])} | [{fmt(c['cross_pp_ci95'][0])}, {fmt(c['cross_pp_ci95'][1])}] | {fmt(c['family_pp_delta'])} | {fmt(c['family_mmd_delta'])} | {fmt(c['min_dataset_cross_pp_delta'])} |",
        "",
        "## Selected Spec Counts",
        "",
        "| spec | held-out selections |",
        "|---|---:|",
    ]
    for name, n in main_lodo["selected_spec_counts"].items():
        lines.append(f"| `{name}` | {n} |")
    lines.extend([
        "",
        "## Decision Reasons",
        "",
    ])
    lines.extend([f"- `{reason}`" for reason in reasons] or ["- `none`"])
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The adaptive count-knee idea is not supported by this nested proxy gate unless the main policy improves cross-background performance over cap120 while preserving family pp/MMD and separating from controls. A failure means no GPU should be spent on an adaptive split until a materially stronger train-only proxy is introduced.",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
