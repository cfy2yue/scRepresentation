#!/usr/bin/env python3
"""CPU-only gate for anchor-relative risk-constrained update ideas.

This script deliberately uses only train/internal condition-mean artifacts.
It tests whether a simple anchor-relative update constraint has enough
train-only evidence to justify implementing a new training hook.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
MEAN_DIR = ROOT / "reports/latentfm_xverse_nuisance_condition_means_20260624"
ANCHOR_SPLIT = MEAN_DIR / "split_group_eval_anchor_internal_means_ode20.json"
CAP120_SPLIT = MEAN_DIR / "split_group_eval_cap120_internal_means_ode20.json"
OUT_JSON = ROOT / "reports/latentfm_xverse_risk_constrained_update_cpu_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_XVERSE_RISK_CONSTRAINED_UPDATE_CPU_GATE_20260624.md"

GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
BOOT_N = 1000
SEED = 42


@dataclass(frozen=True)
class Row:
    group: str
    dataset: str
    condition: str
    anchor_pred: np.ndarray
    cap_pred: np.ndarray
    gt: np.ndarray
    ctrl: np.ndarray
    pert: np.ndarray


@dataclass(frozen=True)
class Policy:
    name: str
    deployable: bool
    family: str
    fn: Callable[[Row, list[Row]], float]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def arr(row: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(row[key], dtype=np.float64)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    aa = a - float(np.mean(a))
    bb = b - float(np.mean(b))
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(aa, bb) / den)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / den)


def pp(pred: np.ndarray, row: Row) -> float:
    return corr(pred - row.ctrl, row.pert - row.ctrl)


def residual(pred: np.ndarray, row: Row) -> float:
    return float(np.linalg.norm(pred - row.gt) / math.sqrt(pred.size))


def load_rows() -> list[Row]:
    anchor = load_json(ANCHOR_SPLIT)
    cap120 = load_json(CAP120_SPLIT)
    rows: list[Row] = []
    for group in GROUPS:
        a_rows = {
            (str(r["dataset"]), str(r["condition"])): r
            for r in anchor["groups"][group]["condition_metrics"]
        }
        c_rows = {
            (str(r["dataset"]), str(r["condition"])): r
            for r in cap120["groups"][group]["condition_metrics"]
        }
        for dataset, condition in sorted(set(a_rows) & set(c_rows)):
            a = a_rows[(dataset, condition)]
            c = c_rows[(dataset, condition)]
            rows.append(
                Row(
                    group=group,
                    dataset=dataset,
                    condition=condition,
                    anchor_pred=arr(a, "pred_mean"),
                    cap_pred=arr(c, "pred_mean"),
                    gt=arr(a, "gt_mean"),
                    ctrl=arr(a, "ctrl_mean"),
                    pert=arr(a, "pert_mean"),
                )
            )
    return rows


def update(row: Row) -> np.ndarray:
    return row.cap_pred - row.anchor_pred


def make_policies(rows: list[Row]) -> list[Policy]:
    norms = np.asarray([np.linalg.norm(update(r)) for r in rows], dtype=np.float64)
    q25, q50, q75 = [float(np.quantile(norms, q)) for q in (0.25, 0.5, 0.75)]

    policies: list[Policy] = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        policies.append(
            Policy(
                name=f"global_alpha_{alpha:.2f}",
                deployable=True,
                family="global_shrink",
                fn=lambda _row, _train, a=alpha: a,
            )
        )

    for name, threshold in (("q25", q25), ("q50", q50), ("q75", q75)):
        policies.append(
            Policy(
                name=f"update_norm_clip_{name}",
                deployable=True,
                family="update_norm_clip",
                fn=lambda row, _train, t=threshold: float(min(1.0, t / max(np.linalg.norm(update(row)), 1e-12))),
            )
        )

    for cut in (0.0, 0.25, 0.5):
        policies.append(
            Policy(
                name=f"oracle_target_cos_ge_{cut:.2f}",
                deployable=False,
                family="target_alignment_oracle",
                fn=lambda row, _train, c=cut: 1.0 if cosine(update(row), row.pert - row.ctrl) >= c else 0.0,
            )
        )

    for margin in (0.0, 0.0005, 0.0010):
        policies.append(
            Policy(
                name=f"oracle_residual_no_worse_{margin:.4f}",
                deployable=False,
                family="residual_oracle",
                fn=lambda row, _train, m=margin: 1.0
                if residual(row.cap_pred, row) <= residual(row.anchor_pred, row) + m
                else 0.0,
            )
        )

    return policies


def apply_policy(row: Row, policy: Policy, train_rows: list[Row], *, invert: bool = False, shuffled_update: np.ndarray | None = None) -> tuple[float, float, float]:
    u = update(row) if shuffled_update is None else shuffled_update
    if invert:
        u = -u
    alpha = float(policy.fn(row, train_rows))
    alpha = max(-1.0, min(1.0, alpha))
    pred = row.anchor_pred + alpha * u
    return pp(pred, row), residual(pred, row), alpha


def summarize(rows: list[Row], policy: Policy, train_rows: list[Row]) -> dict[str, Any]:
    group_payload = []
    all_delta = []
    all_resid_delta = []
    all_alphas = []
    for group in GROUPS:
        gr = [r for r in rows if r.group == group]
        deltas = []
        resid_deltas = []
        by_ds: dict[str, list[float]] = {}
        for row in gr:
            p_anchor = pp(row.anchor_pred, row)
            r_anchor = residual(row.anchor_pred, row)
            p_policy, r_policy, alpha = apply_policy(row, policy, train_rows)
            delta = p_policy - p_anchor
            deltas.append(delta)
            resid_deltas.append(r_policy - r_anchor)
            by_ds.setdefault(row.dataset, []).append(delta)
            all_alphas.append(alpha)
        ds_delta = {ds: float(np.nanmean(vals)) for ds, vals in by_ds.items()}
        group_payload.append(
            {
                "group": group,
                "n": len(gr),
                "mean_pp_delta_vs_anchor": float(np.nanmean(deltas)),
                "mean_residual_delta_vs_anchor": float(np.nanmean(resid_deltas)),
                "dataset_min_pp_delta_vs_anchor": float(min(ds_delta.values())),
                "p_harm_condition": float(np.mean(np.asarray(deltas) < 0.0)),
                "mean_alpha": float(np.nanmean(all_alphas)) if all_alphas else float("nan"),
            }
        )
        all_delta.extend(deltas)
        all_resid_delta.extend(resid_deltas)
    return {
        "policy": policy.name,
        "family": policy.family,
        "deployable": policy.deployable,
        "groups": group_payload,
        "mean_pp_delta_vs_anchor": float(np.nanmean(all_delta)),
        "mean_residual_delta_vs_anchor": float(np.nanmean(all_resid_delta)),
    }


def lodo_select(rows: list[Row], policies: list[Policy], *, deployable_only: bool) -> dict[str, Any]:
    datasets = sorted({r.dataset for r in rows})
    heldout_rows = []
    chosen = []
    candidate_policies = [p for p in policies if p.deployable or not deployable_only]
    for ds in datasets:
        train_rows = [r for r in rows if r.dataset != ds]
        test_rows = [r for r in rows if r.dataset == ds]
        scored = []
        for policy in candidate_policies:
            s = summarize(train_rows, policy, train_rows)
            group_ok = all(
                g["dataset_min_pp_delta_vs_anchor"] >= -0.02
                and g["mean_residual_delta_vs_anchor"] <= 0.0005
                for g in s["groups"]
            )
            score = (
                min(g["mean_pp_delta_vs_anchor"] for g in s["groups"]),
                -max(g["p_harm_condition"] for g in s["groups"]),
                -abs(s["mean_residual_delta_vs_anchor"]),
            )
            scored.append((group_ok, score, policy))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        policy = scored[0][2]
        chosen.append({"heldout_dataset": ds, "policy": policy.name, "deployable": policy.deployable})
        for row in test_rows:
            p_anchor = pp(row.anchor_pred, row)
            r_anchor = residual(row.anchor_pred, row)
            p_policy, r_policy, alpha = apply_policy(row, policy, train_rows)
            heldout_rows.append(
                {
                    "group": row.group,
                    "dataset": row.dataset,
                    "condition": row.condition,
                    "pp_delta_vs_anchor": p_policy - p_anchor,
                    "residual_delta_vs_anchor": r_policy - r_anchor,
                    "alpha": alpha,
                }
            )
    return summarize_eval_rows(heldout_rows) | {"chosen": chosen}


def summarize_eval_rows(eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    group_payload = []
    all_delta = []
    for group in GROUPS:
        rows = [r for r in eval_rows if r["group"] == group]
        deltas = np.asarray([r["pp_delta_vs_anchor"] for r in rows], dtype=np.float64)
        residuals = np.asarray([r["residual_delta_vs_anchor"] for r in rows], dtype=np.float64)
        ds_delta: dict[str, list[float]] = {}
        for r in rows:
            ds_delta.setdefault(str(r["dataset"]), []).append(float(r["pp_delta_vs_anchor"]))
        group_payload.append(
            {
                "group": group,
                "n": len(rows),
                "mean_pp_delta_vs_anchor": float(np.nanmean(deltas)),
                "mean_residual_delta_vs_anchor": float(np.nanmean(residuals)),
                "dataset_min_pp_delta_vs_anchor": float(min(float(np.nanmean(v)) for v in ds_delta.values())),
                "p_harm_condition": float(np.mean(deltas < 0.0)),
                "bootstrap_p_harm_mean_delta": bootstrap_p_harm(deltas),
            }
        )
        all_delta.extend([float(x) for x in deltas])
    return {
        "groups": group_payload,
        "mean_pp_delta_vs_anchor": float(np.nanmean(all_delta)),
    }


def bootstrap_p_harm(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    rng = random.Random(SEED)
    harms = 0
    vals = [float(v) for v in values]
    for _ in range(BOOT_N):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        if float(np.nanmean(sample)) < 0.0:
            harms += 1
    return harms / BOOT_N


def shuffled_update_control(rows: list[Row], policies: list[Policy]) -> dict[str, Any]:
    rng = random.Random(SEED)
    deployable = [p for p in policies if p.deployable]
    policy = next(p for p in deployable if p.name == "global_alpha_1.00")
    updates = [update(r) for r in rows]
    shuffled = updates[:]
    rng.shuffle(shuffled)
    eval_rows = []
    for row, u in zip(rows, shuffled):
        p_anchor = pp(row.anchor_pred, row)
        r_anchor = residual(row.anchor_pred, row)
        p_policy, r_policy, alpha = apply_policy(row, policy, rows, shuffled_update=u)
        eval_rows.append(
            {
                "group": row.group,
                "dataset": row.dataset,
                "condition": row.condition,
                "pp_delta_vs_anchor": p_policy - p_anchor,
                "residual_delta_vs_anchor": r_policy - r_anchor,
                "alpha": alpha,
            }
        )
    return summarize_eval_rows(eval_rows)


def inverted_update_control(rows: list[Row]) -> dict[str, Any]:
    policy = Policy(
        name="global_alpha_1.00_inverted_update",
        deployable=True,
        family="inverted_control",
        fn=lambda _row, _train: 1.0,
    )
    eval_rows = []
    for row in rows:
        p_anchor = pp(row.anchor_pred, row)
        r_anchor = residual(row.anchor_pred, row)
        p_policy, r_policy, alpha = apply_policy(row, policy, rows, invert=True)
        eval_rows.append(
            {
                "group": row.group,
                "dataset": row.dataset,
                "condition": row.condition,
                "pp_delta_vs_anchor": p_policy - p_anchor,
                "residual_delta_vs_anchor": r_policy - r_anchor,
                "alpha": alpha,
            }
        )
    return summarize_eval_rows(eval_rows)


def passes_gate(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    for group in summary["groups"]:
        name = group["group"]
        if group["mean_pp_delta_vs_anchor"] < 0.01:
            reasons.append(f"{name}_mean_pp_delta_lt_0p01")
        if group["p_harm_condition"] > 0.20:
            reasons.append(f"{name}_condition_p_harm_gt_0p20")
        if group["bootstrap_p_harm_mean_delta"] > 0.20:
            reasons.append(f"{name}_bootstrap_p_harm_gt_0p20")
        if group["dataset_min_pp_delta_vs_anchor"] < -0.02:
            reasons.append(f"{name}_dataset_min_delta_lt_neg0p02")
        if group["mean_residual_delta_vs_anchor"] > 0.0:
            reasons.append(f"{name}_residual_proxy_worse_than_anchor")
    return not reasons, reasons


def main() -> int:
    rows = load_rows()
    policies = make_policies(rows)
    policy_summaries = [summarize(rows, p, rows) for p in policies]
    deployable_lodo = lodo_select(rows, policies, deployable_only=True)
    all_policy_lodo = lodo_select(rows, policies, deployable_only=False)
    shuffled = shuffled_update_control(rows, policies)
    inverted = inverted_update_control(rows)

    deployable_pass, deployable_reasons = passes_gate(deployable_lodo)
    all_pass, all_reasons = passes_gate(all_policy_lodo)
    control_reasons = []
    for control_name, control in (("shuffled_update", shuffled), ("inverted_update", inverted)):
        for group in control["groups"]:
            if group["mean_pp_delta_vs_anchor"] >= 0.003:
                control_reasons.append(f"{control_name}_{group['group']}_did_not_collapse")

    status = "risk_constrained_update_cpu_gate_pass_code_gate_next_no_gpu"
    action = "design_default_off_training_hook_and_resource_audit"
    reasons = []
    if not deployable_pass:
        reasons.extend(deployable_reasons)
    if control_reasons:
        reasons.extend(control_reasons)
    if not deployable_pass or control_reasons:
        status = "risk_constrained_update_cpu_gate_fail_no_gpu"
        action = "do_not_launch_risk_constrained_update_gpu"
    if all_pass and not deployable_pass:
        reasons.append("only_nondeployable_oracle_policy_passed_if_any")
    if not all_pass:
        reasons.append("even_oracle_policy_lodo_did_not_pass_strict_gate")

    payload = {
        "status": status,
        "gpu_authorization": "none",
        "action": action,
        "inputs": {
            "anchor_split": str(ANCHOR_SPLIT),
            "cap120_split": str(CAP120_SPLIT),
            "groups": list(GROUPS),
        },
        "boundary": {
            "train_internal_only": True,
            "canonical_test_read": False,
            "canonical_multi_read": False,
            "heldout_query_read": False,
            "active_log_read": False,
            "new_gpu_artifact_read": False,
        },
        "gate": {
            "mean_pp_delta_vs_anchor_min": 0.01,
            "condition_p_harm_max": 0.20,
            "bootstrap_p_harm_max": 0.20,
            "dataset_min_pp_delta_vs_anchor_min": -0.02,
            "mean_residual_delta_vs_anchor_max": 0.0,
            "controls_must_collapse_delta_lt": 0.003,
            "selection": "leave_one_dataset_out",
        },
        "decision_reasons": reasons,
        "deployable_lodo": deployable_lodo,
        "all_policy_lodo": all_policy_lodo,
        "controls": {
            "shuffled_update": shuffled,
            "inverted_update": inverted,
        },
        "policy_summaries": policy_summaries,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM xverse Risk-Constrained Update CPU Gate",
        "",
        f"Status: `{status}`",
        "GPU authorization: `none`",
        "",
        "## Boundary",
        "",
        "- Reads only train/internal condition-mean artifacts for xverse 8k anchor and cap120.",
        "- Does not read canonical test, canonical multi, held-out Track C query, active logs, or new GPU artifacts.",
        "- Tests whether anchor-relative update magnitude/direction can be constrained strongly enough to justify a new training hook.",
        "",
        "## Gate",
        "",
        "- Leave-one-dataset-out policy selection.",
        "- Both internal groups must satisfy mean pp delta vs anchor `>= +0.01`.",
        "- Condition-level and bootstrap p_harm must be `<= 0.20`.",
        "- Dataset-min pp delta must be `>= -0.02`.",
        "- Mean residual proxy must be no worse than anchor.",
        "- Shuffled-update and inverted-update controls must collapse below `+0.003` mean pp delta.",
        "",
        "## Deployable LODO Result",
        "",
        "| group | n | mean pp delta | p_harm | boot p_harm | dataset min | residual delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in deployable_lodo["groups"]:
        lines.append(
            f"| {group['group']} | {group['n']} | {group['mean_pp_delta_vs_anchor']:+.6f} | "
            f"{group['p_harm_condition']:.3f} | {group['bootstrap_p_harm_mean_delta']:.3f} | "
            f"{group['dataset_min_pp_delta_vs_anchor']:+.6f} | {group['mean_residual_delta_vs_anchor']:+.6f} |"
        )
    lines.extend([
        "",
        "## All-Policy LODO Result",
        "",
        "| group | n | mean pp delta | p_harm | boot p_harm | dataset min | residual delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for group in all_policy_lodo["groups"]:
        lines.append(
            f"| {group['group']} | {group['n']} | {group['mean_pp_delta_vs_anchor']:+.6f} | "
            f"{group['p_harm_condition']:.3f} | {group['bootstrap_p_harm_mean_delta']:.3f} | "
            f"{group['dataset_min_pp_delta_vs_anchor']:+.6f} | {group['mean_residual_delta_vs_anchor']:+.6f} |"
        )
    lines.extend(["", "## Controls", ""])
    for control_name, control in payload["controls"].items():
        lines.append(f"### {control_name}")
        lines.append("")
        lines.append("| group | mean pp delta | p_harm | dataset min |")
        lines.append("|---|---:|---:|---:|")
        for group in control["groups"]:
            lines.append(
                f"| {group['group']} | {group['mean_pp_delta_vs_anchor']:+.6f} | "
                f"{group['p_harm_condition']:.3f} | {group['dataset_min_pp_delta_vs_anchor']:+.6f} |"
            )
        lines.append("")
    lines.extend(["## Decision Reasons", ""])
    lines.extend([f"- `{r}`" for r in reasons] or ["- none"])
    lines.extend([
        "",
        "## Interpretation",
        "",
        "A pass would authorize only a default-off code/protocol gate, not direct GPU.",
        "A fail closes this specific anchor-relative risk-constrained update route until a materially new train-only signal appears.",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
        "",
    ])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorization": "none", "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
