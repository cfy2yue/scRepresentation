#!/usr/bin/env python3
"""Summarize the anchor-gated support-teacher residual CPU gate.

This script consumes condition-mean artifacts generated with
``--save-condition-means``.  It does not read held-out Track C query outputs and
does not use canonical multi groups for selection.

The evaluated mechanism is an offline, anchor-preserving calibrator:

    pred = anchor_pred + gate * alpha * (support_teacher_pred - anchor_pred)

where the support route is evaluated on the safe trainselect support-val rows
and canonical single/background rows use ``gate=0`` by design.  Passing this
gate authorizes a code/provenance gate for an implemented support-teacher
calibrator; it is not a formal multi-capability claim.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_RUN_ROOT = (
    ROOT
    / "runs/latentfm_trackc_anchor_gated_support_teacher_artifacts_20260623/"
    "xverse_support_film_retry1_condition_means_artifacts"
)
CPU_GATE_JSON = ROOT / "reports/latentfm_trackc_alternative_support_conditioning_cpu_gate_20260623.json"
OUT_JSON = ROOT / "reports/latentfm_trackc_anchor_gated_support_teacher_cpu_gate_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_ANCHOR_GATED_SUPPORT_TEACHER_CPU_GATE_20260623.md"
ALPHAS = (0.25, 0.50, 0.75, 1.00)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def pearson_np(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return None
    x = x[mask]
    y = y[mask]
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return None
    return float(np.dot(x, y) / denom)


def group_rows(payload: dict[str, Any], group: str) -> list[dict[str, Any]]:
    return ((payload.get("groups") or {}).get(group) or {}).get("condition_metrics") or []


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("dataset")), str(row.get("condition"))


def require_mean(row: dict[str, Any], key: str) -> np.ndarray:
    value = row.get(key)
    if value is None:
        raise ValueError(f"row missing {key}: {row_key(row)}")
    return np.asarray(value, dtype=np.float32)


def paired_rows(anchor_payload: dict[str, Any], candidate_payload: dict[str, Any], group: str) -> list[dict[str, Any]]:
    anchors = {row_key(row): row for row in group_rows(anchor_payload, group)}
    candidates = {row_key(row): row for row in group_rows(candidate_payload, group)}
    missing = sorted((set(anchors) ^ set(candidates)))
    if missing:
        raise ValueError(f"anchor/candidate condition mismatch for {group}: {len(missing)}")
    out = []
    for key in sorted(anchors):
        a = anchors[key]
        c = candidates[key]
        pred_anchor = require_mean(a, "pred_mean")
        pred_teacher = require_mean(c, "pred_mean")
        gt_mean = require_mean(a, "gt_mean")
        pert_mean = require_mean(a, "pert_mean")
        out.append(
            {
                "dataset": key[0],
                "condition": key[1],
                "anchor_pp": pearson_np(pred_anchor - pert_mean, gt_mean - pert_mean),
                "teacher_pp": pearson_np(pred_teacher - pert_mean, gt_mean - pert_mean),
                "pred_anchor": pred_anchor,
                "pred_teacher": pred_teacher,
                "gt_mean": gt_mean,
                "pert_mean": pert_mean,
            }
        )
    if not out:
        raise ValueError(f"no rows for group {group}")
    return out


def cpu_route_gap_by_dataset(path: Path) -> dict[str, float]:
    payload = load_json(path)
    status = payload.get("status")
    if status != "trackc_alternative_support_conditioning_cpu_gate_pass_authorize_one_capped_gpu_smoke":
        raise ValueError(f"unexpected CPU gate status: {status}")
    out = {}
    for row in ((payload.get("real") or {}).get("dataset_breakdown") or []):
        ds = str(row.get("dataset"))
        route = row.get("support_selected_route")
        target = row.get("candidate")
        if route is not None and target is not None:
            out[ds] = float(target) - float(route)
    return out


def score_blend(rows: list[dict[str, Any]], alpha: float, *, shuffle: bool = False, seed: int = 42) -> list[dict[str, Any]]:
    residuals = [row["pred_teacher"] - row["pred_anchor"] for row in rows]
    if shuffle:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(residuals))
        residuals = [residuals[int(i)] for i in order]
    scored = []
    for row, residual in zip(rows, residuals):
        pred = row["pred_anchor"] + float(alpha) * residual
        pp = pearson_np(pred - row["pert_mean"], row["gt_mean"] - row["pert_mean"])
        scored.append(
            {
                "dataset": row["dataset"],
                "condition": row["condition"],
                "anchor_pp": row["anchor_pp"],
                "teacher_pp": row["teacher_pp"],
                "blend_pp": pp,
                "delta_vs_anchor": None if pp is None or row["anchor_pp"] is None else pp - row["anchor_pp"],
            }
        )
    return scored


def dataset_means(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is not None and np.isfinite(float(value)):
            by_ds[str(row["dataset"])].append(float(value))
    return {ds: mean(vals) for ds, vals in by_ds.items() if vals}


def equal_dataset_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = list(dataset_means(rows, key).values())
    return mean(vals) if vals else None


def bootstrap_delta(rows: list[dict[str, Any]], candidate: str, baseline: str, *, n_boot: int, seed: int) -> dict[str, Any]:
    by_ds: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        c = row.get(candidate)
        b = row.get(baseline)
        if c is not None and b is not None and np.isfinite(float(c)) and np.isfinite(float(b)):
            by_ds[str(row["dataset"])].append((float(c), float(b)))
    ds_items = sorted((ds, vals) for ds, vals in by_ds.items() if vals)
    if not ds_items:
        return {"status": "missing", "delta_mean": None, "p_harm": None, "ci95": [None, None], "n_datasets": 0}
    ds_delta = {ds: mean(c - b for c, b in vals) for ds, vals in ds_items}
    delta_mean = mean(ds_delta.values())
    rng = np.random.default_rng(seed)
    names = [ds for ds, _ in ds_items]
    vals = []
    for _ in range(int(n_boot)):
        sample = rng.choice(names, size=len(names), replace=True)
        vals.append(mean(ds_delta[str(ds)] for ds in sample))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "status": "ok",
        "delta_mean": float(delta_mean),
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_improvement": float(np.mean(arr > 0.0)),
        "p_harm": float(np.mean(arr < 0.0)),
        "n_datasets": len(names),
        "n_conditions": sum(len(v) for _, v in ds_items),
        "dataset_deltas": ds_delta,
    }


def support_summary(rows: list[dict[str, Any]], route_gaps: dict[str, float], alpha: float, *, shuffle: bool = False) -> dict[str, Any]:
    scored = score_blend(rows, alpha, shuffle=shuffle)
    by_ds = dataset_means(scored, "delta_vs_anchor")
    summary = []
    for ds, delta in sorted(by_ds.items()):
        gap = route_gaps.get(ds)
        summary.append(
            {
                "dataset": ds,
                "mean_delta_pp": delta,
                "route_gap_pp": gap,
                "route_gap_closed_fraction": None if gap is None or abs(gap) <= 1e-12 else delta / gap,
                "n_conditions": sum(1 for row in scored if row["dataset"] == ds),
            }
        )
    return {
        "alpha": float(alpha),
        "shuffle": bool(shuffle),
        "scored_rows": scored,
        "paired": bootstrap_delta(scored, "blend_pp", "anchor_pp", n_boot=2000, seed=100 + int(alpha * 1000) + (17 if shuffle else 0)),
        "dataset_summary": summary,
    }


def find_dataset(summary: dict[str, Any], dataset: str) -> dict[str, Any]:
    for row in summary.get("dataset_summary") or []:
        if row.get("dataset") == dataset:
            return row
    return {}


def alpha_passes(summary: dict[str, Any]) -> bool:
    wessels = find_dataset(summary, "Wessels")
    norman = find_dataset(summary, "NormanWeissman2019_filtered")
    paired = summary.get("paired") or {}
    wessels_delta = wessels.get("mean_delta_pp")
    wessels_closure = wessels.get("route_gap_closed_fraction")
    norman_delta = norman.get("mean_delta_pp")
    return (
        float(wessels_delta if wessels_delta is not None else -999.0) >= 0.02
        and float(wessels_closure if wessels_closure is not None else -999.0) >= 0.05
        and float(norman_delta if norman_delta is not None else -999.0) >= -0.02
        and float(paired.get("p_harm") if paired.get("p_harm") is not None else 1.0) <= 0.20
    )


def canonical_noop_summary(rows: list[dict[str, Any]], group: str) -> dict[str, Any]:
    scored = score_blend(rows, 0.0)
    return {
        "group": group,
        "gate": "g_zero_anchor_preserving",
        "paired": bootstrap_delta(scored, "blend_pp", "anchor_pp", n_boot=2000, seed=300 + len(group)),
        "max_abs_delta_pp": max(abs(float(row["delta_vs_anchor"] or 0.0)) for row in scored),
        "n_conditions": len(scored),
    }


def candidate_harm_diagnostic(rows: list[dict[str, Any]], group: str, alpha: float) -> dict[str, Any]:
    scored = score_blend(rows, alpha)
    return {
        "group": group,
        "gate": "diagnostic_apply_support_residual_to_canonical",
        "paired": bootstrap_delta(scored, "blend_pp", "anchor_pp", n_boot=2000, seed=400 + len(group)),
        "n_conditions": len(scored),
    }


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    selected = payload.get("selected_support_alpha_summary") or {}
    if not selected:
        reasons.append("no_alpha_passed_support_gate")
    else:
        wessels = find_dataset(selected, "Wessels")
        norman = find_dataset(selected, "NormanWeissman2019_filtered")
        paired = selected.get("paired") or {}
        wessels_delta = wessels.get("mean_delta_pp")
        wessels_closure = wessels.get("route_gap_closed_fraction")
        norman_delta = norman.get("mean_delta_pp")
        if float(wessels_delta if wessels_delta is not None else -999.0) < 0.02:
            reasons.append("wessels_delta_below_0p02")
        if float(wessels_closure if wessels_closure is not None else -999.0) < 0.05:
            reasons.append("wessels_closure_below_0p05")
        if float(norman_delta if norman_delta is not None else -999.0) < -0.02:
            reasons.append("norman_material_loss")
        if float(paired.get("p_harm") if paired.get("p_harm") is not None else 1.0) > 0.20:
            reasons.append("support_pp_p_harm_above_0p20")

    shuffled = payload.get("selected_shuffled_summary") or {}
    if selected and shuffled:
        selected_w = float((find_dataset(selected, "Wessels") or {}).get("mean_delta_pp") or 0.0)
        shuffled_w = float((find_dataset(shuffled, "Wessels") or {}).get("mean_delta_pp") or 0.0)
        if shuffled_w >= 0.02 or shuffled_w >= 0.5 * selected_w:
            reasons.append("shuffled_support_control_did_not_collapse")

    for row in payload.get("canonical_noop") or []:
        paired = row.get("paired") or {}
        if float(paired.get("p_harm") if paired.get("p_harm") is not None else 1.0) > 0.35:
            reasons.append(f"{row.get('group')}_canonical_pp_harm")
        max_abs_delta = row.get("max_abs_delta_pp")
        if max_abs_delta is None or float(max_abs_delta) > 1e-8:
            reasons.append(f"{row.get('group')}_not_exact_noop")

    status = (
        "trackc_anchor_gated_support_teacher_cpu_gate_pass_code_gate_next"
        if not reasons
        else "trackc_anchor_gated_support_teacher_cpu_gate_fail_no_gpu"
    )
    return {
        "status": status,
        "gpu_authorization": "none",
        "next_authorization": "code_provenance_gate_only" if not reasons else "none",
        "reasons": reasons,
        "rules": [
            "small predeclared alpha grid; choose smallest alpha passing support gate",
            "support Wessels pp delta >= +0.02",
            "support Wessels route-gap closure >= +0.05",
            "support Norman pp delta >= -0.02",
            "support paired pp p_harm <= 0.20",
            "shuffled support residual must lose Wessels signal",
            "canonical test_single/family_gene must be exact anchor no-op when gate=0",
            "mean-vector CPU gate does not evaluate MMD; later code/gpu gate must include MMD hard-harm",
        ],
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track C Anchor-Gated Support-Teacher CPU Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"GPU authorization: `{payload['decision']['gpu_authorization']}`",
        f"Next authorization: `{payload['decision']['next_authorization']}`",
        "",
        "## Scope",
        "",
        "This gate reads condition-mean posthoc artifacts generated from the support-FiLM retry1 anchor/candidate checkpoints.",
        "It does not read held-out Track C query outputs and does not use canonical multi groups for selection.",
        "MMD is not evaluated in this mean-vector CPU gate.",
        "",
        "## Selected Support Alpha",
        "",
    ]
    selected = payload.get("selected_support_alpha_summary")
    if selected:
        lines += [
            f"- alpha: `{selected['alpha']}`",
            f"- support paired pp delta: `{fmt((selected.get('paired') or {}).get('delta_mean'))}`",
            f"- support pp p_harm: `{fmt((selected.get('paired') or {}).get('p_harm'))}`",
            "",
            "| dataset | n | mean delta pp | route gap | closure |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in selected.get("dataset_summary") or []:
            lines.append(
                f"| {row['dataset']} | {row['n_conditions']} | {fmt(row['mean_delta_pp'])} | "
                f"{fmt(row.get('route_gap_pp'))} | {fmt(row.get('route_gap_closed_fraction'))} |"
            )
    else:
        lines.append("- no alpha passed the support gate")
    lines += ["", "## Shuffled Support Control", ""]
    shuffled = payload.get("selected_shuffled_summary")
    if shuffled:
        for row in shuffled.get("dataset_summary") or []:
            lines.append(
                f"- {row['dataset']}: delta {fmt(row['mean_delta_pp'])}, closure {fmt(row.get('route_gap_closed_fraction'))}"
            )
    lines += ["", "## Canonical No-Op Check", ""]
    for row in payload.get("canonical_noop") or []:
        paired = row.get("paired") or {}
        lines.append(
            f"- `{row['group']}`: delta {fmt(paired.get('delta_mean'))}, p_harm {fmt(paired.get('p_harm'))}, "
            f"max_abs_delta {fmt(row.get('max_abs_delta_pp'))}"
        )
    lines += ["", "## Diagnostic If Support Residual Were Applied To Canonical", ""]
    for row in payload.get("canonical_apply_diagnostic") or []:
        paired = row.get("paired") or {}
        lines.append(
            f"- `{row['group']}`: delta {fmt(paired.get('delta_mean'))}, p_harm {fmt(paired.get('p_harm'))}"
        )
    lines += ["", "## Decision Reasons", ""]
    reasons = payload["decision"].get("reasons") or []
    lines.extend([f"- `{reason}`" for reason in reasons] if reasons else ["- none"])
    lines += ["", "## Rules", ""]
    lines.extend(f"- {rule}" for rule in payload["decision"].get("rules") or [])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--cpu-gate-json", type=Path, default=CPU_GATE_JSON)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    args = parser.parse_args()

    out_dir = args.run_root / "condition_means"
    paths = {
        "support_anchor": out_dir / "support_anchor_split_condition_means_ode20.json",
        "support_candidate": out_dir / "support_candidate_split_condition_means_ode20.json",
        "canonical_anchor_single": out_dir / "canonical_anchor_split_test_single_condition_means_ode20.json",
        "canonical_candidate_single": out_dir / "canonical_candidate_split_test_single_condition_means_ode20.json",
        "canonical_anchor_family": out_dir / "canonical_anchor_family_gene_condition_means_ode20.json",
        "canonical_candidate_family": out_dir / "canonical_candidate_family_gene_condition_means_ode20.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing condition-mean artifacts: {missing}")

    route_gaps = cpu_route_gap_by_dataset(args.cpu_gate_json)
    support_rows = paired_rows(load_json(paths["support_anchor"]), load_json(paths["support_candidate"]), "test_multi")
    canonical_single_rows = paired_rows(
        load_json(paths["canonical_anchor_single"]),
        load_json(paths["canonical_candidate_single"]),
        "test_single",
    )
    canonical_family_rows = paired_rows(
        load_json(paths["canonical_anchor_family"]),
        load_json(paths["canonical_candidate_family"]),
        "family_gene",
    )

    support_grid = [support_summary(support_rows, route_gaps, alpha) for alpha in ALPHAS]
    passing = [row for row in support_grid if alpha_passes(row)]
    selected = passing[0] if passing else None
    selected_alpha = float(selected["alpha"]) if selected else float(ALPHAS[-1])
    shuffled = support_summary(support_rows, route_gaps, selected_alpha, shuffle=True) if selected else None

    payload: dict[str, Any] = {
        "run_root": str(args.run_root),
        "inputs": {k: str(v) for k, v in paths.items()},
        "cpu_gate_json": str(args.cpu_gate_json),
        "heldout_query_used": False,
        "canonical_multi_selection_used": False,
        "mmd_evaluated": False,
        "alpha_grid": list(ALPHAS),
        "support_alpha_grid": support_grid,
        "selected_support_alpha_summary": selected,
        "selected_shuffled_summary": shuffled,
        "canonical_noop": [
            canonical_noop_summary(canonical_single_rows, "test_single"),
            canonical_noop_summary(canonical_family_rows, "family_gene"),
        ],
        "canonical_apply_diagnostic": [
            candidate_harm_diagnostic(canonical_single_rows, "test_single", selected_alpha),
            candidate_harm_diagnostic(canonical_family_rows, "family_gene", selected_alpha),
        ],
    }
    payload["decision"] = decide(payload)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "out_md": str(args.out_md)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
