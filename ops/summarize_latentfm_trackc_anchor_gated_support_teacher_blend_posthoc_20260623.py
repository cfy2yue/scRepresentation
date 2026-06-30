#!/usr/bin/env python3
"""Summarize the full anchor-gated support-teacher blend posthoc gate."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
PASS_STATUS = "trackc_anchor_gated_support_teacher_blend_posthoc_gate_pass"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_for(payload: dict[str, Any], group: str) -> list[dict[str, Any]]:
    return ((payload.get("groups") or {}).get(group) or {}).get("condition_metrics") or []


def finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def fmt(value: Any) -> str:
    value = finite(value)
    return "NA" if value is None else f"{value:+.6f}"


def equal_ds_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = finite(row.get(key))
        if value is not None:
            by_ds[str(row.get("dataset"))].append(value)
    vals = [mean(v) for v in by_ds.values() if v]
    return mean(vals) if vals else None


def bootstrap_equal_ds(
    rows: list[dict[str, Any]],
    key: str,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = finite(row.get(key))
        if value is not None:
            by_ds[str(row.get("dataset"))].append(value)
    observed = equal_ds_mean(rows, key)
    if observed is None or not by_ds:
        return {"observed": None, "ci_low": None, "ci_high": None, "samples": []}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(int(n_boot)):
        ds_vals = []
        for vals in by_ds.values():
            arr = np.asarray(vals, dtype=np.float64)
            picked = rng.choice(arr, size=arr.size, replace=True)
            ds_vals.append(float(np.mean(picked)))
        samples.append(float(np.mean(ds_vals)))
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "observed": float(observed),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "p_harm_pp": float(np.mean(arr < -0.02)),
        "p_improve_pp": float(np.mean(arr > 0.02)),
        "p_harm_mmd": float(np.mean(arr > 0.005)),
        "p_improve_mmd": float(np.mean(arr < -0.005)),
    }


def max_abs_delta(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in keys:
        vals = [abs(v) for row in rows if (v := finite(row.get(key))) is not None]
        out[key] = max(vals) if vals else None
    return out


def safety_ok(payloads: list[dict[str, Any]]) -> bool:
    for payload in payloads:
        safety = payload.get("safety") or {}
        if safety.get("heldout_query_read") is not False:
            return False
        if safety.get("canonical_multi_selection") is not False:
            return False
    return True


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    support = load_json(args.support_json)
    canonical_single = load_json(args.canonical_test_single_json)
    canonical_family = load_json(args.canonical_family_gene_json)
    support_rows = rows_for(support, args.support_group)
    single_rows = rows_for(canonical_single, "test_single")
    family_rows = rows_for(canonical_family, "family_gene")

    support_pp = bootstrap_equal_ds(
        support_rows,
        "blend_delta_vs_anchor_pearson_pert",
        n_boot=int(args.n_boot),
        seed=42,
    )
    support_mmd = bootstrap_equal_ds(
        support_rows,
        "blend_delta_vs_anchor_test_mmd",
        n_boot=int(args.n_boot),
        seed=43,
    )
    support_mmd_biased = bootstrap_equal_ds(
        support_rows,
        "blend_delta_vs_anchor_test_mmd_biased",
        n_boot=int(args.n_boot),
        seed=44,
    )
    noop_keys = [
        "blend_delta_vs_anchor_test_mmd",
        "blend_delta_vs_anchor_test_mmd_biased",
        "blend_delta_vs_anchor_test_mmd_clamped",
        "blend_delta_vs_anchor_direct_pearson",
        "blend_delta_vs_anchor_pearson_ctrl",
        "blend_delta_vs_anchor_pearson_pert",
    ]
    single_noop = max_abs_delta(single_rows, noop_keys)
    family_noop = max_abs_delta(family_rows, noop_keys)

    reasons: list[str] = []
    if not safety_ok([support, canonical_single, canonical_family]):
        reasons.append("payload_safety_flags_not_clean")
    if not support_rows:
        reasons.append("missing_support_rows")
    if not single_rows:
        reasons.append("missing_canonical_test_single_rows")
    if not family_rows:
        reasons.append("missing_canonical_family_gene_rows")

    pp_obs = finite(support_pp.get("observed"))
    pp_p_harm = finite(support_pp.get("p_harm_pp"))
    mmd_obs = finite(support_mmd.get("observed"))
    mmd_p_harm = finite(support_mmd.get("p_harm_mmd"))
    mmd_b_obs = finite(support_mmd_biased.get("observed"))
    mmd_b_p_harm = finite(support_mmd_biased.get("p_harm_mmd"))
    if pp_obs is None or pp_obs < 0.02:
        reasons.append("support_pearson_pert_delta_below_0p02")
    if pp_p_harm is None or pp_p_harm > 0.10:
        reasons.append("support_pearson_pert_bootstrap_harm_p_above_0p10")
    if mmd_obs is None or mmd_obs > 0.005:
        reasons.append("support_unbiased_mmd_delta_above_0p005")
    if mmd_p_harm is None or mmd_p_harm > 0.10:
        reasons.append("support_unbiased_mmd_bootstrap_harm_p_above_0p10")
    if mmd_b_obs is None or mmd_b_obs > 0.005:
        reasons.append("support_biased_mmd_delta_above_0p005")
    if mmd_b_p_harm is None or mmd_b_p_harm > 0.10:
        reasons.append("support_biased_mmd_bootstrap_harm_p_above_0p10")

    noop_tol = float(args.noop_tol)
    for label, values in (("test_single", single_noop), ("family_gene", family_noop)):
        for key, value in values.items():
            if value is None or value > noop_tol:
                reasons.append(f"canonical_{label}_{key}_not_exact_noop")

    status = PASS_STATUS if not reasons else "trackc_anchor_gated_support_teacher_blend_posthoc_gate_fail"
    action = (
        "freeze_route_and_consider_one_time_heldout_query_eval"
        if status == PASS_STATUS
        else "fail_closed_do_not_run_query_or_claim_multi"
    )
    return {
        "status": status,
        "action": action,
        "reasons": reasons,
        "rules": [
            "support safe trainselect only; held-out query forbidden",
            "support equal-dataset mean pearson_pert delta >= +0.02",
            "support pearson_pert bootstrap p(delta < -0.02) <= 0.10",
            "support unbiased and biased MMD deltas <= +0.005",
            "support MMD bootstrap p(delta > +0.005) <= 0.10",
            f"canonical test_single/family_gene gate=0 max absolute metric delta <= {noop_tol:g}",
        ],
        "inputs": {
            "support_json": str(args.support_json),
            "canonical_test_single_json": str(args.canonical_test_single_json),
            "canonical_family_gene_json": str(args.canonical_family_gene_json),
        },
        "support": {
            "group": str(args.support_group),
            "n_rows": len(support_rows),
            "pearson_pert_delta": support_pp,
            "test_mmd_delta": support_mmd,
            "test_mmd_biased_delta": support_mmd_biased,
        },
        "canonical_noop": {
            "test_single_n_rows": len(single_rows),
            "family_gene_n_rows": len(family_rows),
            "test_single_max_abs_delta": single_noop,
            "family_gene_max_abs_delta": family_noop,
        },
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track C Anchor-Gated Support-Teacher Blend Posthoc Gate",
        "",
        f"Status: `{payload['status']}`",
        f"Action: `{payload['action']}`",
        "",
        "## Scope",
        "",
        "This report reads only the safe support-trainselect posthoc JSON and canonical single/family no-harm JSON.",
        "Held-out Track C query and canonical multi selection are not used.",
        "",
        "## Gate Summary",
        "",
        f"* support rows: `{payload['support']['n_rows']}`",
        f"* support pp delta: `{fmt(payload['support']['pearson_pert_delta'].get('observed'))}`",
        f"* support pp p_harm: `{fmt(payload['support']['pearson_pert_delta'].get('p_harm_pp'))}`",
        f"* support unbiased MMD delta: `{fmt(payload['support']['test_mmd_delta'].get('observed'))}`",
        f"* support unbiased MMD p_harm: `{fmt(payload['support']['test_mmd_delta'].get('p_harm_mmd'))}`",
        f"* support biased MMD delta: `{fmt(payload['support']['test_mmd_biased_delta'].get('observed'))}`",
        f"* support biased MMD p_harm: `{fmt(payload['support']['test_mmd_biased_delta'].get('p_harm_mmd'))}`",
        f"* canonical test_single rows: `{payload['canonical_noop']['test_single_n_rows']}`",
        f"* canonical family_gene rows: `{payload['canonical_noop']['family_gene_n_rows']}`",
        "",
        "## Reasons",
        "",
    ]
    if payload["reasons"]:
        lines.extend(f"* `{reason}`" for reason in payload["reasons"])
    else:
        lines.append("* none")
    lines.extend(["", "## Rules", ""])
    lines.extend(f"* {rule}" for rule in payload["rules"])
    lines.extend(["", "## Inputs", ""])
    lines.extend(f"* {k}: `{v}`" for k, v in payload["inputs"].items())
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--support-json", type=Path, required=True)
    ap.add_argument("--canonical-test-single-json", type=Path, required=True)
    ap.add_argument("--canonical-family-gene-json", type=Path, required=True)
    ap.add_argument("--support-group", type=str, default="support_val_multi")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--noop-tol", type=float, default=1e-8)
    args = ap.parse_args()

    payload = summarize(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
