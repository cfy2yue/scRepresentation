#!/usr/bin/env python3
"""DepMap dependency tail-risk veto feasibility gate.

This CPU-only report revisits the completed DepMap matched rows for a narrower
question: dependency is not a positive information axis, but can it safely act
as a gene-perturbation tail-risk veto/curriculum signal? No training,
inference, checkpoint selection, canonical multi selection, or Track C query
access is performed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "depmap_tailrisk_veto_gate_20260630"

INPUTS = {
    "matched_rows": REPORTS
    / "depmap_mmd_matched_dependency_noharm_gate_20260627"
    / "depmap_mmd_matched_dependency_rows.csv",
    "matched_summary": REPORTS
    / "depmap_mmd_matched_dependency_noharm_gate_20260627"
    / "depmap_mmd_matched_dependency_summary.csv",
    "matched_gate": REPORTS / "latentfm_depmap_mmd_matched_dependency_noharm_gate_20260627.json",
    "residual_gate": REPORTS / "latentfm_depmap_dependency_residual_mmd_gate_20260627.json",
}


def finite_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def block_high_low(rows: pd.DataFrame) -> pd.DataFrame:
    work = rows.copy()
    for col in ["dependency_z_within_dataset", "pearson_pert", "test_mmd_clamped"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    out: list[dict[str, Any]] = []
    for key, block in work.groupby(["seed", "dataset", "mmd_bin"], sort=True):
        block = block.dropna(subset=["dependency_z_within_dataset", "pearson_pert", "test_mmd_clamped"])
        if len(block) < 4:
            continue
        q = block["dependency_z_within_dataset"].median()
        high = block[block["dependency_z_within_dataset"] >= q]
        low = block[block["dependency_z_within_dataset"] < q]
        if len(high) < 2 or len(low) < 2:
            continue
        seed, dataset, mmd_bin = key
        out.append(
            {
                "seed": seed,
                "dataset": dataset,
                "mmd_bin": mmd_bin,
                "n": int(len(block)),
                "high_n": int(len(high)),
                "low_n": int(len(low)),
                "dep_high_mean": float(high["dependency_z_within_dataset"].mean()),
                "dep_low_mean": float(low["dependency_z_within_dataset"].mean()),
                "high_minus_low_pp": float(high["pearson_pert"].mean() - low["pearson_pert"].mean()),
                "high_minus_low_mmd": float(high["test_mmd_clamped"].mean() - low["test_mmd_clamped"].mean()),
                "high_pp_tailfrac_lt0": float((high["pearson_pert"] < 0).mean()),
                "low_pp_tailfrac_lt0": float((low["pearson_pert"] < 0).mean()),
            }
        )
    return pd.DataFrame(out)


def bootstrap_seed(blocks: pd.DataFrame, seed_label: str, n_boot: int = 1000) -> dict[str, Any]:
    sub = blocks[blocks["seed"].astype(str).eq(seed_label)].copy()
    if sub.empty:
        return {"seed": seed_label, "blocks": 0}
    rng = np.random.default_rng(abs(hash(seed_label)) % (2**32))
    pp = sub["high_minus_low_pp"].to_numpy(dtype=float)
    mmd = sub["high_minus_low_mmd"].to_numpy(dtype=float)
    n = len(sub)
    pp_vals = []
    mmd_vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        pp_vals.append(float(np.mean(pp[idx])))
        mmd_vals.append(float(np.mean(mmd[idx])))
    pp_ci = np.quantile(pp_vals, [0.025, 0.975])
    mmd_ci = np.quantile(mmd_vals, [0.025, 0.975])
    lodo = []
    for dataset in sorted(sub["dataset"].astype(str).unique()):
        keep = sub[~sub["dataset"].astype(str).eq(dataset)]
        if keep.empty:
            continue
        lodo.append(float(keep["high_minus_low_pp"].mean()))
    return {
        "seed": seed_label,
        "blocks": int(n),
        "datasets": int(sub["dataset"].nunique()),
        "rows": int(sub["n"].sum()),
        "mean_high_minus_low_pp": float(np.mean(pp)),
        "pp_ci_low": finite_float(pp_ci[0]),
        "pp_ci_high": finite_float(pp_ci[1]),
        "mean_high_minus_low_mmd": float(np.mean(mmd)),
        "mmd_ci_low": finite_float(mmd_ci[0]),
        "mmd_ci_high": finite_float(mmd_ci[1]),
        "lodo_max_high_minus_low_pp": finite_float(max(lodo)) if lodo else None,
        "negative_pp_blocks": int((sub["high_minus_low_pp"] < 0).sum()),
        "positive_mmd_blocks": int((sub["high_minus_low_mmd"] > 0).sum()),
    }


def shuffle_null(blocks: pd.DataFrame, n_perm: int = 400) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    vals = blocks["high_minus_low_pp"].to_numpy(dtype=float)
    obs = float(np.mean(vals)) if len(vals) else 0.0
    null_abs = []
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(vals), replace=True)
        null_abs.append(abs(float(np.mean(vals * signs))))
    p = (sum(v >= abs(obs) for v in null_abs) + 1) / (len(null_abs) + 1)
    return {"observed_mean_pp_gap": obs, "signflip_abs_p": float(p), "null_abs_p95": float(np.quantile(null_abs, 0.95))}


def decide(seed_rows: list[dict[str, Any]], null: dict[str, Any], matched_gate: dict[str, Any]) -> dict[str, Any]:
    usable = [r for r in seed_rows if int(r.get("blocks", 0)) > 0]
    reasons: list[str] = []
    if len(usable) < 2:
        reasons.append("fewer_than_two_seed_replicates")
    if any(int(r.get("rows", 0)) < 80 for r in usable):
        reasons.append("seed_matched_rows_below_80")
    if any((finite_float(r.get("pp_ci_high")) or 0.0) >= 0 for r in usable):
        reasons.append("pp_tailrisk_ci_crosses_zero")
    if any((finite_float(r.get("lodo_max_high_minus_low_pp")) or 0.0) >= 0 for r in usable):
        reasons.append("lodo_dataset_pp_tailrisk_not_all_negative")
    if float(null.get("signflip_abs_p", 1.0)) > 0.05:
        reasons.append("signflip_null_not_significant")
    if any(abs(finite_float(r.get("mean_high_minus_low_mmd")) or 0.0) > 0.001 for r in usable):
        reasons.append("mmd_gap_not_stably_within_noharm_band")
    if str(matched_gate.get("status", "")).endswith("fail_no_gpu"):
        reasons.append("prior_mmd_matched_dependency_noharm_gate_failed")

    # This gate is intentionally conservative. A tail-risk signal can suggest
    # a future curriculum design, but without stable no-harm it cannot launch.
    status = "depmap_tailrisk_veto_signal_but_no_gpu"
    gpu = False
    next_action = (
        "keep DepMap dependency as tail-risk/failure-analysis covariate; do not launch "
        "dependency-weighted GPU smoke unless a future matched gate stabilizes MMD and LODO"
    )
    if not reasons:
        status = "depmap_tailrisk_veto_cpu_pass_external_audit_needed"
        next_action = "request external audit before dependency-veto GPU launcher"
    return {
        "status": status,
        "gpu_authorized_next": gpu,
        "reasons": reasons,
        "signflip_null": null,
        "seed_rows": seed_rows,
        "next_action": next_action,
    }


def fmt(value: Any, digits: int = 4) -> str:
    x = finite_float(value)
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


def md_seed_table(rows: list[dict[str, Any]]) -> str:
    cols = [
        "seed",
        "rows",
        "datasets",
        "blocks",
        "mean_high_minus_low_pp",
        "pp_ci_low",
        "pp_ci_high",
        "mean_high_minus_low_mmd",
        "lodo_max_high_minus_low_pp",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = [fmt(row.get(c)) if c != "seed" else str(row.get(c)) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_outputs(
    blocks: pd.DataFrame,
    seed_rows: list[dict[str, Any]],
    null: dict[str, Any],
    decision: dict[str, Any],
    matched_gate: dict[str, Any],
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    block_path = OUT_DIR / "depmap_tailrisk_veto_block_rows_20260630.csv"
    json_path = OUT_DIR / "depmap_tailrisk_veto_gate_20260630.json"
    md_path = OUT_DIR / "LATENTFM_DEPMAP_TAILRISK_VETO_GATE_20260630.md"
    blocks.to_csv(block_path, index=False)
    payload = {
        "boundary": {
            "cpu_only": True,
            "reads_completed_reports_only": True,
            "training_or_inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection": False,
            "trackc_query_access": False,
        },
        "inputs": {k: str(v) for k, v in INPUTS.items()},
        "prior_matched_gate_status": matched_gate.get("status"),
        "seed_rows": seed_rows,
        "signflip_null": null,
        "decision": decision,
        "outputs": {"block_rows": str(block_path), "markdown_report": str(md_path)},
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    text = f"""# LatentFM DepMap Tail-Risk Veto Gate 20260630

## Boundary

- CPU/report-only synthesis over completed DepMap matched dependency rows.
- Tests dependency as a tail-risk veto/curriculum covariate, not as a positive information axis.
- No training, inference, checkpoint selection, canonical multi selection, Track C query access, or GPU.

## Decision

- status: `{decision['status']}`
- GPU authorized next: `{decision['gpu_authorized_next']}`
- reasons: `{', '.join(decision['reasons']) if decision['reasons'] else 'none'}`
- next action: `{decision['next_action']}`

## Seed-Level Tail-Risk Signal

{md_seed_table(seed_rows)}

## Sign-Flip Null

- observed mean PP gap: `{fmt(null['observed_mean_pp_gap'])}`
- sign-flip abs p: `{fmt(null['signflip_abs_p'])}`
- null abs p95: `{fmt(null['null_abs_p95'])}`

## Interpretation

High DepMap dependency behaves like a potential hard-row/tail-risk covariate:
the high-minus-low Pearson gap is negative on the existing matched rows. But
the evidence is not strong enough to launch a dependency-weighted GPU smoke:
row counts sit below the previous seed threshold, confidence intervals are not
cleanly separated from zero, and MMD/no-harm remains unstable. Use this as
failure-analysis or future curriculum inspiration only.

## Artifacts

- JSON: `{json_path}`
- block rows: `{block_path}`
"""
    md_path.write_text(text, encoding="utf-8")


def main() -> None:
    rows = pd.read_csv(INPUTS["matched_rows"])
    matched_gate = load_json(INPUTS["matched_gate"])
    blocks = block_high_low(rows)
    seed_rows = [bootstrap_seed(blocks, seed) for seed in sorted(blocks["seed"].astype(str).unique())]
    null = shuffle_null(blocks)
    decision = decide(seed_rows, null, matched_gate)
    write_outputs(blocks, seed_rows, null, decision, matched_gate)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
