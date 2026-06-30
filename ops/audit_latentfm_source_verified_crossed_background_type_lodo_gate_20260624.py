#!/usr/bin/env python3
"""Source-verified crossed background/type LODO gate for scaling.

This is a narrow CPU diagnostic gate. It tests cap120-cap30 scaling only on
source-verified train-only rows with real cap-gain support, then checks
background/type leave-one-out tails and simple label-shuffle controls.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
MIXED = REPORTS / "latentfm_scaling_mixed_effect_lodo_condition_count_gate_20260624.json"
PROV = REPORTS / "latentfm_scaling_provenance_estimand_matrix_gate_20260624.json"
NOHARM = REPORTS / "latentfm_noharm_calibration_positive_controls_gate_20260624.json"
OUT_JSON = REPORTS / "latentfm_source_verified_crossed_background_type_lodo_gate_20260624.json"
OUT_MD = REPORTS / "LATENTFM_SOURCE_VERIFIED_CROSSED_BACKGROUND_TYPE_LODO_GATE_20260624.md"

SEED = 20260624
N_BOOT = 5000
N_SHUFFLE = 2000


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def weighted(rows: list[dict[str, Any]], key: str) -> float | None:
    num = 0.0
    den = 0.0
    for row in rows:
        n = float(row.get("n") or 0.0)
        val = row.get(key)
        if n > 0 and val is not None:
            num += n * float(val)
            den += n
    return None if den <= 0 else num / den


def total_n(rows: list[dict[str, Any]]) -> int:
    return int(sum(int(row.get("n") or 0) for row in rows))


def leave_one(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    out = []
    for val in sorted({str(row.get(field) or "") for row in rows}):
        kept = [row for row in rows if str(row.get(field) or "") != val]
        out.append(
            {
                "left_out": val,
                "n_kept": total_n(kept),
                "pp_delta_mean": weighted(kept, "pp_delta_mean"),
                "mmd_delta_mean": weighted(kept, "mmd_delta_mean"),
            }
        )
    return out


def group_rows(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    out = []
    for val in sorted({str(row.get(field) or "") for row in rows}):
        group = [row for row in rows if str(row.get(field) or "") == val]
        out.append(
            {
                field: val,
                "datasets": [row["dataset"] for row in group],
                "dataset_count": len(group),
                "n": total_n(group),
                "pp_delta_mean": weighted(group, "pp_delta_mean"),
                "mmd_delta_mean": weighted(group, "mmd_delta_mean"),
            }
        )
    return out


def bootstrap_ci(rows: list[dict[str, Any]], *, seed: int = SEED, n_boot: int = N_BOOT) -> dict[str, Any]:
    rng = random.Random(seed)
    vals = []
    if not rows:
        return {"n_boot": 0, "ci95": None, "p_le_zero": None, "median": None}
    for _ in range(n_boot):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        val = weighted(sample, "pp_delta_mean")
        if val is not None:
            vals.append(val)
    vals.sort()
    if not vals:
        return {"n_boot": 0, "ci95": None, "p_le_zero": None, "median": None}
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    med = vals[int(0.5 * (len(vals) - 1))]
    return {
        "n_boot": len(vals),
        "ci95": [lo, hi],
        "median": med,
        "p_le_zero": sum(1 for x in vals if x <= 0.0) / len(vals),
    }


def shuffle_controls(rows: list[dict[str, Any]], *, field: str, seed: int = SEED, n_shuffle: int = N_SHUFFLE) -> dict[str, Any]:
    """Permutation control for metadata association with pp tails.

    Overall weighted pp is invariant to label shuffling, so this evaluates the
    distribution of worst stratum means after pp values are permuted across
    fixed background/type labels.
    """

    rng = random.Random(seed)
    labels = [row[field] for row in rows]
    values = [float(row["pp_delta_mean"]) for row in rows]
    mmd_values = [float(row["mmd_delta_mean"]) for row in rows]
    ns = [int(row["n"]) for row in rows]
    worst_vals = []
    best_vals = []
    for _ in range(n_shuffle):
        perm = values[:]
        rng.shuffle(perm)
        pseudo = [
            {**row, "pp_delta_mean": perm[i], "mmd_delta_mean": mmd_values[i], "n": ns[i], field: labels[i]}
            for i, row in enumerate(rows)
        ]
        strata = group_rows(pseudo, field)
        means = [float(s["pp_delta_mean"]) for s in strata if s["pp_delta_mean"] is not None]
        if means:
            worst_vals.append(min(means))
            best_vals.append(max(means))
    worst_vals.sort()
    best_vals.sort()
    if not worst_vals:
        return {"n_shuffle": 0, "worst_stratum_median": None, "worst_stratum_q10": None, "best_stratum_q90": None}
    return {
        "n_shuffle": len(worst_vals),
        "worst_stratum_median": worst_vals[int(0.5 * (len(worst_vals) - 1))],
        "worst_stratum_q10": worst_vals[int(0.1 * (len(worst_vals) - 1))],
        "best_stratum_q90": best_vals[int(0.9 * (len(best_vals) - 1))],
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def main() -> int:
    mixed = load(MIXED)
    prov = load(PROV)
    noharm = load(NOHARM) if NOHARM.exists() else {}
    all_rows = [dict(row) for row in (mixed.get("dataset_rows") or [])]
    source_verified = [row for row in all_rows if row.get("source_quality") == "source_verified"]
    primary = [row for row in source_verified if int(row.get("cap_gain") or 0) > 0]
    zero_controls = [row for row in source_verified if int(row.get("cap_gain") or 0) <= 0]

    primary_pp = weighted(primary, "pp_delta_mean")
    primary_mmd = weighted(primary, "mmd_delta_mean")
    primary_min = min((float(row["pp_delta_mean"]) for row in primary), default=None)
    primary_neg_tails = sum(1 for row in primary if float(row["pp_delta_mean"]) < -0.020)
    zero_pp = weighted(zero_controls, "pp_delta_mean")
    all_source_pp = weighted(source_verified, "pp_delta_mean")
    boot = bootstrap_ci(primary)
    lobg = leave_one(primary, "background")
    lotype = leave_one(primary, "perturbation_type")
    bg_strata = group_rows(primary, "background")
    type_strata = group_rows(primary, "perturbation_type")
    bg_ctrl = shuffle_controls(primary, field="background", seed=SEED + 1)
    type_ctrl = shuffle_controls(primary, field="perturbation_type", seed=SEED + 2)

    min_lobg = min((float(row["pp_delta_mean"]) for row in lobg if row["pp_delta_mean"] is not None), default=None)
    min_lotype = min((float(row["pp_delta_mean"]) for row in lotype if row["pp_delta_mean"] is not None), default=None)
    reasons: list[str] = []
    if len(primary) < 8:
        reasons.append("too_few_source_verified_cap_gain_datasets")
    if primary_pp is None or primary_pp < 0.010:
        reasons.append("source_verified_cap_gain_weighted_pp_lt_0p010")
    if not boot.get("ci95") or float(boot["ci95"][0]) <= 0.0:
        reasons.append("bootstrap_ci_lower_not_positive")
    if primary_min is None or primary_min < -0.020:
        reasons.append("source_verified_cap_gain_dataset_tail_below_minus_0p020")
    if primary_neg_tails:
        reasons.append("source_verified_cap_gain_negative_tails_present")
    if min_lobg is None or min_lobg < -0.020:
        reasons.append("leave_background_min_below_minus_0p020")
    if min_lotype is None or min_lotype < -0.020:
        reasons.append("leave_type_min_below_minus_0p020")
    if primary_mmd is None or primary_mmd > 0.001:
        reasons.append("source_verified_cap_gain_mmd_mean_gt_0p001")
    if zero_pp is not None and zero_pp > (primary_pp or -999):
        reasons.append("zero_cap_gain_controls_not_lower_than_primary")
    if (noharm.get("status") or "") != "noharm_calibration_positive_controls_pass_external_review_next":
        reasons.append("noharm_calibration_not_passed_gpu_blocked")

    status = "source_verified_crossed_background_type_lodo_pass_external_review_no_gpu_yet" if not reasons else "source_verified_crossed_background_type_lodo_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "reads_train_only_completed_reports": True,
            "reads_canonical_metrics": False,
            "reads_canonical_multi": False,
            "reads_heldout_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "inputs": {
            "mixed_effect_lodo": str(MIXED),
            "provenance": str(PROV),
            "noharm_calibration": str(NOHARM),
        },
        "summary": {
            "source_verified_rows": len(source_verified),
            "source_verified_cap_gain_rows": len(primary),
            "source_verified_cap_gain_n": total_n(primary),
            "source_verified_cap_gain_weighted_pp": primary_pp,
            "source_verified_cap_gain_weighted_mmd": primary_mmd,
            "source_verified_cap_gain_min_dataset_pp": primary_min,
            "source_verified_cap_gain_negative_tails_lt_minus_0p020": primary_neg_tails,
            "source_verified_all_weighted_pp": all_source_pp,
            "zero_cap_gain_control_rows": len(zero_controls),
            "zero_cap_gain_control_weighted_pp": zero_pp,
            "bootstrap": boot,
            "min_leave_background_pp": min_lobg,
            "min_leave_type_pp": min_lotype,
            "background_shuffle_control": bg_ctrl,
            "type_shuffle_control": type_ctrl,
            "global_family_proxy_pp_delta": (mixed.get("summary") or {}).get("family_proxy_pp_delta"),
            "global_family_proxy_mmd_delta": (mixed.get("summary") or {}).get("family_proxy_mmd_delta"),
            "noharm_calibration_status": noharm.get("status"),
            "provenance_status": prov.get("status"),
        },
        "primary_rows": primary,
        "zero_cap_gain_controls": zero_controls,
        "background_strata": bg_strata,
        "type_strata": type_strata,
        "leave_one_background": lobg,
        "leave_one_type": lotype,
        "reasons": reasons,
        "decision": {
            "can_authorize_gpu_by_itself": False,
            "gpu_next_action": "none",
            "if_passed": "external_review_then_require_separate_noharm_safety_gate_before_gpu",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Source-Verified Crossed Background/Type LODO Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only diagnostic gate using completed train-only scaling reports.",
        "- Primary rows are source-verified datasets with cap120 > cap30 train conditions.",
        "- Zero-cap-gain source-verified datasets are controls, not positive evidence.",
        "- Does not read canonical metrics, canonical multi, Track C query, train, infer, or use GPU.",
        "",
        "## Summary",
        "",
        f"- source-verified cap-gain rows: `{len(primary)}`",
        f"- source-verified cap-gain n: `{total_n(primary)}`",
        f"- weighted pp/MMD: `{fmt(primary_pp)}` / `{fmt(primary_mmd)}`",
        f"- min dataset pp: `{fmt(primary_min)}`",
        f"- negative tails `< -0.020`: `{primary_neg_tails}`",
        f"- bootstrap pp CI: `{[fmt(x) for x in (boot.get('ci95') or [])]}`; p<=0 `{fmt(boot.get('p_le_zero'))}`",
        f"- min leave-background/type pp: `{fmt(min_lobg)}` / `{fmt(min_lotype)}`",
        f"- zero-cap-gain control weighted pp: `{fmt(zero_pp)}`",
        f"- no-harm calibration status: `{noharm.get('status')}`",
        "",
        "## Primary Rows",
        "",
        "| dataset | background | type | n | pp delta | MMD delta |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            f"| `{row['dataset']}` | `{row['background']}` | `{row['perturbation_type']}` | {row['n']} | {fmt(row['pp_delta_mean'])} | {fmt(row['mmd_delta_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Controls",
            "",
            f"- background shuffle worst-stratum median/q10: `{fmt(bg_ctrl.get('worst_stratum_median'))}` / `{fmt(bg_ctrl.get('worst_stratum_q10'))}`",
            f"- type shuffle worst-stratum median/q10: `{fmt(type_ctrl.get('worst_stratum_median'))}` / `{fmt(type_ctrl.get('worst_stratum_q10'))}`",
            "",
            "## Decision",
            "",
            f"- reasons: `{reasons}`",
            "- GPU authorized: `False`",
            "- Even a pass here would require external review and a separate no-harm/safety gate before any GPU smoke.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "reasons": reasons}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
