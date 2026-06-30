#!/usr/bin/env python3
"""CPU-only target observability v2 gate.

This gate reuses the completed cap120-vs-cap30 target-activity audit rows and
asks whether any predeclared target-observability policy is tail-safe enough to
unlock a future GPU protocol.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
TARGET_JSON = REPORTS / "latentfm_scaling_target_activity_gate_20260624.json"
CONDITION_JSON = REPORTS / "latentfm_scaling_mixed_effect_lodo_condition_count_gate_20260624.json"
OUT_JSON = REPORTS / "latentfm_target_observability_v2_gate_20260625.json"
OUT_MD = REPORTS / "LATENTFM_TARGET_OBSERVABILITY_V2_GATE_20260625.md"
SEED = 20260625


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_meta() -> dict[str, dict[str, Any]]:
    data = load_json(CONDITION_JSON)
    return {str(row["dataset"]): row for row in data.get("dataset_rows", [])}


def eligible(row: dict[str, Any]) -> bool:
    return row.get("activity_status") == "ok"


def pp(row: dict[str, Any]) -> float:
    return float(row.get("pp_delta_cap120_minus_cap30") or 0.0)


def mmd(row: dict[str, Any]) -> float:
    return float(row.get("mmd_delta_cap120_minus_cap30") or 0.0)


def summarize(rows: list[dict[str, Any]], *, policy: str, description: str) -> dict[str, Any]:
    if not rows:
        return {
            "policy": policy,
            "description": description,
            "pass_gate": False,
            "reasons": ["empty_policy"],
            "n_conditions": 0,
            "dataset_count": 0,
        }
    vals = [pp(r) for r in rows]
    mmd_vals = [mmd(r) for r in rows]
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_ds[str(row["dataset"])].append(pp(row))
    ds_rows = [{"dataset": ds, "n": len(v), "pp_mean": float(mean(v))} for ds, v in sorted(by_ds.items())]
    rng = random.Random(SEED)
    boot = []
    for _ in range(2000):
        sample_ds = [rng.choice(ds_rows) for _ in ds_rows]
        boot.append(float(mean([x["pp_mean"] for x in sample_ds])))
    boot.sort()
    ds_min = min(row["pp_mean"] for row in ds_rows)
    reasons: list[str] = []
    if len(rows) < 40:
        reasons.append("too_few_conditions")
    if len(ds_rows) < 5:
        reasons.append("too_few_datasets")
    if mean(vals) < 0.010:
        reasons.append("pp_mean_below_0p010")
    if max(mmd_vals) > 0.004:
        reasons.append("condition_mmd_tail_above_0p004")
    if mean(mmd_vals) > 0.001:
        reasons.append("mmd_mean_above_0p001")
    if ds_min < -0.020:
        reasons.append("dataset_tail_below_minus_0p020")
    if sum(1 for row in ds_rows if row["pp_mean"] < -0.020) > 0:
        reasons.append("negative_dataset_tail_present")
    if boot[50] <= 0.0:
        reasons.append("dataset_bootstrap_ci_lower_not_positive")
    if sum(v < -0.005 for v in vals) / len(vals) > 0.35:
        reasons.append("condition_hard_harm_frac_above_0p35")
    return {
        "policy": policy,
        "description": description,
        "pass_gate": not reasons,
        "reasons": reasons,
        "n_conditions": len(rows),
        "dataset_count": len(ds_rows),
        "pp_mean": float(mean(vals)),
        "pp_min": float(min(vals)),
        "mmd_mean": float(mean(mmd_vals)),
        "mmd_max": float(max(mmd_vals)),
        "condition_hard_harm_frac": float(sum(v < -0.005 for v in vals) / len(vals)),
        "dataset_min_pp": float(ds_min),
        "negative_dataset_tail_count": int(sum(1 for row in ds_rows if row["pp_mean"] < -0.020)),
        "bootstrap_ci": [float(boot[50]), float(boot[1950])],
        "dataset_rows": ds_rows,
    }


def permutation_control(rows: list[dict[str, Any]], score_key: str, *, high_count: int) -> dict[str, Any]:
    ok = [r for r in rows if eligible(r)]
    vals = [pp(r) for r in ok]
    scores = [float(r.get(score_key) or 0.0) for r in ok]
    observed_idx = sorted(range(len(ok)), key=lambda i: scores[i], reverse=True)[:high_count]
    observed = mean([vals[i] for i in observed_idx]) if observed_idx else 0.0
    rng = random.Random(SEED)
    controls = []
    for _ in range(2000):
        idx = rng.sample(range(len(ok)), k=high_count)
        controls.append(mean([vals[i] for i in idx]))
    controls.sort()
    return {
        "score_key": score_key,
        "observed_high_mean": float(observed),
        "shuffle_mean": float(mean(controls)),
        "shuffle_p95": float(controls[int(0.95 * len(controls))]),
        "observed_minus_shuffle_mean": float(observed - mean(controls)),
    }


def main() -> int:
    target = load_json(TARGET_JSON)
    meta = dataset_meta()
    rows = [dict(row) for row in target.get("rows", []) if eligible(row)]
    for row in rows:
        m = meta.get(str(row["dataset"]), {})
        row["source_quality"] = m.get("source_quality")
        row["perturbation_type"] = m.get("perturbation_type")
        row["background"] = m.get("background")
    policies: list[tuple[str, str, Callable[[dict[str, Any]], bool]]] = [
        ("all_activity_ok", "all rows with target activity available", lambda r: True),
        ("nonzero_target_expr", "target nonzero fraction > 0", lambda r: float(r.get("target_expr_nonzero_fraction") or 0.0) > 0.0),
        ("top_quartile_nonzero", "top quartile by target nonzero fraction", lambda r: False),
        ("top_quartile_expr", "top quartile by target expression mean", lambda r: False),
        ("crispri_nonzero", "CRISPRi rows with target nonzero fraction > 0", lambda r: r.get("perturbation_type") == "CRISPRi" and float(r.get("target_expr_nonzero_fraction") or 0.0) > 0.0),
        ("source_verified_nonzero", "source-verified rows with target nonzero fraction > 0", lambda r: r.get("source_quality") == "source_verified" and float(r.get("target_expr_nonzero_fraction") or 0.0) > 0.0),
    ]
    nonzero_scores = sorted([float(r.get("target_expr_nonzero_fraction") or 0.0) for r in rows])
    expr_scores = sorted([float(r.get("target_expr_mean") or 0.0) for r in rows])
    q_nonzero = nonzero_scores[int(0.75 * (len(nonzero_scores) - 1))] if nonzero_scores else 0.0
    q_expr = expr_scores[int(0.75 * (len(expr_scores) - 1))] if expr_scores else 0.0
    evaluated = []
    for name, desc, pred in policies:
        if name == "top_quartile_nonzero":
            subset = [r for r in rows if float(r.get("target_expr_nonzero_fraction") or 0.0) >= q_nonzero]
        elif name == "top_quartile_expr":
            subset = [r for r in rows if float(r.get("target_expr_mean") or 0.0) >= q_expr]
        else:
            subset = [r for r in rows if pred(r)]
        evaluated.append(summarize(subset, policy=name, description=desc))
    evaluated.sort(key=lambda x: (bool(x["pass_gate"]), float(x.get("pp_mean", -999)), -float(x.get("condition_hard_harm_frac", 1))), reverse=True)
    passing = [x for x in evaluated if x["pass_gate"]]
    high_count = max(1, sum(1 for r in rows if float(r.get("target_expr_nonzero_fraction") or 0.0) >= q_nonzero))
    controls = [
        permutation_control(rows, "target_expr_nonzero_fraction", high_count=high_count),
        permutation_control(rows, "target_expr_mean", high_count=high_count),
    ]
    # Controls must collapse relative to observed top-quartile policies. If the
    # best observed policy is no better than shuffled target scores, no GPU.
    best = evaluated[0] if evaluated else None
    control_reasons = []
    if best and best.get("pass_gate"):
        if max(c["observed_minus_shuffle_mean"] for c in controls) < 0.010:
            control_reasons.append("target_score_control_increment_below_0p010")
    status = "target_observability_v2_pass_external_review_no_gpu_yet" if passing and not control_reasons else "target_observability_v2_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "policies": evaluated,
        "best_policy": best,
        "score_quantiles": {"nonzero_q75": q_nonzero, "expr_q75": q_expr},
        "permutation_controls": controls,
        "reasons": [] if status.startswith("target_observability_v2_pass") else ["no_predeclared_target_observability_policy_passed_controls_and_tail_gate", *control_reasons],
        "next_action": (
            "external review, then write a bounded target-observability protocol"
            if status.startswith("target_observability_v2_pass")
            else "no GPU; target observability remains a weak/confounded mechanism hint"
        ),
        "boundary": {
            "cpu_only": True,
            "reads_completed_train_only_reports": True,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM Target Observability V2 Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only v2 gate using completed target-activity rows from cap120-vs-cap30 train-only summaries.",
        "- Tests predeclared target observability policies and permutation controls.",
        "- Does not train, infer, use GPU, read canonical multi, or read Track C query.",
        "",
        "## Policy Table",
        "",
        "| policy | pass | datasets | conditions | pp | MMD | min ds | CI lower | hard-harm frac | reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in evaluated:
        ci0 = (row.get("bootstrap_ci") or [None, None])[0]
        lines.append(
            f"| `{row['policy']}` | `{row['pass_gate']}` | {row.get('dataset_count', 0)} | {row.get('n_conditions', 0)} | "
            f"{float(row.get('pp_mean', 0.0)):+.6f} | {float(row.get('mmd_mean', 0.0)):+.6f} | "
            f"{float(row.get('dataset_min_pp', 0.0)):+.6f} | {ci0 if ci0 is not None else 'NA'} | "
            f"{float(row.get('condition_hard_harm_frac', 0.0)):.3f} | `{row['reasons']}` |"
        )
    lines.extend(
        [
            "",
            "## Controls",
            "",
        ]
    )
    for ctrl in controls:
        lines.append(
            f"- `{ctrl['score_key']}` observed high mean `{ctrl['observed_high_mean']:+.6f}`, "
            f"shuffle mean `{ctrl['shuffle_mean']:+.6f}`, delta `{ctrl['observed_minus_shuffle_mean']:+.6f}`"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- GPU authorized: `{payload['gpu_authorized']}`",
            f"- best policy: `{best['policy'] if best else 'NA'}`",
            f"- reasons: `{payload['reasons']}`",
            f"- next action: `{payload['next_action']}`",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorized": False, "out_md": str(OUT_MD)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
