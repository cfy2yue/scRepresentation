#!/usr/bin/env python3
"""Source-verified background/type v2 tail-localization gate."""

from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean
from typing import Any, Callable

ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
V1 = REPORTS / "latentfm_source_verified_crossed_background_type_lodo_gate_20260624.json"
OUT_JSON = REPORTS / "latentfm_source_verified_background_type_v2_gate_20260625.json"
OUT_MD = REPORTS / "LATENTFM_SOURCE_VERIFIED_BACKGROUND_TYPE_V2_GATE_20260625.md"
SEED = 20260625


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def weighted(rows: list[dict[str, Any]], key: str) -> float:
    n = sum(int(r.get("n") or 0) for r in rows)
    if n <= 0:
        return 0.0
    return sum(float(r.get(key) or 0.0) * int(r.get("n") or 0) for r in rows) / n


def bootstrap(rows: list[dict[str, Any]], n_boot: int = 2000) -> dict[str, Any]:
    rng = random.Random(SEED)
    if not rows:
        return {"ci": [None, None], "p_le_zero": None, "n_boot": 0}
    vals = []
    for _ in range(n_boot):
        sample = [rng.choice(rows) for _ in rows]
        vals.append(weighted(sample, "pp_delta_mean"))
    vals.sort()
    return {
        "ci": [vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]],
        "p_le_zero": sum(v <= 0.0 for v in vals) / len(vals),
        "n_boot": n_boot,
    }


def leave_one_min(rows: list[dict[str, Any]]) -> float | None:
    if len(rows) <= 1:
        return None
    vals = []
    for i in range(len(rows)):
        vals.append(weighted([r for j, r in enumerate(rows) if j != i], "pp_delta_mean"))
    return min(vals)


def group_summary(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    out = []
    for value in sorted({str(r.get(field) or "") for r in rows}):
        subset = [r for r in rows if str(r.get(field) or "") == value]
        out.append(
            {
                field: value,
                "datasets": [r["dataset"] for r in subset],
                "n": sum(int(r.get("n") or 0) for r in subset),
                "dataset_count": len(subset),
                "pp_delta": weighted(subset, "pp_delta_mean"),
                "mmd_delta": weighted(subset, "mmd_delta_mean"),
            }
        )
    return out


def evaluate_policy(name: str, rows: list[dict[str, Any]], desc: str) -> dict[str, Any]:
    n = sum(int(r.get("n") or 0) for r in rows)
    pp = weighted(rows, "pp_delta_mean")
    mmd = weighted(rows, "mmd_delta_mean")
    ds_min = min((float(r["pp_delta_mean"]) for r in rows), default=0.0)
    neg = sum(float(r["pp_delta_mean"]) < -0.020 for r in rows)
    max_weight = max((int(r.get("n") or 0) / n for r in rows), default=1.0)
    boot = bootstrap(rows)
    loo = leave_one_min(rows)
    bg_groups = group_summary(rows, "background")
    type_groups = group_summary(rows, "perturbation_type")
    min_bg = min((float(r["pp_delta"]) for r in bg_groups), default=0.0)
    min_type = min((float(r["pp_delta"]) for r in type_groups), default=0.0)
    reasons = []
    if len(rows) < 5:
        reasons.append("too_few_datasets")
    if n < 80:
        reasons.append("too_few_conditions")
    if pp < 0.010:
        reasons.append("weighted_pp_below_0p010")
    if mmd > 0.001:
        reasons.append("weighted_mmd_above_0p001")
    if ds_min < -0.020:
        reasons.append("dataset_tail_below_minus_0p020")
    if neg:
        reasons.append("negative_dataset_tail_present")
    if boot["ci"][0] is None or float(boot["ci"][0]) <= 0.0:
        reasons.append("bootstrap_ci_lower_not_positive")
    if loo is None or loo < 0.005:
        reasons.append("leave_one_dataset_min_below_0p005")
    if min_bg < 0.005:
        reasons.append("background_stratum_min_below_0p005")
    if min_type < 0.005:
        reasons.append("type_stratum_min_below_0p005")
    if max_weight > 0.35:
        reasons.append("single_dataset_weight_above_0p35")
    return {
        "policy": name,
        "description": desc,
        "pass_gate": not reasons,
        "reasons": reasons,
        "dataset_count": len(rows),
        "n_conditions": n,
        "weighted_pp": pp,
        "weighted_mmd": mmd,
        "dataset_min_pp": ds_min,
        "negative_tail_count": neg,
        "bootstrap": boot,
        "leave_one_dataset_min_pp": loo,
        "background_min_pp": min_bg,
        "type_min_pp": min_type,
        "max_dataset_weight": max_weight,
        "datasets": [r["dataset"] for r in rows],
    }


def main() -> int:
    v1 = load_json(V1)
    primary = list(v1.get("primary_rows") or [])
    policies: list[tuple[str, str, Callable[[dict[str, Any]], bool]]] = [
        ("all_source_verified_cap_gain", "all source-verified cap-gain rows", lambda r: True),
        ("non_neuron", "exclude Human neurons tail background", lambda r: r.get("background") != "Human neurons"),
        ("non_neuron_non_crisprko", "exclude Human neurons and CRISPRko tail axis", lambda r: r.get("background") != "Human neurons" and r.get("perturbation_type") != "CRISPRko"),
        ("k562_background", "K562 source-verified cap-gain background", lambda r: r.get("background") == "K562"),
        ("crispri_source_verified", "source-verified CRISPRi cap-gain rows", lambda r: r.get("perturbation_type") == "CRISPRi"),
        ("non_hard_tail_backgrounds", "exclude backgrounds whose v1 stratum pp < -0.020", lambda r: r.get("background") not in {"Human neurons", "A375"}),
    ]
    evaluated = []
    for name, desc, pred in policies:
        evaluated.append(evaluate_policy(name, [r for r in primary if pred(r)], desc))
    evaluated.sort(key=lambda r: (bool(r["pass_gate"]), float(r["weighted_pp"]), -float(r["max_dataset_weight"])), reverse=True)
    passing = [r for r in evaluated if r["pass_gate"]]
    status = "source_verified_background_type_v2_pass_external_review_no_gpu_yet" if passing else "source_verified_background_type_v2_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "policies": evaluated,
        "best_policy": evaluated[0] if evaluated else None,
        "background_strata_v1": v1.get("background_strata", []),
        "type_strata_v1": v1.get("type_strata", []),
        "reasons": [] if passing else ["no_predeclared_background_type_policy_passed_tail_safe_gate"],
        "next_action": (
            "external review, then write a bounded source/type/background-tail-aware GPU protocol"
            if passing
            else "no GPU; background/type remains failure-localization evidence until a new tail-protection mechanism is proposed"
        ),
        "boundary": {
            "cpu_only": True,
            "reads_completed_train_only_reports": True,
            "reads_canonical_multi": False,
            "reads_heldout_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Source-Verified Background/Type V2 Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only v2 tail-localization gate using the completed source-verified cap120-vs-cap30 rows.",
        "- Tests predeclared background/type policies; it does not train, infer, use GPU, read canonical multi, or read Track C query.",
        "",
        "## Policy Table",
        "",
        "| policy | pass | datasets | conditions | pp | MMD | min ds | CI lower | LOO min | bg min | type min | max weight | reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in evaluated:
        ci0 = row["bootstrap"]["ci"][0]
        lines.append(
            f"| `{row['policy']}` | `{row['pass_gate']}` | {row['dataset_count']} | {row['n_conditions']} | "
            f"{row['weighted_pp']:+.6f} | {row['weighted_mmd']:+.6f} | {row['dataset_min_pp']:+.6f} | "
            f"{ci0 if ci0 is not None else 'NA'} | {row['leave_one_dataset_min_pp'] if row['leave_one_dataset_min_pp'] is not None else 'NA'} | "
            f"{row['background_min_pp']:+.6f} | {row['type_min_pp']:+.6f} | {row['max_dataset_weight']:.3f} | `{row['reasons']}` |"
        )
    lines.extend(
        [
            "",
            "## V1 Tail Localization",
            "",
            "- Human neurons and A375/CRISPRko are the dominant negative source-verified strata.",
            "- K562 is the best-looking background, but the v2 policy remains too small and dataset-weight dominated for GPU authorization.",
            "",
            "## Decision",
            "",
            f"- GPU authorized: `{payload['gpu_authorized']}`",
            f"- best policy: `{payload['best_policy']['policy'] if payload['best_policy'] else 'NA'}`",
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
