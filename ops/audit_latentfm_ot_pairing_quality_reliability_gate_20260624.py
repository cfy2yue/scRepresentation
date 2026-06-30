#!/usr/bin/env python3
"""CPU-only OT pairing-quality reopen gate.

This gate asks whether train-only OT pairing-quality diagnostics predict
internal response reliability strongly enough to justify reopening OT.
It does not train, select checkpoints, or read canonical/query artifacts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path("/data/cyx/1030/scLatent")
PAIR_JSON = ROOT / "reports/latentfm_ot_pairing_signal_audit_20260624.json"
ANCHOR_JSON = ROOT / "reports/latentfm_xverse_tracka_anchor_internal_val_error_map_20260622.json"
OUT_JSON = ROOT / "reports/latentfm_ot_pairing_quality_reliability_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_OT_PAIRING_QUALITY_RELIABILITY_GATE_20260624.md"

SEED = 42
N_PERM = 2000


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 3:
        return float("nan")
    mx = mean(x)
    my = mean(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(rankdata(x), rankdata(y))


def permutation_p_abs(x: list[float], y: list[float], seed: int, n_perm: int = N_PERM) -> float:
    import random

    obs = abs(spearman(x, y))
    if not math.isfinite(obs):
        return float("nan")
    rng = random.Random(seed)
    yp = list(y)
    count = 1
    for _ in range(n_perm):
        rng.shuffle(yp)
        r = spearman(x, yp)
        if math.isfinite(r) and abs(r) >= obs:
            count += 1
    return count / (n_perm + 1)


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def build_joined_rows(pair_payload: dict[str, Any], anchor_payload: dict[str, Any]) -> list[dict[str, Any]]:
    pair_by_dataset = pair_payload.get("summary", {}).get("by_dataset", {})
    if isinstance(pair_by_dataset, list):
        pair_by_dataset = dict(pair_by_dataset)

    internal_by_dataset: dict[str, dict[str, dict[str, Any]]] = {}
    for row in anchor_payload.get("dataset_summary", []):
        ds = str(row.get("dataset", ""))
        group = str(row.get("group", ""))
        if not ds or not group:
            continue
        internal_by_dataset.setdefault(ds, {})[group] = row

    joined: list[dict[str, Any]] = []
    for ds, prow in sorted(pair_by_dataset.items()):
        groups = internal_by_dataset.get(ds, {})
        cross = groups.get("internal_val_cross_background_seen_gene_proxy")
        family = groups.get("internal_val_family_gene_proxy")
        if not cross or not family:
            continue
        item = {
            "dataset": ds,
            "n_pair_batches": int(prow.get("n_batches", 0) or 0),
            "ot_cost_gain": -float(prow["multinomial_cost_delta_frac_mean"]),
            "ot_marginal_noise": float(prow["multinomial_delta_rel_error_mean"]),
            "ot_unique_gt": float(prow["multinomial_unique_gt_frac_mean"]),
            "cross_anchor_pp": safe_float(cross.get("anchor_pearson_pert")),
            "family_anchor_pp": safe_float(family.get("anchor_pearson_pert")),
            "cross_anchor_minus_gene": safe_float(cross.get("anchor_minus_gene_raw_mean")),
            "family_anchor_minus_gene": safe_float(family.get("anchor_minus_gene_raw_mean")),
            "cross_mmd": safe_float(cross.get("anchor_mmd_clamped")),
            "family_mmd": safe_float(family.get("anchor_mmd_clamped")),
        }
        if all(item.get(k) is not None for k in item if k not in {"dataset", "n_pair_batches"}):
            joined.append(item)
    return joined


def correlation_rows(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features = {
        "ot_cost_gain": "positive",
        "ot_unique_gt": "positive",
        "ot_marginal_noise": "negative",
    }
    targets = {
        "cross_anchor_pp": "positive",
        "family_anchor_pp": "positive",
        "cross_anchor_minus_gene": "positive",
        "family_anchor_minus_gene": "positive",
        "cross_mmd": "negative",
        "family_mmd": "negative",
    }
    rows: list[dict[str, Any]] = []
    for f_idx, (feature, f_dir) in enumerate(features.items()):
        x = [float(r[feature]) for r in joined]
        for t_idx, (target, t_dir) in enumerate(targets.items()):
            y = [float(r[target]) for r in joined]
            rho = spearman(x, y)
            p_abs = permutation_p_abs(x, y, seed=SEED + f_idx * 100 + t_idx)
            expected_sign = 1
            if f_dir != t_dir:
                expected_sign = -1
            direction_ok = math.isfinite(rho) and (rho * expected_sign) > 0
            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "n": len(joined),
                    "spearman": rho,
                    "perm_p_abs": p_abs,
                    "expected_sign": expected_sign,
                    "direction_ok": direction_ok,
                    "material": bool(direction_ok and abs(rho) >= 0.35 and p_abs <= 0.10),
                }
            )
    return rows


def decide(joined: list[dict[str, Any]], rows: list[dict[str, Any]], condition_overlap: int) -> dict[str, Any]:
    material = [r for r in rows if r["material"]]
    contradictions = [
        r
        for r in rows
        if math.isfinite(r["spearman"]) and abs(r["spearman"]) >= 0.35 and not r["direction_ok"]
    ]
    reasons: list[str] = []
    if condition_overlap == 0:
        reasons.append("no_condition_level_overlap_pairing_audit_vs_internal_val")
    if len(joined) < 12:
        reasons.append("dataset_overlap_below_12")
    if len(material) < 2:
        reasons.append("fewer_than_two_material_expected_direction_correlations")
    if contradictions:
        reasons.append("material_contradictory_correlations_present")
    status = "ot_pairing_quality_reliability_gate_pass_code_gate_next_no_gpu"
    if reasons:
        status = "ot_pairing_quality_reliability_gate_fail_no_gpu"
    return {
        "status": status,
        "gpu_authorized": False,
        "n_joined_datasets": len(joined),
        "condition_overlap": condition_overlap,
        "material_expected_correlations": len(material),
        "material_contradictions": len(contradictions),
        "reasons": reasons,
        "next_action": (
            "If pass, design exactly one default-off OT smoke with preserved marginals and fixed fail-close gate."
            if not reasons
            else "Keep OT closed; do not run OT GPU sweeps without a stronger train-only signal."
        ),
    }


def render_md(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# LatentFM OT Pairing Quality Reliability Gate",
        "",
        f"Status: `{decision['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only gate using existing train-only OT pairing diagnostics and train-only internal validation summaries.",
        "- No training, checkpoint selection, canonical/test/query reads, or GPU use.",
        "- Because condition-level overlap is zero, this is a dataset-level proxy gate, not condition-level evidence.",
        "",
        "## Decision",
        "",
        f"- joined datasets: `{decision['n_joined_datasets']}`",
        f"- condition overlap: `{decision['condition_overlap']}`",
        f"- material expected-direction correlations: `{decision['material_expected_correlations']}`",
        f"- material contradictory correlations: `{decision['material_contradictions']}`",
        f"- GPU authorized: `{decision['gpu_authorized']}`",
        "",
        "Reasons:",
    ]
    if decision["reasons"]:
        lines.extend([f"- `{r}`" for r in decision["reasons"]])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Correlations",
            "",
            "| feature | target | n | spearman | perm_p_abs | expected sign | material |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["correlations"]:
        lines.append(
            "| {feature} | {target} | {n} | {spearman:.4f} | {perm_p_abs:.4f} | {expected_sign:+d} | {material} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Current OT pairing changes minibatch coupling, but this gate asks a different question: whether pairing-quality variation predicts internal response reliability. Without that link, more OT cost/mode sweeps are not a disciplined use of GPU.",
            "",
            "## Next Action",
            "",
            decision["next_action"],
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    pair_payload = load_json(PAIR_JSON)
    anchor_payload = load_json(ANCHOR_JSON)
    pair_keys = {(r["dataset"], r["condition"]) for r in pair_payload.get("rows", [])}
    anchor_keys = {(r["dataset"], r["condition"]) for r in anchor_payload.get("condition_rows", [])}
    condition_overlap = len(pair_keys & anchor_keys)
    joined = build_joined_rows(pair_payload, anchor_payload)
    rows = correlation_rows(joined)
    payload = {
        "boundary": {
            "pairing_json": str(PAIR_JSON),
            "anchor_internal_val_json": str(ANCHOR_JSON),
            "no_training": True,
            "no_canonical_or_query": True,
            "no_gpu": True,
            "n_perm": N_PERM,
            "seed": SEED,
        },
        "decision": decide(joined, rows, condition_overlap),
        "joined_rows": joined,
        "correlations": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    OUT_MD.write_text(render_md(payload))
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
