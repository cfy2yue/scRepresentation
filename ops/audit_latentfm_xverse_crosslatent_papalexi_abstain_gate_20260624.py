#!/usr/bin/env python3
"""CPU-only guard for the cross-latent source router Papalexi harm mode.

This reads only the already-built train-only/internal-val cross-latent source
features. It does not read canonical test, canonical multi, Track C query, or
posthoc held-out outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from audit_latentfm_xverse_crosslatent_deployable_source_gate_20260622 import (
    GROUPS,
    SEED,
    build_table,
    equal_dataset_mean,
    lodo_ridge,
    paired_bootstrap,
    r2_score,
    spearman,
)


ROOT = Path("/data/cyx/1030/scLatent")
OUT_JSON = ROOT / "reports/latentfm_xverse_crosslatent_papalexi_abstain_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_XVERSE_CROSSLATENT_PAPALEXI_ABSTAIN_GATE_20260624.md"
GUARD_DATASETS = ("Papalexi",)


def _f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _dataset_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        ds = str(row["dataset"])
        out[ds] = out.get(ds, 0) + 1
    return out


def _score_group(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    *,
    seed_offset: int,
) -> dict[str, Any]:
    pred, meta = lodo_ridge(rows, feature_names)
    xverse = np.asarray([_f(row["xverse_anchor"]) for row in rows], dtype=float)
    gene = np.asarray([_f(row["gene_raw_mean"]) for row in rows], dtype=float)
    y = np.asarray([_f(row["target_anchor_minus_gene"]) for row in rows], dtype=float)
    datasets = [str(row["dataset"]) for row in rows]
    route = np.where(pred > 0.0, xverse, gene)
    guarded = np.asarray(
        [gene[i] if datasets[i] in GUARD_DATASETS else route[i] for i in range(len(rows))],
        dtype=float,
    )
    route_delta = paired_bootstrap(route, gene, datasets, seed=SEED + seed_offset)
    guarded_delta = paired_bootstrap(guarded, gene, datasets, seed=SEED + seed_offset + 17)
    guarded_harms = {
        ds: val for ds, val in guarded_delta["by_dataset"].items() if float(val) < -0.02
    }
    original_harms = {
        ds: val for ds, val in route_delta["by_dataset"].items() if float(val) < -0.02
    }
    counts = _dataset_counts(rows)
    guard_counts = {ds: counts.get(ds, 0) for ds in GUARD_DATASETS}
    return {
        "n": len(rows),
        "n_datasets": len(set(datasets)),
        "guard_datasets": list(GUARD_DATASETS),
        "guard_counts": guard_counts,
        "lodo": {
            "spearman": spearman(pred, y),
            "r2": r2_score(y, pred),
            "predicted_anchor_fraction": float(np.mean(pred > 0.0)),
            "meta": meta,
        },
        "equal_dataset_scores": {
            "gene_raw_mean": equal_dataset_mean(gene, datasets),
            "route": equal_dataset_mean(route, datasets),
            "papalexi_abstain_route": equal_dataset_mean(guarded, datasets),
        },
        "paired_deltas": {
            "route_vs_gene_raw_mean": route_delta,
            "papalexi_abstain_route_vs_gene_raw_mean": guarded_delta,
        },
        "original_material_harms_vs_gene": original_harms,
        "guarded_material_harms_vs_gene": guarded_harms,
    }


def _decide(groups: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    for group in GROUPS:
        item = groups[group]
        lodo = item["lodo"]
        if (lodo["spearman"] is None or abs(float(lodo["spearman"])) < 0.35) and (
            lodo["r2"] is None or float(lodo["r2"]) < 0.15
        ):
            reasons.append(f"{group}_lodo_predictor_too_weak")
        delta = item["paired_deltas"]["papalexi_abstain_route_vs_gene_raw_mean"]
        if float(delta["delta"]) < 0.02:
            reasons.append(f"{group}_guarded_route_not_0p02_better_than_gene")
        if float(delta["p_harm"]) > 0.20:
            reasons.append(f"{group}_guarded_route_harm_risk")
        if item["guarded_material_harms_vs_gene"]:
            reasons.append(f"{group}_guarded_route_dataset_material_harm")
        for ds, n in item["guard_counts"].items():
            if int(n) < 3:
                reasons.append(f"{group}_{ds}_guard_support_lt_3")
        original_harms = set(item["original_material_harms_vs_gene"])
        if not set(GUARD_DATASETS).issubset(original_harms):
            reasons.append(f"{group}_guard_dataset_not_original_trainonly_harm")
    status = (
        "papalexi_abstain_gate_pass_protocol_only_no_gpu"
        if not reasons
        else "papalexi_abstain_gate_partial_or_fail_no_gpu"
    )
    action = (
        "freeze_route_protocol_then_consider_no_training_canonical_posthoc"
        if not reasons
        else "do_not_launch_gpu_or_canonical_posthoc_from_papalexi_abstain"
    )
    return {"status": status, "recommended_action": action, "reasons": reasons}


def _fmt(value: Any) -> str:
    val = _f(value)
    if not np.isfinite(val):
        return "NA"
    return f"{val:+.6f}"


def _write_md(payload: dict[str, Any]) -> None:
    lines = [
        "# LatentFM xverse Cross-Latent Papalexi Abstain Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"Recommended action: `{payload['decision']['recommended_action']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only train-only/internal-val source-router guard.",
        "- No canonical test, canonical multi, Track C query, active logs, or held-out posthoc outcomes are read.",
        "- Tests whether the only old source-router material harm dataset can be safely abstained to `gene_raw_mean`.",
        "",
        "## Rule",
        "",
        "`crosslatent_source_route`: choose xverse anchor when LODO predicted anchor-minus-gene is positive, otherwise `gene_raw_mean`.",
        "",
        "`papalexi_abstain_route`: same route, except Papalexi always uses `gene_raw_mean`.",
        "",
        "## Group Summary",
        "",
        "| group | n | Papalexi n | LODO Spearman | LODO R2 | route-gene delta | guarded-gene delta | guarded p_harm | guarded harms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for group in GROUPS:
        item = payload["groups"][group]
        lodo = item["lodo"]
        route = item["paired_deltas"]["route_vs_gene_raw_mean"]
        guarded = item["paired_deltas"]["papalexi_abstain_route_vs_gene_raw_mean"]
        lines.append(
            "| {group} | {n} | {pn} | {sp} | {r2} | {rd} | {gd} | {ph} | {harms} |".format(
                group=group,
                n=item["n"],
                pn=item["guard_counts"].get("Papalexi", 0),
                sp=_fmt(lodo["spearman"]),
                r2=_fmt(lodo["r2"]),
                rd=_fmt(route["delta"]),
                gd=_fmt(guarded["delta"]),
                ph=_fmt(guarded["p_harm"]),
                harms=", ".join(item["guarded_material_harms_vs_gene"]) or "none",
            )
        )
    lines.extend(["", "## Gate Reasons", ""])
    if payload["decision"]["reasons"]:
        lines.extend([f"- `{reason}`" for reason in payload["decision"]["reasons"]])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Numeric rescue of the old source-router harm is not enough if the guard dataset has too few train-only proxy rows.",
            "- A fail/partial result keeps this branch closed for GPU and held-out canonical posthoc.",
            "- A full pass would authorize only a frozen no-training route protocol, not model-training promotion.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows, feature_names = build_table()
    groups = {
        group: _score_group(
            [row for row in rows if row["group"] == group],
            feature_names,
            seed_offset=i * 100,
        )
        for i, group in enumerate(GROUPS)
    }
    payload = {
        "leakage_status": "train_only_internal_val_crosslatent_features_no_canonical_no_multi_no_query",
        "source_gate": str(ROOT / "reports/latentfm_xverse_crosslatent_deployable_source_gate_20260622.json"),
        "guard_datasets": list(GUARD_DATASETS),
        "groups": groups,
    }
    payload["decision"] = _decide(groups)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_md(payload)
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
