#!/usr/bin/env python3
"""CPU-only gate for cross-latent signals as deployable Track A sources.

This script tests whether existing cross-latent internal-val anchor outputs can
predict when the xverse anchor should be trusted over the train-only gene
baseline.  It reads train-only/internal-val artifacts only.  It does not read
canonical test, canonical multi, or Track C query outputs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
XVERSE_JSON = ROOT / "reports/latentfm_xverse_tracka_residual_forensics_20260622.json"
CROSSLATENT_JSONS = {
    "stack": ROOT / "reports/latentfm_crosslatent_stack_tracka_anchor_internal_val_20260622.json",
    "scfoundation": ROOT
    / "reports/latentfm_crosslatent_scfoundation_tracka_anchor_internal_val_20260622.json",
    "scldm": ROOT / "reports/latentfm_crosslatent_scldm_tracka_anchor_internal_val_20260622.json",
}
OUT_JSON = ROOT / "reports/latentfm_xverse_crosslatent_deployable_source_gate_20260622.json"
OUT_MD = ROOT / "reports/LATENTFM_XVERSE_CROSSLATENT_DEPLOYABLE_SOURCE_GATE_20260622.md"
GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
BOOT_N = 2000
SEED = 20260622
RIDGE_ALPHA = 1.0


def key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("group")),
        str(row.get("dataset")),
        str(row.get("condition")),
        str(row.get("gene")),
    )


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        payload = json.load(handle)
    rows = payload.get("condition_rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path} lacks condition_rows")
    return rows


def as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return None
    xr = rankdata_average(x[mask])
    yr = rankdata_average(y[mask])
    if float(np.std(xr)) <= 1e-12 or float(np.std(yr)) <= 1e-12:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def r2_score(y: np.ndarray, pred: np.ndarray) -> float | None:
    mask = np.isfinite(y) & np.isfinite(pred)
    if int(mask.sum()) < 3:
        return None
    yy = y[mask]
    pp = pred[mask]
    denom = float(np.sum((yy - yy.mean()) ** 2))
    if denom <= 1e-12:
        return None
    return float(1.0 - np.sum((yy - pp) ** 2) / denom)


def equal_dataset_mean(values: np.ndarray, datasets: list[str]) -> float:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for value, ds in zip(values, datasets):
        if np.isfinite(value):
            by_ds[str(ds)].append(float(value))
    return float(np.mean([np.mean(vals) for _, vals in sorted(by_ds.items()) if vals]))


def paired_bootstrap(
    candidate: np.ndarray, baseline: np.ndarray, datasets: list[str], *, seed: int
) -> dict[str, Any]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for cand, base, ds in zip(candidate, baseline, datasets):
        if np.isfinite(cand) and np.isfinite(base):
            by_ds[str(ds)].append(float(cand) - float(base))
    keys = sorted(ds for ds, vals in by_ds.items() if vals)
    point_by_ds = {ds: float(np.mean(by_ds[ds])) for ds in keys}
    point = float(np.mean(list(point_by_ds.values()))) if point_by_ds else float("nan")
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(BOOT_N):
        sampled_ds = rng.choice(keys, size=len(keys), replace=True)
        boot.append(float(np.mean([np.mean(rng.choice(by_ds[str(ds)], size=len(by_ds[str(ds)]), replace=True)) for ds in sampled_ds])))
    arr = np.asarray(boot, dtype=float)
    return {
        "delta": point,
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_improve": float(np.mean(arr > 0.0)),
        "p_harm": float(np.mean(arr < 0.0)),
        "by_dataset": point_by_ds,
    }


def build_table() -> tuple[list[dict[str, Any]], list[str]]:
    xverse = load_rows(XVERSE_JSON)
    cross = {name: {key(row): row for row in load_rows(path)} for name, path in CROSSLATENT_JSONS.items()}
    rows: list[dict[str, Any]] = []
    for row in xverse:
        k = key(row)
        if any(k not in table for table in cross.values()):
            continue
        out = {
            "group": str(row["group"]),
            "dataset": str(row["dataset"]),
            "condition": str(row["condition"]),
            "gene": str(row["gene"]),
            "xverse_anchor": as_float(row.get("anchor_pearson_pert")),
            "xverse_mmd": as_float(row.get("anchor_mmd_clamped")),
            "gene_raw_mean": as_float(row.get("gene_raw_mean")),
            "dataset_mean": as_float(row.get("dataset_mean")),
            "global_mean": as_float(row.get("global_mean")),
            "target_anchor_minus_gene": as_float(row.get("anchor_minus_gene_raw_mean")),
        }
        for name, table in cross.items():
            crow = table[k]
            out[f"{name}_anchor"] = as_float(crow.get("anchor_pearson_pert"))
            out[f"{name}_mmd"] = as_float(crow.get("anchor_mmd_clamped"))
            out[f"{name}_minus_gene"] = as_float(crow.get("anchor_minus_gene_raw_mean"))
            out[f"{name}_minus_dataset"] = as_float(crow.get("anchor_minus_dataset_mean"))
        anchors = np.asarray([out[f"{name}_anchor"] for name in CROSSLATENT_JSONS], dtype=float)
        mmds = np.asarray([out[f"{name}_mmd"] for name in CROSSLATENT_JSONS], dtype=float)
        out["cross_anchor_mean"] = float(np.nanmean(anchors))
        out["cross_anchor_max"] = float(np.nanmax(anchors))
        out["cross_anchor_min"] = float(np.nanmin(anchors))
        out["cross_anchor_std"] = float(np.nanstd(anchors))
        out["cross_minus_xverse_mean"] = out["cross_anchor_mean"] - out["xverse_anchor"]
        out["cross_mmd_mean"] = float(np.nanmean(mmds))
        out["cross_mmd_min"] = float(np.nanmin(mmds))
        rows.append(out)
    feature_names = []
    for name in CROSSLATENT_JSONS:
        feature_names.extend(
            [
                f"{name}_anchor",
                f"{name}_mmd",
                f"{name}_minus_gene",
                f"{name}_minus_dataset",
            ]
        )
    feature_names.extend(
        [
            "cross_anchor_mean",
            "cross_anchor_max",
            "cross_anchor_min",
            "cross_anchor_std",
            "cross_minus_xverse_mean",
            "cross_mmd_mean",
            "cross_mmd_min",
            "xverse_mmd",
        ]
    )
    return rows, feature_names


def lodo_ridge(rows: list[dict[str, Any]], feature_names: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    datasets = sorted({row["dataset"] for row in rows})
    X = np.asarray([[as_float(row.get(feat)) for feat in feature_names] for row in rows], dtype=float)
    y = np.asarray([as_float(row["target_anchor_minus_gene"]) for row in rows], dtype=float)
    pred = np.full(len(rows), np.nan, dtype=float)
    folds = []
    for ds in datasets:
        test_mask = np.asarray([row["dataset"] == ds for row in rows], dtype=bool)
        train_mask = ~test_mask
        X_train = X[train_mask]
        y_train = y[train_mask]
        finite_train = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
        finite_test = np.isfinite(X[test_mask]).all(axis=1)
        usable = []
        for j in range(X_train.shape[1]):
            col = X_train[finite_train, j]
            if len(col) >= 3 and float(np.std(col)) > 1e-12:
                usable.append(j)
        fallback = float(np.nanmean(y_train))
        if len(usable) == 0 or int(finite_train.sum()) < 5:
            pred[test_mask] = fallback
            folds.append({"dataset": ds, "status": "fallback", "n_test": int(test_mask.sum())})
            continue
        Xtr = X_train[finite_train][:, usable]
        ytr = y_train[finite_train]
        mean = Xtr.mean(axis=0)
        std = Xtr.std(axis=0)
        std[std < 1e-12] = 1.0
        Z = (Xtr - mean) / std
        design = np.c_[np.ones(len(Z)), Z]
        penalty = np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        coef = np.linalg.solve(design.T @ design + RIDGE_ALPHA * penalty, design.T @ ytr)
        fold_pred = np.full(int(test_mask.sum()), fallback, dtype=float)
        if finite_test.any():
            Xte = X[test_mask][:, usable]
            Zte = (Xte[finite_test] - mean) / std
            fold_pred[finite_test] = np.c_[np.ones(int(finite_test.sum())), Zte] @ coef
        pred[test_mask] = fold_pred
        folds.append(
            {
                "dataset": ds,
                "status": "ridge",
                "n_train": int(finite_train.sum()),
                "n_test": int(test_mask.sum()),
                "n_features": int(len(usable)),
            }
        )
    return pred, {"folds": folds}


def summarize_group(rows: list[dict[str, Any]], feature_names: list[str], *, seed_offset: int) -> dict[str, Any]:
    pred, meta = lodo_ridge(rows, feature_names)
    y = np.asarray([row["target_anchor_minus_gene"] for row in rows], dtype=float)
    xverse_anchor = np.asarray([row["xverse_anchor"] for row in rows], dtype=float)
    gene = np.asarray([row["gene_raw_mean"] for row in rows], dtype=float)
    dataset = np.asarray([row["dataset_mean"] for row in rows], dtype=float)
    cross_mean = np.asarray([row["cross_anchor_mean"] for row in rows], dtype=float)
    route = np.where(pred > 0.0, xverse_anchor, gene)
    oracle = np.where(y > 0.0, xverse_anchor, gene)
    datasets = [row["dataset"] for row in rows]
    feature_tests = []
    for feat in feature_names:
        x = np.asarray([row[feat] for row in rows], dtype=float)
        feature_tests.append({"feature": feat, "spearman": spearman(x, y), "r2": r2_score(y, x)})
    feature_tests.sort(
        key=lambda item: (
            -1.0 if item["spearman"] is None else -abs(float(item["spearman"])),
            item["feature"],
        )
    )
    route_vs_gene = paired_bootstrap(route, gene, datasets, seed=SEED + seed_offset)
    route_vs_dataset = paired_bootstrap(route, dataset, datasets, seed=SEED + seed_offset + 17)
    cross_vs_gene = paired_bootstrap(cross_mean, gene, datasets, seed=SEED + seed_offset + 31)
    material_harms = [
        ds for ds, val in route_vs_gene["by_dataset"].items() if float(val) < -0.02
    ]
    return {
        "n": len(rows),
        "n_datasets": len(set(datasets)),
        "feature_names": feature_names,
        "top_feature_tests": feature_tests[:12],
        "lodo": {
            "spearman": spearman(pred, y),
            "r2": r2_score(y, pred),
            "predicted_anchor_fraction": float(np.mean(pred > 0.0)),
            "meta": meta,
        },
        "equal_dataset_scores": {
            "xverse_anchor": equal_dataset_mean(xverse_anchor, datasets),
            "gene_raw_mean": equal_dataset_mean(gene, datasets),
            "dataset_mean": equal_dataset_mean(dataset, datasets),
            "cross_anchor_mean": equal_dataset_mean(cross_mean, datasets),
            "crosslatent_routed_xverse_or_gene": equal_dataset_mean(route, datasets),
            "oracle_xverse_or_gene": equal_dataset_mean(oracle, datasets),
        },
        "paired_deltas": {
            "route_vs_gene_raw_mean": route_vs_gene,
            "route_vs_dataset_mean": route_vs_dataset,
            "cross_anchor_mean_vs_gene_raw_mean": cross_vs_gene,
        },
        "material_harm_datasets_vs_gene": material_harms,
    }


def decide(groups: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    for group in GROUPS:
        item = groups[group]
        lodo = item["lodo"]
        if (lodo["spearman"] is None or abs(float(lodo["spearman"])) < 0.35) and (
            lodo["r2"] is None or float(lodo["r2"]) < 0.15
        ):
            reasons.append(f"{group}_crosslatent_lodo_predictor_too_weak")
        delta = item["paired_deltas"]["route_vs_gene_raw_mean"]
        if float(delta["delta"]) < 0.02:
            reasons.append(f"{group}_route_not_0p02_better_than_gene_raw_mean")
        if float(delta["p_harm"]) > 0.20:
            reasons.append(f"{group}_route_harm_risk_vs_gene_raw_mean")
        if item["material_harm_datasets_vs_gene"]:
            reasons.append(f"{group}_dataset_level_material_harm_vs_gene_raw_mean")
    status = (
        "cpu_gate_pass_crosslatent_deployable_source_candidate"
        if not reasons
        else "cpu_gate_fail_close_crosslatent_deployable_source"
    )
    action = (
        "design_one_capped_tracka_gpu_smoke_with_noharm_gate"
        if not reasons
        else "do_not_launch_gpu_from_crosslatent_deployable_source"
    )
    return {"status": status, "recommended_action": action, "reasons": reasons}


def write_md(payload: dict[str, Any]) -> None:
    lines = [
        "# LatentFM xverse Cross-Latent Deployable Source Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"Recommended action: `{payload['decision']['recommended_action']}`",
        "",
        "## Provenance",
        "",
        f"- xverse train-only residual JSON: `{XVERSE_JSON}`",
        *[f"- {name} cross-latent internal-val JSON: `{path}`" for name, path in CROSSLATENT_JSONS.items()],
        "- leakage status: `train_only_internal_val_crosslatent_features_no_canonical_no_multi_no_query`",
        f"- aligned condition rows: `{payload['n_rows']}`",
        "",
        "## Gate",
        "",
        "- LODO predictor must explain xverse anchor-minus-gene risk in both groups:",
        "  `abs(Spearman) >= 0.35` or `R2 >= 0.15`.",
        "- The induced deployable route chooses xverse anchor only when predicted",
        "  anchor-minus-gene is positive.",
        "- Route must beat `gene_raw_mean` by at least `+0.02` equal-dataset pp in",
        "  both groups with `p_harm <= 0.20`.",
        "- No dataset may have route-minus-gene delta `< -0.02`.",
        "",
        "## Group Summary",
        "",
        "| group | n | LODO Spearman | LODO R2 | route score | gene score | route-gene delta | p_harm | oracle-gene delta | harmed datasets |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        item = payload["groups"][group]
        scores = item["equal_dataset_scores"]
        delta = item["paired_deltas"]["route_vs_gene_raw_mean"]
        oracle_delta = scores["oracle_xverse_or_gene"] - scores["gene_raw_mean"]
        spearman_val = item["lodo"]["spearman"]
        r2_val = item["lodo"]["r2"]
        lines.append(
            "| {group} | {n} | {sp} | {r2} | {route:+.6f} | {gene:+.6f} | {delta:+.6f} | {harm:+.6f} | {oracle:+.6f} | {harmed} |".format(
                group=group,
                n=item["n"],
                sp="NA" if spearman_val is None else f"{float(spearman_val):+.6f}",
                r2="NA" if r2_val is None else f"{float(r2_val):+.6f}",
                route=scores["crosslatent_routed_xverse_or_gene"],
                gene=scores["gene_raw_mean"],
                delta=delta["delta"],
                harm=delta["p_harm"],
                oracle=oracle_delta,
                harmed=len(item["material_harm_datasets_vs_gene"]),
            )
        )
    lines.extend(["", "## Top Feature Correlations", ""])
    for group in GROUPS:
        lines.extend(
            [
                f"### {group}",
                "",
                "| feature | Spearman to xverse anchor-minus-gene | raw R2 proxy |",
                "|---|---:|---:|",
            ]
        )
        for feat in payload["groups"][group]["top_feature_tests"][:8]:
            sp = feat["spearman"]
            r2 = feat["r2"]
            lines.append(
                "| {feature} | {sp} | {r2} |".format(
                    feature=f"`{feat['feature']}`",
                    sp="NA" if sp is None else f"{float(sp):+.6f}",
                    r2="NA" if r2 is None else f"{float(r2):+.6f}",
                )
            )
        lines.append("")
    lines.extend(["## Gate Reasons", ""])
    if payload["decision"]["reasons"]:
        lines.extend([f"- `{reason}`" for reason in payload["decision"]["reasons"]])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a CPU-only deployable-source gate. Passing would authorize only",
            "  one capped GPU smoke with a frozen route and canonical single/background",
            "  no-harm gate.",
            "- Failure closes cross-latent disagreement/anchor-score features as the next",
            "  Track A GPU unlock route.",
            "- No canonical test, canonical multi, or Track C query evidence is used.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines))


def main() -> None:
    rows, feature_names = build_table()
    groups = {}
    for i, group in enumerate(GROUPS):
        group_rows = [row for row in rows if row["group"] == group]
        groups[group] = summarize_group(group_rows, feature_names, seed_offset=i * 100)
    payload = {
        "xverse_json": str(XVERSE_JSON),
        "crosslatent_jsons": {name: str(path) for name, path in CROSSLATENT_JSONS.items()},
        "leakage_status": "train_only_internal_val_crosslatent_features_no_canonical_no_multi_no_query",
        "n_rows": len(rows),
        "feature_names": feature_names,
        "groups": groups,
    }
    payload["decision"] = decide(groups)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_md(payload)
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
