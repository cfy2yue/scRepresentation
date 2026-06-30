#!/usr/bin/env python3
"""CPU gate for a continuous multi-latent state/agreement router.

This is a query-free Track A internal-proxy gate.  It tests one predeclared
continuous multi-latent rule and fails closed if no-harm or control separation
criteria are not met.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

ROOT = Path("/data/cyx/1030/scLatent")
OUT_MD = ROOT / "reports/LATENTFM_SOFT_ARCHETYPE_MULTILATENT_STATE_CPU_GATE_20260623.md"
OUT_JSON = ROOT / "reports/latentfm_soft_archetype_multilatent_state_cpu_gate_20260623.json"
INPUT_AUDIT = ROOT / "reports/latentfm_soft_archetype_multilatent_state_input_audit_20260623.json"

LATENT_JSONS = {
    "xverse": ROOT / "reports/latentfm_xverse_tracka_anchor_internal_val_error_map_20260622.json",
    "stack": ROOT / "reports/latentfm_crosslatent_stack_tracka_anchor_internal_val_20260622.json",
    "scfoundation": ROOT / "reports/latentfm_crosslatent_scfoundation_tracka_anchor_internal_val_20260622.json",
    "scldm": ROOT / "reports/latentfm_crosslatent_scldm_tracka_anchor_internal_val_20260622.json",
}
FEATURE_LATENTS = ("stack", "scfoundation", "scldm")
GROUPS = ("internal_val_cross_background_seen_gene_proxy", "internal_val_family_gene_proxy")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("group", "")),
        str(row.get("dataset", "")),
        str(row.get("condition", "")),
        str(row.get("gene", "")),
    )


def f(row: dict[str, Any], name: str, default: float = 0.0) -> float:
    value = row.get(name)
    if value is None:
        return default
    return float(value)


def margin(row: dict[str, Any], baseline: str) -> float:
    return f(row, "anchor_pearson_pert") - f(row, baseline)


def feature_vec(indexed: dict[str, dict], row_key: tuple[str, str, str, str]) -> np.ndarray:
    values: list[float] = []
    dataset_margins = []
    gene_margins = []
    for latent in FEATURE_LATENTS:
        row = indexed[latent][row_key]
        ds_margin = margin(row, "dataset_mean")
        gene_margin = margin(row, "gene_raw_mean")
        dataset_margins.append(ds_margin)
        gene_margins.append(gene_margin)
        values.extend(
            [
                ds_margin,
                gene_margin,
                f(row, "anchor_pearson_pert"),
                f(row, "anchor_mmd_clamped"),
                f(row, "gene_train_count"),
            ]
        )
    values.extend(
        [
            float(np.mean(dataset_margins)),
            float(np.std(dataset_margins)),
            float(np.min(dataset_margins)),
            float(np.mean(gene_margins)),
            float(np.std(gene_margins)),
            float(np.min(gene_margins)),
        ]
    )
    return np.asarray(values, dtype=np.float64)


def lodo_predictions(
    indexed: dict[str, dict],
    keys: list[tuple[str, str, str, str]],
    *,
    shuffled: bool,
    seed: int = 42,
    ridge_lambda: float = 10.0,
) -> dict[tuple[str, str, str, str], float]:
    rng = np.random.RandomState(seed)
    datasets = sorted({k[1] for k in keys})
    out: dict[tuple[str, str, str, str], float] = {}
    for holdout in datasets:
        train_keys = [k for k in keys if k[1] != holdout]
        test_keys = [k for k in keys if k[1] == holdout]
        x_train_keys = list(train_keys)
        if shuffled:
            rng.shuffle(x_train_keys)
        x_train = np.vstack([feature_vec(indexed, k) for k in x_train_keys])
        y_train = np.asarray([margin(indexed["xverse"][k], "dataset_mean") for k in train_keys], dtype=np.float64)
        mu = x_train.mean(axis=0)
        sigma = x_train.std(axis=0) + 1e-6
        x_scaled = (x_train - mu) / sigma
        beta = np.linalg.solve(
            x_scaled.T @ x_scaled + ridge_lambda * np.eye(x_scaled.shape[1]),
            x_scaled.T @ y_train,
        )
        for k in test_keys:
            out[k] = float(((feature_vec(indexed, k) - mu) / sigma) @ beta)
    return out


def evaluate_policy(
    indexed: dict[str, dict],
    keys: list[tuple[str, str, str, str]],
    predictions: dict[tuple[str, str, str, str], float],
) -> list[dict[str, Any]]:
    rows = []
    for k in keys:
        xverse = indexed["xverse"][k]
        other_ds = [margin(indexed[latent][k], "dataset_mean") for latent in FEATURE_LATENTS]
        predicted_gain = float(predictions[k])
        gate = (
            predicted_gain >= 0.02
            and min(other_ds) >= -0.05
            and float(np.std(other_ds)) <= 0.35
        )
        candidate = f(xverse, "dataset_mean") + (f(xverse, "anchor_pearson_pert") - f(xverse, "dataset_mean")) * float(gate)
        rows.append(
            {
                "group": k[0],
                "dataset": k[1],
                "condition": k[2],
                "gene": k[3],
                "gate": bool(gate),
                "predicted_gain": predicted_gain,
                "other_dataset_margin_min": float(min(other_ds)),
                "other_dataset_margin_std": float(np.std(other_ds)),
                "candidate": float(candidate),
                "dataset_mean": f(xverse, "dataset_mean"),
                "gene_raw_mean": f(xverse, "gene_raw_mean"),
                "xverse_anchor": f(xverse, "anchor_pearson_pert"),
                "delta_vs_dataset_mean": float(candidate - f(xverse, "dataset_mean")),
                "delta_vs_gene_raw_mean": float(candidate - f(xverse, "gene_raw_mean")),
            }
        )
    return rows


def bootstrap(rows: list[dict[str, Any]], field: str, *, seed: int = 42, n_boot: int = 2000) -> dict[str, Any]:
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(float(row[field]))
    datasets = sorted(by_dataset)
    ds_means = np.asarray([mean(by_dataset[ds]) for ds in datasets], dtype=np.float64)
    rng = np.random.RandomState(seed)
    samples = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(ds_means), size=len(ds_means))
        samples.append(float(np.mean(ds_means[idx])))
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "delta_mean": float(np.mean(ds_means)),
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_improve": float(np.mean(arr > 0.0)),
        "p_harm": float(np.mean(arr < 0.0)),
        "leave_one_min": float(min(np.mean(np.delete(ds_means, i)) for i in range(len(ds_means)))) if len(ds_means) > 1 else float(ds_means[0]),
        "dataset_min": float(np.min(ds_means)),
        "n_datasets": int(len(ds_means)),
        "n_conditions": int(len(rows)),
    }


def summarize_group(rows: list[dict[str, Any]], shuffled_rows: list[dict[str, Any]]) -> dict[str, Any]:
    real_dataset = bootstrap(rows, "delta_vs_dataset_mean")
    real_gene = bootstrap(rows, "delta_vs_gene_raw_mean")
    shuf_dataset = bootstrap(shuffled_rows, "delta_vs_dataset_mean")
    shuf_gene = bootstrap(shuffled_rows, "delta_vs_gene_raw_mean")
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row["delta_vs_dataset_mean"])
    worst = sorted(
        (
            {
                "dataset": ds,
                "n": len(vals),
                "delta_vs_dataset_mean": float(mean(vals)),
                "used_fraction": float(np.mean([r["gate"] for r in rows if r["dataset"] == ds])),
            }
            for ds, vals in by_dataset.items()
        ),
        key=lambda x: x["delta_vs_dataset_mean"],
    )[:8]
    reasons = []
    if real_dataset["delta_mean"] < 0.02:
        reasons.append("dataset_mean_delta_below_0p02")
    if real_dataset["p_harm"] > 0.20:
        reasons.append("dataset_mean_p_harm_above_0p20")
    if real_dataset["dataset_min"] < -0.02:
        reasons.append("dataset_mean_dataset_min_below_minus_0p02")
    if real_gene["delta_mean"] < 0.02:
        reasons.append("gene_raw_mean_delta_below_0p02")
    if real_gene["p_harm"] > 0.20:
        reasons.append("gene_raw_mean_p_harm_above_0p20")
    if real_dataset["delta_mean"] - shuf_dataset["delta_mean"] < 0.02:
        reasons.append("shuffled_control_not_separated_vs_dataset")
    if real_gene["delta_mean"] - shuf_gene["delta_mean"] < 0.02:
        reasons.append("shuffled_control_not_separated_vs_gene")
    return {
        "n_rows": len(rows),
        "coverage_fraction": float(np.mean([row["gate"] for row in rows])),
        "real_vs_dataset_mean": real_dataset,
        "real_vs_gene_raw_mean": real_gene,
        "shuffled_vs_dataset_mean": shuf_dataset,
        "shuffled_vs_gene_raw_mean": shuf_gene,
        "shuffled_separation_dataset": float(real_dataset["delta_mean"] - shuf_dataset["delta_mean"]),
        "shuffled_separation_gene": float(real_gene["delta_mean"] - shuf_gene["delta_mean"]),
        "worst_datasets": worst,
        "reasons": reasons,
        "status": "pass" if not reasons else "fail",
    }


def main() -> int:
    input_audit = load_json(INPUT_AUDIT)
    indexed = {
        latent: {key(row): row for row in load_json(path).get("condition_rows") or []}
        for latent, path in LATENT_JSONS.items()
    }
    common_keys = sorted(set.intersection(*(set(rows) for rows in indexed.values())))
    group_summaries: dict[str, Any] = {}
    for group in GROUPS:
        group_keys = [k for k in common_keys if k[0] == group]
        predictions = lodo_predictions(indexed, group_keys, shuffled=False)
        shuffled_predictions = lodo_predictions(indexed, group_keys, shuffled=True)
        rows = evaluate_policy(indexed, group_keys, predictions)
        shuffled_rows = evaluate_policy(indexed, group_keys, shuffled_predictions)
        group_summaries[group] = summarize_group(rows, shuffled_rows)

    failed_reasons = []
    if input_audit.get("status") != "soft_archetype_multilatent_state_inputs_ready_cpu_gate_next_no_gpu":
        failed_reasons.append("input_audit_not_ready")
    for group, summary in group_summaries.items():
        failed_reasons.extend(f"{group}:{reason}" for reason in summary["reasons"])
    status = (
        "soft_archetype_multilatent_state_cpu_gate_pass_code_gate_next_no_gpu"
        if not failed_reasons
        else "soft_archetype_multilatent_state_cpu_gate_fail_no_gpu"
    )
    payload = {
        "status": status,
        "gpu_authorization": "none",
        "next_authorization": "code_gate_only" if not failed_reasons else "none",
        "leakage_status": "trainonly_internal_proxy_no_canonical_no_multi_no_query_no_active_gpu_artifacts",
        "rule": {
            "name": "lodo_ridge_multilatent_agreement_abstain",
            "prediction_target": "xverse_anchor_minus_dataset_mean",
            "gate": "predicted_gain>=0.02 and min(other_latent_dataset_margins)>=-0.05 and std(other_latent_dataset_margins)<=0.35",
            "feature_latents": list(FEATURE_LATENTS),
            "ridge_lambda": 10.0,
            "shuffled_control": "permute feature rows inside each LODO training fold with fixed seed 42",
        },
        "groups": group_summaries,
        "failed_reasons": failed_reasons,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM Soft-Archetype Multi-Latent State CPU Gate",
        "",
        f"Status: `{status}`",
        "GPU authorization: `none`",
        f"Next authorization: `{payload['next_authorization']}`",
        "",
        "## Rule",
        "",
        "- LODO ridge predicts xverse anchor margin over dataset_mean from stack/scFoundation/SCLDM continuous internal-val features.",
        "- Gate opens only when predicted gain is at least `+0.02`, all other-latent dataset margins are at least `-0.05`, and other-latent disagreement std is at most `0.35`.",
        "- Candidate abstains to dataset_mean when the gate is off.",
        "- Shuffled control permutes feature rows inside each LODO training fold.",
        "",
        "## Group Results",
        "",
        "| group | status | coverage | delta vs dataset | p_harm dataset | dataset min | delta vs gene | p_harm gene | shuffled sep dataset | shuffled sep gene |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, summary in group_summaries.items():
        ds = summary["real_vs_dataset_mean"]
        gene = summary["real_vs_gene_raw_mean"]
        lines.append(
            f"| `{group}` | `{summary['status']}` | {summary['coverage_fraction']:.3f} | "
            f"{ds['delta_mean']:+.6f} | {ds['p_harm']:.6f} | {ds['dataset_min']:+.6f} | "
            f"{gene['delta_mean']:+.6f} | {gene['p_harm']:.6f} | "
            f"{summary['shuffled_separation_dataset']:+.6f} | {summary['shuffled_separation_gene']:+.6f} |"
        )
    lines.extend(["", "## Worst Dataset Effects", ""])
    for group, summary in group_summaries.items():
        lines.append(f"### {group}")
        lines.append("")
        lines.append("| dataset | n | used frac | delta vs dataset_mean |")
        lines.append("|---|---:|---:|---:|")
        for row in summary["worst_datasets"]:
            lines.append(
                f"| `{row['dataset']}` | {row['n']} | {row['used_fraction']:.3f} | {row['delta_vs_dataset_mean']:+.6f} |"
            )
        lines.append("")
    lines.extend(["## Failed Reasons", ""])
    lines.extend([f"- `{reason}`" for reason in failed_reasons] or ["- none"])
    lines.extend(
        [
            "",
            "## Decision Boundary",
            "",
            "This gate is CPU-only and query-free.  Failure keeps the current continuous multi-latent state rule diagnostic-only; it does not close all future state-prior ideas, but it blocks GPU for this rule.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
