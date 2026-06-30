#!/usr/bin/env python3
"""Association gate using train-only condition/residual information metrics.

CPU/report-only. This joins frozen downstream scaling outcomes with the
condition-level train-support geometry materialized from xVERSE H5 bundles.
It does not train, infer, use canonical multi, use Track C query, or select
checkpoints.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "reports/downstream_condition_residual_association_gate_20260628"
OUT_MD = ROOT / "reports/LATENTFM_DOWNSTREAM_CONDITION_RESIDUAL_ASSOCIATION_GATE_20260628.md"
OUT_JSON = ROOT / "reports/latentfm_downstream_condition_residual_association_gate_20260628.json"
OUT_JOIN = OUT_DIR / "condition_residual_information_outcome_join_rows.csv"
OUT_ASSOC = OUT_DIR / "condition_residual_association_rows.csv"
OUT_PAIRS = OUT_DIR / "condition_residual_matched_pair_candidates.csv"

INFO_CSV = ROOT / "reports/downstream_information_scaling_preflight_20260628/split_information_metrics.csv"
COND_CSV = ROOT / "reports/trainonly_condition_residual_information_20260628/trainonly_condition_residual_information_rows.csv"
CONDITION_OUTCOME_CSV = ROOT / "reports/scaling_figure_data_20260625/condition_exposure_curve.csv"
TRUECELL_OUTCOME_CSV = ROOT / "reports/scaling_figure_data_20260625/truecell_budget_curve.csv"

BASE_METRIC_TAGS = {
    "n_train_conditions": "technical",
    "dataset_effective_count": "mixed",
    "background_effective_count": "mixed",
    "perturbation_type_effective_count": "mixed",
    "target_gene_effective_count": "biological",
    "drug_condition_fraction": "mixed",
    "gene_condition_fraction": "mixed",
}

CONDITION_METRIC_TAGS = {
    "n_train_conditions_with_vectors": "technical",
    "dataset_effective_count_condition_vectors": "mixed",
    "perturbation_type_effective_count_condition_vectors": "mixed",
    "target_gene_effective_count_condition_vectors": "biological",
    "response_norm_mean": "latent_bio_mixed",
    "response_norm_cv": "latent_bio_mixed",
    "residual_effective_rank": "latent_bio_mixed",
    "residual_rank_entropy_norm": "latent_bio_mixed",
    "residual_pairwise_l2_mean": "latent_bio_mixed",
    "residual_pairwise_cosine_distance_mean": "latent_bio_mixed",
    "residual_vendi_rbf_effective_count": "latent_bio_mixed",
    "ctrl_center_effective_rank": "technical_latent",
    "gt_center_effective_rank": "latent_bio_mixed",
}

METRIC_TAGS = {**BASE_METRIC_TAGS, **CONDITION_METRIC_TAGS}
PREDICTORS = list(METRIC_TAGS)
OUTCOMES = ["cross_pp_delta", "family_pp_delta", "family_mmd_delta", "tail_score"]
CONFOUNDS = [
    "n_train_conditions",
    "dataset_effective_count",
    "background_effective_count",
    "perturbation_type_effective_count",
    "drug_condition_fraction",
]
PAIR_AXES = [
    "residual_vendi_rbf_effective_count",
    "residual_effective_rank",
    "response_norm_mean",
    "residual_pairwise_l2_mean",
    "target_gene_effective_count_condition_vectors",
    "perturbation_type_effective_count_condition_vectors",
    "dataset_effective_count_condition_vectors",
]
PAIR_CONFOUNDS = [
    "n_train_conditions_with_vectors",
    "dataset_effective_count_condition_vectors",
    "perturbation_type_effective_count_condition_vectors",
    "target_gene_effective_count_condition_vectors",
    "drug_condition_fraction",
    "gene_condition_fraction",
]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: Any) -> float:
    if value in ("", None):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    out = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        out[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return out


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    x = x.astype(float) - float(np.mean(x))
    y = y.astype(float) - float(np.mean(y))
    denom = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denom <= 0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(ranks(x), ranks(y))


def permutation_p(x: np.ndarray, y: np.ndarray, observed: float, n_perm: int = 5000) -> float:
    if len(x) < 5 or not np.isfinite(observed):
        return float("nan")
    rng = np.random.default_rng(42)
    extreme = 1
    for _ in range(n_perm):
        val = spearman(x, rng.permutation(y))
        if np.isfinite(val) and abs(val) >= abs(observed):
            extreme += 1
    return extreme / (n_perm + 1)


def residualize(v: np.ndarray, confounds: np.ndarray) -> np.ndarray:
    if confounds.size == 0:
        return v - np.mean(v)
    x = np.column_stack([np.ones(len(v)), confounds])
    coef, *_ = np.linalg.lstsq(x, v, rcond=None)
    return v - x @ coef


def partial_corr(x: np.ndarray, y: np.ndarray, confounds: np.ndarray) -> float:
    if len(x) < 5:
        return float("nan")
    return pearson(residualize(x, confounds), residualize(y, confounds))


def loo_stability(x: np.ndarray, y: np.ndarray, observed: float) -> float:
    if len(x) < 5 or not np.isfinite(observed) or observed == 0:
        return float("nan")
    ok = 0
    total = 0
    for i in range(len(x)):
        mask = np.ones(len(x), dtype=bool)
        mask[i] = False
        val = spearman(x[mask], y[mask])
        if np.isfinite(val):
            total += 1
            ok += int(math.copysign(1.0, val) == math.copysign(1.0, observed))
    return ok / total if total else float("nan")


def finite_matrix(rows: list[dict[str, Any]], keys: list[str]) -> np.ndarray:
    return np.asarray([[to_float(row.get(key)) for key in keys] for row in rows], dtype=float)


def mean_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {}
    out: dict[str, Any] = {}
    keys = rows[0].keys()
    for key in keys:
        vals = [to_float(row.get(key)) for row in rows]
        if all(np.isfinite(v) for v in vals):
            out[key] = float(np.mean(vals))
        else:
            out[key] = rows[0].get(key, "")
    return out


def select_row(rows: list[dict[str, str]], contains: str) -> dict[str, str] | None:
    matches = [row for row in rows if contains in row["split_name"]]
    if not matches:
        return None
    v2 = [row for row in matches if row["split_name"].endswith("_v2")]
    return sorted(v2 or matches, key=lambda r: (len(r["split_name"]), r["split_name"]))[0]


def condition_arm_row(rows: list[dict[str, str]], arm: str) -> dict[str, Any] | None:
    mapping = {
        "cap30_all": "scaling_cap30_all",
        "cap120_all": "scaling_cap120_all",
        "gene_cap120_allbg": "gene_cap120_allbg",
        "gene_cap120_k562bg": "gene_cap120_k562bg",
        "type_balanced_cap120": "type_balanced_cap120",
        "jiang_exposure_capped": "jiang_exposure_capped",
        "general_exposure_cap_v2": "general_exposure_cap_v2",
        "breadth_few_deep_4ds_cap120_budget480": "breadth_few_deep_4ds_cap120_budget480",
        "breadth_mid_8ds_cap60_budget480": "breadth_mid_8ds_cap60_budget480",
        "breadth_many_shallow_19ds_cap30_budget480": "breadth_many_shallow_19ds_cap30_budget480",
        "cap60_primary19": "cap60_primary19",
    }
    if arm == "full_trainonly":
        return select_row(rows, "scaling_cap120_all")
    return select_row(rows, mapping.get(arm, arm))


def truecell_row(rows: list[dict[str, str]], budget: int) -> dict[str, Any] | None:
    matches = [
        row
        for row in rows
        if "xverse_true_cell_count_scaling_nested_splits_20260624" in row["split_file"]
        and re.search(rf"_budget{budget}_seed\d+$", row["split_name"])
    ]
    return mean_rows(matches) if matches else None


def merge_metrics(base: dict[str, Any], cond: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(base)
    if cond:
        for key in CONDITION_METRIC_TAGS:
            out[key] = to_float(cond.get(key))
    return out


def add_predictors(out: dict[str, Any], metrics: dict[str, Any]) -> None:
    for key in PREDICTORS:
        out[key] = to_float(metrics.get(key))


def build_join_rows() -> list[dict[str, Any]]:
    base_rows = load_csv(INFO_CSV)
    cond_rows = load_csv(COND_CSV)
    cond_by_split = {row["split_name"]: row for row in cond_rows}
    joined: list[dict[str, Any]] = []

    for outcome in load_csv(CONDITION_OUTCOME_CSV):
        base = condition_arm_row(base_rows, outcome["arm"])
        if base is None:
            continue
        cond = cond_by_split.get(base["split_name"])
        metrics = merge_metrics(base, cond)
        out = {
            "source_family": outcome["source_family"],
            "axis_family": "condition_exposure",
            "arm": outcome["arm"],
            "split_name": base.get("split_name", ""),
            "role": outcome.get("role", ""),
            "status": outcome.get("status", ""),
            "cross_pp_delta": to_float(outcome.get("cross_pp_delta")),
            "family_pp_delta": to_float(outcome.get("family_pp_delta")),
            "family_mmd_delta": to_float(outcome.get("family_mmd_delta")),
            "tail_score": -1.0 if to_float(outcome.get("family_mmd_delta")) > 0.001 else 0.0,
            "weak_proxy_info": outcome["arm"] == "full_trainonly",
        }
        add_predictors(out, metrics)
        joined.append(out)

    for outcome in load_csv(TRUECELL_OUTCOME_CSV):
        base = truecell_row(base_rows, int(float(outcome["budget"])))
        if base is None:
            continue
        cond = cond_by_split.get(base["split_name"])
        metrics = merge_metrics(base, cond)
        out = {
            "source_family": "truecell_budget_curve",
            "axis_family": "true_cell_count",
            "arm": f"{outcome['series']}_budget{outcome['budget']}",
            "split_name": base.get("split_name", f"mean_truecell_budget{outcome['budget']}"),
            "role": outcome["series"],
            "status": outcome.get("status", ""),
            "cross_pp_delta": to_float(outcome.get("cross_pp_mean")),
            "family_pp_delta": to_float(outcome.get("family_pp_mean")),
            "family_mmd_delta": to_float(outcome.get("family_mmd_mean")),
            "tail_score": -to_float(outcome.get("cross_pp_negative_tails")),
            "weak_proxy_info": False,
        }
        add_predictors(out, metrics)
        joined.append(out)
    return joined


def association_rows(join_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for axis_filter in ["all", "condition_exposure", "true_cell_count"]:
        subset = [row for row in join_rows if axis_filter == "all" or row["axis_family"] == axis_filter]
        for outcome in OUTCOMES:
            y = np.asarray([to_float(row.get(outcome)) for row in subset], dtype=float)
            for predictor in PREDICTORS:
                x = np.asarray([to_float(row.get(predictor)) for row in subset], dtype=float)
                confound_keys = [key for key in CONFOUNDS if key != predictor]
                c = finite_matrix(subset, confound_keys)
                mask = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(c), axis=1)
                if int(mask.sum()) < 5:
                    continue
                xm = x[mask]
                ym = y[mask]
                cm = c[mask]
                rho = spearman(xm, ym)
                part = partial_corr(xm, ym, cm)
                sign_consistent = (
                    np.isfinite(rho)
                    and np.isfinite(part)
                    and rho != 0
                    and part != 0
                    and math.copysign(1.0, rho) == math.copysign(1.0, part)
                )
                tag = METRIC_TAGS[predictor]
                row = {
                    "axis_filter": axis_filter,
                    "outcome": outcome,
                    "predictor": predictor,
                    "metric_tag": tag,
                    "n": int(mask.sum()),
                    "spearman_rho": rho,
                    "spearman_perm_p": permutation_p(xm, ym, rho),
                    "partial_corr": part,
                    "partial_sign_consistent": sign_consistent,
                    "loo_sign_stability": loo_stability(xm, ym, rho),
                }
                row["gate_signal"] = (
                    tag in {"biological", "mixed", "latent_bio_mixed"}
                    and np.isfinite(row["spearman_rho"])
                    and abs(row["spearman_rho"]) >= 0.55
                    and np.isfinite(row["spearman_perm_p"])
                    and row["spearman_perm_p"] <= 0.05
                    and np.isfinite(row["partial_corr"])
                    and abs(row["partial_corr"]) >= 0.35
                    and sign_consistent
                    and np.isfinite(row["loo_sign_stability"])
                    and row["loo_sign_stability"] >= 0.8
                )
                rows.append(row)
    return rows


def zscores(rows: list[dict[str, str]], keys: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {row["split_name"]: {} for row in rows}
    for key in keys:
        vals = np.asarray([to_float(row.get(key)) for row in rows], dtype=float)
        mask = np.isfinite(vals)
        if mask.sum() < 2:
            continue
        mean = float(vals[mask].mean())
        std = float(vals[mask].std())
        if std <= 1e-12:
            std = 1.0
        for row, val in zip(rows, vals, strict=True):
            out[row["split_name"]][key] = float((val - mean) / std) if np.isfinite(val) else float("nan")
    return out


def matched_pair_candidates() -> list[dict[str, Any]]:
    rows = load_csv(COND_CSV)
    keys = sorted(set(PAIR_AXES + PAIR_CONFOUNDS))
    z = zscores(rows, keys)
    out: list[dict[str, Any]] = []
    for axis in PAIR_AXES:
        confounds = [key for key in PAIR_CONFOUNDS if key != axis]
        for i, left in enumerate(rows):
            for right in rows[i + 1 :]:
                lz = z.get(left["split_name"], {})
                rz = z.get(right["split_name"], {})
                axis_delta = abs(lz.get(axis, float("nan")) - rz.get(axis, float("nan")))
                if not np.isfinite(axis_delta):
                    continue
                conf_deltas = [abs(lz.get(key, float("nan")) - rz.get(key, float("nan"))) for key in confounds]
                conf_deltas = [v for v in conf_deltas if np.isfinite(v)]
                if not conf_deltas:
                    continue
                max_conf = max(conf_deltas)
                mean_conf = float(np.mean(conf_deltas))
                candidate = axis_delta >= 0.65 and max_conf <= 1.25 and mean_conf <= 0.75
                out.append(
                    {
                        "axis": axis,
                        "left_split": left["split_name"],
                        "right_split": right["split_name"],
                        "axis_delta_z": axis_delta,
                        "max_confound_z": max_conf,
                        "mean_confound_z": mean_conf,
                        "candidate_equal_cell_or_matched_info": candidate,
                        "left_axis_value": to_float(left.get(axis)),
                        "right_axis_value": to_float(right.get(axis)),
                        "left_n_train_vectors": to_float(left.get("n_train_conditions_with_vectors")),
                        "right_n_train_vectors": to_float(right.get("n_train_conditions_with_vectors")),
                    }
                )
    return sorted(out, key=lambda r: (not r["candidate_equal_cell_or_matched_info"], -r["axis_delta_z"], r["max_confound_z"]))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def fmt(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return f"{float(value):+.4f}"
    return str(value)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    join_rows = build_join_rows()
    assoc = association_rows(join_rows)
    pairs = matched_pair_candidates()
    signals = [row for row in assoc if row["gate_signal"]]
    latent_signals = [row for row in signals if row["metric_tag"] == "latent_bio_mixed"]
    bio_signals = [row for row in signals if row["metric_tag"] in {"biological", "latent_bio_mixed"}]
    candidate_pairs = [row for row in pairs if row["candidate_equal_cell_or_matched_info"]]
    strongest = sorted(assoc, key=lambda row: -abs(row["spearman_rho"]) if np.isfinite(row["spearman_rho"]) else 0.0)[:15]
    top_pairs = pairs[:15]

    gpu_authorized = False
    status = "condition_residual_association_gate_no_gpu"
    reasons = [
        "association_gate_is_hypothesis_generation_only",
        "matched_pair_training_requires_predeclared_launcher_and_noharm_gate",
    ]
    if not bio_signals:
        reasons.append("no_biological_or_latent_bio_metric_passed_all_controls")
    if not candidate_pairs:
        reasons.append("no_residual_axis_pair_met_matching_threshold")

    join_fields = [
        "source_family",
        "axis_family",
        "arm",
        "split_name",
        "role",
        "status",
        "cross_pp_delta",
        "family_pp_delta",
        "family_mmd_delta",
        "tail_score",
        "weak_proxy_info",
    ] + PREDICTORS
    assoc_fields = [
        "axis_filter",
        "outcome",
        "predictor",
        "metric_tag",
        "n",
        "spearman_rho",
        "spearman_perm_p",
        "partial_corr",
        "partial_sign_consistent",
        "loo_sign_stability",
        "gate_signal",
    ]
    write_csv(OUT_JOIN, join_rows, join_fields)
    write_csv(OUT_ASSOC, assoc, assoc_fields)
    write_csv(OUT_PAIRS, pairs)

    payload = {
        "status": status,
        "gpu_authorized": gpu_authorized,
        "n_join_rows": len(join_rows),
        "n_association_rows": len(assoc),
        "n_gate_signals": len(signals),
        "n_latent_gate_signals": len(latent_signals),
        "n_candidate_pairs": len(candidate_pairs),
        "reasons": reasons,
        "join_csv": str(OUT_JOIN),
        "association_csv": str(OUT_ASSOC),
        "pair_csv": str(OUT_PAIRS),
        "signals": signals,
        "strongest": strongest,
        "top_pairs": top_pairs,
    }
    with OUT_JSON.open("w") as f:
        json.dump(jsonable({"summary": payload, "join_rows": join_rows, "association_rows": assoc, "pairs": pairs}), f, indent=2)

    lines = [
        "# LatentFM Downstream Condition-Residual Association Gate",
        "",
        f"Timestamp: `{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}`",
        "",
        f"Status: `{status}`",
        "",
        f"GPU authorized: `{gpu_authorized}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only over frozen downstream outcomes plus train-only condition/residual geometry.",
        "- Uses only split `train` conditions for x-axis construction.",
        "- Does not train, infer, read canonical multi, read Track C held-out query, or select checkpoints.",
        "",
        "## Summary",
        "",
        f"- Joined outcome rows: `{len(join_rows)}`.",
        f"- Association tests: `{len(assoc)}`.",
        f"- Gate signals: `{len(signals)}`; latent-bio signals: `{len(latent_signals)}`.",
        f"- Matched-pair candidate rows: `{len(candidate_pairs)}`.",
        "",
        "## Passing Association Rows",
        "",
        "| axis | outcome | predictor | tag | n | rho | p | partial | loo |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    if signals:
        for row in sorted(signals, key=lambda r: (r["axis_filter"], r["outcome"], r["predictor"])):
            lines.append(
                f"| `{row['axis_filter']}` | `{row['outcome']}` | `{row['predictor']}` | `{row['metric_tag']}` | {row['n']} | {fmt(row['spearman_rho'])} | {fmt(row['spearman_perm_p'])} | {fmt(row['partial_corr'])} | {fmt(row['loo_sign_stability'])} |"
            )
    else:
        lines.append("| none |  |  |  |  |  |  |  |  |")

    lines.extend(
        [
            "",
            "## Strongest Associations",
            "",
            "| axis | outcome | predictor | tag | n | rho | p | partial | gate |",
            "|---|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in strongest:
        lines.append(
            f"| `{row['axis_filter']}` | `{row['outcome']}` | `{row['predictor']}` | `{row['metric_tag']}` | {row['n']} | {fmt(row['spearman_rho'])} | {fmt(row['spearman_perm_p'])} | {fmt(row['partial_corr'])} | `{row['gate_signal']}` |"
        )

    lines.extend(
        [
            "",
            "## Top Matched-Pair Candidates",
            "",
            "| axis | left | right | axis delta z | max confound z | candidate |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for row in top_pairs:
        lines.append(
            f"| `{row['axis']}` | `{row['left_split']}` | `{row['right_split']}` | {fmt(row['axis_delta_z'])} | {fmt(row['max_confound_z'])} | `{row['candidate_equal_cell_or_matched_info']}` |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The condition/residual geometry is now a measurable information axis, but this gate remains hypothesis-generating. A GPU run should only follow if the selected matched pair is converted into a leakage-safe split/launcher with a fixed no-harm gate and provenance.",
            "",
            "Current interpretation: residual Vendi/effective-rank distinguishes broad, balanced split designs from narrow/high-response designs, while response-norm mean identifies strong but potentially narrow perturbation regimes. This supports defining scaling in terms of train-support information content rather than raw condition count alone.",
            "",
            "## Outputs",
            "",
            f"- Join rows: `{OUT_JOIN}`",
            f"- Association rows: `{OUT_ASSOC}`",
            f"- Pair candidates: `{OUT_PAIRS}`",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps(jsonable(payload), indent=2))


if __name__ == "__main__":
    main()
