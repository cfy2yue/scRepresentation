#!/usr/bin/env python3
"""CPU-only association gate for downstream information scaling.

Inputs are frozen report artifacts:
- split-level train-support information metrics from the 2026-06-28 preflight;
- completed scaling/count-smoke and true-cell outcome tables.

The script does not train, infer, read canonical multi, read Track C held-out
query, select checkpoints, or use GPU.
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
OUT_DIR = ROOT / "reports" / "downstream_information_association_gate_20260628"
OUT_MD = ROOT / "reports" / "LATENTFM_DOWNSTREAM_INFORMATION_ASSOCIATION_GATE_20260628.md"
OUT_JSON = ROOT / "reports" / "latentfm_downstream_information_association_gate_20260628.json"
OUT_JOIN = OUT_DIR / "information_outcome_join_rows.csv"
OUT_ASSOC = OUT_DIR / "association_rows.csv"

INFO_CSV = ROOT / "reports/downstream_information_scaling_preflight_20260628/split_information_metrics.csv"
CONDITION_OUTCOME_CSV = ROOT / "reports/scaling_figure_data_20260625/condition_exposure_curve.csv"
TRUECELL_OUTCOME_CSV = ROOT / "reports/scaling_figure_data_20260625/truecell_budget_curve.csv"

METRIC_TAGS = {
    "n_train_conditions": "technical",
    "n_dataset_labels": "mixed",
    "dataset_entropy_norm": "technical",
    "dataset_effective_count": "technical",
    "max_dataset_share": "technical",
    "n_background_labels": "mixed",
    "background_entropy_norm": "mixed",
    "background_effective_count": "mixed",
    "max_background_share": "mixed",
    "n_perturbation_types": "mixed",
    "perturbation_type_entropy_norm": "mixed",
    "perturbation_type_effective_count": "mixed",
    "max_perturbation_type_share": "mixed",
    "n_target_genes": "biological",
    "target_gene_entropy_norm": "biological",
    "target_gene_effective_count": "biological",
    "drug_condition_fraction": "mixed",
    "gene_condition_fraction": "mixed",
    "dataset_mean_effective_rank": "technical",
    "dataset_mean_rank_entropy_norm": "technical",
    "dataset_mean_pairwise_l2": "technical",
}

PREDICTORS = list(METRIC_TAGS)
OUTCOMES = ["cross_pp_delta", "family_pp_delta", "family_mmd_delta", "tail_score"]
CONFOUNDS = [
    "n_train_conditions",
    "n_dataset_labels",
    "max_dataset_share",
    "n_background_labels",
    "n_perturbation_types",
    "drug_condition_fraction",
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
    x = x.astype(float)
    y = y.astype(float)
    x = x - x.mean()
    y = y - y.mean()
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
        yp = rng.permutation(y)
        val = spearman(x, yp)
        if np.isfinite(val) and abs(val) >= abs(observed):
            extreme += 1
    return extreme / (n_perm + 1)


def residualize(v: np.ndarray, confounds: np.ndarray) -> np.ndarray:
    if confounds.size == 0:
        return v - v.mean()
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
    signs = []
    for i in range(len(x)):
        mask = np.ones(len(x), dtype=bool)
        mask[i] = False
        val = spearman(x[mask], y[mask])
        if np.isfinite(val):
            signs.append(math.copysign(1.0, val) == math.copysign(1.0, observed))
    return sum(signs) / len(signs) if signs else float("nan")


def mean_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {}
    out: dict[str, Any] = {}
    for key in rows[0]:
        vals = [to_float(r.get(key)) for r in rows]
        finite = [v for v in vals if np.isfinite(v)]
        if finite and len(finite) == len(rows):
            out[key] = float(np.mean(finite))
        else:
            out[key] = rows[0].get(key, "")
    return out


def select_info_row(info_rows: list[dict[str, str]], contains: str) -> dict[str, str] | None:
    matches = [r for r in info_rows if contains in r["split_name"]]
    if not matches:
        return None
    # Prefer v2 when both v1 and v2 exist, otherwise shortest name is usually most direct.
    v2 = [r for r in matches if r["split_name"].endswith("_v2")]
    return sorted(v2 or matches, key=lambda r: (len(r["split_name"]), r["split_name"]))[0]


def truecell_info_row(info_rows: list[dict[str, str]], budget: int) -> dict[str, Any] | None:
    rows = [
        r
        for r in info_rows
        if "xverse_true_cell_count_scaling_nested_splits_20260624" in r["split_file"]
        and re.search(rf"_budget{budget}_seed\d+$", r["split_name"])
    ]
    return mean_rows(rows) if rows else None


def condition_info_row(info_rows: list[dict[str, str]], arm: str) -> dict[str, Any] | None:
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
        # There is no direct full split artifact in the current preflight. Use the
        # broadest cap120 arm only as a weak proxy and mark it downstream.
        return select_info_row(info_rows, "scaling_cap120_all")
    needle = mapping.get(arm, arm)
    return select_info_row(info_rows, needle)


def add_numeric_info(out: dict[str, Any], info: dict[str, Any]) -> None:
    for pred in PREDICTORS:
        out[pred] = to_float(info.get(pred))


def build_join_rows() -> list[dict[str, Any]]:
    info_rows = load_csv(INFO_CSV)
    joined: list[dict[str, Any]] = []

    for row in load_csv(CONDITION_OUTCOME_CSV):
        info = condition_info_row(info_rows, row["arm"])
        if info is None:
            continue
        out = {
            "source_family": row["source_family"],
            "axis_family": "condition_exposure",
            "arm": row["arm"],
            "split_name": info.get("split_name", ""),
            "role": row.get("role", ""),
            "status": row.get("status", ""),
            "outcome_source_report": row.get("source_report", ""),
            "cross_pp_delta": to_float(row.get("cross_pp_delta")),
            "family_pp_delta": to_float(row.get("family_pp_delta")),
            "family_mmd_delta": to_float(row.get("family_mmd_delta")),
            "tail_score": -1.0 if to_float(row.get("family_mmd_delta")) > 0.001 else 0.0,
            "weak_proxy_info": row["arm"] == "full_trainonly",
        }
        add_numeric_info(out, info)
        joined.append(out)

    for row in load_csv(TRUECELL_OUTCOME_CSV):
        info = truecell_info_row(info_rows, int(float(row["budget"])))
        if info is None:
            continue
        out = {
            "source_family": "truecell_budget_curve",
            "axis_family": "true_cell_count",
            "arm": f"{row['series']}_budget{row['budget']}",
            "split_name": info.get("split_name", f"mean_truecell_budget{row['budget']}"),
            "role": row["series"],
            "status": row.get("status", ""),
            "outcome_source_report": row.get("source_report", ""),
            "cross_pp_delta": to_float(row.get("cross_pp_mean")),
            "family_pp_delta": to_float(row.get("family_pp_mean")),
            "family_mmd_delta": to_float(row.get("family_mmd_mean")),
            "tail_score": -to_float(row.get("cross_pp_negative_tails")),
            "weak_proxy_info": False,
        }
        add_numeric_info(out, info)
        joined.append(out)
    return joined


def finite_matrix(rows: list[dict[str, Any]], keys: list[str]) -> np.ndarray:
    return np.asarray([[to_float(r.get(k)) for k in keys] for r in rows], dtype=float)


def association_rows(join_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for axis_filter in ["all", "condition_exposure", "true_cell_count"]:
        subset = [r for r in join_rows if axis_filter == "all" or r["axis_family"] == axis_filter]
        for outcome in OUTCOMES:
            y = np.asarray([to_float(r.get(outcome)) for r in subset], dtype=float)
            for pred in PREDICTORS:
                x = np.asarray([to_float(r.get(pred)) for r in subset], dtype=float)
                confound_keys = [c for c in CONFOUNDS if c != pred]
                c = finite_matrix(subset, confound_keys)
                mask = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(c), axis=1)
                if int(mask.sum()) < 5:
                    continue
                xm = x[mask]
                ym = y[mask]
                cm = c[mask]
                rho = spearman(xm, ym)
                row = {
                    "axis_filter": axis_filter,
                    "outcome": outcome,
                    "predictor": pred,
                    "metric_tag": METRIC_TAGS[pred],
                    "n": int(mask.sum()),
                    "spearman_rho": rho,
                    "spearman_perm_p": permutation_p(xm, ym, rho),
                    "partial_corr": partial_corr(xm, ym, cm),
                    "loo_sign_stability": loo_stability(xm, ym, rho),
                }
                sign_consistent = (
                    np.isfinite(row["spearman_rho"])
                    and np.isfinite(row["partial_corr"])
                    and row["spearman_rho"] != 0
                    and row["partial_corr"] != 0
                    and math.copysign(1.0, row["spearman_rho"]) == math.copysign(1.0, row["partial_corr"])
                )
                row["partial_sign_consistent"] = sign_consistent
                row["gate_signal"] = (
                    row["metric_tag"] in {"biological", "mixed"}
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


def fmt(v: Any) -> str:
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return ""
        return f"{float(v):+.4f}"
    return str(v)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    join_rows = build_join_rows()
    assoc = association_rows(join_rows)
    signals = [r for r in assoc if r["gate_signal"]]
    biological_signals = [r for r in signals if r["metric_tag"] == "biological"]
    mixed_signals = [r for r in signals if r["metric_tag"] == "mixed"]

    # Strongest rows by absolute rho, for reporting.
    strongest = sorted(
        assoc,
        key=lambda r: (
            r["axis_filter"] != "all",
            -abs(r["spearman_rho"]) if np.isfinite(r["spearman_rho"]) else 0,
        ),
    )[:12]
    signal_rows = sorted(signals, key=lambda r: (r["axis_filter"], r["outcome"], r["predictor"]))

    status = "information_association_gate_no_gpu"
    gpu_authorized = False
    reasons = []
    if not biological_signals:
        reasons.append("no_biological_metric_passed_all_association_controls")
    if len(join_rows) < 15:
        reasons.append("small_completed_outcome_panel")
    reasons.append("candidate_generation_requires_predeclared_equal_cell_or_matched_entropy_split")
    reasons.append("canonical_noharm_and_dual_baseline_veto_still_required")

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

    payload = {
        "status": status,
        "gpu_authorized": gpu_authorized,
        "n_join_rows": len(join_rows),
        "n_association_rows": len(assoc),
        "n_gate_signals": len(signals),
        "n_biological_gate_signals": len(biological_signals),
        "n_mixed_gate_signals": len(mixed_signals),
        "reasons": reasons,
        "join_csv": str(OUT_JOIN),
        "association_csv": str(OUT_ASSOC),
        "signal_rows": signal_rows,
        "strongest_rows": strongest,
    }
    with OUT_JSON.open("w") as f:
        json.dump(jsonable({"summary": payload, "join_rows": join_rows, "association_rows": assoc}), f, indent=2)

    lines = [
        "# LatentFM Downstream Information Association Gate",
        "",
        "Timestamp: `2026-06-28 05:05 CST`",
        "",
        f"Status: `{status}`",
        "",
        f"GPU authorized: `{str(gpu_authorized)}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only over frozen split-information metrics and completed scaling outcome reports.",
        "- Does not train, infer, read canonical multi, read Track C held-out query, select checkpoints, or use GPU.",
        "- Existing canonical no-harm vetoes and source/control dual-baseline requirements remain binding.",
        "",
        "## Summary",
        "",
        f"- Joined information/outcome rows: `{len(join_rows)}`.",
        f"- Association tests: `{len(assoc)}`.",
        f"- Gate signals: `{len(signals)}`.",
        f"- Biological gate signals: `{len(biological_signals)}`.",
        f"- Mixed biological/technical gate signals: `{len(mixed_signals)}`.",
        "",
        "## Gate Rule",
        "",
        "A predictor can nominate a GPU split only if it is biological or mixed, has `abs(Spearman rho) >= 0.55`, permutation `p <= 0.05`, `abs(partial_corr) >= 0.35` after controlling basic composition confounds, partial-correlation direction agrees with Spearman direction, and leave-one-out sign stability >= `0.8`. A passing association is still not a model claim; it only authorizes designing an equal-cell/different-info or matched-entropy split with dual-baseline promotion rules.",
        "",
        "## Passing Association Rows",
        "",
        "| axis | outcome | predictor | tag | n | rho | p | partial | loo |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    if signal_rows:
        for r in signal_rows:
            lines.append(
                f"| `{r['axis_filter']}` | `{r['outcome']}` | `{r['predictor']}` | `{r['metric_tag']}` | {r['n']} | {fmt(r['spearman_rho'])} | {fmt(r['spearman_perm_p'])} | {fmt(r['partial_corr'])} | {fmt(r['loo_sign_stability'])} |"
            )
    else:
        lines.append("| none |  |  |  |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "## Strongest Associations For Interpretation",
            "",
            "| axis | outcome | predictor | tag | n | rho | p | partial | loo | gate |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for r in strongest:
        lines.append(
            f"| `{r['axis_filter']}` | `{r['outcome']}` | `{r['predictor']}` | `{r['metric_tag']}` | {r['n']} | {fmt(r['spearman_rho'])} | {fmt(r['spearman_perm_p'])} | {fmt(r['partial_corr'])} | {fmt(r['loo_sign_stability'])} | `{r['gate_signal']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "No GPU is authorized. The completed outcome panel is useful for mechanistic interpretation but does not yet produce a biological information metric that survives all association controls. The strongest signals should guide the next CPU split-design step, not a direct training launch.",
            "",
            "Next CPU action: construct a predeclared equal-cell-count/different-information split design using biologically interpretable axes where possible, especially target/pathway/background entropy and ZSCAPE-derived cell-type/state diversity once available.",
            "",
            "## Outputs",
            "",
            f"- Join rows: `{OUT_JOIN}`",
            f"- Association rows: `{OUT_ASSOC}`",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
