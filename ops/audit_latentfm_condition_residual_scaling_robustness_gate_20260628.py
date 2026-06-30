#!/usr/bin/env python3
"""Robustness/FDR gate for condition-residual scaling axes.

CPU/report-only. This de-duplicates the joined outcome panel several ways,
tests prespecified latent-bio axes, and reports whether the signals are stable
enough to guide GPU smokes. It does not train, infer, use canonical multi, use
Track C query, or select checkpoints.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
JOIN_CSV = ROOT / "reports/downstream_condition_residual_association_gate_20260628/condition_residual_information_outcome_join_rows.csv"
PAIR_CSV = ROOT / "reports/downstream_condition_residual_association_gate_20260628/condition_residual_matched_pair_candidates.csv"
OUT_DIR = ROOT / "reports/condition_residual_scaling_robustness_gate_20260628"
OUT_MD = ROOT / "reports/LATENTFM_CONDITION_RESIDUAL_SCALING_ROBUSTNESS_GATE_20260628.md"
OUT_CSV = OUT_DIR / "condition_residual_scaling_robustness_rows.csv"
OUT_JSON = ROOT / "reports/latentfm_condition_residual_scaling_robustness_gate_20260628.json"

PREDICTORS = [
    "response_norm_mean",
    "residual_pairwise_l2_mean",
    "residual_effective_rank",
    "residual_vendi_rbf_effective_count",
    "dataset_effective_count_condition_vectors",
    "perturbation_type_effective_count_condition_vectors",
]
OUTCOMES = [
    "cross_pp_delta",
    "family_pp_delta",
    "family_mmd_delta",
    "tail_score",
]
PRIMARY_PAIRS = [
    ("response_norm_mean", "family_mmd_delta"),
    ("response_norm_mean", "tail_score"),
    ("residual_pairwise_l2_mean", "tail_score"),
    ("residual_effective_rank", "cross_pp_delta"),
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
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denom <= 0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(ranks(x), ranks(y))


def permutation_p(x: np.ndarray, y: np.ndarray, observed: float, n_perm: int = 5000, seed: int = 42) -> float:
    if len(x) < 5 or not np.isfinite(observed):
        return float("nan")
    rng = np.random.default_rng(seed)
    extreme = 1
    for _ in range(n_perm):
        val = spearman(x, rng.permutation(y))
        if np.isfinite(val) and abs(val) >= abs(observed):
            extreme += 1
    return extreme / (n_perm + 1)


def bootstrap_sign(x: np.ndarray, y: np.ndarray, observed: float, n_boot: int = 3000, seed: int = 7) -> dict[str, float]:
    if len(x) < 5 or not np.isfinite(observed) or observed == 0:
        return {"boot_sign_stability": float("nan"), "boot_rho_median": float("nan"), "boot_rho_lo": float("nan"), "boot_rho_hi": float("nan")}
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    target_sign = math.copysign(1.0, observed)
    stable = 0
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        if len(set(int(i) for i in idx)) < 3:
            continue
        val = spearman(x[idx], y[idx])
        if not np.isfinite(val):
            continue
        vals.append(val)
        stable += int(math.copysign(1.0, val) == target_sign)
    if not vals:
        return {"boot_sign_stability": float("nan"), "boot_rho_median": float("nan"), "boot_rho_lo": float("nan"), "boot_rho_hi": float("nan")}
    arr = np.asarray(vals, dtype=float)
    return {
        "boot_sign_stability": stable / len(vals),
        "boot_rho_median": float(np.median(arr)),
        "boot_rho_lo": float(np.quantile(arr, 0.025)),
        "boot_rho_hi": float(np.quantile(arr, 0.975)),
    }


def bh_qvalues(pvalues: list[float]) -> list[float]:
    indexed = [(i, p) for i, p in enumerate(pvalues) if np.isfinite(p)]
    qvalues = [float("nan")] * len(pvalues)
    if not indexed:
        return qvalues
    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    running = 1.0
    for rank, (i, p) in reversed(list(enumerate(indexed, start=1))):
        running = min(running, p * m / rank)
        qvalues[i] = running
    return qvalues


def design_family(row: dict[str, Any]) -> str:
    arm = str(row.get("arm", ""))
    if arm.startswith("3k_nested") or arm.startswith("6k_budget"):
        return f"truecell::{row.get('role', '')}::{arm.split('_budget')[-1]}"
    if "breadth_" in arm:
        return f"breadth::{row.get('role', arm)}"
    if arm in {"cap30_all", "cap120_all", "full_trainonly"}:
        return f"count::{arm}"
    if arm.startswith("gene_cap120"):
        return f"gene::{arm}"
    if "exposure" in arm or "type_balanced" in arm:
        return f"composition::{arm}"
    return arm


def aggregate_rows(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == "raw":
        return [dict(row) for row in rows]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if mode == "split_dedup":
            key = str(row.get("split_name", row.get("arm", "")))
        elif mode == "design_family":
            key = design_family(row)
        else:
            raise ValueError(mode)
        groups[key].append(row)
    out: list[dict[str, Any]] = []
    keys = sorted(set().union(*(set(row) for row in rows)))
    for key, group in sorted(groups.items()):
        merged: dict[str, Any] = {"aggregate_key": key, "n_aggregated_rows": len(group)}
        for col in keys:
            vals = [to_float(row.get(col)) for row in group]
            finite = [v for v in vals if np.isfinite(v)]
            if finite and len(finite) == len(group):
                merged[col] = float(np.mean(finite))
            else:
                merged[col] = group[0].get(col, "")
        out.append(merged)
    return out


def test_rows(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for predictor in PREDICTORS:
        for outcome in OUTCOMES:
            x = np.asarray([to_float(row.get(predictor)) for row in rows], dtype=float)
            y = np.asarray([to_float(row.get(outcome)) for row in rows], dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if int(mask.sum()) < 5:
                continue
            xm = x[mask]
            ym = y[mask]
            rho = spearman(xm, ym)
            p = permutation_p(xm, ym, rho)
            boot = bootstrap_sign(xm, ym, rho)
            primary = (predictor, outcome) in PRIMARY_PAIRS
            results.append(
                {
                    "mode": mode,
                    "predictor": predictor,
                    "outcome": outcome,
                    "primary_pair": primary,
                    "n": int(mask.sum()),
                    "spearman_rho": rho,
                    "perm_p": p,
                    **boot,
                }
            )
    q = bh_qvalues([row["perm_p"] for row in results])
    for row, qv in zip(results, q, strict=True):
        row["bh_q"] = qv
        row["robust_signal"] = (
            row["primary_pair"]
            and np.isfinite(row["spearman_rho"])
            and abs(float(row["spearman_rho"])) >= 0.5
            and np.isfinite(row["boot_sign_stability"])
            and float(row["boot_sign_stability"]) >= 0.8
            and np.isfinite(row["bh_q"])
            and float(row["bh_q"]) <= 0.15
        )
        row["exploratory_stable_signal"] = (
            row["primary_pair"]
            and np.isfinite(row["spearman_rho"])
            and abs(float(row["spearman_rho"])) >= 0.5
            and np.isfinite(row["boot_sign_stability"])
            and float(row["boot_sign_stability"]) >= 0.8
            and np.isfinite(row["perm_p"])
            and float(row["perm_p"]) <= 0.10
        )
    return results


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
    rows = load_csv(JOIN_CSV)
    pair_rows = load_csv(PAIR_CSV)
    all_results: list[dict[str, Any]] = []
    mode_counts: dict[str, int] = {}
    for mode in ["raw", "split_dedup", "design_family"]:
        agg = aggregate_rows(rows, mode)
        mode_counts[mode] = len(agg)
        all_results.extend(test_rows(agg, mode))

    robust = [row for row in all_results if row["robust_signal"]]
    stable = [row for row in all_results if row["exploratory_stable_signal"]]
    primary_results = [row for row in all_results if row["primary_pair"]]
    best_primary = sorted(primary_results, key=lambda r: (not r["exploratory_stable_signal"], float(r["perm_p"]) if np.isfinite(r["perm_p"]) else 9.0))[:12]
    candidate_pairs = [row for row in pair_rows if str(row.get("candidate_equal_cell_or_matched_info")) == "True"]

    if robust:
        status = "condition_residual_scaling_robust_pass_no_gpu"
    elif stable:
        status = "condition_residual_scaling_exploratory_stable_no_gpu"
    else:
        status = "condition_residual_scaling_robust_fail_no_gpu"

    write_csv(OUT_CSV, all_results)
    payload = {
        "status": status,
        "gpu_authorized": False,
        "mode_counts": mode_counts,
        "n_tests": len(all_results),
        "n_robust_signals": len(robust),
        "n_exploratory_stable_signals": len(stable),
        "n_candidate_pairs": len(candidate_pairs),
        "robust_signals": robust,
        "exploratory_stable_signals": stable,
        "best_primary": best_primary,
        "top_candidate_pairs": candidate_pairs[:20],
        "outputs": {"rows_csv": str(OUT_CSV), "report": str(OUT_MD)},
    }
    with OUT_JSON.open("w") as f:
        json.dump(jsonable(payload), f, indent=2)

    lines = [
        "# LatentFM Condition-Residual Scaling Robustness Gate",
        "",
        f"Timestamp: `{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only robustness check over frozen outcome and condition-residual information rows.",
        "- No training, inference, canonical multi, Track C query, or checkpoint selection.",
        "- This gate can nominate exploratory GPU smokes but cannot promote a model.",
        "",
        "## Summary",
        "",
        f"- Raw joined rows: `{mode_counts['raw']}`.",
        f"- Split-deduplicated rows: `{mode_counts['split_dedup']}`.",
        f"- Design-family rows: `{mode_counts['design_family']}`.",
        f"- Tests run: `{len(all_results)}`.",
        f"- Robust FDR signals: `{len(robust)}`.",
        f"- Exploratory stable primary signals: `{len(stable)}`.",
        f"- Matched-pair candidates available: `{len(candidate_pairs)}`.",
        "",
        "## Best Primary Rows",
        "",
        "| mode | predictor | outcome | n | rho | p | q | boot sign | robust | exploratory |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in best_primary:
        lines.append(
            f"| `{row['mode']}` | `{row['predictor']}` | `{row['outcome']}` | {row['n']} | {fmt(row['spearman_rho'])} | {fmt(row['perm_p'])} | {fmt(row['bh_q'])} | {fmt(row['boot_sign_stability'])} | `{row['robust_signal']}` | `{row['exploratory_stable_signal']}` |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )
    if robust:
        lines.append(
            "At least one prespecified latent-bio axis survives bootstrap sign stability and BH-FDR under the tested row handling. This supports a small, predeclared GPU matched-pair slate, but not promotion or a final scaling law."
        )
    elif stable:
        lines.append(
            "No primary axis survives BH-FDR across the small panel, but at least one prespecified latent-bio axis is sign-stable with permutation support. This supports only exploratory, clearly labeled GPU smokes with strict fail-close rules."
        )
    else:
        lines.append(
            "The primary axes do not survive the robustness screen. Do not use them as the basis for GPU selection without a stronger split/outcome panel."
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Rows: `{OUT_CSV}`",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps(jsonable(payload), indent=2))


if __name__ == "__main__":
    main()
