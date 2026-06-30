#!/usr/bin/env python3
"""Try to salvage a source-balanced exact/nonstatic scaling design.

CPU/report-only. Uses completed matched-pair artifacts and does not train,
infer, select checkpoints, read canonical multi, read Track C query, or use GPU.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "source_balanced_exact_nonstatic_design_gate_20260630"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAIR_CSV = (
    REPORTS
    / "exact_coverage_crossdataset_matched_feasibility_20260629"
    / "best_crossdataset_matched_pairs.csv"
)
V3_DECISION = (
    REPORTS
    / "hvg_advantage_resid_v3_highlow_smoke_20260630"
    / "latentfm_hvg_advantage_resid_v3_highlow_decision_20260630.json"
)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": str(path)}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def js(left: Counter[str], right: Counter[str]) -> float:
    keys = sorted(set(left) | set(right))
    lt = float(sum(left.values()))
    rt = float(sum(right.values()))
    if lt <= 0 or rt <= 0:
        return float("nan")
    p = np.asarray([left.get(k, 0) / lt for k in keys], dtype=float)
    q = np.asarray([right.get(k, 0) / rt for k in keys], dtype=float)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / np.maximum(b[mask], 1e-12))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def smd(high: pd.Series, low: pd.Series) -> float:
    h = pd.to_numeric(high, errors="coerce").dropna().astype(float)
    l = pd.to_numeric(low, errors="coerce").dropna().astype(float)
    if len(h) < 2 or len(l) < 2:
        return float("nan")
    pooled = math.sqrt((float(np.var(h)) + float(np.var(l))) / 2.0)
    if pooled <= 1e-12:
        return 0.0
    return float((float(np.mean(h)) - float(np.mean(l))) / pooled)


def max_share(values: list[str]) -> float:
    if not values:
        return float("nan")
    counts = Counter(values)
    return max(counts.values()) / len(values)


def ptype_count(df: pd.DataFrame) -> int:
    return int(df["perturbation_type"].nunique()) if "perturbation_type" in df else 0


def metrics(name: str, df: pd.DataFrame, note: str) -> dict[str, Any]:
    if df.empty:
        return {
            "policy": name,
            "n_pairs": 0,
            "feasibility_pass": False,
            "reasons": "empty_subset",
            "note": note,
        }
    hds = Counter(df["high_dataset"].astype(str))
    lds = Counter(df["low_dataset"].astype(str))
    hsrc = Counter(df["high_source_family"].astype(str))
    lsrc = Counter(df["low_source_family"].astype(str))
    log_high_ctrl = np.log1p(pd.to_numeric(df["high_n_ctrl"], errors="coerce"))
    log_low_ctrl = np.log1p(pd.to_numeric(df["low_n_ctrl"], errors="coerce"))
    log_high_gt = np.log1p(pd.to_numeric(df["high_n_gt"], errors="coerce"))
    log_low_gt = np.log1p(pd.to_numeric(df["low_n_gt"], errors="coerce"))
    smds = {
        "smd_log_n_ctrl": smd(log_high_ctrl, log_low_ctrl),
        "smd_log_n_gt": smd(log_high_gt, log_low_gt),
        "smd_response_norm": smd(df["high_response_norm"], df["low_response_norm"]),
        "smd_analog_support_dataset_count": smd(
            df["high_analog_support_dataset_count"], df["low_analog_support_dataset_count"]
        ),
    }
    max_abs = max(abs(v) for v in smds.values() if math.isfinite(v))
    datasets = set(df["high_dataset"].astype(str)) | set(df["low_dataset"].astype(str))
    reasons: list[str] = []
    if len(df) < 300:
        reasons.append("pairs_below_300")
    if len(datasets) < 12:
        reasons.append("datasets_below_12")
    if ptype_count(df) < 2:
        reasons.append("perturbation_types_below_2")
    if js(hsrc, lsrc) > 0.25:
        reasons.append("source_js_gt_0p25")
    if max_abs > 0.15:
        reasons.append("max_abs_smd_gt_0p15")
    if max(max_share(list(hds.elements())), max_share(list(lds.elements()))) > 0.20:
        reasons.append("dataset_max_share_gt_0p20")
    return {
        "policy": name,
        "n_pairs": int(len(df)),
        "n_total_datasets": int(len(datasets)),
        "n_perturbation_types": ptype_count(df),
        "dataset_js_divergence": js(hds, lds),
        "source_js_divergence": js(hsrc, lsrc),
        "high_source_max_share": max_share(list(hsrc.elements())),
        "low_source_max_share": max_share(list(lsrc.elements())),
        "high_dataset_max_share": max_share(list(hds.elements())),
        "low_dataset_max_share": max_share(list(lds.elements())),
        "top_dataset_pair_fraction": max(max_share(list(hds.elements())), max_share(list(lds.elements()))),
        **smds,
        "max_abs_covariate_smd": max_abs,
        "feasibility_pass": len(reasons) == 0,
        "reasons": ";".join(reasons) if reasons else "none",
        "note": note,
    }


def greedy_salvage(df: pd.DataFrame, min_pairs: int = 300) -> pd.DataFrame:
    """Greedily remove pairs that worsen source/cell-count imbalance."""
    work = df.copy()
    work["log_delta_abs"] = (
        np.log1p(pd.to_numeric(work["high_n_ctrl"], errors="coerce"))
        - np.log1p(pd.to_numeric(work["low_n_ctrl"], errors="coerce"))
    ).abs()
    while len(work) > min_pairs:
        base = metrics("tmp", work, "")
        if (
            base.get("source_js_divergence", 1.0) <= 0.25
            and base.get("max_abs_covariate_smd", 1.0) <= 0.15
            and base.get("top_dataset_pair_fraction", 1.0) <= 0.20
        ):
            break
        hsrc = Counter(work["high_source_family"].astype(str))
        lsrc = Counter(work["low_source_family"].astype(str))
        all_src = set(hsrc) | set(lsrc)
        h_total = sum(hsrc.values())
        l_total = sum(lsrc.values())
        imbalance = {
            s: (hsrc.get(s, 0) / h_total if h_total else 0.0)
            - (lsrc.get(s, 0) / l_total if l_total else 0.0)
            for s in all_src
        }
        scores = []
        for idx, row in work.iterrows():
            h_bad = max(imbalance.get(str(row["high_source_family"]), 0.0), 0.0)
            l_bad = max(-imbalance.get(str(row["low_source_family"]), 0.0), 0.0)
            score = h_bad + l_bad + 0.10 * float(row["log_delta_abs"])
            scores.append((score, idx))
        scores.sort(reverse=True)
        work = work.drop(index=scores[0][1])
    return work.drop(columns=["log_delta_abs"], errors="ignore")


def main() -> int:
    df = pd.read_csv(PAIR_CSV)
    candidates: list[tuple[str, pd.DataFrame, str]] = []
    candidates.append(("all_390", df, "original best exact cross-dataset pair set"))
    for cap in [380, 360, 340, 320, 300]:
        candidates.append((f"lowest_cost_cap{cap}", df.sort_values("cost").head(cap), "lowest matching-cost prefix"))
    log_delta = (
        np.log1p(pd.to_numeric(df["high_n_ctrl"], errors="coerce"))
        - np.log1p(pd.to_numeric(df["low_n_ctrl"], errors="coerce"))
    ).abs()
    for threshold in [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]:
        candidates.append((f"abs_log_count_delta_le_{threshold:g}", df[log_delta <= threshold], "pairwise count-gap filter"))
    same_source = df[df["high_source_family"] == df["low_source_family"]]
    candidates.append(("same_source_only", same_source, "strict same-source subset"))
    candidates.append(("greedy_source_count_salvage_min300", greedy_salvage(df, 300), "greedy source/count balancing with >=300-pair floor"))

    rows = [metrics(name, cand, note) for name, cand, note in candidates]
    summary = pd.DataFrame(rows).sort_values(
        ["feasibility_pass", "n_pairs", "max_abs_covariate_smd"], ascending=[False, False, True]
    )

    v3 = read_json(V3_DECISION)
    v3_checks = v3.get("decision", {}).get("checks", {})
    v3_closed = (
        float(v3_checks.get("high_minus_low_cross_pp_delta", 0.0)) < 0
        and float(v3_checks.get("high_minus_low_family_pp_delta", 0.0)) < 0
    )
    pass_count = int(summary["feasibility_pass"].sum())
    status = (
        "source_balanced_exact_nonstatic_design_gate_pass_prepare_cpu_outcome_gate"
        if pass_count
        else "source_balanced_exact_nonstatic_design_gate_fail_no_gpu"
    )

    csv_path = OUT_DIR / "source_balanced_exact_nonstatic_design_gate_rows_20260630.csv"
    json_path = OUT_DIR / "source_balanced_exact_nonstatic_design_gate_20260630.json"
    md_path = OUT_DIR / "LATENTFM_SOURCE_BALANCED_EXACT_NONSTATIC_DESIGN_GATE_20260630.md"
    summary.to_csv(csv_path, index=False)
    best = summary.iloc[0].to_dict()

    out = {
        "timestamp": now(),
        "status": status,
        "gpu_authorized": False,
        "feasibility_pass_count": pass_count,
        "best_policy": best,
        "nonstatic_v3_experimental_closed": v3_closed,
        "boundary": {
            "cpu_report_only": True,
            "training_or_inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection": False,
            "trackc_query_access": False,
            "uses_gpu": False,
        },
        "inputs": {"exact_pairs": str(PAIR_CSV), "v3_decision": str(V3_DECISION)},
        "outputs": {"rows": str(csv_path), "json": str(json_path), "markdown": str(md_path)},
    }
    json_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Source-Balanced Exact/Nonstatic Design Gate",
        "",
        f"Created: `{out['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only salvage attempt over completed exact cross-dataset pairs and v3 nonstatic result.",
        "- No training, inference, checkpoint selection, canonical multi selection, Track C query, or GPU.",
        "",
        "## Decision",
        "",
    ]
    if pass_count:
        lines.append(
            "At least one exact-coverage subset passes balance feasibility, but this still only authorizes a later CPU outcome/no-harm gate, not GPU."
        )
    else:
        lines.append(
            "No source-balanced exact-coverage subset reaches the matrix gate. The nonstatic v3 derivative is also experimentally closed, so this branch does not restore GPU eligibility."
        )
    lines += [
        "",
        "## Best Row",
        "",
        f"- policy: `{best.get('policy')}`",
        f"- pairs/datasets/perturbation types: `{int(best.get('n_pairs', 0))}` / `{int(best.get('n_total_datasets', 0))}` / `{int(best.get('n_perturbation_types', 0))}`",
        f"- source JS: `{best.get('source_js_divergence'):.6f}`",
        f"- max abs SMD: `{best.get('max_abs_covariate_smd'):.6f}`",
        f"- top dataset fraction: `{best.get('top_dataset_pair_fraction'):.6f}`",
        f"- reasons: `{best.get('reasons')}`",
        "",
        "## Nonstatic Cross-Check",
        "",
        f"- v3 high-minus-low cross pp: `{v3_checks.get('high_minus_low_cross_pp_delta')}`",
        f"- v3 high-minus-low family pp: `{v3_checks.get('high_minus_low_family_pp_delta')}`",
        f"- v3 experimentally closed: `{v3_closed}`",
        "",
        "## Candidate Rows",
        "",
        "| policy | pairs | datasets | source JS | max SMD | top dataset | pass | reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| `{row['policy']}` | {int(row['n_pairs'])} | {int(row.get('n_total_datasets', 0))} | "
            f"{row.get('source_js_divergence', float('nan')):.4f} | {row.get('max_abs_covariate_smd', float('nan')):.4f} | "
            f"{row.get('top_dataset_pair_fraction', float('nan')):.4f} | `{bool(row['feasibility_pass'])}` | {row['reasons']} |"
        )
    lines += [
        "",
        "## Outputs",
        "",
        f"- rows: `{csv_path}`",
        f"- JSON: `{json_path}`",
        f"- Markdown: `{md_path}`",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "best_policy": best.get("policy"), "pass_count": pass_count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
