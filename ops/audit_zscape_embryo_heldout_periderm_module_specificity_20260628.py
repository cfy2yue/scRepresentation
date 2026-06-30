#!/usr/bin/env python3
"""Embryo-heldout module specificity gate for ZSCAPE periderm noto/smo.

CPU-only. This strengthens the OT dynamic response finding by asking whether
each perturb embryo acts like a held-out replicate with the same module
direction, and whether wrong-target/time/lineage controls remain weaker.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

import audit_zscape_expression_latent_biology_preflight_20260628 as base
import audit_zscape_periderm_substate_time_qc_ot_module_gate_20260628 as module_gate


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_ENRICHMENT_SUMMARY = (
    ROOT
    / "reports/zscape_formal_gprofiler_enrichment_20260628"
    / "zscape_formal_gprofiler_enrichment_20260628_130129"
    / "zscape_gprofiler_enrichment_summary.csv"
)
DEFAULT_OT_ROWS = ROOT / "reports/zscape_ot_dynamic_response_gate_20260628/zscape_ot_dynamic_response_rows.csv"
DEFAULT_OUT = ROOT / "reports/zscape_embryo_heldout_periderm_module_specificity_20260628"
PRIMARY_ROWS = {"periderm__noto__24p0h", "periderm__smo__24p0h"}


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def residualize(scores: np.ndarray, manifest: list[dict[str, str]]) -> np.ndarray:
    return module_gate.residualize(scores, manifest)


def embryo_mean_scores(
    manifest: list[dict[str, str]],
    values: np.ndarray,
    row_id: str,
    role: str,
) -> dict[str, float]:
    by_embryo: dict[str, list[float]] = defaultdict(list)
    for idx, row in enumerate(manifest):
        if row.get("row_id") != row_id or row.get("selection_role") != role:
            continue
        embryo = row.get("embryo") or row.get("sample") or f"cell_{idx}"
        by_embryo[embryo].append(float(values[idx]))
    return {embryo: float(np.nanmean(vals)) for embryo, vals in sorted(by_embryo.items()) if vals}


def query_scores(
    query: dict[str, str],
    matrix,
    manifest: list[dict[str, str]],
    sym_to_idx: dict[str, list[int]],
    max_symbols: int,
) -> tuple[np.ndarray, list[str], list[str]]:
    symbols = module_gate.parse_symbols(query.get("top_symbols", ""))[:max_symbols]
    gene_idx: list[int] = []
    missing: list[str] = []
    for symbol in symbols:
        hits = sym_to_idx.get(symbol.lower(), [])
        if hits:
            gene_idx.extend(hits)
        else:
            missing.append(symbol)
    gene_idx = sorted(set(gene_idx))
    if not gene_idx:
        return np.full(len(manifest), np.nan, dtype=float), symbols, missing
    scores = np.asarray(matrix[np.asarray(gene_idx, dtype=int), :].mean(axis=0)).ravel().astype(float)
    return residualize(scores, manifest), symbols, missing


def bootstrap_ci(values: np.ndarray, seed: int, repeats: int) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(repeats):
        sample = values[rng.integers(0, values.size, values.size)]
        boot.append(float(np.mean(sample)))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def control_candidates(manifest: list[dict[str, str]], query: dict[str, str]) -> dict[str, set[str]]:
    q_row = query.get("row_id", "")
    q_time = str(query.get("timepoint", ""))
    out = {
        "periderm_wrong_target_or_time": set(),
        "periderm_wrong_time": set(),
        "wrong_lineage": set(),
    }
    for row in manifest:
        rid = row.get("row_id", "")
        if not rid or rid == q_row:
            continue
        lineage = row.get("cell_type_broad", "")
        time = str(row.get("manifest_timepoint") or row.get("timepoint") or "")
        if lineage == "periderm":
            out["periderm_wrong_target_or_time"].add(rid)
            if time != q_time:
                out["periderm_wrong_time"].add(rid)
        else:
            out["wrong_lineage"].add(rid)
    return out


def control_effects(
    manifest: list[dict[str, str]],
    values: np.ndarray,
    row_ids: set[str],
    sign: float,
) -> list[float]:
    effects: list[float] = []
    for row_id in sorted(row_ids):
        perturb = embryo_mean_scores(manifest, values, row_id, "perturb")
        control = embryo_mean_scores(manifest, values, row_id, "control")
        if not perturb or not control:
            continue
        effects.append(sign * (float(np.mean(list(perturb.values()))) - float(np.mean(list(control.values())))))
    return effects


def evaluate_query(
    query: dict[str, str],
    values: np.ndarray,
    manifest: list[dict[str, str]],
    max_wrong_quantile: float,
    min_positive_fraction: float,
    repeats: int,
) -> dict[str, Any]:
    sign = 1.0 if query.get("direction") == "up" else -1.0
    perturb = embryo_mean_scores(manifest, values, query["row_id"], "perturb")
    control = embryo_mean_scores(manifest, values, query["row_id"], "control")
    control_vals = np.asarray(list(control.values()), dtype=float)
    control_mean = float(np.mean(control_vals)) if control_vals.size else float("nan")
    heldout_rows = []
    heldout_effects = []
    for embryo, value in perturb.items():
        effect = sign * (value - control_mean)
        heldout_effects.append(effect)
        heldout_rows.append(
            {
                "query_name": query.get("query_name", ""),
                "row_id": query["row_id"],
                "direction": query.get("direction", ""),
                "embryo": embryo,
                "heldout_directed_effect": effect,
                "positive": bool(effect > 0),
            }
        )
    arr = np.asarray(heldout_effects, dtype=float)
    if arr.size >= 2:
        ttest = stats.ttest_1samp(arr, popmean=0.0, alternative="greater")
        p_value = float(ttest.pvalue) if math.isfinite(float(ttest.pvalue)) else 1.0
    else:
        p_value = 1.0
    ci_low, ci_high = bootstrap_ci(arr, stable_seed(query["query_name"]), repeats)

    controls = control_candidates(manifest, query)
    wrong_rows: list[dict[str, Any]] = []
    wrong_thresholds: dict[str, float] = {}
    for label, row_ids in controls.items():
        vals = control_effects(manifest, values, row_ids, sign)
        if vals:
            threshold = float(np.quantile(vals, max_wrong_quantile))
            vmax = float(np.max(vals))
        else:
            threshold = float("nan")
            vmax = float("nan")
        wrong_thresholds[label] = threshold
        for rid, val in zip(sorted(row_ids), vals):
            wrong_rows.append(
                {
                    "query_name": query.get("query_name", ""),
                    "row_id": query["row_id"],
                    "direction": query.get("direction", ""),
                    "control_type": label,
                    "control_effect": val,
                }
            )
    finite_thresholds = [v for v in wrong_thresholds.values() if math.isfinite(v)]
    wrong_qmax = max(finite_thresholds) if finite_thresholds else float("nan")
    mean_effect = float(np.mean(arr)) if arr.size else float("nan")
    positive_fraction = float(np.mean(arr > 0.0)) if arr.size else float("nan")
    specificity_gate = bool(math.isfinite(wrong_qmax) and mean_effect > max(0.0, wrong_qmax))
    heldout_gate = bool(
        arr.size >= 4
        and mean_effect > 0.0
        and ci_low > 0.0
        and positive_fraction >= min_positive_fraction
        and p_value <= 0.10
    )
    return {
        "summary": {
            "query_name": query.get("query_name", ""),
            "row_id": query["row_id"],
            "direction": query.get("direction", ""),
            "lineage": query.get("lineage", ""),
            "target": query.get("target", ""),
            "timepoint": query.get("timepoint", ""),
            "n_perturb_embryos": len(perturb),
            "n_control_embryos": len(control),
            "heldout_mean_effect": mean_effect,
            "heldout_ci_low": ci_low,
            "heldout_ci_high": ci_high,
            "heldout_p_value": p_value,
            "heldout_positive_fraction": positive_fraction,
            "wrong_control_qmax": wrong_qmax,
            "wrong_periderm_q95": wrong_thresholds.get("periderm_wrong_target_or_time", float("nan")),
            "wrong_time_q95": wrong_thresholds.get("periderm_wrong_time", float("nan")),
            "wrong_lineage_q95": wrong_thresholds.get("wrong_lineage", float("nan")),
            "heldout_gate": heldout_gate,
            "specificity_gate": specificity_gate,
            "query_gate": bool(heldout_gate and specificity_gate),
        },
        "heldout_rows": heldout_rows,
        "wrong_rows": wrong_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts-npz", type=Path, default=base.DEFAULT_COUNTS)
    parser.add_argument("--cell-index", type=Path, default=base.DEFAULT_CELL_INDEX)
    parser.add_argument("--matched-manifest", type=Path, default=base.DEFAULT_MANIFEST)
    parser.add_argument("--gene-names", type=Path, default=base.DEFAULT_GENES)
    parser.add_argument("--gene-metadata", type=Path, default=base.DEFAULT_GENE_META)
    parser.add_argument("--enrichment-summary", type=Path, default=DEFAULT_ENRICHMENT_SUMMARY)
    parser.add_argument("--ot-rows", type=Path, default=DEFAULT_OT_ROWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-symbols", type=int, default=50)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--min-positive-fraction", type=float, default=0.75)
    parser.add_argument("--wrong-quantile", type=float, default=0.95)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = base.load_csc(args.counts_npz)
    manifest = base.index_manifest(args.cell_index, args.matched_manifest, counts.shape[1])
    matrix = base.lognorm_counts(counts, np.arange(len(manifest), dtype=int))
    gene_ids = [line.strip() for line in args.gene_names.read_text(encoding="utf-8").splitlines() if line.strip()]
    symbol_map = base.read_gene_metadata(args.gene_metadata)
    gene_symbols = [symbol_map.get(gid, gid) for gid in gene_ids]
    sym_to_idx = module_gate.symbol_index(gene_symbols)
    ot_positive = {
        row["row_id"]
        for row in read_csv(args.ot_rows)
        if row.get("row_id") in PRIMARY_ROWS and truthy(row.get("dynamic_response_gate"))
    }
    queries = [
        row
        for row in read_csv(args.enrichment_summary)
        if row.get("row_id") in ot_positive
        and row.get("direction") in {"up", "down"}
        and int(float(row.get("significant_term_count") or 0)) > 0
    ]

    summary_rows: list[dict[str, Any]] = []
    heldout_rows: list[dict[str, Any]] = []
    wrong_rows: list[dict[str, Any]] = []
    for query in queries:
        values, symbols, missing = query_scores(query, matrix, manifest, sym_to_idx, args.max_symbols)
        result = evaluate_query(
            query,
            values,
            manifest,
            max_wrong_quantile=args.wrong_quantile,
            min_positive_fraction=args.min_positive_fraction,
            repeats=args.bootstrap_repeats,
        )
        summary = result["summary"]
        summary["n_symbols"] = len(symbols)
        summary["n_missing_symbols"] = len(missing)
        summary_rows.append(summary)
        heldout_rows.extend(result["heldout_rows"])
        wrong_rows.extend(result["wrong_rows"])

    row_summary: list[dict[str, Any]] = []
    for row_id in sorted(ot_positive):
        sub = [row for row in summary_rows if row["row_id"] == row_id]
        row_summary.append(
            {
                "row_id": row_id,
                "queries": len(sub),
                "query_gates": sum(bool(row["query_gate"]) for row in sub),
                "all_query_gates": bool(sub and all(bool(row["query_gate"]) for row in sub)),
                "min_heldout_ci_low": float(np.nanmin([row["heldout_ci_low"] for row in sub])) if sub else float("nan"),
                "max_wrong_control_q95": float(np.nanmax([row["wrong_control_qmax"] for row in sub])) if sub else float("nan"),
            }
        )

    status = (
        "zscape_embryo_heldout_periderm_module_specificity_pass_no_gpu"
        if row_summary and all(bool(row["all_query_gates"]) for row in row_summary)
        else "zscape_embryo_heldout_periderm_module_specificity_partial_or_fail_no_gpu"
    )

    summary_csv = args.out_dir / "zscape_embryo_heldout_periderm_module_specificity_summary.csv"
    heldout_csv = args.out_dir / "zscape_embryo_heldout_periderm_module_specificity_heldout_rows.csv"
    wrong_csv = args.out_dir / "zscape_embryo_heldout_periderm_module_specificity_wrong_controls.csv"
    row_csv = args.out_dir / "zscape_embryo_heldout_periderm_module_specificity_row_summary.csv"
    write_csv(summary_csv, summary_rows)
    write_csv(heldout_csv, heldout_rows)
    write_csv(wrong_csv, wrong_rows)
    write_csv(row_csv, row_summary)

    payload = {
        "timestamp": now_cst(),
        "status": status,
        "gpu_authorized_next": False,
        "queries": len(summary_rows),
        "row_summary": row_summary,
        "outputs": {
            "summary_csv": str(summary_csv),
            "heldout_csv": str(heldout_csv),
            "wrong_csv": str(wrong_csv),
            "row_summary_csv": str(row_csv),
        },
    }
    json_path = args.out_dir / "zscape_embryo_heldout_periderm_module_specificity_20260628.json"
    write_json(json_path, payload)

    lines = [
        "# ZSCAPE Embryo-Heldout Periderm Module Specificity Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized next: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only expression-space gate over OT-positive periderm `noto/smo` rows.",
        "- Uses selected raw counts normalized to 1e4 and one `log1p`; module scores are QC-residualized.",
        "- Each perturb embryo is treated as a held-out replicate and compared against control embryo means.",
        "- Wrong-target/time and wrong-lineage rows are scored with the same module and direction.",
        "- No latent extraction, model training, checkpoint selection, canonical multi, or Track C query use.",
        "",
        "## Query Summary",
        "",
        "| query | heldout embryos | mean effect | CI low | positive fraction | wrong q95 max | heldout gate | specificity | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {query} | {emb}/{ctrl} | {mean} | {ci} | {pos} | {wrong} | `{held}` | `{spec}` | `{gate}` |".format(
                query=row["query_name"],
                emb=row["n_perturb_embryos"],
                ctrl=row["n_control_embryos"],
                mean=fmt(row["heldout_mean_effect"]),
                ci=fmt(row["heldout_ci_low"]),
                pos=fmt(row["heldout_positive_fraction"]),
                wrong=fmt(row["wrong_control_qmax"]),
                held=row["heldout_gate"],
                spec=row["specificity_gate"],
                gate=row["query_gate"],
            )
        )
    lines.extend(
        [
            "",
            "## Row Summary",
            "",
            "| row | query gates | min heldout CI low | max wrong-control q95 | all pass |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in row_summary:
        lines.append(
            f"| `{row['row_id']}` | {row['query_gates']}/{row['queries']} | "
            f"{fmt(row['min_heldout_ci_low'])} | {fmt(row['max_wrong_control_q95'])} | `{row['all_query_gates']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- A pass would support a stronger expression-space periderm dynamic module insight, still not a LatentFM training constraint.",
            "- A fail means ZSCAPE remains a hypothesis generator and scaling covariate source only.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{json_path}`",
            f"- Query summary: `{summary_csv}`",
            f"- Heldout rows: `{heldout_csv}`",
            f"- Wrong controls: `{wrong_csv}`",
        ]
    )
    md_path = args.out_dir / "LATENTFM_ZSCAPE_EMBRYO_HELDOUT_PERIDERM_MODULE_SPECIFICITY_20260628.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
