#!/usr/bin/env python3
"""Periderm substate/time/QC-residual module gate for ZSCAPE.

This CPU-only gate follows the external audit recommendation after the ZSCAPE
expression branch became a narrow positive.  It tests whether the supported
periderm noto/smo modules survive embryo-replicate uncertainty, periderm
substate checks, wrong-target/wrong-time/wrong-lineage controls, and simple
QC residualization.
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
from scipy.stats import spearmanr

import audit_zscape_expression_latent_biology_preflight_20260628 as base


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_ENRICHMENT_SUMMARY = (
    ROOT
    / "reports/zscape_formal_gprofiler_enrichment_20260628"
    / "zscape_formal_gprofiler_enrichment_20260628_130129"
    / "zscape_gprofiler_enrichment_summary.csv"
)
DEFAULT_SNAPSHOT_ROWS = (
    ROOT
    / "reports"
    / "zscape_snapshot_dynamic_constraint_spec_20260628"
    / "zscape_snapshot_dynamic_constraint_rows.csv"
)
DEFAULT_OUT = ROOT / "reports" / "zscape_periderm_substate_time_qc_ot_module_gate_20260628"
PRIMARY_ROW_IDS = {"periderm__noto__24p0h", "periderm__smo__24p0h"}


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
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


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


def parse_symbols(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in str(text or "").split(";"):
        symbol = item.strip()
        if not symbol:
            continue
        key = symbol.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(symbol)
    return out


def symbol_index(gene_symbols: list[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for idx, symbol in enumerate(gene_symbols):
        out.setdefault(symbol.lower(), []).append(idx)
    return out


def mito_like(symbol: str) -> bool:
    s = symbol.strip().lower()
    return s.startswith("mt-") or s.startswith("mt.") or s.startswith("nc_002333")


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return float("nan")
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return float("nan")
    rho, _ = spearmanr(a, b)
    return float(rho) if math.isfinite(float(rho)) else float("nan")


def row_indices(manifest: list[dict[str, str]], row_id: str) -> dict[str, list[int]]:
    out = {"perturb": [], "control": []}
    for idx, row in enumerate(manifest):
        if row.get("row_id") != row_id:
            continue
        role = row.get("selection_role", "")
        if role in out:
            out[role].append(idx)
    return out


def role_indices_for_row_ids(
    manifest: list[dict[str, str]],
    row_ids: set[str],
) -> dict[str, dict[str, list[int]]]:
    out: dict[str, dict[str, list[int]]] = {
        row_id: {"perturb": [], "control": []} for row_id in sorted(row_ids)
    }
    for idx, row in enumerate(manifest):
        row_id = row.get("row_id", "")
        if row_id not in out:
            continue
        role = row.get("selection_role", "")
        if role in {"perturb", "control"}:
            out[row_id][role].append(idx)
    return out


def embryo_values(
    manifest: list[dict[str, str]],
    values: np.ndarray,
    indices: list[int],
) -> tuple[np.ndarray, dict[str, int]]:
    by_embryo: dict[str, list[float]] = defaultdict(list)
    for idx in indices:
        row = manifest[idx]
        embryo = row.get("embryo") or row.get("sample") or f"cell_{idx}"
        by_embryo[embryo].append(float(values[idx]))
    arr = np.asarray([np.mean(v) for _, v in sorted(by_embryo.items())], dtype=float)
    return arr, {key: len(val) for key, val in by_embryo.items()}


def directed_diff(
    perturb: np.ndarray,
    control: np.ndarray,
    sign: float,
    seed: int,
    repeats: int,
) -> dict[str, float]:
    if perturb.size == 0 or control.size == 0:
        return {
            "raw_diff": float("nan"),
            "directed_diff": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p": float("nan"),
            "cohen_d": float("nan"),
        }
    raw = float(np.mean(perturb) - np.mean(control))
    directed = sign * raw
    if perturb.size >= 2 and control.size >= 2:
        ttest = stats.ttest_ind(perturb, control, equal_var=False)
        p_value = float(ttest.pvalue) if math.isfinite(float(ttest.pvalue)) else 1.0
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(repeats):
            p = perturb[rng.integers(0, perturb.size, size=perturb.size)]
            c = control[rng.integers(0, control.size, size=control.size)]
            boots.append(sign * float(np.mean(p) - np.mean(c)))
        ci_low, ci_high = np.quantile(boots, [0.025, 0.975])
        sd = math.sqrt(float((np.var(perturb, ddof=1) + np.var(control, ddof=1)) / 2.0) + 1e-12)
        cohen = directed / sd
    else:
        p_value = 1.0
        ci_low = ci_high = float("nan")
        cohen = float("nan")
    return {
        "raw_diff": raw,
        "directed_diff": directed,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p": p_value,
        "cohen_d": cohen,
    }


def residualize(scores: np.ndarray, manifest: list[dict[str, str]]) -> np.ndarray:
    log_umi = np.log1p(np.asarray([base.as_float(row, "n_umi") for row in manifest], dtype=float))
    log_genes = np.log1p(np.asarray([base.as_float(row, "num_genes_expressed") for row in manifest], dtype=float))
    design = np.vstack([np.ones(scores.size), log_umi, log_genes]).T
    mask = np.isfinite(scores) & np.all(np.isfinite(design), axis=1)
    resid = np.full(scores.shape, np.nan, dtype=float)
    beta, *_ = np.linalg.lstsq(design[mask], scores[mask], rcond=None)
    resid[mask] = scores[mask] - design[mask] @ beta
    return resid


def substate_rows(
    manifest: list[dict[str, str]],
    raw_scores: np.ndarray,
    resid_scores: np.ndarray,
    row_id: str,
    sign: float,
    min_cells: int,
) -> list[dict[str, Any]]:
    rows = []
    by_sub: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"perturb": [], "control": []})
    for idx, row in enumerate(manifest):
        if row.get("row_id") != row_id:
            continue
        role = row.get("selection_role", "")
        if role not in {"perturb", "control"}:
            continue
        sub = row.get("cell_type_sub") or "unknown"
        by_sub[sub][role].append(idx)
    for sub, roles in sorted(by_sub.items()):
        p_idx = roles["perturb"]
        c_idx = roles["control"]
        if len(p_idx) < min_cells or len(c_idx) < min_cells:
            evaluable = False
            raw_diff = resid_diff = float("nan")
        else:
            evaluable = True
            raw_diff = sign * float(np.nanmean(raw_scores[p_idx]) - np.nanmean(raw_scores[c_idx]))
            resid_diff = sign * float(np.nanmean(resid_scores[p_idx]) - np.nanmean(resid_scores[c_idx]))
        rows.append(
            {
                "row_id": row_id,
                "substate": sub,
                "n_perturb": len(p_idx),
                "n_control": len(c_idx),
                "evaluable": evaluable,
                "raw_directed_diff": raw_diff,
                "residual_directed_diff": resid_diff,
                "positive_residual": bool(evaluable and resid_diff > 0.0),
            }
        )
    return rows


def placebo_rows(
    manifest: list[dict[str, str]],
    values: np.ndarray,
    query_row_id: str,
    sign: float,
    repeats: int,
    label: str,
    row_filter,
) -> list[dict[str, Any]]:
    row_ids = {
        row.get("row_id", "")
        for row in manifest
        if row.get("row_id", "") and row.get("row_id") != query_row_id and row_filter(row)
    }
    roles_by_row = role_indices_for_row_ids(manifest, row_ids)
    rows = []
    for row_id, roles in sorted(roles_by_row.items()):
        p, _ = embryo_values(manifest, values, roles["perturb"])
        c, _ = embryo_values(manifest, values, roles["control"])
        stats_row = directed_diff(
            p,
            c,
            sign,
            stable_seed(f"{query_row_id}|{row_id}|{label}"),
            repeats,
        )
        rows.append(
            {
                "query_row_id": query_row_id,
                "control_type": label,
                "control_row_id": row_id,
                "n_perturb_embryos": int(p.size),
                "n_control_embryos": int(c.size),
                **stats_row,
            }
        )
    return rows


def quantile_or_nan(values: list[float], q: float) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(np.quantile(vals, q))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts-npz", type=Path, default=base.DEFAULT_COUNTS)
    parser.add_argument("--cell-index", type=Path, default=base.DEFAULT_CELL_INDEX)
    parser.add_argument("--matched-manifest", type=Path, default=base.DEFAULT_MANIFEST)
    parser.add_argument("--gene-names", type=Path, default=base.DEFAULT_GENES)
    parser.add_argument("--gene-metadata", type=Path, default=base.DEFAULT_GENE_META)
    parser.add_argument("--enrichment-summary", type=Path, default=DEFAULT_ENRICHMENT_SUMMARY)
    parser.add_argument("--snapshot-rows", type=Path, default=DEFAULT_SNAPSHOT_ROWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-symbols", type=int, default=50)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--min-substate-cells", type=int, default=8)
    parser.add_argument("--min-embryos", type=int, default=4)
    parser.add_argument("--max-p", type=float, default=0.05)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = base.load_csc(args.counts_npz)
    manifest = base.index_manifest(args.cell_index, args.matched_manifest, counts.shape[1])
    keep_cols = np.arange(len(manifest), dtype=int)
    matrix = base.lognorm_counts(counts, keep_cols)
    gene_ids = [line.strip() for line in args.gene_names.read_text(encoding="utf-8").splitlines() if line.strip()]
    symbol_map = base.read_gene_metadata(args.gene_metadata)
    gene_symbols = [symbol_map.get(gid, gid) for gid in gene_ids]
    sym_to_idx = symbol_index(gene_symbols)

    enrichment = [
        row
        for row in read_csv(args.enrichment_summary)
        if row.get("row_id") in PRIMARY_ROW_IDS
        and row.get("direction") in {"up", "down"}
        and int(float(row.get("significant_term_count") or 0)) > 0
    ]
    snapshots = {row["row_id"]: row for row in read_csv(args.snapshot_rows)}

    query_rows: list[dict[str, Any]] = []
    sub_rows: list[dict[str, Any]] = []
    all_placebo_rows: list[dict[str, Any]] = []

    for query in enrichment:
        row_id = query["row_id"]
        direction = query["direction"]
        sign = 1.0 if direction == "up" else -1.0
        symbols = parse_symbols(query.get("top_symbols", ""))[: args.max_symbols]
        gene_idx: list[int] = []
        missing: list[str] = []
        for symbol in symbols:
            hits = sym_to_idx.get(symbol.lower(), [])
            if hits:
                gene_idx.extend(hits)
            else:
                missing.append(symbol)
        gene_idx = sorted(set(gene_idx))
        scores = np.asarray(matrix[np.asarray(gene_idx, dtype=int), :].mean(axis=0)).ravel().astype(float)
        residual = residualize(scores, manifest)
        roles = row_indices(manifest, row_id)
        p_raw, p_cells = embryo_values(manifest, scores, roles["perturb"])
        c_raw, c_cells = embryo_values(manifest, scores, roles["control"])
        p_resid, _ = embryo_values(manifest, residual, roles["perturb"])
        c_resid, _ = embryo_values(manifest, residual, roles["control"])
        raw_stats = directed_diff(
            p_raw,
            c_raw,
            sign,
            stable_seed(f"{row_id}|{direction}|raw"),
            args.bootstrap_repeats,
        )
        resid_stats = directed_diff(
            p_resid,
            c_resid,
            sign,
            stable_seed(f"{row_id}|{direction}|resid"),
            args.bootstrap_repeats,
        )

        target_idx = np.asarray(roles["perturb"] + roles["control"], dtype=int)
        log_umi = np.log1p(np.asarray([base.as_float(manifest[i], "n_umi") for i in target_idx], dtype=float))
        log_genes = np.log1p(
            np.asarray([base.as_float(manifest[i], "num_genes_expressed") for i in target_idx], dtype=float)
        )
        target_scores = scores[target_idx]
        max_abs_qc_rho = np.nanmax(
            np.abs(
                np.asarray(
                    [
                        safe_spearman(target_scores, log_umi),
                        safe_spearman(target_scores, log_genes),
                    ],
                    dtype=float,
                )
            )
        )
        if not math.isfinite(float(max_abs_qc_rho)):
            max_abs_qc_rho = float("nan")

        q_sub_rows = substate_rows(
            manifest,
            scores,
            residual,
            row_id,
            sign,
            args.min_substate_cells,
        )
        sub_rows.extend([{**row, "query_name": query["query_name"], "direction": direction} for row in q_sub_rows])
        eval_sub = [row for row in q_sub_rows if row["evaluable"]]
        substate_positive_fraction = (
            float(np.mean([row["positive_residual"] for row in eval_sub])) if eval_sub else float("nan")
        )
        substate_min_resid = (
            float(np.nanmin([row["residual_directed_diff"] for row in eval_sub])) if eval_sub else float("nan")
        )
        substate_gate = bool(eval_sub and substate_positive_fraction >= 1.0 and substate_min_resid > 0.0)

        periderm_placebo = placebo_rows(
            manifest,
            residual,
            row_id,
            sign,
            args.bootstrap_repeats,
            "periderm_wrong_target_or_time",
            lambda row: row.get("cell_type_broad") == "periderm",
        )
        wrong_time = placebo_rows(
            manifest,
            residual,
            row_id,
            sign,
            args.bootstrap_repeats,
            "periderm_wrong_time",
            lambda row: row.get("cell_type_broad") == "periderm"
            and str(row.get("manifest_timepoint") or row.get("timepoint")) != str(query.get("timepoint")),
        )
        wrong_lineage = placebo_rows(
            manifest,
            residual,
            row_id,
            sign,
            args.bootstrap_repeats,
            "wrong_lineage",
            lambda row: row.get("cell_type_broad") != "periderm",
        )
        all_placebo_rows.extend(periderm_placebo + wrong_time + wrong_lineage)
        periderm_p95 = quantile_or_nan([row["directed_diff"] for row in periderm_placebo], 0.95)
        wrong_time_max = quantile_or_nan([row["directed_diff"] for row in wrong_time], 1.0)
        wrong_lineage_p95 = quantile_or_nan([row["directed_diff"] for row in wrong_lineage], 0.95)
        specificity_gate = bool(
            resid_stats["directed_diff"] > max(0.0, periderm_p95 if math.isfinite(periderm_p95) else 0.0)
            and resid_stats["directed_diff"]
            > max(0.0, wrong_lineage_p95 if math.isfinite(wrong_lineage_p95) else 0.0)
            and (
                not math.isfinite(wrong_time_max)
                or resid_stats["directed_diff"] > max(0.0, wrong_time_max)
            )
        )
        snapshot = snapshots.get(row_id, {})
        snapshot_gate = bool(truthy(snapshot.get("expression_constraint_candidate")))
        qc_residual_gate = bool(
            p_resid.size >= args.min_embryos
            and c_resid.size >= args.min_embryos
            and resid_stats["directed_diff"] > 0.0
            and resid_stats["ci_low"] > 0.0
            and resid_stats["p"] <= args.max_p
        )
        query_gate = bool(qc_residual_gate and substate_gate and specificity_gate and snapshot_gate)
        query_rows.append(
            {
                "query_name": query["query_name"],
                "row_id": row_id,
                "direction": direction,
                "lineage": query["lineage"],
                "target": query["target"],
                "timepoint": query["timepoint"],
                "n_symbols": len(symbols),
                "n_mapped_genes": len(gene_idx),
                "n_missing_symbols": len(missing),
                "mito_like_symbol_fraction": float(np.mean([mito_like(s) for s in symbols])) if symbols else 0.0,
                "n_perturb_embryos": int(p_resid.size),
                "n_control_embryos": int(c_resid.size),
                "median_perturb_cells_per_embryo": float(np.median(list(p_cells.values()))) if p_cells else 0.0,
                "median_control_cells_per_embryo": float(np.median(list(c_cells.values()))) if c_cells else 0.0,
                "raw_directed_diff": raw_stats["directed_diff"],
                "raw_ci_low": raw_stats["ci_low"],
                "raw_p": raw_stats["p"],
                "residual_directed_diff": resid_stats["directed_diff"],
                "residual_ci_low": resid_stats["ci_low"],
                "residual_ci_high": resid_stats["ci_high"],
                "residual_p": resid_stats["p"],
                "residual_cohen_d": resid_stats["cohen_d"],
                "max_abs_raw_qc_spearman": max_abs_qc_rho,
                "evaluable_substates": len(eval_sub),
                "substate_positive_fraction": substate_positive_fraction,
                "substate_min_residual_diff": substate_min_resid,
                "periderm_placebo_p95": periderm_p95,
                "wrong_time_max": wrong_time_max,
                "wrong_lineage_p95": wrong_lineage_p95,
                "snapshot_expression_candidate": snapshot_gate,
                "qc_residual_gate": qc_residual_gate,
                "substate_gate": substate_gate,
                "specificity_gate": specificity_gate,
                "query_gate": query_gate,
                "top_terms": query.get("top_terms", ""),
                "missing_symbols": ";".join(missing[:20]),
            }
        )

    row_summary = []
    for row_id in sorted(PRIMARY_ROW_IDS):
        qs = [row for row in query_rows if row["row_id"] == row_id]
        row_summary.append(
            {
                "row_id": row_id,
                "queries": len(qs),
                "query_gates": sum(bool(row["query_gate"]) for row in qs),
                "all_query_gates": bool(qs and all(bool(row["query_gate"]) for row in qs)),
                "min_residual_ci_low": float(np.nanmin([row["residual_ci_low"] for row in qs])) if qs else float("nan"),
                "max_qc_spearman": float(np.nanmax([row["max_abs_raw_qc_spearman"] for row in qs])) if qs else float("nan"),
            }
        )

    status = (
        "zscape_periderm_substate_time_qc_ot_module_gate_pass_no_gpu"
        if len(query_rows) == 4 and all(bool(row["query_gate"]) for row in query_rows)
        else "zscape_periderm_substate_time_qc_ot_module_gate_partial_or_fail_no_gpu"
    )

    query_csv = args.out_dir / "zscape_periderm_substate_time_qc_module_query_rows.csv"
    substate_csv = args.out_dir / "zscape_periderm_substate_time_qc_module_substate_rows.csv"
    placebo_csv = args.out_dir / "zscape_periderm_substate_time_qc_module_placebo_rows.csv"
    summary_csv = args.out_dir / "zscape_periderm_substate_time_qc_module_row_summary.csv"
    write_csv(query_csv, query_rows)
    write_csv(substate_csv, sub_rows)
    write_csv(placebo_csv, all_placebo_rows)
    write_csv(summary_csv, row_summary)

    payload = {
        "status": status,
        "gpu_authorized": False,
        "inputs": {
            "counts_npz": str(args.counts_npz),
            "matched_manifest": str(args.matched_manifest),
            "enrichment_summary": str(args.enrichment_summary),
            "snapshot_rows": str(args.snapshot_rows),
        },
        "query_rows": query_rows,
        "row_summary": row_summary,
        "outputs": {
            "query_csv": str(query_csv),
            "substate_csv": str(substate_csv),
            "placebo_csv": str(placebo_csv),
            "summary_csv": str(summary_csv),
        },
    }
    json_path = args.out_dir / "zscape_periderm_substate_time_qc_ot_module_gate_20260628.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    md_path = args.out_dir / "LATENTFM_ZSCAPE_PERIDERM_SUBSTATE_TIME_QC_OT_MODULE_GATE_20260628.md"
    lines = [
        "# LatentFM ZSCAPE Periderm Substate/Time/QC Module Gate",
        "",
        f"Timestamp: `{now_cst()}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only expression-space gate over frozen ZSCAPE selected counts and reports.",
        "- Uses raw counts -> size-factor normalization -> exactly one `log1p`.",
        "- Tests periderm `noto/smo` enriched modules after embryo aggregation, QC residualization, substate checks, wrong-target/time controls, and wrong-lineage controls.",
        "- Does not train, infer, extract true scFM embeddings, select checkpoints, use canonical multi, or read Track C query.",
        "",
        "## Query Gate Summary",
        "",
        "| query | residual diff | CI low | p | QC rho | substate | specificity | snapshot | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in query_rows:
        lines.append(
            "| {query} | {diff} | {ci} | {p} | {rho} | {sub} | {spec} | {snap} | {gate} |".format(
                query=row["query_name"],
                diff=fmt(row["residual_directed_diff"]),
                ci=fmt(row["residual_ci_low"]),
                p=fmt(row["residual_p"], 4),
                rho=fmt(row["max_abs_raw_qc_spearman"]),
                sub=str(row["substate_gate"]),
                spec=str(row["specificity_gate"]),
                snap=str(row["snapshot_expression_candidate"]),
                gate=str(row["query_gate"]),
            )
        )
    lines.extend(
        [
            "",
            "## Row Summary",
            "",
            "| row | query gates | min residual CI low | max QC rho | all pass |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in row_summary:
        lines.append(
            f"| {row['row_id']} | {row['query_gates']}/{row['queries']} | "
            f"{fmt(row['min_residual_ci_low'])} | {fmt(row['max_qc_spearman'])} | {row['all_query_gates']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A pass supports the narrow expression-space claim that periderm `noto/smo` programs are not explained by simple library/gene-count QC structure, are visible within evaluable substates, and are stronger than matched wrong-target/time or wrong-lineage module reuse. It still does not authorize latent constraints or GPU training.",
            "",
            "A partial/fail keeps ZSCAPE as biological insight and scaling-axis evidence but blocks using these modules as model constraints until the failed component is repaired with a new hypothesis.",
            "",
            "## Outputs",
            "",
            f"- query rows: `{query_csv}`",
            f"- substate rows: `{substate_csv}`",
            f"- placebo rows: `{placebo_csv}`",
            f"- row summary: `{summary_csv}`",
            f"- JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": status, "out_dir": str(args.out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
