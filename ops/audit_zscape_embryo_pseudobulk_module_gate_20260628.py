#!/usr/bin/env python3
"""Embryo-level ZSCAPE module replicate gate.

This CPU-only audit reuses the frozen ZSCAPE selected-count matrix and
g:Profiler-derived module gene lists. It tests whether module shifts that pass
cell-level scoring also persist when each embryo is treated as the replicate.
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


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_ENRICHMENT_SUMMARY = (
    ROOT
    / "reports/zscape_formal_gprofiler_enrichment_20260628"
    / "zscape_formal_gprofiler_enrichment_20260628_130129"
    / "zscape_gprofiler_enrichment_summary.csv"
)
DEFAULT_OUT = ROOT / "reports/zscape_embryo_pseudobulk_module_gate_20260628"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def embryo_groups(
    manifest: list[dict[str, str]],
    old_to_new: dict[int, int],
    row_id: str,
) -> dict[str, dict[str, list[int]]]:
    groups: dict[str, dict[str, list[int]]] = {"perturb": defaultdict(list), "control": defaultdict(list)}
    for old_idx, meta in enumerate(manifest):
        if meta.get("row_id") != row_id:
            continue
        new_idx = old_to_new.get(old_idx)
        if new_idx is None:
            continue
        role = meta.get("selection_role", "")
        if role not in groups:
            continue
        embryo = meta.get("embryo") or meta.get("sample") or f"cell_{old_idx}"
        groups[role][embryo].append(new_idx)
    return groups


def embryo_module_scores(
    matrix,
    gene_idx: list[int],
    groups: dict[str, dict[str, list[int]]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not gene_idx:
        return np.asarray([]), np.asarray([]), {}
    sub = matrix[np.asarray(gene_idx, dtype=int), :]
    cell_scores = np.asarray(sub.mean(axis=0)).ravel().astype(float)
    out: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    for role in ("perturb", "control"):
        vals = []
        cell_counts = []
        for _, cols in sorted(groups[role].items()):
            if not cols:
                continue
            arr = cell_scores[np.asarray(cols, dtype=int)]
            vals.append(float(np.mean(arr)))
            cell_counts.append(len(cols))
        out[role] = np.asarray(vals, dtype=float)
        diagnostics[f"n_{role}_embryos"] = int(len(vals))
        diagnostics[f"n_{role}_cells"] = int(sum(cell_counts))
        diagnostics[f"median_{role}_cells_per_embryo"] = (
            float(np.median(cell_counts)) if cell_counts else 0.0
        )
    return out["perturb"], out["control"], diagnostics


def bootstrap_ci(
    perturb: np.ndarray,
    control: np.ndarray,
    sign: float,
    seed: int,
    repeats: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(repeats):
        p = perturb[rng.integers(0, perturb.size, size=perturb.size)]
        c = control[rng.integers(0, control.size, size=control.size)]
        vals.append(sign * float(np.mean(p) - np.mean(c)))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def effect_size(perturb: np.ndarray, control: np.ndarray, sign: float) -> float:
    if perturb.size < 2 or control.size < 2:
        return float("nan")
    sd = math.sqrt(float((np.var(perturb, ddof=1) + np.var(control, ddof=1)) / 2.0) + 1e-12)
    return sign * float(np.mean(perturb) - np.mean(control)) / sd


def run_variant(
    counts,
    manifest: list[dict[str, str]],
    qc_flags: np.ndarray,
    gene_symbols: list[str],
    query_rows: list[dict[str, str]],
    apply_qc: bool,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    keep_old = np.where(qc_flags if apply_qc else np.ones(len(manifest), dtype=bool))[0]
    old_to_new = {int(old): int(new) for new, old in enumerate(keep_old)}
    matrix = base.lognorm_counts(counts, keep_old)
    sym_to_idx = symbol_index(gene_symbols)
    rows: list[dict[str, Any]] = []
    for query in query_rows:
        row_id = query.get("row_id", "")
        direction = query.get("direction", "")
        sign = 1.0 if direction == "up" else -1.0
        symbols = parse_symbols(query.get("top_symbols", ""))[: args.max_symbols]
        gene_idx: list[int] = []
        missing = []
        for symbol in symbols:
            hits = sym_to_idx.get(symbol.lower(), [])
            if hits:
                gene_idx.extend(hits)
            else:
                missing.append(symbol)
        gene_idx = sorted(set(gene_idx))
        groups = embryo_groups(manifest, old_to_new, row_id)
        perturb, control, diag = embryo_module_scores(matrix, gene_idx, groups)
        if perturb.size and control.size:
            raw = float(np.mean(perturb) - np.mean(control))
            directed = sign * raw
        else:
            raw = float("nan")
            directed = float("nan")
        if perturb.size >= 2 and control.size >= 2:
            ttest = stats.ttest_ind(perturb, control, equal_var=False)
            p_value = float(ttest.pvalue) if math.isfinite(float(ttest.pvalue)) else 1.0
            ci_low, ci_high = bootstrap_ci(
                perturb,
                control,
                sign,
                stable_seed(f"{row_id}|{direction}|{apply_qc}|embryo"),
                args.bootstrap_repeats,
            )
        else:
            p_value = 1.0
            ci_low, ci_high = float("nan"), float("nan")
        d_value = effect_size(perturb, control, sign)
        gate = bool(
            diag.get("n_perturb_embryos", 0) >= args.min_embryos
            and diag.get("n_control_embryos", 0) >= args.min_embryos
            and np.isfinite(ci_low)
            and ci_low > 0.0
            and directed > 0.0
            and p_value <= args.max_p_value
        )
        rows.append(
            {
                "qc_filtered": bool(apply_qc),
                "query_name": query.get("query_name", ""),
                "row_id": row_id,
                "direction": direction,
                "lineage": query.get("lineage", ""),
                "target": query.get("target", ""),
                "timepoint": query.get("timepoint", ""),
                "constraint_feasibility_class": query.get("constraint_feasibility_class", ""),
                "strict_row_gate": query.get("strict_row_gate", ""),
                "trajectory_alignment_gate": query.get("trajectory_alignment_gate", ""),
                "significant_term_count": query.get("significant_term_count", ""),
                "n_module_symbols": len(symbols),
                "n_mapped_genes": len(gene_idx),
                "n_missing_symbols": len(missing),
                **diag,
                "mean_embryo_score_perturb": float(np.mean(perturb)) if perturb.size else float("nan"),
                "mean_embryo_score_control": float(np.mean(control)) if control.size else float("nan"),
                "raw_embryo_mean_diff": raw,
                "directed_embryo_mean_diff": directed,
                "directed_diff_ci95_low": ci_low,
                "directed_diff_ci95_high": ci_high,
                "welch_p_value": p_value,
                "directed_cohen_d": d_value,
                "embryo_module_gate": gate,
                "top_terms": query.get("top_terms", ""),
                "missing_symbols": ";".join(missing[:20]),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts-npz", type=Path, default=base.DEFAULT_COUNTS)
    parser.add_argument("--cell-index", type=Path, default=base.DEFAULT_CELL_INDEX)
    parser.add_argument("--matched-manifest", type=Path, default=base.DEFAULT_MANIFEST)
    parser.add_argument("--gene-names", type=Path, default=base.DEFAULT_GENES)
    parser.add_argument("--gene-metadata", type=Path, default=base.DEFAULT_GENE_META)
    parser.add_argument("--enrichment-summary", type=Path, default=DEFAULT_ENRICHMENT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-umi", type=float, default=100.0)
    parser.add_argument("--min-genes", type=float, default=100.0)
    parser.add_argument("--max-symbols", type=int, default=50)
    parser.add_argument("--min-embryos", type=int, default=3)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--max-p-value", type=float, default=0.10)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = base.load_csc(args.counts_npz)
    manifest = base.index_manifest(args.cell_index, args.matched_manifest, counts.shape[1])
    gene_ids = [line.strip() for line in args.gene_names.read_text(encoding="utf-8").splitlines() if line.strip()]
    symbol_map = base.read_gene_metadata(args.gene_metadata)
    gene_symbols = [symbol_map.get(gid, gid) for gid in gene_ids]
    qc_flags = np.asarray([base.qc_pass(row, args.min_umi, args.min_genes) for row in manifest], dtype=bool)
    query_rows = [
        row
        for row in read_csv(args.enrichment_summary)
        if row.get("audit_role") == "primary_mechanism_test"
        and row.get("direction") in {"up", "down"}
        and int(float_or_zero(row.get("significant_term_count"))) > 0
    ]

    rows: list[dict[str, Any]] = []
    rows.extend(run_variant(counts, manifest, qc_flags, gene_symbols, query_rows, False, args))
    rows.extend(run_variant(counts, manifest, qc_flags, gene_symbols, query_rows, True, args))

    fields = [
        "qc_filtered",
        "query_name",
        "row_id",
        "direction",
        "lineage",
        "target",
        "timepoint",
        "constraint_feasibility_class",
        "strict_row_gate",
        "trajectory_alignment_gate",
        "significant_term_count",
        "n_module_symbols",
        "n_mapped_genes",
        "n_missing_symbols",
        "n_perturb_embryos",
        "n_control_embryos",
        "n_perturb_cells",
        "n_control_cells",
        "median_perturb_cells_per_embryo",
        "median_control_cells_per_embryo",
        "mean_embryo_score_perturb",
        "mean_embryo_score_control",
        "raw_embryo_mean_diff",
        "directed_embryo_mean_diff",
        "directed_diff_ci95_low",
        "directed_diff_ci95_high",
        "welch_p_value",
        "directed_cohen_d",
        "embryo_module_gate",
        "top_terms",
        "missing_symbols",
    ]
    rows_csv = args.out_dir / "zscape_embryo_pseudobulk_module_rows.csv"
    write_csv(rows_csv, rows, fields)

    primary_no_qc = [r for r in rows if not r["qc_filtered"]]
    primary_qc = [r for r in rows if r["qc_filtered"]]
    periderm_no_qc = [r for r in primary_no_qc if r["lineage"] == "periderm"]
    muscle_no_qc = [r for r in primary_no_qc if r["lineage"] == "mature fast muscle"]
    best_periderm = [
        r
        for r in periderm_no_qc
        if truthy(r["strict_row_gate"])
        and truthy(r["trajectory_alignment_gate"])
        and r["target"] in {"noto", "smo"}
    ]
    qc_by_query = {r["query_name"]: r for r in primary_qc}
    stable = 0
    for row in primary_no_qc:
        other = qc_by_query.get(row["query_name"])
        if other and bool(row["embryo_module_gate"]) == bool(other["embryo_module_gate"]):
            stable += 1
    stability_fraction = stable / max(len(primary_no_qc), 1)
    status = "zscape_embryo_pseudobulk_module_gate_ready_no_gpu"
    if best_periderm and sum(bool(r["embryo_module_gate"]) for r in best_periderm) >= 2:
        status = "zscape_embryo_pseudobulk_module_gate_periderm_support_no_gpu"
    elif best_periderm:
        status = "zscape_embryo_pseudobulk_module_gate_partial_no_gpu"

    json_path = args.out_dir / "zscape_embryo_pseudobulk_module_gate_20260628.json"
    json_path.write_text(
        json.dumps(
            {
                "timestamp_cst": now_cst(),
                "status": status,
                "gpu_authorized": False,
                "runtime_classification": "short_task",
                "query_rows": len(query_rows),
                "no_qc_rows": len(primary_no_qc),
                "qc_rows": len(primary_qc),
                "no_qc_gate_count": int(sum(bool(r["embryo_module_gate"]) for r in primary_no_qc)),
                "periderm_no_qc_gate_count": int(sum(bool(r["embryo_module_gate"]) for r in periderm_no_qc)),
                "muscle_no_qc_gate_count": int(sum(bool(r["embryo_module_gate"]) for r in muscle_no_qc)),
                "best_periderm_gate_count": int(sum(bool(r["embryo_module_gate"]) for r in best_periderm)),
                "qc_gate_stability_fraction": stability_fraction,
                "rows_csv": str(rows_csv),
                "input_counts": str(args.counts_npz),
                "enrichment_summary": str(args.enrichment_summary),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    md_path = args.out_dir / "LATENTFM_ZSCAPE_EMBRYO_PSEUDOBULK_MODULE_GATE_20260628.md"
    lines = [
        "# LatentFM ZSCAPE Embryo Pseudobulk Module Gate",
        "",
        f"Timestamp: `{now_cst()}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only replicate-level check over frozen ZSCAPE selected counts.",
        "- Uses raw counts -> size-factor normalization -> exactly one `log1p`.",
        "- Treats each embryo as the replicate for module scores.",
        "- Does not train, infer, extract true scFM embeddings, read canonical multi, or read Track C query.",
        "",
        "## Summary",
        "",
        f"- query modules tested: `{len(query_rows)}`.",
        f"- no-QC embryo-module gates: `{sum(bool(r['embryo_module_gate']) for r in primary_no_qc)}/{len(primary_no_qc)}`.",
        f"- periderm no-QC gates: `{sum(bool(r['embryo_module_gate']) for r in periderm_no_qc)}/{len(periderm_no_qc)}`.",
        f"- mature-fast-muscle no-QC gates: `{sum(bool(r['embryo_module_gate']) for r in muscle_no_qc)}/{len(muscle_no_qc)}`.",
        f"- best periderm strict+trajectory rows (`noto/smo`) gates: `{sum(bool(r['embryo_module_gate']) for r in best_periderm)}/{len(best_periderm)}`.",
        f"- QC gate stability fraction: `{fmt(stability_fraction)}`.",
        "",
        "## Best Periderm Rows",
        "",
        "| query | embryos perturb/control | directed diff | CI low | CI high | p | gate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_periderm:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["query_name"]),
                    f"{row['n_perturb_embryos']}/{row['n_control_embryos']}",
                    fmt(row["directed_embryo_mean_diff"]),
                    fmt(row["directed_diff_ci95_low"]),
                    fmt(row["directed_diff_ci95_high"]),
                    fmt(row["welch_p_value"], 4),
                    str(row["embryo_module_gate"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This gate can strengthen expression-space biological interpretation, but it does not authorize latent-space or LatentFM model claims.",
            "A positive result supports ZSCAPE as a replicate-aware periderm biological-information axis.",
            "A true latent/flow constraint still requires a species-compatible encoder or frozen orthology-loss audit plus a separate no-harm design gate.",
            "",
            "## Outputs",
            "",
            f"- rows: `{rows_csv}`",
            f"- JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
