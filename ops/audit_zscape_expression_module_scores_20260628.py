#!/usr/bin/env python3
"""ZSCAPE expression-space module score audit.

Scores frozen DE/enrichment query gene modules per cell to test whether
pathway-like up/down signatures show directional perturb-control shifts and
whether those shifts are stable to the simple QC rule.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import audit_zscape_expression_latent_biology_preflight_20260628 as base


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_ENRICHMENT_SUMMARY = (
    ROOT
    / "reports/zscape_formal_gprofiler_enrichment_20260628"
    / "zscape_formal_gprofiler_enrichment_20260628_130129"
    / "zscape_gprofiler_enrichment_summary.csv"
)
DEFAULT_OUT = ROOT / "reports/zscape_expression_module_scores_20260628"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        base.write_csv(path, rows, list(rows[0].keys()))
    else:
        path.write_text("", encoding="utf-8")


def symbol_index(gene_symbols: list[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for idx, symbol in enumerate(gene_symbols):
        key = symbol.lower()
        out.setdefault(key, []).append(idx)
    return out


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


def cell_groups(manifest: list[dict[str, str]], old_to_new: dict[int, int]) -> dict[str, dict[str, list[int]]]:
    groups: dict[str, dict[str, list[int]]] = {}
    for old_idx, row in enumerate(manifest):
        new_idx = old_to_new.get(old_idx)
        if new_idx is None:
            continue
        row_id = row.get("row_id", "")
        role = row.get("selection_role", "")
        if role not in {"perturb", "control"}:
            continue
        groups.setdefault(row_id, {"perturb": [], "control": []})[role].append(new_idx)
    return groups


def scores_for_cols(matrix, gene_idx: list[int], cols: list[int]) -> np.ndarray:
    if not gene_idx or not cols:
        return np.asarray([], dtype=float)
    sub = matrix[np.asarray(gene_idx, dtype=int), :][:, np.asarray(cols, dtype=int)]
    return np.asarray(sub.mean(axis=0)).ravel().astype(float)


def bootstrap_ci(
    perturb: np.ndarray,
    control: np.ndarray,
    direction: str,
    seed: int,
    repeats: int,
) -> tuple[float, float, float, float]:
    if perturb.size == 0 or control.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    raw_diff = float(np.mean(perturb) - np.mean(control))
    sign = 1.0 if direction == "up" else -1.0
    directed = sign * raw_diff
    pooled_sd = float(np.sqrt((np.var(perturb) + np.var(control)) / 2.0 + 1e-12))
    effect = directed / pooled_sd
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(repeats):
        p = perturb[rng.integers(0, perturb.size, size=perturb.size)]
        c = control[rng.integers(0, control.size, size=control.size)]
        vals.append(sign * float(np.mean(p) - np.mean(c)))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return directed, float(lo), float(hi), effect


def run_one(
    counts,
    manifest: list[dict[str, str]],
    qc_flags: np.ndarray,
    query_rows: list[dict[str, str]],
    gene_symbols: list[str],
    apply_qc: bool,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    keep_old = np.where(qc_flags if apply_qc else np.ones(len(manifest), dtype=bool))[0]
    old_to_new = {int(old): int(new) for new, old in enumerate(keep_old)}
    groups = cell_groups(manifest, old_to_new)
    matrix = base.lognorm_counts(counts, keep_old)
    sym_to_idx = symbol_index(gene_symbols)
    rows: list[dict[str, Any]] = []
    for query in query_rows:
        row_id = query.get("row_id", "")
        direction = query.get("direction", "")
        if direction not in {"up", "down"}:
            continue
        group = groups.get(row_id)
        if not group:
            continue
        symbols = parse_symbols(query.get("top_symbols", ""))
        gene_idx: list[int] = []
        missing: list[str] = []
        for symbol in symbols[: args.max_symbols]:
            hits = sym_to_idx.get(symbol.lower(), [])
            if hits:
                gene_idx.extend(hits)
            else:
                missing.append(symbol)
        gene_idx = sorted(set(gene_idx))
        p = scores_for_cols(matrix, gene_idx, group["perturb"])
        c = scores_for_cols(matrix, gene_idx, group["control"])
        directed, ci_lo, ci_hi, effect = bootstrap_ci(
            p,
            c,
            direction,
            seed=base.stable_seed(f"{row_id}|{direction}|{apply_qc}") if hasattr(base, "stable_seed") else args.seed,
            repeats=args.bootstrap_repeats,
        )
        gate = bool(np.isfinite(ci_lo) and ci_lo > 0.0 and effect > args.min_effect)
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
                "n_module_symbols": len(symbols[: args.max_symbols]),
                "n_mapped_genes": len(gene_idx),
                "n_missing_symbols": len(missing),
                "n_perturb": int(p.size),
                "n_control": int(c.size),
                "mean_score_perturb": float(np.mean(p)) if p.size else float("nan"),
                "mean_score_control": float(np.mean(c)) if c.size else float("nan"),
                "directed_mean_diff": directed,
                "directed_diff_ci95_low": ci_lo,
                "directed_diff_ci95_high": ci_hi,
                "directed_cohen_d": effect,
                "module_direction_gate": gate,
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
    parser.add_argument("--bootstrap-repeats", type=int, default=300)
    parser.add_argument("--min-effect", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260628)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = base.load_csc(args.counts_npz)
    manifest = base.index_manifest(args.cell_index, args.matched_manifest, counts.shape[1])
    gene_ids = [line.strip() for line in args.gene_names.read_text(encoding="utf-8").splitlines() if line.strip()]
    symbol_map = base.read_gene_metadata(args.gene_metadata)
    gene_symbols = [symbol_map.get(gid, gid) for gid in gene_ids]
    qc_flags = np.asarray([base.qc_pass(row, args.min_umi, args.min_genes) for row in manifest], dtype=bool)

    raw_queries = read_csv(args.enrichment_summary)
    query_rows = [
        row
        for row in raw_queries
        if row.get("audit_role") == "primary_mechanism_test"
        and int(float(row.get("significant_term_count") or 0)) > 0
    ]

    all_rows = []
    all_rows.extend(run_one(counts, manifest, qc_flags, query_rows, gene_symbols, False, args))
    all_rows.extend(run_one(counts, manifest, qc_flags, query_rows, gene_symbols, True, args))

    by_query = {}
    for row in all_rows:
        by_query.setdefault((row["query_name"], bool(row["qc_filtered"])), row)
    stability_rows: list[dict[str, Any]] = []
    for query in sorted({row["query_name"] for row in all_rows}):
        a = by_query.get((query, False))
        b = by_query.get((query, True))
        if not a or not b:
            continue
        diff_delta = float(b["directed_mean_diff"]) - float(a["directed_mean_diff"])
        stability_rows.append(
            {
                "query_name": query,
                "row_id": a["row_id"],
                "direction": a["direction"],
                "lineage": a["lineage"],
                "target": a["target"],
                "no_qc_directed_diff": a["directed_mean_diff"],
                "qc_directed_diff": b["directed_mean_diff"],
                "qc_minus_no_qc_directed_diff": diff_delta,
                "no_qc_gate": a["module_direction_gate"],
                "qc_gate": b["module_direction_gate"],
                "gate_stable": bool(a["module_direction_gate"]) == bool(b["module_direction_gate"]),
            }
        )

    no_qc = [row for row in all_rows if not row["qc_filtered"]]
    periderm_best = [
        row
        for row in no_qc
        if row["constraint_feasibility_class"] == "best_candidate_pending_fixedcell_placebo"
    ]
    periderm_pass = sum(bool(row["module_direction_gate"]) for row in periderm_best)
    gate_stable_frac = (
        float(np.mean([bool(row["gate_stable"]) for row in stability_rows])) if stability_rows else float("nan")
    )
    status = (
        "zscape_expression_module_scores_pass_no_gpu"
        if periderm_best and periderm_pass == len(periderm_best) and gate_stable_frac >= 0.95
        else "zscape_expression_module_scores_partial_no_gpu"
    )

    row_csv = args.out_dir / "zscape_expression_module_score_rows.csv"
    stability_csv = args.out_dir / "zscape_expression_module_score_qc_stability.csv"
    json_path = args.out_dir / "zscape_expression_module_scores_20260628.json"
    md_path = args.out_dir / "LATENTFM_ZSCAPE_EXPRESSION_MODULE_SCORES_20260628.md"
    write_csv(row_csv, all_rows)
    write_csv(stability_csv, stability_rows)

    result = {
        "timestamp_cst": now_cst(),
        "status": status,
        "gpu_authorized": False,
        "n_cells_before_qc": int(len(manifest)),
        "n_cells_after_qc": int(qc_flags.sum()),
        "n_primary_significant_queries": int(len(query_rows)),
        "periderm_best_module_gate_pass": int(periderm_pass),
        "periderm_best_module_gate_total": int(len(periderm_best)),
        "qc_gate_stable_fraction": gate_stable_frac,
        "outputs": {"rows": str(row_csv), "qc_stability": str(stability_csv)},
    }
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM ZSCAPE Expression Module Scores",
        "",
        f"Timestamp: `{now_cst()}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only expression-space module scoring over selected ZSCAPE cells.",
        "- Uses size-factor normalization and exactly one `log1p`.",
        "- No model training, no inference, no true scFM embedding extraction, no canonical multi, and no Track C query.",
        "",
        "## Summary",
        "",
        f"- cells after QC rule: `{len(manifest)} -> {int(qc_flags.sum())}`.",
        f"- primary significant query modules scored: `{len(query_rows)}`.",
        f"- best-candidate periderm module gates: `{periderm_pass}/{len(periderm_best)}`.",
        f"- QC gate stability fraction: `{gate_stable_frac:.4f}`.",
        "",
        "## Primary No-QC Module Rows",
        "",
        "| query | lineage | target | direction | directed diff | CI low | CI high | d | gate | top terms |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in no_qc:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["query_name"]),
                    str(row["lineage"]),
                    str(row["target"]),
                    str(row["direction"]),
                    f"{float(row['directed_mean_diff']):.4f}",
                    f"{float(row['directed_diff_ci95_low']):.4f}",
                    f"{float(row['directed_diff_ci95_high']):.4f}",
                    f"{float(row['directed_cohen_d']):.4f}",
                    str(row["module_direction_gate"]),
                    str(row["top_terms"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- Module-score support is expression-space evidence only; it cannot authorize model training or a mechanism claim by itself.",
            "- Periderm electron-transport and intermediate-filament modules still need fixed-cell/placebo controls to exclude cell-state or sampling confounds.",
            "- Muscle modules are retained as negative/confounded controls under the current strict-gate interpretation.",
            "",
            "## Outputs",
            "",
            f"- row scores: `{row_csv}`",
            f"- QC stability: `{stability_csv}`",
            f"- JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
