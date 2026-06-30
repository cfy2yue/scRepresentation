#!/usr/bin/env python3
"""Embryo-heldout specificity gate for ZSCAPE periderm dynamic modules.

This CPU-only gate asks a narrower question than the original module screen:
given the current fixed periderm noto/smo module hypotheses, are their effects
stable in held-out perturb embryos and stronger than wrong controls?

The module gene sets are not rediscovered inside each split, so a pass would be
replication evidence, not a de-novo pathway discovery claim.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import audit_zscape_expression_latent_biology_preflight_20260628 as base
import audit_zscape_periderm_substate_time_qc_ot_module_gate_20260628 as module_gate


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_OUT = ROOT / "reports/zscape_embryo_heldout_dynamic_specificity_gate_20260628"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


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


def quantile(values: list[float], q: float) -> float:
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, q))


def embryo_index_map(
    manifest: list[dict[str, str]],
    indices: list[int],
) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for idx in indices:
        embryo = manifest[idx].get("embryo") or manifest[idx].get("sample") or f"cell_{idx}"
        out[str(embryo)].append(idx)
    return dict(out)


def embryo_mean_values(
    manifest: list[dict[str, str]],
    values: np.ndarray,
    indices: list[int],
) -> np.ndarray:
    arr, _ = module_gate.embryo_values(manifest, values, indices)
    return arr


def split_iter(embryos: list[str], heldout_size: int, max_splits: int, seed: int) -> list[tuple[str, ...]]:
    combos = list(itertools.combinations(sorted(embryos), heldout_size))
    if len(combos) <= max_splits:
        return combos
    rng = np.random.default_rng(seed)
    take = rng.choice(len(combos), size=max_splits, replace=False)
    return [combos[int(i)] for i in sorted(take)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts-npz", type=Path, default=base.DEFAULT_COUNTS)
    parser.add_argument("--cell-index", type=Path, default=base.DEFAULT_CELL_INDEX)
    parser.add_argument("--matched-manifest", type=Path, default=base.DEFAULT_MANIFEST)
    parser.add_argument("--gene-names", type=Path, default=base.DEFAULT_GENES)
    parser.add_argument("--gene-metadata", type=Path, default=base.DEFAULT_GENE_META)
    parser.add_argument("--enrichment-summary", type=Path, default=module_gate.DEFAULT_ENRICHMENT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-symbols", type=int, default=50)
    parser.add_argument("--bootstrap-repeats", type=int, default=300)
    parser.add_argument("--max-splits", type=int, default=128)
    parser.add_argument("--min-heldout-embryos", type=int, default=4)
    parser.add_argument("--positive-fraction", type=float, default=0.75)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = base.load_csc(args.counts_npz)
    manifest = base.index_manifest(args.cell_index, args.matched_manifest, counts.shape[1])
    keep_cols = np.arange(len(manifest), dtype=int)
    matrix = base.lognorm_counts(counts, keep_cols)
    gene_ids = [line.strip() for line in args.gene_names.read_text(encoding="utf-8").splitlines() if line.strip()]
    symbol_map = base.read_gene_metadata(args.gene_metadata)
    gene_symbols = [symbol_map.get(gid, gid) for gid in gene_ids]
    sym_to_idx = module_gate.symbol_index(gene_symbols)

    enrichment = [
        row
        for row in module_gate.read_csv(args.enrichment_summary)
        if row.get("row_id") in module_gate.PRIMARY_ROW_IDS
        and row.get("direction") in {"up", "down"}
        and int(float(row.get("significant_term_count") or 0)) > 0
    ]

    query_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    placebo_rows_all: list[dict[str, Any]] = []

    for query in enrichment:
        row_id = query["row_id"]
        direction = query["direction"]
        sign = 1.0 if direction == "up" else -1.0
        query_name = query["query_name"]
        symbols = module_gate.parse_symbols(query.get("top_symbols", ""))[: args.max_symbols]
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
            query_rows.append(
                {
                    "query_name": query_name,
                    "row_id": row_id,
                    "direction": direction,
                    "status": "no_mapped_genes",
                    "query_gate": False,
                }
            )
            continue

        scores = np.asarray(matrix[np.asarray(gene_idx, dtype=int), :].mean(axis=0)).ravel().astype(float)
        residual = module_gate.residualize(scores, manifest)
        roles = module_gate.row_indices(manifest, row_id)
        perturb_by_embryo = embryo_index_map(manifest, roles["perturb"])
        control_values = embryo_mean_values(manifest, residual, roles["control"])
        embryos = sorted(perturb_by_embryo)
        heldout_size = max(int(args.min_heldout_embryos), len(embryos) // 2)
        heldout_size = min(heldout_size, len(embryos))
        splits = split_iter(
            embryos,
            heldout_size,
            int(args.max_splits),
            module_gate.stable_seed(f"{query_name}|heldout_splits"),
        )

        periderm_placebo = module_gate.placebo_rows(
            manifest,
            residual,
            row_id,
            sign,
            args.bootstrap_repeats,
            "periderm_wrong_target_or_time",
            lambda row: row.get("cell_type_broad") == "periderm",
        )
        wrong_time = module_gate.placebo_rows(
            manifest,
            residual,
            row_id,
            sign,
            args.bootstrap_repeats,
            "periderm_wrong_time",
            lambda row: row.get("cell_type_broad") == "periderm"
            and str(row.get("manifest_timepoint") or row.get("timepoint")) != str(query.get("timepoint")),
        )
        wrong_lineage = module_gate.placebo_rows(
            manifest,
            residual,
            row_id,
            sign,
            args.bootstrap_repeats,
            "wrong_lineage",
            lambda row: row.get("cell_type_broad") != "periderm",
        )
        placebo_rows_all.extend(
            [{**row, "query_name": query_name} for row in periderm_placebo + wrong_time + wrong_lineage]
        )
        periderm_p95 = quantile([row["directed_diff"] for row in periderm_placebo], 0.95)
        wrong_time_max = quantile([row["directed_diff"] for row in wrong_time], 1.0)
        wrong_lineage_p95 = quantile([row["directed_diff"] for row in wrong_lineage], 0.95)
        specificity_threshold = max(
            0.0,
            periderm_p95 if math.isfinite(periderm_p95) else 0.0,
            wrong_time_max if math.isfinite(wrong_time_max) else 0.0,
            wrong_lineage_p95 if math.isfinite(wrong_lineage_p95) else 0.0,
        )

        directed_diffs: list[float] = []
        ci_lows: list[float] = []
        split_positive: list[bool] = []
        split_specific: list[bool] = []
        for split_id, heldout in enumerate(splits):
            heldout_indices: list[int] = []
            for embryo in heldout:
                heldout_indices.extend(perturb_by_embryo[embryo])
            perturb_values = embryo_mean_values(manifest, residual, heldout_indices)
            stats = module_gate.directed_diff(
                perturb_values,
                control_values,
                sign,
                module_gate.stable_seed(f"{query_name}|heldout|{split_id}"),
                args.bootstrap_repeats,
            )
            directed_diffs.append(float(stats["directed_diff"]))
            ci_lows.append(float(stats["ci_low"]))
            effect_positive = bool(stats["directed_diff"] > 0.0 and stats["ci_low"] > 0.0)
            specificity_positive = bool(stats["ci_low"] > specificity_threshold)
            split_positive.append(effect_positive)
            split_specific.append(specificity_positive)
            split_rows.append(
                {
                    "query_name": query_name,
                    "row_id": row_id,
                    "direction": direction,
                    "split_id": split_id,
                    "heldout_embryos": ";".join(heldout),
                    "n_heldout_embryos": int(len(heldout)),
                    "n_control_embryos": int(control_values.size),
                    "heldout_directed_diff": stats["directed_diff"],
                    "heldout_ci_low": stats["ci_low"],
                    "heldout_ci_high": stats["ci_high"],
                    "heldout_p": stats["p"],
                    "specificity_threshold": specificity_threshold,
                    "effect_positive": effect_positive,
                    "specificity_positive": specificity_positive,
                }
            )

        effect_positive_fraction = float(np.mean(split_positive)) if split_positive else float("nan")
        specificity_positive_fraction = float(np.mean(split_specific)) if split_specific else float("nan")
        heldout_ci_low_q05 = quantile(ci_lows, 0.05)
        heldout_diff_median = quantile(directed_diffs, 0.50)
        effect_gate = bool(
            effect_positive_fraction >= float(args.positive_fraction)
            and math.isfinite(heldout_ci_low_q05)
            and heldout_ci_low_q05 > 0.0
        )
        specificity_gate = bool(
            specificity_positive_fraction >= float(args.positive_fraction)
            and math.isfinite(heldout_ci_low_q05)
            and heldout_ci_low_q05 > specificity_threshold
        )
        query_gate = bool(effect_gate and specificity_gate)
        query_rows.append(
            {
                "query_name": query_name,
                "row_id": row_id,
                "direction": direction,
                "status": "ok",
                "n_symbols": int(len(symbols)),
                "n_mapped_genes": int(len(gene_idx)),
                "n_missing_symbols": int(len(missing)),
                "n_perturb_embryos": int(len(embryos)),
                "heldout_embryos_per_split": int(heldout_size),
                "n_splits": int(len(splits)),
                "heldout_diff_median": heldout_diff_median,
                "heldout_ci_low_q05": heldout_ci_low_q05,
                "heldout_effect_positive_fraction": effect_positive_fraction,
                "heldout_specificity_positive_fraction": specificity_positive_fraction,
                "periderm_placebo_p95": periderm_p95,
                "wrong_time_max": wrong_time_max,
                "wrong_lineage_p95": wrong_lineage_p95,
                "specificity_threshold": specificity_threshold,
                "effect_gate": effect_gate,
                "specificity_gate": specificity_gate,
                "query_gate": query_gate,
                "top_terms": query.get("top_terms", ""),
                "missing_symbols": ";".join(missing[:20]),
            }
        )

    row_summary: list[dict[str, Any]] = []
    for row_id in sorted(module_gate.PRIMARY_ROW_IDS):
        qs = [row for row in query_rows if row.get("row_id") == row_id]
        row_summary.append(
            {
                "row_id": row_id,
                "queries": len(qs),
                "query_gates": sum(bool(row.get("query_gate")) for row in qs),
                "effect_gates": sum(bool(row.get("effect_gate")) for row in qs),
                "specificity_gates": sum(bool(row.get("specificity_gate")) for row in qs),
                "all_query_gates": bool(qs and all(bool(row.get("query_gate")) for row in qs)),
                "min_heldout_ci_low_q05": quantile(
                    [row.get("heldout_ci_low_q05", float("nan")) for row in qs], 0.0
                ),
            }
        )

    status = (
        "zscape_embryo_heldout_dynamic_specificity_gate_pass_no_gpu"
        if len(query_rows) == 4 and all(bool(row.get("query_gate")) for row in query_rows)
        else "zscape_embryo_heldout_dynamic_specificity_gate_fail_no_gpu"
    )

    query_csv = args.out_dir / "zscape_embryo_heldout_dynamic_specificity_query_rows.csv"
    split_csv = args.out_dir / "zscape_embryo_heldout_dynamic_specificity_split_rows.csv"
    placebo_csv = args.out_dir / "zscape_embryo_heldout_dynamic_specificity_placebo_rows.csv"
    summary_csv = args.out_dir / "zscape_embryo_heldout_dynamic_specificity_row_summary.csv"
    write_csv(query_csv, query_rows)
    write_csv(split_csv, split_rows)
    write_csv(placebo_csv, placebo_rows_all)
    write_csv(summary_csv, row_summary)

    payload = {
        "status": status,
        "gpu_authorized": False,
        "timestamp": now_cst(),
        "boundary": "CPU-only fixed-module embryo-heldout replication/specificity gate",
        "inputs": {
            "counts_npz": str(args.counts_npz),
            "matched_manifest": str(args.matched_manifest),
            "enrichment_summary": str(args.enrichment_summary),
        },
        "query_rows": query_rows,
        "row_summary": row_summary,
        "outputs": {
            "query_csv": str(query_csv),
            "split_csv": str(split_csv),
            "placebo_csv": str(placebo_csv),
            "summary_csv": str(summary_csv),
        },
    }
    json_path = args.out_dir / "zscape_embryo_heldout_dynamic_specificity_gate_20260628.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    report = [
        "# LatentFM ZSCAPE Embryo-Heldout Dynamic Specificity Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only expression-space gate over frozen ZSCAPE selected counts.",
        "- Uses fixed module gene sets from the existing enrichment report; modules are not rediscovered inside heldout splits.",
        "- Heldout unit is perturb embryo split-half; controls remain pooled snapshot controls because ZSCAPE does not provide true same-embryo lineage controls.",
        "- No model training, inference, GPU, canonical multi, Track C query, or checkpoint selection.",
        "",
        "## Query Summary",
        "",
        "| query | splits | median diff | q05 CI low | effect frac | specificity frac | threshold | effect | specificity | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in query_rows:
        report.append(
            "| {query} | {splits} | {diff} | {ci} | {efrac} | {sfrac} | {thr} | {egate} | {sgate} | {gate} |".format(
                query=row.get("query_name"),
                splits=row.get("n_splits", 0),
                diff=fmt(row.get("heldout_diff_median")),
                ci=fmt(row.get("heldout_ci_low_q05")),
                efrac=fmt(row.get("heldout_effect_positive_fraction")),
                sfrac=fmt(row.get("heldout_specificity_positive_fraction")),
                thr=fmt(row.get("specificity_threshold")),
                egate=row.get("effect_gate", False),
                sgate=row.get("specificity_gate", False),
                gate=row.get("query_gate", False),
            )
        )
    report.extend(
        [
            "",
            "## Row Summary",
            "",
            "| row | query gates | effect gates | specificity gates | min heldout q05 CI | all pass |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in row_summary:
        report.append(
            f"| {row['row_id']} | {row['query_gates']}/{row['queries']} | "
            f"{row['effect_gates']}/{row['queries']} | {row['specificity_gates']}/{row['queries']} | "
            f"{fmt(row['min_heldout_ci_low_q05'])} | {row['all_query_gates']} |"
        )
    report.extend(
        [
            "",
            "## Decision",
            "",
            "A pass would support using the periderm modules as candidates for a separate model-constraint design gate. A fail means the modules remain biological hypotheses and negative-control material only.",
            "",
        ]
    )
    if status.endswith("_pass_no_gpu"):
        report.append("Decision: heldout embryo replication and specificity passed; proceed only to a separate constraint-design audit before training.")
    else:
        report.append("Decision: heldout effect and/or specificity failed; do not convert current ZSCAPE modules into LatentFM/RawFM losses.")
    report.extend(
        [
            "",
            "## Outputs",
            "",
            f"- query rows: `{query_csv}`",
            f"- split rows: `{split_csv}`",
            f"- placebo rows: `{placebo_csv}`",
            f"- row summary: `{summary_csv}`",
            f"- JSON: `{json_path}`",
            "",
        ]
    )
    md_path = args.out_dir / "LATENTFM_ZSCAPE_EMBRYO_HELDOUT_DYNAMIC_SPECIFICITY_GATE_20260628.md"
    md_path.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": status, "query_csv": str(query_csv), "report": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
