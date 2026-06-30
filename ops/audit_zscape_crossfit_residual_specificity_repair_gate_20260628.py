#!/usr/bin/env python3
"""Crossfit residual specificity-repair gate for selected ZSCAPE rows.

CPU-only. This differs from the fixed-module specificity gate by discovering
module genes inside train perturb embryos for each heldout split, after
subtracting wrong-control gene effects, then evaluating on heldout embryos.
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
DEFAULT_OUT = ROOT / "reports/zscape_crossfit_residual_specificity_repair_gate_20260628"
PRIMARY_ROW_IDS = ("periderm__noto__24p0h", "periderm__smo__24p0h")
DIRECTIONS = ("down", "up")


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


def ensure_output_dir(path: Path, force: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = [p for p in path.iterdir() if p.name != ".DS_Store"]
    if existing and not force:
        raise SystemExit(f"Refusing to overwrite nonempty output directory: {path}")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def quantile(values: list[float] | np.ndarray, q: float) -> float:
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, q))


def embryo_index_map(manifest: list[dict[str, str]], indices: list[int]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for idx in indices:
        row = manifest[idx]
        embryo = row.get("embryo") or row.get("sample") or f"cell_{idx}"
        out[str(embryo)].append(idx)
    return dict(out)


def split_iter(embryos: list[str], heldout_size: int, max_splits: int, seed: int) -> list[tuple[str, ...]]:
    combos = list(itertools.combinations(sorted(embryos), heldout_size))
    if len(combos) <= max_splits:
        return combos
    rng = np.random.default_rng(seed)
    take = rng.choice(len(combos), size=max_splits, replace=False)
    return [combos[int(i)] for i in sorted(take)]


def role_indices(manifest: list[dict[str, str]], row_id: str) -> dict[str, list[int]]:
    out = {"perturb": [], "control": []}
    for idx, row in enumerate(manifest):
        if row.get("row_id") != row_id:
            continue
        role = row.get("selection_role", "")
        if role in out:
            out[role].append(idx)
    return out


def row_meta(manifest: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in manifest:
        rid = row.get("row_id", "")
        if rid and rid not in out:
            out[rid] = row
    return out


def mean_gene(matrix, cols: list[int]) -> np.ndarray:
    if not cols:
        return np.zeros(matrix.shape[0], dtype=float)
    return np.asarray(matrix[:, cols].mean(axis=1)).ravel().astype(float)


def module_values(matrix, gene_idx: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[gene_idx, :].mean(axis=0)).ravel().astype(float)


def directed_gene_effect(matrix, perturb_cols: list[int], control_cols: list[int], sign: float) -> np.ndarray:
    return sign * (mean_gene(matrix, perturb_cols) - mean_gene(matrix, control_cols))


def stable_control_split(control_embryos: list[str], split_key: str) -> tuple[set[str], set[str]]:
    embryos = sorted(control_embryos)
    if len(embryos) <= 1:
        return set(), set(embryos)
    rng = np.random.default_rng(module_gate.stable_seed(split_key))
    perm = list(rng.permutation(embryos))
    n_hold = max(1, len(perm) // 2)
    heldout = set(map(str, perm[:n_hold]))
    train = set(map(str, perm[n_hold:]))
    return train, heldout


def indices_for_embryos(mapping: dict[str, list[int]], embryos: set[str] | tuple[str, ...]) -> list[int]:
    out: list[int] = []
    for embryo in embryos:
        out.extend(mapping.get(str(embryo), []))
    return out


def wrong_row_ids(
    meta: dict[str, dict[str, str]],
    query_row_id: str,
    *,
    kind: str,
    query_lineage: str,
    query_time: str,
) -> list[str]:
    rows = []
    for rid, row in meta.items():
        if rid == query_row_id:
            continue
        lineage = row.get("manifest_cell_type_broad") or row.get("cell_type_broad") or ""
        timepoint = str(row.get("manifest_timepoint") or row.get("timepoint") or "")
        if kind == "same_lineage" and lineage == query_lineage:
            rows.append(rid)
        elif kind == "wrong_time" and lineage == query_lineage and timepoint != str(query_time):
            rows.append(rid)
        elif kind == "wrong_lineage" and lineage != query_lineage:
            rows.append(rid)
    return sorted(rows)


def wrong_gene_penalty(
    matrix,
    manifest: list[dict[str, str]],
    row_ids: list[str],
    sign: float,
) -> np.ndarray:
    penalty = np.zeros(matrix.shape[0], dtype=float)
    for rid in row_ids:
        roles = role_indices(manifest, rid)
        if not roles["perturb"] or not roles["control"]:
            continue
        effect = directed_gene_effect(matrix, roles["perturb"], roles["control"], sign)
        penalty = np.maximum(penalty, np.maximum(effect, 0.0))
    return penalty


def quantile_bins(score: np.ndarray, bins: int) -> np.ndarray:
    q = np.linspace(0.0, 1.0, max(2, int(bins)) + 1)
    edges = np.quantile(np.asarray(score, dtype=float), q)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.digitize(score, edges[1:-1], right=True).astype(int)


def matched_random_keep(
    selected: np.ndarray,
    scores: list[np.ndarray],
    rng: np.random.Generator,
    bins: int,
) -> np.ndarray:
    n = int(scores[0].shape[0])
    bin_arrays = [quantile_bins(score, bins) for score in scores]
    selected_set = set(map(int, selected))
    selected_keys = [tuple(int(arr[i]) for arr in bin_arrays) for i in selected]
    all_keys: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i in range(n):
        all_keys[tuple(int(arr[i]) for arr in bin_arrays)].append(i)
    chosen: list[int] = []
    used: set[int] = set()
    for key in sorted(set(selected_keys)):
        need = int(sum(k == key for k in selected_keys))
        pool = [i for i in all_keys.get(key, []) if i not in selected_set and i not in used]
        if len(pool) < need:
            pool = [i for i in all_keys.get(key, []) if i not in used]
        if not pool:
            continue
        take = rng.choice(np.asarray(pool, dtype=int), size=min(need, len(pool)), replace=False)
        chosen.extend(map(int, take))
        used.update(map(int, take))
    if len(chosen) < len(selected):
        remaining = np.asarray([i for i in range(n) if i not in used and i not in selected_set], dtype=int)
        if remaining.size < len(selected) - len(chosen):
            remaining = np.asarray([i for i in range(n) if i not in used], dtype=int)
        extra = rng.choice(remaining, size=len(selected) - len(chosen), replace=False)
        chosen.extend(map(int, extra))
    return np.asarray(chosen[: len(selected)], dtype=int)


def directed_split_stats(
    manifest: list[dict[str, str]],
    values: np.ndarray,
    perturb_indices: list[int],
    control_indices: list[int],
    sign: float,
    seed: int,
    repeats: int,
) -> dict[str, float]:
    p, _ = module_gate.embryo_values(manifest, values, perturb_indices)
    c, _ = module_gate.embryo_values(manifest, values, control_indices)
    return module_gate.directed_diff(p, c, sign, seed, repeats)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts-npz", type=Path, default=base.DEFAULT_COUNTS)
    parser.add_argument("--cell-index", type=Path, default=base.DEFAULT_CELL_INDEX)
    parser.add_argument("--matched-manifest", type=Path, default=base.DEFAULT_MANIFEST)
    parser.add_argument("--gene-names", type=Path, default=base.DEFAULT_GENES)
    parser.add_argument("--gene-metadata", type=Path, default=base.DEFAULT_GENE_META)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--module-size", type=int, default=24)
    parser.add_argument("--bootstrap-repeats", type=int, default=120)
    parser.add_argument("--random-repeats", type=int, default=40)
    parser.add_argument("--max-splits", type=int, default=70)
    parser.add_argument("--heldout-embryos", type=int, default=4)
    parser.add_argument("--positive-fraction", type=float, default=0.75)
    parser.add_argument("--specificity-margin", type=float, default=0.02)
    parser.add_argument(
        "--row-pass-fraction",
        type=float,
        default=0.75,
        help="Fraction of primary rows that must have at least one passing direction for biological_pass.",
    )
    parser.add_argument(
        "--min-pass-rows",
        type=int,
        default=None,
        help="Override row-level biological_pass threshold. Defaults to ceil(row-pass-fraction * n_primary_rows).",
    )
    parser.add_argument("--force", action="store_true", help="Allow overwriting a nonempty output directory.")
    parser.add_argument(
        "--primary-row-ids",
        default=",".join(PRIMARY_ROW_IDS),
        help="Comma-separated ZSCAPE row_ids to evaluate. Defaults to original periderm rows.",
    )
    args = parser.parse_args()

    ensure_output_dir(args.out_dir, args.force)
    counts = base.load_csc(args.counts_npz)
    manifest = base.index_manifest(args.cell_index, args.matched_manifest, counts.shape[1])
    matrix = base.lognorm_counts(counts, np.arange(len(manifest), dtype=int))
    gene_ids = [line.strip() for line in args.gene_names.read_text(encoding="utf-8").splitlines() if line.strip()]
    symbol_map = base.read_gene_metadata(args.gene_metadata)
    gene_symbols = [symbol_map.get(gid, gid) for gid in gene_ids]
    meta = row_meta(manifest)

    discovered_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    wrong_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []

    primary_row_ids = tuple(x.strip() for x in str(args.primary_row_ids).split(",") if x.strip())
    min_pass_rows = (
        int(args.min_pass_rows)
        if args.min_pass_rows is not None
        else max(1, int(math.ceil(float(args.row_pass_fraction) * len(primary_row_ids))))
    )

    for row_id in primary_row_ids:
        roles = role_indices(manifest, row_id)
        if not roles["perturb"] or not roles["control"] or row_id not in meta:
            query_rows.append(
                {
                    "query_name": f"{row_id}.missing",
                    "row_id": row_id,
                    "direction": "missing",
                    "n_splits": 0,
                    "n_mapped_genes_min": 0,
                    "heldout_diff_median": float("nan"),
                    "heldout_ci_low_q05": float("nan"),
                    "effect_positive_fraction": float("nan"),
                    "specificity_positive_fraction": float("nan"),
                    "specificity_margin_q05": float("nan"),
                    "random_margin_q05": float("nan"),
                    "signflip_margin_median": float("nan"),
                    "query_gate": False,
                }
            )
            continue
        perturb_by_embryo = embryo_index_map(manifest, roles["perturb"])
        control_by_embryo = embryo_index_map(manifest, roles["control"])
        perturb_embryos = sorted(perturb_by_embryo)
        control_embryos = sorted(control_by_embryo)
        qmeta = meta[row_id]
        query_lineage = qmeta.get("manifest_cell_type_broad") or qmeta.get("cell_type_broad") or ""
        query_time = str(qmeta.get("manifest_timepoint") or qmeta.get("timepoint") or "")
        same_lineage = wrong_row_ids(
            meta, row_id, kind="same_lineage", query_lineage=query_lineage, query_time=query_time
        )
        wrong_time = wrong_row_ids(
            meta, row_id, kind="wrong_time", query_lineage=query_lineage, query_time=query_time
        )
        wrong_lineage = wrong_row_ids(
            meta, row_id, kind="wrong_lineage", query_lineage=query_lineage, query_time=query_time
        )
        splits = split_iter(
            perturb_embryos,
            min(int(args.heldout_embryos), len(perturb_embryos)),
            int(args.max_splits),
            module_gate.stable_seed(f"{row_id}|crossfit_splits"),
        )
        for direction in DIRECTIONS:
            sign = 1.0 if direction == "up" else -1.0
            query_name = f"{row_id}.{direction}.crossfit"
            heldout_diffs: list[float] = []
            heldout_ci_lows: list[float] = []
            specificity_margins: list[float] = []
            random_margins: list[float] = []
            signflip_margins: list[float] = []
            effect_positive: list[bool] = []
            specificity_positive: list[bool] = []
            mapped_counts: list[int] = []
            for split_id, heldout in enumerate(splits):
                heldout_set = set(heldout)
                train_set = set(perturb_embryos) - heldout_set
                train_control_set, heldout_control_set = stable_control_split(
                    control_embryos,
                    f"{query_name}|control|{split_id}",
                )
                train_p = indices_for_embryos(perturb_by_embryo, train_set)
                heldout_p = indices_for_embryos(perturb_by_embryo, heldout)
                train_c = indices_for_embryos(control_by_embryo, train_control_set)
                heldout_c = indices_for_embryos(control_by_embryo, heldout_control_set)
                if not train_p or not train_c or not heldout_p or not heldout_c:
                    continue

                train_effect = directed_gene_effect(matrix, train_p, train_c, sign)
                penalty = np.maximum(
                    wrong_gene_penalty(matrix, manifest, same_lineage, sign),
                    wrong_gene_penalty(matrix, manifest, wrong_lineage, sign),
                )
                residual_score = train_effect - penalty
                order = np.argsort(-residual_score, kind="mergesort")
                selected = order[: max(1, min(int(args.module_size), len(order)))]
                selected = np.asarray([int(i) for i in selected if math.isfinite(float(residual_score[int(i)]))], dtype=int)
                mapped_counts.append(int(selected.size))
                symbols = [gene_symbols[int(i)] for i in selected]
                discovered_rows.append(
                    {
                        "query_name": query_name,
                        "row_id": row_id,
                        "direction": direction,
                        "split_id": split_id,
                        "n_train_perturb_embryos": int(len(train_set)),
                        "n_heldout_perturb_embryos": int(len(heldout_set)),
                        "n_train_control_embryos": int(len(train_control_set)),
                        "n_heldout_control_embryos": int(len(heldout_control_set)),
                        "n_genes": int(selected.size),
                        "score_min": float(np.min(residual_score[selected])) if selected.size else float("nan"),
                        "score_median": float(np.median(residual_score[selected])) if selected.size else float("nan"),
                        "top_symbols": ";".join(symbols[:24]),
                    }
                )
                values = module_gate.residualize(module_values(matrix, selected), manifest)
                stats = directed_split_stats(
                    manifest,
                    values,
                    heldout_p,
                    heldout_c,
                    sign,
                    module_gate.stable_seed(f"{query_name}|heldout|{split_id}"),
                    int(args.bootstrap_repeats),
                )

                def placebo(kind: str, row_ids: list[str]) -> list[float]:
                    vals = []
                    for wrong_id in row_ids:
                        wroles = role_indices(manifest, wrong_id)
                        wstats = directed_split_stats(
                            manifest,
                            values,
                            wroles["perturb"],
                            wroles["control"],
                            sign,
                            module_gate.stable_seed(f"{query_name}|{kind}|{wrong_id}|{split_id}"),
                            int(args.bootstrap_repeats),
                        )
                        vals.append(float(wstats["directed_diff"]))
                        wrong_rows.append(
                            {
                                "query_name": query_name,
                                "split_id": split_id,
                                "control_type": kind,
                                "control_row_id": wrong_id,
                                "directed_diff": wstats["directed_diff"],
                                "ci_low": wstats["ci_low"],
                                "ci_high": wstats["ci_high"],
                            }
                        )
                    return vals

                same_p95 = quantile(placebo("same_lineage_wrong_target_or_time", same_lineage), 0.95)
                wrong_time_max = quantile(placebo("same_lineage_wrong_time", wrong_time), 1.0)
                wrong_lineage_p95 = quantile(placebo("wrong_lineage", wrong_lineage), 0.95)

                abundance, variance, detection = base.gene_moments(matrix, train_c)
                rng = np.random.default_rng(module_gate.stable_seed(f"{query_name}|matched_random|{split_id}"))
                random_effects = []
                for ridx in range(int(args.random_repeats)):
                    rkeep = matched_random_keep(selected, [abundance, variance, detection], rng, bins=8)
                    rvalues = module_gate.residualize(module_values(matrix, rkeep), manifest)
                    rstats = directed_split_stats(
                        manifest,
                        rvalues,
                        heldout_p,
                        heldout_c,
                        sign,
                        module_gate.stable_seed(f"{query_name}|random|{split_id}|{ridx}"),
                        max(30, int(args.bootstrap_repeats) // 3),
                    )
                    random_effects.append(float(rstats["directed_diff"]))
                random_p95 = quantile(random_effects, 0.95)
                random_rows.append(
                    {
                        "query_name": query_name,
                        "split_id": split_id,
                        "n_random_sets": int(args.random_repeats),
                        "matched_random_p95": random_p95,
                    }
                )
                threshold = max(
                    0.0,
                    same_p95 if math.isfinite(same_p95) else 0.0,
                    wrong_time_max if math.isfinite(wrong_time_max) else 0.0,
                    wrong_lineage_p95 if math.isfinite(wrong_lineage_p95) else 0.0,
                    random_p95 if math.isfinite(random_p95) else 0.0,
                )
                margin = float(stats["ci_low"]) - threshold
                rand_margin = float(stats["ci_low"]) - random_p95
                signflip_margin = -float(stats["ci_low"]) - threshold
                heldout_diffs.append(float(stats["directed_diff"]))
                heldout_ci_lows.append(float(stats["ci_low"]))
                specificity_margins.append(margin)
                random_margins.append(rand_margin)
                signflip_margins.append(signflip_margin)
                effect_ok = bool(stats["directed_diff"] > 0.0 and stats["ci_low"] > 0.0)
                specificity_ok = bool(margin > float(args.specificity_margin))
                effect_positive.append(effect_ok)
                specificity_positive.append(specificity_ok)
                split_rows.append(
                    {
                        "query_name": query_name,
                        "row_id": row_id,
                        "direction": direction,
                        "split_id": split_id,
                        "heldout_embryos": ";".join(heldout),
                        "n_genes": int(selected.size),
                        "heldout_directed_diff": stats["directed_diff"],
                        "heldout_ci_low": stats["ci_low"],
                        "heldout_ci_high": stats["ci_high"],
                        "same_lineage_wrong_p95": same_p95,
                        "wrong_time_max": wrong_time_max,
                        "wrong_lineage_p95": wrong_lineage_p95,
                        "matched_random_p95": random_p95,
                        "specificity_threshold": threshold,
                        "specificity_margin": margin,
                        "random_margin": rand_margin,
                        "signflip_margin": signflip_margin,
                        "effect_positive": effect_ok,
                        "specificity_positive": specificity_ok,
                    }
                )

            n_mapped_min = min(mapped_counts) if mapped_counts else 0
            effect_frac = float(np.mean(effect_positive)) if effect_positive else float("nan")
            specificity_frac = float(np.mean(specificity_positive)) if specificity_positive else float("nan")
            margin_q05 = quantile(specificity_margins, 0.05)
            random_margin_q05 = quantile(random_margins, 0.05)
            signflip_median = quantile(signflip_margins, 0.50)
            heldout_ci_q05 = quantile(heldout_ci_lows, 0.05)
            query_gate = bool(
                n_mapped_min >= 8
                and effect_frac >= float(args.positive_fraction)
                and math.isfinite(heldout_ci_q05)
                and heldout_ci_q05 > 0.0
                and specificity_frac >= float(args.positive_fraction)
                and math.isfinite(margin_q05)
                and margin_q05 > float(args.specificity_margin)
                and math.isfinite(signflip_median)
                and signflip_median <= 0.0
                and math.isfinite(random_margin_q05)
                and random_margin_q05 >= 0.01
            )
            query_rows.append(
                {
                    "query_name": query_name,
                    "row_id": row_id,
                    "direction": direction,
                    "n_splits": int(len(splits)),
                    "n_mapped_genes_min": int(n_mapped_min),
                    "heldout_diff_median": quantile(heldout_diffs, 0.50),
                    "heldout_ci_low_q05": heldout_ci_q05,
                    "effect_positive_fraction": effect_frac,
                    "specificity_positive_fraction": specificity_frac,
                    "specificity_margin_q05": margin_q05,
                    "random_margin_q05": random_margin_q05,
                    "signflip_margin_median": signflip_median,
                    "query_gate": query_gate,
                }
            )

    row_summary = []
    for row_id in primary_row_ids:
        qs = [row for row in query_rows if row["row_id"] == row_id]
        row_summary.append(
            {
                "row_id": row_id,
                "queries": len(qs),
                "query_gates": sum(bool(row["query_gate"]) for row in qs),
                "any_query_gate": any(bool(row["query_gate"]) for row in qs),
            }
        )
    total_pass = sum(bool(row["query_gate"]) for row in query_rows)
    row_pass_count = sum(bool(row["any_query_gate"]) for row in row_summary)
    biological_pass = bool(row_pass_count >= min_pass_rows)
    model_constraint_precondition = bool(total_pass == len(primary_row_ids) * len(DIRECTIONS))
    status = (
        "zscape_crossfit_residual_specificity_repair_gate_biological_pass_no_gpu"
        if biological_pass
        else "zscape_crossfit_residual_specificity_repair_gate_fail_no_gpu"
    )

    discovered_csv = args.out_dir / "zscape_crossfit_specificity_discovered_modules.csv"
    split_csv = args.out_dir / "zscape_crossfit_specificity_split_rows.csv"
    query_csv = args.out_dir / "zscape_crossfit_specificity_query_rows.csv"
    wrong_csv = args.out_dir / "zscape_crossfit_specificity_wrong_control_rows.csv"
    random_csv = args.out_dir / "zscape_crossfit_specificity_matched_random_rows.csv"
    summary_csv = args.out_dir / "zscape_crossfit_specificity_row_summary.csv"
    write_csv(discovered_csv, discovered_rows)
    write_csv(split_csv, split_rows)
    write_csv(query_csv, query_rows)
    write_csv(wrong_csv, wrong_rows)
    write_csv(random_csv, random_rows)
    write_csv(summary_csv, row_summary)
    payload = {
        "timestamp": now_cst(),
        "status": status,
        "gpu_authorized": False,
        "biological_pass": biological_pass,
        "model_constraint_precondition": model_constraint_precondition,
        "row_pass_count": int(row_pass_count),
        "min_pass_rows": int(min_pass_rows),
        "row_pass_fraction_threshold": float(args.row_pass_fraction),
        "query_gates": int(total_pass),
        "queries": int(len(query_rows)),
        "primary_row_ids": list(primary_row_ids),
        "outputs": {
            "discovered_modules": str(discovered_csv),
            "split_rows": str(split_csv),
            "query_rows": str(query_csv),
            "wrong_control_rows": str(wrong_csv),
            "matched_random_rows": str(random_csv),
            "row_summary": str(summary_csv),
        },
    }
    json_path = args.out_dir / "zscape_crossfit_specificity_gate_20260628.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM ZSCAPE Crossfit Residual Specificity Repair Gate",
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
        "- Discovers module genes inside train perturb embryos for each split; heldout perturb embryos are not used for gene selection.",
        "- Control embryos are split into train/heldout pools for discovery/evaluation.",
        "- Wrong target/time/lineage effects and abundance/variance/detection-matched random sets are negative controls.",
        "- No LatentFM/RawFM training, no GPU, no checkpoint selection, no canonical multi, no Track C query.",
        "",
        "## Query Summary",
        "",
        "| query | splits | genes min | median diff | q05 CI low | effect frac | specificity frac | margin q05 | random margin q05 | signflip median | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in query_rows:
        lines.append(
            f"| {row['query_name']} | {row['n_splits']} | {row['n_mapped_genes_min']} | "
            f"{fmt(row['heldout_diff_median'])} | {fmt(row['heldout_ci_low_q05'])} | "
            f"{fmt(row['effect_positive_fraction'])} | {fmt(row['specificity_positive_fraction'])} | "
            f"{fmt(row['specificity_margin_q05'])} | {fmt(row['random_margin_q05'])} | "
            f"{fmt(row['signflip_margin_median'])} | {row['query_gate']} |"
        )
    lines.extend(["", "## Row Summary", "", "| row | query gates | any pass |", "|---|---:|---:|"])
    for row in row_summary:
        lines.append(f"| {row['row_id']} | {row['query_gates']}/{row['queries']} | {row['any_query_gate']} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- biological repair pass: `{biological_pass}`",
            f"- model-constraint precondition: `{model_constraint_precondition}`",
            f"- row pass count / threshold: `{row_pass_count}` / `{min_pass_rows}`",
            f"- query gates / total queries: `{total_pass}` / `{len(query_rows)}`",
        ]
    )
    if model_constraint_precondition:
        lines.append("Decision: all query modules passed this CPU gate; still require a separate model no-harm/design gate before any GPU constraint.")
    elif biological_pass:
        lines.append("Decision: biological specificity repair is promising, but not sufficient for model constraints; keep as biology/scaling insight.")
    else:
        lines.append("Decision: crossfit residual rediscovery did not repair specificity enough; ZSCAPE remains report-only for current modeling.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- discovered modules: `{discovered_csv}`",
            f"- split rows: `{split_csv}`",
            f"- query rows: `{query_csv}`",
            f"- wrong controls: `{wrong_csv}`",
            f"- matched random: `{random_csv}`",
            f"- row summary: `{summary_csv}`",
            f"- JSON: `{json_path}`",
            "",
        ]
    )
    md_path = args.out_dir / "LATENTFM_ZSCAPE_CROSSFIT_RESIDUAL_SPECIFICITY_REPAIR_GATE_20260628.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "query_csv": str(query_csv), "report": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
