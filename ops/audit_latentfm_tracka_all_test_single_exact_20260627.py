#!/usr/bin/env python3
"""Exact provenance audit for Track A all_test_single.

The canonical evaluator exposes `test_single`. This script verifies whether
that group is exactly the canonical split's `test_single` condition set and
therefore can be cited as `all_test_single` with explicit provenance. It reads
only frozen posthoc JSONs.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
METADATA = ROOT / "dataset/latentfm_full/xverse/condition_metadata.json"
SEED_EVALS = {
    "seed42": ROOT / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/posthoc_eval_uncapped_20260621/split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json",
    "seed43": ROOT / "CoupledFM/output/latentfm_runs/xverse_8k_seed_replicate_20260621/xverse_comp006_endpoint5_8k_seed43_fulleval/posthoc_eval_uncapped_20260621/split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json",
}
OUT_DIR = ROOT / "reports/tracka_all_test_single_exact_20260627"
OUT_JSON = ROOT / "reports/latentfm_tracka_all_test_single_exact_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_ALL_TEST_SINGLE_EXACT_20260627.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_condition(metadata: dict[str, Any], dataset: str, condition: str) -> dict[str, Any]:
    meta = ((metadata.get(dataset) or {}).get(condition) or {})
    ptype = str(meta.get("perturbation_type_raw", "")).strip()
    genes = meta.get("genes") if isinstance(meta.get("genes"), list) else []
    lower = ptype.lower()
    if lower.startswith("crispr") or lower in {"cas13", "gene"} or genes:
        modality = "gene"
    elif "drug" in lower or "trt_cp" in lower or "compound" in lower or "chemical" in lower:
        modality = "chemical"
    else:
        modality = "unknown"
    return {
        "perturbation_type_raw": ptype,
        "n_genes": len(genes),
        "genes": ";".join(str(g) for g in genes),
        "modality_guess": modality,
    }


def split_test_single_set(split: dict[str, Any]) -> set[tuple[str, str]]:
    rows = set()
    for dataset, groups in split.items():
        for condition in groups.get("test_single", []) or []:
            rows.add((str(dataset), str(condition)))
    return rows


def eval_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list(((payload.get("groups") or {}).get("test_single") or {}).get("condition_metrics") or [])


def fmean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return mean(vals) if vals else None


def summarize(seed: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_ds[str(row["dataset"])].append(row)
    modality_counts = Counter(str(r["modality_guess"]) for r in rows)
    ptype_counts = Counter(str(r["perturbation_type_raw"]) for r in rows)
    dataset_rows = []
    for dataset, ds_rows in sorted(by_ds.items()):
        dataset_rows.append(
            {
                "seed": seed,
                "dataset": dataset,
                "n": len(ds_rows),
                "pearson_pert_mean": fmean(ds_rows, "pearson_pert"),
                "test_mmd_clamped_mean": fmean(ds_rows, "test_mmd_clamped"),
                "modality_counts": ";".join(f"{k}:{v}" for k, v in Counter(str(r["modality_guess"]) for r in ds_rows).most_common()),
            }
        )
    return {
        "seed": seed,
        "n_conditions": len(rows),
        "n_datasets": len(by_ds),
        "pearson_pert_mean": fmean(rows, "pearson_pert"),
        "pearson_ctrl_mean": fmean(rows, "pearson_ctrl"),
        "direct_pearson_mean": fmean(rows, "direct_pearson"),
        "test_mmd_clamped_mean": fmean(rows, "test_mmd_clamped"),
        "dataset_min_pearson_pert_mean": min((r["pearson_pert_mean"] for r in dataset_rows if r["pearson_pert_mean"] is not None), default=None),
        "dataset_max_mmd_mean": max((r["test_mmd_clamped_mean"] for r in dataset_rows if r["test_mmd_clamped_mean"] is not None), default=None),
        "modality_counts": modality_counts.most_common(),
        "perturbation_type_counts": ptype_counts.most_common(),
        "dataset_breakdown": dataset_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def main() -> int:
    split = load_json(SPLIT)
    metadata = load_json(METADATA)
    split_rows = split_test_single_set(split)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    for seed, path in SEED_EVALS.items():
        payload = load_json(path)
        rows = []
        eval_set = set()
        for row in eval_rows(payload):
            key = (str(row.get("dataset")), str(row.get("condition")))
            eval_set.add(key)
            cls = classify_condition(metadata, key[0], key[1])
            out = {
                "seed": seed,
                "dataset": key[0],
                "condition": key[1],
                **cls,
                "pearson_pert": float(row["pearson_pert"]),
                "pearson_ctrl": float(row["pearson_ctrl"]),
                "direct_pearson": float(row["direct_pearson"]),
                "test_mmd_clamped": float(row["test_mmd_clamped"]),
                "n_src_eval": row.get("n_src_eval", ""),
                "n_gt_eval": row.get("n_gt_eval", ""),
            }
            rows.append(out)
            all_rows.append(out)
        missing = sorted(split_rows - eval_set)
        extra = sorted(eval_set - split_rows)
        provenance[seed] = {
            "split_test_single_rows": len(split_rows),
            "eval_test_single_rows": len(eval_set),
            "missing_from_eval": len(missing),
            "extra_in_eval": len(extra),
            "exact_set_match": len(missing) == 0 and len(extra) == 0,
            "missing_examples": missing[:10],
            "extra_examples": extra[:10],
        }
        summaries.append(summarize(seed, rows))

    exact = all(v["exact_set_match"] for v in provenance.values())
    status = "tracka_all_test_single_exact_ready_no_gpu" if exact else "tracka_all_test_single_exact_mismatch_no_gpu"
    out = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "training": False,
            "inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection_used": False,
            "trackc_query_used": False,
            "definition": "all_test_single is exactly canonical split_seed42.json test_single when frozen posthoc test_single rows match the split set",
        },
        "input_manifest": [
            {"key": "split", "path": str(SPLIT), "sha256": sha256(SPLIT)},
            {"key": "metadata", "path": str(METADATA), "sha256": sha256(METADATA)},
            *[
                {"key": key, "path": str(path), "sha256": sha256(path)}
                for key, path in SEED_EVALS.items()
            ],
        ],
        "provenance": provenance,
        "summaries": summaries,
        "outputs": {
            "rows_csv": str(OUT_DIR / "all_test_single_rows.csv"),
            "summary_csv": str(OUT_DIR / "all_test_single_summary.csv"),
            "dataset_breakdown_csv": str(OUT_DIR / "all_test_single_dataset_breakdown.csv"),
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "all_test_single_rows.csv", all_rows)
    write_csv(OUT_DIR / "all_test_single_summary.csv", [{k: v for k, v in s.items() if k != "dataset_breakdown"} for s in summaries])
    write_csv(OUT_DIR / "all_test_single_dataset_breakdown.csv", [r for s in summaries for r in s["dataset_breakdown"]])
    OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Track A All Test Single Exact Provenance",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only over canonical split and frozen xverse 8k posthoc JSONs.",
        "- No training, inference, checkpoint selection, canonical multi selection, Track C query, or GPU.",
        "",
        "## Set Match",
        "",
        "| seed | split rows | eval rows | missing | extra | exact match |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for seed, rec in provenance.items():
        lines.append(
            f"| `{seed}` | {rec['split_test_single_rows']} | {rec['eval_test_single_rows']} | "
            f"{rec['missing_from_eval']} | {rec['extra_in_eval']} | {rec['exact_set_match']} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| seed | n | datasets | pp mean | ctrl mean | direct mean | MMD mean | modality counts |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for summary in summaries:
        lines.append(
            f"| `{summary['seed']}` | {summary['n_conditions']} | {summary['n_datasets']} | "
            f"{fmt(summary['pearson_pert_mean'])} | {fmt(summary['pearson_ctrl_mean'])} | "
            f"{fmt(summary['direct_pearson_mean'])} | {fmt(summary['test_mmd_clamped_mean'])} | "
            f"`{summary['modality_counts']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Canonical `test_single` is an exact all-test-single set if both seeds show exact set match above. This is provenance/benchmark evidence only and does not authorize GPU.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- rows: `{OUT_DIR / 'all_test_single_rows.csv'}`",
            f"- dataset breakdown: `{OUT_DIR / 'all_test_single_dataset_breakdown.csv'}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorized": False, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
