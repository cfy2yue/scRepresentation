#!/usr/bin/env python3
"""Build a canonical Track A cross-background seen-gene evaluation slice.

This is a CPU/report-only evaluator over existing frozen xverse_8k anchor
posthoc JSONs. It does not train, infer, read Track C query, or use canonical
multi for selection. Because the canonical metadata does not expose one
uniform cell/background field for all datasets, this report uses dataset as the
background unit and states that boundary explicitly.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
METADATA = ROOT / "dataset/latentfm_full/xverse/condition_metadata.json"
SEED_EVALS = {
    "seed42": ROOT / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/posthoc_eval_uncapped_20260621/split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json",
    "seed43": ROOT / "CoupledFM/output/latentfm_runs/xverse_8k_seed_replicate_20260621/xverse_comp006_endpoint5_8k_seed43_fulleval/posthoc_eval_uncapped_20260621/split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json",
}
OUT_DIR = ROOT / "reports/tracka_cross_background_seen_gene_exact_20260627"
OUT_JSON = ROOT / "reports/latentfm_tracka_cross_background_seen_gene_exact_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_CROSS_BACKGROUND_SEEN_GENE_EXACT_20260627.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def genes_for(metadata: dict[str, Any], ds: str, cond: str) -> tuple[str, ...]:
    entry = ((metadata.get(ds) or {}).get(cond) or {})
    genes = entry.get("genes")
    if not isinstance(genes, list):
        parts = [p.strip() for p in str(cond).split("+") if p.strip()]
        genes = parts if parts else [str(cond)]
    return tuple(str(g).strip().upper() for g in genes if str(g).strip())


def mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=float)))


def boot_ci(values: list[float], *, seed: int = 20260627, n_boot: int = 2000) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.shape[0], size=(int(n_boot), arr.shape[0]))
    stats = arr[idx].mean(axis=1)
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def classify_rows(split: dict[str, Any], metadata: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    train_any: dict[str, set[str]] = defaultdict(set)
    train_single: dict[str, set[str]] = defaultdict(set)
    train_same_dataset_any: dict[tuple[str, str], bool] = {}
    for ds, groups in split.items():
        for cond in groups.get("train", []) or []:
            genes = genes_for(metadata, ds, str(cond))
            for gene in genes:
                train_any[gene].add(str(ds))
                if len(genes) == 1:
                    train_single[gene].add(str(ds))
                train_same_dataset_any[(str(ds), gene)] = True

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for ds, groups in split.items():
        for cond in groups.get("test_single", []) or []:
            genes = genes_for(metadata, ds, str(cond))
            if len(genes) != 1:
                continue
            gene = genes[0]
            other_any = sorted(d for d in train_any.get(gene, set()) if d != str(ds))
            other_single = sorted(d for d in train_single.get(gene, set()) if d != str(ds))
            same_any = bool(train_same_dataset_any.get((str(ds), gene), False))
            out[(str(ds), str(cond))] = {
                "gene": gene,
                "other_train_any_datasets": other_any,
                "other_train_single_datasets": other_single,
                "same_dataset_train_any": same_any,
                "is_cross_background_seen_gene": bool(other_any),
                "is_cross_background_seen_gene_single_only": bool(other_single),
                "is_cross_background_seen_gene_strict_other_only": bool(other_any and not same_any),
            }
    return out


def row_metrics(eval_json: dict[str, Any]) -> list[dict[str, Any]]:
    return list(((eval_json.get("groups") or {}).get("test_single") or {}).get("condition_metrics") or [])


def summarize(seed: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    pp = [float(r["pearson_pert"]) for r in rows if r.get("pearson_pert") is not None]
    mmd = [float(r["test_mmd_clamped"]) for r in rows if r.get("test_mmd_clamped") is not None]
    pp_lo, pp_hi = boot_ci(pp)
    mmd_lo, mmd_hi = boot_ci(mmd)
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_ds[str(r["dataset"])].append(r)
    ds_rows = []
    for ds, ds_rows_raw in sorted(by_ds.items()):
        ds_pp = [float(r["pearson_pert"]) for r in ds_rows_raw if r.get("pearson_pert") is not None]
        ds_mmd = [float(r["test_mmd_clamped"]) for r in ds_rows_raw if r.get("test_mmd_clamped") is not None]
        ds_rows.append(
            {
                "seed": seed,
                "dataset": ds,
                "n": len(ds_rows_raw),
                "pearson_pert_mean": mean(ds_pp),
                "test_mmd_clamped_mean": mean(ds_mmd),
            }
        )
    return {
        "seed": seed,
        "n_conditions": len(rows),
        "n_datasets": len(by_ds),
        "pearson_pert_mean": mean(pp),
        "pearson_pert_ci95_low": pp_lo,
        "pearson_pert_ci95_high": pp_hi,
        "test_mmd_clamped_mean": mean(mmd),
        "test_mmd_clamped_ci95_low": mmd_lo,
        "test_mmd_clamped_ci95_high": mmd_hi,
        "dataset_min_pearson_pert": min(
            (float(r["pearson_pert_mean"]) for r in ds_rows if r["pearson_pert_mean"] is not None),
            default=None,
        ),
        "dataset_breakdown": ds_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(v: Any) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:+.6f}"
    return str(v)


def main() -> int:
    split = load_json(SPLIT)
    metadata = load_json(METADATA)
    classes = classify_rows(split, metadata)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for seed, eval_path in SEED_EVALS.items():
        payload = load_json(eval_path)
        if Path(str(payload.get("split_file", ""))).resolve() != SPLIT.resolve():
            raise RuntimeError(f"{seed} split mismatch: {payload.get('split_file')}")
        seed_rows = []
        for row in row_metrics(payload):
            key = (str(row.get("dataset")), str(row.get("condition")))
            cls = classes.get(key)
            if not cls or not cls["is_cross_background_seen_gene"]:
                continue
            out = {
                "seed": seed,
                "dataset": key[0],
                "condition": key[1],
                **cls,
                "other_train_any_datasets": ";".join(cls["other_train_any_datasets"]),
                "other_train_single_datasets": ";".join(cls["other_train_single_datasets"]),
                "pearson_pert": float(row["pearson_pert"]),
                "pearson_ctrl": float(row["pearson_ctrl"]),
                "direct_pearson": float(row["direct_pearson"]),
                "test_mmd_clamped": float(row["test_mmd_clamped"]),
            }
            seed_rows.append(out)
            all_rows.append(out)
        summaries.append(summarize(seed, seed_rows))

    seed42 = next(s for s in summaries if s["seed"] == "seed42")
    seed43 = next(s for s in summaries if s["seed"] == "seed43")
    replicate_delta = None
    if seed42["pearson_pert_mean"] is not None and seed43["pearson_pert_mean"] is not None:
        replicate_delta = float(seed43["pearson_pert_mean"] - seed42["pearson_pert_mean"])
    status = "tracka_cross_background_seen_gene_exact_ready_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "training": False,
            "inference": False,
            "canonical_multi_selection_used": False,
            "trackc_query_used": False,
            "background_unit": "dataset",
            "primary_definition": "canonical test_single row whose single perturbed gene appears in train conditions of at least one different dataset",
            "split_file": str(SPLIT),
        },
        "input_manifest": [
            {"key": "split", "path": str(SPLIT), "sha256": sha256(SPLIT)},
            {"key": "metadata", "path": str(METADATA), "sha256": sha256(METADATA)},
            *[
                {"key": key, "path": str(path), "sha256": sha256(path)}
                for key, path in SEED_EVALS.items()
            ],
        ],
        "summaries": summaries,
        "seed43_minus_seed42_pearson_pert": replicate_delta,
        "outputs": {
            "rows_csv": str(OUT_DIR / "cross_background_seen_gene_rows.csv"),
            "dataset_breakdown_csv": str(OUT_DIR / "cross_background_seen_gene_dataset_breakdown.csv"),
            "summary_csv": str(OUT_DIR / "cross_background_seen_gene_summary.csv"),
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
        },
        "decision": (
            "Exact dataset-background seen-gene slice is now available for Track A reporting/gates. "
            "This does not authorize GPU by itself."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "cross_background_seen_gene_rows.csv", all_rows)
    write_csv(OUT_DIR / "cross_background_seen_gene_summary.csv", [
        {k: v for k, v in s.items() if k != "dataset_breakdown"} for s in summaries
    ])
    write_csv(
        OUT_DIR / "cross_background_seen_gene_dataset_breakdown.csv",
        [r for s in summaries for r in s["dataset_breakdown"]],
    )
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Track A Exact Cross-Background Seen-Gene Slice",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only over frozen xverse_8k seed42/seed43 canonical posthoc JSONs.",
        "- No training, inference, Track C query, split recut, or canonical multi selection.",
        "- Background unit is `dataset` because the canonical metadata lacks a uniform cell/background field across all datasets.",
        "- Primary definition: canonical `test_single` row whose single perturbed gene appears in canonical train conditions of at least one different dataset.",
        "",
        "## Summary",
        "",
        "| seed | n cond | n ds | pearson_pert | CI95 | MMD clamped | MMD CI95 | dataset min pp |",
        "|---|---:|---:|---:|---|---:|---|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| `{s['seed']}` | {s['n_conditions']} | {s['n_datasets']} | "
            f"{fmt(s['pearson_pert_mean'])} | [{fmt(s['pearson_pert_ci95_low'])}, {fmt(s['pearson_pert_ci95_high'])}] | "
            f"{fmt(s['test_mmd_clamped_mean'])} | [{fmt(s['test_mmd_clamped_ci95_low'])}, {fmt(s['test_mmd_clamped_ci95_high'])}] | "
            f"{fmt(s['dataset_min_pearson_pert'])} |"
        )
    lines.extend(
        [
            "",
            f"Seed43 - seed42 pearson_pert: `{fmt(replicate_delta)}`",
            "",
            "## Decision",
            "",
            payload["decision"],
            "",
            "## Outputs",
            "",
        ]
    )
    for key, path in payload["outputs"].items():
        lines.append(f"- {key}: `{path}`")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
