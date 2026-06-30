#!/usr/bin/env python
"""Summarize LatentFM condition-residual strata with bootstrap CIs.

Inputs are ``condition_residual_full128_best.csv`` files produced by
``model.latent.eval_condition_residuals``.  The script is CPU-only and does not
touch checkpoints or GPUs.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


RUNS = {
    "prior010_no_injection": Path(
        "/data/cyx/1030/scLatent/CoupledFM/output/latentfm_runs/"
        "condition_prior_teacher_probe_20260619/scf_prior010_e2_4k/"
        "posthoc_eval/condition_residual_full128_best.csv"
    ),
    "prior010_injection": Path(
        "/data/cyx/1030/scLatent/CoupledFM/output/latentfm_runs/"
        "condition_prior_teacher_injection_20260619/scf_prior010_inject_e2_4k/"
        "posthoc_eval/condition_residual_full128_best.csv"
    ),
    "prioradd005_injection": Path(
        "/data/cyx/1030/scLatent/CoupledFM/output/latentfm_runs/"
        "condition_prior_additive_head_20260619/scf_prioradd005_prior010_inject_e2_4k/"
        "posthoc_eval/condition_residual_full128_best.csv"
    ),
}

OUT_PREFIX = Path("/data/cyx/1030/scLatent/reports/latentfm_residual_strata_prior_branches_20260619")
BOOTSTRAPS = 1000
SEED = 20260619


def _float(row: dict[str, str], key: str) -> float | None:
    val = str(row.get(key, "")).strip()
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _load_run(name: str, path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            target_norm = _float(row, "target_norm")
            pred_norm = _float(row, "pred_norm")
            pearson = _float(row, "pred_target_pearson")
            cosine = _float(row, "pred_target_cosine")
            if target_norm is None or pred_norm is None or pearson is None:
                continue
            groups = [
                g.strip()
                for g in str(row.get("groups", "")).split(",")
                if g.strip()
            ]
            rows.append(
                {
                    "run": name,
                    "dataset": row.get("dataset", ""),
                    "condition": row.get("condition", ""),
                    "groups": groups,
                    "target_norm": target_norm,
                    "pred_norm": pred_norm,
                    "norm_ratio": pred_norm / target_norm if target_norm else None,
                    "pearson": pearson,
                    "cosine": cosine,
                    "family": row.get("perturbation_family", ""),
                    "n_genes": int(float(row.get("n_genes", "0") or 0)),
                    "is_multi": str(row.get("is_multi", "")).lower() == "true",
                }
            )
    return rows


def _quantile_edges(values: list[float], n_bins: int = 4) -> list[float]:
    vals = sorted(values)
    if not vals:
        return []
    edges = []
    for i in range(1, n_bins):
        idx = min(len(vals) - 1, max(0, round(i * (len(vals) - 1) / n_bins)))
        edges.append(vals[idx])
    return edges


def _bin(value: float, edges: list[float]) -> str:
    labels = ["q1_low", "q2", "q3", "q4_high"]
    for i, edge in enumerate(edges):
        if value <= edge:
            return labels[i]
    return labels[min(len(edges), len(labels) - 1)]


def _ci(vals: list[float], *, rng: random.Random) -> tuple[float, float, float]:
    if not vals:
        return float("nan"), float("nan"), float("nan")
    point = mean(vals)
    if len(vals) == 1:
        return point, point, point
    boots = []
    n = len(vals)
    for _ in range(BOOTSTRAPS):
        boots.append(mean(vals[rng.randrange(n)] for _ in range(n)))
    boots.sort()
    lo = boots[int(0.025 * (len(boots) - 1))]
    hi = boots[int(0.975 * (len(boots) - 1))]
    return point, lo, hi


def _summarize(items: list[dict[str, Any]], key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        key = tuple(item[k] for k in key_fields)
        grouped[key].append(item)
    rng = random.Random(SEED)
    rows = []
    for key, vals in sorted(grouped.items()):
        pearsons = [float(v["pearson"]) for v in vals]
        cosines = [float(v["cosine"]) for v in vals if v["cosine"] is not None]
        ratios = [float(v["norm_ratio"]) for v in vals if v["norm_ratio"] is not None]
        pp, lo, hi = _ci(pearsons, rng=rng)
        row = {field: value for field, value in zip(key_fields, key)}
        row.update(
            {
                "n": len(vals),
                "pearson_mean": pp,
                "pearson_ci95_low": lo,
                "pearson_ci95_high": hi,
                "cosine_mean": mean(cosines) if cosines else None,
                "target_norm_median": median([float(v["target_norm"]) for v in vals]),
                "norm_ratio_median": median(ratios) if ratios else None,
            }
        )
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    group_rows = payload["by_group"]
    norm_rows = payload["by_norm_bin"]
    dataset_rows = payload["by_key_dataset_group"]

    def rows_for_group(group: str) -> list[dict[str, Any]]:
        return [r for r in group_rows if r["group"] == group]

    lines = [
        "# LatentFM Residual Strata: Prior Branches",
        "",
        "Generated: 2026-06-19 18:55 CST",
        "",
        "Inputs:",
    ]
    for name, path_s in payload["inputs"].items():
        lines.append(f"- `{name}`: `{path_s}`")
    lines += [
        "",
        "## Key Groups",
        "",
        "| group | run | n | pearson mean | 95% CI | median target norm | median pred/target norm |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for group in ["test_single", "test_multi_seen", "test_multi_unseen1", "test_multi_unseen2", "family_gene", "family_drug"]:
        for r in rows_for_group(group):
            lines.append(
                f"| `{group}` | `{r['run']}` | {r['n']} | "
                f"{r['pearson_mean']:.4f} | [{r['pearson_ci95_low']:.4f}, {r['pearson_ci95_high']:.4f}] | "
                f"{r['target_norm_median']:.3f} | {r['norm_ratio_median']:.3f} |"
            )
    lines += [
        "",
        "## Target-Norm Bins",
        "",
        "| run | norm bin | n | pearson mean | 95% CI | median target norm | median pred/target norm |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in norm_rows:
        lines.append(
            f"| `{r['run']}` | `{r['target_norm_bin']}` | {r['n']} | "
            f"{r['pearson_mean']:.4f} | [{r['pearson_ci95_low']:.4f}, {r['pearson_ci95_high']:.4f}] | "
            f"{r['target_norm_median']:.3f} | {r['norm_ratio_median']:.3f} |"
        )
    lines += [
        "",
        "## Multi-Unseen2 Dataset Detail",
        "",
        "| dataset | run | n | pearson mean | 95% CI | median target norm | median pred/target norm |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in dataset_rows:
        if r["group"] != "test_multi_unseen2":
            continue
        lines.append(
            f"| `{r['dataset']}` | `{r['run']}` | {r['n']} | "
            f"{r['pearson_mean']:.4f} | [{r['pearson_ci95_low']:.4f}, {r['pearson_ci95_high']:.4f}] | "
            f"{r['target_norm_median']:.3f} | {r['norm_ratio_median']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- These are condition-level residual diagnostics, not full ODE/MMD metrics.",
        "- CIs are condition-bootstrap intervals and should be used for branch triage, not formal final statistics.",
        "- A useful sampling or preprocessing branch should improve multi-unseen2 without worsening family_gene or aggregate residual pearson.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    all_rows: list[dict[str, Any]] = []
    inputs = {}
    for name, path in RUNS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        inputs[name] = str(path)
        all_rows.extend(_load_run(name, path))

    edges = _quantile_edges([float(r["target_norm"]) for r in all_rows], n_bins=4)
    binned_rows: list[dict[str, Any]] = []
    expanded: list[dict[str, Any]] = []
    for row in all_rows:
        base = dict(row)
        base["target_norm_bin"] = _bin(float(row["target_norm"]), edges)
        binned_rows.append(base)
        for group in row["groups"]:
            item = dict(base)
            item["group"] = group
            expanded.append(item)

    by_group = _summarize(expanded, ("group", "run"))
    by_norm = _summarize(binned_rows, ("run", "target_norm_bin"))
    by_dataset_group = _summarize(expanded, ("dataset", "group", "run"))
    payload = {
        "inputs": inputs,
        "target_norm_edges": edges,
        "by_group": by_group,
        "by_norm_bin": by_norm,
        "by_key_dataset_group": by_dataset_group,
    }
    OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    json_path = OUT_PREFIX.with_suffix(".json")
    group_csv = OUT_PREFIX.with_name(OUT_PREFIX.name + "_by_group.csv")
    norm_csv = OUT_PREFIX.with_name(OUT_PREFIX.name + "_by_norm_bin.csv")
    dataset_csv = OUT_PREFIX.with_name(OUT_PREFIX.name + "_by_dataset_group.csv")
    report_path = OUT_PREFIX.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(group_csv, by_group)
    _write_csv(norm_csv, by_norm)
    _write_csv(dataset_csv, by_dataset_group)
    _write_report(report_path, payload)
    print(f"wrote {json_path}")
    print(f"wrote {group_csv}")
    print(f"wrote {norm_csv}")
    print(f"wrote {dataset_csv}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
