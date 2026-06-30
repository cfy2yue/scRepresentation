#!/usr/bin/env python3
"""CPU-only Wessels-context adapter diagnostic for additive gene-response priors.

The adapter is fit only on canonical train single-gene conditions. Wessels held
out multi-condition groups are used only for evaluation and gating.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


BAD_TOKENS = {
    "CONTROL",
    "CTRL",
    "NON-TARGETING",
    "NONTARGETING",
    "POS",
    "TSS",
    "KLANN",
    "MOSAIC",
    "INTERGENIC",
}
WESSELS_GROUPS = ("test_multi_seen", "test_multi_unseen1", "test_multi_unseen2")


def parse_genes(condition: str) -> list[str]:
    genes = []
    for token in re.split(r"\+", str(condition)):
        gene = token.strip().upper()
        if not gene or gene in BAD_TOKENS or gene.startswith("CONTROL"):
            continue
        if re.fullmatch(r"[A-Z0-9][A-Z0-9.-]*", gene):
            genes.append(gene)
    return sorted(set(genes))


class H5Means:
    def __init__(self, h5_path: Path):
        self.handle = h5py.File(h5_path, "r")
        self.conditions = self.handle["conditions"].asstr()[:].tolist()
        self.cond2idx = {cond: idx for idx, cond in enumerate(self.conditions)}
        self.ctrl_key = "ctrl" if "ctrl/offsets" in self.handle else "ir"
        self.ctrl_offsets = self.handle[f"{self.ctrl_key}/offsets"][:]
        self.gt_offsets = self.handle["gt/offsets"][:]

    def close(self) -> None:
        self.handle.close()

    def mean_delta(self, condition: str) -> tuple[np.ndarray, int, int]:
        idx = self.cond2idx[condition]
        cs, ce = int(self.ctrl_offsets[idx]), int(self.ctrl_offsets[idx + 1])
        gs, ge = int(self.gt_offsets[idx]), int(self.gt_offsets[idx + 1])
        ctrl = self.handle[f"{self.ctrl_key}/emb"][cs:ce]
        gt = self.handle["gt/emb"][gs:ge]
        return np.asarray(gt.mean(axis=0) - ctrl.mean(axis=0), dtype=np.float32), int(ce - cs), int(ge - gs)


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    aa = a.astype(np.float64) - float(np.mean(a))
    bb = b.astype(np.float64) - float(np.mean(b))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-12:
        return None
    return float(np.dot(aa, bb) / denom)


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return None
    return float(np.dot(a, b) / denom)


def mean_or_none(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(clean)) if clean else None


def load_split(path: Path) -> dict[str, dict[str, list[str]]]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_handle(cache: dict[str, H5Means], data_dir: Path, dataset: str) -> H5Means:
    if dataset not in cache:
        cache[dataset] = H5Means(data_dir / f"{dataset}.h5")
    return cache[dataset]


def build_global_gene_priors(
    *,
    data_dir: Path,
    split: dict[str, dict[str, list[str]]],
    target_genes: set[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    handles: dict[str, H5Means] = {}
    vectors_by_gene: dict[str, list[np.ndarray]] = defaultdict(list)
    meta: dict[str, dict[str, Any]] = defaultdict(lambda: {"n_conditions": 0, "datasets": defaultdict(int)})
    for dataset, groups in split.items():
        for condition in groups.get("train", []):
            genes = parse_genes(condition)
            if len(genes) != 1 or genes[0] not in target_genes:
                continue
            handle = get_handle(handles, data_dir, dataset)
            if condition not in handle.cond2idx:
                continue
            delta, _, _ = handle.mean_delta(condition)
            gene = genes[0]
            vectors_by_gene[gene].append(delta)
            meta[gene]["n_conditions"] += 1
            meta[gene]["datasets"][dataset] += 1
    for handle in handles.values():
        handle.close()
    priors = {gene: np.mean(vectors, axis=0).astype(np.float32) for gene, vectors in vectors_by_gene.items()}
    clean_meta = {
        gene: {
            "n_conditions": info["n_conditions"],
            "datasets": dict(sorted(info["datasets"].items())),
        }
        for gene, info in meta.items()
    }
    return priors, clean_meta


def build_wessels_train_targets(
    *,
    data_dir: Path,
    wessels_split: dict[str, list[str]],
    priors: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    handle = H5Means(data_dir / "Wessels.h5")
    targets: dict[str, np.ndarray] = {}
    condition_by_gene: dict[str, str] = {}
    for condition in wessels_split.get("train", []):
        genes = parse_genes(condition)
        if len(genes) != 1 or genes[0] not in priors or condition not in handle.cond2idx:
            continue
        delta, _, _ = handle.mean_delta(condition)
        gene = genes[0]
        targets[gene] = delta
        condition_by_gene[gene] = condition
    handle.close()
    return targets, condition_by_gene


class Adapter:
    def __init__(self, name: str, alpha: float = 1.0, residual: np.ndarray | None = None):
        self.name = name
        self.alpha = float(alpha)
        self.residual = residual

    def apply(self, x: np.ndarray) -> np.ndarray:
        y = (self.alpha * x).astype(np.float32)
        if self.residual is not None:
            y = y + self.residual.astype(np.float32)
        return y.astype(np.float32)


def fit_adapter(name: str, x_rows: list[np.ndarray], y_rows: list[np.ndarray], residual_lambda: float = 1.0) -> Adapter:
    x = np.stack(x_rows).astype(np.float64)
    y = np.stack(y_rows).astype(np.float64)
    if name == "raw_global":
        return Adapter(name=name)
    if name == "scalar":
        denom = float(np.sum(x * x))
        alpha = 1.0 if denom <= 1e-12 else float(np.sum(x * y) / denom)
        return Adapter(name=name, alpha=alpha)
    if name == "mean_residual":
        residual = residual_lambda * np.mean(y - x, axis=0)
        return Adapter(name=f"{name}_lambda{residual_lambda:g}", residual=residual.astype(np.float32))
    if name == "scalar_mean_residual":
        denom = float(np.sum(x * x))
        alpha = 1.0 if denom <= 1e-12 else float(np.sum(x * y) / denom)
        residual = residual_lambda * np.mean(y - alpha * x, axis=0)
        return Adapter(name=f"{name}_lambda{residual_lambda:g}", alpha=alpha, residual=residual.astype(np.float32))
    raise ValueError(f"unknown adapter: {name}")


def train_table(
    *,
    priors: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    condition_by_gene: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    genes = sorted(set(priors) & set(targets))
    adapter_specs: list[tuple[str, float]] = [("raw_global", 0.0), ("scalar", 0.0)]
    for lam in (0.25, 0.5, 0.75, 1.0):
        adapter_specs.append(("mean_residual", lam))
        adapter_specs.append(("scalar_mean_residual", lam))
    rows: list[dict[str, Any]] = []
    for spec_name, lam in adapter_specs:
        for gene in genes:
            train_genes = [g for g in genes if g != gene]
            if not train_genes:
                continue
            adapter = fit_adapter(
                spec_name,
                [priors[g] for g in train_genes],
                [targets[g] for g in train_genes],
                residual_lambda=lam,
            )
            pred = adapter.apply(priors[gene])
            target = targets[gene]
            rows.append(
                {
                    "adapter": adapter.name,
                    "heldout_gene": gene,
                    "heldout_condition": condition_by_gene.get(gene, ""),
                    "pearson": pearson(pred, target),
                    "cosine": cosine(pred, target),
                    "norm_ratio": float(np.linalg.norm(pred) / max(np.linalg.norm(target), 1e-12)),
                }
            )
    summary = []
    for adapter in sorted({row["adapter"] for row in rows}):
        adapter_rows = [row for row in rows if row["adapter"] == adapter]
        summary.append(
            {
                "adapter": adapter,
                "loocv_n": len(adapter_rows),
                "mean_pearson": mean_or_none([row["pearson"] for row in adapter_rows]),
                "mean_cosine": mean_or_none([row["cosine"] for row in adapter_rows]),
                "mean_norm_ratio": mean_or_none([row["norm_ratio"] for row in adapter_rows]),
            }
        )
    best = sorted(
        summary,
        key=lambda row: (
            row["mean_pearson"] if row["mean_pearson"] is not None else -999.0,
            -(abs((row["mean_norm_ratio"] or 999.0) - 1.0)),
        ),
        reverse=True,
    )[0] if summary else {"adapter": "raw_global"}
    return rows, best


def fit_named_adapter(
    *,
    adapter_name: str,
    priors: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
) -> Adapter:
    genes = sorted(set(priors) & set(targets))
    if adapter_name == "raw_global":
        return fit_adapter("raw_global", [priors[g] for g in genes], [targets[g] for g in genes])
    if adapter_name == "scalar":
        return fit_adapter("scalar", [priors[g] for g in genes], [targets[g] for g in genes])
    match = re.fullmatch(r"(mean_residual|scalar_mean_residual)_lambda([0-9.]+)", adapter_name)
    if not match:
        raise ValueError(f"cannot refit adapter from name: {adapter_name}")
    return fit_adapter(match.group(1), [priors[g] for g in genes], [targets[g] for g in genes], float(match.group(2)))


def evaluate_additive_prior(
    *,
    data_dir: Path,
    wessels_split: dict[str, list[str]],
    priors: dict[str, np.ndarray],
    prior_meta: dict[str, Any],
    adapters: list[Adapter],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    handle = H5Means(data_dir / "Wessels.h5")
    condition_rows: list[dict[str, Any]] = []
    for adapter in adapters:
        adapted = {gene: adapter.apply(vec) for gene, vec in priors.items()}
        for group_name in WESSELS_GROUPS:
            for condition in wessels_split.get(group_name, []):
                genes = parse_genes(condition)
                available = [gene for gene in genes if gene in adapted]
                missing = [gene for gene in genes if gene not in adapted]
                target, n_ctrl, n_gt = handle.mean_delta(condition)
                pred = None
                if available and len(available) == len(genes):
                    pred = np.sum([adapted[gene] for gene in available], axis=0).astype(np.float32)
                condition_rows.append(
                    {
                        "adapter": adapter.name,
                        "group": group_name,
                        "condition": condition,
                        "genes": "+".join(genes),
                        "n_genes": len(genes),
                        "coverage_n": len(available),
                        "coverage_fraction": len(available) / max(len(genes), 1),
                        "missing_genes": "+".join(missing),
                        "prior_train_condition_count_sum": sum(
                            int(prior_meta.get(gene, {}).get("n_conditions", 0)) for gene in available
                        ),
                        "n_ctrl": n_ctrl,
                        "n_gt": n_gt,
                        "target_norm": float(np.linalg.norm(target)),
                        "pred_norm": None if pred is None else float(np.linalg.norm(pred)),
                        "pearson": None if pred is None else pearson(pred, target),
                        "cosine": None if pred is None else cosine(pred, target),
                        "norm_ratio": None
                        if pred is None
                        else float(np.linalg.norm(pred) / max(np.linalg.norm(target), 1e-12)),
                    }
                )
    handle.close()
    group_rows: list[dict[str, Any]] = []
    for adapter in sorted({row["adapter"] for row in condition_rows}):
        for group_name in WESSELS_GROUPS:
            rows = [row for row in condition_rows if row["adapter"] == adapter and row["group"] == group_name]
            covered = [row for row in rows if row["coverage_fraction"] >= 1.0 and row["pearson"] is not None]
            group_rows.append(
                {
                    "adapter": adapter,
                    "group": group_name,
                    "n_conditions": len(rows),
                    "n_full_coverage": len(covered),
                    "mean_pearson": mean_or_none([row["pearson"] for row in covered]),
                    "mean_cosine": mean_or_none([row["cosine"] for row in covered]),
                    "mean_norm_ratio": mean_or_none([row["norm_ratio"] for row in covered]),
                }
            )
    return condition_rows, group_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Wessels Context-Adapted Additive Prior Diagnostic",
        "",
        "This CPU-only gate fits context adapters only on Wessels canonical train single-gene conditions.",
        "Held-out Wessels multi-condition groups are used only for evaluation.",
        "",
        f"Selected adapter by train-single LOOCV: `{payload['selected_adapter']}`",
        "",
        f"CPU gate: `{payload['gate']['status']}`; unseen2 Pearson delta "
        f"{fmt(payload['gate']['unseen2_pearson_delta_vs_raw'])} vs >= {payload['gate']['min_delta']}, "
        f"mean pred/target norm {fmt(payload['gate']['selected_unseen2_norm_ratio'])} vs <= {payload['gate']['max_norm_ratio']}.",
        "",
        "## Train-Single LOOCV Summary",
        "",
        "| adapter | n | mean Pearson | mean cosine | mean pred/target norm |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["train_summary"]:
        lines.append(
            f"| `{row['adapter']}` | {row['loocv_n']} | {fmt(row['mean_pearson'])} | "
            f"{fmt(row['mean_cosine'])} | {fmt(row['mean_norm_ratio'])} |"
        )
    lines.extend(
        [
            "",
            "## Held-Out Multi Summary",
            "",
            "| adapter | group | n | full coverage | mean Pearson | mean cosine | mean pred/target norm |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["group_rows"]:
        lines.append(
            f"| `{row['adapter']}` | `{row['group']}` | {row['n_conditions']} | "
            f"{row['n_full_coverage']}/{row['n_conditions']} | {fmt(row['mean_pearson'])} | "
            f"{fmt(row['mean_cosine'])} | {fmt(row['mean_norm_ratio'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `pass` means the adapted additive prior is strong enough to justify a narrow GPU diagnostic.",
            "- `fail` means context adaptation does not improve the train-only additive prior enough; prioritize other condition-modeling diagnostics before GPU training.",
            "- This is diagnostic screening only, not paper-level evidence.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/data/cyx/1030/dataset/latentfm_full/scfoundation"))
    parser.add_argument("--split-file", type=Path, default=Path("/data/cyx/1030/dataset/biFlow_data/split_seed42.json"))
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-conditions-csv", type=Path, required=True)
    parser.add_argument("--out-groups-csv", type=Path, required=True)
    parser.add_argument("--out-train-csv", type=Path, required=True)
    args = parser.parse_args()

    split = load_split(args.split_file)
    wessels_split = split["Wessels"]
    target_genes = {
        gene
        for group_name in (*WESSELS_GROUPS, "train")
        for condition in wessels_split.get(group_name, [])
        for gene in parse_genes(condition)
    }
    priors, prior_meta = build_global_gene_priors(data_dir=args.data_dir, split=split, target_genes=target_genes)
    train_targets, condition_by_gene = build_wessels_train_targets(
        data_dir=args.data_dir,
        wessels_split=wessels_split,
        priors=priors,
    )
    train_rows, selected = train_table(priors=priors, targets=train_targets, condition_by_gene=condition_by_gene)
    selected_adapter = str(selected.get("adapter", "raw_global"))
    adapters = [
        fit_named_adapter(adapter_name="raw_global", priors=priors, targets=train_targets),
        fit_named_adapter(adapter_name=selected_adapter, priors=priors, targets=train_targets),
    ]
    condition_rows, group_rows = evaluate_additive_prior(
        data_dir=args.data_dir,
        wessels_split=wessels_split,
        priors=priors,
        prior_meta=prior_meta,
        adapters=adapters,
    )
    raw_u2 = next(row for row in group_rows if row["adapter"] == "raw_global" and row["group"] == "test_multi_unseen2")
    selected_u2 = next(row for row in group_rows if row["adapter"] == selected_adapter and row["group"] == "test_multi_unseen2")
    delta = None
    if raw_u2["mean_pearson"] is not None and selected_u2["mean_pearson"] is not None:
        delta = float(selected_u2["mean_pearson"] - raw_u2["mean_pearson"])
    gate = {
        "status": "pass"
        if delta is not None
        and delta >= 0.05
        and selected_u2["mean_norm_ratio"] is not None
        and selected_u2["mean_norm_ratio"] <= 2.0
        else "fail",
        "selected_adapter": selected_adapter,
        "unseen2_pearson_delta_vs_raw": delta,
        "selected_unseen2_norm_ratio": selected_u2["mean_norm_ratio"],
        "min_delta": 0.05,
        "max_norm_ratio": 2.0,
    }
    train_summary = []
    for adapter in sorted({row["adapter"] for row in train_rows}):
        rows = [row for row in train_rows if row["adapter"] == adapter]
        train_summary.append(
            {
                "adapter": adapter,
                "loocv_n": len(rows),
                "mean_pearson": mean_or_none([row["pearson"] for row in rows]),
                "mean_cosine": mean_or_none([row["cosine"] for row in rows]),
                "mean_norm_ratio": mean_or_none([row["norm_ratio"] for row in rows]),
            }
        )
    payload = {
        "data_dir": str(args.data_dir),
        "split_file": str(args.split_file),
        "wessels_train_genes_used": sorted(train_targets),
        "n_wessels_train_genes_used": len(train_targets),
        "selected_adapter": selected_adapter,
        "gate": gate,
        "train_summary": train_summary,
        "group_rows": group_rows,
        "interpretation": "CPU-only diagnostic; no held-out multi GT used for fitting or adapter selection.",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(args.out_conditions_csv, condition_rows)
    write_csv(args.out_groups_csv, group_rows)
    write_csv(args.out_train_csv, train_rows)
    write_md(args.out_md, payload)
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "gate": gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
