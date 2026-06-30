#!/usr/bin/env python3
"""CPU gate for a train-only gene-reliability router.

The question is whether a deployable reliability shrink/router between
train-only `gene_raw_mean` and `dataset_mean` residual baselines contains enough
Track A single/background signal to justify a new GPU mechanism.

This script uses only the xverse train-only cross-background split. It does not
read canonical test posthoc, held-out multi query, or model checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
DEFAULT_PERT_MEANS = (
    ROOT
    / "runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/"
    "xverse_trainonly_pert_means_split_seed42_crossbgval_v2.npz"
)
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_gene_reliability_router_gate_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_GENE_RELIABILITY_ROUTER_GATE_20260622.md"

GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
K_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
THRESH_GRID = (1, 2, 3, 5, 10, 20, 50)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_condition_metadata_file(data_dir: Path, manifest: dict[str, Any]) -> tuple[Path, str]:
    configured = manifest.get("condition_metadata_file")
    if configured:
        path = Path(str(configured))
        if not path.is_absolute():
            path = data_dir / path
        return path, "manifest.condition_metadata_file"
    fallback = data_dir / "condition_metadata.json"
    if fallback.is_file():
        return fallback, "data_dir_condition_metadata_json_fallback"
    raise KeyError(
        "condition_metadata_file missing from manifest and "
        f"fallback does not exist: {fallback}"
    )


def decode(values: np.ndarray) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def stable_subset(items: list[str], k: int, key: str) -> list[str]:
    if k <= 0 or len(items) <= k:
        return list(items)
    return sorted(items, key=lambda x: hashlib.sha256(f"{key}|{x}".encode()).hexdigest())[:k]


def condition_mean(handle: h5py.File, group: str, idx: int, max_cells: int) -> np.ndarray | None:
    offsets = np.asarray(handle[f"{group}/offsets"])
    start, end = int(offsets[idx]), int(offsets[idx + 1])
    if end <= start:
        return None
    if max_cells > 0 and end - start > max_cells:
        end = start + max_cells
    return np.asarray(handle[f"{group}/emb"][start:end], dtype=np.float32).mean(axis=0)


def residual_for_condition(
    handle: h5py.File,
    by_cond: dict[str, int],
    cond: str,
    max_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    idx = by_cond.get(cond)
    if idx is None:
        return None
    ctrl = condition_mean(handle, "ctrl", idx, max_cells)
    gt = condition_mean(handle, "gt", idx, max_cells)
    if ctrl is None or gt is None:
        return None
    return ctrl.astype(np.float32), gt.astype(np.float32), (gt - ctrl).astype(np.float32)


def single_gene(metadata: dict[str, Any], ds: str, cond: str) -> str | None:
    meta = (metadata.get(ds) or {}).get(cond) or {}
    genes = [str(g).strip() for g in meta.get("genes") or [] if str(g).strip()]
    if len(genes) != 1:
        return None
    raw = str(meta.get("perturbation_type_raw") or "").lower()
    if "drug" in raw or "compound" in raw or "chemical" in raw:
        return None
    return genes[0]


def collect_rows(
    data_dir: Path,
    split: dict[str, Any],
    metadata: dict[str, Any],
    *,
    max_train_per_dataset: int,
    max_cells: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for ds, obj in sorted(split.items()):
        path = data_dir / f"{ds}.h5"
        if not path.is_file():
            continue
        train_single = []
        for cond in obj.get("train") or []:
            cond = str(cond)
            gene = single_gene(metadata, ds, cond)
            if gene is not None:
                train_single.append((cond, gene))
        chosen_train = set(stable_subset([c for c, _ in train_single], max_train_per_dataset, f"gene_router|train|{ds}"))
        with h5py.File(path, "r") as handle:
            by_cond = {c: i for i, c in enumerate(decode(np.asarray(handle["conditions"])))}
            for cond, gene in train_single:
                if cond not in chosen_train:
                    continue
                vals = residual_for_condition(handle, by_cond, cond, max_cells)
                if vals is None:
                    continue
                ctrl, gt, residual = vals
                train_rows.append(
                    {
                        "dataset": ds,
                        "condition": cond,
                        "gene": gene,
                        "ctrl": ctrl,
                        "gt": gt,
                        "residual": residual,
                    }
                )
            for group in GROUPS:
                for cond in obj.get(group) or []:
                    cond = str(cond)
                    gene = single_gene(metadata, ds, cond)
                    if gene is None:
                        continue
                    vals = residual_for_condition(handle, by_cond, cond, max_cells)
                    if vals is None:
                        continue
                    ctrl, gt, residual = vals
                    val_rows.append(
                        {
                            "dataset": ds,
                            "condition": cond,
                            "gene": gene,
                            "group": group,
                            "ctrl": ctrl,
                            "gt": gt,
                            "residual": residual,
                        }
                    )
    return train_rows, val_rows


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 3 or x.size != y.size:
        return None
    x -= x.mean()
    y -= y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return None
    return float(np.dot(x, y) / denom)


def score(row: dict[str, Any], pred_residual: np.ndarray, pert_means: dict[str, np.ndarray]) -> float | None:
    pert = pert_means.get(str(row["dataset"]))
    pred_endpoint = np.asarray(row["ctrl"], dtype=np.float32) + np.asarray(pred_residual, dtype=np.float32)
    gt_endpoint = np.asarray(row["gt"], dtype=np.float32)
    if pert is None:
        return pearson(pred_residual, row["residual"])
    return pearson(pred_endpoint - pert, gt_endpoint - pert)


def build_sums(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_ds_sum: dict[str, np.ndarray] = {}
    by_gene_sum: dict[str, np.ndarray] = {}
    by_ds_count: dict[str, int] = defaultdict(int)
    by_gene_count: dict[str, int] = defaultdict(int)
    total = None
    n_total = 0
    for row in rows:
        residual = np.asarray(row["residual"], dtype=np.float32)
        ds = str(row["dataset"])
        gene = str(row["gene"])
        by_ds_sum[ds] = residual.copy() if ds not in by_ds_sum else by_ds_sum[ds] + residual
        by_gene_sum[gene] = residual.copy() if gene not in by_gene_sum else by_gene_sum[gene] + residual
        by_ds_count[ds] += 1
        by_gene_count[gene] += 1
        total = residual.copy() if total is None else total + residual
        n_total += 1
    if total is None:
        raise ValueError("no train rows")
    return {
        "by_ds_sum": by_ds_sum,
        "by_gene_sum": by_gene_sum,
        "by_ds_count": dict(by_ds_count),
        "by_gene_count": dict(by_gene_count),
        "total_sum": total,
        "total_count": n_total,
    }


def mean_or_global(sum_arr: np.ndarray | None, count: int, global_mean: np.ndarray) -> np.ndarray:
    if sum_arr is None or count <= 0:
        return global_mean
    return (sum_arr / float(count)).astype(np.float32)


def component_means(
    sums: dict[str, Any],
    row: dict[str, Any] | None = None,
    *,
    exclude_row: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    total_sum = sums["total_sum"]
    total_count = int(sums["total_count"])
    ds = str(row["dataset"]) if row is not None else ""
    gene = str(row["gene"]) if row is not None else ""
    residual = np.asarray(row["residual"], dtype=np.float32) if row is not None else None

    if exclude_row and residual is not None:
        global_sum = total_sum - residual
        global_count = total_count - 1
    else:
        global_sum = total_sum
        global_count = total_count
    global_mean = (global_sum / float(max(global_count, 1))).astype(np.float32)

    ds_sum = sums["by_ds_sum"].get(ds)
    ds_count = int(sums["by_ds_count"].get(ds, 0))
    gene_sum = sums["by_gene_sum"].get(gene)
    gene_count = int(sums["by_gene_count"].get(gene, 0))
    if exclude_row and residual is not None:
        if ds_sum is not None:
            ds_sum = ds_sum - residual
            ds_count -= 1
        if gene_sum is not None:
            gene_sum = gene_sum - residual
            gene_count -= 1
    dataset_mean = mean_or_global(ds_sum, ds_count, global_mean)
    gene_mean = mean_or_global(gene_sum, gene_count, global_mean)
    return dataset_mean, gene_mean, global_mean, gene_count


def predict_from_components(
    dataset_mean: np.ndarray,
    gene_mean: np.ndarray,
    global_mean: np.ndarray,
    gene_count: int,
    model: str,
) -> np.ndarray:
    if model == "dataset_mean":
        return dataset_mean
    if model == "gene_raw_mean":
        return gene_mean
    if model == "global_mean":
        return global_mean
    if model.startswith("shrink_k"):
        k = float(model.removeprefix("shrink_k"))
        alpha = float(gene_count) / (float(gene_count) + k)
        return (alpha * gene_mean + (1.0 - alpha) * dataset_mean).astype(np.float32)
    if model.startswith("route_t"):
        thresh = int(model.removeprefix("route_t"))
        return gene_mean if gene_count >= thresh else dataset_mean
    raise ValueError(model)


def candidate_models() -> list[str]:
    return (
        ["dataset_mean", "gene_raw_mean", "global_mean"]
        + [f"shrink_k{k:g}" for k in K_GRID]
        + [f"route_t{t}" for t in THRESH_GRID]
    )


def evaluate_rows(
    rows: list[dict[str, Any]],
    sums: dict[str, Any],
    pert_means: dict[str, np.ndarray],
    models: list[str],
    *,
    exclude_row: bool,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        dataset_mean, gene_mean, global_mean, gene_count = component_means(sums, row, exclude_row=exclude_row)
        scored = {
            "dataset": row["dataset"],
            "condition": row["condition"],
            "gene": row["gene"],
            "group": row.get("group", "train_leave_one"),
            "gene_train_count": gene_count,
        }
        for model in models:
            pred = predict_from_components(dataset_mean, gene_mean, global_mean, gene_count, model)
            scored[model] = score(row, pred, pert_means)
        out.append(scored)
    return out


def equal_dataset_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        val = row.get(key)
        if val is not None:
            by_ds[str(row["dataset"])].append(float(val))
    vals = [float(np.mean(v)) for v in by_ds.values() if v]
    return None if not vals else float(np.mean(vals))


def paired_bootstrap(
    rows: list[dict[str, Any]],
    candidate: str,
    baseline: str,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    diffs_by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        a = row.get(candidate)
        b = row.get(baseline)
        if a is not None and b is not None:
            diffs_by_ds[str(row["dataset"])].append(float(a) - float(b))
    datasets = sorted(ds for ds, vals in diffs_by_ds.items() if vals)
    if not datasets:
        return {"status": "missing", "candidate": candidate, "baseline": baseline}
    point = float(np.mean([np.mean(diffs_by_ds[ds]) for ds in datasets]))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        sample_ds = rng.choice(datasets, size=len(datasets), replace=True)
        means = []
        for ds in sample_ds:
            vals = np.asarray(diffs_by_ds[str(ds)], dtype=np.float64)
            means.append(float(np.mean(rng.choice(vals, size=len(vals), replace=True))))
        boot.append(float(np.mean(means)))
    arr = np.asarray(boot, dtype=np.float64)
    leave_one = {}
    for ds in datasets:
        rest = [d for d in datasets if d != ds]
        if rest:
            leave_one[ds] = float(np.mean([np.mean(diffs_by_ds[d]) for d in rest]))
    return {
        "status": "ok",
        "candidate": candidate,
        "baseline": baseline,
        "n_conditions": int(sum(len(diffs_by_ds[d]) for d in datasets)),
        "n_datasets": int(len(datasets)),
        "delta_mean": point,
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_improve": float(np.mean(arr > 0.0)),
        "p_harm": float(np.mean(arr < 0.0)),
        "leave_one_min": float(min(leave_one.values())) if leave_one else point,
    }


def select_model(train_eval: list[dict[str, Any]], models: list[str]) -> dict[str, Any]:
    scores = [{"model": model, "train_leave_one_pp": equal_dataset_mean(train_eval, model)} for model in models]
    nontrivial = [row for row in scores if row["model"] not in {"dataset_mean", "gene_raw_mean", "global_mean"}]
    best = max(nontrivial, key=lambda row: float(row["train_leave_one_pp"] if row["train_leave_one_pp"] is not None else -1e9))
    return {"selected_model": best["model"], "train_leave_one_scores": scores}


def group_rows(rows: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("group") == group]


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    selected = payload["selected_model"]
    by_key = {
        (row["group"], row["candidate"], row["baseline"]): row
        for row in payload["paired_deltas"]
        if row.get("status") == "ok"
    }
    reasons = []
    primary_gene = by_key.get(("internal_val_cross_background_seen_gene_proxy", selected, "gene_raw_mean"), {})
    primary_ds = by_key.get(("internal_val_cross_background_seen_gene_proxy", selected, "dataset_mean"), {})
    family_gene = by_key.get(("internal_val_family_gene_proxy", selected, "gene_raw_mean"), {})
    family_ds = by_key.get(("internal_val_family_gene_proxy", selected, "dataset_mean"), {})
    if float(primary_gene.get("delta_mean") or 0.0) < 0.01 and not (
        (primary_gene.get("ci95") or [0.0])[0] > 0.0
    ):
        reasons.append("cross_background_not_materially_better_than_gene_raw_mean")
    if float(primary_gene.get("p_improve") or 0.0) < 0.75:
        reasons.append("cross_background_vs_gene_raw_not_bootstrap_supported")
    if float(primary_ds.get("p_harm") if primary_ds.get("p_harm") is not None else 1.0) > 0.20:
        reasons.append("cross_background_vs_dataset_mean_harm_risk")
    if float(family_gene.get("p_harm") if family_gene.get("p_harm") is not None else 1.0) > 0.20:
        reasons.append("family_vs_gene_raw_harm_risk")
    if float(family_ds.get("p_harm") if family_ds.get("p_harm") is not None else 1.0) > 0.20:
        reasons.append("family_vs_dataset_mean_harm_risk")
    status = "cpu_gate_pass_design_one_gene_reliability_adapter" if not reasons else "cpu_gate_fail_do_not_launch_gpu"
    action = (
        "design_one_small_ema_consistent_gene_reliability_adapter_smoke"
        if not reasons
        else "close_simple_gene_reliability_router"
    )
    return {"status": status, "action": action, "reasons": reasons}


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def render(payload: dict[str, Any]) -> str:
    selected = payload["selected_model"]
    latent_label = payload.get("latent_label") or "xverse"
    lines = [
        f"# LatentFM {latent_label} Gene-Reliability Router CPU Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"Recommended action: `{payload['decision']['action']}`",
        "",
        "## Provenance",
        "",
        f"- data_dir: `{payload['data_dir']}`",
        f"- split_file: `{payload['split_file']}`",
        f"- condition_metadata_file: `{payload.get('condition_metadata_file')}`",
        f"- condition_metadata_source: `{payload.get('condition_metadata_source')}`",
        f"- pert_means_file: `{payload['pert_means_file']}`",
        f"- leakage status: `{payload['leakage_status']}`",
        f"- train rows: `{payload['n_train_rows']}`",
        f"- validation rows: `{payload['n_val_rows']}`",
        f"- selected model from train leave-one: `{selected}`",
        "",
        "## Train Leave-One Scores",
        "",
        "| model | equal-dataset pp |",
        "|---|---:|",
    ]
    for row in payload["train_leave_one_scores"]:
        if row["model"] in {"dataset_mean", "gene_raw_mean", "global_mean", selected} or row["model"].startswith("shrink_k"):
            lines.append(f"| `{row['model']}` | {fmt(row['train_leave_one_pp'])} |")
    lines += [
        "",
        "## Internal Validation Absolute Scores",
        "",
        "| group | model | equal-dataset pp |",
        "|---|---|---:|",
    ]
    for row in payload["absolute_scores"]:
        lines.append(f"| {row['group']} | `{row['model']}` | {fmt(row['pp'])} |")
    lines += [
        "",
        "## Paired Deltas",
        "",
        "| group | candidate | baseline | n cond | n ds | delta | 95% CI | p improve | p harm | leave-one min |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in payload["paired_deltas"]:
        ci = row.get("ci95") or [None, None]
        lines.append(
            f"| {row['group']} | `{row['candidate']}` | `{row['baseline']}` | "
            f"{row.get('n_conditions', 0)} | {row.get('n_datasets', 0)} | "
            f"{fmt(row.get('delta_mean'))} | [{fmt(ci[0])}, {fmt(ci[1])}] | "
            f"{fmt(row.get('p_improve'))} | {fmt(row.get('p_harm'))} | "
            f"{fmt(row.get('leave_one_min'))} |"
        )
    lines += ["", "## Gate Reasons", ""]
    reasons = payload["decision"].get("reasons") or []
    lines.extend([f"- `{r}`" for r in reasons] if reasons else ["- none"])
    lines += [
        "",
        "## Usage Rule",
        "",
        "- This is a CPU-only train/internal-val diagnostic.",
        "- It does not read canonical test, held-out multi query, or model posthoc outputs.",
        "- Passing would authorize at most one small EMA-consistent Track A adapter smoke.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--pert-means-file", type=Path, default=DEFAULT_PERT_MEANS)
    parser.add_argument("--max-train-per-dataset", type=int, default=768)
    parser.add_argument("--max-cells-per-condition", type=int, default=256)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    split = load_json(args.split_file)
    manifest = load_json(data_dir / "manifest.json")
    metadata_file, metadata_source = resolve_condition_metadata_file(data_dir, manifest)
    metadata = load_json(metadata_file)
    pert_means = {k: v.astype(np.float32) for k, v in np.load(args.pert_means_file).items()}
    train_rows, val_rows = collect_rows(
        data_dir,
        split,
        metadata,
        max_train_per_dataset=args.max_train_per_dataset,
        max_cells=args.max_cells_per_condition,
    )
    sums = build_sums(train_rows)
    models = candidate_models()
    train_eval = evaluate_rows(train_rows, sums, pert_means, models, exclude_row=True)
    selection = select_model(train_eval, models)
    selected = selection["selected_model"]
    eval_models = ["dataset_mean", "gene_raw_mean", "global_mean", selected]
    val_eval = evaluate_rows(val_rows, sums, pert_means, eval_models, exclude_row=False)

    absolute = []
    for group in GROUPS:
        rows = group_rows(val_eval, group)
        for model in eval_models:
            absolute.append({"group": group, "model": model, "pp": equal_dataset_mean(rows, model)})

    deltas = []
    for group in GROUPS:
        rows = group_rows(val_eval, group)
        for baseline in ("gene_raw_mean", "dataset_mean", "global_mean"):
            deltas.append(
                {
                    "group": group,
                    **paired_bootstrap(rows, selected, baseline, n_boot=args.n_boot, seed=args.seed + len(deltas)),
                }
            )

    payload = {
        "latent_label": data_dir.name,
        "data_dir": str(data_dir),
        "split_file": str(args.split_file),
        "condition_metadata_file": str(metadata_file),
        "condition_metadata_source": metadata_source,
        "pert_means_file": str(args.pert_means_file),
        "max_train_per_dataset": args.max_train_per_dataset,
        "max_cells_per_condition": args.max_cells_per_condition,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "leakage_status": "train_only_v2_train_leave_one_to_internal_proxy_no_canonical_test_no_posthoc_no_heldout_multi",
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
        "selected_model": selected,
        "train_leave_one_scores": selection["train_leave_one_scores"],
        "absolute_scores": absolute,
        "paired_deltas": deltas,
        "val_condition_rows": val_eval,
    }
    payload["decision"] = decide(payload)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "selected_model": selected, "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
