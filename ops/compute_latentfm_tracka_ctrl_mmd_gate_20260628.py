#!/usr/bin/env python3
"""Compute control/source-vs-GT MMD for explicit Track A proxy rows.

This is an eval-only diagnostic: no model checkpoint is loaded and no
threshold/model selection is performed. It pairs the control/source endpoint
MMD against the frozen anchor MMD already present in the explicit proxy rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

ROOT = Path("/data/cyx/1030/scLatent")
sys.path.insert(0, str(ROOT / "CoupledFM"))

from model.latent.fm_ot import median_sigmas, mmd2_biased, mmd2_unbiased  # noqa: E402


DEFAULT_ROWS = ROOT / "reports/tracka_explicit_group_proxy_benchmark_20260628/condition_rows.csv"
DEFAULT_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
DEFAULT_OUT_DIR = ROOT / "reports/tracka_ctrl_mmd_gate_20260628"
GROUP_ORDER = (
    "all_test_single_proxy",
    "cross_background_seen_gene_proxy",
    "family_gene",
    "simple_single_unseen_global_gene_proxy",
)


def stable_int_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def eval_rank(seed: int, *parts: object) -> int:
    text = "eval_select:" + ":".join(str(p) for p in (seed, *parts))
    return stable_int_hash(text)


def read_rows(ds: h5py.Dataset, start: int, rel_idx: np.ndarray) -> np.ndarray:
    rel_idx = np.asarray(rel_idx, dtype=np.int64)
    if rel_idx.size == 0:
        return np.empty((0, int(ds.shape[1])), dtype=np.float32)
    uniq, inverse = np.unique(rel_idx, return_inverse=True)
    block = ds[uniq + int(start)]
    return np.asarray(block[inverse], dtype=np.float32)


class H5Cache:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.handles: dict[str, h5py.File] = {}
        self.conds: dict[str, dict[str, int]] = {}

    def open(self, dataset: str) -> h5py.File:
        if dataset not in self.handles:
            path = self.data_dir / f"{dataset}.h5"
            if not path.is_file():
                raise FileNotFoundError(path)
            handle = h5py.File(path, "r")
            self.handles[dataset] = handle
            self.conds[dataset] = {str(c): i for i, c in enumerate(handle["conditions"].asstr()[:].tolist())}
        return self.handles[dataset]

    def sample_pair(self, *, seed: int, dataset: str, condition: str, max_cells: int) -> tuple[np.ndarray, np.ndarray]:
        handle = self.open(dataset)
        cond_idx = self.conds[dataset][condition]
        ctrl_offsets = handle["ctrl/offsets"][:] if "ctrl/offsets" in handle else handle["ir/offsets"][:]
        ctrl_key = "ctrl" if "ctrl/emb" in handle else "ir"
        gt_offsets = handle["gt/offsets"][:]
        n_src = int(ctrl_offsets[cond_idx + 1] - ctrl_offsets[cond_idx])
        n_gt = int(gt_offsets[cond_idx + 1] - gt_offsets[cond_idx])
        n_src_eval = min(n_src, int(max_cells))
        n_gt_eval = min(n_gt, int(max_cells))
        rng = np.random.RandomState(eval_rank(seed, "mmd_cells", dataset, condition) % (2**32 - 1))
        src_rel = rng.permutation(n_src)[:n_src_eval]
        gt_rel = rng.permutation(n_gt)[:n_gt_eval]
        src = read_rows(handle[f"{ctrl_key}/emb"], int(ctrl_offsets[cond_idx]), src_rel)
        gt = read_rows(handle["gt/emb"], int(gt_offsets[cond_idx]), gt_rel)
        return src, gt

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                {
                    "seed": str(row["seed"]),
                    "explicit_group": str(row["explicit_group"]),
                    "dataset": str(row["dataset"]),
                    "condition": str(row["condition"]),
                    "anchor_mmd_clamped": float(row["test_mmd_clamped"]),
                    "anchor_pearson_pert": float(row["pearson_pert"]),
                    "ctrl_pearson_pert": float(row["pearson_ctrl"]),
                    "n_src_eval_reported": int(float(row["n_src_eval"])),
                    "n_gt_eval_reported": int(float(row["n_gt_eval"])),
                }
            )
    return rows


def seed_int(seed_name: str) -> int:
    return int(str(seed_name).replace("seed", ""))


def compute_ctrl_mmd(src: np.ndarray, gt: np.ndarray, device: torch.device) -> dict[str, float]:
    x = torch.from_numpy(src).float().to(device)
    y = torch.from_numpy(gt).float().to(device)
    sigmas, dyy = median_sigmas(y, return_D2=True)
    mmd = float(mmd2_unbiased(x, y, sigmas, Dyy=dyy).detach().cpu().item())
    mmd_biased = float(mmd2_biased(x, y, sigmas, Dyy=dyy).detach().cpu().item())
    return {
        "ctrl_mmd": mmd,
        "ctrl_mmd_biased": mmd_biased,
        "ctrl_mmd_clamped": max(mmd, 0.0),
        "n_src_eval": int(src.shape[0]),
        "n_gt_eval": int(gt.shape[0]),
    }


def bootstrap(vals: list[float], *, seed: int = 20260628) -> dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return {"ci_low": 0.0, "ci_high": 0.0, "p_le0": 0.0, "p_gt0": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(5000, arr.size))
    means = arr[idx].mean(axis=1)
    return {
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_le0": float(np.mean(means <= 0.0)),
        "p_gt0": float(np.mean(means > 0.0)),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for seed in sorted({str(r["seed"]) for r in rows}):
        out[seed] = {}
        for group in GROUP_ORDER:
            part = [r for r in rows if str(r["seed"]) == seed and str(r["explicit_group"]) == group]
            if not part:
                continue
            by_ds: dict[str, list[float]] = defaultdict(list)
            for row in part:
                by_ds[str(row["dataset"])].append(float(row["mmd_delta"]))
            ds_means = [float(np.mean(v)) for v in by_ds.values()]
            out[seed][group] = {
                "n": len(part),
                "n_datasets": len(by_ds),
                "anchor_mmd_clamped": float(np.mean([r["anchor_mmd_clamped"] for r in part])),
                "ctrl_mmd_clamped": float(np.mean([r["ctrl_mmd_clamped"] for r in part])),
                "mmd_delta": float(np.mean(ds_means)) if ds_means else 0.0,
                "dataset_min_delta": float(min(ds_means)) if ds_means else 0.0,
                "dataset_max_delta": float(max(ds_means)) if ds_means else 0.0,
                "bootstrap_dataset_delta": bootstrap(ds_means, seed=20260628 + seed_int(seed)),
                "datasets_worse_gt_0p01": int(sum(v > 0.01 for v in ds_means)),
                "per_dataset_delta": {ds: float(np.mean(vals)) for ds, vals in sorted(by_ds.items())},
            }
    return out


def decision(summary: dict[str, Any], *, cap: int) -> tuple[str, list[str]]:
    reasons: list[str] = []
    target_groups = ("all_test_single_proxy", "cross_background_seen_gene_proxy", "family_gene")
    for seed, groups in summary.items():
        for group in target_groups:
            if group not in groups:
                reasons.append(f"{seed}_{group}_missing")
                continue
            s = groups[group]
            if s["mmd_delta"] > 0.0:
                reasons.append(f"{seed}_{group}_ctrl_mmd_delta_positive")
            if s["bootstrap_dataset_delta"]["ci_high"] > 0.003:
                reasons.append(f"{seed}_{group}_ctrl_mmd_ci_high_gt_0p003")
            if s["dataset_max_delta"] > 0.01:
                reasons.append(f"{seed}_{group}_dataset_harm_gt_0p01")
    if cap < 2048:
        reasons.append(f"pilot_cap_{cap}_not_exact_2048")
    status = "tracka_ctrl_mmd_gate_fail_or_pilot_only"
    severe = [r for r in reasons if not r.startswith("pilot_cap_")]
    if not severe and cap >= 2048:
        status = "tracka_ctrl_mmd_gate_diagnostic_pass_no_model_route"
    elif not severe:
        status = "tracka_ctrl_mmd_gate_pilot_nonharm_needs_2048"
    return status, reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--eval-max-mmd-cells", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.threads)))
    device = torch.device(args.device)
    rows = load_rows(args.rows)
    unique = sorted({(r["seed"], r["dataset"], r["condition"]) for r in rows})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cache = H5Cache(args.data_dir)
    computed: dict[tuple[str, str, str], dict[str, float]] = {}
    try:
        for i, (seed_name, ds, cond) in enumerate(unique, start=1):
            src, gt = cache.sample_pair(
                seed=seed_int(seed_name),
                dataset=ds,
                condition=cond,
                max_cells=int(args.eval_max_mmd_cells),
            )
            computed[(seed_name, ds, cond)] = compute_ctrl_mmd(src, gt, device)
            if args.progress_every > 0 and (i == 1 or i % args.progress_every == 0 or i == len(unique)):
                elapsed = time.time() - started
                print(
                    f"[ctrl-mmd] {i}/{len(unique)} cap={args.eval_max_mmd_cells} elapsed={elapsed:.1f}s",
                    flush=True,
                )
    finally:
        cache.close()

    enriched = []
    for row in rows:
        mmd = computed[(row["seed"], row["dataset"], row["condition"])]
        out = dict(row)
        out.update(mmd)
        out["mmd_delta"] = float(out["ctrl_mmd_clamped"] - out["anchor_mmd_clamped"])
        enriched.append(out)

    summary = summarize(enriched)
    status, reasons = decision(summary, cap=int(args.eval_max_mmd_cells))
    tag = f"cap{int(args.eval_max_mmd_cells)}"
    rows_csv = args.out_dir / f"tracka_ctrl_mmd_rows_{tag}.csv"
    json_path = args.out_dir / f"latentfm_tracka_ctrl_mmd_gate_{tag}.json"
    md_path = args.out_dir / f"LATENTFM_TRACKA_CTRL_MMD_GATE_{tag}.md"

    fieldnames = [
        "seed",
        "explicit_group",
        "dataset",
        "condition",
        "anchor_mmd_clamped",
        "ctrl_mmd_clamped",
        "mmd_delta",
        "anchor_pearson_pert",
        "ctrl_pearson_pert",
        "n_src_eval",
        "n_gt_eval",
    ]
    with rows_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in enriched:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "eval_only_no_model": True,
            "explicit_rows_frozen": str(args.rows),
            "canonical_multi_selection_used": False,
            "trackc_query_used": False,
            "selection_weight": 0,
            "cell_sampling": "eval_select:<seed>:mmd_cells:<dataset>:<condition>",
            "eval_max_mmd_cells": int(args.eval_max_mmd_cells),
            "device": str(device),
            "threads": int(args.threads),
        },
        "runtime_seconds": time.time() - started,
        "n_unique_seed_dataset_condition": len(unique),
        "n_rows": len(enriched),
        "summary": summary,
        "decision_reasons": reasons,
        "outputs": {"rows_csv": str(rows_csv), "json": str(json_path), "md": str(md_path)},
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Track A Control Baseline MMD Gate",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        f"Eval-only control/source-vs-GT MMD diagnostic over frozen explicit Track A proxy rows. No model checkpoint, training, threshold selection, canonical multi, or Track C query is used. Cell cap: `{int(args.eval_max_mmd_cells)}`.",
        "",
        "## Results",
        "",
        "| seed | group | n | datasets | anchor MMD | ctrl MMD | ctrl-anchor | 95% CI | dataset max | datasets harm >0.01 |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for seed, groups in sorted(summary.items()):
        for group in GROUP_ORDER:
            if group not in groups:
                continue
            s = groups[group]
            boot = s["bootstrap_dataset_delta"]
            lines.append(
                f"| `{seed}` | `{group}` | {s['n']} | {s['n_datasets']} | "
                f"{s['anchor_mmd_clamped']:+.6f} | {s['ctrl_mmd_clamped']:+.6f} | {s['mmd_delta']:+.6f} | "
                f"[{boot['ci_low']:+.6f},{boot['ci_high']:+.6f}] | {s['dataset_max_delta']:+.6f} | "
                f"{s['datasets_worse_gt_0p01']} |"
            )
    lines.extend(["", "## Decision Reasons", ""])
    lines.extend(f"- `{reason}`" for reason in reasons)
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This is a metric/baseline diagnostic, not a model route. If the control endpoint passes MMD non-harm, it should be reported as a baseline pathology/control before making strong Track A perturbation claims; it does not authorize training a model to predict control.",
            "",
            "## Outputs",
            "",
            f"- Rows: `{rows_csv}`",
            f"- JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "md": str(md_path), "json": str(json_path), "rows": str(rows_csv)}, indent=2))


if __name__ == "__main__":
    main()
