#!/usr/bin/env python3
"""CPU-only OT pair-quality audit for LatentFM train conditions."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


ROOT = Path("/data/cyx/1030/scLatent")
COUPLED = ROOT / "CoupledFM"
DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
MANIFEST = DATA_DIR / "manifest.json"
SPLIT_FILE = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
TAIL_SENTINEL = ROOT / "reports/latentfm_scaling_provenance_tail_sentinel_gate_20260624.json"
BUDGET64_DECISION = ROOT / "reports/latentfm_true_cell_count_budget64_tail_stability_6k_decision_20260625.json"
OUT_JSON = ROOT / "reports/latentfm_ot_pair_quality_gate_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_OT_PAIR_QUALITY_GATE_20260625.md"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def risk_datasets() -> list[str]:
    out: set[str] = set()
    if BUDGET64_DECISION.exists():
        data = load_json(BUDGET64_DECISION)
        for budget_row in (data.get("matrix_summary") or {}).get("budget_rows", []):
            tail = budget_row.get("cross_background_pp_dataset_tail") or {}
            for row in tail.get("dataset_rows", []):
                if float(row.get("mean", 0.0)) < -0.020:
                    out.add(str(row["dataset"]))
    if TAIL_SENTINEL.exists():
        data = load_json(TAIL_SENTINEL)
        for sim in data.get("simulations", []):
            if not sim.get("pass_gate", False):
                out.update(str(x) for x in sim.get("sentinel_datasets", []))
    return sorted(out)


def choose_conditions(manifest: dict[str, Any], split: dict[str, Any], *, max_datasets: int = 8, per_dataset: int = 3) -> list[tuple[str, str]]:
    risks = [ds for ds in risk_datasets() if ds in split]
    fallback = [ds for ds in sorted(split) if ds not in risks]
    datasets = (risks + fallback)[:max_datasets]
    selected: list[tuple[str, str]] = []
    for ds in datasets:
        train = sorted(str(c) for c in split[ds].get("train", []))
        if not train:
            continue
        h5_path = Path(manifest["datasets"][ds]["out_path"])
        with h5py.File(h5_path, "r") as handle:
            conds = [decode(x) for x in handle["conditions"][()]]
            gt_offsets = [int(x) for x in handle["gt/offsets"][()]]
            ctrl_offsets = [int(x) for x in handle["ctrl/offsets"][()]]
        sizes = []
        for idx, cond in enumerate(conds):
            if cond in train:
                n_gt = gt_offsets[idx + 1] - gt_offsets[idx]
                n_ctrl = ctrl_offsets[idx + 1] - ctrl_offsets[idx]
                if n_gt >= 16 and n_ctrl >= 16:
                    sizes.append((min(n_gt, n_ctrl), cond))
        # Large conditions stress pair quality most; deterministic top-k keeps
        # provenance simple and makes the audit stable.
        for _, cond in sorted(sizes, reverse=True)[:per_dataset]:
            selected.append((ds, cond))
    return selected


def read_batch(manifest: dict[str, Any], ds: str, cond: str, *, seed: int, batch_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    h5_path = Path(manifest["datasets"][ds]["out_path"])
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, "r") as handle:
        conds = [decode(x) for x in handle["conditions"][()]]
        idx = conds.index(cond)
        gt_offsets = [int(x) for x in handle["gt/offsets"][()]]
        ctrl_offsets = [int(x) for x in handle["ctrl/offsets"][()]]
        gt_start, gt_end = gt_offsets[idx], gt_offsets[idx + 1]
        ctrl_start, ctrl_end = ctrl_offsets[idx], ctrl_offsets[idx + 1]
        n_gt = gt_end - gt_start
        n_ctrl = ctrl_end - ctrl_start
        n = min(batch_size, n_gt, n_ctrl)
        gt_idx = np.sort(rng.choice(n_gt, size=n, replace=False) + gt_start)
        ctrl_idx = np.sort(rng.choice(n_ctrl, size=n, replace=False) + ctrl_start)
        gt = np.asarray(handle["gt/emb"][gt_idx], dtype=np.float32)
        ctrl = np.asarray(handle["ctrl/emb"][ctrl_idx], dtype=np.float32)
    return torch.from_numpy(ctrl), torch.from_numpy(gt)


def dup_rate(idx: torch.Tensor) -> float:
    n = int(idx.numel())
    if n == 0:
        return math.nan
    return 1.0 - (int(torch.unique(idx).numel()) / float(n))


def summarize_pairing(src: torch.Tensor, gt: torch.Tensor, *, seed: int) -> dict[str, Any]:
    sys.path.insert(0, str(COUPLED))
    from model.utils.data.ot_pairer import compute_ot_cost, hungarian_pair, sinkhorn_pair

    torch.manual_seed(seed)
    n = int(src.shape[0])
    cost = compute_ot_cost(src.float(), gt.float())
    baseline_diag = cost[torch.arange(n), torch.arange(n)]
    out: dict[str, Any] = {
        "n": n,
        "full_cost_mean": float(cost.mean().item()),
        "full_cost_median": float(cost.median().item()),
        "random_index_cost_mean": float(baseline_diag.mean().item()),
        "modes": {},
    }
    for mode in ("multinomial", "assignment", "hungarian"):
        if mode == "hungarian":
            i, j = hungarian_pair(src.float(), gt.float(), n_samples=n)
        else:
            i, j = sinkhorn_pair(
                src.float(),
                gt.float(),
                n_samples=n,
                reg=0.05,
                n_iter=30,
                use_assignment=(mode == "assignment"),
            )
        paired = cost[i, j]
        out["modes"][mode] = {
            "paired_cost_mean": float(paired.mean().item()),
            "paired_cost_median": float(paired.median().item()),
            "src_duplicate_rate": dup_rate(i.cpu()),
            "gt_duplicate_rate": dup_rate(j.cpu()),
            "cost_vs_random_index_delta": float(paired.mean().item() - baseline_diag.mean().item()),
        }
    return out


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM OT Pair Quality Gate",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only deterministic pair-quality audit on train-only conditions.",
        "- Does not train, infer, read canonical multi, read held-out Track C query, or use GPU.",
        "- A pass here does not authorize GPU by itself; it only identifies whether a specific OT failure mode deserves a future launcher.",
        "",
        "## Aggregate",
        "",
        "| mode | mean paired cost | mean delta vs random index | src dup | gt dup |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, row in payload["aggregate"]["modes"].items():
        lines.append(
            f"| `{mode}` | {row['paired_cost_mean']:.6f} | {row['cost_vs_random_index_delta']:+.6f} | {row['src_duplicate_rate']:.4f} | {row['gt_duplicate_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- GPU authorized: `{payload['gpu_authorized']}`",
            f"- reasons: `{payload['reasons']}`",
            f"- next action: `{payload['next_action']}`",
            "",
            "## Sampled Conditions",
            "",
            "| dataset | condition |",
            "|---|---|",
        ]
    )
    for ds, cond in payload["sampled_conditions"]:
        lines.append(f"| `{ds}` | `{cond}` |")
    lines.extend(["", "## JSON", "", f"`{OUT_JSON}`"])
    return "\n".join(lines) + "\n"


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    modes: dict[str, dict[str, float]] = {}
    for mode in ("multinomial", "assignment", "hungarian"):
        vals = [r["pair_quality"]["modes"][mode] for r in rows]
        modes[mode] = {
            "paired_cost_mean": float(np.mean([v["paired_cost_mean"] for v in vals])),
            "cost_vs_random_index_delta": float(np.mean([v["cost_vs_random_index_delta"] for v in vals])),
            "src_duplicate_rate": float(np.mean([v["src_duplicate_rate"] for v in vals])),
            "gt_duplicate_rate": float(np.mean([v["gt_duplicate_rate"] for v in vals])),
        }
    return {"modes": modes}


def main() -> int:
    manifest = load_json(MANIFEST)
    split = load_json(SPLIT_FILE)
    sampled = choose_conditions(manifest, split)
    rows = []
    for idx, (ds, cond) in enumerate(sampled):
        src, gt = read_batch(manifest, ds, cond, seed=20260625 + idx)
        rows.append(
            {
                "dataset": ds,
                "condition": cond,
                "pair_quality": summarize_pairing(src, gt, seed=20260625 + idx),
            }
        )
    agg = aggregate(rows)
    reasons: list[str] = []
    assignment_gain = -agg["modes"]["assignment"]["cost_vs_random_index_delta"]
    multinomial_dup = agg["modes"]["multinomial"]["gt_duplicate_rate"]
    if assignment_gain < 0.05:
        reasons.append("assignment_cost_gain_too_small_for_new_gpu_branch")
    if multinomial_dup < 0.05:
        reasons.append("multinomial_duplicate_rate_not_high_enough_to_explain_failures")
    status = "ot_pair_quality_no_gpu_without_failure_correlation"
    next_action = "do not launch OT GPU; pair quality alone does not overcome existing negative OT smoke evidence"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "sampled_conditions": sampled,
        "aggregate": agg,
        "rows": rows,
        "reasons": reasons,
        "next_action": next_action,
        "boundary": {
            "canonical_multi_selection": False,
            "trackc_query_read": False,
            "gpu_used": False,
            "train_selection": "train_only",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "out_json": str(OUT_JSON), "out_md": str(OUT_MD)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
