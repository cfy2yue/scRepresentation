#!/usr/bin/env python3
"""Condition-overlap OT pairing-quality reliability gate.

This CPU-only audit fixes a limitation of the earlier dataset-level OT gate:
the random pairing audit batches did not overlap the internal validation
conditions. Here we compute OT pairing-quality features directly on the
train-only internal-validation condition rows and test whether those features
predict response/no-harm reliability.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/data/cyx/1030/scLatent")
COUPLED = ROOT / "CoupledFM"
sys.path.insert(0, str(COUPLED))

from model.latent.dataset import _DatasetHandle  # noqa: E402
from model.utils.data.ot_pairer import sinkhorn_pair  # noqa: E402

ANCHOR_JSON = ROOT / "reports/latentfm_xverse_tracka_anchor_internal_val_error_map_20260622.json"
DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
OUT_JSON = ROOT / "reports/latentfm_ot_condition_overlap_reliability_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_OT_CONDITION_OVERLAP_RELIABILITY_GATE_20260624.md"


def _f(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _sq_cost(src: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return (src.float() - gt.float()).pow(2).sum(dim=1)


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 3:
        return float("nan")
    mx = statistics.fmean(x)
    my = statistics.fmean(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)


def _spearman(x: list[float], y: list[float]) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _perm_p_abs(x: list[float], y: list[float], *, seed: int, n_perm: int) -> float:
    obs = abs(_spearman(x, y))
    if not math.isfinite(obs):
        return float("nan")
    rng = random.Random(seed)
    yp = list(y)
    count = 1
    for _ in range(int(n_perm)):
        rng.shuffle(yp)
        r = _spearman(x, yp)
        if math.isfinite(r) and abs(r) >= obs:
            count += 1
    return count / (int(n_perm) + 1)


def _sample_pairing_features(
    handle: _DatasetHandle,
    cond: str,
    *,
    batch_size: int,
    seed: int,
    row_id: int,
    reg: float,
    n_iter: int,
) -> dict[str, float]:
    n_src, n_gt = handle.cond_sizes(cond)
    if n_src <= 0 or n_gt <= 0:
        raise ValueError(f"empty condition {cond}")
    rng = np.random.RandomState(int(seed) + int(row_id) * 7919)
    src_idx = rng.choice(n_src, size=int(batch_size), replace=(n_src < int(batch_size)))
    gt_idx = rng.choice(n_gt, size=int(batch_size), replace=(n_gt < int(batch_size)))
    src = torch.from_numpy(handle.read_src_rows(cond, src_idx).astype(np.float32))
    gt = torch.from_numpy(handle.read_gt_rows(cond, gt_idx).astype(np.float32))
    gen = torch.Generator(device=src.device)
    gen.manual_seed(int(seed) + int(row_id) * 1009)
    im, jm = sinkhorn_pair(src, gt, int(batch_size), reg=float(reg), n_iter=int(n_iter), generator=gen, use_assignment=False)
    ia, ja = sinkhorn_pair(src, gt, int(batch_size), reg=float(reg), n_iter=int(n_iter), use_assignment=True)

    identity_cost = float(_sq_cost(src, gt).mean().item())
    multinomial_cost = float(_sq_cost(src[im], gt[jm]).mean().item())
    assignment_cost = float(_sq_cost(src[ia], gt[ja]).mean().item())
    raw_delta = gt.float().mean(dim=0) - src.float().mean(dim=0)
    denom = max(float(raw_delta.norm().item()), 1e-8)
    multinomial_delta = gt[jm].float().mean(dim=0) - src[im].float().mean(dim=0)
    assignment_delta = gt[ja].float().mean(dim=0) - src[ia].float().mean(dim=0)
    return {
        "identity_cost": identity_cost,
        "ot_cost_gain_multinomial": -((multinomial_cost - identity_cost) / max(identity_cost, 1e-8)),
        "ot_cost_gain_assignment": -((assignment_cost - identity_cost) / max(identity_cost, 1e-8)),
        "ot_unique_gt_multinomial": float(torch.unique(jm).numel() / int(batch_size)),
        "ot_unique_src_multinomial": float(torch.unique(im).numel() / int(batch_size)),
        "ot_delta_noise_multinomial": float((multinomial_delta - raw_delta).norm().item() / denom),
        "ot_delta_noise_assignment": float((assignment_delta - raw_delta).norm().item() / denom),
    }


def _load_condition_rows(path: Path, max_conditions: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    seen: set[tuple[str, str, str]] = set()
    allowed_groups = {
        "internal_val_cross_background_seen_gene_proxy",
        "internal_val_family_gene_proxy",
    }
    for row in payload.get("condition_rows", []):
        group = str(row.get("group", ""))
        ds = str(row.get("dataset", ""))
        cond = str(row.get("condition", ""))
        key = (group, ds, cond)
        if group not in allowed_groups or not ds or not cond or key in seen:
            continue
        seen.add(key)
        rows.append(row)
    if max_conditions > 0:
        rows = rows[: int(max_conditions)]
    return rows


def _compute_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = _load_condition_rows(Path(args.anchor_json), int(args.max_conditions))
    handles: dict[str, _DatasetHandle] = {}
    out: list[dict[str, Any]] = []
    feature_cache: dict[tuple[str, str], dict[str, float]] = {}
    try:
        for row in rows:
            ds = str(row["dataset"])
            cond = str(row["condition"])
            h5_path = Path(args.data_dir) / f"{ds}.h5"
            if not h5_path.is_file():
                continue
            if ds not in handles:
                handles[ds] = _DatasetHandle(str(h5_path))
            key = (ds, cond)
            if key not in feature_cache:
                feature_cache[key] = _sample_pairing_features(
                    handles[ds],
                    cond,
                    batch_size=int(args.batch_size),
                    seed=int(args.seed),
                    row_id=len(feature_cache),
                    reg=float(args.reg),
                    n_iter=int(args.n_iter),
                )
            item: dict[str, Any] = {
                "group": str(row["group"]),
                "dataset": ds,
                "condition": cond,
                **feature_cache[key],
                "anchor_pearson_pert": _f(row.get("anchor_pearson_pert")),
                "anchor_minus_gene_raw_mean": _f(
                    (_f(row.get("anchor_pearson_pert")) or 0.0) - (_f(row.get("gene_raw_mean")) or 0.0)
                ),
                "anchor_mmd_clamped": _f(row.get("anchor_mmd_clamped")),
            }
            if all(item.get(k) is not None for k in ("anchor_pearson_pert", "anchor_minus_gene_raw_mean", "anchor_mmd_clamped")):
                out.append(item)
    finally:
        for handle in handles.values():
            handle.close()
    return out


def _correlation_rows(rows: list[dict[str, Any]], *, n_perm: int, seed: int) -> list[dict[str, Any]]:
    features = {
        "ot_cost_gain_multinomial": {"target_sign": 1, "control": False},
        "ot_cost_gain_assignment": {"target_sign": 1, "control": False},
        "ot_unique_gt_multinomial": {"target_sign": 1, "control": False},
        "ot_delta_noise_multinomial": {"target_sign": -1, "control": False},
        "ot_delta_noise_assignment": {"target_sign": -1, "control": False},
    }
    targets = {
        "anchor_pearson_pert": 1,
        "anchor_minus_gene_raw_mean": 1,
        "anchor_mmd_clamped": -1,
    }
    out: list[dict[str, Any]] = []
    groups = sorted({str(r["group"]) for r in rows})
    groups.append("pooled")
    for group in groups:
        sub = rows if group == "pooled" else [r for r in rows if r["group"] == group]
        if len(sub) < 12:
            continue
        for f_idx, (feature, finfo) in enumerate(features.items()):
            x = [float(r[feature]) for r in sub]
            for t_idx, (target, target_sign) in enumerate(targets.items()):
                y = [float(r[target]) for r in sub]
                rho = _spearman(x, y)
                expected_sign = int(finfo["target_sign"]) * int(target_sign)
                p_abs = _perm_p_abs(x, y, seed=int(seed) + 1000 * f_idx + 17 * t_idx + len(group), n_perm=int(n_perm))
                direction_ok = math.isfinite(rho) and rho * expected_sign > 0
                out.append(
                    {
                        "group": group,
                        "feature": feature,
                        "target": target,
                        "n": len(sub),
                        "spearman": rho,
                        "perm_p_abs": p_abs,
                        "expected_sign": expected_sign,
                        "direction_ok": direction_ok,
                        "material": bool(direction_ok and abs(rho) >= 0.25 and p_abs <= 0.10),
                        "contradictory": bool((not direction_ok) and math.isfinite(rho) and abs(rho) >= 0.25 and p_abs <= 0.10),
                    }
                )
    return out


def _decide(rows: list[dict[str, Any]], corr: list[dict[str, Any]]) -> dict[str, Any]:
    material = [r for r in corr if r["material"]]
    contradictions = [r for r in corr if r["contradictory"]]
    material_groups = sorted({str(r["group"]) for r in material})
    reasons: list[str] = []
    if len(rows) < 100:
        reasons.append("condition_overlap_below_100")
    if len(material) < 3:
        reasons.append("fewer_than_three_material_expected_direction_correlations")
    if "pooled" not in material_groups:
        reasons.append("no_pooled_material_expected_direction_correlation")
    if contradictions:
        reasons.append("material_contradictory_correlations_present")
    status = "ot_condition_overlap_reliability_gate_pass_design_one_default_off_smoke"
    if reasons:
        status = "ot_condition_overlap_reliability_gate_fail_no_gpu"
    return {
        "status": status,
        "gpu_authorized": False,
        "n_condition_rows": len(rows),
        "n_unique_conditions": len({(r["dataset"], r["condition"]) for r in rows}),
        "material_expected_correlations": len(material),
        "material_groups": material_groups,
        "material_contradictions": len(contradictions),
        "reasons": reasons,
        "next_action": (
            "If this is accepted as pass, design exactly one default-off OT smoke; do not run broad OT sweeps."
            if not reasons
            else "Keep OT closed; do not run OT GPU without a new non-contradictory condition-level gate."
        ),
    }


def _render_md(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# LatentFM OT Condition-Overlap Reliability Gate",
        "",
        f"Status: `{decision['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only audit over train-only internal-validation condition rows.",
        "- Reads LatentFM HDF5 embeddings for the same `(dataset, condition)` rows used by the internal-val error map.",
        "- No model training, checkpoint selection, canonical/test/query reads, or GPU use.",
        "",
        "## Decision",
        "",
        f"- condition rows: `{decision['n_condition_rows']}`",
        f"- unique conditions: `{decision['n_unique_conditions']}`",
        f"- material expected-direction correlations: `{decision['material_expected_correlations']}`",
        f"- material groups: `{decision['material_groups']}`",
        f"- material contradictory correlations: `{decision['material_contradictions']}`",
        f"- GPU authorized: `{decision['gpu_authorized']}`",
        "",
        "Reasons:",
    ]
    if decision["reasons"]:
        lines.extend([f"- `{r}`" for r in decision["reasons"]])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Correlations",
            "",
            "| group | feature | target | n | spearman | perm p | expected sign | material | contradictory |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["correlations"]:
        lines.append(
            "| {group} | {feature} | {target} | {n} | {spearman:+.4f} | {perm_p_abs:.4f} | {expected_sign:+d} | {material} | {contradictory} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Next Action",
            "",
            decision["next_action"],
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-json", default=str(ANCHOR_JSON))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-conditions", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reg", type=float, default=0.05)
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-md", default=str(OUT_MD))
    args = parser.parse_args()

    rows = _compute_rows(args)
    corr = _correlation_rows(rows, n_perm=int(args.n_perm), seed=int(args.seed))
    payload = {
        "boundary": {
            "anchor_internal_val_json": str(args.anchor_json),
            "data_dir": str(args.data_dir),
            "batch_size": int(args.batch_size),
            "max_conditions": int(args.max_conditions),
            "seed": int(args.seed),
            "reg": float(args.reg),
            "n_iter": int(args.n_iter),
            "n_perm": int(args.n_perm),
            "no_training": True,
            "no_canonical_or_query": True,
            "no_gpu": True,
        },
        "decision": _decide(rows, corr),
        "condition_rows": rows,
        "correlations": corr,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.out_md).write_text(_render_md(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
