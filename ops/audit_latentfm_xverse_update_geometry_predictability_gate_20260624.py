#!/usr/bin/env python3
"""CPU gate: can deployable update geometry recover oracle-safe updates?"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
MEAN_DIR = ROOT / "reports/latentfm_xverse_nuisance_condition_means_20260624"
ANCHOR_SPLIT = MEAN_DIR / "split_group_eval_anchor_internal_means_ode20.json"
CAP120_SPLIT = MEAN_DIR / "split_group_eval_cap120_internal_means_ode20.json"
OUT_JSON = ROOT / "reports/latentfm_xverse_update_geometry_predictability_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_XVERSE_UPDATE_GEOMETRY_PREDICTABILITY_GATE_20260624.md"
GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
SEED = 42
BOOT_N = 1000
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
FEATURE_CACHE: dict[int, dict[str, float]] = {}
METRIC_CACHE: dict[tuple[int, float], tuple[float, float]] = {}


@dataclass(frozen=True)
class Row:
    group: str
    dataset: str
    condition: str
    anchor_pred: np.ndarray
    cap_pred: np.ndarray
    gt: np.ndarray
    ctrl: np.ndarray
    pert: np.ndarray


@dataclass(frozen=True)
class Stump:
    feature: str
    op: str
    threshold: float
    alpha_true: float
    alpha_false: float

    @property
    def name(self) -> str:
        return f"{self.feature}_{self.op}_{self.threshold:.6g}_a{self.alpha_true:.2f}_{self.alpha_false:.2f}"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def arr(row: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(row[key], dtype=np.float64)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = a - float(np.mean(a))
    bb = b - float(np.mean(b))
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(aa, bb) / den)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / den)


def pp(pred: np.ndarray, row: Row) -> float:
    return corr(pred - row.ctrl, row.pert - row.ctrl)


def residual(pred: np.ndarray, row: Row) -> float:
    return float(np.linalg.norm(pred - row.gt) / math.sqrt(pred.size))


def update(row: Row) -> np.ndarray:
    return row.cap_pred - row.anchor_pred


def load_rows() -> list[Row]:
    anchor = load_json(ANCHOR_SPLIT)
    cap120 = load_json(CAP120_SPLIT)
    rows: list[Row] = []
    for group in GROUPS:
        a_rows = {
            (str(r["dataset"]), str(r["condition"])): r
            for r in anchor["groups"][group]["condition_metrics"]
        }
        c_rows = {
            (str(r["dataset"]), str(r["condition"])): r
            for r in cap120["groups"][group]["condition_metrics"]
        }
        for key in sorted(set(a_rows) & set(c_rows)):
            a = a_rows[key]
            c = c_rows[key]
            rows.append(
                Row(
                    group=group,
                    dataset=key[0],
                    condition=key[1],
                    anchor_pred=arr(a, "pred_mean"),
                    cap_pred=arr(c, "pred_mean"),
                    gt=arr(a, "gt_mean"),
                    ctrl=arr(a, "ctrl_mean"),
                    pert=arr(a, "pert_mean"),
                )
            )
    return rows


def features(row: Row) -> dict[str, float]:
    cached = FEATURE_CACHE.get(id(row))
    if cached is not None:
        return cached
    u = update(row)
    a_delta = row.anchor_pred - row.ctrl
    c_delta = row.cap_pred - row.ctrl
    out = {
        "update_norm": float(np.linalg.norm(u)),
        "anchor_delta_norm": float(np.linalg.norm(a_delta)),
        "cap_delta_norm": float(np.linalg.norm(c_delta)),
        "delta_norm_ratio": float(np.linalg.norm(c_delta) / max(np.linalg.norm(a_delta), 1e-12)),
        "update_cos_anchor_delta": cosine(u, a_delta),
        "update_cos_cap_delta": cosine(u, c_delta),
        "anchor_cap_delta_cos": cosine(a_delta, c_delta),
    }
    FEATURE_CACHE[id(row)] = out
    return out


def precompute(rows: list[Row]) -> None:
    for row in rows:
        features(row)
        for alpha in ALPHAS:
            pred = row.anchor_pred + alpha * update(row)
            METRIC_CACHE[(id(row), alpha)] = (
                pp(pred, row) - pp(row.anchor_pred, row),
                residual(pred, row) - residual(row.anchor_pred, row),
            )


def cached_delta(row: Row, alpha: float) -> tuple[float, float]:
    key_alpha = min(ALPHAS, key=lambda a: abs(a - alpha))
    return METRIC_CACHE[(id(row), key_alpha)]


def alpha_for(row: Row, stump: Stump) -> float:
    val = features(row)[stump.feature]
    hit = val <= stump.threshold if stump.op == "<=" else val >= stump.threshold
    return stump.alpha_true if hit else stump.alpha_false


def eval_rows(rows: list[Row], stump: Stump) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        alpha = alpha_for(row, stump)
        pp_delta, resid_delta = cached_delta(row, alpha)
        out.append(
            {
                "group": row.group,
                "dataset": row.dataset,
                "condition": row.condition,
                "pp_delta_vs_anchor": pp_delta,
                "residual_delta_vs_anchor": resid_delta,
                "alpha": alpha,
            }
        )
    return out


def summarize_eval(rows: list[dict[str, Any]], *, with_bootstrap: bool = True) -> dict[str, Any]:
    groups = []
    for group in GROUPS:
        gr = [r for r in rows if r["group"] == group]
        deltas = np.asarray([r["pp_delta_vs_anchor"] for r in gr], dtype=np.float64)
        residuals = np.asarray([r["residual_delta_vs_anchor"] for r in gr], dtype=np.float64)
        by_ds: dict[str, list[float]] = {}
        for r in gr:
            by_ds.setdefault(str(r["dataset"]), []).append(float(r["pp_delta_vs_anchor"]))
        groups.append(
            {
                "group": group,
                "n": len(gr),
                "mean_pp_delta_vs_anchor": float(np.nanmean(deltas)),
                "p_harm_condition": float(np.mean(deltas < 0.0)),
                "bootstrap_p_harm_mean_delta": bootstrap_p_harm(deltas) if with_bootstrap else float("nan"),
                "dataset_min_pp_delta_vs_anchor": float(min(np.nanmean(v) for v in by_ds.values())),
                "mean_residual_delta_vs_anchor": float(np.nanmean(residuals)),
                "mean_alpha": float(np.nanmean([r["alpha"] for r in gr])),
            }
        )
    return {"groups": groups}


def bootstrap_p_harm(vals: np.ndarray) -> float:
    rng = random.Random(SEED)
    values = [float(v) for v in vals]
    harms = 0
    for _ in range(BOOT_N):
        sample = [values[rng.randrange(len(values))] for _ in values]
        if float(np.nanmean(sample)) < 0.0:
            harms += 1
    return harms / BOOT_N


def score(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    groups = summary["groups"]
    return (
        min(g["mean_pp_delta_vs_anchor"] for g in groups),
        -max(g["p_harm_condition"] for g in groups),
        min(g["dataset_min_pp_delta_vs_anchor"] for g in groups),
        -max(g["mean_residual_delta_vs_anchor"] for g in groups),
    )


def make_stumps(train_rows: list[Row]) -> list[Stump]:
    feats = [features(r) for r in train_rows]
    stumps: list[Stump] = []
    for feature in feats[0]:
        vals = np.asarray([f[feature] for f in feats if np.isfinite(f[feature])], dtype=np.float64)
        if vals.size < 8:
            continue
        for q in (0.2, 0.4, 0.6, 0.8):
            t = float(np.quantile(vals, q))
            for op in ("<=", ">="):
                for alpha_true, alpha_false in ((1.0, 0.0), (0.75, 0.0), (0.5, 0.0), (1.0, 0.25), (0.25, 1.0)):
                    stumps.append(Stump(feature, op, t, alpha_true, alpha_false))
    stumps.extend(
        [
            Stump("update_norm", ">=", -1.0, 0.0, 0.0),
            Stump("update_norm", ">=", -1.0, 1.0, 1.0),
            Stump("update_norm", ">=", -1.0, 0.5, 0.5),
        ]
    )
    return stumps


def lodo(rows: list[Row]) -> dict[str, Any]:
    chosen = []
    heldout_eval = []
    for ds in sorted({r.dataset for r in rows}):
        train_rows = [r for r in rows if r.dataset != ds]
        test_rows = [r for r in rows if r.dataset == ds]
        candidates = []
        for stump in make_stumps(train_rows):
            summary = summarize_eval(eval_rows(train_rows, stump), with_bootstrap=False)
            candidates.append((score(summary), stump, summary))
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best, best_summary = candidates[0]
        chosen.append(
            {
                "heldout_dataset": ds,
                "policy": best.name,
                "train_score": list(best_score),
                "train_summary": best_summary,
            }
        )
        heldout_eval.extend(eval_rows(test_rows, best))
    return summarize_eval(heldout_eval) | {"chosen": chosen}


def passes_gate(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    for g in summary["groups"]:
        name = g["group"]
        if g["mean_pp_delta_vs_anchor"] < 0.01:
            reasons.append(f"{name}_mean_pp_delta_lt_0p01")
        if g["p_harm_condition"] > 0.20:
            reasons.append(f"{name}_condition_p_harm_gt_0p20")
        if g["bootstrap_p_harm_mean_delta"] > 0.20:
            reasons.append(f"{name}_bootstrap_p_harm_gt_0p20")
        if g["dataset_min_pp_delta_vs_anchor"] < -0.02:
            reasons.append(f"{name}_dataset_min_delta_lt_neg0p02")
        if g["mean_residual_delta_vs_anchor"] > 0.0:
            reasons.append(f"{name}_residual_proxy_worse_than_anchor")
    return not reasons, reasons


def main() -> int:
    rows = load_rows()
    precompute(rows)
    result = lodo(rows)
    passed, reasons = passes_gate(result)
    status = (
        "update_geometry_predictability_gate_pass_code_gate_next_no_gpu"
        if passed
        else "update_geometry_predictability_gate_fail_no_gpu"
    )
    payload = {
        "status": status,
        "gpu_authorization": "none",
        "inputs": {"anchor_split": str(ANCHOR_SPLIT), "cap120_split": str(CAP120_SPLIT)},
        "boundary": {
            "train_internal_only": True,
            "canonical_test_read": False,
            "canonical_multi_read": False,
            "heldout_query_read": False,
            "active_log_read": False,
            "new_gpu_artifact_read": False,
        },
        "gate": {
            "selection": "leave_one_dataset_out",
            "feature_family": "deployable_anchor_candidate_geometry_stumps",
            "mean_pp_delta_vs_anchor_min": 0.01,
            "condition_p_harm_max": 0.20,
            "bootstrap_p_harm_max": 0.20,
            "dataset_min_pp_delta_vs_anchor_min": -0.02,
            "mean_residual_delta_vs_anchor_max": 0.0,
        },
        "decision_reasons": reasons,
        "lodo": result,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM xverse Update-Geometry Predictability Gate",
        "",
        f"Status: `{status}`",
        "GPU authorization: `none`",
        "",
        "## Boundary",
        "",
        "- Reads only train/internal xverse anchor and cap120 condition-mean artifacts.",
        "- Does not read canonical test, canonical multi, held-out Track C query, active logs, or new GPU artifacts.",
        "- Tests whether deployable anchor/candidate geometry features can recover the target-cosine oracle update signal.",
        "",
        "## LODO Result",
        "",
        "| group | n | mean pp delta | p_harm | boot p_harm | dataset min | residual delta | mean alpha |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g in result["groups"]:
        lines.append(
            f"| {g['group']} | {g['n']} | {g['mean_pp_delta_vs_anchor']:+.6f} | "
            f"{g['p_harm_condition']:.3f} | {g['bootstrap_p_harm_mean_delta']:.3f} | "
            f"{g['dataset_min_pp_delta_vs_anchor']:+.6f} | {g['mean_residual_delta_vs_anchor']:+.6f} | "
            f"{g['mean_alpha']:.3f} |"
        )
    lines.extend(["", "## Chosen Policy Counts", ""])
    counts: dict[str, int] = {}
    for item in result["chosen"]:
        counts[item["policy"]] = counts.get(item["policy"], 0) + 1
    for name, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- `{name}`: {n}")
    lines.extend(["", "## Decision Reasons", ""])
    lines.extend([f"- `{r}`" for r in reasons] or ["- none"])
    lines.extend([
        "",
        "## Interpretation",
        "",
        "A pass would authorize only a default-off code/protocol gate, not GPU.",
        "A fail means the target-cosine oracle signal is not recoverable by these deployable geometry stumps.",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
        "",
    ])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorization": "none", "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
