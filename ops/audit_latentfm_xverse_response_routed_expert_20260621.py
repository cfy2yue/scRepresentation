#!/usr/bin/env python3
"""CPU gate for routed use of an xverse response-repair expert.

Routes use train-only/deployable covariates. This report still selects route
families posthoc, so it is a CPU gate only, not a promotion claim.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_xverse_response_repair_smoke_20260621/xverse_response_pca32_aux025_replay1_4k/posthoc_eval_stablecaps"
DEFAULT_BASE_SPLIT = RUN_ROOT / "split_group_eval_anchor_ode20_stablecaps.json"
DEFAULT_CAND_SPLIT = RUN_ROOT / "split_group_eval_candidate_ode20_stablecaps.json"
DEFAULT_BASE_FAMILY = RUN_ROOT / "condition_family_eval_anchor_ode20_stablecaps.json"
DEFAULT_CAND_FAMILY = RUN_ROOT / "condition_family_eval_candidate_ode20_stablecaps.json"
DEFAULT_COVARIATES = ROOT / "reports/latentfm_xverse_trainonly_repair_covariates_20260621.json"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_response_routed_expert_cpu_gate_20260621.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_RESPONSE_ROUTED_EXPERT_CPU_GATE_20260621.md"

METRICS = ("pearson_pert", "test_mmd_clamped")
GROUPS = ("test", "test_multi", "test_multi_unseen2", "family_gene", "family_drug", "structure_multi")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float | None:
    try:
        if value is None:
            return None
        val = float(value)
        if val != val:
            return None
        return val
    except (TypeError, ValueError):
        return None


def condition_table(payload: dict[str, Any], group: str) -> dict[tuple[str, str], dict[str, Any]]:
    rows = (payload.get("groups", {}).get(group, {}) or {}).get("condition_metrics") or []
    return {
        (str(row.get("dataset")), str(row.get("condition"))): row
        for row in rows
        if isinstance(row, dict) and row.get("dataset") and row.get("condition")
    }


def feature_lookup(cov_payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for row in cov_payload.get("rows") or []:
        out[(str(row["dataset"]), str(row["condition"]))] = row
    return out


def build_rows(
    base_split: dict[str, Any],
    cand_split: dict[str, Any],
    base_family: dict[str, Any],
    cand_family: dict[str, Any],
    features: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in GROUPS:
        bp = base_split if group in (base_split.get("groups") or {}) else base_family
        cp = cand_split if group in (cand_split.get("groups") or {}) else cand_family
        btab = condition_table(bp, group)
        ctab = condition_table(cp, group)
        for key in sorted(set(btab) & set(ctab)):
            b = btab[key]
            c = ctab[key]
            feat = features.get(key, {})
            row: dict[str, Any] = {
                "group": group,
                "dataset": key[0],
                "condition": key[1],
                "is_multi_feature_available": bool(feat),
                **{k: v for k, v in feat.items() if k not in {"group", "dataset", "condition"}},
            }
            for metric in METRICS:
                row[f"anchor__{metric}"] = fnum(b.get(metric))
                row[f"candidate__{metric}"] = fnum(c.get(metric))
            rows.append(row)
    return rows


def route_defs(cov_payload: dict[str, Any]) -> dict[str, Callable[[dict[str, Any]], bool]]:
    vals: dict[str, list[float]] = defaultdict(list)
    for row in cov_payload.get("rows") or []:
        for key in ("global_mean_resid_norm", "scgpt_pair_cos", "cellnavi_pair_cos"):
            if row.get(key) is not None:
                vals[key].append(float(row[key]))
    q75 = {k: float(np.percentile(v, 75)) for k, v in vals.items() if v}
    med = {k: float(np.median(v)) for k, v in vals.items() if v}

    def target(row: dict[str, Any]) -> bool:
        return bool(row.get("is_multi_feature_available"))

    routes: dict[str, Callable[[dict[str, Any]], bool]] = {
        "none_anchor_only": lambda r: False,
        "all_featured_multi": target,
        "global_full": lambda r: target(r) and bool(r.get("global_full")),
        "same_not_full": lambda r: target(r) and not bool(r.get("same_full")),
    }
    if "global_mean_resid_norm" in q75:
        routes["global_mean_resid_norm_ge_q75"] = (
            lambda r, t=q75["global_mean_resid_norm"]: target(r)
            and r.get("global_mean_resid_norm") is not None
            and float(r["global_mean_resid_norm"]) >= t
        )
    if "global_mean_resid_norm" in med:
        routes["global_mean_resid_norm_ge_median"] = (
            lambda r, t=med["global_mean_resid_norm"]: target(r)
            and r.get("global_mean_resid_norm") is not None
            and float(r["global_mean_resid_norm"]) >= t
        )
    if "scgpt_pair_cos" in q75:
        routes["scgpt_pair_cos_ge_q75"] = (
            lambda r, t=q75["scgpt_pair_cos"]: target(r)
            and r.get("scgpt_pair_cos") is not None
            and float(r["scgpt_pair_cos"]) >= t
        )
    if "cellnavi_pair_cos" in q75:
        routes["cellnavi_pair_cos_ge_q75"] = (
            lambda r, t=q75["cellnavi_pair_cos"]: target(r)
            and r.get("cellnavi_pair_cos") is not None
            and float(r["cellnavi_pair_cos"]) >= t
        )
    return routes


def delta_for(row: dict[str, Any], route: Callable[[dict[str, Any]], bool], metric: str) -> float | None:
    base = row.get(f"anchor__{metric}")
    cand = row.get(f"candidate__{metric}")
    if base is None or cand is None:
        return None
    return (float(cand) if route(row) else float(base)) - float(base)


def bootstrap(
    rows: list[dict[str, Any]],
    route_name: str,
    route: Callable[[dict[str, Any]], bool],
    *,
    n_boot: int,
    rng: np.random.Generator,
    resample_datasets: bool,
) -> list[dict[str, Any]]:
    out = []
    for group in GROUPS:
        group_rows = [r for r in rows if r["group"] == group]
        if not group_rows:
            continue
        selected = sum(1 for r in group_rows if route(r))
        for metric in METRICS:
            by_ds: dict[str, list[float]] = defaultdict(list)
            for row in group_rows:
                d = delta_for(row, route, metric)
                if d is not None:
                    by_ds[str(row["dataset"])].append(float(d))
            datasets = sorted(ds for ds, vals in by_ds.items() if vals)
            if not datasets:
                continue
            observed = float(np.mean([np.mean(by_ds[ds]) for ds in datasets]))
            samples = []
            for _ in range(n_boot):
                picked = rng.choice(datasets, size=len(datasets), replace=True) if resample_datasets else datasets
                vals = []
                for ds in picked:
                    arr = np.asarray(by_ds[str(ds)], dtype=float)
                    idx = rng.integers(0, len(arr), size=len(arr))
                    vals.append(float(np.mean(arr[idx])))
                samples.append(float(np.mean(vals)))
            arr = np.asarray(samples, dtype=float)
            lo, hi = np.quantile(arr, [0.025, 0.975])
            if metric == "pearson_pert":
                p_improve = float(np.mean(arr > 0.0))
                p_harm = float(np.mean(arr < 0.0))
            else:
                p_improve = float(np.mean(arr < 0.0))
                p_harm = float(np.mean(arr > 0.0))
            out.append(
                {
                    "route": route_name,
                    "group": group,
                    "metric": metric,
                    "selected": selected,
                    "n_conditions": len(group_rows),
                    "n_datasets": len(datasets),
                    "delta": observed,
                    "ci95": [float(lo), float(hi)],
                    "p_improve": p_improve,
                    "p_harm": p_harm,
                    "bootstrap": "dataset_resampled" if resample_datasets else "within_dataset",
                }
            )
    return out


def assess(rows: list[dict[str, Any]], route_name: str) -> dict[str, Any]:
    by = {(r["bootstrap"], r["group"], r["metric"]): r for r in rows if r["route"] == route_name}
    reasons = []
    status = "strict_cpu_route_candidate"
    for boot in ("within_dataset", "dataset_resampled"):
        test_pp = by.get((boot, "test", "pearson_pert"))
        fam_pp = by.get((boot, "family_gene", "pearson_pert"))
        u2_pp = by.get((boot, "test_multi_unseen2", "pearson_pert"))
        u2_mmd = by.get((boot, "test_multi_unseen2", "test_mmd_clamped"))
        multi_mmd = by.get((boot, "test_multi", "test_mmd_clamped"))
        if not all([test_pp, fam_pp, u2_pp, u2_mmd, multi_mmd]):
            reasons.append(f"{boot}:missing_required")
            status = "incomplete"
            continue
        if float(test_pp["p_harm"]) > 0.80:
            reasons.append(f"{boot}:test_pp_harm")
        if float(fam_pp["p_harm"]) > 0.80:
            reasons.append(f"{boot}:family_pp_harm")
        if float(u2_pp["delta"]) < 0.02 or float(u2_pp["p_improve"]) < 0.90:
            reasons.append(f"{boot}:unseen2_pp_not_supported")
        if float(u2_mmd["p_harm"]) > 0.80:
            reasons.append(f"{boot}:unseen2_mmd_harm")
        if float(multi_mmd["p_harm"]) > 0.80:
            reasons.append(f"{boot}:multi_mmd_harm")
    if reasons and status != "incomplete":
        status = "diagnostic_or_fail"
    return {"route": route_name, "status": status, "reasons": reasons}


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM xverse Response Routed Expert CPU Gate 2026-06-21",
        "",
        "Routes use train-only/deployable covariates, but route families were screened posthoc. This is not promotion evidence.",
        "",
        "## Decisions",
        "",
        "| route | status | reasons |",
        "|---|---|---|",
    ]
    for dec in payload["decisions"]:
        lines.append(f"| {dec['route']} | {dec['status']} | {', '.join(dec['reasons']) or '-'} |")
    for boot in ("within_dataset", "dataset_resampled"):
        lines.extend([
            "",
            f"## {boot} Bootstrap",
            "",
            "| route | group | metric | selected | delta | ci95 | p_improve | p_harm |",
            "|---|---|---|---:|---:|---|---:|---:|",
        ])
        for row in payload["bootstrap_rows"]:
            if row["bootstrap"] != boot:
                continue
            if row["group"] not in {"test", "test_multi", "test_multi_unseen2", "family_gene"}:
                continue
            if row["metric"] not in METRICS:
                continue
            ci = row["ci95"]
            lines.append(
                "| {route} | {group} | {metric} | {selected} | {delta} | [{lo}, {hi}] | {pi:.3f} | {ph:.3f} |".format(
                    route=row["route"],
                    group=row["group"],
                    metric=row["metric"],
                    selected=row["selected"],
                    delta=fmt(row["delta"]),
                    lo=fmt(ci[0]),
                    hi=fmt(ci[1]),
                    pi=float(row["p_improve"]),
                    ph=float(row["p_harm"]),
                )
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- A strict candidate must survive both within-dataset and dataset-resampled bootstrap.",
        "- If no strict candidate appears, do not start another response-weight GPU run from this design.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-split-json", type=Path, default=DEFAULT_BASE_SPLIT)
    parser.add_argument("--candidate-split-json", type=Path, default=DEFAULT_CAND_SPLIT)
    parser.add_argument("--baseline-family-json", type=Path, default=DEFAULT_BASE_FAMILY)
    parser.add_argument("--candidate-family-json", type=Path, default=DEFAULT_CAND_FAMILY)
    parser.add_argument("--covariates-json", type=Path, default=DEFAULT_COVARIATES)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    cov = load_json(args.covariates_json)
    features = feature_lookup(cov)
    rows = build_rows(
        load_json(args.baseline_split_json),
        load_json(args.candidate_split_json),
        load_json(args.baseline_family_json),
        load_json(args.candidate_family_json),
        features,
    )
    routes = route_defs(cov)
    rng = np.random.default_rng(args.seed)
    boot_rows = []
    for route_name, pred in routes.items():
        boot_rows.extend(bootstrap(rows, route_name, pred, n_boot=args.n_boot, rng=rng, resample_datasets=False))
        boot_rows.extend(bootstrap(rows, route_name, pred, n_boot=args.n_boot, rng=rng, resample_datasets=True))
    payload = {
        "baseline_split_json": str(args.baseline_split_json),
        "candidate_split_json": str(args.candidate_split_json),
        "baseline_family_json": str(args.baseline_family_json),
        "candidate_family_json": str(args.candidate_family_json),
        "covariates_json": str(args.covariates_json),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "routes": sorted(routes),
        "bootstrap_rows": boot_rows,
        "decisions": [assess(boot_rows, route_name) for route_name in sorted(routes)],
    }
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
