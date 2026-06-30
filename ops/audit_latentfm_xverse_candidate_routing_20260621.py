#!/usr/bin/env python3
"""CPU gate for deployable routing among xverse LatentFM candidates.

The policies in this script use only train-only/deployable covariates, but the
choice among policies is still posthoc. Treat positive results as hypotheses
for a pre-registered GPU smoke, not as promotion evidence.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
RESP_ROOT = ROOT / "runs/latentfm_xverse_response_repair_smoke_20260621/xverse_response_pca32_aux025_replay1_4k/posthoc_eval_stablecaps"
PRIOR_ROOT = ROOT / "runs/latentfm_xverse_condition_prior_adapter_smoke_20260621/xverse_prior_adapter_global_genemean_w005_add002_replay1_4k/posthoc_eval_stablecaps"
DEFAULT_COVARIATES = ROOT / "reports/latentfm_xverse_trainonly_repair_covariates_20260621.json"
DEFAULT_FEASIBILITY = ROOT / "reports/latentfm_xverse_synthetic_composition_feasibility_20260621.json"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_candidate_routing_cpu_gate_20260621.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_CANDIDATE_ROUTING_CPU_GATE_20260621.md"

METRICS = ("pearson_pert", "test_mmd_clamped")
GROUPS = ("test", "test_multi", "test_multi_unseen2", "family_gene", "family_drug", "structure_multi")
EXPERTS = ("response_aux025", "prior_adapter")


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


def feature_lookup(covariates: dict[str, Any], feasibility: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in feasibility.get("rows") or []:
        key = (str(row["dataset"]), str(row["condition"]))
        out[key] = {
            "genes": row.get("genes") or [],
            "n_genes": int(row.get("n_genes") or len(row.get("genes") or [])),
            "same_hits": int(row.get("same_hits") or 0),
            "global_hits": int(row.get("global_hits") or 0),
            "same_full": bool(row.get("same_full")),
            "global_full": bool(row.get("global_full")),
            "same_prior_norm": fnum(row.get("same_prior_norm")),
            "global_prior_norm": fnum(row.get("global_prior_norm")),
            "same_pair_resid_cos": fnum(row.get("same_pair_resid_cos")),
            "global_pair_resid_cos": fnum(row.get("global_pair_resid_cos")),
        }
    for row in covariates.get("rows") or []:
        key = (str(row["dataset"]), str(row["condition"]))
        feat = out.setdefault(key, {})
        for name in (
            "genes",
            "n_genes",
            "same_hits",
            "global_hits",
            "same_full",
            "global_full",
            "same_hit_frac",
            "global_hit_frac",
            "same_mean_resid_norm",
            "same_max_resid_norm",
            "global_mean_resid_norm",
            "global_max_resid_norm",
            "scgpt_pair_cos",
            "cellnavi_pair_cos",
        ):
            if name in row:
                feat[name] = row[name]
    return out


def read_eval_bundle(root: Path) -> dict[str, dict[str, Any]]:
    return {
        "split_anchor": load_json(root / "split_group_eval_anchor_ode20_stablecaps.json"),
        "split_candidate": load_json(root / "split_group_eval_candidate_ode20_stablecaps.json"),
        "family_anchor": load_json(root / "condition_family_eval_anchor_ode20_stablecaps.json"),
        "family_candidate": load_json(root / "condition_family_eval_candidate_ode20_stablecaps.json"),
    }


def payload_for_group(bundle: dict[str, dict[str, Any]], group: str, candidate: bool) -> dict[str, Any]:
    if group in (bundle["split_anchor"].get("groups") or {}):
        return bundle["split_candidate" if candidate else "split_anchor"]
    return bundle["family_candidate" if candidate else "family_anchor"]


def build_rows(
    response: dict[str, dict[str, Any]],
    prior: dict[str, dict[str, Any]],
    features: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in GROUPS:
        anchor_tab = condition_table(payload_for_group(response, group, False), group)
        resp_tab = condition_table(payload_for_group(response, group, True), group)
        prior_tab = condition_table(payload_for_group(prior, group, True), group)
        for key in sorted(set(anchor_tab) & set(resp_tab) & set(prior_tab)):
            feat = features.get(key, {})
            row: dict[str, Any] = {
                "group": group,
                "dataset": key[0],
                "condition": key[1],
                "has_deployable_features": bool(feat),
                **feat,
            }
            if "n_genes" not in row:
                row["n_genes"] = len(row.get("genes") or [])
            for metric in METRICS:
                row[f"anchor__{metric}"] = fnum(anchor_tab[key].get(metric))
                row[f"response_aux025__{metric}"] = fnum(resp_tab[key].get(metric))
                row[f"prior_adapter__{metric}"] = fnum(prior_tab[key].get(metric))
            rows.append(row)
    return rows


def percentiles(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    vals: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if not is_gene_multi(row):
            continue
        for name in (
            "global_mean_resid_norm",
            "global_prior_norm",
            "scgpt_pair_cos",
            "cellnavi_pair_cos",
            "global_pair_resid_cos",
        ):
            val = fnum(row.get(name))
            if val is not None:
                vals[name].append(val)
    out = {}
    for name, arr in vals.items():
        if arr:
            out[name] = {
                "q25": float(np.percentile(arr, 25)),
                "median": float(np.median(arr)),
                "q75": float(np.percentile(arr, 75)),
            }
    return out


def is_gene_multi(row: dict[str, Any]) -> bool:
    return bool(row.get("has_deployable_features")) and int(row.get("n_genes") or 0) >= 2


Policy = Callable[[dict[str, Any]], str]


def expert_if(expert: str, pred: Callable[[dict[str, Any]], bool]) -> Policy:
    return lambda row: expert if pred(row) else "anchor"


def policy_defs(rows: list[dict[str, Any]]) -> dict[str, Policy]:
    pct = percentiles(rows)

    def ge(name: str, which: str) -> Callable[[dict[str, Any]], bool]:
        threshold = pct.get(name, {}).get(which)
        return lambda row, n=name, t=threshold: (
            is_gene_multi(row) and t is not None and row.get(n) is not None and float(row[n]) >= float(t)
        )

    policies: dict[str, Policy] = {
        "anchor_only": lambda row: "anchor",
        "response_all_gene_multi": expert_if("response_aux025", is_gene_multi),
        "prior_all_gene_multi": expert_if("prior_adapter", is_gene_multi),
        "response_global_full": expert_if("response_aux025", lambda r: is_gene_multi(r) and bool(r.get("global_full"))),
        "prior_global_full": expert_if("prior_adapter", lambda r: is_gene_multi(r) and bool(r.get("global_full"))),
        "response_global_resid_ge_median": expert_if("response_aux025", ge("global_mean_resid_norm", "median")),
        "response_global_resid_ge_q75": expert_if("response_aux025", ge("global_mean_resid_norm", "q75")),
        "prior_global_prior_norm_ge_median": expert_if("prior_adapter", ge("global_prior_norm", "median")),
        "prior_global_prior_norm_ge_q75": expert_if("prior_adapter", ge("global_prior_norm", "q75")),
        "response_scgpt_pair_cos_ge_q75": expert_if("response_aux025", ge("scgpt_pair_cos", "q75")),
        "response_cellnavi_pair_cos_ge_q75": expert_if("response_aux025", ge("cellnavi_pair_cos", "q75")),
        "response_global_pair_resid_cos_ge_q75": expert_if("response_aux025", ge("global_pair_resid_cos", "q75")),
    }

    def hybrid_resid_then_prior(row: dict[str, Any]) -> str:
        if ge("global_mean_resid_norm", "q75")(row):
            return "response_aux025"
        if ge("global_prior_norm", "q75")(row):
            return "prior_adapter"
        return "anchor"

    def hybrid_prior_then_resid(row: dict[str, Any]) -> str:
        if ge("global_prior_norm", "q75")(row):
            return "prior_adapter"
        if ge("global_mean_resid_norm", "q75")(row):
            return "response_aux025"
        return "anchor"

    policies["hybrid_response_resid_q75_then_prior_norm_q75"] = hybrid_resid_then_prior
    policies["hybrid_prior_norm_q75_then_response_resid_q75"] = hybrid_prior_then_resid
    return policies


def metric_delta(row: dict[str, Any], policy: Policy, metric: str) -> float | None:
    base = row.get(f"anchor__{metric}")
    expert = policy(row)
    cand = row.get(f"{expert}__{metric}") if expert != "anchor" else base
    if base is None or cand is None:
        return None
    return float(cand) - float(base)


def selected_counts(rows: list[dict[str, Any]], policy: Policy) -> dict[str, int]:
    counts = {"anchor": 0, "response_aux025": 0, "prior_adapter": 0}
    for row in rows:
        counts[policy(row)] = counts.get(policy(row), 0) + 1
    return counts


def bootstrap_policy(
    rows: list[dict[str, Any]],
    policy_name: str,
    policy: Policy,
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
        counts = selected_counts(group_rows, policy)
        for metric in METRICS:
            by_ds: dict[str, list[float]] = defaultdict(list)
            for row in group_rows:
                delta = metric_delta(row, policy, metric)
                if delta is not None:
                    by_ds[str(row["dataset"])].append(delta)
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
                    "policy": policy_name,
                    "group": group,
                    "metric": metric,
                    "n_conditions": len(group_rows),
                    "n_datasets": len(datasets),
                    "selected": counts,
                    "delta": observed,
                    "ci95": [float(lo), float(hi)],
                    "p_improve": p_improve,
                    "p_harm": p_harm,
                    "bootstrap": "dataset_resampled" if resample_datasets else "within_dataset",
                }
            )
    return out


def assess(boot_rows: list[dict[str, Any]], policy_name: str) -> dict[str, Any]:
    by = {(r["bootstrap"], r["group"], r["metric"]): r for r in boot_rows if r["policy"] == policy_name}
    status = "strict_cpu_route_candidate"
    reasons = []
    for boot in ("within_dataset", "dataset_resampled"):
        required = {
            "test_pp": by.get((boot, "test", "pearson_pert")),
            "family_gene_pp": by.get((boot, "family_gene", "pearson_pert")),
            "unseen2_pp": by.get((boot, "test_multi_unseen2", "pearson_pert")),
            "unseen2_mmd": by.get((boot, "test_multi_unseen2", "test_mmd_clamped")),
            "test_multi_mmd": by.get((boot, "test_multi", "test_mmd_clamped")),
        }
        if not all(required.values()):
            status = "incomplete"
            reasons.append(f"{boot}:missing_required")
            continue
        if float(required["test_pp"]["p_harm"]) > 0.80:
            reasons.append(f"{boot}:test_pp_harm")
        if float(required["family_gene_pp"]["p_harm"]) > 0.80:
            reasons.append(f"{boot}:family_gene_pp_harm")
        if float(required["unseen2_pp"]["delta"]) < 0.02 or float(required["unseen2_pp"]["p_improve"]) < 0.90:
            reasons.append(f"{boot}:unseen2_pp_not_supported")
        if float(required["unseen2_mmd"]["p_harm"]) > 0.80:
            reasons.append(f"{boot}:unseen2_mmd_harm")
        if float(required["test_multi_mmd"]["p_harm"]) > 0.80:
            reasons.append(f"{boot}:test_multi_mmd_harm")
    if reasons and status != "incomplete":
        status = "diagnostic_or_fail"
    return {"policy": policy_name, "status": status, "reasons": reasons}


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM xverse Candidate Routing CPU Gate 2026-06-21",
        "",
        "This is a CPU-only posthoc policy screen. Policy inputs are train-only/deployable covariates; held-out outcomes are used only for scoring.",
        "",
        "## Provenance",
        "",
        f"- response root: `{payload['response_root']}`",
        f"- prior root: `{payload['prior_root']}`",
        f"- covariates: `{payload['covariates_json']}`",
        f"- feasibility: `{payload['feasibility_json']}`",
        f"- bootstrap n: `{payload['n_boot']}`",
        "",
        "## Decisions",
        "",
        "| policy | status | reasons |",
        "|---|---|---|",
    ]
    for row in payload["decisions"]:
        lines.append(f"| {row['policy']} | {row['status']} | {', '.join(row['reasons']) or '-'} |")

    for boot in ("within_dataset", "dataset_resampled"):
        lines.extend(
            [
                "",
                f"## {boot} Bootstrap",
                "",
                "| policy | group | metric | selected response/prior | delta | ci95 | p_improve | p_harm |",
                "|---|---|---|---:|---:|---|---:|---:|",
            ]
        )
        for row in payload["bootstrap_rows"]:
            if row["bootstrap"] != boot:
                continue
            if row["group"] not in {"test", "test_multi", "test_multi_unseen2", "family_gene"}:
                continue
            ci = row["ci95"]
            selected = row["selected"]
            lines.append(
                "| {policy} | {group} | {metric} | {resp}/{prior} | {delta} | [{lo}, {hi}] | {pi:.3f} | {ph:.3f} |".format(
                    policy=row["policy"],
                    group=row["group"],
                    metric=row["metric"],
                    resp=selected.get("response_aux025", 0),
                    prior=selected.get("prior_adapter", 0),
                    delta=fmt(row["delta"]),
                    lo=fmt(ci[0]),
                    hi=fmt(ci[1]),
                    pi=float(row["p_improve"]),
                    ph=float(row["p_harm"]),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A strict route candidate must preserve aggregate/family pp and avoid MMD harm in both bootstrap modes.",
            "- If every policy is diagnostic_or_fail, the next GPU step should not be another blind response/prior sweep.",
            "- Positive policies here must be converted into a pre-declared GPU smoke before any promotion claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-root", type=Path, default=RESP_ROOT)
    parser.add_argument("--prior-root", type=Path, default=PRIOR_ROOT)
    parser.add_argument("--covariates-json", type=Path, default=DEFAULT_COVARIATES)
    parser.add_argument("--feasibility-json", type=Path, default=DEFAULT_FEASIBILITY)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    features = feature_lookup(load_json(args.covariates_json), load_json(args.feasibility_json))
    rows = build_rows(read_eval_bundle(args.response_root), read_eval_bundle(args.prior_root), features)
    policies = policy_defs(rows)
    rng = np.random.default_rng(args.seed)
    boot_rows = []
    for policy_name, policy in sorted(policies.items()):
        boot_rows.extend(bootstrap_policy(rows, policy_name, policy, n_boot=args.n_boot, rng=rng, resample_datasets=False))
        boot_rows.extend(bootstrap_policy(rows, policy_name, policy, n_boot=args.n_boot, rng=rng, resample_datasets=True))
    payload = {
        "response_root": str(args.response_root),
        "prior_root": str(args.prior_root),
        "covariates_json": str(args.covariates_json),
        "feasibility_json": str(args.feasibility_json),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "n_rows": len(rows),
        "groups": GROUPS,
        "experts": EXPERTS,
        "policies": sorted(policies),
        "bootstrap_rows": boot_rows,
        "decisions": [assess(boot_rows, policy_name) for policy_name in sorted(policies)],
    }
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "n_rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
