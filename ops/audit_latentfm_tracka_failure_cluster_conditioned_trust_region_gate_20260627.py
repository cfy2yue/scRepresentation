#!/usr/bin/env python3
"""CPU gate for Track A failure-cluster conditioned trust-region policies.

This is a query-blind/report-only gate over existing train-only/internal
forensics rows. It tests whether predeclared failure clusters can select a
deployable proxy policy that improves internal cross/family Pearson without
using target-derived forensic columns. It does not train, infer, read canonical
multi, read Track C query, or use GPU.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
FORENSICS = ROOT / "reports/latentfm_xverse_tracka_residual_forensics_conditions_20260622.csv"
FAILURES = ROOT / "reports/tracka_deployable_benchmark_failure_taxonomy_20260627/failure_cases.csv"
OUT_ROWS = ROOT / "reports/latentfm_tracka_failure_cluster_conditioned_trust_region_gate_rows_20260627.csv"
OUT_JSON = ROOT / "reports/latentfm_tracka_failure_cluster_conditioned_trust_region_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_FAILURE_CLUSTER_CONDITIONED_TRUST_REGION_GATE_20260627.md"

CANDIDATES = ["gene_raw_mean", "dataset_mean", "global_mean", "shrink_k8"]
FORBIDDEN_COLUMNS = {"target_residual_norm", "gene_target_cosine", "dataset_target_cosine"}
GROUP_CROSS = "internal_val_cross_background_seen_gene_proxy"
GROUP_FAMILY = "internal_val_family_gene_proxy"
SEED = 20260627


def norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def fnum(value: Any) -> float | None:
    text = norm(value)
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def cluster_for(dataset: str, condition: str) -> str:
    if dataset.startswith("Jiang_"):
        return "jiang_cytokine_single_gene"
    if dataset == "Replogle_RPE1essential":
        return "replogle_rpe1_essential_crispr"
    if dataset == "Adamson":
        return "adamson_stress_translation_like"
    if dataset == "GasperiniShendure2019_lowMOI":
        return "gasperini_lowmoi"
    if dataset.startswith("Nadig_"):
        return "nadig_cellline_single_gene"
    if dataset == "Wessels":
        return "wessels_combinatorial_diagnostic"
    return "non_target"


def read_forensics() -> list[dict[str, Any]]:
    rows = []
    with FORENSICS.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rec = dict(row)
            rec["dataset"] = norm(row.get("dataset"))
            rec["condition"] = norm(row.get("condition"))
            rec["group"] = norm(row.get("group"))
            rec["cluster"] = cluster_for(rec["dataset"], rec["condition"])
            for col in ["anchor_pearson_pert", "anchor_mmd_clamped", "gene_train_count", *CANDIDATES]:
                rec[col] = fnum(row.get(col))
            rows.append(rec)
    return rows


def read_failure_clusters() -> dict[str, Any]:
    unique = {}
    with FAILURES.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            unique.setdefault((row["dataset"], row["condition"]), row)
    counts = defaultdict(int)
    examples = defaultdict(list)
    for dataset, condition in unique:
        cluster = cluster_for(dataset, condition)
        counts[cluster] += 1
        examples[cluster].append(f"{dataset}:{condition}")
    return {"unique_cases": len(unique), "counts": dict(counts), "examples": dict(examples)}


def candidate_value(row: dict[str, Any], candidate: str) -> float:
    val = row.get(candidate)
    if val is None:
        return row.get("anchor_pearson_pert") or 0.0
    return float(val)


def row_pred(row: dict[str, Any], policy: dict[str, str]) -> float:
    anchor = row.get("anchor_pearson_pert")
    if anchor is None:
        return 0.0
    candidate = policy.get(row["cluster"], "anchor")
    if candidate == "anchor":
        return float(anchor)
    return candidate_value(row, candidate)


def evaluate(rows: list[dict[str, Any]], policy: dict[str, str]) -> dict[str, Any]:
    by_group = {}
    row_out = []
    for group in [GROUP_CROSS, GROUP_FAMILY]:
        group_rows = [r for r in rows if r["group"] == group and r.get("anchor_pearson_pert") is not None]
        deltas = []
        target_deltas = []
        non_target_deltas = []
        dataset_deltas = defaultdict(list)
        for row in group_rows:
            pred = row_pred(row, policy)
            delta = pred - float(row["anchor_pearson_pert"])
            deltas.append(delta)
            dataset_deltas[row["dataset"]].append(delta)
            if row["cluster"] == "non_target":
                non_target_deltas.append(delta)
            else:
                target_deltas.append(delta)
            row_out.append(
                {
                    "group": group,
                    "dataset": row["dataset"],
                    "condition": row["condition"],
                    "cluster": row["cluster"],
                    "selected_policy": policy.get(row["cluster"], "anchor"),
                    "anchor_pearson_pert": row["anchor_pearson_pert"],
                    "policy_pearson_pert": pred,
                    "pp_delta": delta,
                    "anchor_mmd_clamped": row.get("anchor_mmd_clamped"),
                }
            )
        dataset_means = {ds: mean(vals) for ds, vals in dataset_deltas.items() if vals}
        by_group[group] = {
            "n_rows": len(group_rows),
            "pp_gain": mean(deltas) if deltas else None,
            "dataset_min": min(dataset_means.values()) if dataset_means else None,
            "target_cluster_gain": mean(target_deltas) if target_deltas else None,
            "non_target_min": min(non_target_deltas) if non_target_deltas else None,
            "p_harm": sum(1 for x in deltas if x < 0) / len(deltas) if deltas else None,
            "dataset_means": dataset_means,
        }
    return {"groups": by_group, "rows": row_out}


def policy_space(clusters: list[str]) -> list[dict[str, str]]:
    policies = []
    choices = ["anchor", *CANDIDATES]
    # Conservative space: all target clusters share one candidate, or one cluster
    # is targeted at a time. This prevents hidden high-dimensional label fitting.
    for candidate in choices:
        policies.append({cluster: candidate for cluster in clusters})
    for cluster in clusters:
        for candidate in choices:
            p = {c: "anchor" for c in clusters}
            p[cluster] = candidate
            policies.append(p)
    dedup = []
    seen = set()
    for p in policies:
        key = tuple(sorted(p.items()))
        if key not in seen:
            dedup.append(p)
            seen.add(key)
    return dedup


def score_eval(ev: dict[str, Any]) -> float:
    cross = ev["groups"][GROUP_CROSS]
    fam = ev["groups"][GROUP_FAMILY]
    return (cross.get("pp_gain") or -999) + (fam.get("pp_gain") or -999)


def select_policy_lodo(rows: list[dict[str, Any]], target_clusters: list[str]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    policies = policy_space(target_clusters)
    datasets = sorted({r["dataset"] for r in rows})
    heldout_evals = []
    votes = defaultdict(int)
    for held in datasets:
        train = [r for r in rows if r["dataset"] != held]
        test = [r for r in rows if r["dataset"] == held]
        ranked = sorted(((score_eval(evaluate(train, p)), p) for p in policies), key=lambda x: x[0], reverse=True)
        selected = ranked[0][1]
        votes[tuple(sorted(selected.items()))] += 1
        ev = evaluate(test, selected)
        heldout_evals.append({"heldout_dataset": held, "selected_policy": selected, "score": ranked[0][0], "evaluation": ev["groups"]})
    final_key = max(votes.items(), key=lambda x: x[1])[0]
    return dict(final_key), heldout_evals


def bootstrap_ci(values: list[float], n_boot: int = 2000) -> dict[str, Any]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return {"mean": None, "ci_low": None, "ci_high": None, "p_improve": None}
    rng = random.Random(SEED)
    samples = []
    for _ in range(n_boot):
        samples.append(sum(rng.choice(vals) for _ in vals) / len(vals))
    samples.sort()
    return {
        "mean": mean(vals),
        "ci_low": samples[int(0.025 * (len(samples) - 1))],
        "ci_high": samples[int(0.975 * (len(samples) - 1))],
        "p_improve": sum(1 for x in samples if x > 0) / len(samples),
    }


def control_eval(rows: list[dict[str, Any]], policy: dict[str, str], mode: str) -> dict[str, Any]:
    control_rows = [dict(r) for r in rows]
    rng = random.Random(SEED + len(mode))
    if mode == "shuffled_cluster":
        clusters = [r["cluster"] for r in control_rows]
        rng.shuffle(clusters)
        for r, c in zip(control_rows, clusters):
            r["cluster"] = c
    elif mode == "sign_inverted":
        ev = evaluate(control_rows, policy)
        for row in ev["rows"]:
            row["pp_delta"] = -float(row["pp_delta"])
        return {"groups": summarize_rows(ev["rows"])}
    elif mode == "count_only":
        for r in control_rows:
            count = r.get("gene_train_count")
            r["cluster"] = "high_count" if count is not None and count >= 4 else "low_count"
    elif mode == "dataset_identity":
        for r in control_rows:
            r["cluster"] = r["dataset"]
    else:
        raise ValueError(mode)
    return evaluate(control_rows, policy)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for group in [GROUP_CROSS, GROUP_FAMILY]:
        vals = [float(r["pp_delta"]) for r in rows if r["group"] == group]
        by_ds = defaultdict(list)
        for r in rows:
            if r["group"] == group:
                by_ds[r["dataset"]].append(float(r["pp_delta"]))
        ds_means = [mean(v) for v in by_ds.values() if v]
        out[group] = {
            "n_rows": len(vals),
            "pp_gain": mean(vals) if vals else None,
            "dataset_min": min(ds_means) if ds_means else None,
            "p_harm": sum(1 for v in vals if v < 0) / len(vals) if vals else None,
        }
    return out


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "group",
        "dataset",
        "condition",
        "cluster",
        "selected_policy",
        "anchor_pearson_pert",
        "policy_pearson_pert",
        "pp_delta",
        "anchor_mmd_clamped",
    ]
    with OUT_ROWS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def fmt(v: Any) -> str:
    if v is None:
        return "NA"
    return f"{float(v):+.6f}"


def main() -> int:
    rows = read_forensics()
    failure_summary = read_failure_clusters()
    target_clusters = sorted(c for c in failure_summary["counts"] if c != "non_target")
    selected_lodo, lodo = select_policy_lodo(rows, target_clusters)
    final_eval = evaluate(rows, selected_lodo)
    write_csv(final_eval["rows"])

    pp_values = {
        group: [float(r["pp_delta"]) for r in final_eval["rows"] if r["group"] == group]
        for group in [GROUP_CROSS, GROUP_FAMILY]
    }
    boot = {group: bootstrap_ci(vals) for group, vals in pp_values.items()}
    controls = {
        mode: control_eval(rows, selected_lodo, mode)
        for mode in ["shuffled_cluster", "sign_inverted", "count_only", "dataset_identity"]
    }
    control_summary = {}
    for mode, ev in controls.items():
        groups = ev.get("groups", {})
        control_summary[mode] = {
            group: {
                "pp_gain": groups.get(group, {}).get("pp_gain"),
                "dataset_min": groups.get(group, {}).get("dataset_min"),
                "p_harm": groups.get(group, {}).get("p_harm"),
            }
            for group in [GROUP_CROSS, GROUP_FAMILY]
        }

    reasons = []
    cross = final_eval["groups"][GROUP_CROSS]
    fam = final_eval["groups"][GROUP_FAMILY]
    if (cross["pp_gain"] or -999) < 0.010:
        reasons.append("cross_pp_gain_below_0p010")
    if (fam["pp_gain"] or -999) < 0.010:
        reasons.append("family_pp_gain_below_0p010")
    if (cross["dataset_min"] or -999) < -0.020 or (fam["dataset_min"] or -999) < -0.020:
        reasons.append("dataset_min_below_minus_0p020")
    if (cross["target_cluster_gain"] or -999) < 0.020 or (fam["target_cluster_gain"] or -999) < 0.020:
        reasons.append("target_cluster_gain_below_0p020")
    if (cross["non_target_min"] or 0) < -0.010 or (fam["non_target_min"] or 0) < -0.010:
        reasons.append("non_target_noharm_min_below_minus_0p010")
    if (cross["p_harm"] or 1) > 0.20 or (fam["p_harm"] or 1) > 0.20:
        reasons.append("p_harm_above_0p20")
    if (boot[GROUP_CROSS]["ci_low"] or -999) <= 0 or (boot[GROUP_FAMILY]["ci_low"] or -999) <= 0:
        reasons.append("bootstrap_ci_low_not_above_0")
    reasons.append("candidate_mmd_unavailable_for_policy_no_gpu")

    status = (
        "tracka_failure_cluster_conditioned_trust_region_gate_fail_no_gpu"
        if reasons
        else "tracka_failure_cluster_conditioned_trust_region_gate_pass_needs_adapter_design_no_gpu"
    )
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_report_only": True,
            "no_training": True,
            "no_inference": True,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "forbidden_columns_excluded": sorted(FORBIDDEN_COLUMNS),
            "selection_inputs": "train-only/internal residual forensics rows plus predeclared failure cluster families",
            "limitation": "policy proxy has no candidate MMD, so it cannot authorize GPU by itself",
        },
        "inputs": {"forensics_csv": str(FORENSICS), "failure_cases_csv": str(FAILURES)},
        "failure_clusters": failure_summary,
        "selected_policy_lodo_vote": selected_lodo,
        "lodo": lodo,
        "final_eval": final_eval["groups"],
        "bootstrap": boot,
        "controls": control_summary,
        "reasons": reasons,
        "outputs": {"rows": str(OUT_ROWS), "json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Track A Failure-Cluster Conditioned Trust-Region Gate",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only over train-only/internal residual forensics rows.",
        "- Predeclared failure clusters come from the Track A taxonomy, but taxonomy rows are not used as model scores.",
        "- Excludes target-derived forensic columns: `target_residual_norm`, `gene_target_cosine`, `dataset_target_cosine`.",
        "- Does not train, infer, read canonical multi, read Track C query, or use GPU.",
        "- Candidate MMD is unavailable for this proxy policy, so this gate cannot directly authorize GPU.",
        "",
        "## Failure Clusters",
        "",
        "| cluster | unique worst cases | examples |",
        "|---|---:|---|",
    ]
    for cluster, count in sorted(failure_summary["counts"].items()):
        examples = "; ".join(failure_summary["examples"][cluster][:5])
        lines.append(f"| `{cluster}` | {count} | {examples} |")
    lines += [
        "",
        "## Selected Policy",
        "",
        "```json",
        json.dumps(selected_lodo, indent=2, sort_keys=True),
        "```",
        "",
        "## Gate Metrics",
        "",
        "| group | pp gain | bootstrap CI | dataset min | target gain | non-target min | p_harm |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in [GROUP_CROSS, GROUP_FAMILY]:
        g = final_eval["groups"][group]
        b = boot[group]
        lines.append(
            f"| `{group}` | {fmt(g['pp_gain'])} | [{fmt(b['ci_low'])}, {fmt(b['ci_high'])}] | "
            f"{fmt(g['dataset_min'])} | {fmt(g['target_cluster_gain'])} | {fmt(g['non_target_min'])} | {fmt(g['p_harm'])} |"
        )
    lines += [
        "",
        "## Controls",
        "",
        "| control | cross pp gain | family pp gain |",
        "|---|---:|---:|",
    ]
    for mode, summary in control_summary.items():
        lines.append(
            f"| `{mode}` | {fmt(summary[GROUP_CROSS]['pp_gain'])} | {fmt(summary[GROUP_FAMILY]['pp_gain'])} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- reasons: `{reasons}`",
        "- If this fails, do not launch a failure-cluster adapter GPU smoke from these proxy features.",
        "- If a future adapter is proposed, it needs real candidate MMD/no-harm metrics and negative-control collapse before GPU.",
        "",
        "## Outputs",
        "",
        f"- rows: `{OUT_ROWS}`",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "reasons": reasons}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
