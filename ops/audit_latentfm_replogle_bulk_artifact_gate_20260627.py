#!/usr/bin/env python3
"""Strict CPU association gate for Replogle author bulk artifacts.

This gate joins Replogle source artifacts to frozen xverse_8k anchor condition
metrics for legal Track A single-gene/family contexts. It is report-only: no
training, inference, checkpoint selection, canonical multi selection, Track C
query, or GPU.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
ARTIFACT_CSV = ROOT / "reports/replogle_bulk_artifacts_20260627/replogle_bulk_condition_artifacts.csv"
OUT_DIR = ROOT / "reports/replogle_bulk_artifact_gate_20260627"
OUT_JOINED = OUT_DIR / "replogle_bulk_artifact_gate_joined_rows.csv"
OUT_SUMMARY = OUT_DIR / "replogle_bulk_artifact_gate_summary.csv"
OUT_JSON = ROOT / "reports/latentfm_replogle_bulk_artifact_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_REPLOGLE_BULK_ARTIFACT_GATE_20260627.md"

ANCHOR_ROOT = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/"
    / "xverse_comp006_endpoint5_8k_seed42_fulleval"
)
REPLICATE_ROOT = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_seed_replicate_20260621/"
    / "xverse_comp006_endpoint5_8k_seed43_fulleval"
)

INPUTS = {
    "seed42_split": ANCHOR_ROOT
    / "posthoc_eval_uncapped_20260621/split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json",
    "seed42_family": ANCHOR_ROOT
    / "posthoc_eval_uncapped_20260621/condition_family_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json",
    "seed43_split": REPLICATE_ROOT
    / "posthoc_eval_uncapped_20260621/split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json",
    "seed43_family": REPLICATE_ROOT
    / "posthoc_eval_uncapped_20260621/condition_family_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json",
}

GROUPS = {"test_single", "family_gene", "test_all"}
DATASETS = {"ReplogleWeissman2022_K562_gwps", "Replogle_RPE1essential"}


def fnum(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:12], 16) % (2**32)


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    mx, my = mean(x), mean(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 0 or vy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(rankdata(x), rankdata(y))


def load_eval_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, path in INPUTS.items():
        seed, source = key.split("_", 1)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for group, data in payload.get("groups", {}).items():
            if group not in GROUPS:
                continue
            for row in data.get("condition_metrics") or []:
                dataset = str(row.get("dataset", ""))
                condition = str(row.get("condition", ""))
                if dataset not in DATASETS:
                    continue
                pp = fnum(row.get("pearson_pert"))
                mmd = fnum(row.get("test_mmd_clamped"))
                if pp is None or mmd is None:
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "source": source,
                        "group": group,
                        "dataset": dataset,
                        "condition": condition,
                        "pearson_pert": pp,
                        "test_mmd_clamped": mmd,
                    }
                )
    return rows


def load_artifacts() -> list[dict[str, Any]]:
    rows = []
    with ARTIFACT_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] not in DATASETS:
                continue
            value = fnum(row.get("artifact_value"))
            if value is None:
                continue
            rows.append(
                {
                    "dataset": row["dataset"],
                    "condition": row["condition"],
                    "artifact": row["artifact"],
                    "artifact_role": row["artifact_role"],
                    "source_label": row["source_label"],
                    "raw_column": row["raw_column"],
                    "artifact_value": value,
                }
            )
            # Formal pass candidates must generalize across the matched K562
            # gwps and RPE1 sources. K562_essential is retained as a diagnostic
            # source-specific artifact but is not pooled into the cross-background
            # gate because the local dataset is K562_gwps.
            if row["source_label"] in {"K562_gwps", "RPE1"}:
                rows.append(
                    {
                        "dataset": row["dataset"],
                        "condition": row["condition"],
                        "artifact": f"replogle_bulk_pooled_{row['raw_column']}",
                        "artifact_role": row["artifact_role"],
                        "source_label": "pooled_K562_gwps_RPE1",
                        "raw_column": row["raw_column"],
                        "artifact_value": value,
                    }
                )
    return rows


def join_rows(eval_rows: list[dict[str, Any]], artifact_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in artifact_rows:
        by_key[(row["dataset"], row["condition"])].append(row)
    joined: list[dict[str, Any]] = []
    for erow in eval_rows:
        for arow in by_key.get((erow["dataset"], erow["condition"]), []):
            joined.append({**erow, **arow})
    return joined


def shuffle_p(rows: list[dict[str, Any]], direction: float, observed_signed: float, label: str, n_perm: int = 1000) -> float | None:
    if len(rows) < 20:
        return None
    rng = random.Random(stable_seed(label))
    pp = [float(r["pearson_pert"]) for r in rows]
    values_by_dataset: dict[str, list[float]] = defaultdict(list)
    idx_by_dataset: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        values_by_dataset[row["dataset"]].append(float(row["artifact_value"]))
        idx_by_dataset[row["dataset"]].append(i)
    ge = 0
    valid = 0
    for _ in range(n_perm):
        shuffled = [0.0] * len(rows)
        for ds, vals in values_by_dataset.items():
            vals2 = list(vals)
            rng.shuffle(vals2)
            for idx, value in zip(idx_by_dataset[ds], vals2):
                shuffled[idx] = value
        rho = spearman(shuffled, pp)
        if rho is None:
            continue
        valid += 1
        if direction * rho >= observed_signed - 1e-12:
            ge += 1
    if valid == 0:
        return None
    return (ge + 1) / (valid + 1)


def summarize_artifact(rows: list[dict[str, Any]], artifact: str) -> dict[str, Any]:
    arows = [r for r in rows if r["artifact"] == artifact]
    role = sorted({r["artifact_role"] for r in arows})[0] if arows else ""
    raw_column = sorted({r["raw_column"] for r in arows})[0] if arows else ""
    source_label = sorted({r["source_label"] for r in arows})[0] if arows else ""
    primary = [r for r in arows if r["seed"] == "seed42" and r["group"] == "test_single"]
    if len(primary) < 20:
        direction = 1.0
    else:
        rho0 = spearman([float(r["artifact_value"]) for r in primary], [float(r["pearson_pert"]) for r in primary]) or 0.0
        direction = 1.0 if rho0 >= 0 else -1.0

    parts = []
    for seed in ("seed42", "seed43"):
        for group in ("test_single", "family_gene", "test_all"):
            sub = [r for r in arows if r["seed"] == seed and r["group"] == group]
            vals = [float(r["artifact_value"]) for r in sub]
            pp = [float(r["pearson_pert"]) for r in sub]
            mmd = [float(r["test_mmd_clamped"]) for r in sub]
            rho_pp = spearman(vals, pp)
            rho_mmd = spearman(vals, mmd)
            signed = None if rho_pp is None else direction * rho_pp
            ds_signed = []
            for ds in sorted({r["dataset"] for r in sub}):
                dsub = [r for r in sub if r["dataset"] == ds]
                drho = spearman([float(r["artifact_value"]) for r in dsub], [float(r["pearson_pert"]) for r in dsub])
                if drho is not None:
                    ds_signed.append(direction * drho)
            pval = shuffle_p(sub, direction, signed or -999.0, f"{artifact}:{seed}:{group}") if signed is not None else None
            parts.append(
                {
                    "seed": seed,
                    "group": group,
                    "n": len(sub),
                    "datasets": len({r["dataset"] for r in sub}),
                    "rho_pp": rho_pp,
                    "signed_rho_pp": signed,
                    "rho_mmd": rho_mmd,
                    "dataset_min_signed_rho": min(ds_signed) if ds_signed else None,
                    "shuffle_p": pval,
                }
            )

    key_parts = [p for p in parts if p["group"] in {"test_single", "family_gene"}]
    signed_vals = [p["signed_rho_pp"] for p in key_parts if p["signed_rho_pp"] is not None]
    pvals = [p["shuffle_p"] for p in key_parts if p["shuffle_p"] is not None]
    dsmins = [p["dataset_min_signed_rho"] for p in key_parts if p["dataset_min_signed_rho"] is not None]
    mmd_abs = [abs(p["rho_mmd"]) for p in key_parts if p["rho_mmd"] is not None]
    reasons = []
    if not signed_vals or min(signed_vals) < 0.20:
        reasons.append("min_signed_rho_below_0p20")
    if len(pvals) < len(key_parts):
        reasons.append("missing_shuffle_p_for_one_or_more_key_groups")
    elif max(pvals) > 0.05:
        reasons.append("max_shuffle_p_above_0p05")
    if not dsmins or min(dsmins) < 0.05:
        reasons.append("dataset_min_signed_rho_below_0p05")
    if mmd_abs and signed_vals and max(mmd_abs) >= min(signed_vals):
        reasons.append("mmd_correlation_not_weaker_than_pp_signal")
    key_ns = [int(p["n"]) for p in key_parts]
    key_dataset_counts = [int(p["datasets"]) for p in key_parts]
    if not key_ns or min(key_ns) < 50:
        reasons.append("key_group_n_below_50")
    if not key_dataset_counts or min(key_dataset_counts) < 2:
        reasons.append("key_group_datasets_below_2")
    if not artifact.startswith("replogle_bulk_pooled_"):
        reasons.append("source_specific_diagnostic_only_not_formal_pass")
    status = "pass_needs_external_audit_no_gpu" if not reasons and role == "response_candidate" else "fail_no_gpu"

    out: dict[str, Any] = {
        "artifact": artifact,
        "artifact_role": role,
        "raw_column": raw_column,
        "source_label": source_label,
        "direction_from_seed42_test_single": direction,
        "min_signed_rho_key_groups": min(signed_vals) if signed_vals else None,
        "mean_signed_rho_key_groups": mean(signed_vals) if signed_vals else None,
        "max_shuffle_p_key_groups": max(pvals) if pvals else None,
        "min_dataset_signed_rho_key_groups": min(dsmins) if dsmins else None,
        "max_abs_rho_mmd_key_groups": max(mmd_abs) if mmd_abs else None,
        "status": status,
        "reasons": ";".join(reasons) if reasons else "",
    }
    for part in parts:
        prefix = f"{part['seed']}_{part['group']}"
        for key, value in part.items():
            if key not in {"seed", "group"}:
                out[f"{prefix}_{key}"] = value
    return out


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):+.6f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_rows = load_eval_rows()
    artifact_rows = load_artifacts()
    joined = join_rows(eval_rows, artifact_rows)
    artifacts = sorted({r["artifact"] for r in joined})
    summaries = [summarize_artifact(joined, artifact) for artifact in artifacts]
    summaries.sort(
        key=lambda r: (
            0 if r["artifact_role"] == "response_candidate" else 1,
            -(r["min_signed_rho_key_groups"] or -999.0),
            r["max_shuffle_p_key_groups"] if r["max_shuffle_p_key_groups"] is not None else 999.0,
        )
    )
    pass_candidates = [r["artifact"] for r in summaries if r["status"] == "pass_needs_external_audit_no_gpu"]
    best_response = next((r for r in summaries if r["artifact_role"] == "response_candidate"), None)
    best_qc = next((r for r in summaries if r["artifact_role"] == "qc_control"), None)
    global_reasons = []
    if not pass_candidates:
        global_reasons.append("no_response_candidate_passed_strict_gate")
    if best_response and best_qc and (best_response["min_signed_rho_key_groups"] or -999.0) <= (best_qc["min_signed_rho_key_groups"] or -999.0) + 0.05:
        global_reasons.append("response_candidate_not_clearly_above_qc_control")
    status = "replogle_bulk_artifact_gate_pass_needs_external_audit_no_gpu" if pass_candidates and not global_reasons else "replogle_bulk_artifact_gate_fail_no_gpu"

    join_fields = [
        "seed",
        "source",
        "group",
        "dataset",
        "condition",
        "pearson_pert",
        "test_mmd_clamped",
        "artifact",
        "artifact_value",
        "artifact_role",
        "source_label",
        "raw_column",
    ]
    with OUT_JOINED.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=join_fields)
        writer.writeheader()
        for row in joined:
            writer.writerow({k: row.get(k, "") for k in join_fields})
    summary_fields = list(summaries[0].keys()) if summaries else []
    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_report_only": True,
            "canonical_multi_selection_weight": 0,
            "no_trackc_query": True,
            "no_training": True,
            "no_inference": True,
        },
        "eval_rows": len(eval_rows),
        "artifact_rows": len(artifact_rows),
        "joined_rows": len(joined),
        "artifacts_tested": len(summaries),
        "pass_candidates": pass_candidates,
        "global_reasons": global_reasons,
        "best_response": best_response,
        "best_qc_control": best_qc,
        "outputs": {"joined_csv": str(OUT_JOINED), "summary_csv": str(OUT_SUMMARY), "json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Replogle Bulk Artifact Gate 2026-06-27",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only association gate over Replogle author bulk artifacts and frozen xverse_8k seed42/43 condition metrics.",
        "- Canonical multi is not used for selection; Track C query is not read.",
        "- Passing this gate would require external audit before any GPU smoke.",
        "",
        "## Summary",
        "",
        f"- eval rows: `{len(eval_rows)}`",
        f"- joined rows: `{len(joined)}`",
        f"- artifacts tested: `{len(summaries)}`",
        f"- pass candidates: `{pass_candidates}`",
        f"- global reasons: `{global_reasons}`",
        "",
        "## Top Artifacts",
        "",
        "| artifact | role | min signed rho | max shuffle p | dataset min | max abs rho MMD | status | reasons |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in summaries[:12]:
        lines.append(
            f"| `{row['artifact']}` | `{row['artifact_role']}` | {fmt(row.get('min_signed_rho_key_groups'))} | "
            f"{fmt(row.get('max_shuffle_p_key_groups'))} | {fmt(row.get('min_dataset_signed_rho_key_groups'))} | "
            f"{fmt(row.get('max_abs_rho_mmd_key_groups'))} | `{row['status']}` | `{row['reasons']}` |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "No GPU is authorized from this report. A response artifact must pass strict signal, shuffle, dataset-min, MMD, and QC-control checks plus external audit before GPU design.",
        "",
        f"- joined rows: `{OUT_JOINED}`",
        f"- summary: `{OUT_SUMMARY}`",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "pass_candidates": pass_candidates, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
