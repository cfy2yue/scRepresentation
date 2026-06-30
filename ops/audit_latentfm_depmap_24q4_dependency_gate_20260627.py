#!/usr/bin/env python3
"""Gate DepMap 24Q4 dependency artifacts against frozen xverse outcomes.

CPU/report-only. Reads materialized DepMap matched dependency rows and frozen
seed42/seed43 xverse_8k condition-family metrics. No training, inference,
checkpoint selection, canonical multi selection, Track C query, or GPU.
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
ARTIFACT_CSV = ROOT / "reports/depmap_24q4_dependency_artifacts_20260627/depmap_24q4_dependency_artifacts.csv"
OUT_DIR = ROOT / "reports/depmap_24q4_dependency_gate_20260627"
OUT_ROWS = OUT_DIR / "depmap_24q4_dependency_gate_joined_rows.csv"
OUT_SUMMARY = OUT_DIR / "depmap_24q4_dependency_gate_summary.csv"
OUT_JSON = ROOT / "reports/latentfm_depmap_24q4_dependency_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_DEPMAP_24Q4_DEPENDENCY_GATE_20260627.md"

SEED42_FAMILY = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/"
    / "xverse_comp006_endpoint5_8k_seed42_fulleval/posthoc_eval_uncapped_20260621/"
    / "condition_family_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json"
)
SEED43_FAMILY = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_seed_replicate_20260621/"
    / "xverse_comp006_endpoint5_8k_seed43_fulleval/posthoc_eval_uncapped_20260621/"
    / "condition_family_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json"
)

GROUPS = ["test_single", "family_gene", "test_all"]
PRIMARY_GROUPS = {"test_single", "family_gene"}


def stable_seed(*parts: Any) -> int:
    label = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:12], 16) % (2**32)


def norm(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if text.lower() in {"", "na", "nan", "none", "<na>"}:
        return ""
    return text


def to_float(value: Any) -> float | None:
    text = norm(value)
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


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
    mx = mean(x)
    my = mean(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 0 or vy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(rankdata(x), rankdata(y))


def permutation_p_abs_rho(rows: list[dict[str, Any]], seed: int, n_perm: int = 2000) -> float | None:
    vals = [float(r["artifact_value"]) for r in rows]
    pp = [float(r["pearson_pert"]) for r in rows]
    obs = spearman(vals, pp)
    if obs is None:
        return None
    rng = random.Random(seed)
    by_ds: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_ds[row["dataset"]].append(i)
    ge = 1
    for _ in range(n_perm):
        shuffled = vals[:]
        for idxs in by_ds.values():
            sub = [shuffled[i] for i in idxs]
            rng.shuffle(sub)
            for i, val in zip(idxs, sub):
                shuffled[i] = val
        rho = spearman(shuffled, pp)
        if rho is not None and abs(rho) >= abs(obs):
            ge += 1
    return ge / (n_perm + 1)


def load_artifacts() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ARTIFACT_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            val = to_float(row.get("artifact_value"))
            raw = to_float(row.get("depmap_gene_effect_raw"))
            if val is None:
                continue
            rows.append({**row, "artifact_value": val, "depmap_gene_effect_raw": raw})
    return rows


def load_eval_rows(path: Path, seed: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for group in GROUPS:
        data = payload.get("groups", {}).get(group, {})
        for row in data.get("condition_metrics") or []:
            dataset = norm(row.get("dataset"))
            pp = to_float(row.get("pearson_pert"))
            mmd = to_float(row.get("test_mmd_clamped"))
            if pp is None:
                continue
            out.append(
                {
                    "seed": seed,
                    "group": group,
                    "dataset": dataset,
                    "condition": norm(row.get("condition")),
                    "pearson_pert": pp,
                    "test_mmd_clamped": mmd,
                    "failure_flag": pp < 0.05 or (mmd is not None and mmd > 0.05),
                }
            )
    return out


def join_rows(artifact_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(r["dataset"], r["condition"]): r for r in artifact_rows}
    out = []
    for ev in eval_rows:
        art = by_key.get((ev["dataset"], ev["condition"]))
        if art is None:
            continue
        out.append(
            {
                **ev,
                "split": art["split"],
                "cell_background": art["cell_background"],
                "depmap_model_id": art["depmap_model_id"],
                "target_gene": art["target_gene"],
                "artifact": art["artifact"],
                "artifact_value": float(art["artifact_value"]),
                "depmap_gene_effect_raw": art["depmap_gene_effect_raw"],
            }
        )
    return out


def summarize(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_seed_group[(row["seed"], row["group"])].append(row)
    out = []
    for (seed, group), rows in sorted(by_seed_group.items()):
        vals = [float(r["artifact_value"]) for r in rows]
        pp = [float(r["pearson_pert"]) for r in rows]
        rho = spearman(vals, pp)
        perm_p = permutation_p_abs_rho(rows, seed=stable_seed("depmap", seed, group))
        datasets = sorted({r["dataset"] for r in rows})
        lodo = []
        lodo_details = []
        for ds in datasets:
            sub = [r for r in rows if r["dataset"] != ds]
            sub_rho = spearman([float(r["artifact_value"]) for r in sub], [float(r["pearson_pert"]) for r in sub])
            if sub_rho is not None:
                lodo.append(sub_rho)
                lodo_details.append((ds, sub_rho))
        sign = 1.0 if (rho is not None and rho >= 0) else -1.0
        signed_lodo = [sign * v for v in lodo]
        failure_vals = [float(r["artifact_value"]) for r in rows if r["failure_flag"]]
        ok_vals = [float(r["artifact_value"]) for r in rows if not r["failure_flag"]]
        fail_minus_ok = mean(failure_vals) - mean(ok_vals) if failure_vals and ok_vals else None
        out.append(
            {
                "artifact": "depmap_24q4_dependency_score",
                "seed": seed,
                "group": group,
                "n": len(rows),
                "datasets": len(datasets),
                "varying_datasets": sum(
                    len({round(float(r["artifact_value"]), 8) for r in rows if r["dataset"] == ds}) >= 2
                    for ds in datasets
                ),
                "spearman_artifact_vs_pp": rho,
                "within_dataset_shuffle_p_abs": perm_p,
                "lodo_min_signed_rho": min(signed_lodo) if signed_lodo else None,
                "lodo_rhos": ";".join(f"{ds}:{v:+.4f}" for ds, v in lodo_details),
                "failure_count": len(failure_vals),
                "failure_minus_ok_artifact_mean": fail_minus_ok,
            }
        )
    return out


def decide(summaries: list[dict[str, Any]]) -> tuple[str, list[str], list[dict[str, Any]]]:
    primary = [s for s in summaries if s["group"] in PRIMARY_GROUPS]
    required = {("seed42", "test_single"), ("seed42", "family_gene"), ("seed43", "test_single"), ("seed43", "family_gene")}
    got = {(s["seed"], s["group"]) for s in primary}
    reasons = []
    if not required.issubset(got):
        reasons.append("missing_required_seed_group_rows")
    usable = [
        s
        for s in primary
        if s["n"] >= 100
        and s["datasets"] >= 5
        and s["varying_datasets"] >= 5
        and s["spearman_artifact_vs_pp"] is not None
        and s["within_dataset_shuffle_p_abs"] is not None
    ]
    if len(usable) < 4:
        reasons.append("insufficient_usable_primary_summaries")
    if usable:
        rhos = [float(s["spearman_artifact_vs_pp"]) for s in usable]
        sign = 1.0 if mean(rhos) >= 0 else -1.0
        signed = [sign * r for r in rhos]
        max_p = max(float(s["within_dataset_shuffle_p_abs"]) for s in usable)
        lodo = [float(s["lodo_min_signed_rho"]) for s in usable if s["lodo_min_signed_rho"] is not None]
        if min(signed) < 0.20:
            reasons.append("min_cross_seed_primary_signed_rho_below_0p20")
        if max_p > 0.05:
            reasons.append("within_dataset_shuffle_p_above_0p05")
        if not lodo or min(lodo) < 0.05:
            reasons.append("lodo_min_signed_rho_below_0p05")
    status = "depmap_24q4_dependency_signal_gate_pass_needs_external_audit_no_gpu" if not reasons else "depmap_24q4_dependency_signal_gate_fail_no_gpu"
    return status, reasons, summaries


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not ARTIFACT_CSV.is_file():
        payload = {
            "status": "depmap_24q4_dependency_gate_missing_artifacts_no_gpu",
            "gpu_authorized": False,
            "artifact_csv": str(ARTIFACT_CSV),
        }
        OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 2
    artifacts = load_artifacts()
    eval_rows = load_eval_rows(SEED42_FAMILY, "seed42") + load_eval_rows(SEED43_FAMILY, "seed43")
    joined = join_rows(artifacts, eval_rows)
    summaries = summarize(joined)
    status, reasons, _ = decide(summaries)

    joined_fields = [
        "seed",
        "group",
        "dataset",
        "condition",
        "split",
        "cell_background",
        "depmap_model_id",
        "target_gene",
        "artifact",
        "artifact_value",
        "depmap_gene_effect_raw",
        "pearson_pert",
        "test_mmd_clamped",
        "failure_flag",
    ]
    with OUT_ROWS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=joined_fields)
        writer.writeheader()
        for row in joined:
            writer.writerow({k: row.get(k, "") for k in joined_fields})

    summary_fields = [
        "artifact",
        "seed",
        "group",
        "n",
        "datasets",
        "varying_datasets",
        "spearman_artifact_vs_pp",
        "within_dataset_shuffle_p_abs",
        "lodo_min_signed_rho",
        "lodo_rhos",
        "failure_count",
        "failure_minus_ok_artifact_mean",
    ]
    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({k: row.get(k, "") for k in summary_fields})

    payload = {
        "status": status,
        "gpu_authorized": False,
        "reasons": reasons,
        "artifact_csv": str(ARTIFACT_CSV),
        "joined_rows": len(joined),
        "summary_rows": len(summaries),
        "summaries": summaries,
        "outputs": {"joined_rows": str(OUT_ROWS), "summary_csv": str(OUT_SUMMARY), "json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM DepMap 24Q4 Dependency Gate 2026-06-27",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only association gate for matched DepMap gene dependency artifacts.",
        "- Reads frozen seed42/seed43 xverse_8k condition-family eval JSON only.",
        "- Does not train, infer, select checkpoints, read canonical multi for selection, read Track C query, or use GPU.",
        "",
        "## Gate",
        "",
        "A strict signal candidate must cover at least `100` primary rows, `>=5` datasets, varying values in all covered datasets, consistent seed42/seed43 `test_single` and `family_gene` Spearman signal, within-dataset shuffle `p<=0.05`, and leave-one-dataset signed rho `>=0.05`. Passing still only authorizes external audit/adapter design, not GPU training.",
        "",
        "## Summary",
        "",
        f"- joined rows: `{len(joined)}`",
        f"- summary rows: `{len(summaries)}`",
        f"- reasons: `{', '.join(reasons) or 'none'}`",
        "",
        "| seed | group | n | datasets | rho | shuffle p | min LODO signed rho | failure minus ok artifact |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| `{row['seed']}` | `{row['group']}` | {row['n']} | {row['datasets']} | "
            f"{fmt(row['spearman_artifact_vs_pp'])} | {fmt(row['within_dataset_shuffle_p_abs'])} | "
            f"{fmt(row['lodo_min_signed_rho'])} | {fmt(row['failure_minus_ok_artifact_mean'])} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "If status fails, keep DepMap dependency as negative/control evidence. If status passes, request external audit and design a bounded adapter CPU gate before any GPU smoke.",
        "",
        f"- joined rows CSV: `{OUT_ROWS}`",
        f"- summary CSV: `{OUT_SUMMARY}`",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "joined_rows": len(joined), "reasons": reasons, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
