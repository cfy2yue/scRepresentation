#!/usr/bin/env python3
"""Gate Jiang author-DE artifacts against frozen xverse anchor outcomes.

CPU/report-only. Uses frozen seed42/seed43 xverse_8k anchor condition metrics
and materialized Jiang author-DE condition artifacts. No training, inference,
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
ARTIFACT_CSV = ROOT / "reports/jiang_author_de_artifacts_20260627/jiang_author_de_condition_aggregate_artifacts.csv"
OUT_DIR = ROOT / "reports/jiang_author_de_artifact_gate_20260627"
OUT_ROWS = OUT_DIR / "jiang_author_de_artifact_joined_rows.csv"
OUT_SUMMARY = OUT_DIR / "jiang_author_de_artifact_gate_summary.csv"
OUT_JSON = ROOT / "reports/latentfm_jiang_author_de_artifact_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_JIANG_AUTHOR_DE_ARTIFACT_GATE_20260627.md"

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
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / math.sqrt(vx * vy)


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(rankdata(x), rankdata(y))


def permutation_p_abs_rho(rows: list[dict[str, Any]], seed: int, n_perm: int = 1000) -> float | None:
    vals = [float(r["artifact_value"]) for r in rows]
    pp = [float(r["pearson_pert"]) for r in rows]
    obs = spearman(vals, pp)
    if obs is None:
        return None
    rng = random.Random(seed)
    ge = 1
    datasets = sorted({r["dataset"] for r in rows})
    by_ds = {ds: [i for i, r in enumerate(rows) if r["dataset"] == ds] for ds in datasets}
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


def bootstrap_ci(values: list[float], seed: int, n_boot: int = 1000) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(seed)
    samples = []
    n = len(values)
    for _ in range(n_boot):
        samples.append(sum(rng.choice(values) for _ in range(n)) / n)
    samples.sort()
    return samples[int(0.025 * (n_boot - 1))], samples[int(0.975 * (n_boot - 1))]


def load_artifacts() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ARTIFACT_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = to_float(row.get("artifact_value"))
            if value is None:
                continue
            rows.append({**row, "artifact_value": value})
    return rows


def load_eval_rows(path: Path, seed: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for group in GROUPS:
        data = payload.get("groups", {}).get(group, {})
        for row in data.get("condition_metrics") or []:
            dataset = norm(row.get("dataset"))
            if not dataset.startswith("Jiang_"):
                continue
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
    artifacts_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in artifact_rows:
        artifacts_by_key[(row["dataset"], row["condition"])].append(row)
    out: list[dict[str, Any]] = []
    for ev in eval_rows:
        for art in artifacts_by_key.get((ev["dataset"], ev["condition"]), []):
            out.append(
                {
                    **ev,
                    "split": art["split"],
                    "artifact": art["artifact"],
                    "artifact_metric": art["artifact_metric"],
                    "aggregation": art["aggregation"],
                    "artifact_value": float(art["artifact_value"]),
                    "background_count": art["background_count"],
                }
            )
    return out


def summarize(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_artifact_seed_group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_artifact_seed_group[(row["artifact"], row["seed"], row["group"])].append(row)

    summaries: list[dict[str, Any]] = []
    for (artifact, seed, group), rows in sorted(by_artifact_seed_group.items()):
        vals = [float(r["artifact_value"]) for r in rows]
        pp = [float(r["pearson_pert"]) for r in rows]
        rho = spearman(vals, pp)
        perm_p = permutation_p_abs_rho(rows, seed=stable_seed("perm", artifact, seed, group))
        datasets = sorted({r["dataset"] for r in rows})
        lodo = []
        lodo_details = []
        for ds in datasets:
            sub = [r for r in rows if r["dataset"] != ds]
            sub_rho = spearman([float(r["artifact_value"]) for r in sub], [float(r["pearson_pert"]) for r in sub])
            if sub_rho is not None:
                lodo.append(sub_rho)
                lodo_details.append((ds, sub_rho))
        failure_values = [float(r["artifact_value"]) for r in rows if r["failure_flag"]]
        ok_values = [float(r["artifact_value"]) for r in rows if not r["failure_flag"]]
        failure_minus_ok = None
        fail_ci_low = None
        fail_ci_high = None
        if failure_values and ok_values:
            failure_minus_ok = mean(failure_values) - mean(ok_values)
            boot_vals = []
            rng = random.Random(stable_seed("boot", artifact, seed, group))
            for _ in range(1000):
                boot_vals.append(
                    mean(rng.choice(failure_values) for _ in failure_values)
                    - mean(rng.choice(ok_values) for _ in ok_values)
                )
            boot_vals.sort()
            fail_ci_low = boot_vals[24]
            fail_ci_high = boot_vals[974]
        signed_lodo = []
        if rho is not None:
            sign = 1.0 if rho >= 0 else -1.0
            signed_lodo = [sign * v for v in lodo]
        summaries.append(
            {
                "artifact": artifact,
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
                "failure_count": len(failure_values),
                "failure_minus_ok_artifact_mean": failure_minus_ok,
                "failure_minus_ok_ci_low": fail_ci_low,
                "failure_minus_ok_ci_high": fail_ci_high,
            }
        )
    return summaries


def candidate_artifact_decisions(summaries: list[dict[str, Any]]) -> tuple[str, bool, list[dict[str, Any]], list[str]]:
    primary = [s for s in summaries if s["group"] in PRIMARY_GROUPS]
    by_artifact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primary:
        by_artifact[row["artifact"]].append(row)
    candidates = []
    for artifact, rows in by_artifact.items():
        usable = [
            r
            for r in rows
            if r["n"] >= 80
            and r["datasets"] >= 5
            and r["varying_datasets"] >= 5
            and r["spearman_artifact_vs_pp"] is not None
            and r["within_dataset_shuffle_p_abs"] is not None
        ]
        seed_group = {(r["seed"], r["group"]) for r in usable}
        if not {("seed42", "test_single"), ("seed42", "family_gene"), ("seed43", "test_single"), ("seed43", "family_gene")}.issubset(seed_group):
            continue
        rhos = [float(r["spearman_artifact_vs_pp"]) for r in usable]
        sign = 1.0 if mean(rhos) >= 0 else -1.0
        signed = [sign * v for v in rhos]
        ps = [float(r["within_dataset_shuffle_p_abs"]) for r in usable]
        lodo = [r["lodo_min_signed_rho"] for r in usable if r["lodo_min_signed_rho"] is not None]
        score = min(signed) if signed else None
        pass_like = (
            score is not None
            and score >= 0.20
            and max(ps) <= 0.05
            and lodo
            and min(float(v) for v in lodo) >= 0.05
        )
        candidates.append(
            {
                "artifact": artifact,
                "mean_signed_rho": mean(signed) if signed else None,
                "min_signed_rho": score,
                "max_shuffle_p": max(ps) if ps else None,
                "min_lodo_signed_rho": min(float(v) for v in lodo) if lodo else None,
                "pass_strict_signal_gate": pass_like,
            }
        )
    candidates.sort(
        key=lambda r: (
            bool(r["pass_strict_signal_gate"]),
            r["min_signed_rho"] if r["min_signed_rho"] is not None else -999,
            -(r["max_shuffle_p"] if r["max_shuffle_p"] is not None else 999),
        ),
        reverse=True,
    )
    passing = [c for c in candidates if c["pass_strict_signal_gate"]]
    reasons = []
    if not passing:
        reasons.append("no_artifact_passed_cross_seed_primary_shuffle_lodo_gate")
    status = "jiang_author_de_signal_gate_pass_needs_external_audit_no_gpu" if passing else "jiang_author_de_signal_gate_fail_no_gpu"
    return status, False, candidates[:10], reasons


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):+.6f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = load_artifacts()
    eval_rows = load_eval_rows(SEED42_FAMILY, "seed42") + load_eval_rows(SEED43_FAMILY, "seed43")
    joined = join_rows(artifacts, eval_rows)
    summaries = summarize(joined)
    status, gpu_authorized, top_candidates, reasons = candidate_artifact_decisions(summaries)

    joined_fields = [
        "seed",
        "group",
        "dataset",
        "condition",
        "split",
        "artifact",
        "artifact_metric",
        "aggregation",
        "artifact_value",
        "pearson_pert",
        "test_mmd_clamped",
        "failure_flag",
        "background_count",
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
        "failure_minus_ok_ci_low",
        "failure_minus_ok_ci_high",
    ]
    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({k: row.get(k, "") for k in summary_fields})

    payload = {
        "status": status,
        "gpu_authorized": gpu_authorized,
        "reasons": reasons,
        "artifact_csv": str(ARTIFACT_CSV),
        "seed42_eval": str(SEED42_FAMILY),
        "seed43_eval": str(SEED43_FAMILY),
        "joined_rows": len(joined),
        "summary_rows": len(summaries),
        "top_candidates": top_candidates,
        "outputs": {
            "joined_rows": str(OUT_ROWS),
            "summary_csv": str(OUT_SUMMARY),
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Jiang Author-DE Artifact Gate 2026-06-27",
        "",
        f"Status: `{status}`",
        "",
        f"GPU authorized: `{gpu_authorized}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only association gate for materialized Jiang author-DE artifacts.",
        "- Reads frozen xverse_8k seed42/seed43 condition-family eval JSON only.",
        "- Uses Jiang `test_single`, `family_gene`, and `test_all`; canonical multi has selection weight 0 and is not read here.",
        "- Does not train, infer, select checkpoints, read Track C query, or use GPU.",
        "",
        "## Gate",
        "",
        "A strict signal candidate must have `n>=80`, `datasets>=5`, varying artifact values in all five Jiang datasets, consistent seed42/seed43 `test_single` and `family_gene` Spearman signal, within-dataset shuffle `p<=0.05`, and leave-one-dataset signed rho `>=0.05`. Passing this gate still only authorizes external audit/adapter design, not GPU training.",
        "",
        "## Summary",
        "",
        f"- joined rows: `{len(joined)}`",
        f"- summary rows: `{len(summaries)}`",
        f"- reasons: `{', '.join(reasons) or 'none'}`",
        "",
        "| artifact | pass | min signed rho | mean signed rho | max shuffle p | min LODO signed rho |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in top_candidates[:10]:
        lines.append(
            f"| `{row['artifact']}` | `{row['pass_strict_signal_gate']}` | "
            f"{fmt(row['min_signed_rho'])} | {fmt(row['mean_signed_rho'])} | "
            f"{fmt(row['max_shuffle_p'])} | {fmt(row['min_lodo_signed_rho'])} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        (
            "If status fails, close Jiang author-DE as a direct artifact gate and keep it only as failure-anatomy evidence. "
            "If status passes, request external audit and design a bounded adapter CPU gate before any GPU smoke."
        ),
        "",
        f"- joined rows CSV: `{OUT_ROWS}`",
        f"- summary CSV: `{OUT_SUMMARY}`",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "joined_rows": len(joined), "top": top_candidates[:3], "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
