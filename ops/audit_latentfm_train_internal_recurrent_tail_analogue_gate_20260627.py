#!/usr/bin/env python3
"""Train/internal recurrent-tail analogue gate.

CPU/report-only. Builds query-blind recurrent-tail analogues from frozen
seed42/seed43 train/internal condition metrics. This asks whether internal tail
severity is stable enough to serve as a future tail-protection benchmark. It
does not create a model, select a checkpoint, read canonical multi for
selection, read Track C query, train, infer, or use GPU.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
SEED42 = REPORTS / "latentfm_xverse_8k_seed_ensemble_internal_means_20260627/seed42_internal_split_group_means_evalseed42.json"
SEED43 = REPORTS / "latentfm_xverse_8k_seed_ensemble_internal_means_20260627/seed43_internal_split_group_means_evalseed42.json"
OUT_DIR = REPORTS / "train_internal_recurrent_tail_analogue_gate_20260627"
OUT_ROWS = OUT_DIR / "train_internal_recurrent_tail_analogue_rows.csv"
OUT_SUMMARY = OUT_DIR / "train_internal_recurrent_tail_analogue_summary.csv"
OUT_JSON = REPORTS / "latentfm_train_internal_recurrent_tail_analogue_gate_20260627.json"
OUT_MD = REPORTS / "LATENTFM_TRAIN_INTERNAL_RECURRENT_TAIL_ANALOGUE_GATE_20260627.md"
RNG_SEED = 20260627


def norm(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def fnum(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(rank(xs), rank(ys)) if len(xs) >= 3 else None


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def read_seed(path: Path) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for group, gdata in data.get("groups", {}).items():
        rows: dict[tuple[str, str], dict[str, Any]] = {}
        for row in gdata.get("condition_metrics", []):
            key = (norm(row.get("dataset")), norm(row.get("condition")))
            if key[0] and key[1]:
                rows[key] = {
                    "dataset": key[0],
                    "condition": key[1],
                    "pearson_pert": fnum(row.get("pearson_pert")),
                    "test_mmd_clamped": fnum(row.get("test_mmd_clamped")),
                    "n_src_eval": fnum(row.get("n_src_eval")),
                    "n_gt_eval": fnum(row.get("n_gt_eval")),
                }
        out[group] = rows
    return out


def build_rows() -> list[dict[str, Any]]:
    s42 = read_seed(SEED42)
    s43 = read_seed(SEED43)
    rows: list[dict[str, Any]] = []
    for group in sorted(set(s42) & set(s43)):
        for key in sorted(set(s42[group]) & set(s43[group])):
            r42 = s42[group][key]
            r43 = s43[group][key]
            pp42 = r42.get("pearson_pert")
            pp43 = r43.get("pearson_pert")
            if pp42 is None or pp43 is None:
                continue
            mmd42 = r42.get("test_mmd_clamped")
            mmd43 = r43.get("test_mmd_clamped")
            mmd_max = max([v for v in [mmd42, mmd43] if v is not None], default=None)
            rows.append(
                {
                    "group": group,
                    "dataset": key[0],
                    "condition": key[1],
                    "pp_seed42": pp42,
                    "pp_seed43": pp43,
                    "bad_pp_seed42": -float(pp42),
                    "bad_pp_seed43": -float(pp43),
                    "bad_pp_mean": -(float(pp42) + float(pp43)) / 2.0,
                    "pp_abs_seed_delta": abs(float(pp42) - float(pp43)),
                    "mmd_seed42": mmd42,
                    "mmd_seed43": mmd43,
                    "mmd_max": mmd_max,
                }
            )
    # Define tail labels within each group using only internal rows.
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
    for group, subset in by_group.items():
        q75 = quantile([float(row["bad_pp_mean"]) for row in subset], 0.75)
        hard_threshold = max(0.5, q75 or 0.5)
        for row in subset:
            row["internal_tail_top_quartile"] = bool(q75 is not None and row["bad_pp_mean"] >= q75)
            row["internal_recurrent_negative_tail"] = bool(row["pp_seed42"] < 0 and row["pp_seed43"] < 0)
            row["internal_recurrent_hard_tail"] = bool(row["bad_pp_mean"] >= hard_threshold and row["pp_seed42"] < 0 and row["pp_seed43"] < 0)
    return rows


def lodo_min(rows: list[dict[str, Any]], x_key: str, y_key: str) -> float | None:
    vals: list[float] = []
    for dataset in sorted({row["dataset"] for row in rows}):
        sub = [row for row in rows if row["dataset"] != dataset]
        xs = [float(row[x_key]) for row in sub if fnum(row.get(x_key)) is not None and fnum(row.get(y_key)) is not None]
        ys = [float(row[y_key]) for row in sub if fnum(row.get(x_key)) is not None and fnum(row.get(y_key)) is not None]
        rho = spearman(xs, ys)
        if rho is not None:
            vals.append(rho)
    return min(vals) if vals else None


def shuffle_p(rows: list[dict[str, Any]], x_key: str, y_key: str, *, n_perm: int = 1000) -> float | None:
    pairs = [(float(row[x_key]), float(row[y_key]), row["dataset"]) for row in rows if fnum(row.get(x_key)) is not None and fnum(row.get(y_key)) is not None]
    if len(pairs) < 10:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    actual = spearman(xs, ys)
    if actual is None:
        return None
    by_dataset: dict[str, list[int]] = defaultdict(list)
    for idx, (_, _, dataset) in enumerate(pairs):
        by_dataset[dataset].append(idx)
    rng = random.Random(RNG_SEED)
    hits = total = 0
    for _ in range(n_perm):
        shuffled = xs[:]
        for idxs in by_dataset.values():
            vals = [shuffled[i] for i in idxs]
            rng.shuffle(vals)
            for i, val in zip(idxs, vals):
                shuffled[i] = val
        rho = spearman(shuffled, ys)
        if rho is None:
            continue
        total += 1
        if rho >= actual:
            hits += 1
    return (hits + 1) / (total + 1) if total else None


def bootstrap_ci_low(rows: list[dict[str, Any]], x_key: str, y_key: str, *, n_boot: int = 1000) -> float | None:
    datasets = sorted({row["dataset"] for row in rows})
    if len(datasets) < 3:
        return None
    by_dataset = {dataset: [row for row in rows if row["dataset"] == dataset] for dataset in datasets}
    rng = random.Random(RNG_SEED + 17)
    vals: list[float] = []
    for _ in range(n_boot):
        sample: list[dict[str, Any]] = []
        for dataset in [rng.choice(datasets) for _ in datasets]:
            pool = by_dataset[dataset]
            sample.extend(pool[rng.randrange(len(pool))] for _ in range(len(pool)))
        xs = [float(row[x_key]) for row in sample if fnum(row.get(x_key)) is not None and fnum(row.get(y_key)) is not None]
        ys = [float(row[y_key]) for row in sample if fnum(row.get(x_key)) is not None and fnum(row.get(y_key)) is not None]
        rho = spearman(xs, ys)
        if rho is not None:
            vals.append(rho)
    if not vals:
        return None
    vals.sort()
    return vals[int(0.025 * (len(vals) - 1))]


def summarize_group(group: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    xs = [float(row["bad_pp_seed42"]) for row in rows]
    ys = [float(row["bad_pp_seed43"]) for row in rows]
    rho = spearman(xs, ys)
    lodo = lodo_min(rows, "bad_pp_seed42", "bad_pp_seed43")
    sp = shuffle_p(rows, "bad_pp_seed42", "bad_pp_seed43")
    ci_low = bootstrap_ci_low(rows, "bad_pp_seed42", "bad_pp_seed43")
    mmd_values = [float(row["mmd_max"]) for row in rows if fnum(row.get("mmd_max")) is not None]
    high_low_mmd = None
    mmd_rho = None
    if mmd_values:
        ordered = sorted(rows, key=lambda row: float(row["bad_pp_mean"]))
        k = max(5, len(ordered) // 4)
        low = [fnum(row.get("mmd_max")) for row in ordered[:k]]
        high = [fnum(row.get("mmd_max")) for row in ordered[-k:]]
        low = [float(v) for v in low if v is not None]
        high = [float(v) for v in high if v is not None]
        if low and high:
            high_low_mmd = mean(high) - mean(low)
        mmd_x = [float(row["bad_pp_mean"]) for row in rows if fnum(row.get("mmd_max")) is not None]
        mmd_y = [float(row["mmd_max"]) for row in rows if fnum(row.get("mmd_max")) is not None]
        mmd_rho = spearman(mmd_x, mmd_y)
    reasons = []
    if len(rows) < 120:
        reasons.append("rows_below_120")
    if len({row["dataset"] for row in rows}) < 12:
        reasons.append("datasets_below_12")
    if rho is None or rho < 0.25:
        reasons.append("seed_stability_rho_below_0p25")
    if lodo is None or lodo < 0.10:
        reasons.append("lodo_min_below_0p10")
    if ci_low is None or ci_low <= 0:
        reasons.append("bootstrap_ci_low_not_above_0")
    if sp is None or sp > 0.01:
        reasons.append("within_dataset_shuffle_p_gt_0p01")
    if high_low_mmd is not None and high_low_mmd > 0.001:
        reasons.append("mmd_high_low_gt_0p001")
    if mmd_rho is not None and abs(mmd_rho) >= 0.15:
        reasons.append("bad_pp_mmd_abs_rho_ge_0p15")
    return {
        "group": group,
        "n": len(rows),
        "datasets": len({row["dataset"] for row in rows}),
        "seed_stability_rho": rho,
        "shuffle_p": sp,
        "lodo_min": lodo,
        "bootstrap_ci_low": ci_low,
        "mmd_high_low": high_low_mmd,
        "bad_pp_mmd_abs_rho": None if mmd_rho is None else abs(mmd_rho),
        "recurrent_negative_tail": sum(bool(row["internal_recurrent_negative_tail"]) for row in rows),
        "recurrent_hard_tail": sum(bool(row["internal_recurrent_hard_tail"]) for row in rows),
        "status": "pass_benchmark_only_no_gpu" if not reasons else "fail_no_gpu",
        "reasons": ";".join(reasons),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    rows = build_rows()
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
    summaries = [summarize_group(group, subset) for group, subset in sorted(by_group.items())]
    passed = [row for row in summaries if row["status"] == "pass_benchmark_only_no_gpu"]
    status = "train_internal_recurrent_tail_analogue_pass_benchmark_only_no_gpu" if passed else "train_internal_recurrent_tail_analogue_fail_no_gpu"

    write_csv(
        OUT_ROWS,
        rows,
        [
            "group",
            "dataset",
            "condition",
            "pp_seed42",
            "pp_seed43",
            "bad_pp_seed42",
            "bad_pp_seed43",
            "bad_pp_mean",
            "pp_abs_seed_delta",
            "mmd_seed42",
            "mmd_seed43",
            "mmd_max",
            "internal_tail_top_quartile",
            "internal_recurrent_negative_tail",
            "internal_recurrent_hard_tail",
        ],
    )
    write_csv(
        OUT_SUMMARY,
        summaries,
        [
            "group",
            "n",
            "datasets",
            "seed_stability_rho",
            "shuffle_p",
            "lodo_min",
            "bootstrap_ci_low",
            "mmd_high_low",
            "bad_pp_mmd_abs_rho",
            "recurrent_negative_tail",
            "recurrent_hard_tail",
            "status",
            "reasons",
        ],
    )
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
        "status": status,
        "gpu_authorized": False,
        "inputs": [str(SEED42), str(SEED43)],
        "summaries": summaries,
        "outputs": {"rows": str(OUT_ROWS), "summary": str(OUT_SUMMARY), "json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = []
    for row in summaries:
        lines.append(
            "| {group} | {n} | {datasets} | {rho} | {sp} | {lodo} | {ci} | {mmd} | {hard} | `{status}` | {reasons} |".format(
                group=row["group"],
                n=row["n"],
                datasets=row["datasets"],
                rho=f"{row['seed_stability_rho']:+.6f}" if isinstance(row["seed_stability_rho"], float) else "NA",
                sp=f"{row['shuffle_p']:.6f}" if isinstance(row["shuffle_p"], float) else "NA",
                lodo=f"{row['lodo_min']:+.6f}" if isinstance(row["lodo_min"], float) else "NA",
                ci=f"{row['bootstrap_ci_low']:+.6f}" if isinstance(row["bootstrap_ci_low"], float) else "NA",
                mmd=f"{row['mmd_high_low']:+.6f}" if isinstance(row["mmd_high_low"], float) else "NA",
                hard=row["recurrent_hard_tail"],
                status=row["status"],
                reasons=row["reasons"],
            )
        )
    md = f"""# LatentFM Train/Internal Recurrent-Tail Analogue Gate 2026-06-27

Timestamp: `{payload['timestamp']}`

Status: `{status}`

GPU authorized: `False`

## Boundary

- CPU/report-only recurrent-tail analogue from frozen seed42/seed43 internal
  validation condition metrics.
- No training, inference, checkpoint selection, canonical multi selection,
  Track C query, or GPU.
- Passing this gate would only define a future benchmark/tail set, not authorize
  training.

## Summary

| group | n | datasets | seed stability rho | shuffle p | LODO min | boot CI low | MMD high-low | hard tails | status | reasons |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
{chr(10).join(lines)}

## Decision

The internal recurrent-tail analogue is benchmark evidence only. GPU remains
unauthorized because this gate does not create a candidate model or no-harm
delta, and any failed status means the tail labels are MMD-confounded or not
stable enough for promotion.

## Outputs

- JSON: `{OUT_JSON}`
- rows: `{OUT_ROWS}`
- summary: `{OUT_SUMMARY}`
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
