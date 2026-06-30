#!/usr/bin/env python3
"""CPU-only admission-veto gate for failed LatentFM adapter trajectories.

This gate does not train or evaluate a model.  It replays accepted-step logs
from the current lookahead/low-rank adapter experiments and asks whether a
predeclared support/reliability admission rule could have vetoed the low-rank
negative trajectories while preserving a nontrivial, multi-dataset step
portfolio.  It is intentionally conservative: controls that match the signal
close the route.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import h5py
import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
SAFE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
OUT_JSON = ROOT / "reports/latentfm_adapter_support_reliability_admission_gate_20260628.json"
OUT_MD = ROOT / "reports/LATENTFM_ADAPTER_SUPPORT_RELIABILITY_ADMISSION_GATE_20260628.md"
OUT_ROWS = ROOT / "reports/adapter_support_reliability_admission_gate_20260628/step_rows.csv"
OUT_RULES = ROOT / "reports/adapter_support_reliability_admission_gate_20260628/rule_rows.csv"

RUNS = {
    "lowrank_5accepted_internal_fail": {
        "path": ROOT / "runs/latentfm_lowrank_residual_adapter_smoke_20260627/xverse_lowrank_residual_adapter_seed42_5accepted_20260628_0000/train_metrics.csv",
        "role": "lowrank_fail",
        "accepted_target": 5,
    },
    "lowrank_10accepted_internal_fail": {
        "path": ROOT / "runs/latentfm_lowrank_residual_adapter_smoke_20260627/xverse_lowrank_residual_adapter_seed42_10accepted_20260628_0002/train_metrics.csv",
        "role": "lowrank_fail",
        "accepted_target": 10,
    },
    "lowrank_20accepted_internal_fail": {
        "path": ROOT / "runs/latentfm_lowrank_residual_adapter_smoke_20260627/xverse_lowrank_residual_adapter_seed42_20accepted_20260627_2356/train_metrics.csv",
        "role": "lowrank_fail",
        "accepted_target": 20,
    },
    "condition_delta_40accepted_internal_pass_canonical_fail": {
        "path": ROOT / "runs/latentfm_lookahead_trust_region_adapter_smoke_20260627/xverse_lookahead_trust_region_adapter_seed42_40accepted_20260627_2300/train_metrics.csv",
        "role": "viability_reference",
        "accepted_target": 40,
    },
}

RAW_FEATURES = (
    "log_n_gt",
    "log_n_ctrl",
    "response_norm",
    "mean_var",
    "sem_proxy",
    "snr_proxy",
)
COUNTLIKE_FEATURES = {
    "log_n_gt",
    "log_n_ctrl",
    "ds_pct_log_n_gt",
    "ds_pct_log_n_ctrl",
    "support_datasets",
    "log_support_datasets",
    "dataset_train_count",
    "log_dataset_train_count",
}
MAX_CELLS_PER_CONDITION = 128
MAX_TRAIN_NORMALIZER_CONDITIONS = 384
SEED = 20260628


@dataclass(frozen=True)
class Rule:
    name: str
    feature: str
    op: str
    threshold: float
    control: str


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_u64(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def decode(values: np.ndarray) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def sample_indices(n: int, *, limit: int, key: str) -> np.ndarray:
    if n <= limit:
        return np.arange(n)
    rng = np.random.default_rng(stable_u64(key) % (2**32 - 1))
    return np.sort(rng.choice(n, size=limit, replace=False))


def read_condition_features(h5: h5py.File, ds: str, cond: str, cidx: dict[str, int]) -> dict[str, float] | None:
    if cond not in cidx:
        return None
    i = cidx[cond]
    ctrl = h5["ctrl/emb"] if "ctrl/emb" in h5 else h5["ir/emb"]
    gt = h5["gt/emb"]
    ctrl_offsets = np.asarray(h5["ctrl/offsets"] if "ctrl/offsets" in h5 else h5["ir/offsets"])
    gt_offsets = np.asarray(h5["gt/offsets"])
    c0, c1 = int(ctrl_offsets[i]), int(ctrl_offsets[i + 1])
    g0, g1 = int(gt_offsets[i]), int(gt_offsets[i + 1])
    if c1 <= c0 or g1 <= g0:
        return None
    cidxs = sample_indices(c1 - c0, limit=MAX_CELLS_PER_CONDITION, key=f"ctrl|{ds}|{cond}")
    gidxs = sample_indices(g1 - g0, limit=MAX_CELLS_PER_CONDITION, key=f"gt|{ds}|{cond}")
    ctrl_arr = np.asarray(ctrl[c0 + cidxs], dtype=np.float64)
    gt_arr = np.asarray(gt[g0 + gidxs], dtype=np.float64)
    ctrl_mean = ctrl_arr.mean(axis=0)
    gt_mean = gt_arr.mean(axis=0)
    ctrl_var = float(np.mean(np.var(ctrl_arr, axis=0)))
    gt_var = float(np.mean(np.var(gt_arr, axis=0)))
    sem = math.sqrt(ctrl_var / max(1, c1 - c0) + gt_var / max(1, g1 - g0))
    response_norm = float(np.linalg.norm(gt_mean - ctrl_mean))
    return {
        "n_ctrl": float(c1 - c0),
        "n_gt": float(g1 - g0),
        "log_n_ctrl": float(math.log1p(c1 - c0)),
        "log_n_gt": float(math.log1p(g1 - g0)),
        "response_norm": response_norm,
        "mean_var": float((ctrl_var + gt_var) / 2.0),
        "sem_proxy": sem,
        "snr_proxy": float(response_norm / (sem + 1e-8)),
    }


def median_mad(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return 0.0, 1.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, mad if mad > 1e-12 else 1.0


def pct(value: float, vals: list[float]) -> float:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr <= float(value)))


def stable_subset(items: list[str], *, limit: int, salt: str) -> list[str]:
    return sorted(items, key=lambda x: (stable_u64(f"{salt}|{x}"), x))[:limit]


def read_logs() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for run, meta in RUNS.items():
        with Path(meta["path"]).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("accepted", "")).lower() != "true":
                    continue
                item = dict(row)
                item["run"] = run
                item["role"] = meta["role"]
                item["accepted_target"] = int(meta["accepted_target"])
                item["attempt"] = int(float(item["attempt"]))
                item["proj_footprint_mean_l2"] = float(item.get("proj_footprint_mean_l2") or 0.0)
                item["proj_anchor_delta"] = float(item.get("proj_anchor_delta") or 0.0)
                item["proj_task_delta"] = float(item.get("proj_task_delta") or 0.0)
                out.append(item)
    return out


def collect_feature_rows(log_rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, Any]]:
    split = load_json(SAFE_SPLIT)
    needed: dict[str, set[str]] = {}
    for row in log_rows:
        needed.setdefault(row["task_dataset"], set()).add(row["task_condition"])
        needed.setdefault(row["noharm_dataset"], set()).add(row["noharm_condition"])
    support_by_condition: dict[str, set[str]] = {}
    for ds, groups in split.items():
        for cond in groups.get("train", []):
            support_by_condition.setdefault(str(cond), set()).add(str(ds))

    raw: dict[tuple[str, str], dict[str, float]] = {}
    train_samples: dict[str, list[dict[str, float]]] = {}
    for ds, groups in split.items():
        h5_path = DATA_DIR / f"{ds}.h5"
        if not h5_path.is_file():
            continue
        wanted = set(needed.get(ds, set()))
        train_conds = [str(c) for c in groups.get("train", [])]
        wanted.update(stable_subset(train_conds, limit=MAX_TRAIN_NORMALIZER_CONDITIONS, salt=ds))
        with h5py.File(h5_path, "r") as h5:
            cidx = {c: i for i, c in enumerate(decode(np.asarray(h5["conditions"])))}
            for cond in sorted(wanted):
                feats = read_condition_features(h5, str(ds), cond, cidx)
                if feats is None:
                    continue
                feats["support_datasets"] = float(len(support_by_condition.get(cond, set())))
                feats["log_support_datasets"] = float(math.log1p(feats["support_datasets"]))
                feats["dataset_train_count"] = float(len(train_conds))
                feats["log_dataset_train_count"] = float(math.log1p(len(train_conds)))
                raw[(str(ds), cond)] = feats
                if cond in train_conds:
                    train_samples.setdefault(str(ds), []).append(feats)

    enriched: dict[tuple[str, str], dict[str, float]] = {}
    for key, feats in raw.items():
        ds, _cond = key
        ds_train = train_samples.get(ds, [])
        out = dict(feats)
        for feat in RAW_FEATURES:
            vals = [float(x[feat]) for x in ds_train if feat in x]
            med, mad = median_mad(vals)
            out[f"ds_z_{feat}"] = float((float(feats[feat]) - med) / mad)
            out[f"ds_pct_{feat}"] = pct(float(feats[feat]), vals)
        enriched[key] = out

    meta = {
        "split_file": str(SAFE_SPLIT),
        "n_feature_conditions": len(enriched),
        "n_needed_log_conditions": sum(len(v) for v in needed.values()),
        "max_cells_per_condition": MAX_CELLS_PER_CONDITION,
        "max_train_normalizer_conditions_per_dataset": MAX_TRAIN_NORMALIZER_CONDITIONS,
        "feature_provenance": "condition features from safe train-only split H5; dataset normalizers from train conditions only",
    }
    return enriched, meta


def step_features(row: dict[str, Any], cond_features: dict[tuple[str, str], dict[str, float]]) -> dict[str, float]:
    task = cond_features.get((row["task_dataset"], row["task_condition"]), {})
    noharm = cond_features.get((row["noharm_dataset"], row["noharm_condition"]), {})
    feats: dict[str, float] = {}
    keys = sorted(set(task) | set(noharm))
    for key in keys:
        tv = float(task.get(key, float("nan")))
        nv = float(noharm.get(key, float("nan")))
        feats[f"task_{key}"] = tv
        feats[f"noharm_{key}"] = nv
        if np.isfinite(tv) and np.isfinite(nv):
            feats[f"min_{key}"] = min(tv, nv)
            feats[f"max_{key}"] = max(tv, nv)
            feats[f"mean_{key}"] = (tv + nv) / 2.0
    # Composite quality features: high is better.
    feats["quality_snr_count_sem"] = min(
        feats.get("min_ds_pct_snr_proxy", 0.0),
        feats.get("min_ds_pct_log_n_gt", 0.0),
        1.0 - feats.get("max_ds_pct_sem_proxy", 1.0),
    )
    feats["quality_snr_response_sem"] = min(
        feats.get("min_ds_pct_snr_proxy", 0.0),
        feats.get("min_ds_pct_response_norm", 0.0),
        1.0 - feats.get("max_ds_pct_sem_proxy", 1.0),
    )
    return feats


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_step_rows(log_rows: list[dict[str, Any]], cond_features: dict[tuple[str, str], dict[str, float]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in log_rows:
        feats = step_features(row, cond_features)
        out.append({**row, **{f"feat_{k}": v for k, v in feats.items()}})
    return out


def feature_names(step_rows: list[dict[str, Any]], *, mode: str) -> list[str]:
    names = sorted(k.removeprefix("feat_") for row in step_rows for k in row if k.startswith("feat_"))
    numeric = []
    for name in names:
        vals = [float(row.get(f"feat_{name}", float("nan"))) for row in step_rows]
        arr = np.asarray(vals, dtype=np.float64)
        if np.isfinite(arr).sum() >= 8 and float(np.nanmax(arr) - np.nanmin(arr)) > 1e-12:
            numeric.append(name)
    if mode == "count_only":
        return [n for n in numeric if any(token in n for token in COUNTLIKE_FEATURES)]
    if mode == "noncount":
        return [n for n in numeric if not any(token in n for token in COUNTLIKE_FEATURES)]
    if mode == "dataset_only":
        return ["task_dataset", "noharm_dataset"]
    return numeric


def candidate_rules(step_rows: list[dict[str, Any]], *, mode: str) -> list[Rule]:
    rules: list[Rule] = []
    if mode == "dataset_only":
        datasets = sorted({row["task_dataset"] for row in step_rows} | {row["noharm_dataset"] for row in step_rows})
        for ds in datasets:
            rules.append(Rule(f"task_dataset_is_{ds}", f"task_dataset::{ds}", "==", 1.0, mode))
            rules.append(Rule(f"task_dataset_not_{ds}", f"task_dataset::{ds}", "!=", 1.0, mode))
            rules.append(Rule(f"noharm_dataset_is_{ds}", f"noharm_dataset::{ds}", "==", 1.0, mode))
            rules.append(Rule(f"noharm_dataset_not_{ds}", f"noharm_dataset::{ds}", "!=", 1.0, mode))
        return rules
    for feat in feature_names(step_rows, mode=mode):
        vals = np.asarray([float(row.get(f"feat_{feat}", float("nan"))) for row in step_rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size < 8:
            continue
        for q in (0.2, 0.35, 0.5, 0.65, 0.8):
            thr = float(np.quantile(vals, q))
            rules.append(Rule(f"{feat}_ge_q{q:.2f}", feat, ">=", thr, mode))
            rules.append(Rule(f"{feat}_le_q{q:.2f}", feat, "<=", thr, mode))
    return rules


def rule_admits(row: dict[str, Any], rule: Rule) -> bool:
    if "::" in rule.feature:
        field, value = rule.feature.split("::", 1)
        hit = str(row.get(field)) == value
        return hit if rule.op == "==" else not hit
    value = float(row.get(f"feat_{rule.feature}", float("nan")))
    if not np.isfinite(value):
        return False
    return value >= rule.threshold if rule.op == ">=" else value <= rule.threshold


def apply_control(rows: list[dict[str, Any]], control: str) -> list[dict[str, Any]]:
    if control not in {"shuffled", "inverted"}:
        return [dict(r) for r in rows]
    out = [dict(r) for r in rows]
    feat_keys = sorted(k for row in rows for k in row if k.startswith("feat_"))
    if control == "inverted":
        for row in out:
            for key in feat_keys:
                if key in row and np.isfinite(float(row[key])):
                    row[key] = -float(row[key])
        return out
    rng = random.Random(SEED)
    for key in feat_keys:
        vals = [row.get(key, float("nan")) for row in out]
        rng.shuffle(vals)
        for row, val in zip(out, vals):
            row[key] = val
    return out


def summarize_rule(rows: list[dict[str, Any]], rule: Rule) -> dict[str, Any]:
    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_run.setdefault(row["run"], []).append(row)
    run_summaries = []
    for run, rrows in sorted(by_run.items()):
        rrows = sorted(rrows, key=lambda r: int(r["attempt"]))
        admitted = [r for r in rrows if rule_admits(r, rule)]
        veto_attempts = [int(r["attempt"]) for r in rrows if not rule_admits(r, rule)]
        first_veto = min(veto_attempts) if veto_attempts else None
        task_ds = {r["task_dataset"] for r in admitted}
        by_task_ds: dict[str, int] = {}
        for row in admitted:
            by_task_ds[row["task_dataset"]] = by_task_ds.get(row["task_dataset"], 0) + 1
        run_summaries.append(
            {
                "run": run,
                "role": rrows[0]["role"],
                "n_total": len(rrows),
                "n_admitted": len(admitted),
                "accepted_rate": len(admitted) / max(len(rrows), 1),
                "task_dataset_coverage": len(task_ds),
                "first_veto_attempt": first_veto,
                "mean_footprint": float(mean(float(r["proj_footprint_mean_l2"]) for r in admitted)) if admitted else 0.0,
                "max_anchor_delta": float(max((float(r["proj_anchor_delta"]) for r in admitted), default=0.0)),
                "task_dataset_counts": by_task_ds,
            }
        )
    return {
        "rule": rule.name,
        "feature": rule.feature,
        "op": rule.op,
        "threshold": rule.threshold,
        "control": rule.control,
        "runs": run_summaries,
    }


def decision_flags(summary: dict[str, Any]) -> dict[str, Any]:
    runs = {r["run"]: r for r in summary["runs"]}
    lowrank = [r for r in runs.values() if r["role"] == "lowrank_fail"]
    viability = runs.get("condition_delta_40accepted_internal_pass_canonical_fail")
    reasons: list[str] = []
    if not lowrank or viability is None:
        reasons.append("missing_required_run")
    for row in lowrank:
        if row["first_veto_attempt"] is None:
            reasons.append(f"{row['run']}:no_veto_in_observed_prefix")
        elif int(row["first_veto_attempt"]) >= int(row["n_total"]):
            reasons.append(f"{row['run']}:veto_only_at_final_step")
    if viability:
        if not (0.25 <= float(viability["accepted_rate"]) <= 0.80):
            reasons.append("viability_accept_rate_outside_0p25_0p80")
        if int(viability["task_dataset_coverage"]) < 4:
            reasons.append("viability_task_dataset_coverage_below_4")
        if float(viability["mean_footprint"]) <= 1e-7:
            reasons.append("viability_mean_footprint_too_small")
        if float(viability["max_anchor_delta"]) > 1e-6:
            reasons.append("viability_anchor_delta_above_1e-6")
    if summary["control"] == "main" and any(token in summary["feature"] for token in COUNTLIKE_FEATURES):
        reasons.append("main_rule_countlike_feature")
    if summary["control"] == "dataset_only":
        reasons.append("dataset_only_control_not_promotable")
    return {
        "passes_mechanical": not reasons,
        "reasons": reasons,
    }


def score_summary(summary: dict[str, Any]) -> tuple[int, int, float, int, float]:
    flags = decision_flags(summary)
    viability = next((r for r in summary["runs"] if r["role"] == "viability_reference"), {})
    lowrank = [r for r in summary["runs"] if r["role"] == "lowrank_fail"]
    blocked = sum(1 for r in lowrank if r["first_veto_attempt"] is not None and int(r["first_veto_attempt"]) < int(r["n_total"]))
    return (
        1 if flags["passes_mechanical"] else 0,
        blocked,
        float(viability.get("accepted_rate", 0.0)),
        int(viability.get("task_dataset_coverage", 0)),
        float(viability.get("mean_footprint", 0.0)),
    )


def select_best(rows: list[dict[str, Any]], *, mode: str, control_transform: str = "main") -> dict[str, Any]:
    work_rows = apply_control(rows, control_transform)
    rules = candidate_rules(work_rows, mode=mode)
    summaries = [summarize_rule(work_rows, rule) for rule in rules]
    if not summaries:
        return {"control": mode if control_transform == "main" else control_transform, "missing": True}
    best = max(summaries, key=score_summary)
    best["decision_flags"] = decision_flags(best)
    return best


def flatten_rule_summary(summary: dict[str, Any]) -> dict[str, Any]:
    out = {
        "control": summary.get("control"),
        "rule": summary.get("rule"),
        "feature": summary.get("feature"),
        "op": summary.get("op"),
        "threshold": summary.get("threshold"),
        "passes_mechanical": (summary.get("decision_flags") or {}).get("passes_mechanical"),
        "reasons": ";".join((summary.get("decision_flags") or {}).get("reasons") or []),
    }
    for run in summary.get("runs") or []:
        prefix = str(run["run"])
        out[f"{prefix}:accepted_rate"] = run["accepted_rate"]
        out[f"{prefix}:first_veto_attempt"] = run["first_veto_attempt"]
        out[f"{prefix}:task_dataset_coverage"] = run["task_dataset_coverage"]
    return out


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Adapter Support/Reliability Admission Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only replay of existing accepted-step logs.",
        "- Feature provenance uses safe train-only split H5 files and train-only dataset normalizers.",
        "- No training, inference, GPU, canonical multi, canonical single/family selection, or Track C query.",
        "",
        "## Decision",
        "",
        f"- GPU authorized: `{payload['gpu_authorized']}`",
        f"- reasons: `{payload['reasons']}`",
        "",
        "## Best Rules",
        "",
        "| control | rule | feature | pass | reasons |",
        "|---|---|---|---:|---|",
    ]
    for summary in payload["best_summaries"]:
        flags = summary.get("decision_flags") or {}
        lines.append(
            f"| `{summary.get('control')}` | `{summary.get('rule')}` | `{summary.get('feature')}` | "
            f"`{flags.get('passes_mechanical')}` | `{flags.get('reasons')}` |"
        )
    lines.extend(
        [
            "",
            "## Run-Level Replay For Best Main Rule",
            "",
            "| run | role | n total | n admitted | accepted rate | first veto | task ds coverage | mean footprint | max anchor delta |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["best_main"].get("runs") or []:
        lines.append(
            f"| `{row['run']}` | `{row['role']}` | {row['n_total']} | {row['n_admitted']} | "
            f"{row['accepted_rate']:.3f} | {row['first_veto_attempt']} | {row['task_dataset_coverage']} | "
            f"{row['mean_footprint']:.6g} | {row['max_anchor_delta']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- step rows: `{OUT_ROWS}`",
            f"- rule rows: `{OUT_RULES}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    global OUT_JSON
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    args = parser.parse_args()
    OUT_JSON = args.out_json

    log_rows = read_logs()
    cond_features, feature_meta = collect_feature_rows(log_rows)
    step_rows = build_step_rows(log_rows, cond_features)
    write_csv(OUT_ROWS, step_rows)

    best_main = select_best(step_rows, mode="noncount")
    best_all = select_best(step_rows, mode="all")
    best_count = select_best(step_rows, mode="count_only")
    best_dataset = select_best(step_rows, mode="dataset_only")
    best_shuffled = select_best(step_rows, mode="noncount", control_transform="shuffled")
    best_inverted = select_best(step_rows, mode="noncount", control_transform="inverted")
    best_summaries = [best_main, best_all, best_count, best_dataset, best_shuffled, best_inverted]
    write_csv(OUT_RULES, [flatten_rule_summary(s) for s in best_summaries if not s.get("missing")])

    reasons: list[str] = []
    main_flags = best_main.get("decision_flags") or {}
    if not main_flags.get("passes_mechanical"):
        reasons.append("best_noncount_rule_fails_mechanical_gate")
    for label, summary in (("count_only", best_count), ("dataset_only", best_dataset), ("shuffled", best_shuffled), ("inverted", best_inverted)):
        if (summary.get("decision_flags") or {}).get("passes_mechanical"):
            reasons.append(f"{label}_control_also_passes")
    if (best_all.get("decision_flags") or {}).get("passes_mechanical") and best_all.get("feature") != best_main.get("feature"):
        # If the best all-feature rule is count-like, the apparent signal is
        # not cleanly support/reliability beyond count metadata.
        if any(token in str(best_all.get("feature")) for token in COUNTLIKE_FEATURES):
            reasons.append("best_all_rule_is_countlike")

    status = "adapter_support_reliability_admission_gate_pass_gpu_smoke_candidate" if not reasons else "adapter_support_reliability_admission_gate_fail_no_gpu"
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
        "status": status,
        "gpu_authorized": status.endswith("pass_gpu_smoke_candidate"),
        "reasons": reasons,
        "boundary": {
            "safe_split": str(SAFE_SPLIT),
            "cpu_only": True,
            "trains_model": False,
            "runs_inference": False,
            "uses_gpu": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
        },
        "feature_meta": feature_meta,
        "n_step_rows": len(step_rows),
        "best_main": best_main,
        "best_summaries": best_summaries,
        "outputs": {
            "json": str(OUT_JSON),
            "md": str(OUT_MD),
            "step_rows": str(OUT_ROWS),
            "rule_rows": str(OUT_RULES),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "report": str(OUT_MD), "reasons": reasons}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
