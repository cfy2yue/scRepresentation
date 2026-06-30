#!/usr/bin/env python3
"""Build and optionally evaluate a SciPlex dose-specific outcome gate.

CPU-only. The builder creates a dose-level diagnostic split from the already
materialized SciPlex dose artifact. If a condition-family eval JSON is present,
it also runs a conservative high-vs-low dose outcome gate over existing model
condition metrics. It does not train, infer, read canonical multi, read Track C
query, or use GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
ARTIFACT_CSV = ROOT / "reports/sciplex_dose_time_artifacts_20260627/sciplex_log_dose_condition_level.csv"
DATA_DIR = ROOT / "runs/latentfm_true_cell_count_allmodality_doseaware_capped_h5_20260625/artifacts/all_modality_doseaware_fixed64_budget16_32_64_budget64_seed42"
OUT_DIR = ROOT / "reports/sciplex_dose_specific_outcome_gate_20260627"
SPLIT_DIR = ROOT / "dataset/biFlow_data/sciplex_dose_specific_splits_20260627"
SPLIT_FILE = SPLIT_DIR / "split_seed42_sciplex_logdose_cap120_all_doseeval.json"
OUT_JSON = ROOT / "reports/latentfm_sciplex_dose_specific_outcome_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_SCIPLEX_DOSE_SPECIFIC_OUTCOME_GATE_20260627.md"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def norm_dose(value: Any) -> str:
    x = float(value)
    if abs(x - round(x)) < 1e-12:
        return f"{x:.1f}"
    return f"{x:g}"


def read_artifact() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ARTIFACT_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ds = row["dataset"]
            bg = row["cell_background"]
            drug = row["drug"]
            dose = float(row["dose"])
            condition = f"{bg}_{drug}_{norm_dose(dose)}"
            rows.append(
                {
                    **row,
                    "dataset": ds,
                    "condition": condition,
                    "drug": drug,
                    "dose": dose,
                    "log_dose": math.log10(dose),
                    "artifact_value": math.log10(dose),
                }
            )
    return rows


def write_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    meta = load_json(DATA_DIR / "condition_metadata.json")
    by_ds: dict[str, list[str]] = defaultdict(list)
    missing: list[dict[str, str]] = []
    for row in rows:
        ds = str(row["dataset"])
        cond = str(row["condition"])
        if cond in meta.get(ds, {}):
            by_ds[ds].append(cond)
        else:
            missing.append({"dataset": ds, "condition": cond})
    split = {
        ds: {
            "train": [],
            "test": sorted(set(conds)),
            "test_single": [],
            "test_multi": [],
            "test_multi_seen": [],
            "test_multi_unseen1": [],
            "test_multi_unseen2": [],
        }
        for ds, conds in sorted(by_ds.items())
    }
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "split_file": str(SPLIT_FILE),
        "split_rows": sum(len(v["test"]) for v in split.values()),
        "datasets": sorted(split),
        "missing_rows": len(missing),
        "missing_examples": missing[:20],
    }


def collect_existing_outcome_readiness(eval_json: Path) -> dict[str, Any] | None:
    if not eval_json.is_file():
        return None
    payload = load_json(eval_json)
    rows = (((payload.get("groups") or {}).get("family_drug") or {}).get("condition_metrics") or [])
    counts: Counter[tuple[str, str]] = Counter()
    dose_counts: Counter[str] = Counter()
    for row in rows:
        ds = str(row.get("dataset"))
        cond = str(row.get("condition"))
        try:
            bg, rest = cond.split("_", 1)
            drug, dose_s = rest.rsplit("_", 1)
            dose = float(dose_s)
        except Exception:
            continue
        counts[(ds, drug)] += 1
        dose_counts[norm_dose(dose)] += 1
    return {
        "eval_json": str(eval_json),
        "family_drug_rows": len(rows),
        "drug_groups": len(counts),
        "drug_group_dose_count_distribution": dict(Counter(counts.values())),
        "multi_dose_drug_groups": sum(1 for v in counts.values() if v >= 2),
        "dose_distribution": dict(dose_counts),
        "readiness_status": "existing_outcomes_underpowered_for_within_drug_dose_gate"
        if counts and max(counts.values()) < 2
        else "existing_outcomes_have_within_drug_dose_variation",
    }


def condition_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["condition"])


def rows_from_eval(eval_json: Path, artifact_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    artifact_by_key = {condition_key(row): row for row in artifact_rows}
    payload = load_json(eval_json)
    metrics = (((payload.get("groups") or {}).get("family_drug") or {}).get("condition_metrics") or [])
    out: list[dict[str, Any]] = []
    for metric in metrics:
        key = (str(metric.get("dataset")), str(metric.get("condition")))
        art = artifact_by_key.get(key)
        if not art:
            continue
        out.append(
            {
                "dataset": key[0],
                "condition": key[1],
                "drug": art["drug"],
                "dose": float(art["dose"]),
                "log_dose": float(art["log_dose"]),
                "pearson_pert": float(metric.get("pearson_pert", "nan")),
                "test_mmd_clamped": float(metric.get("test_mmd_clamped", "nan")),
                "n_src_eval": int(metric.get("n_src_eval", 0) or 0),
                "n_gt_eval": int(metric.get("n_gt_eval", 0) or 0),
            }
        )
    return out


def paired_high_low(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[(row["dataset"], row["drug"])].append(row)
    pairs: list[dict[str, Any]] = []
    for (ds, drug), vals in sorted(by_group.items()):
        finite = [r for r in vals if math.isfinite(r["pearson_pert"]) and math.isfinite(r["test_mmd_clamped"])]
        if len(finite) < 2:
            continue
        finite = sorted(finite, key=lambda r: r["dose"])
        low = finite[0]
        high = finite[-1]
        if high["dose"] <= low["dose"]:
            continue
        pairs.append(
            {
                "dataset": ds,
                "drug": drug,
                "low_condition": low["condition"],
                "high_condition": high["condition"],
                "low_dose": low["dose"],
                "high_dose": high["dose"],
                "delta_log_dose": high["log_dose"] - low["log_dose"],
                "pp_high_minus_low": high["pearson_pert"] - low["pearson_pert"],
                "mmd_high_minus_low": high["test_mmd_clamped"] - low["test_mmd_clamped"],
            }
        )
    return pairs


def bootstrap(values: list[float], *, seed: int = 20260627, n_boot: int = 5000) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None, "p_improve": None}
    rng = random.Random(seed)
    vals = [float(v) for v in values]
    means = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        means.append(mean(sample))
    means.sort()
    return {
        "mean": mean(vals),
        "ci_low": means[int(0.025 * (len(means) - 1))],
        "ci_high": means[int(0.975 * (len(means) - 1))],
        "p_improve": sum(1 for x in means if x > 0.0) / len(means),
    }


def summarize_gate(eval_json: Path, artifact_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = rows_from_eval(eval_json, artifact_rows)
    pairs = paired_high_low(rows)
    pp_vals = [r["pp_high_minus_low"] for r in pairs]
    mmd_vals = [r["mmd_high_minus_low"] for r in pairs]
    pp_boot = bootstrap(pp_vals)
    mmd_boot = bootstrap(mmd_vals)
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        by_ds[row["dataset"]].append(row)
    ds_summary = {
        ds: {
            "n_pairs": len(vals),
            "pp_mean": mean([v["pp_high_minus_low"] for v in vals]),
            "mmd_mean": mean([v["mmd_high_minus_low"] for v in vals]),
        }
        for ds, vals in sorted(by_ds.items())
    }
    reasons = []
    if len(rows) < 50:
        reasons.append("overlap_rows_below_50")
    if len(pairs) < 50:
        reasons.append("within_drug_high_low_pairs_below_50")
    if len(ds_summary) < 3:
        reasons.append("datasets_below_3")
    if pp_boot["ci_low"] is None or pp_boot["ci_low"] <= 0.0:
        reasons.append("pp_bootstrap_lower_not_above_0")
    ds_pp_min = min((x["pp_mean"] for x in ds_summary.values()), default=None)
    if ds_pp_min is None or ds_pp_min < -0.020:
        reasons.append("dataset_min_pp_below_minus_0p020")
    mmd_mean = mmd_boot["mean"]
    ds_mmd_max = max((x["mmd_mean"] for x in ds_summary.values()), default=None)
    if mmd_mean is None or mmd_mean > 0.001 or (ds_mmd_max is not None and ds_mmd_max > 0.001):
        reasons.append("mmd_high_minus_low_above_0p001")
    status = "sciplex_dose_specific_outcome_gate_fail_no_gpu" if reasons else "sciplex_dose_specific_outcome_gate_pass_external_review_next"
    return {
        "eval_json": str(eval_json),
        "status": status,
        "gpu_authorized": False,
        "overlap_rows": len(rows),
        "within_drug_pairs": len(pairs),
        "datasets": sorted(ds_summary),
        "dataset_summary": ds_summary,
        "pp_high_minus_low_bootstrap": pp_boot,
        "mmd_high_minus_low_bootstrap": mmd_boot,
        "reasons": reasons,
    }


def fmt(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v):+.6f}"
    except Exception:
        return str(v)


def write_outputs(payload: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM SciPlex Dose-Specific Outcome Gate 2026-06-27",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"GPU authorized for training: `{False}`",
        "",
        "## Boundary",
        "",
        "- CPU-only split/readiness/gate over materialized SciPlex dose artifact and existing condition metrics if provided.",
        "- Does not train, infer, read canonical multi, read Track C query, select checkpoints, or use GPU.",
        "- Diagnostic posthoc, if launched separately, is anchor-only outcome materialization and remains non-promotional.",
        "",
        "## Split Readiness",
        "",
        f"- artifact rows: `{payload['artifact_rows']}`",
        f"- split rows: `{payload['split']['split_rows']}`",
        f"- split file: `{payload['split']['split_file']}`",
        f"- missing rows: `{payload['split']['missing_rows']}`",
        "",
        "## Existing Outcome Readiness",
        "",
    ]
    readiness = payload.get("existing_outcome_readiness")
    if readiness:
        lines += [
            f"- existing eval JSON: `{readiness['eval_json']}`",
            f"- family_drug rows: `{readiness['family_drug_rows']}`",
            f"- drug groups: `{readiness['drug_groups']}`",
            f"- multi-dose drug groups: `{readiness['multi_dose_drug_groups']}`",
            f"- dose distribution: `{readiness['dose_distribution']}`",
            f"- status: `{readiness['readiness_status']}`",
            "",
        ]
    gate = payload.get("gate")
    if gate:
        lines += [
            "## Gate Result",
            "",
            f"- gate status: `{gate['status']}`",
            f"- overlap rows: `{gate['overlap_rows']}`",
            f"- within-drug high-low pairs: `{gate['within_drug_pairs']}`",
            f"- pp high-low mean: `{fmt(gate['pp_high_minus_low_bootstrap']['mean'])}`",
            f"- pp CI95: `[{fmt(gate['pp_high_minus_low_bootstrap']['ci_low'])}, {fmt(gate['pp_high_minus_low_bootstrap']['ci_high'])}]`",
            f"- MMD high-low mean: `{fmt(gate['mmd_high_minus_low_bootstrap']['mean'])}`",
            f"- reasons: `{gate['reasons']}`",
            "",
            "| dataset | pairs | pp high-low | MMD high-low |",
            "|---|---:|---:|---:|",
        ]
        for ds, row in gate["dataset_summary"].items():
            lines.append(f"| `{ds}` | {row['n_pairs']} | {row['pp_mean']:+.6f} | {row['mmd_mean']:+.6f} |")
        lines.append("")
    lines += [
        "## Decision",
        "",
        payload["decision"],
        "",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-json", type=Path, default=None, help="Optional condition_family_eval JSON with family_drug condition_metrics.")
    ap.add_argument(
        "--existing-eval-json",
        type=Path,
        default=ROOT
        / "runs/latentfm_true_cell_count_allmodality_doseaware_smokes_20260625/xverse_allmod_doseaware_morgan512_budget64_seed42_2500/posthoc_eval_internal/condition_family_eval_anchor_internal_ode20.json",
    )
    args = ap.parse_args()

    artifact_rows = read_artifact()
    split_info = write_split(artifact_rows)
    readiness = collect_existing_outcome_readiness(args.existing_eval_json)
    gate = summarize_gate(args.eval_json, artifact_rows) if args.eval_json and args.eval_json.is_file() else None

    if gate:
        status = gate["status"]
        decision = (
            "Dose-specific outcome gate has valid full outcome rows; use gate reasons above. "
            "No training GPU is authorized unless this gate passes and external review agrees."
        )
    else:
        status = "sciplex_dose_specific_outcome_gate_needs_full_posthoc_no_training_gpu"
        decision = (
            "Existing model outcome rows are not sufficient for a within-drug dose gate because each "
            "dataset+drug appears at only one dose. A bounded anchor-only diagnostic posthoc over the "
            "generated 1440-row split is the next legal artifact-materialization step; it does not "
            "authorize training or checkpoint promotion."
        )
    payload = {
        "status": status,
        "gpu_authorized_for_training": False,
        "posthoc_diagnostic_split_ready": True,
        "artifact_rows": len(artifact_rows),
        "split": split_info,
        "data_dir": str(DATA_DIR),
        "existing_outcome_readiness": readiness,
        "gate": gate,
        "decision": decision,
    }
    write_outputs(payload)
    print(json.dumps({"status": status, "split": str(SPLIT_FILE), "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
