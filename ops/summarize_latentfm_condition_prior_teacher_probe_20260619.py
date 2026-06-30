#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
COUPLED = ROOT / "CoupledFM"
REPORT = ROOT / "reports/LATENTFM_CONDITION_PRIOR_TEACHER_PROBE_20260619.md"
CSV_OUT = ROOT / "reports/latentfm_condition_prior_teacher_probe_20260619.csv"
JSON_OUT = ROOT / "reports/latentfm_condition_prior_teacher_probe_20260619.json"

RUN_DIR = (
    COUPLED
    / "output/latentfm_runs/condition_prior_teacher_probe_20260619/scf_prior005_e2_4k"
)
POSTHOC = RUN_DIR / "posthoc_eval"

PRIMARY_REFERENCE = {
    "name": "primary_scfoundation_comp006_delta_w5",
    "test_mmd": 0.027124,
    "test_pp": 0.0338,
    "family_gene_pp": 0.0437,
    "family_drug_pp": -0.0082,
    "multi_seen_pp": 0.2112,
    "multi_unseen1_pp": -0.0032,
    "multi_unseen2_pp": -0.1386,
    "resid_cosine": 0.0099,
    "resid_multi_seen_cosine": 0.0708,
    "resid_unseen1_cosine": 0.0006,
    "resid_unseen2_cosine": -0.2744,
}

PRIOR_CORRECTION_JSON = ROOT / "reports/latentfm_prior_correction_eval_20260619.json"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        out = float(value)
        return out if math.isfinite(out) else None
    try:
        out = float(str(value))
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def fmt(value: Any, digits: int = 4) -> str:
    num = fnum(value)
    if num is None:
        return "NA"
    return f"{num:.{digits}f}"


def metric(obj: dict[str, Any] | None, group: str, key: str) -> float | None:
    if not obj:
        return None
    group_obj = obj.get("groups", {}).get(group, {})
    if group_obj.get("skipped"):
        return None
    return fnum(group_obj.get(key))


def iid(obj: dict[str, Any] | None, key: str) -> float | None:
    if not obj:
        return None
    return fnum(obj.get(key))


def residual_group_means(obj: dict[str, Any] | None) -> dict[str, float | None]:
    if not obj:
        return {}
    vals: dict[str, list[float]] = defaultdict(list)
    for row in obj.get("rows", []):
        cos = fnum(row.get("pred_target_cosine"))
        if cos is None:
            continue
        vals["test"].append(cos)
        for group in str(row.get("groups", "")).split(","):
            if group:
                vals[group].append(cos)
    return {key: (sum(values) / len(values) if values else None) for key, values in vals.items()}


def collect_row() -> dict[str, Any]:
    iid_obj = load_json(RUN_DIR / "iid_eval_results.json")
    split = load_json(POSTHOC / "split_group_eval_best_ode20_mse1024_mmd1024.json")
    family = load_json(POSTHOC / "condition_family_eval_best_ode20_mse1024_mmd1024.json")
    residual = load_json(POSTHOC / "condition_residual_full128_best.json")
    rgrp = residual_group_means(residual)
    missing = []
    for name, obj in (("iid", iid_obj), ("split", split), ("family", family), ("residual", residual)):
        if obj is None:
            missing.append(name)
    row = {
        "run": "scf_prior005_e2_4k",
        "backbone": "scfoundation",
        "complete": not missing,
        "missing": ",".join(missing),
        "checkpoint_step": None if split is None else split.get("checkpoint_step"),
        "iid_mmd": iid(iid_obj, "test_mmd"),
        "iid_pc": iid(iid_obj, "pearson_ctrl"),
        "iid_pp": iid(iid_obj, "pearson_pert"),
        "test_mmd": metric(split, "test", "test_mmd"),
        "test_pc": metric(split, "test", "pearson_ctrl"),
        "test_pp": metric(split, "test", "pearson_pert"),
        "multi_seen_pp": metric(split, "test_multi_seen", "pearson_pert"),
        "multi_unseen1_pp": metric(split, "test_multi_unseen1", "pearson_pert"),
        "multi_unseen2_pp": metric(split, "test_multi_unseen2", "pearson_pert"),
        "family_gene_pp": metric(family, "family_gene", "pearson_pert"),
        "family_drug_pp": metric(family, "family_drug", "pearson_pert"),
        "resid_cosine": rgrp.get("test"),
        "resid_multi_seen_cosine": rgrp.get("test_multi_seen"),
        "resid_unseen1_cosine": rgrp.get("test_multi_unseen1"),
        "resid_unseen2_cosine": rgrp.get("test_multi_unseen2"),
        "run_dir": str(RUN_DIR),
    }
    for key in (
        "test_pp",
        "multi_seen_pp",
        "multi_unseen1_pp",
        "multi_unseen2_pp",
        "family_gene_pp",
        "family_drug_pp",
        "resid_cosine",
        "resid_multi_seen_cosine",
        "resid_unseen1_cosine",
        "resid_unseen2_cosine",
    ):
        val = fnum(row.get(key))
        ref = fnum(PRIMARY_REFERENCE.get(key))
        row[f"delta_{key}"] = None if val is None or ref is None else val - ref
    mmd = fnum(row.get("test_mmd"))
    ref_mmd = float(PRIMARY_REFERENCE["test_mmd"])
    row["mmd_ratio_to_primary"] = None if mmd is None else mmd / ref_mmd
    row["decision"] = classify(row)
    return row


def classify(row: dict[str, Any]) -> str:
    if not row.get("complete"):
        return "pending"
    test_pp_gain = fnum(row.get("delta_test_pp"))
    u1_gain = fnum(row.get("delta_multi_unseen1_pp"))
    u2_gain = fnum(row.get("delta_multi_unseen2_pp"))
    seen_gain = fnum(row.get("delta_multi_seen_pp"))
    gene_gain = fnum(row.get("delta_family_gene_pp"))
    ratio = fnum(row.get("mmd_ratio_to_primary"))
    required = (test_pp_gain, u1_gain, u2_gain, seen_gain, gene_gain, ratio)
    if any(value is None for value in required):
        return "needs_manual_review"
    if (
        test_pp_gain > 0.02
        and u1_gain > 0.02
        and u2_gain > 0.02
        and seen_gain > -0.02
        and gene_gain > -0.01
        and ratio <= 1.15
    ):
        return "repeat_candidate"
    if (u1_gain > 0 or u2_gain > 0 or test_pp_gain > 0) and ratio <= 1.25:
        return "diagnostic_candidate"
    return "reject_as_is"


def prior_best_rows() -> list[dict[str, Any]]:
    obj = load_json(PRIOR_CORRECTION_JSON)
    if not obj:
        return []
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in obj.get("summary", []):
        dataset = str(row.get("dataset"))
        group = str(row.get("group"))
        pp = fnum(row.get("pp"))
        if pp is None:
            continue
        key = (dataset, group)
        old = best.get(key)
        if old is None or pp > float(old["pp"]):
            best[key] = row
    return [best[key] for key in sorted(best)]


def write_csv(row: dict[str, Any]) -> None:
    fields = [
        "run",
        "backbone",
        "complete",
        "decision",
        "checkpoint_step",
        "test_mmd",
        "mmd_ratio_to_primary",
        "test_pc",
        "test_pp",
        "delta_test_pp",
        "multi_seen_pp",
        "delta_multi_seen_pp",
        "multi_unseen1_pp",
        "delta_multi_unseen1_pp",
        "multi_unseen2_pp",
        "delta_multi_unseen2_pp",
        "family_gene_pp",
        "delta_family_gene_pp",
        "family_drug_pp",
        "delta_family_drug_pp",
        "resid_cosine",
        "delta_resid_cosine",
        "resid_multi_seen_cosine",
        "delta_resid_multi_seen_cosine",
        "resid_unseen1_cosine",
        "delta_resid_unseen1_cosine",
        "resid_unseen2_cosine",
        "delta_resid_unseen2_cosine",
        "missing",
        "run_dir",
    ]
    with CSV_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: row.get(key) for key in fields})


def main() -> int:
    row = collect_row()
    prior_rows = prior_best_rows()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    write_csv(row)
    status = "complete" if row.get("complete") else "pending"
    lines = [
        "# LatentFM Condition-Prior Teacher Probe 2026-06-19",
        "",
        f"Generated: {datetime.now().strftime('%F %T')}",
        f"Status: `{status}`",
        "",
        "## Purpose",
        "",
        "This report evaluates the first capped scFoundation branch that trains with a train-single condition-prior teacher loss. It tests whether the strong posthoc additive/KNN prior can be internalized by the LatentFM velocity model rather than only applied after inference.",
        "",
        "## Primary Reference",
        "",
        "| Reference | MMD | pp | seen pp | unseen1 pp | unseen2 pp | gene pp | drug pp | resid all | resid seen | resid unseen1 | resid unseen2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| `{PRIMARY_REFERENCE['name']}` | {fmt(PRIMARY_REFERENCE['test_mmd'], 6)} | "
            f"{fmt(PRIMARY_REFERENCE['test_pp'])} | {fmt(PRIMARY_REFERENCE['multi_seen_pp'])} | "
            f"{fmt(PRIMARY_REFERENCE['multi_unseen1_pp'])} | {fmt(PRIMARY_REFERENCE['multi_unseen2_pp'])} | "
            f"{fmt(PRIMARY_REFERENCE['family_gene_pp'])} | {fmt(PRIMARY_REFERENCE['family_drug_pp'])} | "
            f"{fmt(PRIMARY_REFERENCE['resid_cosine'])} | {fmt(PRIMARY_REFERENCE['resid_multi_seen_cosine'])} | "
            f"{fmt(PRIMARY_REFERENCE['resid_unseen1_cosine'])} | {fmt(PRIMARY_REFERENCE['resid_unseen2_cosine'])} |"
        ),
        "",
        "## Candidate Result",
        "",
        "| Run | Complete | decision | step | MMD | MMD/ref | pc | pp | d pp | seen pp | d seen | unseen1 pp | d unseen1 | unseen2 pp | d unseen2 | gene pp | d gene | drug pp | resid all | resid unseen2 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| `{row['run']}` | {row['complete']} | `{row['decision']}` | "
            f"{row.get('checkpoint_step') or 'NA'} | {fmt(row.get('test_mmd'), 6)} | "
            f"{fmt(row.get('mmd_ratio_to_primary'))} | {fmt(row.get('test_pc'))} | "
            f"{fmt(row.get('test_pp'))} | {fmt(row.get('delta_test_pp'))} | "
            f"{fmt(row.get('multi_seen_pp'))} | {fmt(row.get('delta_multi_seen_pp'))} | "
            f"{fmt(row.get('multi_unseen1_pp'))} | {fmt(row.get('delta_multi_unseen1_pp'))} | "
            f"{fmt(row.get('multi_unseen2_pp'))} | {fmt(row.get('delta_multi_unseen2_pp'))} | "
            f"{fmt(row.get('family_gene_pp'))} | {fmt(row.get('delta_family_gene_pp'))} | "
            f"{fmt(row.get('family_drug_pp'))} | {fmt(row.get('resid_cosine'))} | "
            f"{fmt(row.get('resid_unseen2_cosine'))} |"
        ),
        "",
    ]
    if row.get("missing"):
        lines += ["Missing artifacts:", "", f"- `{row['missing']}`", ""]
    if prior_rows:
        lines += [
            "## Posthoc Prior Ceiling",
            "",
            "Best alpha/k rows from the evaluation-only train-single prior diagnostic. These are not training results; they are the target signal the teacher branch is trying to internalize.",
            "",
            "| Dataset | Split | alpha | k | pp | pc | direct | n conditions |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for item in prior_rows:
            lines.append(
                f"| `{item.get('dataset')}` | `{item.get('group')}` | {fmt(item.get('alpha'), 2)} | "
                f"{item.get('k')} | {fmt(item.get('pp'))} | {fmt(item.get('pc'))} | "
                f"{fmt(item.get('direct'))} | {item.get('n_conditions')} |"
            )
        lines.append("")
    lines += [
        "## Interpretation Gate",
        "",
        "- `repeat_candidate`: pp improves on test, unseen1, and unseen2, gene pp is preserved, and MMD stays within 15% of the primary run.",
        "- `diagnostic_candidate`: at least one pp stratum improves without a large MMD penalty, but it is not yet strong enough for promotion.",
        "- `reject_as_is`: no meaningful pp improvement or unacceptable MMD degradation.",
        "",
        f"Current decision: `{row['decision']}`.",
        "",
        "## Outputs",
        "",
        f"- CSV: `{CSV_OUT}`",
        f"- JSON: `{JSON_OUT}`",
        f"- Run dir: `{RUN_DIR}`",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    JSON_OUT.write_text(
        json.dumps(
            {
                "generated": datetime.now().isoformat(timespec="seconds"),
                "status": status,
                "row": row,
                "primary_reference": PRIMARY_REFERENCE,
                "prior_correction_best": prior_rows,
                "report": str(REPORT),
                "csv": str(CSV_OUT),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(REPORT)
    print(CSV_OUT)
    print(JSON_OUT)
    return 0 if row.get("complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
