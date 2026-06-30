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
REPORT = ROOT / "reports/LATENTFM_CONDITION_PRIOR_INJECTION_COMPARISON_20260619.md"
CSV_OUT = ROOT / "reports/latentfm_condition_prior_injection_comparison_20260619.csv"
JSON_OUT = ROOT / "reports/latentfm_condition_prior_injection_comparison_20260619.json"

RUNS = [
    {
        "run": "scf_prior010_e2_4k",
        "variant": "prior_teacher_no_injection",
        "run_dir": COUPLED
        / "output/latentfm_runs/condition_prior_teacher_probe_20260619/scf_prior010_e2_4k",
        "desc": "condition_prior_delta_loss_weight=0.10; condition_delta_head_use_in_model=False",
    },
    {
        "run": "scf_prior010_inject_e2_4k",
        "variant": "prior_teacher_with_head_injection",
        "run_dir": COUPLED
        / "output/latentfm_runs/condition_prior_teacher_injection_20260619/scf_prior010_inject_e2_4k",
        "desc": "condition_prior_delta_loss_weight=0.10; condition_delta_head_use_in_model=True",
    },
]

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


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        out = float(value)
    else:
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
    for item in obj.get("rows", []):
        cos = fnum(item.get("pred_target_cosine"))
        if cos is None:
            continue
        vals["test"].append(cos)
        for group in str(item.get("groups", "")).split(","):
            if group:
                vals[group].append(cos)
    return {key: (sum(values) / len(values) if values else None) for key, values in vals.items()}


def composite(row: dict[str, Any]) -> float | None:
    test_pp = fnum(row.get("test_pp"))
    mmd_ratio = fnum(row.get("mmd_ratio_to_primary"))
    seen = fnum(row.get("multi_seen_pp"))
    u1 = fnum(row.get("multi_unseen1_pp"))
    u2 = fnum(row.get("multi_unseen2_pp"))
    gene = fnum(row.get("family_gene_pp"))
    resid_u2 = fnum(row.get("resid_unseen2_cosine"))
    if None in (test_pp, mmd_ratio, seen, u1, u2, gene, resid_u2):
        return None
    return (
        float(test_pp)
        + 0.25 * float(seen)
        + 0.55 * (float(u1) + float(u2))
        + 0.20 * float(gene)
        + 0.10 * float(resid_u2)
        - 0.08 * max(0.0, float(mmd_ratio) - 1.0)
    )


def classify(row: dict[str, Any]) -> str:
    if not row.get("complete"):
        return "pending"
    required = [
        row.get("delta_test_pp"),
        row.get("delta_multi_seen_pp"),
        row.get("delta_multi_unseen1_pp"),
        row.get("delta_multi_unseen2_pp"),
        row.get("delta_family_gene_pp"),
        row.get("mmd_ratio_to_primary"),
    ]
    if any(fnum(value) is None for value in required):
        return "needs_manual_review"
    d_test, d_seen, d_u1, d_u2, d_gene, ratio = [float(fnum(v)) for v in required]
    if d_test > 0.02 and d_u1 > 0.02 and d_u2 > 0.02 and d_seen > -0.02 and d_gene > -0.01 and ratio <= 1.15:
        return "repeat_candidate"
    if (d_test > 0 or d_u1 > 0 or d_u2 > 0) and ratio <= 1.25:
        return "diagnostic_candidate"
    return "reject_as_is"


def collect_one(spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(spec["run_dir"])
    posthoc = run_dir / "posthoc_eval"
    iid_obj = load_json(run_dir / "iid_eval_results.json")
    split = load_json(posthoc / "split_group_eval_best_ode20_mse1024_mmd1024.json")
    family = load_json(posthoc / "condition_family_eval_best_ode20_mse1024_mmd1024.json")
    residual = load_json(posthoc / "condition_residual_full128_best.json")
    rgrp = residual_group_means(residual)
    missing = []
    for name, obj in (("iid", iid_obj), ("split", split), ("family", family), ("residual", residual)):
        if obj is None:
            missing.append(name)
    row = {
        "run": spec["run"],
        "variant": spec["variant"],
        "desc": spec["desc"],
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
        "run_dir": str(run_dir),
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
        value = fnum(row.get(key))
        ref = fnum(PRIMARY_REFERENCE.get(key))
        row[f"delta_{key}"] = None if value is None or ref is None else value - ref
    mmd = fnum(row.get("test_mmd"))
    row["mmd_ratio_to_primary"] = None if mmd is None else mmd / float(PRIMARY_REFERENCE["test_mmd"])
    row["score"] = composite(row)
    row["decision"] = classify(row)
    return row


def add_injection_delta(rows: list[dict[str, Any]]) -> None:
    baseline = next((r for r in rows if r["variant"] == "prior_teacher_no_injection"), None)
    if baseline is None or not baseline.get("complete"):
        return
    keys = [
        "test_mmd",
        "test_pc",
        "test_pp",
        "multi_seen_pp",
        "multi_unseen1_pp",
        "multi_unseen2_pp",
        "family_gene_pp",
        "family_drug_pp",
        "resid_cosine",
        "resid_unseen2_cosine",
        "score",
    ]
    for row in rows:
        for key in keys:
            a = fnum(row.get(key))
            b = fnum(baseline.get(key))
            row[f"delta_vs_no_injection_{key}"] = None if a is None or b is None else a - b


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "run",
        "variant",
        "complete",
        "decision",
        "checkpoint_step",
        "test_mmd",
        "mmd_ratio_to_primary",
        "test_pc",
        "test_pp",
        "delta_test_pp",
        "delta_vs_no_injection_test_pp",
        "multi_seen_pp",
        "delta_multi_seen_pp",
        "delta_vs_no_injection_multi_seen_pp",
        "multi_unseen1_pp",
        "delta_multi_unseen1_pp",
        "delta_vs_no_injection_multi_unseen1_pp",
        "multi_unseen2_pp",
        "delta_multi_unseen2_pp",
        "delta_vs_no_injection_multi_unseen2_pp",
        "family_gene_pp",
        "delta_family_gene_pp",
        "family_drug_pp",
        "resid_cosine",
        "resid_unseen2_cosine",
        "delta_resid_unseen2_cosine",
        "score",
        "delta_vs_no_injection_score",
        "missing",
        "run_dir",
    ]
    with CSV_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def main() -> int:
    rows = [collect_one(spec) for spec in RUNS]
    add_injection_delta(rows)
    complete_rows = [row for row in rows if row.get("complete") and fnum(row.get("score")) is not None]
    status = "complete" if len(complete_rows) == len(rows) else "pending"
    best = max(complete_rows, key=lambda row: float(row["score"])) if complete_rows else None
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows)

    lines = [
        "# LatentFM Condition-Prior Injection Comparison 2026-06-19",
        "",
        f"Generated: {datetime.now().strftime('%F %T')}",
        f"Status: `{status}`",
        "",
        "## Purpose",
        "",
        "Compare the best condition-prior teacher dose with and without explicit `condition_delta_head_use_in_model` injection. This tests whether the train-single prior signal must enter the velocity path directly rather than only through an auxiliary prior-delta loss.",
        "",
        "## Primary Reference",
        "",
        "| Reference | MMD | pp | seen pp | unseen1 pp | unseen2 pp | gene pp | drug pp | resid all | resid unseen2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| `{PRIMARY_REFERENCE['name']}` | {fmt(PRIMARY_REFERENCE['test_mmd'], 6)} | "
            f"{fmt(PRIMARY_REFERENCE['test_pp'])} | {fmt(PRIMARY_REFERENCE['multi_seen_pp'])} | "
            f"{fmt(PRIMARY_REFERENCE['multi_unseen1_pp'])} | {fmt(PRIMARY_REFERENCE['multi_unseen2_pp'])} | "
            f"{fmt(PRIMARY_REFERENCE['family_gene_pp'])} | {fmt(PRIMARY_REFERENCE['family_drug_pp'])} | "
            f"{fmt(PRIMARY_REFERENCE['resid_cosine'])} | {fmt(PRIMARY_REFERENCE['resid_unseen2_cosine'])} |"
        ),
        "",
        "## Comparison Table",
        "",
        "| Run | Variant | Complete | Decision | step | MMD | MMD/ref | pc | pp | d pp | d pp vs no-inj | seen pp | unseen1 pp | unseen2 pp | gene pp | drug pp | resid all | resid unseen2 | score | d score vs no-inj |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['run']}` | `{row['variant']}` | {row['complete']} | `{row['decision']}` | "
            f"{row.get('checkpoint_step') or 'NA'} | {fmt(row.get('test_mmd'), 6)} | "
            f"{fmt(row.get('mmd_ratio_to_primary'))} | {fmt(row.get('test_pc'))} | "
            f"{fmt(row.get('test_pp'))} | {fmt(row.get('delta_test_pp'))} | "
            f"{fmt(row.get('delta_vs_no_injection_test_pp'))} | {fmt(row.get('multi_seen_pp'))} | "
            f"{fmt(row.get('multi_unseen1_pp'))} | {fmt(row.get('multi_unseen2_pp'))} | "
            f"{fmt(row.get('family_gene_pp'))} | {fmt(row.get('family_drug_pp'))} | "
            f"{fmt(row.get('resid_cosine'))} | {fmt(row.get('resid_unseen2_cosine'))} | "
            f"{fmt(row.get('score'))} | {fmt(row.get('delta_vs_no_injection_score'))} |"
        )
    lines.append("")
    missing = [f"{row['run']}:{row['missing']}" for row in rows if row.get("missing")]
    if missing:
        lines += ["## Missing Artifacts", ""]
        lines.extend(f"- `{item}`" for item in missing)
        lines.append("")
    if best:
        lines += [
            "## Current Recommendation",
            "",
            f"Best completed branch by the same provisional score is `{best['run']}`.",
            "",
        ]
        if any(row.get("decision") == "repeat_candidate" for row in rows):
            lines.append("At least one branch meets the repeat-candidate gate. Next action: repeat/deepen exactly that branch before any manuscript claim.")
        elif status == "complete":
            lines.append("No branch meets the strict repeat-candidate gate. Treat the injection result as diagnostic and use split-level deltas to decide whether explicit interaction modeling is needed.")
        else:
            lines.append("Injection branch is still pending; do not interpret it until split/family/residual posthoc artifacts are complete.")
        lines.append("")
    lines += [
        "## Gate",
        "",
        "`repeat_candidate` requires improved aggregate pp, unseen1 pp, unseen2 pp, preserved seen/gene pp, and MMD within 15% of the primary scFoundation branch. A positive no-injection delta is useful but not sufficient unless the strict gate is also satisfied.",
        "",
        "## Outputs",
        "",
        f"- CSV: `{CSV_OUT}`",
        f"- JSON: `{JSON_OUT}`",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    JSON_OUT.write_text(
        json.dumps(
            {
                "generated": datetime.now().isoformat(timespec="seconds"),
                "status": status,
                "rows": rows,
                "best": None if best is None else best["run"],
                "primary_reference": PRIMARY_REFERENCE,
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
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
