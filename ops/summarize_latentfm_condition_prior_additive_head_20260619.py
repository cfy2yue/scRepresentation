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
REPORT = ROOT / "reports/LATENTFM_CONDITION_PRIOR_ADDITIVE_HEAD_COMPARISON_20260619.md"
CSV_OUT = ROOT / "reports/latentfm_condition_prior_additive_head_comparison_20260619.csv"
JSON_OUT = ROOT / "reports/latentfm_condition_prior_additive_head_comparison_20260619.json"

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
    {
        "run": "scf_prioradd005_prior010_inject_e2_4k",
        "variant": "prior_additive_head_supervision",
        "run_dir": COUPLED
        / "output/latentfm_runs/condition_prior_additive_head_20260619/scf_prioradd005_prior010_inject_e2_4k",
        "desc": "prior_delta=0.10; injected head=True; prior_additive_delta=0.05",
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
    "resid_unseen2_cosine": -0.2744,
}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
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


def decomp_group_map(obj: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not obj:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for item in obj.get("summary", []):
        group = str(item.get("group", ""))
        dataset = str(item.get("dataset", ""))
        if group:
            out[f"{dataset}:{group}"] = item
            out.setdefault(group, item)
    return out


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
    decomp = load_json(posthoc / "condition_delta_decomposition_full128_best.json")
    rgrp = residual_group_means(residual)
    dgrp = decomp_group_map(decomp)
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
        "resid_unseen2_cosine": rgrp.get("test_multi_unseen2"),
        "decomp_present": decomp is not None,
        "decomp_wessels_unseen2_combo_additive_cosine": fnum(
            dgrp.get("Wessels_2023:test_multi_unseen2", {}).get("mean_combo_additive_cosine")
        ),
        "decomp_wessels_unseen2_additive_norm_ratio": fnum(
            dgrp.get("Wessels_2023:test_multi_unseen2", {}).get("mean_additive_norm_ratio")
        ),
        "decomp_wessels_unseen2_interaction_norm_ratio": fnum(
            dgrp.get("Wessels_2023:test_multi_unseen2", {}).get("mean_interaction_norm_ratio")
        ),
        "run_dir": str(run_dir),
    }
    for key in (
        "test_pp",
        "multi_seen_pp",
        "multi_unseen1_pp",
        "multi_unseen2_pp",
        "family_gene_pp",
        "family_drug_pp",
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


def add_baseline_deltas(rows: list[dict[str, Any]]) -> None:
    baseline = next((r for r in rows if r["variant"] == "prior_teacher_with_head_injection"), None)
    if baseline is None or not baseline.get("complete"):
        return
    for row in rows:
        for key in (
            "test_mmd",
            "test_pp",
            "multi_seen_pp",
            "multi_unseen1_pp",
            "multi_unseen2_pp",
            "family_gene_pp",
            "family_drug_pp",
            "resid_unseen2_cosine",
            "score",
        ):
            a = fnum(row.get(key))
            b = fnum(baseline.get(key))
            row[f"delta_vs_injection_{key}"] = None if a is None or b is None else a - b


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "run",
        "variant",
        "complete",
        "decision",
        "checkpoint_step",
        "test_mmd",
        "mmd_ratio_to_primary",
        "test_pp",
        "delta_test_pp",
        "delta_vs_injection_test_pp",
        "multi_seen_pp",
        "multi_unseen1_pp",
        "multi_unseen2_pp",
        "delta_vs_injection_multi_unseen2_pp",
        "family_gene_pp",
        "family_drug_pp",
        "resid_unseen2_cosine",
        "score",
        "delta_vs_injection_score",
        "decomp_present",
        "decomp_wessels_unseen2_combo_additive_cosine",
        "decomp_wessels_unseen2_additive_norm_ratio",
        "decomp_wessels_unseen2_interaction_norm_ratio",
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
    add_baseline_deltas(rows)
    complete_rows = [row for row in rows if row.get("complete") and fnum(row.get("score")) is not None]
    status = "complete" if len(complete_rows) == len(rows) else "pending"
    best = max(complete_rows, key=lambda row: float(row["score"])) if complete_rows else None
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows)

    lines = [
        "# LatentFM Condition-Prior Additive Head Comparison 2026-06-19",
        "",
        f"Generated: {datetime.now().strftime('%F %T')}",
        f"Status: `{status}`",
        "",
        "## Purpose",
        "",
        "Test whether direct supervision of the additive condition-delta atom head can improve the injected condition-prior branch without increasing ODE/MMD cost. The branch keeps the prior-teacher and head-injection settings fixed, adding only `condition_prior_additive_delta_loss_weight=0.05`.",
        "",
        "## Comparison Table",
        "",
        "| Run | Variant | Complete | Decision | step | MMD | MMD/ref | pp | d pp | d pp vs inject | seen pp | unseen1 pp | unseen2 pp | d unseen2 vs inject | gene pp | drug pp | resid unseen2 | score | d score vs inject | decomp Wessels unseen2 combo/add | add norm | int norm |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['run']}` | `{row['variant']}` | {row['complete']} | `{row['decision']}` | "
            f"{row.get('checkpoint_step') or 'NA'} | {fmt(row.get('test_mmd'), 6)} | "
            f"{fmt(row.get('mmd_ratio_to_primary'))} | {fmt(row.get('test_pp'))} | "
            f"{fmt(row.get('delta_test_pp'))} | {fmt(row.get('delta_vs_injection_test_pp'))} | "
            f"{fmt(row.get('multi_seen_pp'))} | {fmt(row.get('multi_unseen1_pp'))} | "
            f"{fmt(row.get('multi_unseen2_pp'))} | {fmt(row.get('delta_vs_injection_multi_unseen2_pp'))} | "
            f"{fmt(row.get('family_gene_pp'))} | {fmt(row.get('family_drug_pp'))} | "
            f"{fmt(row.get('resid_unseen2_cosine'))} | {fmt(row.get('score'))} | "
            f"{fmt(row.get('delta_vs_injection_score'))} | "
            f"{fmt(row.get('decomp_wessels_unseen2_combo_additive_cosine'))} | "
            f"{fmt(row.get('decomp_wessels_unseen2_additive_norm_ratio'))} | "
            f"{fmt(row.get('decomp_wessels_unseen2_interaction_norm_ratio'))} |"
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
            lines.append("At least one branch meets the repeat-candidate gate. Repeat/deepen that exact branch before any manuscript claim.")
        elif status == "complete":
            lines.append("No branch meets the strict repeat-candidate gate. Use the additive-head deltas and decomposition metrics to decide whether to abandon or deepen this mechanism.")
        else:
            lines.append("Additive-head branch is pending; do not interpret it until split/family/residual/decomposition artifacts are complete.")
        lines.append("")
    lines += [
        "## Gate",
        "",
        "`repeat_candidate` requires improved aggregate pp, unseen1 pp, unseen2 pp, preserved seen/gene pp, and MMD within 15% of the primary scFoundation branch.",
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
