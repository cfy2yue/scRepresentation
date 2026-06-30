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
REPORT = ROOT / "reports/LATENTFM_STRATEGY_PROBE_EXPANDED_20260619.md"
CSV_OUT = ROOT / "reports/latentfm_strategy_probe_expanded_20260619.csv"
JSON_OUT = ROOT / "reports/latentfm_strategy_probe_expanded_20260619.json"

RUNS = [
    ("scf_e1_comp012", "scfoundation", COUPLED / "output/latentfm_runs/scfoundation_strategy_probe_expanded_20260619", "endpoint_delta=1, composition=0.12"),
    ("scf_head_pert005", "scfoundation", COUPLED / "output/latentfm_runs/scfoundation_strategy_probe_expanded_20260619", "condition-delta head, target=pert_residual, weight=0.05, used in model"),
    ("scf_e3_comp012", "scfoundation", COUPLED / "output/latentfm_runs/scfoundation_strategy_probe_expanded_20260619", "endpoint_delta=3, composition=0.12"),
    ("scf_add005", "scfoundation", COUPLED / "output/latentfm_runs/scfoundation_strategy_probe_expanded_20260619", "additive condition-delta weight=0.05"),
    ("stack_e1_comp012", "stack", COUPLED / "output/latentfm_runs/stack_strategy_probe_expanded_20260619", "endpoint_delta=1, composition=0.12"),
    ("stack_head_pert005", "stack", COUPLED / "output/latentfm_runs/stack_strategy_probe_expanded_20260619", "condition-delta head, target=pert_residual, weight=0.05, used in model"),
    ("stack_e3_comp012", "stack", COUPLED / "output/latentfm_runs/stack_strategy_probe_expanded_20260619", "endpoint_delta=3, composition=0.12"),
    ("stack_e2_comp020", "stack", COUPLED / "output/latentfm_runs/stack_strategy_probe_expanded_20260619", "endpoint_delta=2, composition=0.20"),
]

REFERENCE = {
    "primary_scfoundation": {
        "test_mmd": 0.027124,
        "test_pp": 0.0338,
        "family_gene_pp": 0.0437,
        "family_drug_pp": -0.0082,
        "multi_seen_pp": 0.2112,
        "multi_unseen1_pp": -0.0032,
        "multi_unseen2_pp": -0.1386,
        "resid_cosine": 0.0099,
        "resid_multi_seen": 0.0708,
        "resid_unseen1": 0.0006,
        "resid_unseen2": -0.2744,
    },
    "stack_comp006": {
        "test_mmd": 0.039851,
        "test_pp": 0.0063,
        "family_gene_pp": 0.0133,
        "family_drug_pp": -0.0041,
        "multi_seen_pp": 0.1528,
        "multi_unseen1_pp": 0.0265,
        "multi_unseen2_pp": -0.0656,
        "resid_cosine": 0.0096,
        "resid_multi_seen": 0.1425,
        "resid_unseen1": 0.0258,
        "resid_unseen2": -0.1625,
    },
}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        out = float(value)
        if math.isfinite(out):
            return out
    return None


def metric(obj: dict[str, Any] | None, group: str, key: str) -> float | None:
    if not obj:
        return None
    g = obj.get("groups", {}).get(group, {})
    if g.get("skipped"):
        return None
    return fnum(g.get(key))


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
        groups = [g for g in str(row.get("groups", "")).split(",") if g]
        vals["test"].append(cos)
        for group in groups:
            vals[group].append(cos)
    return {k: (sum(v) / len(v) if v else None) for k, v in vals.items()}


def composite(row: dict[str, Any]) -> float | None:
    test_pp = fnum(row.get("test_pp"))
    test_mmd = fnum(row.get("test_mmd"))
    unseen1 = fnum(row.get("multi_unseen1_pp"))
    unseen2 = fnum(row.get("multi_unseen2_pp"))
    gene = fnum(row.get("family_gene_pp"))
    resid_unseen2 = fnum(row.get("resid_unseen2_cosine"))
    if None in (test_pp, test_mmd, unseen1, unseen2, gene, resid_unseen2):
        return None
    return (
        float(test_pp)
        - 0.5 * float(test_mmd)
        + 0.4 * (float(unseen1) + float(unseen2))
        + 0.2 * float(gene)
        + 0.1 * float(resid_unseen2)
    )


def collect_row(tag: str, backbone: str, base: Path, desc: str) -> dict[str, Any]:
    run_dir = base / tag
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
        "run": tag,
        "backbone": backbone,
        "desc": desc,
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
    row["score"] = composite(row)
    return row


def fmt(value: Any, digits: int = 4) -> str:
    num = fnum(value)
    if num is None:
        return "NA"
    return f"{num:.{digits}f}"


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "run",
        "backbone",
        "complete",
        "checkpoint_step",
        "iid_mmd",
        "iid_pc",
        "iid_pp",
        "test_mmd",
        "test_pc",
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
        "score",
        "desc",
        "missing",
        "run_dir",
    ]
    with CSV_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def main() -> int:
    rows = [collect_row(*spec) for spec in RUNS]
    complete = [r for r in rows if r.get("complete") and r.get("score") is not None]
    best = max(complete, key=lambda r: float(r["score"])) if complete else None

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows)
    JSON_OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM Strategy Probe Expanded Report 2026-06-19",
        "",
        f"Generated: {datetime.now().strftime('%F %T')}",
        "",
        "## Purpose",
        "",
        "Expanded short probes use the user's low-util colocation rule: two additional LatentFM strategies were stacked on each of four physical GPUs, for at most three training jobs per GPU including the earlier active probe. The matrix tests endpoint-delta strength, composition strength, and condition-delta head calibration without reopening the rejected relational residual objective.",
        "",
        "## Reference Anchors",
        "",
        "| Reference | MMD | pp | gene pp | drug pp | seen pp | unseen1 pp | unseen2 pp | resid all | resid seen | resid unseen1 | resid unseen2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, ref in REFERENCE.items():
        lines.append(
            f"| `{name}` | {fmt(ref['test_mmd'], 6)} | {fmt(ref['test_pp'])} | "
            f"{fmt(ref['family_gene_pp'])} | {fmt(ref['family_drug_pp'])} | "
            f"{fmt(ref['multi_seen_pp'])} | {fmt(ref['multi_unseen1_pp'])} | "
            f"{fmt(ref['multi_unseen2_pp'])} | {fmt(ref['resid_cosine'])} | "
            f"{fmt(ref['resid_multi_seen'])} | {fmt(ref['resid_unseen1'])} | "
            f"{fmt(ref['resid_unseen2'])} |"
        )
    lines += [
        "",
        "## Candidate Summary",
        "",
        "| Run | Complete | step | MMD | pc | pp | seen pp | unseen1 pp | unseen2 pp | gene pp | drug pp | resid all | resid seen | resid unseen1 | resid unseen2 | score |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: (r["backbone"], r["run"])):
        lines.append(
            f"| `{row['run']}` | {row['complete']} | {row.get('checkpoint_step') or 'NA'} | "
            f"{fmt(row.get('test_mmd'), 6)} | {fmt(row.get('test_pc'))} | {fmt(row.get('test_pp'))} | "
            f"{fmt(row.get('multi_seen_pp'))} | {fmt(row.get('multi_unseen1_pp'))} | {fmt(row.get('multi_unseen2_pp'))} | "
            f"{fmt(row.get('family_gene_pp'))} | {fmt(row.get('family_drug_pp'))} | "
            f"{fmt(row.get('resid_cosine'))} | {fmt(row.get('resid_multi_seen_cosine'))} | "
            f"{fmt(row.get('resid_unseen1_cosine'))} | {fmt(row.get('resid_unseen2_cosine'))} | "
            f"{fmt(row.get('score'))} |"
        )
    lines += ["", "## Interpretation", ""]
    if best:
        lines.append(
            f"Best composite candidate is `{best['run']}` (`{best['backbone']}`): score={fmt(best.get('score'))}, "
            f"MMD={fmt(best.get('test_mmd'), 6)}, pp={fmt(best.get('test_pp'))}, "
            f"unseen2 pp={fmt(best.get('multi_unseen2_pp'))}, residual unseen2 cosine={fmt(best.get('resid_unseen2_cosine'))}."
        )
    else:
        lines.append("No complete candidate summary yet; inspect missing fields and posthoc logs.")
    missing = [f"{row['run']}:{row['missing']}" for row in rows if row["missing"]]
    if missing:
        lines += ["", "Missing outputs:", "", *[f"- {item}" for item in missing]]
    lines += ["", "CSV/JSON companion files are written next to this report for plotting and downstream selection."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
