#!/usr/bin/env python3
"""Summarize the current xverse LatentFM stage evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORT_MD = ROOT / "reports/LATENTFM_XVERSE_STAGE_SUMMARY_20260621.md"
REPORT_JSON = ROOT / "reports/latentfm_xverse_stage_summary_20260621.json"

XVERSE_RUN = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/"
    / "xverse_comp006_endpoint5_8k_seed42_fulleval"
)
XVERSE_2K = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_smoke_20260620/"
    / "xverse_comp006_endpoint5_2k_smoke"
)
SCF_ANCHOR = (
    ROOT
    / "CoupledFM/output/latentfm_runs/condition_prior_teacher_injection_20260619/"
    / "scf_prior010_inject_e2_4k"
)

PAIRED_SPLIT = ROOT / "reports/latentfm_xverse_8k_vs_2k_stablecaps_bootstrap_20260621.split.json"
PAIRED_FAMILY = ROOT / "reports/latentfm_xverse_8k_vs_2k_stablecaps_bootstrap_20260621.family.json"
UNCAPPED_SPLIT_CI = ROOT / "reports/latentfm_xverse_8k_condition_uncapped_split_ci_20260621.json"
UNCAPPED_FAMILY_CI = ROOT / "reports/latentfm_xverse_8k_condition_uncapped_family_ci_20260621.json"
IID_FULL_CI = ROOT / "reports/latentfm_xverse_8k_iid_full_ci_20260621.json"
XVERSE_2K_UNCAPPED_SPLIT_CI = ROOT / "reports/latentfm_xverse_2k_condition_uncapped_split_ci_20260621.json"
XVERSE_2K_UNCAPPED_FAMILY_CI = ROOT / "reports/latentfm_xverse_2k_condition_uncapped_family_ci_20260621.json"
UNCAPPED_PAIRED_SPLIT = ROOT / "reports/latentfm_xverse_8k_vs_2k_condition_uncapped_bootstrap_split_20260621.json"
UNCAPPED_PAIRED_FAMILY = ROOT / "reports/latentfm_xverse_8k_vs_2k_condition_uncapped_bootstrap_family_20260621.json"
SEED43_UNCAPPED_SPLIT_CI = ROOT / "reports/LATENTFM_XVERSE_8K_SEED43_condition_uncapped_split_ci_20260621.json"
SEED43_UNCAPPED_FAMILY_CI = ROOT / "reports/LATENTFM_XVERSE_8K_SEED43_condition_uncapped_family_ci_20260621.json"
SEED43_IID_FULL_CI = ROOT / "reports/latentfm_xverse_8k_seed43_iid_full_ci_20260621.json"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_group_metric(payload: dict[str, Any] | None, group: str, metric: str) -> dict[str, Any] | None:
    if payload is None:
        return None
    for row in payload.get("rows", []):
        if row.get("group") == group and row.get("metric") == metric:
            return row
    return None


def get_ci_metric(payload: dict[str, Any] | None, group: str, metric: str) -> dict[str, Any] | None:
    if payload is None:
        return None
    for grp in payload.get("groups", []):
        if grp.get("group") != group:
            continue
        for row in grp.get("metrics", []):
            if row.get("metric") == metric:
                return row
    return None


def group_metric_from_eval(path: Path, group: str, metric: str) -> float | None:
    payload = load_json(path)
    if payload is None:
        return None
    obj = (payload.get("groups", {}).get(group, {}) or {})
    val = obj.get(metric)
    try:
        return None if val is None else float(val)
    except (TypeError, ValueError):
        return None


def metric_from_iid(path: Path, metric: str) -> float | None:
    payload = load_json(path)
    if payload is None:
        return None
    val = payload.get(metric)
    try:
        return None if val is None else float(val)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "pending"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{f:+.6f}" if signed else f"{f:.6f}"


def ci(row: dict[str, Any] | None, *, delta: bool = False) -> str:
    if row is None:
        return "pending"
    lo = row.get("ci95_low")
    hi = row.get("ci95_high")
    return f"[{fmt(lo, signed=delta)}, {fmt(hi, signed=delta)}]"


def build_payload() -> dict[str, Any]:
    paired_split = load_json(PAIRED_SPLIT)
    paired_family = load_json(PAIRED_FAMILY)
    iid_full = load_json(IID_FULL_CI)
    uncapped_split = load_json(UNCAPPED_SPLIT_CI)
    uncapped_family = load_json(UNCAPPED_FAMILY_CI)
    uncapped_2k_split = load_json(XVERSE_2K_UNCAPPED_SPLIT_CI)
    uncapped_2k_family = load_json(XVERSE_2K_UNCAPPED_FAMILY_CI)
    uncapped_paired_split = load_json(UNCAPPED_PAIRED_SPLIT)
    uncapped_paired_family = load_json(UNCAPPED_PAIRED_FAMILY)
    seed43_split = load_json(SEED43_UNCAPPED_SPLIT_CI)
    seed43_family = load_json(SEED43_UNCAPPED_FAMILY_CI)
    seed43_iid = load_json(SEED43_IID_FULL_CI)

    stable_rows = []
    for source, payload, group, label in [
        ("split", paired_split, "test", "test"),
        ("split", paired_split, "test_multi_unseen2", "test_multi_unseen2"),
        ("family", paired_family, "family_gene", "family_gene"),
        ("family", paired_family, "family_drug", "family_drug"),
        ("family", paired_family, "structure_multi", "structure_multi"),
    ]:
        for metric in ("pearson_pert", "test_mmd_clamped"):
            row = get_group_metric(payload, group, metric)
            stable_rows.append({"source": source, "group": label, "metric": metric, "row": row})

    uncapped_rows = []
    for metric in ("pearson_pert", "test_mmd_clamped"):
        uncapped_rows.append({"group": "test_full_train_eval", "metric": metric, "row": get_ci_metric(iid_full, "test_full", metric)})
    for payload, group in [
        (uncapped_split, "test"),
        (uncapped_split, "test_multi_unseen2"),
        (uncapped_family, "family_gene"),
        (uncapped_family, "family_drug"),
        (uncapped_family, "structure_multi"),
    ]:
        for metric in ("pearson_pert", "test_mmd_clamped"):
            uncapped_rows.append({"group": group, "metric": metric, "row": get_ci_metric(payload, group, metric)})

    uncapped_2k_rows = []
    for payload, group in [
        (uncapped_2k_split, "test"),
        (uncapped_2k_split, "test_multi_unseen2"),
        (uncapped_2k_family, "family_gene"),
        (uncapped_2k_family, "family_drug"),
        (uncapped_2k_family, "structure_multi"),
    ]:
        for metric in ("pearson_pert", "test_mmd_clamped"):
            uncapped_2k_rows.append({"group": group, "metric": metric, "row": get_ci_metric(payload, group, metric)})

    uncapped_paired_rows = []
    for source, payload, group, label in [
        ("split", uncapped_paired_split, "test", "test"),
        ("split", uncapped_paired_split, "test_single", "test_single"),
        ("split", uncapped_paired_split, "test_multi", "test_multi"),
        ("split", uncapped_paired_split, "test_multi_unseen2", "test_multi_unseen2"),
        ("family", uncapped_paired_family, "family_gene", "family_gene"),
        ("family", uncapped_paired_family, "family_drug", "family_drug"),
        ("family", uncapped_paired_family, "structure_multi", "structure_multi"),
    ]:
        for metric in ("pearson_pert", "test_mmd_clamped"):
            row = get_group_metric(payload, group, metric)
            uncapped_paired_rows.append({"source": source, "group": label, "metric": metric, "row": row})

    seed43_rows = []
    for metric in ("pearson_pert", "test_mmd_clamped"):
        seed43_rows.append({"group": "test_full_train_eval", "metric": metric, "row": get_ci_metric(seed43_iid, "test_full", metric)})
    for payload, group in [
        (seed43_split, "test"),
        (seed43_split, "test_multi_unseen2"),
        (seed43_family, "family_gene"),
        (seed43_family, "family_drug"),
        (seed43_family, "structure_multi"),
    ]:
        for metric in ("pearson_pert", "test_mmd_clamped"):
            seed43_rows.append({"group": group, "metric": metric, "row": get_ci_metric(payload, group, metric)})

    desc = {
        "xverse_8k_iid_full": {
            "path": str(XVERSE_RUN / "iid_eval_results.json"),
            "n_conds": (load_json(XVERSE_RUN / "iid_eval_results.json") or {}).get("n_conds"),
            "pearson_pert": metric_from_iid(XVERSE_RUN / "iid_eval_results.json", "pearson_pert"),
            "test_mmd_clamped": metric_from_iid(XVERSE_RUN / "iid_eval_results.json", "test_mmd_clamped"),
        },
        "xverse_8k_stablecaps": {
            "split": str(XVERSE_RUN / "posthoc_eval/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json"),
            "family": str(XVERSE_RUN / "posthoc_eval/condition_family_eval_best_ode20_mse1024_mmd1024_stablecaps.json"),
            "test_pp": group_metric_from_eval(
                XVERSE_RUN / "posthoc_eval/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json",
                "test",
                "pearson_pert",
            ),
            "family_gene_pp": group_metric_from_eval(
                XVERSE_RUN / "posthoc_eval/condition_family_eval_best_ode20_mse1024_mmd1024_stablecaps.json",
                "family_gene",
                "pearson_pert",
            ),
            "unseen2_pp": group_metric_from_eval(
                XVERSE_RUN / "posthoc_eval/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json",
                "test_multi_unseen2",
                "pearson_pert",
            ),
        },
        "scfoundation_anchor_stablecaps": {
            "split": str(SCF_ANCHOR / "posthoc_eval/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json"),
            "family": str(SCF_ANCHOR / "posthoc_eval/condition_family_eval_best_ode20_mse1024_mmd1024_stablecaps.json"),
            "test_pp": group_metric_from_eval(
                SCF_ANCHOR / "posthoc_eval/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json",
                "test",
                "pearson_pert",
            ),
            "family_gene_pp": group_metric_from_eval(
                SCF_ANCHOR / "posthoc_eval/condition_family_eval_best_ode20_mse1024_mmd1024_stablecaps.json",
                "family_gene",
                "pearson_pert",
            ),
            "unseen2_pp": group_metric_from_eval(
                SCF_ANCHOR / "posthoc_eval/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json",
                "test_multi_unseen2",
                "pearson_pert",
            ),
        },
    }

    return {
        "xverse_run": str(XVERSE_RUN),
        "xverse_2k": str(XVERSE_2K),
        "scfoundation_anchor": str(SCF_ANCHOR),
        "stablecaps_paired_available": paired_split is not None and paired_family is not None,
        "iid_full_ci_available": iid_full is not None,
        "condition_uncapped_available": uncapped_split is not None and uncapped_family is not None,
        "xverse_2k_uncapped_available": uncapped_2k_split is not None and uncapped_2k_family is not None,
        "condition_uncapped_paired_available": uncapped_paired_split is not None and uncapped_paired_family is not None,
        "seed43_iid_full_ci_available": seed43_iid is not None,
        "seed43_uncapped_available": seed43_split is not None and seed43_family is not None,
        "descriptive": desc,
        "stablecaps_rows": stable_rows,
        "uncapped_rows": uncapped_rows,
        "uncapped_2k_rows": uncapped_2k_rows,
        "uncapped_paired_rows": uncapped_paired_rows,
        "seed43_rows": seed43_rows,
    }


def render_md(payload: dict[str, Any]) -> str:
    status = (
        "condition_uncapped_available"
        if payload["condition_uncapped_available"]
        else "waiting_for_condition_uncapped_posthoc"
    )
    lines = [
        "# LatentFM xverse Stage Summary 2026-06-21",
        "",
        f"Status: `{status}`",
        "",
        "## Scope",
        "",
        "This report formalizes xverse as a top-latent stage candidate. Same-latent",
        "xverse 8k vs 2k deltas use paired condition bootstrap. Cross-latent",
        "scFoundation vs xverse values are descriptive only because the latent spaces",
        "and MMD scales differ.",
        "",
        "## Descriptive Values",
        "",
        "| run | n conds | test pp | family_gene pp | unseen2 pp | test MMD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    desc = payload["descriptive"]
    lines.append(
        "| `xverse_8k_iid_full` | {n} | {pp} | pending | pending | {mmd} |".format(
            n=desc["xverse_8k_iid_full"].get("n_conds") or "pending",
            pp=fmt(desc["xverse_8k_iid_full"].get("pearson_pert")),
            mmd=fmt(desc["xverse_8k_iid_full"].get("test_mmd_clamped")),
        )
    )
    for key in ("xverse_8k_stablecaps", "scfoundation_anchor_stablecaps"):
        row = desc[key]
        lines.append(
            "| `{key}` | 254 | {test} | {family} | {u2} | pending |".format(
                key=key,
                test=fmt(row.get("test_pp")),
                family=fmt(row.get("family_gene_pp")),
                u2=fmt(row.get("unseen2_pp")),
            )
        )

    lines += [
        "",
        "## xverse 8k vs 2k Stablecaps Paired Bootstrap",
        "",
        "| group | metric | delta | 95% CI | p improve | p harm |",
        "|---|---|---:|---|---:|---:|",
    ]
    for item in payload["stablecaps_rows"]:
        row = item["row"]
        lines.append(
            "| `{group}` | `{metric}` | {delta} | {ci} | {pimp} | {pharm} |".format(
                group=item["group"],
                metric=item["metric"],
                delta=fmt(None if row is None else row.get("delta_mean"), signed=True),
                ci=ci(row, delta=True),
                pimp=fmt(None if row is None else row.get("p_improvement")),
                pharm=fmt(None if row is None else row.get("p_harm")),
            )
        )

    lines += [
        "",
        "## xverse 8k Condition-Uncapped Single-Run CI",
        "",
        "| group | metric | mean | 95% CI | status |",
        "|---|---|---:|---|---|",
    ]
    for item in payload["uncapped_rows"]:
        row = item["row"]
        lines.append(
            "| `{group}` | `{metric}` | {mean} | {ci} | {status} |".format(
                group=item["group"],
                metric=item["metric"],
                mean=fmt(None if row is None else row.get("mean")),
                ci=ci(row),
                status="pending" if row is None else row.get("status", "NA"),
            )
        )

    lines += [
        "",
        "## xverse 2k Condition-Uncapped Single-Run CI",
        "",
        "| group | metric | mean | 95% CI | status |",
        "|---|---|---:|---|---|",
    ]
    for item in payload["uncapped_2k_rows"]:
        row = item["row"]
        lines.append(
            "| `{group}` | `{metric}` | {mean} | {ci} | {status} |".format(
                group=item["group"],
                metric=item["metric"],
                mean=fmt(None if row is None else row.get("mean")),
                ci=ci(row),
                status="pending" if row is None else row.get("status", "NA"),
            )
        )

    lines += [
        "",
        "## xverse 8k vs 2k Condition-Uncapped Paired Bootstrap",
        "",
        "| group | metric | delta | 95% CI | p improve | p harm |",
        "|---|---|---:|---|---:|---:|",
    ]
    for item in payload["uncapped_paired_rows"]:
        row = item["row"]
        lines.append(
            "| `{group}` | `{metric}` | {delta} | {ci} | {pimp} | {pharm} |".format(
                group=item["group"],
                metric=item["metric"],
                delta=fmt(None if row is None else row.get("delta_mean"), signed=True),
                ci=ci(row, delta=True),
                pimp=fmt(None if row is None else row.get("p_improvement")),
                pharm=fmt(None if row is None else row.get("p_harm")),
            )
        )

    lines += [
        "",
        "## xverse 8k Seed43 Condition-Uncapped Single-Run CI",
        "",
        "| group | metric | mean | 95% CI | status |",
        "|---|---|---:|---|---|",
    ]
    for item in payload["seed43_rows"]:
        row = item["row"]
        lines.append(
            "| `{group}` | `{metric}` | {mean} | {ci} | {status} |".format(
                group=item["group"],
                metric=item["metric"],
                mean=fmt(None if row is None else row.get("mean")),
                ci=ci(row),
                status="pending" if row is None else row.get("status", "NA"),
            )
        )

    lines += [
        "",
        "## Decision",
        "",
    ]
    if payload["condition_uncapped_available"]:
        lines += [
            "- xverse has a formal condition-uncapped CI table. Use it to decide whether to write the top-latent stage result.",
            "- Unseen2 composition remains a separate failure mode unless the uncapped CI contradicts the stablecaps pattern.",
        ]
    else:
        lines += [
            "- Wait for the active xverse condition-uncapped posthoc before finalizing the stage result.",
            "- Current stablecaps evidence already supports xverse 8k over xverse 2k for aggregate/family/multi geometry.",
        ]
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    payload = build_payload()
    REPORT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_md": str(REPORT_MD), "condition_uncapped_available": payload["condition_uncapped_available"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
