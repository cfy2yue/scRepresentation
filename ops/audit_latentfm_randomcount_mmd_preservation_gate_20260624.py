#!/usr/bin/env python3
"""CPU-only gate for random-count chemical downsampling MMD preservation.

The random-count smoke produced promising Pearson deltas but failed the internal
MMD gate. This audit localizes the harm using completed train-only internal
posthoc JSONs only. It does not read canonical metrics, canonical multi, Track C
query, active logs, or launch GPU work.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = (
    ROOT
    / "runs/latentfm_modality_pathway_randomcount_control_smoke_20260624"
    / "xverse_scaling_pathway_randomcount_3k_seed42"
)
POSTHOC = RUN_ROOT / "posthoc_eval_internal"
REPORT_JSON = ROOT / "reports/latentfm_randomcount_mmd_preservation_gate_20260624.json"
REPORT_MD = ROOT / "reports/LATENTFM_RANDOMCOUNT_MMD_PRESERVATION_GATE_20260624.md"

INPUTS = {
    "split_anchor": POSTHOC / "split_group_eval_anchor_internal_ode20.json",
    "split_candidate": POSTHOC / "split_group_eval_candidate_internal_ode20.json",
    "family_anchor": POSTHOC / "condition_family_eval_anchor_internal_ode20.json",
    "family_candidate": POSTHOC / "condition_family_eval_candidate_internal_ode20.json",
}

GROUP_SPECS = [
    ("split_cross", "split_anchor", "split_candidate", "internal_val_cross_background_seen_gene_proxy"),
    ("split_family_proxy", "split_anchor", "split_candidate", "internal_val_family_gene_proxy"),
    ("family_gene", "family_anchor", "family_candidate", "family_gene"),
]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(payload: dict[str, Any], group: str) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in ((payload.get("groups") or {}).get(group) or {}).get("condition_metrics") or []:
        if not isinstance(row, dict):
            continue
        ds = str(row.get("dataset") or "")
        cond = str(row.get("condition") or "")
        if ds and cond:
            out[(ds, cond)] = row
    return out


def fnum(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def summarize_group(anchor: dict[str, Any], candidate: dict[str, Any], group: str) -> dict[str, Any]:
    a_rows = rows(anchor, group)
    c_rows = rows(candidate, group)
    common = sorted(set(a_rows) & set(c_rows))
    deltas: list[dict[str, Any]] = []
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key in common:
        ar = a_rows[key]
        cr = c_rows[key]
        pp_a = fnum(ar.get("pearson_pert"))
        pp_c = fnum(cr.get("pearson_pert"))
        mmd_a = fnum(ar.get("test_mmd_clamped"))
        mmd_c = fnum(cr.get("test_mmd_clamped"))
        if pp_a is None or pp_c is None or mmd_a is None or mmd_c is None:
            continue
        row = {
            "dataset": key[0],
            "condition": key[1],
            "pp_delta": pp_c - pp_a,
            "mmd_delta": mmd_c - mmd_a,
            "anchor_pp": pp_a,
            "candidate_pp": pp_c,
            "anchor_mmd": mmd_a,
            "candidate_mmd": mmd_c,
        }
        deltas.append(row)
        by_ds[key[0]].append(row)
    ds_rows = []
    for ds, vals in sorted(by_ds.items()):
        ds_rows.append(
            {
                "dataset": ds,
                "n": len(vals),
                "pp_delta_mean": sum(v["pp_delta"] for v in vals) / len(vals),
                "mmd_delta_mean": sum(v["mmd_delta"] for v in vals) / len(vals),
                "mmd_harm_frac_gt_0": sum(v["mmd_delta"] > 0 for v in vals) / len(vals),
                "mmd_harm_frac_gt_0p005": sum(v["mmd_delta"] > 0.005 for v in vals) / len(vals),
                "mmd_harm_frac_gt_0p01": sum(v["mmd_delta"] > 0.01 for v in vals) / len(vals),
            }
        )
    n = len(deltas)
    pp_ds_equal = sum(r["pp_delta_mean"] for r in ds_rows) / max(1, len(ds_rows))
    mmd_ds_equal = sum(r["mmd_delta_mean"] for r in ds_rows) / max(1, len(ds_rows))
    return {
        "n_conditions": n,
        "n_datasets": len(ds_rows),
        "pp_delta_equal_condition_mean": sum(v["pp_delta"] for v in deltas) / max(1, n),
        "mmd_delta_equal_condition_mean": sum(v["mmd_delta"] for v in deltas) / max(1, n),
        "pp_delta_equal_dataset_mean": pp_ds_equal,
        "mmd_delta_equal_dataset_mean": mmd_ds_equal,
        "mmd_harm_frac_gt_0": sum(v["mmd_delta"] > 0 for v in deltas) / max(1, n),
        "mmd_harm_frac_gt_0p005": sum(v["mmd_delta"] > 0.005 for v in deltas) / max(1, n),
        "mmd_harm_frac_gt_0p01": sum(v["mmd_delta"] > 0.01 for v in deltas) / max(1, n),
        "datasets_mmd_mean_gt_0p005": [r["dataset"] for r in ds_rows if r["mmd_delta_mean"] > 0.005],
        "datasets_pp_mean_lt_0": [r["dataset"] for r in ds_rows if r["pp_delta_mean"] < 0],
        "dataset_rows": ds_rows,
        "top_mmd_harm_rows": sorted(deltas, key=lambda r: r["mmd_delta"], reverse=True)[:15],
        "top_pp_harm_rows": sorted(deltas, key=lambda r: r["pp_delta"])[:15],
        "top_pp_gain_rows": sorted(deltas, key=lambda r: r["pp_delta"], reverse=True)[:15],
    }


def main() -> int:
    missing = [str(path) for path in INPUTS.values() if not path.exists()]
    reasons: list[str] = []
    if missing:
        reasons.append("missing_required_posthoc_json")
    payloads = {name: load(path) for name, path in INPUTS.items() if path.exists()}
    summaries = {}
    for label, a_key, c_key, group in GROUP_SPECS:
        if a_key in payloads and c_key in payloads:
            summaries[label] = summarize_group(payloads[a_key], payloads[c_key], group)

    family = summaries.get("family_gene") or {}
    cross = summaries.get("split_cross") or {}
    if not family:
        reasons.append("missing_family_gene_summary")
    else:
        if float(family.get("mmd_delta_equal_dataset_mean") or 999.0) > 0.001:
            reasons.append("family_gene_mean_mmd_harm_gt_0p001")
        if float(family.get("mmd_harm_frac_gt_0p005") or 1.0) > 0.25:
            reasons.append("family_gene_mmd_harm_broad_frac_gt_0p005")
        if len(family.get("datasets_mmd_mean_gt_0p005") or []) > 2:
            reasons.append("family_gene_mmd_harm_multi_dataset")
        if len(family.get("datasets_pp_mean_lt_0") or []) > 0:
            reasons.append("family_gene_pp_dataset_tail_harm")
    if cross and float(cross.get("pp_delta_equal_dataset_mean") or 0.0) < 0.01:
        reasons.append("cross_pp_signal_weak")

    status = (
        "randomcount_mmd_preservation_gate_pass_design_next"
        if not reasons
        else "randomcount_mmd_preservation_gate_fail_no_gpu"
    )
    out = {
        "status": status,
        "gpu_authorized_by_this_script": False,
        "missing_inputs": missing,
        "reasons": reasons,
        "boundary": {
            "run_root": str(RUN_ROOT),
            "reads_train_only_internal_posthoc": True,
            "reads_canonical_metrics": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "launches_gpu": False,
        },
        "gate_rules": {
            "family_gene_mmd_delta_mean_max": 0.001,
            "family_gene_mmd_harm_frac_gt_0p005_max": 0.25,
            "family_gene_mmd_harm_dataset_count_gt_0p005_max": 2,
            "family_gene_pp_dataset_tail_harm_allowed": 0,
            "cross_pp_delta_mean_min": 0.01,
        },
        "summaries": summaries,
        "next_action": (
            "design MMD-preserving randomcount/downsampling protocol"
            if status.endswith("design_next")
            else "do not continue randomcount/downsampling GPU; treat Pearson gain as MMD-unsafe negative evidence"
        ),
    }
    REPORT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    def fmt(x: Any) -> str:
        if x is None:
            return "NA"
        if isinstance(x, float):
            return f"{x:+.6f}"
        return str(x)

    lines = [
        "# LatentFM Random-Count MMD Preservation Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only localization of completed random-count internal posthoc.",
        "- Does not read canonical metrics, canonical multi, Track C query, active logs, or launch GPU.",
        "",
        "## Group Summary",
        "",
        "| group | n | datasets | pp ds-equal | MMD ds-equal | pp cond-equal | MMD cond-equal | MMD harm >0.005 | bad MMD datasets | pp-harm datasets |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for label, row in summaries.items():
        lines.append(
            f"| `{label}` | {row['n_conditions']} | {row['n_datasets']} | "
            f"{fmt(row['pp_delta_equal_dataset_mean'])} | {fmt(row['mmd_delta_equal_dataset_mean'])} | "
            f"{fmt(row['pp_delta_equal_condition_mean'])} | {fmt(row['mmd_delta_equal_condition_mean'])} | "
            f"{fmt(row['mmd_harm_frac_gt_0p005'])} | "
            f"`{row['datasets_mmd_mean_gt_0p005']}` | `{row['datasets_pp_mean_lt_0']}` |"
        )
    lines.extend(
        [
            "",
            "## Top Family-Gene MMD Harm Rows",
            "",
            "| dataset | condition | pp delta | MMD delta |",
            "|---|---|---:|---:|",
        ]
    )
    for row in (family.get("top_mmd_harm_rows") or [])[:10]:
        lines.append(
            f"| `{row['dataset']}` | `{row['condition']}` | {fmt(row['pp_delta'])} | {fmt(row['mmd_delta'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- reasons: `{reasons}`",
            f"- next action: `{out['next_action']}`",
            "",
            "## JSON",
            "",
            f"`{REPORT_JSON}`",
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "reasons": reasons, "out_md": str(REPORT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
