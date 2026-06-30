#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
OUT = ROOT / "reports/LATENTFM_FULLCAP_POSTHOC_REPORT_20260618.md"
OUT_JSON = ROOT / "reports/latentfm_fullcap_decision_status_20260618.json"

RUNS = [
    (
        "primary_scfoundation",
        ROOT / "CoupledFM/output/latentfm_runs/full_scfoundation/20260617_scfoundation_comp006_delta_w5_12k",
    ),
    (
        "resid_sum_3k",
        ROOT / "CoupledFM/output/latentfm_runs/full_scfoundation_alignment_smoke/20260617_scfoundation_resid005_ctr002_sum_pool_3k_smoke",
    ),
    (
        "resid_retry2_3k",
        ROOT / "CoupledFM/output/latentfm_runs/full_scfoundation_alignment_smoke/20260617_scfoundation_resid002_ctr0005_comp006_sum_pool_3k_smoke_retry2",
    ),
    (
        "conddelta_3k",
        ROOT / "CoupledFM/output/latentfm_runs/full_scfoundation_alignment_smoke/20260617_scfoundation_conddelta005_comp006_endpoint5_3k_smoke",
    ),
    (
        "conddelta_inject_3k",
        ROOT / "CoupledFM/output/latentfm_runs/full_scfoundation_alignment_smoke/20260617_scfoundation_conddelta005_inject_comp006_endpoint5_3k_smoke",
    ),
    (
        "conddelta_addatom_3k",
        ROOT / "CoupledFM/output/latentfm_runs/full_scfoundation_alignment_smoke/20260617_scfoundation_conddelta005_addatom005_comp006_endpoint5_3k_smoke",
    ),
    (
        "conddelta_pertresid_3k",
        ROOT / "CoupledFM/output/latentfm_runs/full_scfoundation_alignment_smoke/20260617_scfoundation_conddelta005_pertresidtarget_comp006_endpoint5_3k_smoke",
    ),
]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metric(row: dict[str, Any] | None, key: str) -> float | None:
    if row is None:
        return None
    val = row.get(key)
    return float(val) if isinstance(val, (int, float)) else None


def fmt(x: object, digits: int = 4) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "NA"
    if isinstance(x, (int, float)):
        return f"{x:.{digits}f}"
    return str(x)


def group(split: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    if split is None:
        return None
    g = split.get("groups", {}).get(name)
    return g if isinstance(g, dict) else None


def row_summary(r: dict[str, Any]) -> dict[str, Any]:
    split = r["split"]
    fam = r["family"]
    test = group(split, "test")
    single = group(split, "test_single")
    seen = group(split, "test_multi_seen")
    u1 = group(split, "test_multi_unseen1")
    u2 = group(split, "test_multi_unseen2")
    gene = group(fam, "family_gene")
    drug = group(fam, "family_drug")
    multi = group(fam, "structure_multi")
    return {
        "label": r["label"],
        "run_dir": str(r["run_dir"]),
        "complete_split": split is not None,
        "complete_family": fam is not None,
        "test_n": metric(test, "n_conds"),
        "test_mmd": metric(test, "test_mmd"),
        "test_pc": metric(test, "pearson_ctrl"),
        "test_pp": metric(test, "pearson_pert"),
        "single_pp": metric(single, "pearson_pert"),
        "multi_seen_pp": metric(seen, "pearson_pert"),
        "multi_unseen1_pp": metric(u1, "pearson_pert"),
        "multi_unseen2_pp": metric(u2, "pearson_pert"),
        "family_gene_n": metric(gene, "n_conds"),
        "family_gene_pp": metric(gene, "pearson_pert"),
        "family_drug_n": metric(drug, "n_conds"),
        "family_drug_pp": metric(drug, "pearson_pert"),
        "structure_multi_n": metric(multi, "n_conds"),
        "structure_multi_pp": metric(multi, "pearson_pert"),
    }


def choose_recommendation(summaries: list[dict[str, Any]]) -> tuple[str, list[str]]:
    pending = [
        str(r["label"])
        for r in summaries
        if not (r.get("complete_split") and r.get("complete_family"))
    ]
    if pending:
        return (
            "pending",
            [
                "Wait for full-cap posthoc to finish before launching new LatentFM training.",
                "Pending rows: " + ", ".join(pending),
            ],
        )

    primary = next((r for r in summaries if r["label"] == "primary_scfoundation"), None)
    if primary is None:
        return (
            "needs_attention",
            ["Primary scFoundation full-cap reference is missing; inspect the posthoc run."],
        )

    primary_gate = (
        primary.get("family_gene_pp"),
        primary.get("multi_unseen1_pp"),
        primary.get("multi_unseen2_pp"),
        primary.get("test_mmd"),
    )
    if any(v is None for v in primary_gate):
        return (
            "needs_attention",
            ["Primary full-cap metrics are incomplete; inspect generated JSON files."],
        )
    mmd_gate = max(0.028, float(primary_gate[3]) * 1.10)

    viable: list[dict[str, Any]] = []
    for row in summaries:
        if row["label"] == "primary_scfoundation":
            continue
        vals = (
            row.get("family_gene_pp"),
            row.get("multi_unseen1_pp"),
            row.get("multi_unseen2_pp"),
            row.get("test_mmd"),
        )
        if any(v is None for v in vals):
            continue
        if (
            vals[0] >= primary_gate[0]
            and vals[1] >= primary_gate[1]
            and vals[2] >= primary_gate[2]
            and vals[3] <= mmd_gate
        ):
            viable.append(row)

    if viable:
        viable.sort(
            key=lambda r: (
                (r.get("family_gene_pp") or -999)
                + (r.get("multi_unseen1_pp") or -999)
                + (r.get("multi_unseen2_pp") or -999)
                - 0.1 * (r.get("test_mmd") or 0)
            ),
            reverse=True,
        )
        best = viable[0]
        return (
            "promote_candidate",
            [
                f"Best full-cap candidate: {best['label']}.",
                f"Promotion gate required family_gene/multi-unseen pp >= primary and test_mmd <= {mmd_gate:.6f}.",
                "Next: launch a longer capped run for this exact setting before any formal full run.",
            ],
        )

    stack_hint = (
        "Existing broad aggregation showed Stack branches currently dominate multi-unseen pp while "
        "scFoundation dominates MMD/single-gene pp. Treat this as a representation tradeoff, not a "
        "single-score failure."
    )
    return (
        "pivot_from_scfoundation_head_smokes",
        [
            f"No scFoundation smoke preserves/improves primary family-gene pp and both multi-unseen pp groups while keeping test_mmd <= {mmd_gate:.6f}.",
            "Do not launch another condition-delta/head-frame smoke as-is.",
            "Next: pivot to stronger velocity-level composition, or run a Stack-led multi-composition branch while keeping scFoundation for MMD/single-gene analyses.",
            stack_hint,
        ],
    )


def main() -> int:
    rows: list[dict[str, Any]] = []
    for label, run_dir in RUNS:
        split = load_json(run_dir / "posthoc_eval_fullcap/split_group_eval_best_ode20_mse2048_mmd2048_fullcap.json")
        fam = load_json(run_dir / "posthoc_eval_fullcap/condition_family_eval_best_ode20_mse2048_mmd2048_fullcap.json")
        iid = load_json(run_dir / "iid_eval_results.json")
        rows.append({"label": label, "run_dir": run_dir, "split": split, "family": fam, "iid": iid})

    summaries = [row_summary(r) for r in rows]
    status, recommendations = choose_recommendation(summaries)

    lines = [
        "# LatentFM Full-Cap Posthoc Report",
        "",
        f"Generated: {datetime.now().strftime('%F %T %Z')}",
        "",
        "Purpose: compare scFoundation smoke branches under the full canonical test split, overriding checkpoint smoke eval caps. This avoids mixing `n=412` capped smoke posthoc with `n=787` primary posthoc.",
        "",
        "## Decision Gate",
        "",
        f"- Status: `{status}`",
    ]
    for rec in recommendations:
        lines.append(f"- {rec}")
    lines += [
        "",
        "## Split Metrics",
        "",
        "| Run | status | test n | MMD | pc | pp | single pp | multi seen pp | unseen1 pp | unseen2 pp |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r, s in zip(rows, summaries):
        row_status = "complete" if s["complete_split"] else "pending"
        lines.append(
            f"| `{r['label']}` | {row_status} | {fmt(s['test_n'], 0)} | "
            f"{fmt(s['test_mmd'], 6)} | {fmt(s['test_pc'])} | "
            f"{fmt(s['test_pp'])} | {fmt(s['single_pp'])} | "
            f"{fmt(s['multi_seen_pp'])} | {fmt(s['multi_unseen1_pp'])} | "
            f"{fmt(s['multi_unseen2_pp'])} |"
        )

    lines += [
        "",
        "## Family Metrics",
        "",
        "| Run | status | family gene n | family gene pp | family drug n | family drug pp | structure multi n | structure multi pp |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r, s in zip(rows, summaries):
        row_status = "complete" if s["complete_family"] else "pending"
        lines.append(
            f"| `{r['label']}` | {row_status} | {fmt(s['family_gene_n'], 0)} | "
            f"{fmt(s['family_gene_pp'])} | {fmt(s['family_drug_n'], 0)} | "
            f"{fmt(s['family_drug_pp'])} | {fmt(s['structure_multi_n'], 0)} | "
            f"{fmt(s['structure_multi_pp'])} |"
        )

    complete = [r for r in rows if r["split"] is not None]
    if complete:
        primary = next((r for r in complete if r["label"] == "primary_scfoundation"), None)
        lines += ["", "## Interim Interpretation", ""]
        if primary is None:
            lines.append("Primary full-cap reference is not complete yet; wait for the detached posthoc job.")
        else:
            p_u1 = metric(group(primary["split"], "test_multi_unseen1"), "pearson_pert")
            p_u2 = metric(group(primary["split"], "test_multi_unseen2"), "pearson_pert")
            p_gene = metric(group(primary["family"], "family_gene") if primary["family"] else None, "pearson_pert")
            p_mmd = metric(group(primary["split"], "test"), "test_mmd")
            mmd_gate = max(0.028, float(p_mmd) * 1.10) if p_mmd is not None else None
            viable: list[str] = []
            for r in complete:
                if r["label"] == "primary_scfoundation":
                    continue
                u1 = metric(group(r["split"], "test_multi_unseen1"), "pearson_pert")
                u2 = metric(group(r["split"], "test_multi_unseen2"), "pearson_pert")
                gene = metric(group(r["family"], "family_gene") if r["family"] else None, "pearson_pert")
                mmd = metric(group(r["split"], "test"), "test_mmd")
                if (
                    None not in (p_u1, p_u2, p_gene, p_mmd, u1, u2, gene, mmd)
                    and u1 >= p_u1
                    and u2 >= p_u2
                    and gene >= p_gene
                    and mmd <= mmd_gate
                ):
                    viable.append(str(r["label"]))
            if viable:
                lines.append("Full-cap gate currently has viable smoke branch(es): " + ", ".join(f"`{x}`" for x in viable) + ".")
            else:
                if mmd_gate is None:
                    lines.append("No completed smoke branch currently preserves/improves primary on family-gene pp and both multi-unseen pp groups.")
                else:
                    lines.append(
                        "No completed smoke branch currently preserves/improves primary on family-gene pp and both "
                        f"multi-unseen pp groups while keeping test_mmd <= {mmd_gate:.6f}."
                    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_JSON.write_text(
        json.dumps(
            {
                "generated": datetime.now().strftime("%F %T %Z"),
                "status": status,
                "recommendations": recommendations,
                "rows": summaries,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(OUT)
    print(OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
