#!/usr/bin/env python3
"""CPU-only touch-set preflight for a Track A hybrid tail adapter.

This script does not run model inference or training. It estimates how broad
different gene-single adapter gates would be on exact Track A strata and whether
they would mostly touch recurrent hard tails or many non-tail single-gene rows.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
EXACT_ROWS = ROOT / "reports/tracka_simple_single_unseen_exact_20260627/condition_rows.csv"
RECURRENT_ROWS = ROOT / "reports/tracka_recurrent_tail_gate_20260627/recurrent_tail_rows.csv"
XVERSE_BANK = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_condition_prior_adapter_smoke_20260621/"
    / "xverse_prior_adapter_global_genemean_w005_add002_replay1_4k/condition_prior_bank_summary.json"
)
SCF_BANK = (
    ROOT
    / "CoupledFM/output/latentfm_runs/latentfm_tracka_scf_jiang_lowcount_adapter_20260623/"
    / "scfoundation_tracka_gene_shrink_k2_jiang_lowcount_adapter_2k_seed42/condition_prior_bank_summary.json"
)
OUT_DIR = ROOT / "reports/tracka_hybrid_touchset_preflight_20260627"
OUT_JSON = ROOT / "reports/latentfm_tracka_hybrid_touchset_preflight_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_HYBRID_TOUCHSET_PREFLIGHT_20260627.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_bank(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    genes = payload.get("genes") or {}
    return {str(g).strip().upper(): dict(v) for g, v in genes.items()}


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    exact_raw = read_csv(EXACT_ROWS)
    recurrent_raw = read_csv(RECURRENT_ROWS)
    xverse_bank = load_bank(XVERSE_BANK)
    scf_bank = load_bank(SCF_BANK)

    exact: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in exact_raw:
        group = row.get("group")
        if group not in {"simple_single_unseen", "cross_background_seen_gene_exact"}:
            continue
        key = (row["group"], row["dataset"], row["condition"])
        item = exact.setdefault(
            key,
            {
                "group": row["group"],
                "dataset": row["dataset"],
                "condition": row["condition"],
                "gene": str(row["gene"]).strip().upper(),
                "perturbation_type": row.get("perturbation_type", ""),
                "pearson_values": [],
                "mmd_values": [],
            },
        )
        item["pearson_values"].append(fnum(row.get("pearson_pert")))
        item["mmd_values"].append(fnum(row.get("test_mmd_clamped")))

    for item in exact.values():
        vals = item["pearson_values"]
        mmd = item["mmd_values"]
        item["pp_mean"] = sum(vals) / len(vals) if vals else None
        item["pp_min"] = min(vals) if vals else None
        item["mmd_max"] = max(mmd) if mmd else None
        item["xverse_prior_covered"] = item["gene"] in xverse_bank
        item["scf_prior_covered"] = item["gene"] in scf_bank
        item["xverse_raw_condition_count"] = int(
            xverse_bank.get(item["gene"], {}).get("raw_condition_count") or 0
        )
        item["scf_raw_condition_count"] = int(
            scf_bank.get(item["gene"], {}).get("raw_condition_count") or 0
        )

    recurrent: dict[tuple[str, str, str], dict[str, str]] = {}
    hard_keys: set[tuple[str, str, str]] = set()
    hard_genes: set[str] = set()
    cross_hard_genes: set[str] = set()
    simple_hard_genes: set[str] = set()
    for row in recurrent_raw:
        key = (row["group"], row["dataset"], row["condition"])
        recurrent[key] = row
        if str(row.get("recurrent_hard_tail")).lower() == "true":
            hard_keys.add(key)
            gene = str(row.get("gene", "")).strip().upper()
            if gene:
                hard_genes.add(gene)
                if row["group"] == "cross_background_seen_gene_exact":
                    cross_hard_genes.add(gene)
                elif row["group"] == "simple_single_unseen":
                    simple_hard_genes.add(gene)

    for key, item in exact.items():
        rec = recurrent.get(key)
        item["recurrent_hard_tail"] = key in hard_keys
        item["recurrent_negative_tail"] = (
            str((rec or {}).get("recurrent_negative_tail", "")).lower() == "true"
        )
        item["mmd_risk"] = str((rec or {}).get("mmd_risk", "")).lower() == "true"
        item["tail_priority_score"] = fnum((rec or {}).get("tail_priority_score"), 0.0)

    exact_rows = list(exact.values())
    hard_rows = [r for r in exact_rows if r["recurrent_hard_tail"]]
    group_totals = Counter(r["group"] for r in exact_rows)
    group_hard_totals = Counter(r["group"] for r in hard_rows)

    def policy_gene_single(row: dict[str, Any]) -> bool:
        return True

    def policy_xverse_prior_covered(row: dict[str, Any]) -> bool:
        return bool(row["xverse_prior_covered"])

    def policy_tail_gene(row: dict[str, Any]) -> bool:
        return row["gene"] in hard_genes

    def policy_cross_tail_gene(row: dict[str, Any]) -> bool:
        return row["gene"] in cross_hard_genes

    def policy_tail_gene_xverse_count_ge2(row: dict[str, Any]) -> bool:
        return row["gene"] in hard_genes and int(row["xverse_raw_condition_count"]) >= 2

    def policy_tail_gene_xverse_count_ge4(row: dict[str, Any]) -> bool:
        return row["gene"] in hard_genes and int(row["xverse_raw_condition_count"]) >= 4

    def policy_tail_gene_scf_count_ge2(row: dict[str, Any]) -> bool:
        return row["gene"] in hard_genes and int(row["scf_raw_condition_count"]) >= 2

    policies = {
        "current_gene_single_filter": policy_gene_single,
        "xverse_prior_covered_single": policy_xverse_prior_covered,
        "tail_gene_allowlist": policy_tail_gene,
        "cross_tail_gene_allowlist": policy_cross_tail_gene,
        "tail_gene_xverse_prior_count_ge2": policy_tail_gene_xverse_count_ge2,
        "tail_gene_xverse_prior_count_ge4": policy_tail_gene_xverse_count_ge4,
        "tail_gene_scf_prior_count_ge2": policy_tail_gene_scf_count_ge2,
    }

    summaries: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    for name, fn in policies.items():
        touched = [r for r in exact_rows if fn(r)]
        touched_hard = [r for r in touched if r["recurrent_hard_tail"]]
        touched_nonhard = [r for r in touched if not r["recurrent_hard_tail"]]
        touched_mmd_risk = [r for r in touched if r["mmd_risk"]]
        row = {
            "policy": name,
            "touched_total": len(touched),
            "touched_fraction": len(touched) / max(len(exact_rows), 1),
            "hard_tail_coverage": len(touched_hard) / max(len(hard_rows), 1),
            "nonhard_touch_fraction": len(touched_nonhard)
            / max(len([r for r in exact_rows if not r["recurrent_hard_tail"]]), 1),
            "mmd_risk_touch_fraction": len(touched_mmd_risk)
            / max(len([r for r in exact_rows if r["mmd_risk"]]), 1),
            "hard_tail_precision": len(touched_hard) / max(len(touched), 1),
        }
        for group in ("simple_single_unseen", "cross_background_seen_gene_exact"):
            group_touched = [r for r in touched if r["group"] == group]
            group_hard = [r for r in touched_hard if r["group"] == group]
            row[f"{group}_touched"] = len(group_touched)
            row[f"{group}_hard_coverage"] = len(group_hard) / max(group_hard_totals[group], 1)
        summaries.append(row)
        for r in touched:
            policy_rows.append(
                {
                    "policy": name,
                    "group": r["group"],
                    "dataset": r["dataset"],
                    "condition": r["condition"],
                    "gene": r["gene"],
                    "recurrent_hard_tail": r["recurrent_hard_tail"],
                    "mmd_risk": r["mmd_risk"],
                    "xverse_raw_condition_count": r["xverse_raw_condition_count"],
                    "scf_raw_condition_count": r["scf_raw_condition_count"],
                    "pp_min": r["pp_min"],
                    "mmd_max": r["mmd_max"],
                }
            )

    # Gate: broad policies fail if they touch too many non-hard rows. Narrow
    # allowlists are interesting only if they cover most hard tails.
    best_narrow = max(
        (r for r in summaries if "tail_gene" in r["policy"] or "cross_tail" in r["policy"]),
        key=lambda r: (
            float(r["hard_tail_coverage"]),
            -float(r["nonhard_touch_fraction"]),
            float(r["hard_tail_precision"]),
        ),
    )
    broad = {r["policy"]: r for r in summaries}
    reasons: list[str] = []
    if broad["current_gene_single_filter"]["nonhard_touch_fraction"] > 0.50:
        reasons.append("current_gene_single_filter_touches_most_nonhard_exact_rows")
    if broad["xverse_prior_covered_single"]["nonhard_touch_fraction"] > 0.50:
        reasons.append("xverse_prior_covered_single_too_broad")
    if float(best_narrow["hard_tail_coverage"]) < 0.80:
        reasons.append("best_narrow_allowlist_hard_tail_coverage_below_0p80")
    if float(best_narrow["nonhard_touch_fraction"]) > 0.25:
        reasons.append("best_narrow_allowlist_nonhard_touch_fraction_above_0p25")

    existing_code_can_express_best = best_narrow["policy"] in {
        "current_gene_single_filter",
        "xverse_prior_covered_single",
    }
    if not existing_code_can_express_best:
        reasons.append("best_narrow_allowlist_requires_default_off_code_change")

    status = (
        "hybrid_touchset_preflight_pass_code_change_needed"
        if reasons == ["best_narrow_allowlist_requires_default_off_code_change"]
        else "hybrid_touchset_preflight_fail_no_gpu"
    )

    summary_fields = [
        "policy",
        "touched_total",
        "touched_fraction",
        "hard_tail_coverage",
        "nonhard_touch_fraction",
        "mmd_risk_touch_fraction",
        "hard_tail_precision",
        "simple_single_unseen_touched",
        "simple_single_unseen_hard_coverage",
        "cross_background_seen_gene_exact_touched",
        "cross_background_seen_gene_exact_hard_coverage",
    ]
    write_csv(OUT_DIR / "policy_summary.csv", summaries, summary_fields)
    write_csv(
        OUT_DIR / "policy_touched_rows.csv",
        policy_rows,
        [
            "policy",
            "group",
            "dataset",
            "condition",
            "gene",
            "recurrent_hard_tail",
            "mmd_risk",
            "xverse_raw_condition_count",
            "scf_raw_condition_count",
            "pp_min",
            "mmd_max",
        ],
    )

    by_gene = defaultdict(lambda: {"hard_rows": 0, "rows": 0, "datasets": set(), "groups": set()})
    for r in exact_rows:
        item = by_gene[r["gene"]]
        item["rows"] += 1
        item["datasets"].add(r["dataset"])
        item["groups"].add(r["group"])
        if r["recurrent_hard_tail"]:
            item["hard_rows"] += 1
    gene_rows = []
    for gene, item in sorted(by_gene.items(), key=lambda kv: (-kv[1]["hard_rows"], kv[0])):
        if item["hard_rows"] <= 0:
            continue
        gene_rows.append(
            {
                "gene": gene,
                "hard_rows": item["hard_rows"],
                "exact_rows": item["rows"],
                "datasets": ";".join(sorted(item["datasets"])),
                "groups": ";".join(sorted(item["groups"])),
                "xverse_prior_covered": gene in xverse_bank,
                "xverse_raw_condition_count": int(xverse_bank.get(gene, {}).get("raw_condition_count") or 0),
                "scf_prior_covered": gene in scf_bank,
                "scf_raw_condition_count": int(scf_bank.get(gene, {}).get("raw_condition_count") or 0),
            }
        )
    write_csv(
        OUT_DIR / "hard_tail_gene_allowlist_candidates.csv",
        gene_rows,
        [
            "gene",
            "hard_rows",
            "exact_rows",
            "datasets",
            "groups",
            "xverse_prior_covered",
            "xverse_raw_condition_count",
            "scf_prior_covered",
            "scf_raw_condition_count",
        ],
    )
    allowlist_path = OUT_DIR / "tail_gene_allowlist.txt"
    allowlist_path.write_text("\n".join(sorted(hard_genes)) + "\n", encoding="utf-8")

    payload = {
        "status": status,
        "gpu_authorized": False,
        "reasons": reasons,
        "inputs": {
            "exact_rows": str(EXACT_ROWS),
            "recurrent_rows": str(RECURRENT_ROWS),
            "xverse_bank": str(XVERSE_BANK),
            "scf_bank": str(SCF_BANK),
        },
        "totals": {
            "exact_rows": len(exact_rows),
            "hard_tail_rows": len(hard_rows),
            "groups": dict(group_totals),
            "hard_tail_groups": dict(group_hard_totals),
            "hard_tail_genes": len(hard_genes),
            "cross_hard_tail_genes": len(cross_hard_genes),
            "simple_hard_tail_genes": len(simple_hard_genes),
        },
        "best_narrow_policy": best_narrow,
        "existing_code_can_express_best": existing_code_can_express_best,
        "policy_summaries": summaries,
        "outputs": {
            "summary_csv": str(OUT_DIR / "policy_summary.csv"),
            "touched_rows_csv": str(OUT_DIR / "policy_touched_rows.csv"),
            "hard_tail_gene_candidates_csv": str(OUT_DIR / "hard_tail_gene_allowlist_candidates.csv"),
            "tail_gene_allowlist": str(allowlist_path),
            "md": str(OUT_MD),
            "json": str(OUT_JSON),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# LatentFM Track A Hybrid Touch-Set Preflight 2026-06-27",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only touch-set analysis.",
        "- No training, inference, checkpoint selection, canonical multi selection, or Track C query.",
        "- Tests whether a hybrid adapter can be narrowed to recurrent hard-tail single-gene rows.",
        "",
        "## Totals",
        "",
        f"- Exact unique rows: `{len(exact_rows)}`.",
        f"- Recurrent hard-tail rows: `{len(hard_rows)}`.",
        f"- Recurrent hard-tail genes: `{len(hard_genes)}`.",
        "",
        "## Policy Summary",
        "",
        "| policy | touched | hard coverage | non-hard touched | hard precision | simple hard coverage | cross hard coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| `{policy}` | {touched_total} | {hard_tail_coverage:.3f} | "
            "{nonhard_touch_fraction:.3f} | {hard_tail_precision:.3f} | "
            "{simple_single_unseen_hard_coverage:.3f} | "
            "{cross_background_seen_gene_exact_hard_coverage:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Best narrow policy: `{best_narrow['policy']}`.",
            f"- Existing code can express best policy: `{existing_code_can_express_best}`.",
            f"- Reasons: `{', '.join(reasons) if reasons else 'none'}`.",
            "",
            "Interpretation: current coarse `gene_single` or xverse prior-covered gates are too broad for a safe hybrid. "
            "A useful route would need a default-off single-gene allowlist gate keyed by recurrent hard-tail or "
            "validated prior-covered tail genes, followed by the exact-tail candidate gate.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- Policy summary: `{OUT_DIR / 'policy_summary.csv'}`",
            f"- Touched rows: `{OUT_DIR / 'policy_touched_rows.csv'}`",
            f"- Hard-tail gene candidates: `{OUT_DIR / 'hard_tail_gene_allowlist_candidates.csv'}`",
            f"- Tail-gene allowlist: `{allowlist_path}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
