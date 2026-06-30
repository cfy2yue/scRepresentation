#!/usr/bin/env python3
"""CPU-only Track C exogenous row-QA V3 gate.

This gate asks whether the train_multi row-reliability V2 failure can be
rescued by predeclared exogenous row-quality features rather than by
support-val, canonical metrics, or query feedback.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
ARTIFACT = REPORTS / "latentfm_trackc_trainmulti_row_reliability_artifact_20260624.json"
PROVENANCE = REPORTS / "latentfm_scaling_provenance_estimand_matrix_20260624.csv"
OUT_JSON = REPORTS / "latentfm_trackc_exogenous_row_qa_v3_gate_20260624.json"
OUT_MD = REPORTS / "LATENTFM_TRACKC_EXOGENOUS_ROW_QA_V3_GATE_20260624.md"


Row = dict[str, Any]
RuleFn = Callable[[Row], bool]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_provenance(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["dataset"]: row for row in csv.DictReader(handle)}


def delta(row: Row) -> float:
    return float(row["candidate"]) - float(row["support_selected_route"])


def bootstrap_p_harm(values: list[float], *, n_boot: int = 2000, seed: int = 42) -> float:
    if not values:
        return 1.0
    rng = random.Random(seed)
    harmed = 0
    n = len(values)
    for _ in range(n_boot):
        mean = sum(values[rng.randrange(n)] for _ in range(n)) / n
        if mean <= 0.0:
            harmed += 1
    return harmed / n_boot


def summarize(rows: list[Row], pred: RuleFn) -> dict[str, Any]:
    enabled = [row for row in rows if (not row.get("abstained")) and pred(row)]
    values = [delta(row) for row in enabled]
    by_dataset: dict[str, list[float]] = {}
    for row, value in zip(enabled, values):
        by_dataset.setdefault(str(row["dataset"]), []).append(value)
    dataset_pp = {ds: sum(vals) / len(vals) for ds, vals in by_dataset.items()}
    return {
        "enabled_rows": len(enabled),
        "pp_delta": (sum(values) / len(values)) if values else 0.0,
        "p_harm": bootstrap_p_harm(values) if values else 1.0,
        "enabled_negative_rows": sum(1 for value in values if value < 0.0),
        "enabled_min_pp_delta": min(values) if values else None,
        "n_datasets": len(by_dataset),
        "dataset_pp": dataset_pp,
        "norman_pp": dataset_pp.get("NormanWeissman2019_filtered"),
        "wessels_pp": dataset_pp.get("Wessels"),
    }


def pass_reasons(summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if int(summary["enabled_rows"]) < 6:
        reasons.append("enabled_rows_below_6")
    if float(summary["pp_delta"]) < 0.030:
        reasons.append("pp_delta_below_0p030")
    if float(summary["p_harm"]) > 0.20:
        reasons.append("p_harm_above_0p20")
    if int(summary["enabled_negative_rows"]) > 2:
        reasons.append("enabled_negative_rows_gt_2")
    min_pp = summary["enabled_min_pp_delta"]
    if min_pp is None or float(min_pp) < -0.020:
        reasons.append("enabled_min_pp_below_minus_0p020")
    if int(summary["n_datasets"]) < 2:
        reasons.append("single_dataset_rule_not_allowed")
    norman = summary["norman_pp"]
    if norman is None or float(norman) < -0.010:
        reasons.append("norman_pp_below_minus_0p010")
    wessels = summary["wessels_pp"]
    if wessels is None or float(wessels) < 0.020:
        reasons.append("wessels_pp_below_0p020")
    return reasons


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def main() -> int:
    artifact = load_json(ARTIFACT)
    provenance = load_provenance(PROVENANCE)

    # Gene-support coverage is computed from the row universe, not from row
    # outcomes. It is treated as an exogenous support-bank coverage feature.
    all_rows = [row for spec in artifact.get("spec_tables", []) for row in spec.get("rows", [])]
    gene_freq = Counter(gene for row in all_rows for gene in row.get("genes", []))

    def source_quality(row: Row) -> str:
        return (provenance.get(str(row["dataset"])) or {}).get("source_quality", "")

    def pert_type(row: Row) -> str:
        return (provenance.get(str(row["dataset"])) or {}).get("perturbation_type", "")

    rules: list[tuple[str, str, RuleFn]] = [
        ("all_enabled_v2_baseline", "baseline only; expected to fail", lambda row: True),
        ("min_context_ge3", "support-bank context rows >=3", lambda row: int(row.get("n_context_rows") or 0) >= 3),
        ("min_context_ge4", "support-bank context rows >=4", lambda row: int(row.get("n_context_rows") or 0) >= 4),
        ("min_context_ge5", "support-bank context rows >=5", lambda row: int(row.get("n_context_rows") or 0) >= 5),
        ("gene_support_min_ge2", "both genes appear in >=2 row-QA rows", lambda row: min(gene_freq[g] for g in row.get("genes", [])) >= 2),
        ("gene_support_min_ge3", "both genes appear in >=3 row-QA rows", lambda row: min(gene_freq[g] for g in row.get("genes", [])) >= 3),
        (
            "context_ge4_gene_support_min_ge2",
            "context rows >=4 and both genes appear in >=2 row-QA rows",
            lambda row: int(row.get("n_context_rows") or 0) >= 4 and min(gene_freq[g] for g in row.get("genes", [])) >= 2,
        ),
        (
            "context_ge4_gene_support_min_ge3",
            "context rows >=4 and both genes appear in >=3 row-QA rows",
            lambda row: int(row.get("n_context_rows") or 0) >= 4 and min(gene_freq[g] for g in row.get("genes", [])) >= 3,
        ),
        ("jackknife_cos_ge0p75", "jackknife cosine mean >=0.75", lambda row: float(row.get("jackknife_cos_mean") or 0.0) >= 0.75),
        ("jackknife_cos_ge0p90", "jackknife cosine mean >=0.90", lambda row: float(row.get("jackknife_cos_mean") or 0.0) >= 0.90),
        ("jackknife_cv_le0p40", "jackknife norm CV <=0.40", lambda row: float(row.get("jackknife_norm_cv") or 1e9) <= 0.40),
        ("source_verified", "dataset source_quality == source_verified", lambda row: source_quality(row) == "source_verified"),
        ("crispr_or_cas13_multi_sources", "perturbation type CRISPRa or Cas13", lambda row: pert_type(row) in {"CRISPRa", "Cas13"}),
    ]

    rows_out: list[dict[str, Any]] = []
    for spec in artifact.get("spec_tables", []):
        spec_name = str(spec.get("spec"))
        spec_rows = list(spec.get("rows") or [])
        for rule_name, rule_desc, pred in rules:
            summary = summarize(spec_rows, pred)
            reasons = pass_reasons(summary)
            rows_out.append(
                {
                    "spec": spec_name,
                    "rule": rule_name,
                    "rule_description": rule_desc,
                    **summary,
                    "pass": not reasons,
                    "reasons": reasons,
                }
            )

    passing = [row for row in rows_out if row["pass"]]
    rows_out.sort(
        key=lambda row: (
            int(row["pass"]),
            float(row["pp_delta"]),
            -float(row["p_harm"]),
            int(row["enabled_rows"]),
        ),
        reverse=True,
    )
    best = rows_out[0] if rows_out else {}
    reason_counts = Counter(reason for row in rows_out for reason in row.get("reasons", []))
    status = "trackc_exogenous_row_qa_v3_pass_external_review_next" if passing else "trackc_exogenous_row_qa_v3_fail_no_gpu"

    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "reads_train_multi_row_reliability_artifact": True,
            "uses_exogenous_row_qa_features_only": True,
            "support_val_selection": False,
            "reads_canonical_metrics": False,
            "reads_canonical_multi": False,
            "reads_heldout_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "inputs": {
            "artifact": str(ARTIFACT),
            "provenance_csv": str(PROVENANCE),
        },
        "summary": {
            "n_specs": artifact.get("n_specs"),
            "n_rules": len(rules),
            "n_rule_specs": len(rows_out),
            "n_pass": len(passing),
            "best": best,
            "reason_counts": dict(reason_counts.most_common()),
        },
        "decision": {
            "gpu_next_action": "none" if not passing else "external review before one bounded support-only smoke",
            "trackc_query_allowed": False,
            "canonical_noharm_allowed": False,
            "fail_close": not bool(passing),
        },
        "top_rows": rows_out[:25],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Track C Exogenous Row-QA V3 Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only gate over safe trainselect `train_multi` row-reliability artifact.",
        "- Uses predeclared exogenous row-QA/provenance features only: support-bank context count, gene support coverage, jackknife QA, source quality, and perturbation type.",
        "- Does not use support_val for rule search, canonical metrics, canonical multi, held-out Track C query, training, inference, or GPU.",
        "",
        "## Summary",
        "",
        f"- specs: `{artifact.get('n_specs')}`",
        f"- rules: `{len(rules)}`",
        f"- rule-spec combinations: `{len(rows_out)}`",
        f"- passing rule-specs: `{len(passing)}`",
        f"- best spec/rule: `{best.get('spec')}` / `{best.get('rule')}`",
        f"- best pp / p_harm: `{fmt(best.get('pp_delta'))}` / `{fmt(best.get('p_harm'))}`",
        f"- best enabled / negative / min pp: `{best.get('enabled_rows')}` / `{best.get('enabled_negative_rows')}` / `{fmt(best.get('enabled_min_pp_delta'))}`",
        f"- best Norman / Wessels: `{fmt(best.get('norman_pp'))}` / `{fmt(best.get('wessels_pp'))}`",
        "",
        "## Top Rule-Specs",
        "",
        "| spec | rule | pass | enabled | pp | p_harm | neg | min pp | Norman | Wessels | reasons |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows_out[:12]:
        lines.append(
            "| `{spec}` | `{rule}` | `{passed}` | {enabled} | {pp} | {p_harm} | {neg} | {min_pp} | {norman} | {wessels} | `{reasons}` |".format(
                spec=row["spec"],
                rule=row["rule"],
                passed=row["pass"],
                enabled=row["enabled_rows"],
                pp=fmt(row["pp_delta"]),
                p_harm=fmt(row["p_harm"]),
                neg=row["enabled_negative_rows"],
                min_pp=fmt(row["enabled_min_pp_delta"]),
                norman=fmt(row["norman_pp"]),
                wessels=fmt(row["wessels_pp"]),
                reasons=",".join(row["reasons"]),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- GPU authorized: `False`",
            f"- Track C query authorized: `False`",
            f"- canonical no-harm authorized: `False`",
            f"- reason counts: `{dict(reason_counts.most_common(8))}`",
            "- If failed, do not launch a V3 support-only smoke; no exogenous row-QA rule is train-only tail-safe.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "gpu_authorized": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
