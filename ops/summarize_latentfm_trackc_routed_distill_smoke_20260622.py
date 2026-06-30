#!/usr/bin/env python3
"""Summarize the Track C routed-distill smoke without reading held-out query.

This script consumes posthoc JSONs produced by
``launch_latentfm_xverse_trackc_routed_distill_smoke_20260622.sh``.  It only
uses the train-selection support-val split and canonical no-harm posthoc.  It
must not read ``split_seed42_multi_support_v2.json`` query outputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
BOOTSTRAP = ROOT / "ops/bootstrap_latentfm_paired_posthoc_20260621.py"
DEFAULT_RUN_ROOT = (
    ROOT
    / "runs/latentfm_xverse_trackc_routed_distill_20260622"
    / "xverse_trackc_route_condprior_w05_replay1_2k_seed42"
)
DEFAULT_REPORT_JSON = ROOT / "reports/latentfm_trackc_routed_distill_smoke_decision_20260622.json"
DEFAULT_REPORT_MD = ROOT / "reports/LATENTFM_TRACKC_ROUTED_DISTILL_SMOKE_DECISION_20260622.md"

SUPPORT_SPLIT_GROUPS = ["test", "test_multi"]
SUPPORT_FAMILY_GROUPS = ["test_all", "family_gene", "structure_multi", "test_multi"]
CANONICAL_SPLIT_GROUPS = [
    "test",
    "test_single",
    "test_multi",
    "test_multi_seen",
    "test_multi_unseen1",
    "test_multi_unseen2",
]
CANONICAL_FAMILY_GROUPS = [
    "test_all",
    "family_gene",
    "family_drug",
    "structure_single",
    "structure_multi",
    "test_single",
    "test_multi",
]
METRICS = ["pearson_pert", "pearson_ctrl", "direct_pearson", "test_mmd_clamped"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def bootstrap_pair(
    *,
    baseline: Path,
    candidate: Path,
    groups: list[str],
    out_json: Path,
    out_md: Path,
    title: str,
    n_boot: int,
    seed: int,
    python: str,
) -> dict[str, Any]:
    if not baseline.is_file():
        raise FileNotFoundError(f"missing baseline JSON: {baseline}")
    if not candidate.is_file():
        raise FileNotFoundError(f"missing candidate JSON: {candidate}")
    run(
        [
            python,
            str(BOOTSTRAP),
            "--baseline-json",
            str(baseline),
            "--candidate-json",
            str(candidate),
            "--groups",
            *groups,
            "--metrics",
            *METRICS,
            "--n-boot",
            str(int(n_boot)),
            "--seed",
            str(int(seed)),
            "--title",
            title,
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )
    return load_json(out_json)


def index_rows(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload.get("rows") or []:
        out[(str(row.get("group")), str(row.get("metric")))] = row
    return out


def row_float(row: dict[str, Any] | None, key: str, default: float) -> float:
    if not row:
        return default
    value = row.get(key)
    if value is None:
        return default
    return float(value)


def metric_row(
    tables: dict[str, dict[tuple[str, str], dict[str, Any]]],
    table: str,
    group: str,
    metric: str,
) -> dict[str, Any] | None:
    return tables.get(table, {}).get((group, metric))


def usable_metric_row(
    tables: dict[str, dict[tuple[str, str], dict[str, Any]]],
    table: str,
    groups: list[str],
    metric: str,
) -> dict[str, Any] | None:
    for group in groups:
        row = metric_row(tables, table, group, metric)
        if not row:
            continue
        if row.get("status") == "ok" and int(row.get("n_matched_conditions") or 0) > 0:
            return row
    for group in groups:
        row = metric_row(tables, table, group, metric)
        if row:
            return row
    return None


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def ci_fmt(row: dict[str, Any] | None) -> str:
    if not row:
        return "[NA, NA]"
    return f"[{fmt(row.get('ci95_low'))}, {fmt(row.get('ci95_high'))}]"


def evaluate_gate(tables: dict[str, dict[tuple[str, str], dict[str, Any]]]) -> dict[str, Any]:
    reasons: list[str] = []

    support_groups = ["test_multi", "test"]
    support_pp = usable_metric_row(tables, "support_split", support_groups, "pearson_pert")
    support_mmd = usable_metric_row(tables, "support_split", support_groups, "test_mmd_clamped")
    canonical_single_pp = metric_row(tables, "canonical_split", "test_single", "pearson_pert")
    canonical_single_mmd = metric_row(tables, "canonical_split", "test_single", "test_mmd_clamped")
    canonical_family_pp = metric_row(tables, "canonical_family", "family_gene", "pearson_pert")
    canonical_family_mmd = metric_row(tables, "canonical_family", "family_gene", "test_mmd_clamped")

    required = {
        "support_pp": support_pp,
        "support_mmd": support_mmd,
        "canonical_single_pp": canonical_single_pp,
        "canonical_single_mmd": canonical_single_mmd,
        "canonical_family_pp": canonical_family_pp,
        "canonical_family_mmd": canonical_family_mmd,
    }
    for name, row in required.items():
        if not row or row.get("status") != "ok":
            reasons.append(f"missing_or_bad_{name}")

    if not reasons:
        if row_float(support_pp, "delta_mean", 0.0) < 0.02:
            reasons.append("support_pp_delta_below_0p02")
        if row_float(support_pp, "p_improvement", 0.0) < 0.75:
            reasons.append("support_pp_improvement_weak")
        if row_float(support_mmd, "p_harm", 1.0) > 0.80:
            reasons.append("support_mmd_hard_harm")
        if row_float(canonical_single_pp, "p_harm", 1.0) > 0.35:
            reasons.append("canonical_test_single_pp_harm_risk")
        if row_float(canonical_single_mmd, "p_harm", 1.0) > 0.80:
            reasons.append("canonical_test_single_mmd_hard_harm")
        if row_float(canonical_family_pp, "p_harm", 1.0) > 0.35:
            reasons.append("canonical_family_gene_pp_harm_risk")
        if row_float(canonical_family_mmd, "p_harm", 1.0) > 0.80:
            reasons.append("canonical_family_gene_mmd_hard_harm")

    if not reasons:
        status = "trackc_smoke_support_pass_needs_uncapped_noharm_before_query"
        action = "run_condition_uncapped_canonical_noharm_then_one_shot_query_if_pass"
    elif any(r.startswith("missing_or_bad_") for r in reasons):
        status = "trackc_smoke_missing_required_metrics_close_branch"
        action = "close_branch_and_audit_posthoc_metric_coverage"
    elif any(r.startswith("canonical_") for r in reasons):
        status = "trackc_smoke_fail_canonical_harm_close_branch"
        action = "close_branch_or_redesign_noharm_adapter"
    else:
        status = "trackc_smoke_fail_support_gate_close_branch"
        action = "close_branch_or_revisit_support_teacher_design"
    return {
        "status": status,
        "action": action,
        "reasons": reasons,
        "rules": [
            "support-val test_multi pearson_pert delta >= +0.02; fall back to support test alias only when test_multi is empty",
            "support-val test_multi pearson_pert p_improvement >= 0.75; fall back to support test alias only when test_multi is empty",
            "support-val test_multi MMD no hard harm: p_harm <= 0.80; fall back to support test alias only when test_multi is empty",
            "canonical test_single pearson_pert no harm: p_harm <= 0.35",
            "canonical test_single MMD no hard harm: p_harm <= 0.80",
            "canonical family_gene pearson_pert no harm: p_harm <= 0.35",
            "canonical family_gene MMD no hard harm: p_harm <= 0.80",
            "held-out query is forbidden for this decision; evaluate query once only after route/checkpoint freeze and uncapped no-harm pass",
        ],
    }


def row_line(role: str, row: dict[str, Any] | None) -> str:
    if not row:
        return f"| {role} | NA | NA | 0 | 0 | NA | [NA, NA] | NA | NA | missing |"
    return (
        f"| {role} | {row.get('group')} | {row.get('metric')} | "
        f"{row.get('n_matched_conditions', 0)} | {row.get('n_matched_datasets', 0)} | "
        f"{fmt(row.get('delta_mean'))} | {ci_fmt(row)} | "
        f"{fmt(row.get('p_improvement'))} | {fmt(row.get('p_harm'))} | {row.get('status', 'NA')} |"
    )


def render_report(payload: dict[str, Any]) -> str:
    tables = payload["tables"]
    support_groups = ["test_multi", "test"]
    rows = {
        "support pp": usable_metric_row(tables, "support_split", support_groups, "pearson_pert"),
        "support mmd": usable_metric_row(tables, "support_split", support_groups, "test_mmd_clamped"),
        "canonical single pp": metric_row(tables, "canonical_split", "test_single", "pearson_pert"),
        "canonical single mmd": metric_row(tables, "canonical_split", "test_single", "test_mmd_clamped"),
        "canonical family pp": metric_row(tables, "canonical_family", "family_gene", "pearson_pert"),
        "canonical family mmd": metric_row(tables, "canonical_family", "family_gene", "test_mmd_clamped"),
        "canonical multi diagnostic pp": metric_row(tables, "canonical_split", "test_multi", "pearson_pert"),
        "canonical unseen2 diagnostic pp": metric_row(tables, "canonical_split", "test_multi_unseen2", "pearson_pert"),
    }
    lines = [
        "# LatentFM Track C Routed-Distill Smoke Decision",
        "",
        f"Run root: `{payload['run_root']}`",
        f"Status: `{payload['decision']['status']}`",
        f"Recommended action: `{payload['decision']['action']}`",
        "",
        "## Provenance",
        "",
    ]
    for key, value in payload["inputs"].items():
        lines.append(f"- {key}: `{value}`")
    lines += [
        "- leakage boundary: this decision reads only support-val trainselect posthoc and canonical no-harm posthoc; it does not read held-out Track C query outputs.",
        "",
        "## Gate Rows",
        "",
        "| role | group | metric | n cond | n ds | delta | 95% CI | p improve | p harm | status |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for role, row in rows.items():
        lines.append(row_line(role, row))
    lines += ["", "## Gate Reasons", ""]
    reasons = payload["decision"].get("reasons") or []
    if reasons:
        lines.extend(f"- `{r}`" for r in reasons)
    else:
        lines.append("- none")
    lines += ["", "## Rules", ""]
    lines.extend(f"- {rule}" for rule in payload["decision"].get("rules") or [])
    lines += [
        "",
        "Interpretation:",
        "- `trackc_smoke_support_pass_needs_uncapped_noharm_before_query` is not a final multi claim; it only authorizes uncapped canonical no-harm, then a single frozen query evaluation if that passes.",
        "- Any failure status closes this routed-distill smoke unless a new train/support-only mechanism is proposed.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    if not BOOTSTRAP.is_file():
        raise FileNotFoundError(BOOTSTRAP)
    posthoc = args.run_root / "posthoc_eval"
    paths = {
        "support_anchor_split": posthoc / "support_anchor_split_ode20.json",
        "support_candidate_split": posthoc / "support_candidate_split_ode20.json",
        "support_anchor_family": posthoc / "support_anchor_family_ode20.json",
        "support_candidate_family": posthoc / "support_candidate_family_ode20.json",
        "canonical_anchor_split": posthoc / "canonical_anchor_split_ode20_stablecaps.json",
        "canonical_candidate_split": posthoc / "canonical_candidate_split_ode20_stablecaps.json",
        "canonical_anchor_family": posthoc / "canonical_anchor_family_ode20_stablecaps.json",
        "canonical_candidate_family": posthoc / "canonical_candidate_family_ode20_stablecaps.json",
    }
    boot_dir = args.run_root / "posthoc_eval" / "bootstrap_decision"
    boot_dir.mkdir(parents=True, exist_ok=True)
    boot_payloads = {
        "support_split": bootstrap_pair(
            baseline=paths["support_anchor_split"],
            candidate=paths["support_candidate_split"],
            groups=SUPPORT_SPLIT_GROUPS,
            out_json=boot_dir / "support_split_anchor_vs_candidate.json",
            out_md=boot_dir / "support_split_anchor_vs_candidate.md",
            title="Track C Support-Val Split Bootstrap",
            n_boot=args.n_boot,
            seed=args.seed,
            python=args.python,
        ),
        "support_family": bootstrap_pair(
            baseline=paths["support_anchor_family"],
            candidate=paths["support_candidate_family"],
            groups=SUPPORT_FAMILY_GROUPS,
            out_json=boot_dir / "support_family_anchor_vs_candidate.json",
            out_md=boot_dir / "support_family_anchor_vs_candidate.md",
            title="Track C Support-Val Family Bootstrap",
            n_boot=args.n_boot,
            seed=args.seed + 1,
            python=args.python,
        ),
        "canonical_split": bootstrap_pair(
            baseline=paths["canonical_anchor_split"],
            candidate=paths["canonical_candidate_split"],
            groups=CANONICAL_SPLIT_GROUPS,
            out_json=boot_dir / "canonical_split_anchor_vs_candidate.json",
            out_md=boot_dir / "canonical_split_anchor_vs_candidate.md",
            title="Track C Canonical Split No-Harm Bootstrap",
            n_boot=args.n_boot,
            seed=args.seed + 2,
            python=args.python,
        ),
        "canonical_family": bootstrap_pair(
            baseline=paths["canonical_anchor_family"],
            candidate=paths["canonical_candidate_family"],
            groups=CANONICAL_FAMILY_GROUPS,
            out_json=boot_dir / "canonical_family_anchor_vs_candidate.json",
            out_md=boot_dir / "canonical_family_anchor_vs_candidate.md",
            title="Track C Canonical Family No-Harm Bootstrap",
            n_boot=args.n_boot,
            seed=args.seed + 3,
            python=args.python,
        ),
    }
    tables = {name: index_rows(payload) for name, payload in boot_payloads.items()}
    decision = evaluate_gate(tables)
    payload = {
        "run_root": str(args.run_root),
        "inputs": {k: str(v) for k, v in paths.items()},
        "bootstrap_dir": str(boot_dir),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "decision": decision,
        "tables": tables,
    }
    json_payload = dict(payload)
    json_payload["tables"] = {
        name: {f"{group}:{metric}": row for (group, metric), row in table.items()}
        for name, table in tables.items()
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "action": decision["action"], "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
