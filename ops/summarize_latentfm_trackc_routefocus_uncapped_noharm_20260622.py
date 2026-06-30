#!/usr/bin/env python3
"""Summarize Track C route-focused uncapped canonical no-harm posthoc.

This script is intentionally query-blind. It consumes only canonical
``split_seed42.json`` posthoc outputs produced by the pass-only uncapped
no-harm launcher and decides whether a single frozen Track C query evaluation
is allowed.
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
DEFAULT_INDEX = (
    ROOT
    / "reports/latentfm_trackc_routefocus_uncapped_noharm_20260622/uncapped_posthoc_index.json"
)
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_trackc_routefocus_uncapped_noharm_decision_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_TRACKC_ROUTEFOCUS_UNCAPPED_NOHARM_DECISION_20260622.md"

DEFAULT_SPLIT_GROUPS = [
    "test",
    "test_single",
    "test_multi",
    "test_multi_seen",
    "test_multi_unseen1",
    "test_multi_unseen2",
]
DEFAULT_FAMILY_GROUPS = [
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


def first_output(index: dict[str, Any]) -> dict[str, Any]:
    outputs = index.get("outputs") or []
    if not outputs:
        raise ValueError("uncapped posthoc index contains no outputs")
    if len(outputs) > 1:
        raise ValueError(f"expected exactly one route-focused output, found {len(outputs)}")
    row = outputs[0]
    if not isinstance(row, dict):
        raise ValueError("uncapped posthoc output row is not a dict")
    return row


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
        raise FileNotFoundError(f"missing anchor JSON: {baseline}")
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


def metric_row(
    tables: dict[str, dict[tuple[str, str], dict[str, Any]]],
    table: str,
    group: str,
    metric: str,
) -> dict[str, Any] | None:
    return tables.get(table, {}).get((group, metric))


def row_float(row: dict[str, Any] | None, key: str, default: float) -> float:
    if not row:
        return default
    value = row.get(key)
    if value is None:
        return default
    return float(value)


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
    required = {
        "canonical_test_single_pp": metric_row(tables, "split", "test_single", "pearson_pert"),
        "canonical_test_single_mmd": metric_row(tables, "split", "test_single", "test_mmd_clamped"),
        "canonical_family_gene_pp": metric_row(tables, "family", "family_gene", "pearson_pert"),
        "canonical_family_gene_mmd": metric_row(tables, "family", "family_gene", "test_mmd_clamped"),
    }
    for name, row in required.items():
        if not row or row.get("status") != "ok":
            reasons.append(f"missing_or_bad_{name}")

    if not reasons:
        if row_float(required["canonical_test_single_pp"], "p_harm", 1.0) > 0.35:
            reasons.append("canonical_test_single_pp_harm_risk")
        if row_float(required["canonical_test_single_mmd"], "p_harm", 1.0) > 0.80:
            reasons.append("canonical_test_single_mmd_hard_harm")
        if row_float(required["canonical_family_gene_pp"], "p_harm", 1.0) > 0.35:
            reasons.append("canonical_family_gene_pp_harm_risk")
        if row_float(required["canonical_family_gene_mmd"], "p_harm", 1.0) > 0.80:
            reasons.append("canonical_family_gene_mmd_hard_harm")

    if reasons:
        status = "trackc_uncapped_canonical_noharm_fail_close_before_query"
        action = "close_branch_do_not_evaluate_query"
    else:
        status = "trackc_uncapped_canonical_noharm_pass_query_allowed_once"
        action = "run_single_frozen_trackc_query_eval"
    return {
        "status": status,
        "action": action,
        "reasons": reasons,
        "rules": [
            "canonical test_single pearson_pert no harm: p_harm <= 0.35",
            "canonical test_single MMD no hard harm: p_harm <= 0.80",
            "canonical family_gene pearson_pert no harm: p_harm <= 0.35",
            "canonical family_gene MMD no hard harm: p_harm <= 0.80",
            "canonical multi is forbidden unless explicitly included as diagnostic-only by the posthoc protocol; held-out query is allowed only once if this decision passes",
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
    rows = {
        "canonical single pp": metric_row(tables, "split", "test_single", "pearson_pert"),
        "canonical single mmd": metric_row(tables, "split", "test_single", "test_mmd_clamped"),
        "canonical family pp": metric_row(tables, "family", "family_gene", "pearson_pert"),
        "canonical family mmd": metric_row(tables, "family", "family_gene", "test_mmd_clamped"),
        "canonical multi diagnostic pp": metric_row(tables, "split", "test_multi", "pearson_pert"),
        "canonical unseen2 diagnostic pp": metric_row(tables, "split", "test_multi_unseen2", "pearson_pert"),
    }
    lines = [
        f"# {payload.get('report_title') or 'LatentFM Track C Route-Focused Uncapped Canonical No-Harm Decision'}",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"Recommended action: `{payload['decision']['action']}`",
        "",
        "## Provenance",
        "",
    ]
    for key, value in payload["inputs"].items():
        lines.append(f"- {key}: `{value}`")
    lines += [
        "- leakage boundary: canonical `split_seed42.json` only; no held-out Track C query outputs are read.",
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
        lines.extend(f"- `{reason}`" for reason in reasons)
    else:
        lines.append("- none")
    lines += ["", "## Rules", ""]
    lines.extend(f"- {rule}" for rule in payload["decision"].get("rules") or [])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-json", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument(
        "--report-title",
        default="LatentFM Track C Route-Focused Uncapped Canonical No-Harm Decision",
    )
    parser.add_argument("--boot-dir", type=Path, default=None)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--split-groups", nargs="+", default=None)
    parser.add_argument("--family-groups", nargs="+", default=None)
    args = parser.parse_args()

    if not BOOTSTRAP.is_file():
        raise FileNotFoundError(BOOTSTRAP)
    if not args.index_json.is_file():
        raise FileNotFoundError(f"uncapped posthoc index not found: {args.index_json}")
    boot_dir = args.boot_dir or (
        args.out_json.parent / "latentfm_trackc_routefocus_uncapped_noharm_bootstrap_20260622"
    )
    existing = [p for p in (args.out_json, args.out_md, boot_dir) if p.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite uncapped no-harm decision artifacts: "
            + ", ".join(str(p) for p in existing)
        )
    index = load_json(args.index_json)
    row = first_output(index)
    split_groups = args.split_groups or DEFAULT_SPLIT_GROUPS
    family_groups = args.family_groups or DEFAULT_FAMILY_GROUPS
    paths = {
        "index_json": args.index_json,
        "anchor_split": Path(row["baseline_split_json"]),
        "candidate_split": Path(row["run_split_json"]),
        "anchor_family": Path(row["baseline_family_json"]),
        "candidate_family": Path(row["run_family_json"]),
        "split_file": row.get("split_file"),
        "anchor_checkpoint": row.get("anchor_checkpoint"),
        "candidate_checkpoint": row.get("candidate_checkpoint"),
    }
    boot_dir.mkdir(parents=True, exist_ok=True)
    boot_payloads = {
        "split": bootstrap_pair(
            baseline=paths["anchor_split"],
            candidate=paths["candidate_split"],
            groups=split_groups,
            out_json=boot_dir / "canonical_split_anchor_vs_candidate.json",
            out_md=boot_dir / "canonical_split_anchor_vs_candidate.md",
            title="Track C Route-Focused Uncapped Canonical Split No-Harm",
            n_boot=args.n_boot,
            seed=args.seed,
            python=args.python,
        ),
        "family": bootstrap_pair(
            baseline=paths["anchor_family"],
            candidate=paths["candidate_family"],
            groups=family_groups,
            out_json=boot_dir / "canonical_family_anchor_vs_candidate.json",
            out_md=boot_dir / "canonical_family_anchor_vs_candidate.md",
            title="Track C Route-Focused Uncapped Canonical Family No-Harm",
            n_boot=args.n_boot,
            seed=args.seed + 1,
            python=args.python,
        ),
    }
    tables = {name: index_rows(payload) for name, payload in boot_payloads.items()}
    decision = evaluate_gate(tables)
    payload = {
        "inputs": {k: str(v) for k, v in paths.items()},
        "split_groups": split_groups,
        "family_groups": family_groups,
        "bootstrap_dir": str(boot_dir),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "report_title": str(args.report_title),
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
