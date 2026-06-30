#!/usr/bin/env python3
"""Track C OmniPath pair-level regulatory prior preflight.

This short CPU preflight checks whether the acquired OmniPath TF-target prior
actually covers safe-trainselect multi-perturbation gene pairs. It does not fit
or select a model, and it does not read held-out query, canonical test,
canonical multi, active logs, or GPU artifacts.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
SUPPORT_MODULE_PATH = ROOT / "ops/audit_latentfm_trackc_support_route_readiness_20260622.py"
DEFAULT_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
PRIOR_EDGES = ROOT / "dataset/external_priors/omnipath_tf_20260623/omnipath_tf_target_edges.tsv"
PRIOR_SUMMARY = ROOT / "dataset/external_priors/omnipath_tf_20260623/omnipath_tf_prior_summary.json"
FAILURE_JSON = ROOT / "reports/latentfm_trackc_composition_hybrid_failure_cases_20260623.json"
OUT_JSON = ROOT / "reports/latentfm_trackc_omnipath_pair_prior_preflight_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_OMNIPATH_PAIR_PRIOR_PREFLIGHT_20260623.md"
EXPECTED_TRAINSELECT_SHA256 = "5f29dd5b582a40da3736770ca29950c12c54e46b590c3f7705c1d37da89f4f20"


def load_support_module() -> Any:
    spec = importlib.util.spec_from_file_location("trackc_support_route_readiness", SUPPORT_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {SUPPORT_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_edges(path: Path) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            tf = str(row["tf"]).upper()
            target = str(row["target"]).upper()
            sign = int(row.get("sign") or 0)
            out[(tf, target)] = sign
    return out


def genes(row: dict[str, Any]) -> list[str]:
    return [str(g).strip().upper() for g in (row.get("genes") or []) if str(g).strip()]


def pair_features(row: dict[str, Any], edges: dict[tuple[str, str], int]) -> dict[str, Any]:
    gs = genes(row)
    directed = []
    signed = []
    for src in gs:
        for dst in gs:
            if src == dst:
                continue
            if (src, dst) in edges:
                sign = edges[(src, dst)]
                directed.append({"source": src, "target": dst, "sign": sign})
                if sign != 0:
                    signed.append(sign)
    return {
        "dataset": str(row["dataset"]),
        "condition": str(row["condition"]),
        "genes": gs,
        "n_genes": len(gs),
        "has_directed_edge": bool(directed),
        "has_signed_edge": bool(signed),
        "n_directed_edges": len(directed),
        "n_signed_edges": len(signed),
        "signed_balance": int(sum(signed)) if signed else 0,
        "edges": directed,
    }


def summarize_role(rows: list[dict[str, Any]], edges: dict[tuple[str, str], int]) -> dict[str, Any]:
    items = [pair_features(row, edges) for row in rows if len(genes(row)) >= 2]
    by_dataset = {}
    for ds in sorted({item["dataset"] for item in items}):
        sub = [item for item in items if item["dataset"] == ds]
        by_dataset[ds] = {
            "n": len(sub),
            "directed_coverage": float(np.mean([item["has_directed_edge"] for item in sub])) if sub else 0.0,
            "signed_coverage": float(np.mean([item["has_signed_edge"] for item in sub])) if sub else 0.0,
            "mean_directed_edges": float(np.mean([item["n_directed_edges"] for item in sub])) if sub else 0.0,
        }
    return {
        "n_conditions": len(items),
        "directed_coverage": float(np.mean([item["has_directed_edge"] for item in items])) if items else 0.0,
        "signed_coverage": float(np.mean([item["has_signed_edge"] for item in items])) if items else 0.0,
        "mean_directed_edges": float(np.mean([item["n_directed_edges"] for item in items])) if items else 0.0,
        "by_dataset": by_dataset,
        "examples_with_edges": [item for item in items if item["has_directed_edge"]][:10],
        "examples_without_edges": [item for item in items if not item["has_directed_edge"]][:10],
    }


def failure_overlap(edges: dict[tuple[str, str], int]) -> dict[str, Any] | None:
    if not FAILURE_JSON.is_file():
        return None
    payload = json.loads(FAILURE_JSON.read_text(encoding="utf-8"))
    out = []
    for key in ("worst_pp_rows", "best_pp_rows"):
        for row in payload.get(key) or []:
            out.append({**pair_features(row, edges), "failure_group": key, "pp_delta": row.get("pp_delta")})
    return {
        "source": str(FAILURE_JSON),
        "rows": out,
        "worst_with_directed_edge": [row for row in out if row["failure_group"] == "worst_pp_rows" and row["has_directed_edge"]],
        "best_with_directed_edge": [row for row in out if row["failure_group"] == "best_pp_rows" and row["has_directed_edge"]],
    }


def decide(train: dict[str, Any], support: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    if train["directed_coverage"] < 0.25:
        reasons.append("train_multi_directed_pair_coverage_below_0p25")
    if support["directed_coverage"] < 0.25:
        reasons.append("support_val_directed_pair_coverage_below_0p25")
    if train["signed_coverage"] < 0.10:
        reasons.append("train_multi_signed_pair_coverage_below_0p10")
    if support["signed_coverage"] < 0.10:
        reasons.append("support_val_signed_pair_coverage_below_0p10")
    return {
        "status": "trackc_omnipath_pair_prior_preflight_pass_metric_gate_next_no_gpu" if not reasons else "trackc_omnipath_pair_prior_preflight_fail_no_gpu",
        "gpu_authorization": "none",
        "next_authorization": "query_free_metric_gate_only" if not reasons else "none",
        "reasons": reasons,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.6f}"


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track C OmniPath Pair Prior Preflight",
        "",
        f"Status: `{payload['decision']['status']}`",
        "GPU authorization: `none`",
        f"Next authorization: `{payload['decision']['next_authorization']}`",
        "",
        "## Boundary",
        "",
        "- Uses only safe trainselect train_multi/support_val_multi condition genes and the frozen OmniPath prior.",
        "- No model fitting, no held-out query, no canonical test, no canonical multi, no active logs, and no GPU artifacts.",
        "- Existing hybrid failure-case rows are optionally annotated as diagnostic-only, not used for gate selection.",
        "",
        "## Prior",
        "",
        f"- edges TSV: `{payload['prior']['edges_tsv']}`",
        f"- raw TSV SHA256: `{payload['prior']['raw_tsv_sha256']}`",
        f"- deduplicated edges: `{payload['prior']['deduplicated_edges']}`",
        "",
        "## Coverage",
        "",
        "| role | n | directed coverage | signed coverage | mean directed edges |",
        "|---|---:|---:|---:|---:|",
    ]
    for role in ("train_multi", "support_val_multi"):
        row = payload["roles"][role]
        lines.append(f"| {role} | {row['n_conditions']} | {fmt(row['directed_coverage'])} | {fmt(row['signed_coverage'])} | {fmt(row['mean_directed_edges'])} |")
    lines.extend(["", "## Dataset Breakdown", "", "| role | dataset | n | directed coverage | signed coverage |", "|---|---|---:|---:|---:|"])
    for role in ("train_multi", "support_val_multi"):
        for ds, row in payload["roles"][role]["by_dataset"].items():
            lines.append(f"| {role} | {ds} | {row['n']} | {fmt(row['directed_coverage'])} | {fmt(row['signed_coverage'])} |")
    lines.extend(["", "## Gate Reasons", ""])
    reasons = payload["decision"].get("reasons") or []
    lines.extend([f"- `{reason}`" for reason in reasons] or ["- none"])
    overlap = payload.get("failure_case_overlap")
    if overlap:
        lines.extend(["", "## Failure-Case Annotation", ""])
        lines.append(f"- source: `{overlap['source']}`")
        lines.append(f"- worst rows with directed edge: `{len(overlap['worst_with_directed_edge'])}`")
        lines.append(f"- best rows with directed edge: `{len(overlap['best_with_directed_edge'])}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This preflight only decides whether a pair-level regulatory prior has enough safe-trainselect coverage to justify a later query-free metric gate. It never authorizes GPU by itself.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--edges", type=Path, default=PRIOR_EDGES)
    parser.add_argument("--prior-summary", type=Path, default=PRIOR_SUMMARY)
    parser.add_argument("--max-cells", type=int, default=8)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    support = load_support_module()
    split = support.load_json(args.split_file)
    guard = {
        "path": str(args.split_file),
        "sha256": __import__("hashlib").sha256(args.split_file.read_bytes()).hexdigest(),
    }
    if guard["sha256"] != EXPECTED_TRAINSELECT_SHA256:
        raise RuntimeError(f"unexpected trainselect split hash: {guard['sha256']}")
    manifest = support.load_json(args.data_dir / "manifest.json")
    metadata = support.load_json(Path(manifest["condition_metadata_file"]))
    train_multi = support.collect_role_rows(args.data_dir, split, metadata, "train_multi", max_cells=args.max_cells)
    support_val = support.collect_role_rows(args.data_dir, split, metadata, "support_val_multi", max_cells=args.max_cells)
    edges = load_edges(args.edges)
    prior = json.loads(args.prior_summary.read_text(encoding="utf-8"))
    roles = {
        "train_multi": summarize_role(train_multi, edges),
        "support_val_multi": summarize_role(support_val, edges),
    }
    decision = decide(roles["train_multi"], roles["support_val_multi"])
    payload = {
        "status": decision["status"],
        "decision": decision,
        "boundary": {
            "safe_trainselect_only": True,
            "model_fitting": False,
            "heldout_query_read": False,
            "canonical_test_read": False,
            "canonical_multi_read": False,
            "active_log_read": False,
            "gpu_artifact_read": False,
            "failure_case_annotation_diagnostic_only": FAILURE_JSON.is_file(),
            "python": sys.executable,
        },
        "inputs": {
            "data_dir": str(args.data_dir),
            "split_file": str(args.split_file),
            "edges_tsv": str(args.edges),
            "prior_summary": str(args.prior_summary),
        },
        "split_guard": guard,
        "prior": {
            "edges_tsv": str(args.edges),
            "raw_tsv_sha256": prior["hashes"]["raw_tsv"],
            "edges_tsv_sha256": prior["hashes"]["edges_tsv"],
            "deduplicated_edges": prior["deduplicated_edges"],
            "signed_edges": prior["signed_edges"],
        },
        "roles": roles,
        "failure_case_overlap": failure_overlap(edges),
    }
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(args.out_md), "out_json": str(args.out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
