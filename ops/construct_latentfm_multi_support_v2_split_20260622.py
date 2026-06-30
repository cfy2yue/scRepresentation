#!/usr/bin/env python3
"""Construct a formal Track C true-multi support/query split audit.

This split never replaces the canonical Track A split. It exposes a stratified
subset of existing canonical true-multi conditions as support/train and
support-val for future multi adaptation, while keeping disjoint true-multi query
conditions for final evaluation. Gasperini currently remains external because
it has too few multi conditions for a support/query split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_CANONICAL = ROOT / "dataset/biFlow_data/split_seed42.json"
DEFAULT_OUT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2.json"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_multi_support_v2_split_audit_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_MULTI_SUPPORT_V2_SPLIT_AUDIT_20260622.md"

FOCUS_DATASETS = ("NormanWeissman2019_filtered", "Wessels")
EXTERNAL_MULTI_REFERENCE = ("GasperiniShendure2019_lowMOI",)
STRATA = ("test_multi_seen", "test_multi_unseen1", "test_multi_unseen2")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_order(values: list[str], key: str) -> list[str]:
    return sorted(map(str, values), key=lambda x: hashlib.sha256(f"{key}|{x}".encode("utf-8")).hexdigest())


def split_stratum(conds: list[str], *, key: str, train_frac: float, val_frac: float, min_query: int) -> tuple[list[str], list[str], list[str]]:
    ordered = stable_order(conds, key)
    n = len(ordered)
    if n == 0:
        return [], [], []
    max_support = max(0, n - min_query)
    train_n = int(round(train_frac * n))
    val_n = int(round(val_frac * n))
    if n >= 8:
        train_n = max(train_n, 1)
        val_n = max(val_n, 1)
    elif n >= 3:
        train_n = max(train_n, 1)
    if train_n + val_n > max_support:
        overflow = train_n + val_n - max_support
        reduce_val = min(val_n, overflow)
        val_n -= reduce_val
        overflow -= reduce_val
        if overflow > 0:
            train_n = max(0, train_n - overflow)
    train = ordered[:train_n]
    val = ordered[train_n : train_n + val_n]
    query = ordered[train_n + val_n :]
    return train, val, query


def construct(args: argparse.Namespace) -> dict[str, Any]:
    canonical = load_json(args.canonical_split)
    out: dict[str, Any] = {}
    rows = []
    for ds, obj in sorted(canonical.items()):
        ds = str(ds)
        train_single = list(map(str, obj.get("train") or []))
        test_single = list(map(str, obj.get("test_single") or []))
        train_multi: list[str] = []
        val_multi: list[str] = []
        query_multi: list[str] = []
        query_by_stratum: dict[str, list[str]] = {}
        train_by_stratum: dict[str, list[str]] = {}
        val_by_stratum: dict[str, list[str]] = {}
        if ds in FOCUS_DATASETS:
            for stratum in STRATA:
                tr, va, qu = split_stratum(
                    list(map(str, obj.get(stratum) or [])),
                    key=f"multi_support_v2|{ds}|{stratum}|{args.seed}",
                    train_frac=float(args.train_frac),
                    val_frac=float(args.val_frac),
                    min_query=int(args.min_query_per_stratum),
                )
                train_by_stratum[stratum] = tr
                val_by_stratum[stratum] = va
                query_by_stratum[stratum] = qu
                train_multi.extend(tr)
                val_multi.extend(va)
                query_multi.extend(qu)
        else:
            query_multi = list(map(str, obj.get("test_multi") or []))
            for stratum in STRATA:
                query_by_stratum[stratum] = list(map(str, obj.get(stratum) or []))

        out[ds] = {
            "train": train_single + train_multi,
            "train_single": train_single,
            "train_multi": train_multi,
            "support_val_multi": val_multi,
            "test": test_single + query_multi,
            "test_single": test_single,
            "query_multi": query_multi,
            "test_multi": query_multi,
            "query_multi_seen": query_by_stratum.get("test_multi_seen", []),
            "query_multi_unseen1": query_by_stratum.get("test_multi_unseen1", []),
            "query_multi_unseen2": query_by_stratum.get("test_multi_unseen2", []),
            "train_multi_by_stratum": train_by_stratum,
            "support_val_multi_by_stratum": val_by_stratum,
            "source_canonical_test_multi": list(map(str, obj.get("test_multi") or [])),
        }
        rows.append(
            {
                "dataset": ds,
                "role": (
                    "focus_support_query"
                    if ds in FOCUS_DATASETS
                    else ("external_reference" if ds in EXTERNAL_MULTI_REFERENCE else "canonical_no_multi_support")
                ),
                "canonical_train_single": len(train_single),
                "canonical_test_single": len(test_single),
                "canonical_test_multi": len(obj.get("test_multi") or []),
                "train_multi": len(train_multi),
                "support_val_multi": len(val_multi),
                "query_multi": len(query_multi),
                "query_seen": len(query_by_stratum.get("test_multi_seen", [])),
                "query_unseen1": len(query_by_stratum.get("test_multi_unseen1", [])),
                "query_unseen2": len(query_by_stratum.get("test_multi_unseen2", [])),
            }
        )
    return {
        "canonical_split": str(args.canonical_split),
        "out_split": str(args.out_split),
        "seed": int(args.seed),
        "params": {
            "focus_datasets": list(FOCUS_DATASETS),
            "external_multi_reference": list(EXTERNAL_MULTI_REFERENCE),
            "train_frac": float(args.train_frac),
            "val_frac": float(args.val_frac),
            "min_query_per_stratum": int(args.min_query_per_stratum),
        },
        "leakage_status": "condition-split-only; no GT metrics, no posthoc outcomes, canonical split untouched",
        "split": out,
        "rows": rows,
        "decision": decide(rows),
    }


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    focus = [r for r in rows if r["role"] == "focus_support_query"]
    reasons = []
    if len(focus) < 2:
        reasons.append("fewer_than_two_focus_datasets")
    for row in focus:
        if row["train_multi"] < 10:
            reasons.append(f"{row['dataset']}:train_multi_lt_10")
        if row["support_val_multi"] < 5:
            reasons.append(f"{row['dataset']}:support_val_multi_lt_5")
        if row["query_multi"] < 40:
            reasons.append(f"{row['dataset']}:query_multi_lt_40")
        if row["query_unseen2"] < 5:
            reasons.append(f"{row['dataset']}:query_unseen2_lt_5")
    status = "split_audit_pass_no_gpu_without_new_adapter_gate" if not reasons else "split_audit_fail"
    return {
        "status": status,
        "action": "write_split_only_trackc_gpu_requires_new_adapter_gate" if not reasons else "do_not_use_split_for_gpu",
        "reasons": reasons,
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Track C Multi-Support v2 Split Audit",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"Recommended action: `{payload['decision']['action']}`",
        "",
        "## Provenance",
        "",
        f"- canonical_split: `{payload['canonical_split']}`",
        f"- out_split: `{payload['out_split']}`",
        f"- leakage_status: `{payload['leakage_status']}`",
        f"- params: `{payload['params']}`",
        "",
        "## Dataset Counts",
        "",
        "| dataset | role | train single | test single | canonical multi | train multi | val multi | query multi | query seen | query unseen1 | query unseen2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['dataset']} | {row['role']} | {row['canonical_train_single']} | "
            f"{row['canonical_test_single']} | {row['canonical_test_multi']} | {row['train_multi']} | "
            f"{row['support_val_multi']} | {row['query_multi']} | {row['query_seen']} | "
            f"{row['query_unseen1']} | {row['query_unseen2']} |"
        )
    lines += ["", "## Decision Reasons", ""]
    reasons = payload["decision"].get("reasons") or []
    lines.extend([f"- `{r}`" for r in reasons] if reasons else ["- none"])
    lines += [
        "",
        "## Usage Rule",
        "",
        "- This split is a Track C artifact only. It must not replace canonical `split_seed42.json`.",
        "- Query multi conditions are disjoint from train/support-val multi and should be evaluated once after support-internal selection.",
        "- GPU still requires a new adapter/regularizer mechanism gate; the previous v1 pairwise-adapter R2 result was diagnostic only.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-split", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--out-split", type=Path, default=DEFAULT_OUT_SPLIT)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.20)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--min-query-per-stratum", type=int, default=1)
    args = parser.parse_args()

    payload = construct(args)
    args.out_split.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    split = payload.pop("split")
    args.out_split.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["split_sha256"] = hashlib.sha256(args.out_split.read_bytes()).hexdigest()
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "out_split": str(args.out_split), "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
