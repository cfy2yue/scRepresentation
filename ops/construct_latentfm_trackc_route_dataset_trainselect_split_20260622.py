#!/usr/bin/env python3
"""Construct a Track C route-dataset trainselect split.

The output split is for training/sampling only: it keeps the fixed route
datasets from the support-route teacher and preserves their train/support-val
fields from the safe trainselect split. It deliberately omits final query
fields. Routed teacher banks should still point to the full trainselect split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_IN_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
DEFAULT_ROUTE_FILE = ROOT / "reports/latentfm_trackc_support_route_teacher_20260622.json"
DEFAULT_OUT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_route_datasets_trainselect.json"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_trackc_route_dataset_trainselect_split_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_TRACKC_ROUTE_DATASET_TRAINSELECT_SPLIT_20260622.md"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_multi(cond: str) -> bool:
    return "+" in str(cond)


def summarize(split: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for ds, sp in sorted(split.items()):
        train = [str(c) for c in sp.get("train", [])]
        test = [str(c) for c in sp.get("test", [])]
        rows[ds] = {
            "train": len(train),
            "train_multi": sum(1 for c in train if is_multi(c)),
            "train_single": sum(1 for c in train if not is_multi(c)),
            "test": len(test),
            "test_multi": sum(1 for c in test if is_multi(c)),
            "test_single": sum(1 for c in test if not is_multi(c)),
            "forbidden_query_fields_present": sorted(
                key for key in sp if "query" in str(key).lower() or "heldout" in str(key).lower()
            ),
        }
    return rows


def write_md(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# LatentFM Track C Route-Dataset Trainselect Split",
        "",
        f"Status: `{payload['status']}`",
        f"Recommended action: `{payload['recommended_action']}`",
        "",
        "## Provenance",
        "",
        f"- source_trainselect_split: `{payload['source_trainselect_split']}`",
        f"- route_file: `{payload['route_file']}`",
        f"- output_split: `{payload['output_split']}`",
        f"- heldout_query_fields_written: `{payload['heldout_query_fields_written']}`",
        "",
        "## Counts",
        "",
        "| dataset | train | train single | train multi | support-val test | test multi |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for ds, row in payload["counts"].items():
        lines.append(
            f"| `{ds}` | {row['train']} | {row['train_single']} | {row['train_multi']} | "
            f"{row['test']} | {row['test_multi']} |"
        )
    lines += [
        "",
        "## Usage",
        "",
        "- Use this split only as `SPLIT_FILE` for a route-focused Track C support smoke.",
        "- Use the full trainselect split as `TRACKC_ROUTED_DISTILL_BANK_SPLIT_FILE` so Norman additive teachers keep full train-only single-gene coverage.",
        "- Do not use the full v2 query split for training-time selection.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-split", type=Path, default=DEFAULT_IN_SPLIT)
    parser.add_argument("--route-file", type=Path, default=DEFAULT_ROUTE_FILE)
    parser.add_argument("--output-split", type=Path, default=DEFAULT_OUT_SPLIT)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    source = load_json(args.input_split)
    route_payload = load_json(args.route_file)
    routes = route_payload.get("route") or route_payload
    if not isinstance(routes, dict) or not routes:
        raise ValueError(f"route file does not contain a non-empty route mapping: {args.route_file}")

    out = {}
    missing = []
    for ds in sorted(str(k) for k in routes):
        if ds not in source:
            missing.append(ds)
            continue
        sp = source[ds]
        out[ds] = {
            "train": list(sp.get("train", [])),
            "test": list(sp.get("test", [])),
        }
    if missing:
        raise ValueError(f"route datasets missing from source split: {missing}")
    counts = summarize(out)
    forbidden = [
        (ds, key)
        for ds, row in counts.items()
        for key in row["forbidden_query_fields_present"]
    ]
    if forbidden:
        raise ValueError(f"query/heldout fields would be written: {forbidden}")
    if sum(row["test_multi"] for row in counts.values()) != 24:
        raise ValueError("route-dataset trainselect split must preserve 24 support-val multi test rows")

    args.output_split.parent.mkdir(parents=True, exist_ok=True)
    args.output_split.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = {
        "status": "route_dataset_trainselect_split_ready",
        "recommended_action": "use_as_sampling_split_with_full_trainselect_bank_split_only_after_protocol_review",
        "source_trainselect_split": str(args.input_split),
        "route_file": str(args.route_file),
        "output_split": str(args.output_split),
        "heldout_query_fields_written": False,
        "routes": routes,
        "counts": counts,
    }
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(payload, args.out_md)
    print(f"wrote {args.output_split}")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
